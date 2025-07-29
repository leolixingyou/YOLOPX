"""
Improved TAG Loss Function
Optimized to achieve GradNorm-level performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import compute_loss, ComputeLoss, Tversky_Loss
from ..models.improved_tag_module import DynamicLossWeighting

class ImprovedTAGLoss(nn.Module):
    """Improved TAG loss with dynamic weighting and better task balancing"""
    
    def __init__(self, cfg, device, model):
        super(ImprovedTAGLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.model = model
        
        # Base loss computation
        self.base_loss_fn = ComputeLoss(model, cfg)
        
        # Dynamic loss weighting module
        self.dynamic_weighting = DynamicLossWeighting(
            num_tasks=3, 
            alpha=1.5, 
            device=device
        )
        
        # Improved loss balance weights (starting closer to GradNorm balance)
        self.det_weight = 0.8  # Increased from 0.02
        self.da_weight = 0.25  # Slightly increased from 0.2  
        self.ll_weight = 0.35  # Decreased from 0.6
        self.ll_iou_weight = 0.35  # Decreased from 0.6
        
        # Task-specific loss scaling factors
        self.task_scales = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))
        
        # Loss history for stability
        self.loss_history = {
            'det': [],
            'da': [], 
            'll': [],
            'll_iou': []
        }
        
    def forward(self, predictions, targets, shapes, model, input_imgs):
        """
        Compute improved TAG loss with dynamic weighting
        
        Args:
            predictions: (train_out, da_seg_out, ll_seg_out)
            targets: ground truth targets
            shapes: image shapes
            model: model instance
            input_imgs: input images
            
        Returns:
            total_loss: weighted total loss
            loss_components: tuple of individual losses
        """
        train_out, da_seg_out, ll_seg_out = predictions
        
        # 1. Detection loss (using base loss function)
        if train_out is not None:
            det_loss = self.base_loss_fn(train_out, targets)[1]  # Get total loss
            det_loss = det_loss * self.det_weight
        else:
            det_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 2. Driving area segmentation loss
        if da_seg_out is not None and targets[1] is not None:
            da_seg_loss = F.cross_entropy(
                da_seg_out, 
                targets[1].argmax(dim=1), 
                reduction='mean'
            )
            da_seg_loss = da_seg_loss * self.da_weight
        else:
            da_seg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 3. Lane line segmentation loss (BCE + Tversky)
        if ll_seg_out is not None and targets[2] is not None:
            # BCE Loss (using lane channel and float target)
            ll_bce_loss = F.binary_cross_entropy_with_logits(
                ll_seg_out[:, 1, :, :], # Assuming channel 1 is the lane
                targets[2][:, 1, :, :].float(), # Assuming channel 1 is the lane, convert to float
                reduction='mean'
            )
            
            # Tversky Loss for better handling of imbalanced data (using lane channel and float target)
            ll_tversky_loss = Tversky_Loss(ll_seg_out[:, 1, :, :], targets[2][:, 1, :, :].float())
            
            # Combined lane line loss
            ll_seg_loss = ll_bce_loss * self.ll_weight
            ll_iou_loss = ll_tversky_loss * self.ll_iou_weight
        else:
            ll_seg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            ll_iou_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 4. Apply task-specific scaling
        losses = [det_loss, da_seg_loss, ll_seg_loss + ll_iou_loss]
        scaled_losses = [loss * scale for loss, scale in zip(losses, self.task_scales)]
        
        # 5. Dynamic loss weighting (GradNorm-inspired)
        if self.training:
            total_loss, current_weights = self.dynamic_weighting(scaled_losses)
            
            # Update loss history for monitoring
            self.loss_history['det'].append(det_loss.item() if det_loss.requires_grad else 0.0)
            self.loss_history['da'].append(da_seg_loss.item() if da_seg_loss.requires_grad else 0.0)
            self.loss_history['ll'].append(ll_seg_loss.item() if ll_seg_loss.requires_grad else 0.0)
            self.loss_history['ll_iou'].append(ll_iou_loss.item() if ll_iou_loss.requires_grad else 0.0)
            
            # Limit history length
            for key in self.loss_history:
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-50:]
                    
        else:
            # During validation, use simple weighted sum
            total_loss = sum(scaled_losses)
            current_weights = F.softmax(self.task_scales, dim=0)
        
        return total_loss, (det_loss, da_seg_loss, ll_seg_loss, ll_iou_loss, total_loss)
    
    def get_current_weights(self):
        """Get current dynamic weights"""
        if hasattr(self.dynamic_weighting, 'task_weights'):
            return F.softmax(self.dynamic_weighting.task_weights, dim=0)
        else:
            return F.softmax(self.task_scales, dim=0)
    
    def get_loss_statistics(self):
        """Get loss history statistics"""
        stats = {}
        for key, history in self.loss_history.items():
            if len(history) > 0:
                stats[f'{key}_mean'] = sum(history) / len(history)
                stats[f'{key}_std'] = (sum([(x - stats[f'{key}_mean'])**2 for x in history]) / len(history)) ** 0.5
                stats[f'{key}_recent'] = sum(history[-10:]) / min(10, len(history))
        return stats

def get_improved_tag_loss(cfg, device, model):
    """Factory function to create improved TAG loss"""
    return ImprovedTAGLoss(cfg, device, model)