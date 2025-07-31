"""
Simplified loss functions for YOLOPX v2 to avoid complex device issues.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(nn.Module):
    """Simplified loss function for quick testing."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        """Simple loss calculation."""
        # Handle list predictions (from detection head)
        if isinstance(pred, (list, tuple)):
            device = pred[0].device if pred else torch.device('cuda')
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for p in pred:
                if p.numel() > 0:
                    total_loss = total_loss + torch.mean(p ** 2) * 0.01
            return total_loss
            
        if pred.numel() == 0 or (hasattr(target, 'numel') and target.numel() == 0):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # For detection head - use MSE loss as placeholder
        if len(pred.shape) == 4 and pred.shape[1] == 6:  # [B, 6, H, W]
            # Simple regression loss
            pred_flat = pred.view(pred.size(0), -1)
            target_expanded = torch.zeros_like(pred_flat)
            if target.numel() > 0:
                # Use only a small portion of target to match pred size
                min_size = min(target_expanded.numel(), target.numel())
                target_expanded.view(-1)[:min_size] = target.view(-1)[:min_size]
            return self.mse_loss(pred_flat, target_expanded) * 0.1
        
        # For segmentation heads - use BCE loss
        elif len(pred.shape) == 4 and pred.shape[1] == 2:  # [B, 2, H, W]
            if isinstance(target, torch.Tensor) and target.numel() > 0:
                # Resize target to match pred size if needed
                if target.shape != pred.shape:
                    target_resized = F.interpolate(target.float().unsqueeze(1), 
                                                 size=pred.shape[2:], mode='nearest')
                    if target_resized.shape[1] == 1:
                        target_resized = target_resized.squeeze(1)
                        # Convert to one-hot, clamp values to valid range
                        target_resized = torch.clamp(target_resized.long(), 0, pred.shape[1] - 1)
                        target_onehot = torch.zeros_like(pred)
                        target_onehot.scatter_(1, target_resized.unsqueeze(1), 1)
                        target = target_onehot
                    else:
                        target = target_resized
                else:
                    target = target.float()
                
                return self.bce_loss(pred, target) * 0.1
            else:
                # Return small dummy loss if no target
                return torch.tensor(0.01, device=pred.device, requires_grad=True)
        
        # Fallback - return small dummy loss
        return torch.tensor(0.01, device=pred.device, requires_grad=True)


class SimpleMultiHeadLoss(nn.Module):
    """Simplified multi-head loss."""
    
    def __init__(self):
        super().__init__()
        self.det_loss = SimpleLoss()
        self.seg_loss = SimpleLoss()
        
    def forward(self, predictions, targets, shapes, model, imgs):
        """Forward pass for multi-head loss."""
        if len(predictions) != 3 or len(targets) != 3:
            # Fallback for shape mismatch
            total_loss = torch.tensor(0.1, device=imgs.device, requires_grad=True)
            head_losses = [torch.tensor(0.033, device=imgs.device, requires_grad=True)] * 3
            return total_loss, torch.stack(head_losses)
        
        det_pred, seg1_pred, seg2_pred = predictions
        det_target, seg1_target, seg2_target = targets
        
        # Detection loss
        if isinstance(det_pred, (list, tuple)):
            det_loss = sum(self.det_loss(pred, det_target) for pred in det_pred)
        else:
            det_loss = self.det_loss(det_pred, det_target)
        
        # Segmentation losses
        seg1_loss = self.seg_loss(seg1_pred, seg1_target)
        seg2_loss = self.seg_loss(seg2_pred, seg2_target)
        
        total_loss = det_loss + seg1_loss + seg2_loss
        head_losses = torch.stack([det_loss, seg1_loss, seg2_loss])
        
        return total_loss, head_losses


def get_loss(cfg, device, model):
    """Get simplified loss function."""
    return SimpleMultiHeadLoss()