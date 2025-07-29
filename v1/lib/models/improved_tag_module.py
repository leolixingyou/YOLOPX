"""
Improved TAG (Task-adaptive Attention Generator) Module
Optimized for better performance and loss convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTaskAttention(nn.Module):
    """Enhanced Task-Adaptive Attention with better initialization and normalization"""
    def __init__(self, in_channels, num_tasks, reduction=16):
        super(ImprovedTaskAttention, self).__init__()
        self.num_tasks = num_tasks
        self.in_channels = in_channels
        
        # Global Average Pooling for spatial information compression
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Task-specific attention generation
        self.task_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_tasks, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Task-specific bias initialization
        with torch.no_grad():
            # Initialize task-specific biases to encourage specialization
            self.task_attention[-2].bias.data.fill_(0.1)
            for i in range(num_tasks):
                self.task_attention[-2].bias.data[i] = 0.5 + i * 0.1
        
        # Feature refinement layers
        self.feature_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels//4),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Enhanced channel attention
        channel_weights = self.channel_attention(self.global_pool(x))
        x_enhanced = x * channel_weights
        
        # Generate task-specific attention maps
        task_attention_maps = self.task_attention(x_enhanced)
        
        # Apply attention and refine features for each task
        task_specific_features = []
        for i in range(self.num_tasks):
            # Apply task-specific attention
            attended_feature = x_enhanced * task_attention_maps[:, i:i+1, :, :]
            
            # Task-specific feature refinement
            refined_feature = self.feature_refine[i](attended_feature)
            
            # Residual connection for better gradient flow
            final_feature = attended_feature + 0.1 * refined_feature
            
            task_specific_features.append(final_feature)
            
        return task_specific_features

class AdaptiveFeatureFusion(nn.Module):
    """Adaptive feature fusion module with learnable weights"""
    def __init__(self, in_channels, fusion_type='weighted_add'):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        self.in_channels = in_channels
        
        if fusion_type == 'weighted_add':
            self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        elif fusion_type == 'attention':
            self.attention_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 2, 1),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'se_fusion':
            self.se_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels * 2, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, feature1, feature2):
        if self.fusion_type == 'weighted_add':
            weights = F.softmax(self.fusion_weights, dim=0)
            return weights[0] * feature1 + weights[1] * feature2
        
        elif self.fusion_type == 'attention':
            combined = torch.cat([feature1, feature2], dim=1)
            attention_weights = self.attention_conv(combined)
            return attention_weights[:, 0:1] * feature1 + attention_weights[:, 1:2] * feature2
        
        elif self.fusion_type == 'se_fusion':
            combined = torch.cat([feature1, feature2], dim=1)
            se_weights = self.se_module(combined)
            enhanced_f1 = feature1 * se_weights
            enhanced_f2 = feature2 * se_weights
            return enhanced_f1 + enhanced_f2
        
        else:  # Default simple addition
            return feature1 + feature2

class ImprovedSelect(nn.Module):
    """Improved Select module with bounds checking and gradient flow optimization"""
    def __init__(self, task_index, num_tasks=3):
        super(ImprovedSelect, self).__init__()
        self.task_index = task_index
        self.num_tasks = num_tasks
        
        # Add a learnable scaling factor for better gradient flow
        self.scale_factor = nn.Parameter(torch.ones(1))
        
    def forward(self, task_features):
        if isinstance(task_features, list):
            if 0 <= self.task_index < len(task_features):
                selected = task_features[self.task_index]
                return selected * self.scale_factor
            else:
                # Fallback: return zero tensor with proper shape
                if len(task_features) > 0:
                    return torch.zeros_like(task_features[0])
                else:
                    raise ValueError(f"Empty task_features list, cannot select index {self.task_index}")
        else:
            raise TypeError(f"Expected list of task features, got {type(task_features)}")

class DynamicLossWeighting(nn.Module):
    """Dynamic loss weighting inspired by GradNorm but simplified for TAG"""
    def __init__(self, num_tasks=3, alpha=1.5, device='cuda'):
        super(DynamicLossWeighting, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        
        # Learnable task weights (initialized to be balanced)
        self.task_weights = nn.Parameter(torch.ones(num_tasks, device=device))
        
        # Running average of losses
        self.register_buffer('running_loss_avg', torch.ones(num_tasks, device=device))
        self.register_buffer('initial_losses', torch.ones(num_tasks, device=device))
        self.register_buffer('step_count', torch.zeros(1, device=device))
        
    def forward(self, losses):
        """
        Args:
            losses: list of task losses [det_loss, da_loss, ll_loss]
        Returns:
            weighted_loss: combined weighted loss
            task_weights: current task weights
        """
        losses_tensor = torch.stack([l if l is not None else torch.tensor(0.0, device=self.device) 
                                   for l in losses[:self.num_tasks]])
        
        # Update running averages
        if self.step_count == 0:
            self.initial_losses.data = losses_tensor.data.clone()
            self.running_loss_avg.data = losses_tensor.data.clone()
        else:
            # Exponential moving average
            momentum = 0.9
            self.running_loss_avg.data = momentum * self.running_loss_avg.data + (1 - momentum) * losses_tensor.data
        
        self.step_count += 1
        
        # Calculate relative losses
        with torch.no_grad():
            if self.step_count > 10:  # Start dynamic weighting after some steps
                relative_losses = self.running_loss_avg / (self.initial_losses + 1e-8)
                target_relative = relative_losses.mean()
                
                # GradNorm-inspired weight updates
                weight_updates = torch.pow(relative_losses / (target_relative + 1e-8), self.alpha)
                self.task_weights.data = self.task_weights.data * weight_updates
                
                # Normalize weights
                self.task_weights.data = self.task_weights.data / self.task_weights.data.sum() * self.num_tasks
        
        # Apply weights to losses
        weighted_losses = losses_tensor * F.softmax(self.task_weights, dim=0)
        weighted_loss = weighted_losses.sum()
        
        return weighted_loss, F.softmax(self.task_weights, dim=0)