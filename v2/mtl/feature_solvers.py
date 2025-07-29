"""
Feature-level Multi-Task Learning Solvers
Contains implementations of Task-Adaptive Attention Generator (TAG)
and its improved variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Original TAG Implementation ---

class TaskAttention(nn.Module):
    """
    Original Task-Adaptive Attention Generator (TAG).
    Generates a task-specific attention map for each task and applies it
    to the shared feature map.
    """
    def __init__(self, in_channels, num_tasks):
        super(TaskAttention, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels, num_tasks, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is the shared feature map
        task_attention_maps = self.attention_conv(x)
        task_attention_weights = self.sigmoid(task_attention_maps)
        
        # Multiply the shared features by the attention weights for each task
        task_specific_features = []
        for i in range(task_attention_weights.size(1)):
            task_specific_features.append(x * task_attention_weights[:, i:i+1, :, :])
            
        return task_specific_features

# --- Improved TAG Implementation ---

class ImprovedTaskAttention(nn.Module):
    """
    Enhanced Task-Adaptive Attention with better initialization, normalization,
    and feature refinement.
    """
    def __init__(self, in_channels, num_tasks, reduction=16):
        super(ImprovedTaskAttention, self).__init__()
        self.num_tasks = num_tasks
        self.in_channels = in_channels
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.task_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_tasks, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.feature_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels//4),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        channel_weights = self.channel_attention(self.global_pool(x))
        x_enhanced = x * channel_weights
        
        task_attention_maps = self.task_attention(x_enhanced)
        
        task_specific_features = []
        for i in range(self.num_tasks):
            attended_feature = x_enhanced * task_attention_maps[:, i:i+1, :, :]
            refined_feature = self.feature_refine[i](attended_feature)
            final_feature = attended_feature + 0.1 * refined_feature
            task_specific_features.append(final_feature)
            
        return task_specific_features

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion module with learnable weights for combining
    features from different sources (e.g., in a PaFPN).
    """
    def __init__(self, in_channels, fusion_type='weighted_add'):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted_add':
            self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        elif fusion_type == 'attention':
            self.attention_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 2, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, feature1, feature2):
        if self.fusion_type == 'weighted_add':
            weights = F.softmax(self.fusion_weights, dim=0)
            return weights[0] * feature1 + weights[1] * feature2
        elif self.fusion_type == 'attention':
            combined = torch.cat([feature1, feature2], dim=1)
            attention_weights = self.attention_conv(combined)
            return attention_weights[:, 0:1] * feature1 + attention_weights[:, 1:2] * feature2
        else:
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
