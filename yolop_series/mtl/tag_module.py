"""
Task-adaptive Attention Generator (TAG) Module
Based on the paper: "Multi-task Learning for Real-time Autonomous Driving 
Leveraging Task-adaptive Attention Generator" (2403.03468v1)

Core idea:
- Generate task-specific channel attention αt from semantic features
- Generate task-generic spatial attention β from detail features  
- Combine them to create task-adaptive features: ht = αt · h + β
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAdaptiveAttentionGenerator(nn.Module):
    """
    TAG module that generates task-adaptive features through dual attention mechanism.
    
    Based on paper equation:
    αt = σ(Sch^t(xsemantic))  # Task-specific channel attention
    β = σ(Ssp(xdetail))       # Task-generic spatial attention
    ht = αt · h + β           # Task-adaptive features
    """
    
    def __init__(self, semantic_channels, detail_channels, output_channels, num_tasks=3):
        super(TaskAdaptiveAttentionGenerator, self).__init__()
        self.num_tasks = num_tasks
        self.output_channels = output_channels
        
        # Task-specific channel attention layers (one per task)
        self.task_channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
                nn.Conv2d(semantic_channels, semantic_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(semantic_channels // 4, output_channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # Task-generic spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels // 4, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(detail_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion layer to create task-generic feature h
        self.feature_fusion = nn.Conv2d(semantic_channels + detail_channels, output_channels, 1)
        
    def forward(self, inputs):
        """
        Args:
            inputs: List of [detail_features, semantic_features]
            
        Returns:
            List of task-adaptive features [h0, h1, h2, ...] for each task
        """
        if isinstance(inputs, list) and len(inputs) == 2:
            detail_features, semantic_features = inputs
        else:
            raise ValueError(f"TAG expects 2 inputs [detail_features, semantic_features], got {len(inputs) if isinstance(inputs, list) else 'single input'}")
        # Resize detail features to match semantic features if needed
        if semantic_features.shape[2:] != detail_features.shape[2:]:
            detail_features = F.interpolate(
                detail_features, 
                size=semantic_features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Create task-generic feature h by fusing semantic and detail features
        h = self.feature_fusion(torch.cat([semantic_features, detail_features], dim=1))
        
        # Generate task-generic spatial attention β
        beta = self.spatial_attention(detail_features)
        
        # Generate task-adaptive features for each task
        task_adaptive_features = []
        for task_id in range(self.num_tasks):
            # Task-specific channel attention αt
            alpha_t = self.task_channel_attention[task_id](semantic_features)
            
            # Task-adaptive feature: ht = αt · h + β
            h_t = alpha_t * h + beta
            task_adaptive_features.append(h_t)
            
        return task_adaptive_features

class TAGSelect(nn.Module):
    """Select specific task feature from TAG output"""
    
    def __init__(self, task_index):
        super(TAGSelect, self).__init__()
        self.task_index = task_index
        
    def forward(self, task_features):
        """
        Args:
            task_features: List of task-adaptive features from TAG
        Returns:
            Selected task feature
        """
        if isinstance(task_features, list) and len(task_features) > self.task_index:
            return task_features[self.task_index]
        else:
            raise ValueError(f"Cannot select task {self.task_index} from {len(task_features)} tasks")

class TAGFusionLayer(nn.Module):
    """
    Fusion layer that combines TAG features with regular backbone features
    """
    
    def __init__(self, tag_channels, backbone_channels, output_channels):
        super(TAGFusionLayer, self).__init__()
        self.tag_proj = nn.Conv2d(tag_channels, output_channels, 1) if tag_channels != output_channels else nn.Identity()
        self.backbone_proj = nn.Conv2d(backbone_channels, output_channels, 1) if backbone_channels != output_channels else nn.Identity()
        self.fusion_weight = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(self, inputs):
        """
        Weighted fusion of TAG feature and backbone feature
        """
        if isinstance(inputs, list) and len(inputs) == 2:
            tag_feature, backbone_feature = inputs
        else:
            raise ValueError(f"TAGFusionLayer expects 2 inputs [tag_feature, backbone_feature], got {len(inputs) if isinstance(inputs, list) else 'single input'}")
        # Resize to match if needed
        if tag_feature.shape[2:] != backbone_feature.shape[2:]:
            tag_feature = F.interpolate(
                tag_feature, 
                size=backbone_feature.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Project to same channel dimension
        tag_proj = self.tag_proj(tag_feature)
        backbone_proj = self.backbone_proj(backbone_feature)
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weight, dim=0)
        return weights[0] * tag_proj + weights[1] * backbone_proj