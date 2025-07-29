
import torch
import torch.nn as nn

class TaskAttention(nn.Module):
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

class MultiScaleTaskAttention(nn.Module):
    """Task-adaptive Attention Generator for multiple scale features"""
    def __init__(self, feature_channels, num_tasks):
        super(MultiScaleTaskAttention, self).__init__()
        self.num_tasks = num_tasks
        self.feature_channels = feature_channels  # List of channels for each feature scale
        
        # Create attention modules for each scale
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, num_tasks, kernel_size=1),
                nn.Sigmoid()
            ) for channels in feature_channels
        ])

    def forward(self, features):
        # features is a list of multi-scale feature maps
        # Returns a list of [task0_features, task1_features, task2_features]
        # where each task_features contains all scales for that task
        
        task_features = [[] for _ in range(self.num_tasks)]
        
        for scale_idx, (feature, attention_module) in enumerate(zip(features, self.attention_modules)):
            # Generate attention weights for this scale
            attention_weights = attention_module(feature)
            
            # Apply attention for each task
            for task_idx in range(self.num_tasks):
                task_feature = feature * attention_weights[:, task_idx:task_idx+1, :, :]
                task_features[task_idx].append(task_feature)
        
        return task_features
