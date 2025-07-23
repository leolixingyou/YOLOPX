
import torch
import torch.nn as nn

class TaskAttention(nn.Module):
    def __init__(self, in_channels, num_tasks):
        super(TaskAttention, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels, num_tasks, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is the shared feature map
        task_attention_maps = self.attention_conv(x)
        task_attention_weights = self.softmax(task_attention_maps)
        
        # Multiply the shared features by the attention weights for each task
        task_specific_features = []
        for i in range(task_attention_weights.size(1)):
            task_specific_features.append(x * task_attention_weights[:, i:i+1, :, :])
            
        return task_specific_features
