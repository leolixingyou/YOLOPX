"""
改进的模块实现
包含优化的TAG注意力、特征融合和初始化策略
"""

import torch
import torch.nn as nn


class ImprovedTaskAttention(nn.Module):
    def __init__(self, in_channels, num_tasks, init_strategy="improved", temperature=1.0):
        super(ImprovedTaskAttention, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels, num_tasks, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.temperature = temperature
        
        # 改进的权重初始化
        if init_strategy == "improved":
            # 使用Xavier初始化而不是默认初始化
            nn.init.xavier_normal_(self.attention_conv.weight)
            if self.attention_conv.bias is not None:
                # 初始化偏置使得每个任务有不同的初始关注度
                with torch.no_grad():
                    self.attention_conv.bias[0] = 0.1  # 检测任务
                    self.attention_conv.bias[1] = 0.0  # 驾驶区域任务  
                    self.attention_conv.bias[2] = 0.2  # 车道线任务 - 更高初始权重

    def forward(self, x):
        # x is the shared feature map
        task_attention_maps = self.attention_conv(x) / self.temperature
        task_attention_weights = self.sigmoid(task_attention_maps)
        
        # Multiply the shared features by the attention weights for each task
        task_specific_features = []
        for i in range(task_attention_weights.size(1)):
            # 添加残差连接以改善梯度流
            attention_weight = task_attention_weights[:, i:i+1, :, :]
            task_feature = x * attention_weight + x * 0.1  # 10%原始特征保留
            task_specific_features.append(task_feature)
            
        return task_specific_features



class AttentionMergeBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionMergeBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x should be a list of two tensors to merge
        if len(x) != 2:
            return sum(x)  # fallback to simple addition
            
        feat1, feat2 = x
        if feat1.shape != feat2.shape:
            return feat1 + feat2  # fallback
            
        # Attention-weighted fusion
        concat_feat = torch.cat([feat1, feat2], dim=1)
        attention_weights = self.attention(concat_feat)
        
        weighted_feat1 = feat1 * attention_weights[:, 0:1, :, :]
        weighted_feat2 = feat2 * attention_weights[:, 1:2, :, :]
        
        return weighted_feat1 + weighted_feat2



def improved_seg_head_init(module):
    """改进的分割头初始化"""
    if isinstance(module, nn.Conv2d):
        # 对于分割任务，使用更小的初始化方差
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            # 初始化偏置为小负值，使初始输出偏向背景
            nn.init.constant_(module.bias, -0.1)

