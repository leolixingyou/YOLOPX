#!/usr/bin/env python3
"""
验证修复效果的脚本
对比修复前后的性能表现
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn.functional as F
from lib.models.YOLOP_xy import MCnetFromYAML

def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """计算分割指标"""
    # 应用sigmoid和阈值
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred_sigmoid = torch.sigmoid(pred)
    else:
        pred_sigmoid = pred
    
    pred_binary = (pred_sigmoid > threshold).float()
    target_binary = target.float()
    
    # 计算交集和并集
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # IoU
    iou = intersection / (union + 1e-8)
    
    # 准确率
    total_pixels = target_binary.numel()
    correct_pixels = (pred_binary == target_binary).sum()
    accuracy = correct_pixels / total_pixels
    
    return {
        'iou': iou.item(),
        'accuracy': accuracy.item(),
        'intersection': intersection.item(),
        'union': union.item()
    }

def validate_model_performance():
    """验证模型性能"""
    print("=== 验证修复后模型性能 ===")
    
    # 创建模型
    model = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    model.eval()
    
    # 创建更真实的测试数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 创建车道线目标 - 模拟真实的车道线模式
    ll_target = torch.zeros(batch_size, 2, 256, 256)
    
    # 为每个样本创建不同的车道线模式
    import numpy as np
    for b in range(batch_size):
        # 左车道线 - 略微弯曲
        for i in range(256):
            y_left = int(64 + 10 * np.sin(i * np.pi / 128))
            if 0 <= y_left < 256-3:
                ll_target[b, 1, i, y_left:y_left+3] = 1.0
        
        # 右车道线 - 略微弯曲  
        for i in range(256):
            y_right = int(192 - 8 * np.sin(i * np.pi / 96))
            if 0 <= y_right < 256-3:
                ll_target[b, 1, i, y_right:y_right+3] = 1.0
    
    # 设置背景
    ll_target[:, 0, :, :] = 1.0 - ll_target[:, 1, :, :]
    
    print(f"测试数据统计:")
    print(f"  批次大小: {batch_size}")
    print(f"  车道线像素总数: {ll_target[:, 1].sum().item()}")
    print(f"  背景像素总数: {ll_target[:, 0].sum().item()}")
    
    with torch.no_grad():
        outputs = model(x)
        ll_pred = outputs[2]  # 车道线预测
        
        # 计算车道线通道的指标
        metrics = calculate_segmentation_metrics(ll_pred[:, 1], ll_target[:, 1])
        
        print(f"\n车道线分割性能:")
        print(f"  IoU: {metrics['iou']:.6f}")
        print(f"  准确率: {metrics['accuracy']:.6f}")
        print(f"  交集像素数: {metrics['intersection']:.0f}")
        print(f"  并集像素数: {metrics['union']:.0f}")
        
        # 测试不同阈值下的性能
        print(f"\n不同阈值下的IoU:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            metrics_t = calculate_segmentation_metrics(ll_pred[:, 1], ll_target[:, 1], threshold=thresh)
            print(f"  阈值 {thresh}: IoU = {metrics_t['iou']:.6f}")
        
        # 分析预测输出的分布
        pred_sigmoid = torch.sigmoid(ll_pred[:, 1])
        print(f"\n预测输出分析:")
        print(f"  预测范围: [{pred_sigmoid.min().item():.6f}, {pred_sigmoid.max().item():.6f}]")
        print(f"  预测均值: {pred_sigmoid.mean().item():.6f}")
        print(f"  预测标准差: {pred_sigmoid.std().item():.6f}")
        
        # 检查是否仍然是常数输出
        is_constant = torch.all(torch.abs(pred_sigmoid - pred_sigmoid.mean()) < 1e-6)
        print(f"  是否为常数输出: {is_constant.item()}")
        
    return metrics

if __name__ == "__main__":
    try:
        validate_model_performance()
        print("\n验证完成！")
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
