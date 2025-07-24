#!/usr/bin/env python3
"""
评估分析脚本
分析车道线分割在评估阶段可能存在的问题
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn.functional as F
import numpy as np
from lib.models.YOLOP_xy import MCnetFromYAML

def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """计算分割指标"""
    # 应用sigmoid和阈值
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
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
        'union': union.item(),
        'pred_pixels': pred_binary.sum().item(),
        'target_pixels': target_binary.sum().item()
    }

def test_segmentation_evaluation():
    """测试分割评估"""
    print("=== 车道线分割评估测试 ===")
    
    # 创建两个模型
    model_long = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx.yaml')
    model_short = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    
    model_long.eval()
    model_short.eval()
    
    # 生成测试数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 创建简单的车道线目标 - 两条垂直线
    ll_target = torch.zeros(batch_size, 2, 256, 256)
    # 背景通道 (索引0)
    ll_target[:, 0, :, :] = 1.0
    # 车道线通道 (索引1) - 两条垂直线
    ll_target[:, 1, :, 60:65] = 1.0   # 左车道线
    ll_target[:, 1, :, 190:195] = 1.0 # 右车道线
    # 背景通道应该减去前景
    ll_target[:, 0, :, :] -= ll_target[:, 1, :, :]
    
    print(f"目标统计:")
    print(f"  背景像素数: {(ll_target[:, 0] > 0.5).sum().item()}")
    print(f"  车道线像素数: {(ll_target[:, 1] > 0.5).sum().item()}")
    
    with torch.no_grad():
        # 长代码模型预测
        output_long = model_long(x)
        ll_pred_long = output_long[2]  # 车道线预测
        
        # 短代码模型预测
        output_short = model_short(x)
        ll_pred_short = output_short[2]  # 车道线预测
        
        print(f"\n长代码模型预测统计:")
        print(f"  原始输出范围: [{ll_pred_long.min().item():.4f}, {ll_pred_long.max().item():.4f}]")
        ll_sigmoid_long = torch.sigmoid(ll_pred_long)
        print(f"  Sigmoid后范围: [{ll_sigmoid_long.min().item():.4f}, {ll_sigmoid_long.max().item():.4f}]")
        print(f"  背景通道均值: {ll_sigmoid_long[:, 0].mean().item():.4f}")
        print(f"  车道线通道均值: {ll_sigmoid_long[:, 1].mean().item():.4f}")
        
        print(f"\n短代码模型预测统计:")
        print(f"  原始输出范围: [{ll_pred_short.min().item():.4f}, {ll_pred_short.max().item():.4f}]")
        ll_sigmoid_short = torch.sigmoid(ll_pred_short)
        print(f"  Sigmoid后范围: [{ll_sigmoid_short.min().item():.4f}, {ll_sigmoid_short.max().item():.4f}]")
        print(f"  背景通道均值: {ll_sigmoid_short[:, 0].mean().item():.4f}")
        print(f"  车道线通道均值: {ll_sigmoid_short[:, 1].mean().item():.4f}")
        
        # 计算指标 - 只看车道线通道 (索引1)
        print(f"\n=== 车道线通道 (索引1) 评估 ===")
        
        metrics_long = calculate_segmentation_metrics(
            ll_pred_long[:, 1], ll_target[:, 1]
        )
        metrics_short = calculate_segmentation_metrics(
            ll_pred_short[:, 1], ll_target[:, 1]
        )
        
        print(f"长代码模型指标:")
        for key, value in metrics_long.items():
            print(f"  {key}: {value:.6f}")
            
        print(f"\n短代码模型指标:")
        for key, value in metrics_short.items():
            print(f"  {key}: {value:.6f}")
        
        # 尝试不同的阈值
        print(f"\n=== 不同阈值下的IoU对比 ===")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for thresh in thresholds:
            metrics_long_t = calculate_segmentation_metrics(
                ll_pred_long[:, 1], ll_target[:, 1], threshold=thresh
            )
            metrics_short_t = calculate_segmentation_metrics(
                ll_pred_short[:, 1], ll_target[:, 1], threshold=thresh
            )
            
            print(f"阈值 {thresh}:")
            print(f"  长代码 IoU: {metrics_long_t['iou']:.6f}")
            print(f"  短代码 IoU: {metrics_short_t['iou']:.6f}")

def test_with_realistic_targets():
    """使用更真实的目标测试"""
    print(f"\n=== 真实样本测试 ===")
    
    # 创建更真实的车道线目标
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 创建更复杂的车道线模式
    ll_target = torch.zeros(batch_size, 2, 256, 256)
    
    # 第一个样本 - 曲线车道线
    h, w = 256, 256
    for i in range(h):
        y = int(w/4 + 20 * np.sin(i * np.pi / 64))  # 左曲线
        if 0 <= y < w-5:
            ll_target[0, 1, i, y:y+5] = 1.0
            
        y = int(3*w/4 - 15 * np.sin(i * np.pi / 48))  # 右曲线
        if 0 <= y < w-5:
            ll_target[0, 1, i, y:y+5] = 1.0
    
    # 第二个样本 - 直车道线
    ll_target[1, 1, :, 80:85] = 1.0   # 左直线
    ll_target[1, 1, :, 170:175] = 1.0 # 右直线
    
    # 设置背景
    ll_target[:, 0, :, :] = 1.0 - ll_target[:, 1, :, :]
    
    print(f"真实目标统计:")
    print(f"  样本0车道线像素: {ll_target[0, 1].sum().item()}")
    print(f"  样本1车道线像素: {ll_target[1, 1].sum().item()}")
    
    # 测试两个模型
    model_long = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx.yaml')
    model_short = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    
    model_long.eval()
    model_short.eval()
    
    with torch.no_grad():
        output_long = model_long(x)
        output_short = model_short(x)
        
        # 计算指标
        metrics_long = calculate_segmentation_metrics(
            output_long[2][:, 1], ll_target[:, 1]
        )
        metrics_short = calculate_segmentation_metrics(
            output_short[2][:, 1], ll_target[:, 1]
        )
        
        print(f"\n真实样本评估结果:")
        print(f"长代码模型 - IoU: {metrics_long['iou']:.6f}, 准确率: {metrics_long['accuracy']:.6f}")
        print(f"短代码模型 - IoU: {metrics_short['iou']:.6f}, 准确率: {metrics_short['accuracy']:.6f}")

if __name__ == "__main__":
    try:
        test_segmentation_evaluation()
        test_with_realistic_targets()
    except Exception as e:
        print(f"评估测试中出现错误: {e}")
        import traceback
        traceback.print_exc()