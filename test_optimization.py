#!/usr/bin/env python3
"""
测试优化效果
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn as nn
from lib.models.YOLOP_xy import MCnetFromYAML
from lib.core.improved_loss import get_improved_loss
from lib.config import cfg_xy as cfg

def test_optimized_model():
    """测试优化模型"""
    print("=== 测试优化模型 ===")
    
    # 加载原始和优化模型
    try:
        model_original = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
        print("✓ 原始模型加载成功")
    except Exception as e:
        print(f"✗ 原始模型加载失败: {e}")
        return
    
    # 测试前向传播
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    model_original.train()
    # 不使用no_grad，以便计算梯度
    output_original = model_original(x)
    
    print(f"原始模型输出:")
    print(f"  检测输出: {len(output_original[0])} 个尺度")
    print(f"  DA分割输出: {output_original[1].shape}")
    print(f"  LL分割输出: {output_original[2].shape}")
    
    # 测试损失计算
    device = torch.device('cpu')
    criterion_improved = get_improved_loss(cfg, device, model_original)
    
    # 创建测试目标
    det_targets = torch.zeros(batch_size, 50, 5)
    da_targets = torch.zeros(batch_size, 2, 256, 256)
    ll_targets = torch.zeros(batch_size, 2, 256, 256)
    
    # 添加一些目标
    da_targets[:, 1, 100:150, 100:150] = 1.0
    ll_targets[:, 1, 120:130, 50:200] = 1.0
    
    targets = [det_targets, da_targets, ll_targets]
    shapes = [(256, 256)] * batch_size
    
    # 计算改进损失
    total_loss, head_losses = criterion_improved(output_original, targets, shapes, model_original, x)
    
    print(f"\n改进损失函数测试:")
    print(f"  总损失: {total_loss.item():.6f}")
    print(f"  检测损失: {head_losses[0].item():.6f}")
    print(f"  DA分割损失: {head_losses[1].item():.6f}")
    print(f"  LL分割损失: {head_losses[2].item():.6f}")
    print(f"  LL IoU损失: {head_losses[3].item():.6f}")
    
    # 验证梯度流
    total_loss.backward()
    
    grad_norms = {}
    for name, param in model_original.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    # 找出车道线相关的梯度
    ll_gradients = {k: v for k, v in grad_norms.items() 
                   if any(keyword in k.lower() for keyword in ['seg', '33', '34', 'lane'])}
    
    print(f"\n车道线相关梯度:")
    for name, grad_norm in sorted(ll_gradients.items()):
        print(f"  {name}: {grad_norm:.8f}")
    
    print("\n优化测试完成!")

if __name__ == "__main__":
    test_optimized_model()
