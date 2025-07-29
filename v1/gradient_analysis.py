#!/usr/bin/env python3
"""
梯度流分析脚本
分析YOLOP-TAG模型中车道线分割任务的梯度流问题
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn as nn
from lib.models.YOLOP_xy import MCnetFromYAML
from lib.core.loss import get_loss

def analyze_gradient_flow(model_path, model_name):
    """分析模型梯度流"""
    print(f"\n=== 分析 {model_name} 梯度流 ===")
    
    # 创建模型
    model = MCnetFromYAML(model_path)
    model.train()
    
    # 创建损失函数
    import sys
    sys.path.append('/workspace/YOLOPX')
    from lib.config import cfg_xy as cfg
    device = torch.device('cpu')
    criterion = get_loss(cfg, device, model)
    
    # 生成测试数据
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    
    # 生成目标数据
    det_targets = torch.zeros(batch_size, 50, 5)  # [batch, max_objects, 5]
    da_targets = torch.zeros(batch_size, 2, 256, 256)
    ll_targets = torch.zeros(batch_size, 2, 256, 256)
    
    # 添加一些假目标
    da_targets[:, 1, 100:150, 100:150] = 1.0  # 驾驶区域
    ll_targets[:, 1, 120:130, 50:200] = 1.0   # 车道线
    
    targets = [det_targets, da_targets, ll_targets]
    shapes = [(256, 256)] * batch_size
    
    # 前向传播
    outputs = model(x)
    
    # 计算损失
    total_loss, head_losses = criterion(outputs, targets, shapes, model, x)
    
    print(f"总损失: {total_loss.item():.6f}")
    print(f"检测损失: {head_losses[0].item():.6f}")
    print(f"驾驶区域损失: {head_losses[1].item():.6f}")
    print(f"车道线BCE损失: {head_losses[2].item():.6f}")
    print(f"车道线Tversky损失: {head_losses[3].item():.6f}")
    
    # 反向传播
    total_loss.backward()
    
    # 分析关键层的梯度
    gradient_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_info[name] = grad_norm
        else:
            gradient_info[name] = 0.0
    
    # 找出重要的梯度信息
    important_layers = [name for name in gradient_info.keys() 
                       if any(keyword in name.lower() for keyword in ['seg', 'lane', 'll', 'task', 'attention'])]
    
    print(f"\n关键层梯度分析:")
    for layer_name in sorted(important_layers):
        grad_norm = gradient_info[layer_name]
        print(f"  {layer_name}: {grad_norm:.8f}")
    
    # 分析车道线分割输出
    ll_output = outputs[2]  # 车道线分割输出
    print(f"\n车道线输出统计:")
    print(f"  Shape: {ll_output.shape}")
    print(f"  Mean: {ll_output.mean().item():.6f}")
    print(f"  Std: {ll_output.std().item():.6f}")
    print(f"  Min: {ll_output.min().item():.6f}")
    print(f"  Max: {ll_output.max().item():.6f}")
    
    # 分析Sigmoid后的输出
    ll_sigmoid = torch.sigmoid(ll_output)
    print(f"  Sigmoid Mean: {ll_sigmoid.mean().item():.6f}")
    print(f"  Sigmoid Std: {ll_sigmoid.std().item():.6f}")
    
    return gradient_info, outputs

def compare_models():
    """对比两个模型的梯度流"""
    print("开始梯度流对比分析...")
    
    # 分析长代码模型
    long_gradients, long_outputs = analyze_gradient_flow(
        '/workspace/YOLOPX/lib/config/yolopx.yaml', '长代码模型'
    )
    
    # 分析短代码模型
    short_gradients, short_outputs = analyze_gradient_flow(
        '/workspace/YOLOPX/lib/config/yolopx-tag.yaml', '短代码模型'
    )
    
    # 对比分析
    print(f"\n=== 梯度对比分析 ===")
    
    # 找出共同的层
    common_layers = set(long_gradients.keys()) & set(short_gradients.keys())
    seg_layers = [layer for layer in common_layers 
                  if any(keyword in layer.lower() for keyword in ['seg', 'lane', 'll'])]
    
    print(f"分割相关层梯度对比:")
    for layer in sorted(seg_layers):
        long_grad = long_gradients[layer]
        short_grad = short_gradients[layer]
        ratio = short_grad / (long_grad + 1e-8)
        print(f"  {layer}:")
        print(f"    长代码: {long_grad:.8f}")
        print(f"    短代码: {short_grad:.8f}")
        print(f"    比例: {ratio:.4f}")
        
    # 对比输出
    print(f"\n输出对比:")
    ll_long = long_outputs[2]
    ll_short = short_outputs[2]
    
    print(f"车道线输出差异:")
    print(f"  长代码均值: {ll_long.mean().item():.6f}")
    print(f"  短代码均值: {ll_short.mean().item():.6f}")
    print(f"  差异: {(ll_short.mean() - ll_long.mean()).item():.6f}")

def diagnose_tag_attention():
    """诊断TAG注意力机制"""
    print(f"\n=== TAG注意力机制诊断 ===")
    
    from lib.models.tag_module import TaskAttention
    
    # 创建TAG模块
    tag = TaskAttention(512, 3)
    
    # 生成测试输入
    x = torch.randn(2, 512, 32, 32, requires_grad=True)
    
    # 前向传播
    task_features = tag(x)
    
    print(f"TAG输入: {x.shape}")
    print(f"TAG输出数量: {len(task_features)}")
    
    for i, feat in enumerate(task_features):
        print(f"  任务{i}特征: {feat.shape}")
        print(f"  任务{i}均值: {feat.mean().item():.6f}")
        print(f"  任务{i}标准差: {feat.std().item():.6f}")
    
    # 检查注意力权重
    attention_conv = tag.attention_conv
    attention_weights = torch.sigmoid(attention_conv(x))
    
    print(f"注意力权重形状: {attention_weights.shape}")
    for i in range(attention_weights.size(1)):
        weight_map = attention_weights[:, i]
        print(f"  任务{i}注意力 - 均值: {weight_map.mean().item():.6f}, 标准差: {weight_map.std().item():.6f}")
    
    # 模拟损失反向传播
    loss = sum(feat.mean() for feat in task_features)
    loss.backward()
    
    print(f"TAG模块梯度:")
    if tag.attention_conv.weight.grad is not None:
        grad_norm = tag.attention_conv.weight.grad.norm().item()
        print(f"  attention_conv梯度范数: {grad_norm:.8f}")
    
    if x.grad is not None:
        input_grad_norm = x.grad.norm().item()
        print(f"  输入梯度范数: {input_grad_norm:.8f}")

if __name__ == "__main__":
    try:
        compare_models()
        diagnose_tag_attention()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()