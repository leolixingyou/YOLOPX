#!/usr/bin/env python3
"""
初始化分析脚本
检查模型权重初始化和分割头的参数
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn as nn
from lib.models.YOLOP_xy import MCnetFromYAML

def analyze_model_weights(model, model_name):
    """分析模型权重"""
    print(f"\n=== {model_name} 权重分析 ===")
    
    total_params = 0
    zero_params = 0
    constant_params = 0
    
    seg_layers = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 检查零权重
        if torch.all(param == 0):
            zero_params += param.numel()
            print(f"零权重层: {name} - shape: {param.shape}")
        
        # 检查常数权重
        if param.numel() > 1 and torch.all(param == param.flatten()[0]):
            constant_params += param.numel()
            print(f"常数权重层: {name} - shape: {param.shape}, 值: {param.flatten()[0].item()}")
        
        # 收集分割相关层
        if any(keyword in name.lower() for keyword in ['seg', 'lane', 'll']):
            seg_layers.append((name, param))
    
    print(f"总参数: {total_params}")
    print(f"零参数: {zero_params}")
    print(f"常数参数: {constant_params}")
    
    # 分析分割头参数
    print(f"\n分割相关层分析:")
    for name, param in seg_layers:
        print(f"  {name}:")
        print(f"    形状: {param.shape}")
        print(f"    均值: {param.mean().item():.6f}")
        print(f"    标准差: {param.std().item():.6f}")
        print(f"    范围: [{param.min().item():.6f}, {param.max().item():.6f}]")

def analyze_segmentation_heads(model, model_name):
    """分析分割头结构"""
    print(f"\n=== {model_name} 分割头分析 ===")
    
    # 找到分割头
    seg_head_indices = []
    for i, layer in enumerate(model.model):
        if hasattr(layer, '__class__') and 'seg_head' in str(layer.__class__):
            seg_head_indices.append(i)
            print(f"分割头位置: {i}, 类型: {layer.__class__.__name__}")
            
            # 分析分割头内部结构
            if hasattr(layer, 'seg'):
                print(f"  分割层: {layer.seg}")
                if hasattr(layer.seg, 'weight'):
                    weight = layer.seg.weight
                    bias = layer.seg.bias
                    print(f"    权重形状: {weight.shape}")
                    print(f"    权重范围: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
                    if bias is not None:
                        print(f"    偏置范围: [{bias.min().item():.6f}, {bias.max().item():.6f}]")
    
    return seg_head_indices

def test_segmentation_forward(model, model_name):
    """测试分割前向传播"""
    print(f"\n=== {model_name} 分割前向传播测试 ===")
    
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    
    # 逐层前向传播，找到分割输出
    with torch.no_grad():
        intermediate = x
        layer_outputs = []
        
        for i, layer in enumerate(model.model):
            try:
                if hasattr(layer, 'from_'):
                    from_idx = layer.from_
                    if from_idx == -1:
                        layer_input = intermediate
                    elif isinstance(from_idx, list):
                        layer_input = [layer_outputs[j] for j in from_idx]
                    else:
                        layer_input = layer_outputs[from_idx]
                else:
                    layer_input = intermediate
                
                layer_output = layer(layer_input)
                layer_outputs.append(layer_output)
                intermediate = layer_output
                
                # 检查分割相关层
                layer_name = str(layer.__class__.__name__)
                if 'seg_head' in layer_name or i in [model.da_seg_out_idx, model.ll_seg_out_idx]:
                    print(f"层 {i} ({layer_name}) 输出:")
                    if hasattr(layer_output, 'shape'):
                        print(f"  形状: {layer_output.shape}")
                        print(f"  范围: [{layer_output.min().item():.6f}, {layer_output.max().item():.6f}]")
                        print(f"  均值: {layer_output.mean().item():.6f}")
                        print(f"  标准差: {layer_output.std().item():.6f}")
                        
                        # 检查是否为常数
                        if torch.all(layer_output == layer_output.flatten()[0]):
                            print(f"  !!! 输出为常数: {layer_output.flatten()[0].item()}")
                    
            except Exception as e:
                print(f"层 {i} 执行失败: {e}")

def check_weight_initialization():
    """检查权重初始化"""
    print("=== 权重初始化检查 ===")
    
    # 创建模型
    model_long = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx.yaml')
    model_short = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    
    # 分析权重
    analyze_model_weights(model_long, "长代码模型")
    analyze_model_weights(model_short, "短代码模型")
    
    # 分析分割头
    analyze_segmentation_heads(model_long, "长代码模型")
    analyze_segmentation_heads(model_short, "短代码模型")
    
    # 测试前向传播
    test_segmentation_forward(model_long, "长代码模型")
    test_segmentation_forward(model_short, "短代码模型")

if __name__ == "__main__":
    try:
        check_weight_initialization()
    except Exception as e:
        print(f"初始化检查中出现错误: {e}")
        import traceback
        traceback.print_exc()