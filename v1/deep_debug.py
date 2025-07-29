#!/usr/bin/env python3
"""
深度调试脚本
找出车道线分割输出为常数的根本原因
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn as nn
from lib.models.YOLOP_xy import MCnetFromYAML

def trace_forward_pass():
    """追踪前向传播过程"""
    print("=== 深度调试：追踪前向传播 ===")
    
    # 创建模型
    model = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    model.eval()
    
    # 生成输入
    x = torch.randn(1, 3, 256, 256)
    
    print(f"输入统计:")
    print(f"  形状: {x.shape}")
    print(f"  范围: [{x.min().item():.6f}, {x.max().item():.6f}]")
    print(f"  均值: {x.mean().item():.6f}")
    
    # 逐层追踪
    intermediate = x
    layer_outputs = []
    
    for i, layer in enumerate(model.model):
        try:
            # 获取输入
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
            
            # 前向传播
            layer_output = layer(layer_input)
            layer_outputs.append(layer_output)
            intermediate = layer_output
            
            # 分析关键层
            layer_name = str(type(layer).__name__)
            
            # 特别关注车道线分割相关层
            if i in [30, 31, 32, 33, 34]:  # 车道线分割头附近的层
                print(f"\\n层 {i} ({layer_name}):")
                
                if hasattr(layer_output, 'shape'):
                    print(f"  输出形状: {layer_output.shape}")
                    print(f"  输出范围: [{layer_output.min().item():.6f}, {layer_output.max().item():.6f}]")
                    print(f"  输出均值: {layer_output.mean().item():.6f}")
                    print(f"  输出标准差: {layer_output.std().item():.6f}")
                    
                    # 检查是否为常数
                    if torch.all(torch.abs(layer_output - layer_output.mean()) < 1e-6):
                        print(f"  ❌ 输出为常数！")
                        
                        # 如果是Conv层，检查权重
                        if layer_name == 'Conv' and hasattr(layer, 'conv'):
                            conv_layer = layer.conv
                            print(f"  Conv权重检查:")
                            print(f"    权重形状: {conv_layer.weight.shape}")
                            print(f"    权重范围: [{conv_layer.weight.min().item():.8f}, {conv_layer.weight.max().item():.8f}]")
                            print(f"    权重均值: {conv_layer.weight.mean().item():.8f}")
                            print(f"    权重标准差: {conv_layer.weight.std().item():.8f}")
                            
                            if conv_layer.bias is not None:
                                print(f"    偏置范围: [{conv_layer.bias.min().item():.8f}, {conv_layer.bias.max().item():.8f}]")
                            
                            # 检查输入
                            if hasattr(layer_input, 'shape'):
                                print(f"  输入统计:")
                                print(f"    输入形状: {layer_input.shape}")
                                print(f"    输入范围: [{layer_input.min().item():.6f}, {layer_input.max().item():.6f}]")
                                print(f"    输入均值: {layer_input.mean().item():.6f}")
                                
                                if torch.all(torch.abs(layer_input - layer_input.mean()) < 1e-6):
                                    print(f"    ❌ 输入也是常数！")
                                else:
                                    print(f"    ✓ 输入正常")
                    else:
                        print(f"  ✓ 输出正常")
                
                # 如果是seg_head，检查激活函数
                if layer_name == 'seg_head':
                    print(f"  seg_head详情:")
                    print(f"    激活函数: {type(layer.act).__name__}")
                    print(f"    训练模式: {layer.training}")
                    print(f"    模型训练模式: {model.training}")
                    
        except Exception as e:
            print(f"层 {i} 执行失败: {e}")
            break
    
    return layer_outputs

def analyze_constant_output_cause():
    """分析常数输出的原因"""
    print("\n=== 分析常数输出原因 ===")
    
    # 1. 检查是否所有层都输出常数
    model = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    model.eval()
    
    x = torch.randn(1, 3, 256, 256)
    
    # 手动执行关键层
    print("手动执行车道线分割分支:")
    
    # 执行到ELANBlock_Head (层30)
    intermediate = x
    layer_outputs = []
    for i in range(31):  # 执行到层30
        layer = model.model[i]
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
    
    # 现在检查层30的输出
    layer30_output = layer_outputs[30]
    print(f"层30 (ELANBlock_Head) 输出:")
    print(f"  形状: {layer30_output.shape}")
    print(f"  范围: [{layer30_output.min().item():.6f}, {layer30_output.max().item():.6f}]")
    print(f"  均值: {layer30_output.mean().item():.6f}")
    print(f"  标准差: {layer30_output.std().item():.6f}")
    
    # 执行PSA_p (层31)
    layer31 = model.model[31]
    layer31_output = layer31(layer30_output)
    print(f"\\n层31 (PSA_p) 输出:")
    print(f"  形状: {layer31_output.shape}")
    print(f"  范围: [{layer31_output.min().item():.6f}, {layer31_output.max().item():.6f}]")
    print(f"  均值: {layer31_output.mean().item():.6f}")
    print(f"  标准差: {layer31_output.std().item():.6f}")
    
    # 执行Conv (层32)
    layer32 = model.model[32]
    layer32_output = layer32(layer31_output)
    print(f"\\n层32 (Upsample) 输出:")
    print(f"  形状: {layer32_output.shape}")
    print(f"  范围: [{layer32_output.min().item():.6f}, {layer32_output.max().item():.6f}]")
    print(f"  均值: {layer32_output.mean().item():.6f}")
    print(f"  标准差: {layer32_output.std().item():.6f}")
    
    # 执行Conv (层33) - 这是最后的卷积层
    layer33 = model.model[33]
    print(f"\\n层33 (Conv) 权重统计:")
    print(f"  权重形状: {layer33.conv.weight.shape}")
    print(f"  权重范围: [{layer33.conv.weight.min().item():.8f}, {layer33.conv.weight.max().item():.8f}]")
    print(f"  权重均值: {layer33.conv.weight.mean().item():.8f}")
    print(f"  权重是否全为0: {torch.all(layer33.conv.weight == 0).item()}")
    
    layer33_output = layer33(layer32_output)
    print(f"\\n层33 (Conv) 输出:")
    print(f"  形状: {layer33_output.shape}")
    print(f"  范围: [{layer33_output.min().item():.6f}, {layer33_output.max().item():.6f}]")
    print(f"  均值: {layer33_output.mean().item():.6f}")
    print(f"  标准差: {layer33_output.std().item():.6f}")
    
    # 检查输入的每个通道
    print(f"\\n检查输入到Conv33的每个通道:")
    for ch in range(layer32_output.shape[1]):
        ch_data = layer32_output[0, ch]
        print(f"  通道{ch}: 范围[{ch_data.min().item():.6f}, {ch_data.max().item():.6f}], 均值{ch_data.mean().item():.6f}")
    
    return layer33_output

def test_conv_layer_manually():
    """手动测试卷积层"""
    print("\n=== 手动测试卷积层 ===")
    
    # 创建简单的测试卷积层
    test_conv = nn.Conv2d(8, 2, kernel_size=3, padding=1, bias=False)
    
    # 使用相同的初始化
    nn.init.kaiming_normal_(test_conv.weight, mode='fan_out', nonlinearity='relu')
    
    print(f"测试卷积层权重:")
    print(f"  权重形状: {test_conv.weight.shape}")
    print(f"  权重范围: [{test_conv.weight.min().item():.8f}, {test_conv.weight.max().item():.8f}]")
    print(f"  权重均值: {test_conv.weight.mean().item():.8f}")
    
    # 创建测试输入
    test_input = torch.randn(1, 8, 256, 256)
    print(f"\\n测试输入:")
    print(f"  范围: [{test_input.min().item():.6f}, {test_input.max().item():.6f}]")
    print(f"  均值: {test_input.mean().item():.6f}")
    
    # 前向传播
    test_output = test_conv(test_input)
    print(f"\\n测试输出:")
    print(f"  范围: [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
    print(f"  均值: {test_output.mean().item():.6f}")
    print(f"  标准差: {test_output.std().item():.6f}")
    
    return test_output

def main():
    """主函数"""
    print("开始深度调试...")
    
    # 1. 追踪前向传播
    layer_outputs = trace_forward_pass()
    
    # 2. 分析常数输出原因
    layer33_output = analyze_constant_output_cause()
    
    # 3. 手动测试卷积层
    test_output = test_conv_layer_manually()
    
    print(f"\\n=== 调试总结 ===")
    print(f"问题定位：需要找出哪一层开始输出常数")
    print(f"可能原因：")
    print(f"1. 某个层的权重初始化问题")
    print(f"2. BatchNorm层的统计信息问题")
    print(f"3. 激活函数的数值溢出")
    print(f"4. 梯度传播被截断")

if __name__ == "__main__":
    main()