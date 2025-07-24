#!/usr/bin/env python3
"""
找到输出为零的具体层
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
from lib.models.YOLOP_xy import MCnetFromYAML

def find_zero_output_layer():
    """找到开始输出为零的层"""
    print("=== 找到零输出的起始层 ===")
    
    model = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    model.eval()
    
    x = torch.randn(1, 3, 256, 256)
    
    # 逐层执行并检查
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
            
            # 检查输出是否为零或常数
            layer_name = str(type(layer).__name__)
            
            if hasattr(layer_output, 'shape'):
                output_std = layer_output.std().item()
                output_mean = layer_output.mean().item()
                output_max = layer_output.max().item()
                output_min = layer_output.min().item()
                
                is_zero = output_std < 1e-8 and abs(output_mean) < 1e-8
                is_constant = output_std < 1e-8
                
                print(f"层 {i:2d} ({layer_name:15s}): "
                      f"形状={str(layer_output.shape):20s}, "
                      f"范围=[{output_min:8.6f}, {output_max:8.6f}], "
                      f"均值={output_mean:8.6f}, "
                      f"标准差={output_std:8.6f}")
                
                if is_zero:
                    print(f"    ❌ 输出为零！")
                    
                    # 详细分析这一层
                    analyze_layer_details(layer, layer_input, i)
                    
                    # 如果找到了第一个零输出层，停止
                    if i > 20:  # 跳过前面的正常层
                        break
                
                elif is_constant:
                    print(f"    ⚠️  输出为常数！")
                    
        except Exception as e:
            print(f"层 {i} 执行失败: {e}")
            break

def analyze_layer_details(layer, layer_input, layer_idx):
    """分析具体层的详情"""
    print(f"\\n=== 详细分析层 {layer_idx} ===")
    
    layer_name = str(type(layer).__name__)
    print(f"层类型: {layer_name}")
    
    if hasattr(layer_input, 'shape'):
        print(f"输入统计:")
        print(f"  形状: {layer_input.shape}")
        print(f"  范围: [{layer_input.min().item():.6f}, {layer_input.max().item():.6f}]")
        print(f"  均值: {layer_input.mean().item():.6f}")
        print(f"  标准差: {layer_input.std().item():.6f}")
        
        input_is_zero = layer_input.std().item() < 1e-8 and abs(layer_input.mean().item()) < 1e-8
        if input_is_zero:
            print(f"  ❌ 输入已经为零！")
        else:
            print(f"  ✓ 输入正常")
    
    # 如果是ELANBlock_Head，深入分析
    if layer_name == 'ELANBlock_Head':
        print(f"\\nELANBlock_Head 内部分析:")
        
        # 检查每个子模块
        for name, module in layer.named_modules():
            if len(name) > 0:  # 跳过自身
                print(f"  子模块 {name}:")
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight
                    print(f"    权重形状: {weight.shape}")
                    print(f"    权重范围: [{weight.min().item():.8f}, {weight.max().item():.8f}]")
                    print(f"    权重均值: {weight.mean().item():.8f}")
                    print(f"    权重标准差: {weight.std().item():.8f}")
                
                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias
                    print(f"    偏置范围: [{bias.min().item():.8f}, {bias.max().item():.8f}]")
                
                # 如果是BatchNorm，检查running统计
                if 'BatchNorm' in str(type(module)):
                    print(f"    BatchNorm统计:")
                    if hasattr(module, 'running_mean'):
                        print(f"      running_mean: [{module.running_mean.min().item():.8f}, {module.running_mean.max().item():.8f}]")
                    if hasattr(module, 'running_var'):
                        print(f"      running_var: [{module.running_var.min().item():.8f}, {module.running_var.max().item():.8f}]")
                    if hasattr(module, 'num_batches_tracked'):
                        print(f"      num_batches_tracked: {module.num_batches_tracked.item()}")

def test_elan_block_manually():
    """手动测试ELANBlock_Head"""
    print(f"\\n=== 手动测试ELANBlock_Head ===")
    
    from lib.models.common import ELANBlock_Head
    
    # 创建ELANBlock_Head
    elan_block = ELANBlock_Head(128, 64)
    elan_block.eval()
    
    print(f"ELANBlock_Head结构:")
    for name, module in elan_block.named_modules():
        if len(name) > 0:
            print(f"  {name}: {type(module).__name__}")
    
    # 创建测试输入
    test_input = torch.randn(1, 128, 128, 128)
    print(f"\\n测试输入:")
    print(f"  形状: {test_input.shape}")
    print(f"  范围: [{test_input.min().item():.6f}, {test_input.max().item():.6f}]")
    print(f"  均值: {test_input.mean().item():.6f}")
    print(f"  标准差: {test_input.std().item():.6f}")
    
    # 前向传播
    try:
        test_output = elan_block(test_input)
        print(f"\\n测试输出:")
        print(f"  形状: {test_output.shape}")
        print(f"  范围: [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
        print(f"  均值: {test_output.mean().item():.6f}")
        print(f"  标准差: {test_output.std().item():.6f}")
        
        if test_output.std().item() < 1e-8:
            print(f"  ❌ 新创建的ELANBlock也输出常数！")
        else:
            print(f"  ✓ 新创建的ELANBlock输出正常")
            
    except Exception as e:
        print(f"ELANBlock测试失败: {e}")

def main():
    """主函数"""
    print("开始查找零输出层...")
    
    # 1. 找到零输出的起始层
    find_zero_output_layer()
    
    # 2. 手动测试ELANBlock
    test_elan_block_manually()
    
    print(f"\\n=== 分析结论 ===")
    print(f"问题很可能是:")
    print(f"1. 某个BatchNorm层的running_var为0，导致除零")
    print(f"2. 某个层的权重被错误初始化为0")
    print(f"3. 模型加载了预训练权重，但权重不匹配")

if __name__ == "__main__":
    main()