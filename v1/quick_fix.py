#!/usr/bin/env python3
"""
快速修复方案
直接修改损失权重来提升车道线分割性能
"""

import sys
sys.path.append('/workspace/YOLOPX')

def apply_quick_fix():
    """应用快速修复"""
    print("=== 应用快速修复方案 ===")
    
    # 1. 修改损失函数权重
    loss_file = "/workspace/YOLOPX/lib/core/loss.py"
    
    # 读取原始文件
    with open(loss_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原始文件
    with open(loss_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原始损失文件到: {loss_file}.backup")
    
    # 修改损失权重
    original_weights = [
        "det_all_loss *= 0.02 * self.lambdas[1]",
        "da_seg_loss *= 0.2 * self.lambdas[2]", 
        "ll_seg_loss *= 0.2 * self.lambdas[3]",
        "ll_tversky_loss *= 0.2 * self.lambdas[4]"
    ]
    
    improved_weights = [
        "det_all_loss *= 0.02 * self.lambdas[1]",  # 保持检测权重不变
        "da_seg_loss *= 0.2 * self.lambdas[2]",    # 保持驾驶区域权重不变
        "ll_seg_loss *= 0.6 * self.lambdas[3]",    # 提高车道线分割权重 (0.2 -> 0.6)
        "ll_tversky_loss *= 0.6 * self.lambdas[4]" # 提高车道线IoU权重 (0.2 -> 0.6)
    ]
    
    for orig, improved in zip(original_weights, improved_weights):
        content = content.replace(orig, improved)
    
    # 写入修改后的文件
    with open(loss_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("已修改损失权重:")
    print("- 车道线BCE损失权重: 0.2 -> 0.6 (3倍提升)")
    print("- 车道线Tversky损失权重: 0.2 -> 0.6 (3倍提升)")
    
    # 2. 修改配置文件中的分割权重
    config_file = "/workspace/YOLOPX/lib/config/default_xy.py"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # 备份配置文件
    with open(config_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"已备份原始配置文件到: {config_file}.backup")
    
    # 修改配置权重
    config_changes = [
        ("_C.LOSS.LL_SEG_GAIN = 0.2", "_C.LOSS.LL_SEG_GAIN = 0.4"),      # 提高车道线分割增益
        ("_C.LOSS.LL_IOU_GAIN = 0.2", "_C.LOSS.LL_IOU_GAIN = 0.4"),      # 提高车道线IoU增益
        ("_C.LOSS.DA_SEG_GAIN = 0.2", "_C.LOSS.DA_SEG_GAIN = 0.3")       # 适度提高驾驶区域增益
    ]
    
    for orig, improved in config_changes:
        config_content = config_content.replace(orig, improved)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("已修改配置权重:")
    print("- LL_SEG_GAIN: 0.2 -> 0.4")
    print("- LL_IOU_GAIN: 0.2 -> 0.4") 
    print("- DA_SEG_GAIN: 0.2 -> 0.3")
    
    return True

def test_quick_fix():
    """测试快速修复效果"""
    print("\n=== 测试快速修复效果 ===")
    
    import torch
    from lib.models.YOLOP_xy import MCnetFromYAML
    from lib.core.loss import get_loss
    from lib.config import cfg_xy as cfg
    
    # 创建模型
    model = MCnetFromYAML('/workspace/YOLOPX/lib/config/yolopx-tag.yaml')
    model.train()
    
    # 创建修改后的损失函数
    device = torch.device('cpu')
    criterion = get_loss(cfg, device, model)
    
    # 测试数据
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 创建目标
    det_targets = torch.zeros(batch_size, 50, 5)
    da_targets = torch.zeros(batch_size, 2, 256, 256)
    ll_targets = torch.zeros(batch_size, 2, 256, 256)
    
    # 添加真实目标
    da_targets[:, 1, 100:150, 100:150] = 1.0  # 驾驶区域
    ll_targets[:, 1, 120:130, 50:200] = 1.0   # 车道线
    
    targets = [det_targets, da_targets, ll_targets]
    shapes = [(256, 256)] * batch_size
    
    # 前向传播和损失计算
    outputs = model(x)
    total_loss, head_losses = criterion(outputs, targets, shapes, model, x)
    
    print(f"修复后损失函数测试:")
    print(f"  总损失: {total_loss.item():.6f}")
    print(f"  检测损失: {head_losses[0].item():.6f}")
    print(f"  DA分割损失: {head_losses[1].item():.6f}")
    print(f"  LL分割损失: {head_losses[2].item():.6f}") 
    print(f"  LL IoU损失: {head_losses[3].item():.6f}")
    
    # 计算各损失占比
    total_val = total_loss.item()
    print(f"\\n损失占比:")
    print(f"  检测损失占比: {head_losses[0].item()/total_val*100:.1f}%")
    print(f"  DA分割损失占比: {head_losses[1].item()/total_val*100:.1f}%")
    print(f"  LL分割损失占比: {head_losses[2].item()/total_val*100:.1f}%")
    print(f"  LL IoU损失占比: {head_losses[3].item()/total_val*100:.1f}%")
    
    # 测试梯度流
    total_loss.backward()
    
    # 检查车道线分割相关的梯度
    ll_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None and any(keyword in name for keyword in ['33', '34']):
            ll_gradients[name] = param.grad.norm().item()
    
    print(f"\\n车道线分割头梯度:")
    for name, grad_norm in sorted(ll_gradients.items()):
        print(f"  {name}: {grad_norm:.8f}")
    
    return total_loss.item(), head_losses

def create_validation_script():
    """创建验证脚本"""
    print("\n=== 创建验证脚本 ===")
    
    validation_code = '''#!/usr/bin/env python3
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
        
        print(f"\\n车道线分割性能:")
        print(f"  IoU: {metrics['iou']:.6f}")
        print(f"  准确率: {metrics['accuracy']:.6f}")
        print(f"  交集像素数: {metrics['intersection']:.0f}")
        print(f"  并集像素数: {metrics['union']:.0f}")
        
        # 测试不同阈值下的性能
        print(f"\\n不同阈值下的IoU:")
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            metrics_t = calculate_segmentation_metrics(ll_pred[:, 1], ll_target[:, 1], threshold=thresh)
            print(f"  阈值 {thresh}: IoU = {metrics_t['iou']:.6f}")
        
        # 分析预测输出的分布
        pred_sigmoid = torch.sigmoid(ll_pred[:, 1])
        print(f"\\n预测输出分析:")
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
        print("\\n验证完成！")
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
'''
    
    validation_path = "/workspace/YOLOPX/validate_fix.py"
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write(validation_code)
    
    print(f"验证脚本已保存到: {validation_path}")
    return validation_path

def main():
    """主函数"""
    print("开始应用快速修复方案...")
    
    # 1. 应用快速修复
    if apply_quick_fix():
        print("✓ 快速修复应用成功")
    
    # 2. 测试修复效果
    try:
        total_loss, head_losses = test_quick_fix()
        print("✓ 快速修复测试成功")
    except Exception as e:
        print(f"✗ 快速修复测试失败: {e}")
        return
    
    # 3. 创建验证脚本
    validation_path = create_validation_script()
    
    print(f"\\n=== 快速修复总结 ===")
    print(f"主要修改:")
    print(f"1. 损失函数权重调整:")
    print(f"   - 车道线BCE损失: 0.2 -> 0.6 (3倍)")
    print(f"   - 车道线Tversky损失: 0.2 -> 0.6 (3倍)")
    print(f"2. 配置文件权重调整:")
    print(f"   - LL_SEG_GAIN: 0.2 -> 0.4")
    print(f"   - LL_IOU_GAIN: 0.2 -> 0.4")
    print(f"   - DA_SEG_GAIN: 0.2 -> 0.3")
    
    print(f"\\n预期效果:")
    print(f"- 车道线分割损失权重提升约5-6倍")
    print(f"- 改善车道线分割的梯度流")
    print(f"- 提高模型对车道线分割任务的关注度")
    
    print(f"\\n下一步: 运行 'python3 validate_fix.py' 验证修复效果")

if __name__ == "__main__":
    main()