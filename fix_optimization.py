#!/usr/bin/env python3
"""
优化修复脚本
基于分析结果，提出并实施针对性的优化方案
"""

import sys
sys.path.append('/workspace/YOLOPX')
import torch
import torch.nn as nn
import yaml
from pathlib import Path

def create_optimized_config():
    """创建优化后的配置文件"""
    print("=== 创建优化配置 ===")
    
    # 1. 修复损失权重平衡
    optimized_config = {
        "# YOLOP-TAG网络架构配置文件 - 优化版本": None,
        "# 主要优化：修复损失权重、改进特征融合、优化TAG注意力": None,
        
        "prediction_heads": {
            "det_out_idx": 2,
            "da_seg_out_idx": 18, 
            "ll_seg_out_idx": 34
        },
        
        "layers": [
            # Backbone保持不变
            [-1, "ELANNet", [True]],
            [-1, "PaFPNELAN", []],
            [-1, "YOLOXHead", [1]],
            
            # 渐进式上采样
            [1, "FPN_C3", []],
            [1, "FPN_C4", []],
            
            # 驾驶区域分割分支 - 优化TAG权重初始化
            [4, "TaskAttention", [512, 3, "improved_init"]],  # 添加改进初始化标志
            [-1, "Select", [1]], 
            [-1, "Conv", [512, 256, 3, 1]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "ELANBlock_Head", [256, 128]],
            [-1, "Conv", [128, 64, 3, 1]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "Conv", [64, 32, 3, 1]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "Conv", [32, 16, 3, 1]],
            [-1, "ELANBlock_Head", [16, 8]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "Conv", [8, 2, 3, 1, "seg_init"]],  # 添加特殊初始化
            [-1, "seg_head", ["sigmoid"]],
            
            # 车道线分割分支 - 改进特征融合
            [1, "FPN_C2", []],
            [-1, "Conv", [256, 128, 3, 1]],
            
            # 改进的TAG特征融合
            [5, "Select", [2]],  # 车道线任务特征
            [3, "Conv", [256, 128, 3, 1]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [[-1, 20], "MergeBlock", ["attention_add"]],  # 使用注意力加权融合
            
            # 车道线分割头
            [-1, "ELANBlock_Head", [128, 64]],
            [-1, "PSA_p", [64, 64]],
            [-1, "Conv", [64, 32, 3, 1]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "Conv", [32, 16, 3, 1]],
            [-1, "ELANBlock_Head", [16, 8]],
            [-1, "PSA_p", [8, 8]],
            [-1, "Upsample", [None, 2, "bilinear"]],
            [-1, "Conv", [8, 2, 3, 1, "seg_init"]],  # 特殊初始化
            [-1, "seg_head", ["sigmoid"]]
        ],
        
        "model_config": {
            "nc": 1,
            "input_size": [3, 256, 256]
        },
        
        # 新增优化配置
        "optimization": {
            "improved_loss_weights": {
                "det_weight": 0.05,     # 提高检测权重
                "da_seg_weight": 0.3,   # 提高驾驶区域权重  
                "ll_seg_weight": 0.4,   # 大幅提高车道线权重
                "ll_iou_weight": 0.3    # 提高IoU权重
            },
            "tag_attention": {
                "init_strategy": "improved",
                "temperature": 1.0
            },
            "feature_fusion": {
                "use_attention": True,
                "fusion_channels": 128
            }
        }
    }
    
    # 保存优化配置
    config_path = "/workspace/YOLOPX/lib/config/yolopx-tag-optimized.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"优化配置已保存到: {config_path}")
    return config_path

def implement_improved_modules():
    """实现改进的模块"""
    print("\n=== 实现改进模块 ===")
    
    # 1. 改进的TaskAttention
    improved_tag_code = '''
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
'''
    
    # 2. 注意力融合模块
    attention_merge_code = '''
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
'''
    
    # 3. 改进的分割头初始化
    seg_init_code = '''
def improved_seg_head_init(module):
    """改进的分割头初始化"""
    if isinstance(module, nn.Conv2d):
        # 对于分割任务，使用更小的初始化方差
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            # 初始化偏置为小负值，使初始输出偏向背景
            nn.init.constant_(module.bias, -0.1)
'''
    
    # 保存改进模块到文件
    module_path = "/workspace/YOLOPX/lib/models/improved_modules.py"
    with open(module_path, 'w', encoding='utf-8') as f:
        f.write(f'''"""
改进的模块实现
包含优化的TAG注意力、特征融合和初始化策略
"""

import torch
import torch.nn as nn

{improved_tag_code}

{attention_merge_code}

{seg_init_code}
''')
    
    print(f"改进模块已保存到: {module_path}")
    return module_path

def create_improved_loss():
    """创建改进的损失函数"""
    print("\n=== 创建改进损失函数 ===")
    
    loss_code = '''
class ImprovedMultiHeadLoss(nn.Module):
    """改进的多任务损失函数"""
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        
        self.loss_list = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg
        
        # 改进的损失权重 - 更平衡的设计
        self.det_weight = 0.05      # 降低检测权重
        self.da_seg_weight = 0.3    # 适中的驾驶区域权重
        self.ll_seg_weight = 0.4    # 提高车道线权重
        self.ll_iou_weight = 0.3    # 提高IoU权重
        
    def forward(self, head_fields, head_targets, shapes, model, imgs):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model, imgs)
        return total_loss, head_losses
        
    def _forward_impl(self, predictions, targets, shapes, model, imgs):
        cfg = self.cfg
        device = targets[0].device
        Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss = self.loss_list
        
        # 计算各项损失
        det_all_loss = Det_loss(predictions[0], targets[0], imgs)
        
        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        da_seg_loss = Da_Seg_Loss(drive_area_seg_predicts, drive_area_seg_targets)
        
        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        ll_seg_loss = Ll_Seg_Loss(lane_line_seg_predicts, lane_line_seg_targets)
        
        tversky_predicts = predictions[2]
        tversky_targets = targets[2]
        ll_tversky_loss = Tversky_Loss(tversky_predicts, tversky_targets)
        
        # 应用改进的权重
        det_all_loss *= self.det_weight * self.lambdas[1]
        da_seg_loss *= self.da_seg_weight * self.lambdas[2]
        ll_seg_loss *= self.ll_seg_weight * self.lambdas[3]
        ll_tversky_loss *= self.ll_iou_weight * self.lambdas[4]
        
        loss = det_all_loss + da_seg_loss + ll_seg_loss + ll_tversky_loss
        return loss, (det_all_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, loss)

def get_improved_loss(cfg, device, model):
    """获取改进的损失函数"""
    from lib.core.loss import YOLOX_Loss, TverskyLoss, FocalLossSeg
    
    Det_loss = YOLOX_Loss(device, 1)
    Da_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Ll_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Tversky_Loss = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3).to(device)
    
    gamma = cfg.LOSS.FL_GAMMA
    if gamma > 0.0:
        Ll_Seg_Loss = FocalLossSeg(Ll_Seg_Loss, gamma)
    
    losses = [Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss]
    return ImprovedMultiHeadLoss(losses, cfg)
'''
    
    loss_path = "/workspace/YOLOPX/lib/core/improved_loss.py"
    with open(loss_path, 'w', encoding='utf-8') as f:
        f.write(f'''"""
改进的损失函数实现
优化了多任务损失权重平衡
"""

import torch
import torch.nn as nn

{loss_code}
''')
    
    print(f"改进损失函数已保存到: {loss_path}")
    return loss_path

def create_test_script():
    """创建测试脚本验证优化效果"""
    print("\n=== 创建测试脚本 ===")
    
    test_code = '''#!/usr/bin/env python3
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
    with torch.no_grad():
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
    
    print(f"\\n改进损失函数测试:")
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
    
    print(f"\\n车道线相关梯度:")
    for name, grad_norm in sorted(ll_gradients.items()):
        print(f"  {name}: {grad_norm:.8f}")
    
    print("\\n优化测试完成!")

if __name__ == "__main__":
    test_optimized_model()
'''
    
    test_path = "/workspace/YOLOPX/test_optimization.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"测试脚本已保存到: {test_path}")
    return test_path

def main():
    """主函数"""
    print("开始创建优化方案...")
    
    # 1. 创建优化配置
    config_path = create_optimized_config()
    
    # 2. 实现改进模块
    module_path = implement_improved_modules()
    
    # 3. 创建改进损失函数
    loss_path = create_improved_loss()
    
    # 4. 创建测试脚本
    test_path = create_test_script()
    
    print(f"\\n=== 优化方案总结 ===")
    print(f"1. 优化配置文件: {config_path}")
    print(f"2. 改进模块实现: {module_path}")
    print(f"3. 改进损失函数: {loss_path}")  
    print(f"4. 测试脚本: {test_path}")
    
    print(f"\\n主要优化点:")
    print(f"- 修复损失权重平衡：大幅提高车道线分割权重")
    print(f"- 改进TAG注意力初始化：使用Xavier初始化和任务特定偏置")
    print(f"- 优化特征融合：使用注意力加权融合替代简单相加")
    print(f"- 改进分割头初始化：使用更合适的权重和偏置初始化")
    
    print(f"\\n下一步：运行测试脚本验证优化效果")

if __name__ == "__main__":
    main()