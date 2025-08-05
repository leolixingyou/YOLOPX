#!/usr/bin/env python3
"""
YOLOP系列单类检测实验脚本
支持YOLOPv1、v2、v3和YOLOPX的BDD100K单类检测实验
所有参数都有默认值，可以直接在VSCode中按F5运行
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from models.builder import get_net_from_yaml
from core.loss import MultiHeadLoss, get_loss
from data.bdd import BddDataset
from utils.utils import init_weights


def load_config(config_type='train'):
    """加载配置文件"""
    config_paths = {
        'train': PROJECT_ROOT / 'cfgs/train_default.yaml',
        'data': PROJECT_ROOT / 'cfgs/data/bdd100k_single_class.yaml'
    }
    
    with open(config_paths[config_type], 'r') as f:
        return yaml.safe_load(f)


def create_model(model_name, device):
    """创建模型"""
    model_configs = {
        'yolopv1': 'cfgs/models/yolop_v1_official.yaml',
        'yolopv2': 'cfgs/models/yolop.yaml',  # v2 使用 yolop.yaml
        'yolopv3': 'cfgs/models/yolop_v3_official.yaml',
        'yolopx': 'cfgs/models/yolopx.yaml'
    }
    
    model_path = PROJECT_ROOT / model_configs[model_name]
    print(f"加载模型配置: {model_path}")
    
    model = get_net_from_yaml(str(model_path))
    model = model.to(device)
    
    return model


def create_optimizer(model, train_cfg):
    """创建优化器"""
    # 分组参数
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    
    lr = train_cfg['TRAIN']['LR0']
    momentum = train_cfg['TRAIN']['MOMENTUM']
    weight_decay = train_cfg['TRAIN']['WD']
    
    if train_cfg['TRAIN']['OPTIMIZER'] == 'adamw':
        optimizer = optim.AdamW(pg0, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif train_cfg['TRAIN']['OPTIMIZER'] == 'adam':
        optimizer = optim.Adam(pg0, lr=lr, betas=(momentum, 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
    
    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': pg2})
    
    return optimizer


def run_experiment(model_name='yolopx', epochs=3, batch_size=2):
    """运行单个实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"运行实验: {model_name.upper()}")
    print(f"设备: {device}")
    print(f"批大小: {batch_size}, 训练轮数: {epochs}")
    print(f"{'='*60}\n")
    
    # 加载配置
    train_cfg = load_config('train')
    data_cfg = load_config('data')
    
    # 合并配置
    cfg = {**train_cfg, **data_cfg}
    cfg = type('Config', (), cfg)()  # 转换为对象
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = BddDataset(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.DATASET['IMAGE_SIZE'],
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    
    print(f"数据集大小: {len(train_dataset)}")
    print(f"批次数: {len(train_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(model_name, device)
    
    # 创建损失函数
    criterion = get_loss(cfg, device)
    
    # 创建优化器
    optimizer = create_optimizer(model, cfg)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 训练循环
    print("\n开始训练...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, (imgs, labels, seg_label, lane_label, _) in enumerate(pbar):
            # 数据移到GPU
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device)
            seg_label = seg_label.to(device)
            lane_label = lane_label.to(device)
            
            target = [labels, seg_label, lane_label]
            
            # 前向传播
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                shapes = None
                loss, loss_dict = criterion(outputs, target, shapes, model, imgs)
            
            # 检查损失有效性
            if not torch.isfinite(loss):
                print(f"警告: 损失为 {loss.item()}, 跳过该批次")
                continue
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新统计
            epoch_loss += loss.item()
            batch_count += 1
            
            # 更新进度条
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 定期打印详细信息
            if i % 10 == 0 and i > 0:
                det_loss = loss_dict[0].item() if len(loss_dict) > 0 else 0
                seg_loss = loss_dict[1].item() if len(loss_dict) > 1 else 0
                lane_loss = loss_dict[2].item() if len(loss_dict) > 2 else 0
                print(f"\n批次 {i}: 总损失={loss.item():.4f}, "
                      f"检测={det_loss:.4f}, 分割={seg_loss:.4f}, 车道线={lane_loss:.4f}")
        
        # 打印epoch总结
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        print(f"\nEpoch {epoch+1} 完成 - 平均损失: {avg_epoch_loss:.4f}")
    
    print(f"\n实验 {model_name} 完成!")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOP系列单类检测实验')
    parser.add_argument('--model', type=str, 
                       choices=['yolopv1', 'yolopv2', 'yolopv3', 'yolopx', 'all'],
                       default='yolopx',
                       help='选择要运行的模型')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批大小')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # 运行所有模型
        models = ['yolopv1', 'yolopv2', 'yolopv3', 'yolopx']
        for model_name in models:
            try:
                run_experiment(model_name, args.epochs, args.batch_size)
            except Exception as e:
                print(f"\n错误: 模型 {model_name} 运行失败")
                print(f"原因: {str(e)}")
                continue
    else:
        # 运行单个模型
        run_experiment(args.model, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()