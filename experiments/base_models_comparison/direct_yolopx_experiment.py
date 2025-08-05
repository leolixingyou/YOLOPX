#!/usr/bin/env python3
"""
直接构建YOLOPX模型并运行冲突检测实验
避免复杂的YAML配置问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置路径
YOLOP_ROOT = Path(__file__).resolve().parent.parent.parent / 'yolop_series'
sys.path.insert(0, str(YOLOP_ROOT))

from data.bdd import BddDataset
from models.common_modules import ELANNet, PaFPNELAN, Conv, seg_head
from models.heads.yolox_head import YOLOXHead
from core.loss import get_loss
import torch.optim as optim
from easydict import EasyDict as edict
import yaml

class SimplifiedYOLOPX(nn.Module):
    """简化的YOLOPX模型，直接构建而不依赖YAML"""
    
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Backbone
        self.backbone = ELANNet(use_C2=True)
        
        # Neck
        self.neck = PaFPNELAN()
        
        # Detection Head
        self.det_head = YOLOXHead(
            num_classes=num_classes,
            width=0.75,
            strides=[8, 16, 32],
            in_channels=[128, 256, 512]
        )
        
        # 驾驶区域分割头
        self.da_seg = nn.Sequential(
            Conv(512, 256, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(256, 128, 3, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            Conv(128, 64, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(64, 2, 1, 1),
            seg_head('sigmoid')
        )
        
        # 车道线分割头
        self.ll_seg = nn.Sequential(
            Conv(128, 64, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(64, 32, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(32, 16, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(16, 2, 1, 1),
            seg_head('sigmoid')
        )
    
    def forward(self, x):
        # Backbone
        backbone_features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(backbone_features)
        # neck_features: (C2, c5, c8, c12, c13, c16, c19)
        
        # Detection
        det_out = self.det_head(neck_features)
        
        # Segmentation
        # 使用P5特征 (c19) 进行驾驶区域分割
        da_out = self.da_seg(neck_features[6])
        
        # 使用P3特征 (c13) 进行车道线分割
        ll_out = self.ll_seg(neck_features[4])
        
        return det_out, da_out, ll_out

def calculate_task_conflict(gradients):
    """计算任务冲突强度"""
    task_names = list(gradients.keys())
    if len(task_names) < 2:
        return 0.0
    
    conflicts = []
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            task1_grads = gradients[task_names[i]]
            task2_grads = gradients[task_names[j]]
            
            conflict = 0.0
            count = 0
            
            for g1, g2 in zip(task1_grads, task2_grads):
                if g1 is not None and g2 is not None:
                    g1_flat = g1.view(-1)
                    g2_flat = g2.view(-1)
                    
                    dot_product = torch.dot(g1_flat, g2_flat)
                    norm1 = torch.norm(g1_flat)
                    norm2 = torch.norm(g2_flat)
                    
                    if norm1 > 0 and norm2 > 0:
                        cos_sim = dot_product / (norm1 * norm2)
                        if cos_sim < 0:
                            conflict += abs(cos_sim.item())
                            count += 1
            
            if count > 0:
                conflicts.append(conflict / count)
    
    return np.mean(conflicts) if conflicts else 0.0

def run_yolopx_experiment(epochs=5):
    """运行YOLOPX冲突检测实验"""
    print("\n" + "="*60)
    print("运行 YOLOPX 冲突检测实验")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open(YOLOP_ROOT / 'cfgs/data/bdd100k_single_class.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    with open(YOLOP_ROOT / 'cfgs/train_default.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    # 创建配置对象
    cfg = edict()
    cfg.DATASET = edict(data_cfg['DATASET'])
    cfg.AUGMENTATION = edict(data_cfg['AUGMENTATION'])
    cfg.TRAIN = edict(train_cfg['TRAIN'])
    cfg.TEST = edict(train_cfg['TEST'])
    cfg.LOSS = edict(train_cfg['LOSS'])
    cfg.MODEL = edict(train_cfg['MODEL'])
    cfg.WORKERS = train_cfg['WORKERS']
    cfg.PIN_MEMORY = train_cfg['PIN_MEMORY']
    
    # 添加必要的属性
    cfg.mosaic_rate = cfg.AUGMENTATION.MOSAIC_RATE
    cfg.mixup_rate = cfg.AUGMENTATION.MIXUP_RATE
    cfg.num_seg_class = cfg.DATASET.NUM_SEG_CLASS
    cfg.DATASET.HSV_H = cfg.AUGMENTATION.HSV_H
    cfg.DATASET.HSV_S = cfg.AUGMENTATION.HSV_S
    cfg.DATASET.HSV_V = cfg.AUGMENTATION.HSV_V
    cfg.MODEL.NC = cfg.DATASET.NC
    
    # 创建transform
    def numpy_to_tensor(img):
        if isinstance(img, np.ndarray):
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img)
        return img
    
    # 创建数据集
    print("创建数据集...")
    dataset = BddDataset(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.DATASET.IMAGE_SIZE,
        transform=numpy_to_tensor
    )
    
    # 限制数据集大小
    dataset.db = dataset.db[:200]
    print(f"数据集大小: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # 使用0避免多进程问题
        pin_memory=False,
        drop_last=True,
        collate_fn=dataset.collate_fn
    )
    
    # 创建模型
    print("创建YOLOPX模型...")
    model = SimplifiedYOLOPX(num_classes=cfg.DATASET.NC)
    model = model.to(device)
    
    # 创建损失函数
    criterion = get_loss(cfg, device, model)
    
    # 创建优化器
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    optimizer = optim.AdamW(pg0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay': cfg.TRAIN.WD})
    optimizer.add_param_group({'params': pg2})
    
    # 训练循环
    model.train()
    tci_history = []
    
    for epoch in range(epochs):
        epoch_tci = []
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, batch in enumerate(pbar):
            imgs, labels_list, paths, shapes = batch
            labels, seg_label, lane_label = labels_list
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            seg_label = seg_label.to(device)
            lane_label = lane_label.to(device)
            
            target = [labels, seg_label, lane_label]
            
            optimizer.zero_grad()
            
            try:
                outputs = model(imgs)
                loss, loss_dict = criterion(outputs, target, None, model)
                
                if not torch.isfinite(loss) or loss == 0:
                    continue
                
                # 计算任务冲突
                task_gradients = {}
                
                # 检测任务梯度
                if len(loss_dict) > 0 and torch.isfinite(loss_dict[0]) and loss_dict[0] > 0:
                    grads = torch.autograd.grad(loss_dict[0], model.parameters(), 
                                              retain_graph=True, allow_unused=True)
                    task_gradients['detection'] = [g.detach().clone() if g is not None else None for g in grads]
                
                # 驾驶区域分割梯度
                if len(loss_dict) > 1 and torch.isfinite(loss_dict[1]) and loss_dict[1] > 0:
                    grads = torch.autograd.grad(loss_dict[1], model.parameters(), 
                                              retain_graph=True, allow_unused=True)
                    task_gradients['segmentation'] = [g.detach().clone() if g is not None else None for g in grads]
                
                # 车道线分割梯度
                if len(loss_dict) > 2 and torch.isfinite(loss_dict[2]) and loss_dict[2] > 0:
                    grads = torch.autograd.grad(loss_dict[2], model.parameters(), 
                                              retain_graph=True, allow_unused=True)
                    task_gradients['lane'] = [g.detach().clone() if g is not None else None for g in grads]
                
                if len(task_gradients) >= 2:
                    tci = calculate_task_conflict(task_gradients)
                    epoch_tci.append(tci)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if epoch_tci:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tci': f'{np.mean(epoch_tci):.4f}'})
                
            except Exception as e:
                print(f"\n训练步骤错误: {e}")
                continue
        
        avg_tci = np.mean(epoch_tci) if epoch_tci else 0.0
        tci_history.append(avg_tci)
        print(f"Epoch {epoch+1} - 平均TCI: {avg_tci:.4f}")
    
    final_tci = np.mean(tci_history)
    print(f"\nYOLOPX 最终TCI: {final_tci:.4f}")
    
    # 保存结果
    results = {
        'model': 'yolopx',
        'epochs': epochs,
        'final_tci': float(final_tci),
        'tci_history': [float(x) for x in tci_history],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_dir = Path(__file__).parent.parent / 'runs' / 'conflict_detection'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'yolopx_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return final_tci

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    
    args = parser.parse_args()
    
    run_yolopx_experiment(args.epochs)