#!/usr/bin/env python3
"""
YOLOPv1专用接口 - 处理Focus层的特殊输入要求
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
from easydict import EasyDict as edict

# 设置路径
YOLOP_ROOT = Path(__file__).resolve().parent.parent.parent / 'yolop_series'
sys.path.insert(0, str(YOLOP_ROOT))

from models.builder import get_net_from_yaml
from data.bdd import BddDataset
from core.loss import get_loss
import torch.optim as optim

class YOLOPv1Interface:
    """YOLOPv1专用接口，处理其特殊的输入要求"""
    
    def __init__(self):
        self.model_name = 'yolopv1'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_dataset(self, cfg, batch_size=2, num_images=100):
        """创建数据集"""
        def numpy_to_tensor(img):
            if isinstance(img, np.ndarray):
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img)
            return img
        
        dataset = BddDataset(
            cfg=cfg,
            is_train=True,
            inputsize=cfg.DATASET.IMAGE_SIZE,
            transform=numpy_to_tensor
        )
        
        # 限制数据集大小
        dataset.db = dataset.db[:num_images]
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=dataset.collate_fn
        )
        
        return dataloader
    
    def load_model(self):
        """加载YOLOPv1模型"""
        # 使用修正的配置文件
        model_cfg_path = Path(__file__).parent / 'yolopv1_fixed.yaml'
        
        # 检查Focus层的实际实现
        print(f"加载模型配置: {model_cfg_path}")
        
        try:
            model = get_net_from_yaml(str(model_cfg_path))
            model = model.to(self.device)
            
            # 打印模型第一层信息
            first_layer = model.layers[0]
            print(f"第一层类型: {type(first_layer).__name__}")
            
            # 测试输入
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            print(f"测试输入形状: {dummy_input.shape}")
            
            # 尝试通过第一层
            with torch.no_grad():
                try:
                    # Focus层会将输入切片并拼接，所以输入通道会变成4倍
                    output = first_layer(dummy_input)
                    print(f"第一层输出形状: {output.shape}")
                except Exception as e:
                    print(f"第一层前向传播错误: {e}")
                    
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def calculate_task_conflict(self, gradients):
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
    
    def run_experiment(self, epochs=3):
        """运行YOLOPv1实验"""
        print(f"\n{'='*60}")
        print(f"运行 YOLOPv1 专用冲突检测实验")
        print(f"{'='*60}\n")
        
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
        cfg.WORKERS = 0
        cfg.PIN_MEMORY = False
        
        # 添加必要的属性
        cfg.mosaic_rate = cfg.AUGMENTATION.MOSAIC_RATE
        cfg.mixup_rate = cfg.AUGMENTATION.MIXUP_RATE
        cfg.num_seg_class = cfg.DATASET.NUM_SEG_CLASS
        cfg.DATASET.HSV_H = cfg.AUGMENTATION.HSV_H
        cfg.DATASET.HSV_S = cfg.AUGMENTATION.HSV_S
        cfg.DATASET.HSV_V = cfg.AUGMENTATION.HSV_V
        cfg.MODEL.NC = cfg.DATASET.NC
        
        # 创建数据集
        print("创建数据集...")
        dataloader = self.create_dataset(cfg, batch_size=2, num_images=50)
        print(f"数据集大小: {len(dataloader.dataset)}")
        
        # 加载模型
        print("\n加载YOLOPv1模型...")
        model = self.load_model()
        
        if model is None:
            print("模型加载失败，终止实验")
            return None
        
        # 创建损失函数和优化器
        criterion = get_loss(cfg, self.device, model)
        
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
                
                # 调整输入尺寸以适应YOLOPv1
                # YOLOPv1期望640x640的输入
                if imgs.shape[-2:] != (640, 640):
                    imgs = torch.nn.functional.interpolate(
                        imgs, 
                        size=(640, 640), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                seg_label = seg_label.to(self.device)
                lane_label = lane_label.to(self.device)
                
                target = [labels, seg_label, lane_label]
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(imgs)
                    loss, loss_dict = criterion(outputs, target, shapes, model)
                    
                    if torch.isfinite(loss) and loss > 0:
                        # 计算任务冲突
                        task_gradients = {}
                        
                        if len(loss_dict) > 0 and torch.isfinite(loss_dict[0]) and loss_dict[0] > 0:
                            grads = torch.autograd.grad(loss_dict[0], model.parameters(), 
                                                      retain_graph=True, allow_unused=True)
                            task_gradients['detection'] = [g.detach().clone() if g is not None else None for g in grads]
                        
                        if len(loss_dict) > 1 and torch.isfinite(loss_dict[1]) and loss_dict[1] > 0:
                            grads = torch.autograd.grad(loss_dict[1], model.parameters(), 
                                                      retain_graph=True, allow_unused=True)
                            task_gradients['segmentation'] = [g.detach().clone() if g is not None else None for g in grads]
                        
                        if len(loss_dict) > 2 and torch.isfinite(loss_dict[2]) and loss_dict[2] > 0:
                            grads = torch.autograd.grad(loss_dict[2], model.parameters(), 
                                                      retain_graph=True, allow_unused=True)
                            task_gradients['lane'] = [g.detach().clone() if g is not None else None for g in grads]
                        
                        if len(task_gradients) >= 2:
                            tci = self.calculate_task_conflict(task_gradients)
                            epoch_tci.append(tci)
                        
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                        if epoch_tci:
                            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tci': f'{np.mean(epoch_tci):.4f}'})
                    
                except Exception as e:
                    print(f"\n训练步骤错误: {e}")
                    if i == 0:  # 如果第一个batch就失败，说明有严重问题
                        import traceback
                        traceback.print_exc()
                        return None
                    continue
            
            avg_tci = np.mean(epoch_tci) if epoch_tci else 0.0
            tci_history.append(avg_tci)
            print(f"Epoch {epoch+1} - 平均TCI: {avg_tci:.4f}, 平均损失: {epoch_loss/len(dataloader):.4f}")
        
        final_tci = np.mean(tci_history) if tci_history else 0.0
        print(f"\nYOLOPv1 最终TCI: {final_tci:.4f}")
        
        # 保存结果
        results = {
            'model': 'yolopv1',
            'epochs': epochs,
            'final_tci': float(final_tci),
            'tci_history': [float(x) for x in tci_history],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_dir = Path(__file__).parent.parent / 'runs' / 'conflict_detection'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'yolopv1_real_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return final_tci

if __name__ == '__main__':
    interface = YOLOPv1Interface()
    interface.run_experiment(epochs=3)