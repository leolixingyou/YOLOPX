#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspace/YOLOPX/v2')

# 替换原有的导入
import os
import time
import json
import yaml
import wandb
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from easydict import EasyDict as edict

# 使用统一的数据集模块
from data.unified_dataset import BddDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from models.builder import get_net_from_yaml
from utils.utils import get_optimizer

# 复制原有的实验运行器类，但使用新的数据集
class ExperimentRunner:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def run_single_experiment(self, model_config, data_config, train_config, experiment_name):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"开始实验: {experiment_name}")
        print(f"模型配置: {model_config.config_file}")
        print(f"{'='*60}\n")
        
        # 合并配置
        cfg = edict()
        cfg.update(train_config)
        cfg.update({'DATASET': data_config.DATASET})
        cfg.update({'model_family': model_config.model_family})
        
        # 初始化wandb
        wandb.init(
            project="yolop-conflict-analysis",
            name=experiment_name,
            config={
                'model': model_config.config_file,
                'epochs': cfg.TRAIN.END_EPOCH,
                'train_images': cfg.DATASET.NUMBER_IMAGE,
                'val_images': cfg.DATASET.get('NUMBER_VAL', cfg.DATASET.NUMBER_IMAGE // 5)
            }
        )
        
        # 创建数据集 - 使用统一的BddDataset
        print("加载数据集...")
        train_dataset = BddDataset(cfg=cfg, is_train=True, transform=None)
        val_dataset = BddDataset(cfg=cfg, is_train=False, transform=None)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=True,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=BddDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=BddDataset.collate_fn
        )
        
        # 创建模型
        model_cfg_path = f"cfgs/models/{model_config.config_file}"
        model = get_net_from_yaml(model_cfg_path).to(self.device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        # 创建损失函数和优化器
        criterion = get_loss(cfg, self.device)
        optimizer = get_optimizer(cfg, model)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type != 'cpu'))
        
        # 训练循环
        tci_values = []
        best_val_loss = float('inf')
        
        for epoch in range(cfg.TRAIN.END_EPOCH):
            print(f"\nEpoch [{epoch+1}/{cfg.TRAIN.END_EPOCH}]")
            
            # 训练
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for i, (images, targets, _, _) in enumerate(train_loader):
                if i >= 50:  # 限制每个epoch的batch数
                    break
                    
                images = images.to(self.device)
                targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in targets]
                
                # 前向传播
                outputs = model(images)
                total_loss, task_losses = criterion(outputs, targets, model=model)
                
                # 计算任务冲突
                if i % 10 == 0:
                    with torch.no_grad():
                        # 简化的TCI计算
                        losses = [task_losses['det_loss'], task_losses['da_seg_loss'], task_losses['ll_seg_loss']]
                        losses_np = [l.item() if hasattr(l, 'item') else l for l in losses]
                        tci = np.std(losses_np) / (np.mean(losses_np) + 1e-8)
                        tci_values.append(tci)
                
                # 反向传播
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += total_loss.item()
                num_batches += 1
                
                if i % 10 == 0:
                    print(f"  Batch [{i}/{min(50, len(train_loader))}], Loss: {total_loss.item():.4f}, TCI: {tci:.4f}")
            
            avg_train_loss = train_loss / num_batches
            avg_tci = np.mean(tci_values[-10:]) if tci_values else 0
            
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  平均TCI: {avg_tci:.4f}")
            
            # 验证
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for i, (images, targets, _, _) in enumerate(val_loader):
                    if i >= 20:  # 限制验证batch数
                        break
                        
                    images = images.to(self.device)
                    targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in targets]
                    
                    outputs = model(images)
                    total_loss, _ = criterion(outputs, targets, model=model)
                    val_loss += total_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            print(f"  验证损失: {avg_val_loss:.4f}")
            
            # 记录到wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'tci': avg_tci
            })
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        # 计算平均TCI
        avg_tci_overall = np.mean(tci_values) if tci_values else 0
        
        results = {
            'model': model_config.config_file,
            'avg_tci': avg_tci_overall,
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss
        }
        
        wandb.finish()
        
        return results

def main():
    # 解析参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_images', type=int, default=200)
    parser.add_argument('--val_images', type=int, default=100)
    parser.add_argument('--models', nargs='+', default=['yolopx_v2_anchor_free', 'yolop_v1_official', 'yolop_v3_official'])
    args = parser.parse_args()
    
    # 模型配置
    model_configs = {
        'yolopx_v2_anchor_free': {
            'config_file': 'yolopx.yaml',
            'model_family': 'yolopx'
        },
        'yolop_v1_official': {
            'config_file': 'yolop_v1_official.yaml',
            'model_family': 'yolop_v1_official'
        },
        'yolop_v3_official': {
            'config_file': 'yolop_v3_official.yaml',
            'model_family': 'yolop_v3_official'
        }
    }
    
    # 加载配置
    data_config = edict(yaml.safe_load(open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml')))
    train_config = edict(yaml.safe_load(open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml')))
    
    # 更新配置
    train_config.TRAIN.END_EPOCH = args.epochs
    data_config.DATASET.NUMBER_IMAGE = args.train_images
    data_config.DATASET.NUMBER_VAL = args.val_images
    
    # 运行实验
    runner = ExperimentRunner()
    all_results = []
    
    for model_name in args.models:
        if model_name not in model_configs:
            print(f"未知模型: {model_name}")
            continue
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{model_name}_{timestamp}"
        
        try:
            result = runner.run_single_experiment(
                edict(model_configs[model_name]),
                data_config,
                train_config,
                experiment_name
            )
            result['model_name'] = model_name
            all_results.append(result)
        except Exception as e:
            print(f"实验 {model_name} 失败: {str(e)}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'/workspace/YOLOPX/experiments/base_models_comparison/results/comparison_results_{timestamp}.json'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_config': {
                'epochs': args.epochs,
                'train_images': args.train_images,
                'val_images': args.val_images,
                'timestamp': timestamp
            },
            'results': all_results
        }, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for result in all_results:
        print(f"{result['model_name']}: TCI={result['avg_tci']:.4f}, Val Loss={result['best_val_loss']:.4f}")
    
    print(f"\n结果已保存至: {results_file}")

if __name__ == '__main__':
    main()
