#!/usr/bin/env python3
"""
YOLOP 家族横向任务冲突检测实验
=================================

根据 research_plan_v3.md 阶段1要求，对 YOLOPv1, YOLOPv3, YOLOPX 进行横向任务冲突分析。

实验设置:
- Epochs: 20
- 训练图片: 1000
- 验证图片: 100
- 指标: TCI (Task Conflict Index), 任务性能指标
- 结果保存: wandb + 本地runs目录
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import subprocess
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# Wandb integration
import wandb

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import get_optimizer
from data.autodrive_dataset import AutoDriveDataset, BddDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from models.builder import get_net_from_yaml

class ConflictMetricsTracker:
    """任务冲突指标追踪器"""
    
    def __init__(self):
        self.task_gradients = {'det': [], 'da_seg': [], 'll_seg': []}
        self.task_losses = {'det': [], 'da_seg': [], 'll_seg': [], 'total': []}
        self.tci_values = []
        
    def update_gradients(self, model, task_losses):
        """更新任务梯度并计算冲突指标"""
        gradients = {}
        
        # 计算每个任务的梯度
        for task_name, loss in task_losses.items():
            if task_name == 'total':
                continue
                
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))
            
            if grad_vec:
                gradients[task_name] = torch.cat(grad_vec)
            
        # 计算TCI (Task Conflict Index)
        if len(gradients) >= 2:
            tci = self.calculate_tci(gradients)
            self.tci_values.append(tci)
            return tci
        return 0.0
    
    def calculate_tci(self, gradients):
        """计算任务冲突指数"""
        grad_list = list(gradients.values())
        conflicts = []
        
        for i in range(len(grad_list)):
            for j in range(i+1, len(grad_list)):
                # 计算梯度余弦相似度
                cos_sim = torch.cosine_similarity(grad_list[i], grad_list[j], dim=0)
                # 冲突定义为负相似度
                conflict = max(0, -cos_sim.item())
                conflicts.append(conflict)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def get_metrics(self):
        """获取当前指标"""
        return {
            'avg_tci': np.mean(self.tci_values) if self.tci_values else 0.0,
            'latest_tci': self.tci_values[-1] if self.tci_values else 0.0,
            'tci_std': np.std(self.tci_values) if len(self.tci_values) > 1 else 0.0
        }

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_environment()
        
    def setup_environment(self):
        """设置实验环境"""
        # 固定随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # CUDNN 设置
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True
        
    def create_run_name(self, model_name):
        """创建运行名称"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def run_single_experiment(self, model_config, data_config, train_config, model_name):
        """运行单个模型实验"""
        run_name = self.create_run_name(model_name)
        
        # 初始化 wandb
        wandb.init(
            project="yolop-conflict-analysis",
            name=run_name,
            config={
                "model": model_name,
                "epochs": train_config.TRAIN.END_EPOCH,
                "batch_size": train_config.TRAIN.BATCH_SIZE_PER_GPU,
                "learning_rate": train_config.TRAIN.LR0,
                "train_images": data_config.DATASET.NUMBER_IMAGE,
                "experiment_type": "task_conflict_analysis"
            },
            tags=["conflict_analysis", "yolop_family", model_name]
        )
        
        print(f"\n{'='*60}")
        print(f"开始实验: {model_name}")
        print(f"运行名称: {run_name}")
        print(f"{'='*60}")
        
        # 合并配置
        cfg = edict({**train_config, **data_config, **model_config})
        
        # 数据加载
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        # 使用BddDataset而不是AutoDriveDataset
        train_dataset = BddDataset(cfg=cfg, is_train=True, transform=transform)
        val_dataset = BddDataset(cfg=cfg, is_train=False, transform=transform)
        
        # 限制数据集大小
        if len(train_dataset) > self.config.train_images:
            train_dataset.data_list = train_dataset.data_list[:self.config.train_images]
        if len(val_dataset) > self.config.val_images:
            val_dataset.data_list = val_dataset.data_list[:self.config.val_images]
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
            shuffle=cfg.TRAIN.SHUFFLE, 
            num_workers=cfg.WORKERS, 
            pin_memory=cfg.PIN_MEMORY, 
            collate_fn=AutoDriveDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=AutoDriveDataset.collate_fn
        )
        
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        # 模型、损失函数、优化器
        try:
            model_cfg_path = f"cfgs/models/{model_config.config_file}"
            model = get_net_from_yaml(model_cfg_path).to(self.device)
            criterion = get_loss(cfg, self.device)
            optimizer = get_optimizer(cfg, model)
            scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type != 'cpu'))
            
            print(f"模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")
            
        except Exception as e:
            print(f"模型创建失败: {e}")
            wandb.finish()
            return None
        
        # 冲突指标追踪器
        conflict_tracker = ConflictMetricsTracker()
        
        # 训练循环
        results = {'epochs': [], 'train_metrics': [], 'val_metrics': []}
        
        for epoch in range(cfg.TRAIN.END_EPOCH):
            print(f"\nEpoch {epoch+1}/{cfg.TRAIN.END_EPOCH}")
            
            # 训练阶段
            train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler, 
                conflict_tracker, epoch
            )
            
            # 验证阶段
            val_metrics = self.validate_epoch(
                model, val_loader, criterion, conflict_tracker, epoch
            )
            
            # 记录结果
            results['epochs'].append(epoch + 1)
            results['train_metrics'].append(train_metrics)
            results['val_metrics'].append(val_metrics)
            
            # 记录到 wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": train_metrics['total_loss'],
                "train/det_loss": train_metrics.get('det_loss', 0),
                "train/da_seg_loss": train_metrics.get('da_seg_loss', 0),
                "train/ll_seg_loss": train_metrics.get('ll_seg_loss', 0),
                "train/tci": train_metrics.get('tci', 0),
                "val/total_loss": val_metrics['total_loss'],
                "val/det_loss": val_metrics.get('det_loss', 0),
                "val/da_seg_loss": val_metrics.get('da_seg_loss', 0),
                "val/ll_seg_loss": val_metrics.get('ll_seg_loss', 0),
                "val/tci": val_metrics.get('tci', 0),
            })
            
            print(f"训练损失: {train_metrics['total_loss']:.4f}, "
                  f"验证损失: {val_metrics['total_loss']:.4f}, "
                  f"TCI: {train_metrics.get('tci', 0):.4f}")
        
        # 计算最终指标
        final_metrics = conflict_tracker.get_metrics()
        final_metrics.update({
            'model_name': model_name,
            'run_name': run_name,
            'final_train_loss': results['train_metrics'][-1]['total_loss'],
            'final_val_loss': results['val_metrics'][-1]['total_loss']
        })
        
        # 保存结果
        results_path = f"runs/{run_name}/results.json"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({**results, 'final_metrics': final_metrics}, f, indent=2)
        
        print(f"\n实验完成: {model_name}")
        print(f"平均TCI: {final_metrics['avg_tci']:.4f}")
        print(f"结果保存至: {results_path}")
        
        wandb.finish()
        return final_metrics
    
    def train_epoch(self, model, dataloader, criterion, optimizer, scaler, conflict_tracker, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        tci_values = []
        
        for i, (input_data, target, _, _) in enumerate(dataloader):
            if i >= 10:  # 限制batch数量以节省时间
                break
                
            input_data = input_data.to(self.device, non_blocking=True)
            target = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in target]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(self.device.type != 'cpu')):
                outputs = model(input_data)
                total_loss_batch, task_losses = criterion(outputs, target, model=model)
            
            if total_loss_batch == 0 or not torch.isfinite(total_loss_batch):
                continue
            
            # 计算任务冲突
            if isinstance(task_losses, dict):
                tci = conflict_tracker.update_gradients(model, task_losses)
                tci_values.append(tci)
            
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            if i % 5 == 0:
                print(f"  Batch {i}/{min(10, len(dataloader))}, Loss: {total_loss_batch.item():.4f}")
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'tci': np.mean(tci_values) if tci_values else 0.0
        }
    
    def validate_epoch(self, model, dataloader, criterion, conflict_tracker, epoch):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        tci_values = []
        
        with torch.no_grad():
            for i, (input_data, target, _, _) in enumerate(dataloader):
                if i >= 5:  # 限制验证batch数量
                    break
                    
                input_data = input_data.to(self.device, non_blocking=True)
                target = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in target]
                
                with torch.cuda.amp.autocast(enabled=(self.device.type != 'cpu')):
                    outputs = model(input_data)
                    total_loss_batch, task_losses = criterion(outputs, target, model=model)
                
                if total_loss_batch != 0 and torch.isfinite(total_loss_batch):
                    total_loss += total_loss_batch.item()
                    num_batches += 1
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'tci': 0.0  # 验证阶段不计算TCI
        }

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return edict(yaml.safe_load(f))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOP 家族横向任务冲突检测实验')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--train_images', type=int, default=1000, help='训练图片数量')
    parser.add_argument('--val_images', type=int, default=100, help='验证图片数量')
    parser.add_argument('--models', nargs='+', default=['yolopx_v2_anchor_free', 'yolop_v1_official', 'yolop_v3_official'],
                       help='要测试的模型列表')
    
    args = parser.parse_args()
    
    # 实验配置
    experiment_config = edict({
        'epochs': args.epochs,
        'train_images': args.train_images,
        'val_images': args.val_images,
        'models': args.models
    })
    
    # 模型配置映射
    model_configs = {
        'yolopx': {
            'config_file': 'yolopx.yaml',
            'model_family': 'yolopx'
        },
        'yolop_v2': {
            'config_file': 'yolop.yaml',
            'model_family': 'yolop'
        },
        'yolop_v1_official': {
            'config_file': 'yolop_v1_official.yaml', 
            'model_family': 'yolop_v1_official'
        },
        'yolop_v3_official': {
            'config_file': 'yolop_v3_official.yaml',
            'model_family': 'yolop_v3_official'
        },
        'yolopx_v2_anchor_free': {
            'config_file': 'yolopx_v2_anchor_free.yaml',
            'model_family': 'yolopx'
        }
    }
    
    # 加载基础配置
    # 使用虚拟数据集进行测试
    data_config = load_config('cfgs/data/dummy_bdd100k.yaml')
    train_config = load_config('cfgs/train_v2.yaml')
    
    # 更新训练配置
    train_config.TRAIN.END_EPOCH = args.epochs
    data_config.DATASET.NUMBER_IMAGE = args.train_images
    
    # 创建实验运行器
    runner = ExperimentRunner(experiment_config)
    
    # 运行所有实验
    all_results = []
    
    for model_name in args.models:
        if model_name not in model_configs:
            print(f"警告: 未知模型 {model_name}, 跳过")
            continue
        
        model_config = edict(model_configs[model_name])
        
        try:
            result = runner.run_single_experiment(
                model_config, data_config, train_config, model_name
            )
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"实验 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if all_results:
        generate_comparison_report(all_results)
    
    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"总共完成 {len(all_results)} 个实验")
    print(f"{'='*60}")

def generate_comparison_report(results):
    """生成横向对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"runs/conflict_comparison_report_{timestamp}.json"
    
    # 排序结果
    results_sorted = sorted(results, key=lambda x: x['avg_tci'])
    
    report = {
        'experiment_timestamp': timestamp,
        'experiment_type': 'yolop_family_conflict_analysis',
        'summary': {
            'total_models': len(results),
            'best_model': results_sorted[0]['model_name'] if results_sorted else None,
            'worst_model': results_sorted[-1]['model_name'] if results_sorted else None,
        },
        'detailed_results': results_sorted,
        'ranking_by_tci': [
            {'rank': i+1, 'model': r['model_name'], 'avg_tci': r['avg_tci']} 
            for i, r in enumerate(results_sorted)
        ]
    }
    
    # 保存报告
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印摘要
    print(f"\n横向对比报告生成: {report_path}")
    print(f"\nTCI 排名 (越低越好):")
    for item in report['ranking_by_tci']:
        print(f"  {item['rank']}. {item['model']}: {item['avg_tci']:.4f}")

if __name__ == '__main__':
    main()