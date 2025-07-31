#!/usr/bin/env python3
"""
简化的YOLOP任务冲突测试
========================
先测试基本框架，确保能跑通
"""

import os
import sys
import time
import json
from pathlib import Path
import yaml
from easydict import EasyDict as edict
from datetime import datetime
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
from torch.utils.data import DataLoader, TensorDataset
from core.loss import get_loss
from models.builder import get_net_from_yaml

class MockDataset(torch.utils.data.Dataset):
    """模拟数据集用于测试"""
    
    def __init__(self, size=100, img_size=(384, 640)):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟图像数据
        image = torch.randn(3, self.img_size[0], self.img_size[1])
        
        # 模拟检测标签 [class, x, y, w, h]
        det_target = torch.tensor([[0, 0.5, 0.5, 0.3, 0.4]], dtype=torch.float32)
        
        # 模拟分割标签
        da_seg_target = torch.randint(0, 2, (self.img_size[0], self.img_size[1]), dtype=torch.float32)
        ll_seg_target = torch.randint(0, 2, (self.img_size[0], self.img_size[1]), dtype=torch.float32)
        
        return image, [det_target, da_seg_target, ll_seg_target], idx, "mock_path"
    
    @staticmethod
    def collate_fn(batch):
        images, targets, indices, paths = zip(*batch)
        images = torch.stack(images, 0)
        return images, list(targets), indices, paths

class ConflictMetricsTracker:
    """任务冲突指标追踪器"""
    
    def __init__(self):
        self.tci_values = []
        
    def calculate_tci(self, model, task_losses):
        """计算任务冲突指数"""
        if len(task_losses) < 2:
            return 0.0
            
        gradients = {}
        
        # 计算每个任务的梯度
        for task_name, loss in task_losses.items():
            if task_name == 'total' or loss == 0:
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
            grad_list = list(gradients.values())
            conflicts = []
            
            for i in range(len(grad_list)):
                for j in range(i+1, len(grad_list)):
                    cos_sim = torch.cosine_similarity(grad_list[i], grad_list[j], dim=0)
                    conflict = max(0, -cos_sim.item())
                    conflicts.append(conflict)
            
            tci = np.mean(conflicts) if conflicts else 0.0
            self.tci_values.append(tci)
            return tci
        
        return 0.0
    
    def get_metrics(self):
        """获取当前指标"""
        return {
            'avg_tci': np.mean(self.tci_values) if self.tci_values else 0.0,
            'latest_tci': self.tci_values[-1] if self.tci_values else 0.0,
        }

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return edict(yaml.safe_load(f))

def run_single_experiment(model_name, epochs=3):
    """运行单个实验"""
    
    # 创建运行名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"开始实验: {model_name}")
    print(f"运行名称: {run_name}")
    print(f"{'='*60}")
    
    # 初始化 wandb
    wandb.init(
        project="yolop-conflict-analysis",
        name=run_name,
        config={
            "model": model_name,
            "epochs": epochs,
            "experiment_type": "task_conflict_analysis"
        },
        tags=["conflict_analysis", "yolop_family", model_name],
        reinit=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模拟数据集
    train_dataset = MockDataset(size=50)
    val_dataset = MockDataset(size=20)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=MockDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=MockDataset.collate_fn
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 加载模型配置
    model_configs = {
        'yolopx_v2_anchor_free': 'cfgs/models/yolopx_v2_anchor_free.yaml',
        'yolop_v1_official': 'cfgs/models/yolop_v1_official.yaml',
        'yolop_v3_official': 'cfgs/models/yolop_v3_official.yaml'
    }
    
    try:
        model_cfg_path = model_configs[model_name]
        model = get_net_from_yaml(model_cfg_path).to(device)
        
        # 简化的损失函数和优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        wandb.finish()
        return None
    
    # 冲突指标追踪器
    conflict_tracker = ConflictMetricsTracker()
    
    # 简化的训练循环
    results = {'epochs': [], 'train_metrics': [], 'val_metrics': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_tci = 0.0
        
        for i, (input_data, target, _, _) in enumerate(train_loader):
            if i >= 5:  # 限制batch数量
                break
                
            input_data = input_data.to(device)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(input_data)
                
                # 简化的损失计算
                total_loss = torch.randn(1, requires_grad=True).to(device) * 0.1 + 1.0
                
                # 模拟任务损失
                task_losses = {
                    'det_loss': torch.randn(1, requires_grad=True).to(device) * 0.1 + 0.5,
                    'da_seg_loss': torch.randn(1, requires_grad=True).to(device) * 0.1 + 0.3,
                    'll_seg_loss': torch.randn(1, requires_grad=True).to(device) * 0.1 + 0.2
                }
                
                # 计算TCI
                tci = conflict_tracker.calculate_tci(model, task_losses)
                train_tci += tci
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
            except Exception as e:
                print(f"训练批次失败: {e}")
                continue
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i, (input_data, target, _, _) in enumerate(val_loader):
                if i >= 3:  # 限制验证batch数量
                    break
                    
                input_data = input_data.to(device)
                
                try:
                    outputs = model(input_data)
                    total_loss = torch.randn(1).to(device) * 0.1 + 0.8
                    val_loss += total_loss.item()
                    val_batches += 1
                except Exception as e:
                    print(f"验证批次失败: {e}")
                    continue
        
        # 计算平均指标
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        avg_train_tci = train_tci / max(train_batches, 1)
        
        # 记录结果
        train_metrics = {'total_loss': avg_train_loss, 'tci': avg_train_tci}
        val_metrics = {'total_loss': avg_val_loss, 'tci': 0.0}
        
        results['epochs'].append(epoch + 1)
        results['train_metrics'].append(train_metrics)
        results['val_metrics'].append(val_metrics)
        
        # 记录到 wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/total_loss": avg_train_loss,
            "train/tci": avg_train_tci,
            "val/total_loss": avg_val_loss,
        })
        
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, TCI: {avg_train_tci:.4f}")
    
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

def main():
    """主函数"""
    models = ['yolopx_v2_anchor_free', 'yolop_v1_official', 'yolop_v3_official']
    all_results = []
    
    for model_name in models:
        try:
            result = run_single_experiment(model_name, epochs=3)
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"实验 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"runs/conflict_comparison_report_{timestamp}.json"
        
        # 排序结果
        results_sorted = sorted(all_results, key=lambda x: x['avg_tci'])
        
        report = {
            'experiment_timestamp': timestamp,
            'experiment_type': 'yolop_family_conflict_analysis',
            'summary': {
                'total_models': len(all_results),
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
    
    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"总共完成 {len(all_results)} 个实验")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()