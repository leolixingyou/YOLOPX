#!/usr/bin/env python3
"""
简化的YOLOP系列模型横向对比实验
专注于任务冲突检测的核心功能
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from datetime import datetime
import wandb

# 添加项目路径
sys.path.insert(0, '/workspace/YOLOPX/v2')

from easydict import EasyDict as edict
from torch.utils.data import DataLoader

# 导入核心模块
from data.unified_dataset import BddDataset
from models.builder import get_net_from_yaml
from core.loss import get_loss
from utils.utils import get_optimizer

# 实验配置
EXPERIMENT_CONFIG = {
    'epochs': 20,
    'train_images': 200,
    'val_images': 100,
    'batch_size': 4,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 模型配置
MODEL_CONFIGS = {
    'yolopx': {
        'config_file': 'yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)',
        'model_family': 'yolopx'
    },
    'yolop_v1': {
        'config_file': 'yolop_v1_official.yaml', 
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)',
        'model_family': 'yolop_v1_official'
    },
    'yolop_v3': {
        'config_file': 'yolop_v3_official.yaml',
        'description': 'YOLOP v3 (anchor-based, ELAN-W)',
        'model_family': 'yolop_v3_official'
    }
}

def compute_task_conflict_intensity(losses):
    """计算任务冲突强度 (TCI)"""
    if len(losses) < 2:
        return 0.0
    losses_array = np.array(losses)
    return float(np.std(losses_array) / (np.mean(losses_array) + 1e-8))

def train_model(model_name, device='cuda'):
    """训练单个模型并收集冲突指标"""
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODEL_CONFIGS[model_name]['description']}")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载配置
    data_config = yaml.safe_load(open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml'))
    train_config = yaml.safe_load(open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml'))
    
    # 合并配置
    cfg = edict()
    cfg.update(train_config)
    cfg.DATASET = edict(data_config['DATASET'])
    cfg.DATASET.NUMBER_IMAGE = EXPERIMENT_CONFIG['train_images']
    cfg.DATASET.NUMBER_VAL = EXPERIMENT_CONFIG['val_images']
    cfg.TRAIN.END_EPOCH = EXPERIMENT_CONFIG['epochs']
    cfg.TRAIN.BATCH_SIZE_PER_GPU = EXPERIMENT_CONFIG['batch_size']
    cfg.TRAIN.LR0 = EXPERIMENT_CONFIG['lr']
    cfg.model_family = MODEL_CONFIGS[model_name]['model_family']
    
    # 初始化wandb
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project='yolopx-base-comparison',
        name=run_name,
        config={
            'model': model_name,
            'epochs': EXPERIMENT_CONFIG['epochs'],
            'train_images': EXPERIMENT_CONFIG['train_images'],
            'batch_size': EXPERIMENT_CONFIG['batch_size']
        }
    )
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = BddDataset(cfg=cfg, is_train=True)
    val_dataset = BddDataset(cfg=cfg, is_train=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 数据加载器
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
    print(f"创建模型: {MODEL_CONFIGS[model_name]['config_file']}")
    model_path = f'/workspace/YOLOPX/v2/cfgs/models/{MODEL_CONFIGS[model_name]["config_file"]}'
    model = get_net_from_yaml(model_path).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device != 'cpu'))
    
    # 训练循环
    tci_history = []
    loss_history = {'train': [], 'val': []}
    
    for epoch in range(EXPERIMENT_CONFIG['epochs']):
        print(f"\nEpoch [{epoch+1}/{EXPERIMENT_CONFIG['epochs']}]")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        epoch_tci = []
        
        for i, (images, targets, _, _) in enumerate(train_loader):
            if i >= 50:  # 限制每个epoch的批次数
                break
                
            images = images.to(device)
            targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                outputs = model(images)
                total_loss, task_losses = criterion(outputs, targets, model=model)
            
            # 计算TCI
            if i % 10 == 0 and isinstance(task_losses, dict):
                losses_list = []
                for k, v in task_losses.items():
                    if 'loss' in k and hasattr(v, 'item'):
                        losses_list.append(v.item())
                
                if len(losses_list) >= 2:
                    tci = compute_task_conflict_intensity(losses_list)
                    epoch_tci.append(tci)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            
            if i % 10 == 0:
                print(f"  Batch [{i}/50], Loss: {total_loss.item():.4f}")
        
        avg_train_loss = train_loss / min(50, len(train_loader))
        avg_tci = np.mean(epoch_tci) if epoch_tci else 0.0
        tci_history.extend(epoch_tci)
        loss_history['train'].append(avg_train_loss)
        
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  平均TCI: {avg_tci:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (images, targets, _, _) in enumerate(val_loader):
                if i >= 20:  # 限制验证批次
                    break
                    
                images = images.to(device)
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
                
                outputs = model(images)
                total_loss, _ = criterion(outputs, targets, model=model)
                val_loss += total_loss.item()
        
        avg_val_loss = val_loss / min(20, len(val_loader))
        loss_history['val'].append(avg_val_loss)
        
        print(f"  验证损失: {avg_val_loss:.4f}")
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'tci': avg_tci
        })
    
    # 计算总体指标
    overall_tci = np.mean(tci_history) if tci_history else 0.0
    best_val_loss = min(loss_history['val']) if loss_history['val'] else float('inf')
    
    # 关闭wandb
    wandb.finish()
    
    print(f"\n训练完成!")
    print(f"总体平均TCI: {overall_tci:.4f}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    return {
        'model': model_name,
        'description': MODEL_CONFIGS[model_name]['description'],
        'avg_tci': overall_tci,
        'best_val_loss': best_val_loss,
        'tci_history': tci_history,
        'loss_history': loss_history
    }

def main():
    """运行所有模型的对比实验"""
    print("🚀 YOLOP系列模型横向对比实验")
    print(f"📊 配置: {EXPERIMENT_CONFIG}")
    
    results = []
    
    # 只运行能正常工作的模型
    models_to_test = ['yolopx']  # 先测试YOLOPX
    
    for model_name in models_to_test:
        try:
            result = train_model(model_name, device=EXPERIMENT_CONFIG['device'])
            results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'/workspace/YOLOPX/experiments/base_models_comparison/results/comparison_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump({
                'experiment_config': EXPERIMENT_CONFIG,
                'timestamp': timestamp,
                'results': results
            }, f, indent=2)
        
        print(f"\n📊 实验报告已保存: {report_path}")
        
        # 打印总结
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print(f"{'模型':<15} {'描述':<40} {'平均TCI':<10} {'最佳验证损失':<15}")
        print("-"*80)
        
        for r in results:
            print(f"{r['model']:<15} {r['description']:<40} {r['avg_tci']:<10.4f} {r['best_val_loss']:<15.4f}")

if __name__ == '__main__':
    main()