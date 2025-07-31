#!/usr/bin/env python3
"""
YOLOP系列模型横向对比实验 - 修复版
修复了数据集加载问题
"""
import os
import sys
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from easydict import EasyDict as edict
import wandb

# 添加项目路径
sys.path.insert(0, '/workspace/YOLOPX/v2')

from data.bdd_dataset import BddDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from models.builder import get_net_from_yaml
from utils.utils import get_optimizer
from core.conflict_methods import ConflictGradientAnalyzer

# 实验配置
EXPERIMENT_CONFIG = {
    'epochs': 20,
    'train_images': 200,
    'batch_size': 4,
    'lr': 0.001,
    'seed': 42,
    'wandb_project': 'yolopx-base-comparison'
}

# 模型配置
MODEL_CONFIGS = {
    'yolopx': {
        'config_file': 'yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    },
    'yolop_v1': {
        'config_file': 'yolop_v1_official.yaml',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)'
    },
    'yolop_v3': {
        'config_file': 'yolop_v3_official.yaml',
        'description': 'YOLOP v3 (anchor-based, ELAN-W)'
    }
}

def load_config(path):
    """加载yaml配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_model(model_name, debug=False):
    """训练单个模型"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODEL_CONFIGS[model_name]['description']}")
    print(f"时间戳: {timestamp}")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    setup_seed(EXPERIMENT_CONFIG['seed'])
    
    # 加载配置
    model_config = load_config(f'/workspace/YOLOPX/v2/cfgs/models/{MODEL_CONFIGS[model_name]["config_file"]}')
    data_config = load_config('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml')
    train_config = load_config('/workspace/YOLOPX/v2/cfgs/train_v2.yaml')
    
    # 合并配置
    cfg = edict()
    cfg.update(train_config)
    cfg.update({'DATASET': data_config['DATASET']})
    cfg.update({'MODEL': model_config})
    
    # 更新训练参数
    cfg.TRAIN.END_EPOCH = EXPERIMENT_CONFIG['epochs'] if not debug else 2
    cfg.TRAIN.BATCH_SIZE_PER_GPU = EXPERIMENT_CONFIG['batch_size']
    cfg.TRAIN.LR0 = EXPERIMENT_CONFIG['lr']
    cfg.DATASET.NUMBER_IMAGE = EXPERIMENT_CONFIG['train_images'] if not debug else 50
    
    # 创建输出目录
    output_dir = f'/workspace/YOLOPX/experiments/base_models_comparison/results/{run_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化wandb
    wandb.init(
        project=EXPERIMENT_CONFIG['wandb_project'],
        name=run_name,
        config={
            'model': model_name,
            'epochs': cfg.TRAIN.END_EPOCH,
            'batch_size': cfg.TRAIN.BATCH_SIZE_PER_GPU,
            'lr': cfg.TRAIN.LR0,
            'train_images': cfg.DATASET.NUMBER_IMAGE
        }
    )
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = BddDataset(cfg=cfg, is_train=True, transform=None)
    val_dataset = BddDataset(cfg=cfg, is_train=False, transform=None)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("错误：训练集为空！")
        return None
    
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
    print("创建模型...")
    model = get_net_from_yaml(f'/workspace/YOLOPX/v2/cfgs/models/{MODEL_CONFIGS[model_name]["config_file"]}').to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 创建损失函数和优化器
    criterion = get_loss(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type != 'cpu'))
    
    # 创建冲突分析器
    conflict_analyzer = ConflictGradientAnalyzer(num_tasks=3)
    
    # 训练循环
    best_val_loss = float('inf')
    tci_values = []
    
    for epoch in range(cfg.TRAIN.END_EPOCH):
        print(f"\nEpoch [{epoch+1}/{cfg.TRAIN.END_EPOCH}]")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        epoch_tci_values = []
        
        for i, (images, targets, _, _) in enumerate(train_loader):
            if debug and i >= 10:  # Debug模式只训练10个batch
                break
                
            images = images.to(device)
            targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=(device.type != 'cpu')):
                outputs = model(images)
                total_loss, task_losses = criterion(outputs, targets, model=model)
            
            # 计算TCI
            if i % 10 == 0:
                tci = conflict_analyzer.compute_task_conflict_intensity(model, task_losses)
                epoch_tci_values.append(tci)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            
            if i % 10 == 0:
                print(f"  Batch [{i}/{len(train_loader)}], Loss: {total_loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_tci = np.mean(epoch_tci_values) if epoch_tci_values else 0
        tci_values.append(avg_tci)
        
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  任务冲突强度(TCI): {avg_tci:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (images, targets, _, _) in enumerate(val_loader):
                if debug and i >= 5:  # Debug模式只验证5个batch
                    break
                    
                images = images.to(device)
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
                
                outputs = model(images)
                total_loss, _ = criterion(outputs, targets, model=model)
                val_loss += total_loss.item()
        
        avg_val_loss = val_loss / min(len(val_loader), 5 if debug else len(val_loader))
        print(f"  验证损失: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # 记录到wandb
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'tci': avg_tci,
            'epoch': epoch
        })
    
    # 计算平均TCI
    avg_tci_overall = np.mean(tci_values) if tci_values else 0
    
    # 保存最终结果
    results = {
        'model': model_name,
        'description': MODEL_CONFIGS[model_name]['description'],
        'timestamp': timestamp,
        'best_val_loss': best_val_loss,
        'avg_tci': avg_tci_overall,
        'tci_values': tci_values
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    wandb.finish()
    
    print(f"\n✅ 训练完成!")
    print(f"📊 平均TCI: {avg_tci_overall:.4f}")
    print(f"📉 最佳验证损失: {best_val_loss:.4f}")
    
    return results

def generate_comparison_report(all_results):
    """生成对比报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f'/workspace/YOLOPX/experiments/base_models_comparison/results/report_{timestamp}'
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成JSON报告
    json_path = os.path.join(report_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成Markdown报告
    md_path = os.path.join(report_dir, 'comparison_report.md')
    with open(md_path, 'w') as f:
        f.write("# YOLOP系列模型横向对比实验报告\n\n")
        f.write(f"**实验日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验配置\n\n")
        f.write(f"- 训练图片数: {EXPERIMENT_CONFIG['train_images']}\n")
        f.write(f"- 训练轮数: {EXPERIMENT_CONFIG['epochs']}\n")
        f.write(f"- 批次大小: {EXPERIMENT_CONFIG['batch_size']}\n")
        f.write(f"- 学习率: {EXPERIMENT_CONFIG['lr']}\n\n")
        
        f.write("## 模型对比结果\n\n")
        f.write("| 模型 | 描述 | 平均TCI ↓ | 最佳验证损失 ↓ |\n")
        f.write("|------|------|-----------|----------------|\n")
        
        # 按TCI排序
        sorted_results = sorted(all_results, key=lambda x: x['avg_tci'])
        
        for result in sorted_results:
            f.write(f"| {result['model']} | {result['description']} | "
                   f"{result['avg_tci']:.4f} | {result['best_val_loss']:.4f} |\n")
        
        f.write("\n## 关键发现\n\n")
        
        # 找出最佳和最差模型
        best_model = sorted_results[0]
        worst_model = sorted_results[-1]
        
        f.write(f"- **最低任务冲突**: {best_model['model']} (TCI={best_model['avg_tci']:.4f})\n")
        f.write(f"- **最高任务冲突**: {worst_model['model']} (TCI={worst_model['avg_tci']:.4f})\n")
        
        # 计算anchor-based vs anchor-free
        anchor_free = [r for r in all_results if 'yolopx' in r['model'].lower()]
        anchor_based = [r for r in all_results if 'yolop' in r['model'].lower() and 'x' not in r['model'].lower()]
        
        if anchor_free and anchor_based:
            avg_tci_free = np.mean([r['avg_tci'] for r in anchor_free])
            avg_tci_based = np.mean([r['avg_tci'] for r in anchor_based])
            
            f.write(f"\n### Anchor-based vs Anchor-free对比\n")
            f.write(f"- Anchor-based平均TCI: {avg_tci_based:.4f}\n")
            f.write(f"- Anchor-free平均TCI: {avg_tci_free:.4f}\n")
            
            diff = (avg_tci_free - avg_tci_based) / avg_tci_based * 100
            if diff > 0:
                f.write(f"- **结论**: Anchor-free的任务冲突比Anchor-based高{diff:.1f}%\n")
            else:
                f.write(f"- **结论**: Anchor-free的任务冲突比Anchor-based低{-diff:.1f}%\n")
    
    print(f"\n📊 对比报告已生成:")
    print(f"   - JSON: {json_path}")
    print(f"   - Markdown: {md_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()),
                        default=list(MODEL_CONFIGS.keys()),
                        help='要训练的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 YOLOP系列模型横向对比实验")
    print(f"📊 配置: epochs={EXPERIMENT_CONFIG['epochs']}, train_images={EXPERIMENT_CONFIG['train_images']}")
    print(f"🤖 模型: {args.models}")
    print("="*60)
    
    all_results = []
    
    for model_name in args.models:
        try:
            result = train_model(model_name, debug=args.debug)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
    
    if all_results:
        generate_comparison_report(all_results)
    
    print("\n" + "="*60)
    print("✅ 实验完成！")
    print("="*60)

if __name__ == '__main__':
    main()