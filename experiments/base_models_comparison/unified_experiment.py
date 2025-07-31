#!/usr/bin/env python3
"""
统一的YOLOP系列模型任务冲突对比实验
一个文件完成所有模型的测试
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path
import argparse

# 添加v2到路径
sys.path.insert(0, '/workspace/YOLOPX/v2')

from models.builder import get_net_from_yaml
from torch.utils.data import DataLoader, Dataset
from core.loss import get_loss
from easydict import EasyDict as edict
from utils.utils import get_optimizer


# ============ 模拟数据集 ============
class DummyDataset(Dataset):
    """模拟数据集用于测试"""
    def __init__(self, size=200, img_size=(384, 640)):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成随机图像
        img = torch.randn(3, *self.img_size)
        
        # 生成模拟标签
        # 检测标签：[类别, x, y, w, h]
        num_objs = np.random.randint(1, 10)
        det_labels = torch.rand(num_objs, 5)
        det_labels[:, 0] = torch.randint(0, 10, (num_objs,))  # 类别
        
        # 分割标签
        da_seg = torch.randint(0, 2, (1, *self.img_size))  # 二值分割
        ll_seg = torch.randint(0, 2, (1, *self.img_size))  # 二值分割
        
        labels = [det_labels, da_seg, ll_seg]
        shapes = torch.tensor(self.img_size)
        
        return img, labels, shapes, f'dummy_{idx}.jpg'
    
    @staticmethod
    def collate_fn(batch):
        """自定义collate函数"""
        imgs, labels, shapes, paths = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs, 0)
        
        # 处理检测标签
        det_labels = []
        for i, label in enumerate(labels):
            det_label = label[0]
            # 添加batch索引
            batch_idx = torch.full((det_label.shape[0], 1), i)
            det_label = torch.cat([batch_idx, det_label], 1)
            det_labels.append(det_label)
        det_labels = torch.cat(det_labels, 0)
        
        # Stack分割标签
        da_segs = torch.stack([label[1] for label in labels], 0)
        ll_segs = torch.stack([label[2] for label in labels], 0)
        
        # 组合标签
        labels = [det_labels, da_segs, ll_segs]
        shapes = torch.stack(shapes, 0)
        
        return imgs, labels, shapes, paths

# ============ 实验配置 ============
EXPERIMENT_CONFIG = {
    'train_images': 200,
    'val_images': 100, 
    'epochs': 5,  # 减少epoch数以快速验证
    'batch_size': 4,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # 避免多进程问题
    'num_batches_per_epoch': 10  # 每个epoch只跑10个batch快速测试
}

# 模型配置
MODELS_CONFIG = {
    'yolopx': {
        'config_path': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)',
        'model_family': 'yolopx'
    },
    'yolop_v2': {
        'config_path': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)',
        'model_family': 'yolop'
    }
    # v1和v3暂时跳过，因为模块映射问题
}

# ============ 工具函数 ============
def compute_tci(losses_dict):
    """计算任务冲突强度"""
    losses = []
    for k, v in losses_dict.items():
        if 'loss' in k.lower() and hasattr(v, 'item'):
            losses.append(v.item())
    
    if len(losses) < 2:
        return 0.0
    
    losses = np.array(losses)
    return float(np.std(losses) / (np.mean(losses) + 1e-8))

def create_config():
    """创建统一的配置"""
    # 加载基础配置
    with open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    with open('/workspace/YOLOPX/v2/cfgs/data/bdd100k.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # 合并配置
    cfg = edict(train_cfg)
    cfg.DATASET = edict(data_cfg['DATASET'])
    
    # 覆盖实验参数
    cfg.DATASET.DATAROOT = '/workspace/YOLOPX/bdd100k'
    cfg.DATASET.NUMBER_IMAGE = EXPERIMENT_CONFIG['train_images']
    cfg.DATASET.NUMBER_VAL = EXPERIMENT_CONFIG['val_images']
    cfg.TRAIN.BATCH_SIZE_PER_GPU = EXPERIMENT_CONFIG['batch_size']
    cfg.TRAIN.END_EPOCH = EXPERIMENT_CONFIG['epochs']
    cfg.TRAIN.LR0 = EXPERIMENT_CONFIG['lr']
    cfg.TEST.BATCH_SIZE_PER_GPU = EXPERIMENT_CONFIG['batch_size']
    cfg.WORKERS = EXPERIMENT_CONFIG['num_workers']
    
    # 添加损失函数需要的配置
    cfg.MODEL = edict({
        'NC': 10,  # BDD100K类别数
        'LOSS': edict({
            'LOSS_WEIGHTS': {'det': 1.0, 'da': 1.0, 'll': 1.0}
        })
    })
    
    cfg.LOSS = edict({
        'SEG_POS_WEIGHT': 2.0,
        'FL_GAMMA': 0.0,
        'MULTI_HEAD_LAMBDA': [1.0, 0.2, 0.2, 0.2]
    })
    
    return cfg

# ============ 模型包装器 ============
class ModelWrapper(nn.Module):
    """统一不同模型的输出格式"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)
        
        # 统一输出格式
        if isinstance(outputs, tuple) and len(outputs) == 3:
            det, da, ll = outputs
            return {'det': det, 'da_seg': da, 'll_seg': ll}
        elif isinstance(outputs, dict):
            return outputs
        else:
            raise ValueError(f"Unknown output format: {type(outputs)}")

# ============ 实验函数 ============
def run_model_experiment(model_name, cfg):
    """运行单个模型的实验"""
    print(f"\n{'='*60}")
    print(f"测试模型: {MODELS_CONFIG[model_name]['description']}")
    print(f"{'='*60}")
    
    device = torch.device(cfg.get('device', EXPERIMENT_CONFIG['device']))
    
    # 1. 加载模型
    try:
        cfg.model_family = MODELS_CONFIG[model_name]['model_family']
        raw_model = get_net_from_yaml(MODELS_CONFIG[model_name]['config_path'])
        model = ModelWrapper(raw_model).to(device)
        model.train()
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型加载成功: {param_count:.2f}M 参数")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None
    
    # 2. 创建数据集
    try:
        # 使用模拟数据集
        train_dataset = DummyDataset(size=cfg.DATASET.NUMBER_IMAGE)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=True,
            num_workers=cfg.WORKERS,
            collate_fn=DummyDataset.collate_fn
        )
        print(f"✅ 数据集创建成功: {len(train_dataset)} 训练样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {str(e)}")
        return None
    
    # 3. 创建损失函数和优化器
    try:
        criterion = get_loss(cfg, device, raw_model)
        optimizer = get_optimizer(cfg, model)
        print(f"✅ 损失函数和优化器创建成功")
    except Exception as e:
        print(f"❌ 损失函数创建失败: {str(e)}")
        return None
    
    # 4. 训练并收集TCI
    tci_history = []
    loss_history = []
    
    print("\n开始训练...")
    for epoch in range(EXPERIMENT_CONFIG['epochs']):
        epoch_tci = []
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (images, labels, shapes, _) in enumerate(train_loader):
            if batch_count >= EXPERIMENT_CONFIG['num_batches_per_epoch']:
                break
                
            try:
                # 数据准备
                images = images.to(device)
                if isinstance(labels, list):
                    labels = [l.to(device) if isinstance(l, torch.Tensor) else l for l in labels]
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                total_loss, losses_dict = criterion(outputs, labels, shapes, raw_model, images)
                
                # 计算TCI
                tci = compute_tci(losses_dict)
                if tci > 0:
                    epoch_tci.append(tci)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                # 首个epoch的首个batch打印详细信息
                if epoch == 0 and i == 0:
                    print(f"\n  样例损失分解:")
                    for k, v in losses_dict.items():
                        if hasattr(v, 'item'):
                            print(f"    {k}: {v.item():.4f}")
                    print(f"  TCI: {tci:.4f}")
                    
            except Exception as e:
                print(f"  ⚠️ Batch {i} 失败: {str(e)}")
                continue
        
        # 记录epoch统计
        if epoch_tci:
            avg_tci = np.mean(epoch_tci)
            tci_history.extend(epoch_tci)
            loss_history.append(epoch_loss / max(1, batch_count))
            print(f"Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']}: "
                  f"平均损失={epoch_loss/max(1,batch_count):.4f}, "
                  f"平均TCI={avg_tci:.4f}")
    
    # 5. 计算最终统计
    if not tci_history:
        print("❌ 无有效TCI数据")
        return None
        
    results = {
        'model': model_name,
        'description': MODELS_CONFIG[model_name]['description'],
        'avg_tci': np.mean(tci_history),
        'std_tci': np.std(tci_history),
        'min_tci': np.min(tci_history),
        'max_tci': np.max(tci_history),
        'final_loss': loss_history[-1] if loss_history else 0,
        'num_samples': len(tci_history)
    }
    
    print(f"\n📊 实验结果:")
    print(f"  平均TCI: {results['avg_tci']:.4f} (±{results['std_tci']:.4f})")
    print(f"  TCI范围: [{results['min_tci']:.4f}, {results['max_tci']:.4f}]")
    
    return results

# ============ 主函数 ============
def main():
    """运行完整实验"""
    print("🚀 YOLOP系列模型任务冲突对比实验")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ 配置: {EXPERIMENT_CONFIG}")
    
    # 创建结果目录
    results_dir = Path('/workspace/YOLOPX/experiments/base_models_comparison/results')
    results_dir.mkdir(exist_ok=True)
    
    # 创建统一配置
    cfg = create_config()
    
    # 运行所有模型实验
    all_results = []
    for model_name in MODELS_CONFIG.keys():
        try:
            result = run_model_experiment(model_name, cfg)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成最终报告
    if all_results:
        print(f"\n{'='*60}")
        print("📊 最终对比结果")
        print(f"{'='*60}")
        
        # 打印表格
        print(f"\n{'模型':<15} {'描述':<35} {'平均TCI':<10} {'标准差':<10}")
        print("-"*70)
        
        for r in sorted(all_results, key=lambda x: x['avg_tci']):
            print(f"{r['model']:<15} {r['description']:<35} "
                  f"{r['avg_tci']:<10.4f} {r['std_tci']:<10.4f}")
        
        # 分析anchor-based vs anchor-free
        yolopx = next((r for r in all_results if r['model'] == 'yolopx'), None)
        yolop = next((r for r in all_results if r['model'] == 'yolop_v2'), None)
        
        if yolopx and yolop:
            diff = (yolopx['avg_tci'] - yolop['avg_tci']) / yolop['avg_tci'] * 100
            print(f"\n📈 关键发现:")
            print(f"  YOLOP v2 (Anchor-based) TCI: {yolop['avg_tci']:.4f}")
            print(f"  YOLOPX (Anchor-free) TCI: {yolopx['avg_tci']:.4f}")
            print(f"  差异: {diff:+.1f}%")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = results_dir / f'unified_experiment_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump({
                'config': EXPERIMENT_CONFIG,
                'timestamp': timestamp,
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 结果已保存: {output_file}")
        
        return True
    else:
        print("\n❌ 所有实验均失败")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)