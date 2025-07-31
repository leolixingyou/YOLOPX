#!/usr/bin/env python3
"""
最终的YOLOP系列模型任务冲突对比
使用真实数据和损失函数
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path

sys.path.append('/workspace/YOLOPX/v2')

from models.builder import get_net_from_yaml
from data.unified_dataset import BddDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from easydict import EasyDict as edict

# 模型配置
MODELS = {
    'yolopx': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    },
    'yolop': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)' 
    }
}

def compute_tci(task_losses):
    """计算任务冲突强度"""
    if len(task_losses) < 2:
        return 0.0
    losses = np.array(list(task_losses.values()))
    return float(np.std(losses) / (np.mean(losses) + 1e-8))

def analyze_model(model_name, num_batches=10):
    """分析单个模型的任务冲突"""
    print(f"\n分析模型: {MODELS[model_name]['description']}")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    data_config = yaml.safe_load(open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml'))
    train_config = yaml.safe_load(open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml'))
    
    # 合并配置
    cfg = edict()
    cfg.update(train_config)
    cfg.DATASET = edict(data_config['DATASET'])
    cfg.DATASET.NUMBER_IMAGE = 50  # 使用少量图片快速测试
    cfg.DATASET.NUMBER_VAL = 10
    cfg.TRAIN.BATCH_SIZE_PER_GPU = 2
    cfg.TEST.BATCH_SIZE_PER_GPU = 2
    
    # 加载模型
    try:
        model = get_net_from_yaml(MODELS[model_name]['config'])
        model = model.to(device)
        model.train()  # 设置为训练模式
        print(f"✅ 模型加载成功: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None
    
    # 创建数据集
    try:
        dataset = BddDataset(cfg=cfg, is_train=True)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=True,
            num_workers=0,
            collate_fn=BddDataset.collate_fn
        )
        print(f"✅ 数据集创建成功: {len(dataset)} 样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {str(e)}")
        return None
    
    # 创建损失函数
    try:
        # 根据模型类型设置配置
        cfg.model_family = 'yolopx' if 'yolopx' in model_name else 'yolop'
        
        # 添加必要的配置
        cfg.MODEL = edict({
            'NC': 10,  # BDD100K 检测类别数
            'LOSS': edict({
                'LOSS_WEIGHTS': {'det': 1.0, 'da': 1.0, 'll': 1.0}
            })
        })
        
        cfg.LOSS = edict({
            'SEG_POS_WEIGHT': 2.0,
            'FL_GAMMA': 0.0,
            'MULTI_HEAD_LAMBDA': [1.0, 0.2, 0.2, 0.2]
        })
            
        criterion = get_loss(cfg, device, model)
        print(f"✅ 损失函数创建成功")
    except Exception as e:
        print(f"❌ 损失函数创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # 收集TCI数据
    tci_values = []
    batch_count = 0
    
    print("\n开始分析任务冲突...")
    
    for i, (images, labels, shapes, img_file) in enumerate(dataloader):
        if batch_count >= num_batches:
            break
            
        try:
            # 移动数据到设备
            images = images.to(device)
            if isinstance(labels, list):
                labels = [l.to(device) if isinstance(l, torch.Tensor) else l for l in labels]
            else:
                labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            total_loss_val, losses_dict = criterion(outputs, labels, shapes, model, images)
            
            # 提取各个任务的损失
            task_losses = {}
            if isinstance(losses_dict, dict):
                for k, v in losses_dict.items():
                    if 'loss' in k.lower() and hasattr(v, 'item'):
                        # 清理键名
                        task_name = k.replace('_loss', '').replace('loss_', '')
                        task_losses[task_name] = v.item()
            
            # 计算TCI
            if len(task_losses) >= 2:
                tci = compute_tci(task_losses)
                tci_values.append(tci)
                
                if batch_count == 0:
                    print(f"\n  批次 {i} 损失值:")
                    for k, v in task_losses.items():
                        print(f"    {k}: {v:.4f}")
                    print(f"  TCI: {tci:.4f}")
            
            batch_count += 1
            
        except Exception as e:
            print(f"  ⚠️ 批次 {i} 处理失败: {str(e)}")
            continue
    
    if not tci_values:
        print("❌ 无法计算TCI")
        return None
    
    # 计算统计
    results = {
        'model': model_name,
        'description': MODELS[model_name]['description'],
        'avg_tci': np.mean(tci_values),
        'std_tci': np.std(tci_values),
        'min_tci': np.min(tci_values),
        'max_tci': np.max(tci_values),
        'num_batches': len(tci_values)
    }
    
    print(f"\n  统计结果:")
    print(f"  平均TCI: {results['avg_tci']:.4f} (±{results['std_tci']:.4f})")
    print(f"  范围: [{results['min_tci']:.4f}, {results['max_tci']:.4f}]")
    print(f"  分析批次数: {results['num_batches']}")
    
    return results

def main():
    """主函数"""
    print("="*60)
    print("🔍 YOLOP系列模型任务冲突最终对比")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_results = []
    
    # 分析每个模型
    for model_name in MODELS.keys():
        try:
            result = analyze_model(model_name)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ {model_name} 分析异常: {str(e)}")
    
    # 生成报告
    if all_results:
        print("\n" + "="*60)
        print("📊 最终对比报告")
        print("="*60)
        
        # 按TCI排序
        all_results.sort(key=lambda x: x['avg_tci'])
        
        print(f"\n{'模型':<15} {'描述':<35} {'平均TCI':<10} {'标准差':<10}")
        print("-"*70)
        
        for r in all_results:
            print(f"{r['model']:<15} {r['description']:<35} "
                  f"{r['avg_tci']:<10.4f} {r['std_tci']:<10.4f}")
        
        # 分析anchor-based vs anchor-free
        if len(all_results) >= 2:
            yolopx_result = next((r for r in all_results if r['model'] == 'yolopx'), None)
            yolop_result = next((r for r in all_results if r['model'] == 'yolop'), None)
            
            if yolopx_result and yolop_result:
                diff = (yolopx_result['avg_tci'] - yolop_result['avg_tci']) / yolop_result['avg_tci'] * 100
                
                print(f"\n📈 关键发现:")
                print(f"  - YOLOP v2 (Anchor-based) 平均TCI: {yolop_result['avg_tci']:.4f}")
                print(f"  - YOLOPX (Anchor-free) 平均TCI: {yolopx_result['avg_tci']:.4f}")
                
                if diff > 0:
                    print(f"  - 结论: Anchor-free的任务冲突比Anchor-based高 {diff:.1f}%")
                else:
                    print(f"  - 结论: Anchor-free的任务冲突比Anchor-based低 {-diff:.1f}%")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'/workspace/YOLOPX/experiments/base_models_comparison/results/final_comparison_{timestamp}.json'
        
        Path(output_file).parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 结果已保存: {output_file}")

if __name__ == '__main__':
    main()