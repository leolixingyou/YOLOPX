#!/usr/bin/env python3
"""
测试统一数据集模块
"""
import sys
import yaml
sys.path.append('/workspace/YOLOPX/v2')

from easydict import EasyDict
from data.unified_dataset import BddDataset

# 加载配置
with open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)

cfg = EasyDict()
cfg.DATASET = EasyDict(data_cfg['DATASET'])

print("测试统一数据集加载...")
print(f"配置: NUMBER_IMAGE={cfg.DATASET.NUMBER_IMAGE}, NUMBER_VAL={cfg.DATASET.get('NUMBER_VAL', 'not set')}")

# 测试训练集
train_dataset = BddDataset(cfg=cfg, is_train=True)
print(f"✅ 训练集加载成功: {len(train_dataset)} 样本")

# 测试验证集
val_dataset = BddDataset(cfg=cfg, is_train=False)
print(f"✅ 验证集加载成功: {len(val_dataset)} 样本")

# 测试数据加载
if len(train_dataset) > 0:
    sample = train_dataset[0]
    img, target, path, shapes = sample
    print(f"\n样本测试:")
    print(f"  图像shape: {img.shape}")
    print(f"  检测标签shape: {target[0].shape if len(target[0]) > 0 else 'empty'}")
    print(f"  驾驶区域shape: {target[1].shape}")
    print(f"  车道线shape: {target[2].shape}")
    print("\n✅ 数据集完全正常工作！")