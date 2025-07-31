#!/usr/bin/env python3
"""
测试数据集加载
"""
import sys
import yaml
sys.path.append('/workspace/YOLOPX/v2')

from easydict import EasyDict
from data.bdd_dataset import BddDataset

# 加载数据配置
with open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)

cfg = EasyDict()
cfg.DATASET = EasyDict(data_cfg['DATASET'])

print("数据配置:")
print(f"DATAROOT: {cfg.DATASET.DATAROOT}")
print(f"NUMBER_IMAGE: {cfg.DATASET.NUMBER_IMAGE}")
print(f"TRAIN_SET: {cfg.DATASET.TRAIN_SET}")
print(f"TEST_SET: {cfg.DATASET.TEST_SET}")

# 测试训练集
print("\n测试训练集加载...")
try:
    train_dataset = BddDataset(cfg, is_train=True)
    print(f"训练集大小: {len(train_dataset)}")
except Exception as e:
    print(f"训练集加载失败: {e}")

# 测试验证集
print("\n测试验证集加载...")
try:
    val_dataset = BddDataset(cfg, is_train=False)
    print(f"验证集大小: {len(val_dataset)}")
except Exception as e:
    print(f"验证集加载失败: {e}")

# 检查文件是否存在
import os
train_path = os.path.join(cfg.DATASET.DATAROOT, cfg.DATASET.TRAIN_SET)
val_path = os.path.join(cfg.DATASET.DATAROOT, cfg.DATASET.TEST_SET)

print(f"\n训练集路径: {train_path}")
print(f"训练集存在: {os.path.exists(train_path)}")
if os.path.exists(train_path):
    print(f"训练集图片数: {len(os.listdir(train_path))}")

print(f"\n验证集路径: {val_path}")
print(f"验证集存在: {os.path.exists(val_path)}")
if os.path.exists(val_path):
    print(f"验证集图片数: {len(os.listdir(val_path))}")