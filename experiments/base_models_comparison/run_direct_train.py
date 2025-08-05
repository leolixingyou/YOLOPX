#!/usr/bin/env python3
"""
直接使用原有train.py的实验脚本
"""

import os
import sys
import subprocess
from pathlib import Path

# 设置路径
YOLOP_ROOT = Path(__file__).resolve().parent.parent.parent / 'yolop_series'
os.chdir(YOLOP_ROOT)

# 设置环境变量
os.environ['PYTHONPATH'] = str(YOLOP_ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 运行训练
cmd = [
    sys.executable,
    'train.py',
    '--model_cfg', 'cfgs/models/yolopx.yaml',
    '--data_cfg', 'cfgs/data/bdd100k_single_class.yaml', 
    '--train_cfg', 'cfgs/train_default.yaml'
]

print(f"运行命令: {' '.join(cmd)}")
subprocess.run(cmd)