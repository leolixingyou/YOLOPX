#!/usr/bin/env python3
"""
简化的实验脚本，直接调用train.py
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment(model_name='yolopx'):
    """运行单个实验"""
    
    model_configs = {
        'yolopv1': 'yolop_v1_official',
        'yolopv2': 'yolop',  
        'yolopv3': 'yolop_v3_official',
        'yolopx': 'yolopx'
    }
    
    model_cfg = model_configs.get(model_name, 'yolopx')
    
    # 改变工作目录
    os.chdir(PROJECT_ROOT)
    
    # 直接调用train.py
    cmd = [
        sys.executable,
        'train.py',
        '--model_cfg', f'cfgs/models/{model_cfg}.yaml',
        '--data_cfg', 'cfgs/data/bdd100k_single_class.yaml',
        '--train_cfg', 'cfgs/train_default.yaml'
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {model_name.upper()}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    # 禁用多进程以避免卡住
    env['OMP_NUM_THREADS'] = '1'
    
    # 执行命令
    process = subprocess.Popen(cmd, env=env)
    process.wait()
    
    return process.returncode == 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolopx',
                       choices=['yolopv1', 'yolopv2', 'yolopv3', 'yolopx'])
    args = parser.parse_args()
    
    success = run_experiment(args.model)
    print(f"\n实验 {'成功' if success else '失败'}!")