#!/usr/bin/env python3
"""
YOLOP系列任务冲突检测比较实验
使用train.py主训练函数进行实验
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json
import time

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

def run_experiment(model_name='yolopx', epochs=20):
    """运行单个实验"""
    
    # 模型配置映射
    model_configs = {
        'yolopv1': 'yolop_v1_fixed',  # 使用修复版
        'yolopv2': 'yolop',  
        'yolopv3': 'yolop_v3_official',
        'yolopx': 'yolopx'
    }
    
    model_cfg = model_configs.get(model_name, 'yolopx')
    
    # TCI结果保存路径
    results_dir = Path('/workspace/YOLOPX/experiments/runs/conflict_detection')
    results_dir.mkdir(parents=True, exist_ok=True)
    tci_save_path = results_dir / f'{model_name}_tci_results.json'
    
    # 构建命令
    cmd = [
        sys.executable,
        'train.py',
        '--model_cfg', f'cfgs/models/{model_cfg}.yaml',
        '--data_cfg', 'cfgs/data/bdd100k_single_class.yaml',
        '--train_cfg', 'cfgs/train_default.yaml',
        '--use_bdd',  # 使用BddDataset
        '--enable_tci',  # 启用任务冲突检测
        '--tci_freq', '10',  # 每10个batch计算一次TCI
        '--save_tci', str(tci_save_path)  # 保存TCI结果
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {model_name.upper()}")
    print(f"模型配置: {model_cfg}")
    print(f"训练轮数: {epochs}")
    print(f"启用任务冲突检测: 是")
    print(f"TCI结果保存路径: {tci_save_path}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # 改变工作目录到项目根目录
    os.chdir(PROJECT_ROOT)
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU
    
    try:
        # 执行命令
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n✓ 实验 {model_name} 完成!")
        
        # 读取TCI结果
        if tci_save_path.exists():
            with open(tci_save_path, 'r') as f:
                tci_results = json.load(f)
            return True, tci_results.get('final_tci', None)
        return True, None
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 实验 {model_name} 失败: {e}")
        return False, None
    except Exception as e:
        print(f"\n✗ 运行时错误: {e}")
        return False, None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOP系列任务冲突检测比较实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['yolopv1', 'yolopv2', 'yolopv3', 'yolopx', 'all'],
        default='all',
        help='选择要运行的模型'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20,
        help='训练轮数'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # 运行所有模型
        models = ['yolopv1', 'yolopv2', 'yolopv3', 'yolopx']
        results = {}
        tci_results = {}
        
        for model_name in models:
            success, final_tci = run_experiment(model_name, args.epochs)
            results[model_name] = success
            if final_tci is not None:
                tci_results[model_name] = final_tci
        
        # 打印总结
        print("\n" + "="*60)
        print("实验总结:")
        print("="*60)
        print("\n状态:")
        for model, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  {model}: {status}")
        
        if tci_results:
            print("\n任务冲突强度(TCI):")
            for model, tci in tci_results.items():
                print(f"  {model}: {tci:.4f}")
        
        # 保存总结
        summary = {
            'experiment': 'YOLOP系列任务冲突检测比较',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'epochs': args.epochs,
                'models': models
            },
            'results': {
                'status': results,
                'tci': tci_results
            }
        }
        
        results_dir = Path('/workspace/YOLOPX/experiments/runs/conflict_detection')
        with open(results_dir / 'comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print(f"总结已保存至: {results_dir / 'comparison_summary.json'}")
        
    else:
        # 运行单个模型
        run_experiment(args.model, args.epochs)


if __name__ == '__main__':
    main()