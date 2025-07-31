#!/usr/bin/env python3
"""
横向对比实验控制脚本
比较 YOLOP v1, v2, v3 和 YOLOPX 的任务冲突特性
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append('/workspace/YOLOPX/v2')
sys.path.append('/workspace/YOLOPX')

# 实验配置
EXPERIMENT_CONFIG = {
    'train_images': 200,
    'val_images': 100,
    'epochs': 20,
    'batch_size': 4,
    'seed': 42,
    'data_cfg': '/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml'
}

# 模型配置
MODELS = {
    'yolopv1': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop_v1_official.yaml',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)'
    },
    'yolopv2': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',  # v2使用改进的ELAN
        'description': 'YOLOP v2 (anchor-based, E-ELAN)'
    },
    'yolopv3': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolop_v3_official.yaml',
        'description': 'YOLOP v3 (anchor-based, ELAN-W with SimAM)'
    },
    'yolopx': {
        'config': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    }
}

def create_experiment_config(model_name, timestamp):
    """创建实验配置文件"""
    base_dir = Path('/workspace/YOLOPX/experiments/base_models_comparison')
    config_dir = base_dir / 'configs'
    
    # 读取基础训练配置
    with open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml', 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    # 更新配置
    train_cfg['EPOCHS'] = EXPERIMENT_CONFIG['epochs']
    train_cfg['DATASET']['BATCH_SIZE'] = EXPERIMENT_CONFIG['batch_size']
    train_cfg['DATASET']['NUMBER_IMAGE'] = EXPERIMENT_CONFIG['train_images']
    train_cfg['DATASET']['NUMBER_VAL'] = EXPERIMENT_CONFIG['val_images']
    train_cfg['SEED'] = EXPERIMENT_CONFIG['seed']
    
    # 设置wandb项目名和运行名
    experiment_name = f"{model_name}_{timestamp}"
    train_cfg['WANDB'] = {
        'PROJECT': 'yolopx-base-comparison',
        'NAME': experiment_name,
        'MODE': 'online'
    }
    
    # 设置输出目录
    train_cfg['OUTPUT_DIR'] = f'/workspace/YOLOPX/experiments/base_models_comparison/results/{experiment_name}'
    
    # 保存配置
    config_path = config_dir / f'{experiment_name}_train.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(train_cfg, f, default_flow_style=False)
    
    return config_path

def run_training(model_name, config_path, model_config):
    """运行训练"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_config_path = create_experiment_config(model_name, timestamp)
    
    # 构建训练命令
    cmd = [
        'python', '/workspace/YOLOPX/v2/train.py',
        '--cfg', str(train_config_path),
        '--model-cfg', model_config,
        '--data-cfg', EXPERIMENT_CONFIG['data_cfg'],
        '--logdir', f'/workspace/YOLOPX/experiments/base_models_comparison/logs/{model_name}_{timestamp}'
    ]
    
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODELS[model_name]['description']}")
    print(f"时间戳: {timestamp}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{model_name} 训练完成!")
        return {
            'status': 'success',
            'timestamp': timestamp,
            'output_dir': f'/workspace/YOLOPX/experiments/base_models_comparison/results/{model_name}_{timestamp}'
        }
    except subprocess.CalledProcessError as e:
        print(f"{model_name} 训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return {
            'status': 'failed',
            'timestamp': timestamp,
            'error': str(e)
        }

def collect_results(results):
    """收集并保存实验结果"""
    summary = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': EXPERIMENT_CONFIG,
        'models': {}
    }
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            # 尝试读取训练日志获取最终指标
            log_file = Path(result['output_dir']) / 'train.log'
            metrics = {}
            
            # TODO: 解析日志文件获取TCI、损失等指标
            
            summary['models'][model_name] = {
                'description': MODELS[model_name]['description'],
                'timestamp': result['timestamp'],
                'output_dir': result['output_dir'],
                'metrics': metrics
            }
        else:
            summary['models'][model_name] = {
                'description': MODELS[model_name]['description'],
                'status': 'failed',
                'error': result.get('error', 'Unknown error')
            }
    
    # 保存汇总结果
    summary_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n实验汇总已保存至: {summary_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help='要训练的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行1个epoch')
    
    args = parser.parse_args()
    
    if args.debug:
        EXPERIMENT_CONFIG['epochs'] = 1
        EXPERIMENT_CONFIG['train_images'] = 20
        EXPERIMENT_CONFIG['val_images'] = 10
    
    print("YOLOP系列模型横向对比实验")
    print(f"实验配置: {EXPERIMENT_CONFIG}")
    print(f"待训练模型: {args.models}")
    
    # 运行所有模型训练
    results = {}
    for model_name in args.models:
        if model_name in MODELS:
            result = run_training(model_name, MODELS[model_name]['config'], MODELS[model_name]['config'])
            results[model_name] = result
        else:
            print(f"警告: 未知模型 {model_name}")
    
    # 收集结果
    summary = collect_results(results)
    
    print("\n实验完成!")
    print("请查看wandb项目 'yolopx-base-comparison' 获取详细结果")
    
    return summary

if __name__ == '__main__':
    main()