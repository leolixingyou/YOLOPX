#!/usr/bin/env python3
"""
YOLOP系列模型横向对比实验脚本
"""
import os
import sys
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# 实验基础配置
BASE_CONFIG = {
    'train_images': 200,
    'val_images': 100,
    'epochs': 20,
    'batch_size': 4,
    'seed': 42,
}

# 模型定义
MODELS = {
    'yolopx': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    },
    'yolopv1': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolop_v1_official.yaml',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)'
    },
    'yolopv2': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)'
    },
    'yolopv3': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolop_v3_official.yaml',
        'description': 'YOLOP v3 (anchor-based, ELAN-W)'
    }
}

def create_train_config(model_name, timestamp):
    """为每个模型创建训练配置"""
    experiment_name = f"{model_name}_{timestamp}"
    
    # 基础训练配置
    train_cfg = {
        # General Settings
        'LOG_DIR': f'/workspace/YOLOPX/experiments/base_models_comparison/logs/{experiment_name}/',
        'OUTPUT_DIR': f'/workspace/YOLOPX/experiments/base_models_comparison/results/{experiment_name}',
        'WORKERS': 4,
        'PIN_MEMORY': True,
        'PRINT_FREQ': 10,
        'DEBUG': False,
        'SEED': BASE_CONFIG['seed'],
        
        # Cudnn Settings
        'CUDNN': {
            'BENCHMARK': True,
            'DETERMINISTIC': False,
            'ENABLED': True
        },
        
        # Loss Function Settings
        'LOSS': {
            'FL_GAMMA': 2.0,
            'SEG_POS_WEIGHT': 1.0,
            'BOX_GAIN': 0.05,
            'CLS_GAIN': 0.5,
            'OBJ_GAIN': 1.0,
            'DA_SEG_GAIN': 0.2,
            'LL_SEG_GAIN': 0.3,
            'LL_IOU_GAIN': 0.5
        },
        
        # Training Hyperparameters
        'TRAIN': {
            'OPTIMIZER': 'adamw',
            'LR0': 0.001,
            'LRF': 0.1,
            'MOMENTUM': 0.937,
            'WD': 0.0005,
            'WARMUP_EPOCHS': 3.0,
            'WARMUP_MOMENTUM': 0.8,
            'WARMUP_BIASE_LR': 0.1,
            'BATCH_SIZE_PER_GPU': BASE_CONFIG['batch_size'],
            'SHUFFLE': True,
            'BEGIN_EPOCH': 0,
            'END_EPOCH': BASE_CONFIG['epochs']
        },
        
        # Testing/Validation Settings
        'TEST': {
            'BATCH_SIZE_PER_GPU': 8,
            'NMS_CONF_THRESHOLD': 0.001,
            'NMS_IOU_THRESHOLD': 0.6
        },
        
        # WandB配置
        'WANDB': {
            'ENABLE': True,
            'PROJECT': 'yolopx-base-comparison',
            'NAME': experiment_name,
            'MODE': 'online'
        },
        
        # 任务冲突检测
        'TASK_CONFLICT': {
            'ENABLE': True,
            'LOG_INTERVAL': 10
        }
    }
    
    # 保存训练配置
    config_dir = Path('/workspace/YOLOPX/experiments/base_models_comparison/configs')
    config_dir.mkdir(exist_ok=True)
    
    train_cfg_path = config_dir / f'{experiment_name}_train.yaml'
    with open(train_cfg_path, 'w') as f:
        yaml.dump(train_cfg, f, default_flow_style=False)
    
    return str(train_cfg_path)

def run_model_training(model_name):
    """运行单个模型的训练"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODELS[model_name]['description']}")
    print(f"时间戳: {timestamp}")
    print(f"{'='*60}\n")
    
    # 创建训练配置
    train_cfg_path = create_train_config(model_name, timestamp)
    
    # 构建训练命令
    cmd = [
        'python3', '/workspace/YOLOPX/v2/train.py',
        '--model_cfg', MODELS[model_name]['model_cfg'],
        '--data_cfg', '/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml',
        '--train_cfg', train_cfg_path
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 创建日志文件
    log_dir = Path(f'/workspace/YOLOPX/experiments/base_models_comparison/logs/{model_name}_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'
    
    try:
        # 运行训练，将输出同时保存到文件和终端
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, 
                bufsize=1
            )
            
            # 实时显示输出
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if process.returncode == 0:
            print(f"\n✅ {model_name} 训练完成！用时: {duration/60:.2f} 分钟")
            return {
                'status': 'success',
                'timestamp': timestamp,
                'duration': duration,
                'log_file': str(log_file)
            }
        else:
            print(f"\n❌ {model_name} 训练失败！返回码: {process.returncode}")
            return {
                'status': 'failed',
                'timestamp': timestamp,
                'error': f'Process exited with code {process.returncode}',
                'log_file': str(log_file)
            }
            
    except Exception as e:
        print(f"\n❌ {model_name} 训练出错: {str(e)}")
        return {
            'status': 'failed',
            'timestamp': timestamp,
            'error': str(e)
        }

def generate_summary_report(results):
    """生成实验汇总报告"""
    summary = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': BASE_CONFIG
        },
        'models': {}
    }
    
    # 收集每个模型的结果
    for model_name, result in results.items():
        model_summary = {
            'description': MODELS[model_name]['description'],
            'status': result['status'],
            'timestamp': result['timestamp']
        }
        
        if result['status'] == 'success':
            model_summary['duration_minutes'] = result['duration'] / 60
            model_summary['log_file'] = result['log_file']
            
            # TODO: 从日志文件中提取TCI等指标
            
        else:
            model_summary['error'] = result.get('error', 'Unknown error')
        
        summary['models'][model_name] = model_summary
    
    # 保存JSON格式的汇总
    summary_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/experiment_summary.json')
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 生成Markdown报告
    report_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# YOLOP系列模型横向对比实验报告\n\n")
        f.write(f"**实验日期**: {summary['experiment_info']['date']}\n\n")
        
        f.write("## 实验配置\n\n")
        f.write(f"- 训练图片数: {BASE_CONFIG['train_images']}\n")
        f.write(f"- 验证图片数: {BASE_CONFIG['val_images']}\n")
        f.write(f"- 训练轮数: {BASE_CONFIG['epochs']}\n")
        f.write(f"- 批次大小: {BASE_CONFIG['batch_size']}\n")
        f.write(f"- 随机种子: {BASE_CONFIG['seed']}\n\n")
        
        f.write("## 训练结果汇总\n\n")
        f.write("| 模型 | 描述 | 状态 | 训练时长 |\n")
        f.write("|------|------|------|----------|\n")
        
        for name, info in summary['models'].items():
            status = "✅ 成功" if info['status'] == 'success' else "❌ 失败"
            duration = f"{info.get('duration_minutes', 0):.2f} 分钟" if 'duration_minutes' in info else "N/A"
            f.write(f"| {name} | {info['description']} | {status} | {duration} |\n")
        
        f.write("\n## WandB项目链接\n\n")
        f.write("查看详细的训练曲线和指标: [yolopx-base-comparison](https://wandb.ai/your-team/yolopx-base-comparison)\n")
    
    print(f"\n📊 实验汇总已保存至: {summary_path}")
    print(f"📄 实验报告已保存至: {report_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=['yolopx', 'yolopv1'],  # 默认先测试两个模型
                        help='要训练的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式（1个epoch，少量数据）')
    
    args = parser.parse_args()
    
    # 调试模式配置
    if args.debug:
        BASE_CONFIG['epochs'] = 1
        BASE_CONFIG['train_images'] = 20
        BASE_CONFIG['val_images'] = 10
        print("🔧 调试模式启用: epochs=1, train=20, val=10\n")
    
    print("="*60)
    print("🚀 YOLOP系列模型横向对比实验")
    print(f"📊 实验配置: {BASE_CONFIG}")
    print(f"🤖 待训练模型: {args.models}")
    print("="*60)
    
    # 创建必要的目录
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/results', exist_ok=True)
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/logs', exist_ok=True)
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/configs', exist_ok=True)
    
    # 运行训练
    results = {}
    for model_name in args.models:
        result = run_model_training(model_name)
        results[model_name] = result
    
    # 生成报告
    summary = generate_summary_report(results)
    
    print("\n" + "="*60)
    print("✅ 实验完成！")
    print("📁 结果目录: /workspace/YOLOPX/experiments/base_models_comparison/results/")
    print("📈 WandB项目: yolopx-base-comparison")
    print("="*60)

if __name__ == '__main__':
    main()