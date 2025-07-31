#!/usr/bin/env python3
"""
横向对比实验控制脚本 - V2版本
比较 YOLOP v1, v2, v3 和 YOLOPX 的任务冲突特性
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import subprocess

# 添加项目路径
sys.path.append('/workspace/YOLOPX/v2')

# 实验配置
EXPERIMENT_CONFIG = {
    'train_images': 200,
    'val_images': 100,
    'epochs': 20,
    'batch_size': 4,
    'seed': 42,
}

# 模型配置
MODELS = {
    'yolopv1': {
        'config': 'yolop_v1_official',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)'
    },
    'yolopv2': {
        'config': 'yolop',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)'
    },
    'yolopv3': {
        'config': 'yolop_v3_official',
        'description': 'YOLOP v3 (anchor-based, ELAN-W with SimAM)'
    },
    'yolopx': {
        'config': 'yolopx',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    }
}

def run_training(model_name):
    """运行单个模型的训练"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{model_name}_{timestamp}"
    
    # 创建输出目录
    output_dir = f'/workspace/YOLOPX/experiments/base_models_comparison/results/{experiment_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建训练命令 - 使用v2的train.py
    cmd = [
        'python', '/workspace/YOLOPX/v2/train.py',
        '--model_name', MODELS[model_name]['config'],
        '--epochs', str(EXPERIMENT_CONFIG['epochs']),
        '--batch_size', str(EXPERIMENT_CONFIG['batch_size']),
        '--num_train', str(EXPERIMENT_CONFIG['train_images']),
        '--num_val', str(EXPERIMENT_CONFIG['val_images']),
        '--seed', str(EXPERIMENT_CONFIG['seed']),
        '--output_dir', output_dir,
        '--wandb_project', 'yolopx-base-comparison',
        '--wandb_name', experiment_name,
        '--save_dir', output_dir,
        '--log_interval', '10',
        '--enable_tci', 'True'  # 启用任务冲突强度计算
    ]
    
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODELS[model_name]['description']}")
    print(f"时间戳: {timestamp}")
    print(f"输出目录: {output_dir}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行训练
    try:
        # 使用subprocess运行训练，实时显示输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        if process.returncode == 0:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n{model_name} 训练完成!")
            print(f"训练时长: {duration:.2f} 秒")
            
            return {
                'status': 'success',
                'timestamp': timestamp,
                'output_dir': output_dir,
                'duration': duration
            }
        else:
            print(f"\n{model_name} 训练失败，返回码: {process.returncode}")
            return {
                'status': 'failed',
                'timestamp': timestamp,
                'error': f'Process exited with code {process.returncode}'
            }
            
    except Exception as e:
        print(f"\n{model_name} 训练出错: {str(e)}")
        return {
            'status': 'failed',
            'timestamp': timestamp,
            'error': str(e)
        }

def collect_and_save_results(results):
    """收集并保存实验结果"""
    summary = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': EXPERIMENT_CONFIG,
        'models': {}
    }
    
    for model_name, result in results.items():
        model_info = {
            'description': MODELS[model_name]['description'],
            'status': result['status'],
            'timestamp': result['timestamp']
        }
        
        if result['status'] == 'success':
            model_info.update({
                'output_dir': result['output_dir'],
                'duration': result['duration']
            })
            
            # 尝试读取训练产生的指标文件
            metrics_file = Path(result['output_dir']) / 'final_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    model_info['metrics'] = metrics
        else:
            model_info['error'] = result.get('error', 'Unknown error')
        
        summary['models'][model_name] = model_info
    
    # 保存汇总结果
    summary_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n实验汇总已保存至: {summary_path}")
    return summary

def generate_comparison_report(summary):
    """生成对比报告"""
    report_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# YOLOP系列模型横向对比实验报告\n\n")
        f.write(f"实验日期: {summary['experiment_date']}\n\n")
        
        f.write("## 实验配置\n")
        f.write(f"- 训练图片数: {summary['config']['train_images']}\n")
        f.write(f"- 验证图片数: {summary['config']['val_images']}\n")
        f.write(f"- 训练轮数: {summary['config']['epochs']}\n")
        f.write(f"- 批次大小: {summary['config']['batch_size']}\n\n")
        
        f.write("## 模型结果对比\n\n")
        f.write("| 模型 | 描述 | 状态 | 训练时长 | TCI |\n")
        f.write("|------|------|------|----------|-----|\n")
        
        for model_name, info in summary['models'].items():
            status = "✅" if info['status'] == 'success' else "❌"
            duration = f"{info.get('duration', 0):.2f}s" if 'duration' in info else "N/A"
            tci = info.get('metrics', {}).get('final_tci', 'N/A') if 'metrics' in info else "N/A"
            
            f.write(f"| {model_name} | {info['description']} | {status} | {duration} | {tci} |\n")
        
        f.write("\n## 详细指标\n")
        # TODO: 添加更详细的指标对比
    
    print(f"对比报告已生成: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help='要训练的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行1个epoch')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序运行模型训练（默认并行）')
    
    args = parser.parse_args()
    
    if args.debug:
        EXPERIMENT_CONFIG['epochs'] = 1
        EXPERIMENT_CONFIG['train_images'] = 20
        EXPERIMENT_CONFIG['val_images'] = 10
        print("调试模式：epochs=1, train=20, val=10")
    
    print("="*60)
    print("YOLOP系列模型横向对比实验")
    print(f"实验配置: {EXPERIMENT_CONFIG}")
    print(f"待训练模型: {args.models}")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/results', exist_ok=True)
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/logs', exist_ok=True)
    
    # 运行所有模型训练
    results = {}
    for model_name in args.models:
        if model_name in MODELS:
            print(f"\n准备训练模型: {model_name}")
            result = run_training(model_name)
            results[model_name] = result
        else:
            print(f"警告: 未知模型 {model_name}")
    
    # 收集结果并生成报告
    summary = collect_and_save_results(results)
    generate_comparison_report(summary)
    
    print("\n="*60)
    print("实验完成!")
    print(f"结果保存在: /workspace/YOLOPX/experiments/base_models_comparison/results/")
    print("请查看wandb项目 'yolopx-base-comparison' 获取详细的训练曲线和指标")
    print("="*60)
    
    return summary

if __name__ == '__main__':
    main()