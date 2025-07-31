#!/usr/bin/env python3
"""
YOLOP系列模型横向对比实验
基于v2/run_conflict_comparison_experiment.py的简化版本
"""
import os
import sys
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path

# 添加必要的路径
sys.path.insert(0, '/workspace/YOLOPX/v2')

# 实验配置
CONFIG = {
    'train_images': 200,
    'epochs': 20,
    'batch_size': 4,
    'lr': 0.001,
    'seed': 42,
    'wandb_project': 'yolopx-base-comparison'
}

# 模型列表
MODELS = {
    'yolopx': {
        'model_config': 'yolopx',
        'description': 'YOLOPX (anchor-free, ELANNet)',
        'conflict_solver': 'original'
    },
    'yolopv1': {
        'model_config': 'yolop_v1_official',
        'description': 'YOLOP v1 (anchor-based, CSP-Darknet)',
        'conflict_solver': 'original'
    },
    'yolopv2': {
        'model_config': 'yolop',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)',
        'conflict_solver': 'original'
    },
    'yolopv3': {
        'model_config': 'yolop_v3_official',
        'description': 'YOLOP v3 (anchor-based, ELAN-W)',
        'conflict_solver': 'original'
    }
}

def run_model_training(model_name):
    """运行单个模型的训练"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"描述: {MODELS[model_name]['description']}")
    print(f"时间戳: {timestamp}")
    print(f"{'='*60}\n")
    
    # 构建命令
    cmd = [
        'python3', '/workspace/YOLOPX/v2/run_conflict_comparison_experiment.py',
        '--model_name', MODELS[model_name]['model_config'],
        '--epochs', str(CONFIG['epochs']),
        '--train_images', str(CONFIG['train_images']),
        '--batch_size', str(CONFIG['batch_size']),
        '--lr', str(CONFIG['lr']),
        '--seed', str(CONFIG['seed']),
        '--methods', MODELS[model_name]['conflict_solver'],
        '--wandb_project', CONFIG['wandb_project'],
        '--wandb_name', run_name,
        '--save_best_model',
        '--logdir', f'/workspace/YOLOPX/experiments/base_models_comparison/runs/{run_name}'
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    # 创建日志目录
    log_dir = Path(f'/workspace/YOLOPX/experiments/base_models_comparison/logs/{run_name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'
    
    # 记录开始时间
    start_time = datetime.now()
    
    try:
        # 运行训练
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 实时显示和保存输出
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
        
        # 计算训练时长
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if process.returncode == 0:
            print(f"\n✅ {model_name} 训练成功完成！")
            print(f"⏱️  训练时长: {duration/60:.2f} 分钟")
            
            # 尝试读取最终指标
            metrics = {}
            log_content = log_file.read_text()
            
            # 从日志中提取TCI
            for line in log_content.split('\n'):
                if 'Average TCI:' in line:
                    try:
                        tci = float(line.split('Average TCI:')[1].strip())
                        metrics['avg_tci'] = tci
                    except:
                        pass
                elif 'Final validation loss:' in line:
                    try:
                        val_loss = float(line.split('Final validation loss:')[1].strip())
                        metrics['final_val_loss'] = val_loss
                    except:
                        pass
            
            return {
                'status': 'success',
                'timestamp': timestamp,
                'duration': duration,
                'metrics': metrics,
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

def generate_report(results):
    """生成实验报告"""
    report = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'models': {}
    }
    
    # 收集各模型结果
    for model_name, result in results.items():
        model_info = {
            'description': MODELS[model_name]['description'],
            'status': result['status'],
            'timestamp': result['timestamp']
        }
        
        if result['status'] == 'success':
            model_info['duration_minutes'] = result['duration'] / 60
            model_info['metrics'] = result.get('metrics', {})
            model_info['log_file'] = result['log_file']
        else:
            model_info['error'] = result.get('error', 'Unknown error')
        
        report['models'][model_name] = model_info
    
    # 保存JSON报告
    report_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/experiment_report.json')
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成Markdown报告
    md_path = Path('/workspace/YOLOPX/experiments/base_models_comparison/results/comparison_report.md')
    
    with open(md_path, 'w') as f:
        f.write("# YOLOP系列模型横向对比实验报告\n\n")
        f.write(f"**实验日期**: {report['experiment_date']}\n\n")
        
        f.write("## 实验配置\n\n")
        f.write(f"- 训练图片数: {CONFIG['train_images']}\n")
        f.write(f"- 训练轮数: {CONFIG['epochs']}\n")
        f.write(f"- 批次大小: {CONFIG['batch_size']}\n")
        f.write(f"- 学习率: {CONFIG['lr']}\n")
        f.write(f"- 随机种子: {CONFIG['seed']}\n\n")
        
        f.write("## 模型对比结果\n\n")
        f.write("| 模型 | 描述 | 状态 | 训练时长 | 平均TCI | 最终验证损失 |\n")
        f.write("|------|------|------|----------|---------|-------------|\n")
        
        for name, info in report['models'].items():
            status = "✅" if info['status'] == 'success' else "❌"
            duration = f"{info.get('duration_minutes', 0):.1f}分" if 'duration_minutes' in info else "N/A"
            avg_tci = f"{info.get('metrics', {}).get('avg_tci', 0):.4f}" if 'metrics' in info else "N/A"
            val_loss = f"{info.get('metrics', {}).get('final_val_loss', 0):.4f}" if 'metrics' in info else "N/A"
            
            f.write(f"| {name} | {info['description']} | {status} | {duration} | {avg_tci} | {val_loss} |\n")
        
        f.write("\n## 关键发现\n\n")
        
        # 分析TCI差异
        tci_values = {}
        for name, info in report['models'].items():
            if info['status'] == 'success' and 'metrics' in info and 'avg_tci' in info['metrics']:
                tci_values[name] = info['metrics']['avg_tci']
        
        if len(tci_values) >= 2:
            sorted_models = sorted(tci_values.items(), key=lambda x: x[1])
            f.write(f"- **最低任务冲突**: {sorted_models[0][0]} (TCI={sorted_models[0][1]:.4f})\n")
            f.write(f"- **最高任务冲突**: {sorted_models[-1][0]} (TCI={sorted_models[-1][1]:.4f})\n")
            
            # 计算anchor-based vs anchor-free的平均TCI
            anchor_based_tci = []
            anchor_free_tci = []
            
            for name, tci in tci_values.items():
                if name == 'yolopx':
                    anchor_free_tci.append(tci)
                else:
                    anchor_based_tci.append(tci)
            
            if anchor_based_tci and anchor_free_tci:
                avg_anchor_based = sum(anchor_based_tci) / len(anchor_based_tci)
                avg_anchor_free = sum(anchor_free_tci) / len(anchor_free_tci)
                
                f.write(f"\n### Anchor-based vs Anchor-free对比\n")
                f.write(f"- **Anchor-based平均TCI**: {avg_anchor_based:.4f}\n")
                f.write(f"- **Anchor-free平均TCI**: {avg_anchor_free:.4f}\n")
                
                diff_percent = (avg_anchor_free - avg_anchor_based) / avg_anchor_based * 100
                if diff_percent > 0:
                    f.write(f"- **结论**: Anchor-free方法的任务冲突比Anchor-based高{diff_percent:.1f}%\n")
                else:
                    f.write(f"- **结论**: Anchor-free方法的任务冲突比Anchor-based低{-diff_percent:.1f}%\n")
        
        f.write("\n## WandB项目\n\n")
        f.write(f"详细的训练曲线和指标请查看: [{CONFIG['wandb_project']}](https://wandb.ai/{CONFIG['wandb_project']})\n")
    
    print(f"\n📊 实验报告已保存:")
    print(f"   - JSON: {report_path}")
    print(f"   - Markdown: {md_path}")
    
    return report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help='要训练的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式（少量epoch和数据）')
    
    args = parser.parse_args()
    
    if args.debug:
        CONFIG['epochs'] = 2
        CONFIG['train_images'] = 50
        print("🔧 调试模式: epochs=2, train_images=50\n")
    
    print("="*60)
    print("🚀 YOLOP系列模型横向对比实验")
    print(f"📊 配置: epochs={CONFIG['epochs']}, train_images={CONFIG['train_images']}")
    print(f"🤖 模型: {args.models}")
    print("="*60)
    
    # 创建必要的目录
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/runs', exist_ok=True)
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/logs', exist_ok=True)
    os.makedirs('/workspace/YOLOPX/experiments/base_models_comparison/results', exist_ok=True)
    
    # 运行各模型训练
    results = {}
    for model_name in args.models:
        if model_name in MODELS:
            result = run_model_training(model_name)
            results[model_name] = result
        else:
            print(f"⚠️  未知模型: {model_name}")
    
    # 生成报告
    report = generate_report(results)
    
    print("\n" + "="*60)
    print("✅ 实验完成！")
    print(f"📁 结果目录: /workspace/YOLOPX/experiments/base_models_comparison/")
    print(f"📈 WandB项目: {CONFIG['wandb_project']}")
    print("="*60)

if __name__ == '__main__':
    main()