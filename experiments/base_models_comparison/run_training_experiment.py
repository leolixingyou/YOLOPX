#!/usr/bin/env python3
"""
YOLOP系列模型横向对比训练实验
基于v2/train.py的包装脚本
"""
import os
import sys
import yaml
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 添加v2到Python路径
sys.path.insert(0, '/workspace/YOLOPX/v2')

# 实验配置
EXPERIMENT_CONFIG = {
    'epochs': 20,
    'train_images': 200,
    'val_images': 100,
    'batch_size': 4,
    'lr': 0.001,
    'project': 'yolopx-base-comparison'
}

# 模型配置
MODELS = {
    'yolopx': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
        'description': 'YOLOPX (anchor-free, ELANNet)'
    },
    'yolop_v2': {
        'model_cfg': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
        'description': 'YOLOP v2 (anchor-based, E-ELAN)'
    },
    # v1和v3暂时跳过，因为有模块问题
}

def create_experiment_config(model_name):
    """为每个模型创建实验配置"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{model_name}_{timestamp}"
    
    # 创建数据配置
    data_config = {
        'DATASET': {
            'DATAROOT': '/workspace/YOLOPX/bdd100k',
            'LABELROOT': '/workspace/YOLOPX/bdd100k/labels',
            'MASKROOT': '/workspace/YOLOPX/bdd100k/labels',
            'LANEROOT': '/workspace/YOLOPX/bdd100k/labels',
            'TRAIN_FILE': 'train.txt',
            'VAL_FILE': 'val.txt',
            'TEST_FILE': 'val.txt',
            'NUMBER_IMAGE': EXPERIMENT_CONFIG['train_images'],
            'NUMBER_VAL': EXPERIMENT_CONFIG['val_images']
        }
    }
    
    # 创建训练配置（覆盖默认值）
    train_override = {
        'TRAIN': {
            'END_EPOCH': EXPERIMENT_CONFIG['epochs'],
            'BATCH_SIZE_PER_GPU': EXPERIMENT_CONFIG['batch_size'],
            'LR0': EXPERIMENT_CONFIG['lr']
        },
        'TEST': {
            'BATCH_SIZE_PER_GPU': EXPERIMENT_CONFIG['batch_size']
        },
        'WANDB': {
            'PROJECT': EXPERIMENT_CONFIG['project'],
            'NAME': experiment_name,
            'MODE': 'online'
        }
    }
    
    # 保存配置文件
    config_dir = Path('/tmp/yolop_experiments')
    config_dir.mkdir(exist_ok=True)
    
    data_config_path = config_dir / f'{experiment_name}_data.yaml'
    train_override_path = config_dir / f'{experiment_name}_train.yaml'
    
    with open(data_config_path, 'w') as f:
        yaml.dump(data_config, f)
    
    with open(train_override_path, 'w') as f:
        yaml.dump(train_override, f)
    
    return {
        'experiment_name': experiment_name,
        'data_config': str(data_config_path),
        'train_override': str(train_override_path),
        'model_cfg': MODELS[model_name]['model_cfg']
    }

def run_training(model_name):
    """运行单个模型的训练"""
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"描述: {MODELS[model_name]['description']}")
    print(f"{'='*60}\n")
    
    # 创建实验配置
    config = create_experiment_config(model_name)
    
    # 设置日志目录
    log_dir = Path(f'/workspace/YOLOPX/experiments/base_models_comparison/runs/{config["experiment_name"]}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建训练命令
    cmd = [
        'python3', '/workspace/YOLOPX/v2/train.py',
        '--model_cfg', config['model_cfg'],
        '--data_cfg', config['data_config'],
        '--train_cfg', '/workspace/YOLOPX/v2/cfgs/train_v2.yaml',
        '--logdir', str(log_dir)
    ]
    
    # 记录实验信息
    experiment_info = {
        'model': model_name,
        'description': MODELS[model_name]['description'],
        'config': EXPERIMENT_CONFIG,
        'timestamp': datetime.now().isoformat(),
        'command': ' '.join(cmd)
    }
    
    with open(log_dir / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"执行命令: {' '.join(cmd)}")
    print(f"日志目录: {log_dir}")
    
    # 运行训练
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = '/workspace/YOLOPX/v2:' + env.get('PYTHONPATH', '')
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # 执行训练
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd='/workspace/YOLOPX/v2'
        )
        
        # 保存输出
        with open(log_dir / 'stdout.log', 'w') as f:
            f.write(result.stdout)
        with open(log_dir / 'stderr.log', 'w') as f:
            f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练完成!")
            return True
        else:
            print(f"❌ {model_name} 训练失败!")
            print(f"错误信息: {result.stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} 训练异常: {str(e)}")
        return False

def generate_comparison_report(results):
    """生成对比报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'/workspace/YOLOPX/experiments/base_models_comparison/results/training_report_{timestamp}.json'
    
    report = {
        'experiment_config': EXPERIMENT_CONFIG,
        'timestamp': timestamp,
        'models': results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 实验报告已保存: {report_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    for model_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model_name}: {status} - {MODELS[model_name]['description']}")

def main():
    """主函数"""
    print("🚀 YOLOP系列模型横向对比训练实验")
    print(f"📊 配置: {EXPERIMENT_CONFIG}")
    
    results = {}
    
    # 先只运行YOLOPX测试
    for model_name in ['yolopx']:
        success = run_training(model_name)
        results[model_name] = success
    
    # 生成报告
    generate_comparison_report(results)

if __name__ == '__main__':
    main()