#!/usr/bin/env python3
"""
简化版训练脚本，用于运行YOLOP系列单类检测实验
所有参数都有默认值，可以直接在VSCode中按F5运行
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """主函数，调用原始的train.py"""
    parser = argparse.ArgumentParser(description='YOLOP系列单类检测实验')
    parser.add_argument('--model', type=str, 
                       choices=['yolopv1', 'yolopv2', 'yolopv3', 'yolopx'],
                       default='yolopx',
                       help='选择要运行的模型')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批大小')
    
    args = parser.parse_args()
    
    # 模型配置映射
    model_configs = {
        'yolopv1': 'cfgs/models/yolop_v1_official.yaml',
        'yolopv2': 'cfgs/models/yolop.yaml',
        'yolopv3': 'cfgs/models/yolop_v3_official.yaml',
        'yolopx': 'cfgs/models/yolopx.yaml'
    }
    
    # 构建训练命令
    train_script = PROJECT_ROOT / 'train.py'
    model_cfg = model_configs[args.model]
    data_cfg = 'cfgs/data/bdd100k_single_class.yaml'
    train_cfg = 'cfgs/train_default.yaml'
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
    
    # 构建命令行参数
    cmd_args = [
        sys.executable,  # python可执行文件
        str(train_script),
        '--config', train_cfg,
        '--data-cfg', data_cfg,
        '--model-cfg', model_cfg,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--log-dir', f'runs/{args.model}_single_class',
        '--project', 'yolop-single-class-exp'
    ]
    
    # 打印实验信息
    print(f"\n{'='*60}")
    print(f"运行实验: {args.model.upper()}")
    print(f"模型配置: {model_cfg}")
    print(f"数据配置: {data_cfg}")
    print(f"训练配置: {train_cfg}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"{'='*60}\n")
    
    # 执行训练
    import subprocess
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"\n实验 {args.model} 完成!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 实验 {args.model} 失败")
        print(f"返回码: {e.returncode}")
        return 1
    except FileNotFoundError:
        # 如果train.py不存在，打印使用说明
        print(f"\n错误: 找不到训练脚本 {train_script}")
        print("\n请使用以下命令运行实验:")
        print(f"cd {PROJECT_ROOT}")
        print(f"python train.py --data-cfg {data_cfg} --model-cfg {model_cfg}")
        return 1

if __name__ == '__main__':
    sys.exit(main())