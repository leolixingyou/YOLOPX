#!/usr/bin/env python3
"""
YOLOP系列模型横向对比实验包装脚本
使用已验证的run_conflict_comparison_experiment.py
"""
import os
import sys
import subprocess
from datetime import datetime

# 实验配置
EXPERIMENT_CONFIG = {
    'epochs': 20,
    'train_images': 200,
    'val_images': 100
}

def run_experiment(models_list):
    """运行实验"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"""
{'='*60}
🚀 YOLOP系列模型横向对比实验
📅 时间: {timestamp}
📊 配置:
   - 训练图片数: {EXPERIMENT_CONFIG['train_images']}
   - 验证图片数: {EXPERIMENT_CONFIG['val_images']} 
   - 训练轮数: {EXPERIMENT_CONFIG['epochs']}
   - 模型列表: {', '.join(models_list)}
{'='*60}
""")
    
    # 创建结果目录
    result_dir = f'/workspace/YOLOPX/experiments/base_models_comparison/results/{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        'python3', '/workspace/YOLOPX/v2/run_conflict_comparison_experiment.py',
        '--epochs', str(EXPERIMENT_CONFIG['epochs']),
        '--train_images', str(EXPERIMENT_CONFIG['train_images']),
        '--val_images', str(EXPERIMENT_CONFIG['val_images']),
        '--models'] + models_list
    
    print(f"执行命令: {' '.join(cmd)}\n")
    
    # 创建日志文件
    log_file = os.path.join(result_dir, 'experiment.log')
    
    try:
        # 运行实验
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd='/workspace/YOLOPX/v2'  # 设置工作目录
            )
            
            # 实时显示输出
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ 实验成功完成！")
            print(f"📁 结果保存在: {result_dir}")
            print(f"📄 日志文件: {log_file}")
        else:
            print(f"\n❌ 实验失败，返回码: {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ 实验出错: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOP系列模型横向对比实验')
    parser.add_argument('--models', nargs='+', 
                        choices=['yolopx_v2_anchor_free', 'yolop_v1_official', 'yolop_v3_official'],
                        default=['yolopx_v2_anchor_free', 'yolop_v1_official', 'yolop_v3_official'],
                        help='要测试的模型列表')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式（少量数据和epoch）')
    
    args = parser.parse_args()
    
    if args.debug:
        EXPERIMENT_CONFIG['epochs'] = 2
        EXPERIMENT_CONFIG['train_images'] = 50
        EXPERIMENT_CONFIG['val_images'] = 25
        print("🔧 调试模式启用\n")
    
    run_experiment(args.models)

if __name__ == '__main__':
    main()