#!/usr/bin/env python3
"""
运行YOLOP系列模型的完整训练实验
包括YOLOPv1, v2, v3和YOLOPX
"""
import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path

def run_training_experiment(model_name, config_path, epochs=20, batch_size=4):
    """运行单个模型的训练实验"""
    print(f"\n{'='*80}")
    print(f"开始训练: {model_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{model_name}_{timestamp}"
    
    # 确保在正确的目录
    os.chdir('/workspace/YOLOPX/yolop_series')
    
    # 构建训练命令
    cmd = [
        sys.executable, 
        'train.py',
        '--model_cfg', config_path,
        '--data_cfg', 'cfgs/data/bdd100k.yaml',
        '--train_cfg', 'cfgs/train_default.yaml'
    ]
    
    # 设置环境变量以启用wandb
    env = os.environ.copy()
    env['WANDB_PROJECT'] = 'yolopx-base-comparison'
    env['WANDB_NAME'] = f'{model_name}_{timestamp}'
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"日志目录: {log_dir}")
    
    try:
        # 运行训练
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600,  # 1小时超时
            env=env
        )
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练完成")
            # 提取TCI等指标
            output = result.stdout
            tci_lines = [line for line in output.split('\n') if 'TCI' in line or 'conflict' in line.lower()]
            return {
                'success': True,
                'log_dir': log_dir,
                'output': output,
                'tci_info': tci_lines
            }
        else:
            print(f"❌ {model_name} 训练失败")
            print(f"错误信息: {result.stderr}")
            return {
                'success': False,
                'log_dir': log_dir,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 训练超时")
        return {
            'success': False,
            'log_dir': log_dir,
            'error': "Training timeout"
        }
    except Exception as e:
        print(f"💥 {model_name} 训练出错: {str(e)}")
        return {
            'success': False,
            'log_dir': log_dir,
            'error': str(e)
        }

def run_simple_test(model_name, config_path):
    """运行简单的模型加载和前向传播测试"""
    print(f"\n测试模型加载: {model_name}")
    
    try:
        # 测试脚本
        test_script = f"""
import sys
sys.path.insert(0, '/workspace/YOLOPX/yolop_series')
import torch
from models.builder import get_net_from_yaml

# 加载模型
model = get_net_from_yaml('{config_path}')
model.eval()

# 测试前向传播
dummy_input = torch.randn(1, 3, 384, 640)
with torch.no_grad():
    outputs = model(dummy_input)
    
print(f"✅ {model_name} 模型加载成功")
print(f"输出数量: {{len(outputs) if isinstance(outputs, (list, tuple)) else 1}}")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ 加载失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 YOLOP系列模型训练对比实验")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 模型配置
    models = {
        'yolop_v1': 'cfgs/models/yolop_v1_official.yaml',
        'yolop_v2': 'cfgs/models/yolop.yaml', 
        'yolop_v3': 'cfgs/models/yolop_v3_official.yaml',
        'yolopx': 'cfgs/models/yolopx.yaml'
    }
    
    # 首先测试模型是否能加载
    print("\n📋 第一步：测试模型加载")
    print("="*60)
    
    loadable_models = {}
    for model_name, config_path in models.items():
        full_path = f'/workspace/YOLOPX/yolop_series/{config_path}'
        if run_simple_test(model_name, full_path):
            loadable_models[model_name] = full_path
    
    print(f"\n可加载的模型: {list(loadable_models.keys())}")
    
    # 询问用户是否继续
    if not loadable_models:
        print("❌ 没有可加载的模型，退出")
        return
    
    print("\n📋 第二步：运行小规模训练测试")
    print("="*60)
    
    # 运行小规模训练（仅1个epoch，用于测试）
    results = {}
    for model_name, config_path in loadable_models.items():
        print(f"\n开始训练 {model_name} (1 epoch测试)...")
        result = run_training_experiment(
            model_name, 
            config_path,
            epochs=1,
            batch_size=2
        )
        results[model_name] = result
    
    # 保存结果
    output_dir = Path('/workspace/YOLOPX/experiments/base_models_comparison/results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'training_test_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models_tested': list(loadable_models.keys()),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存: {results_file}")
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    
    for model_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{model_name}: {status}")
        if result.get('tci_info'):
            for line in result['tci_info'][:3]:
                print(f"  {line}")
    
    print("\n💡 建议：")
    print("1. 检查wandb项目页面查看训练曲线和可视化结果")
    print("2. 查看runs目录下的日志文件获取详细信息")
    print("3. 如需完整训练，修改epochs参数并重新运行")

if __name__ == '__main__':
    main()