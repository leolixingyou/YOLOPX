#!/usr/bin/env python3
"""
测试模型输入格式
"""
import sys
import torch
import yaml
import numpy as np

sys.path.append('/workspace/YOLOPX/v2')

from models.builder import get_net_from_yaml
from data.unified_dataset import BddDataset
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

# 测试配置
MODELS = {
    'yolopx': '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml',
    'yolop': '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml',
    'yolop_v3': '/workspace/YOLOPX/v2/cfgs/models/yolop_v3_official.yaml'
}

def test_model_input(model_name):
    """测试模型的输入格式"""
    print(f"\n测试模型: {model_name}")
    print("-" * 50)
    
    # 加载模型
    try:
        model = get_net_from_yaml(MODELS[model_name])
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 加载数据集配置
    data_config = yaml.safe_load(open('/workspace/YOLOPX/experiments/base_models_comparison/configs/bdd100k_experiment.yaml'))
    train_config = yaml.safe_load(open('/workspace/YOLOPX/v2/cfgs/train_v2.yaml'))
    
    # 合并配置
    cfg = edict()
    cfg.update(train_config)
    cfg.DATASET = edict(data_config['DATASET'])
    cfg.DATASET.NUMBER_IMAGE = 10  # 只需要几张图片测试
    cfg.DATASET.NUMBER_VAL = 5
    
    # 创建数据集
    try:
        dataset = BddDataset(cfg=cfg, is_train=True)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=BddDataset.collate_fn
        )
        print(f"✅ 数据集创建成功: {len(dataset)} 样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {str(e)}")
        return
    
    # 获取一个批次数据
    try:
        for images, targets, _, _ in dataloader:
            print(f"\n输入形状: {images.shape}")
            print(f"输入范围: [{images.min():.2f}, {images.max():.2f}]")
            
            # 将数据移到设备上
            images = images.to(device)
            
            # 测试前向传播
            with torch.no_grad():
                try:
                    outputs = model(images)
                    print(f"✅ 前向传播成功!")
                    
                    # 打印输出格式
                    if isinstance(outputs, dict):
                        print("\n输出字典内容:")
                        for k, v in outputs.items():
                            if hasattr(v, 'shape'):
                                print(f"  {k}: {v.shape}")
                            elif isinstance(v, (list, tuple)):
                                print(f"  {k}: {type(v).__name__} of {len(v)} items")
                                if len(v) > 0 and hasattr(v[0], 'shape'):
                                    print(f"    First item shape: {v[0].shape}")
                    else:
                        print(f"输出类型: {type(outputs)}")
                        
                except Exception as e:
                    print(f"❌ 前向传播失败: {str(e)}")
                    # 尝试创建简单的测试输入
                    print("\n尝试不同的输入尺寸...")
                    for h, w in [(384, 640), (640, 640), (416, 416)]:
                        test_input = torch.randn(1, 3, h, w).to(device)
                        try:
                            outputs = model(test_input)
                            print(f"✅ 尺寸 {h}x{w} 成功!")
                            break
                        except Exception as e2:
                            print(f"❌ 尺寸 {h}x{w} 失败: {str(e2)}")
            
            break  # 只测试一个批次
            
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")

def main():
    """测试所有模型"""
    print("🔍 测试YOLOP系列模型输入格式")
    
    for model_name in MODELS.keys():
        test_model_input(model_name)

if __name__ == '__main__':
    main()