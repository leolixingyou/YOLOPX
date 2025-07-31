#!/usr/bin/env python3
"""
创建测试数据以运行实验
"""
import os
from pathlib import Path

def create_test_data():
    """创建最小的测试数据集"""
    base_dir = Path('/workspace/YOLOPX/test_bdd100k')
    
    # 创建目录结构
    (base_dir / 'images' / '100k' / 'train').mkdir(parents=True, exist_ok=True)
    (base_dir / 'images' / '100k' / 'val').mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels' / '100k' / 'train').mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels' / '100k' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 创建train.txt和val.txt
    train_images = []
    val_images = []
    
    # 生成虚拟图片名
    for i in range(300):
        img_name = f'image_{i:06d}.jpg'
        if i < 200:
            train_images.append(img_name)
        else:
            val_images.append(img_name)
    
    # 写入文件列表
    with open(base_dir / 'train.txt', 'w') as f:
        for img in train_images:
            f.write(f"{img}\n")
            
    with open(base_dir / 'val.txt', 'w') as f:
        for img in val_images:
            f.write(f"{img}\n")
    
    print(f"✅ 测试数据创建完成:")
    print(f"   路径: {base_dir}")
    print(f"   训练图片: {len(train_images)}")
    print(f"   验证图片: {len(val_images)}")
    
    return str(base_dir)

if __name__ == '__main__':
    create_test_data()