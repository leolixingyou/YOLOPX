#!/usr/bin/env python3
"""
创建用于测试的虚拟BDD100K数据集
"""
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_bdd100k(base_path="/workspace/YOLOPX/dummy_bdd100k", num_images=50):
    """创建虚拟的BDD100K数据集结构"""
    
    # 创建目录结构
    paths = {
        'images': {
            'train': Path(base_path) / 'images' / 'train',
            'val': Path(base_path) / 'images' / 'val'
        },
        'labels': {
            'train': Path(base_path) / 'labels' / 'train',
            'val': Path(base_path) / 'labels' / 'val'
        },
        'det_labels': Path(base_path) / 'det_labels',
        'da_seg_masks': Path(base_path) / 'da_seg_masks',
        'll_seg_masks': Path(base_path) / 'll_seg_masks'
    }
    
    # 创建所有目录
    for category in paths.values():
        if isinstance(category, dict):
            for path in category.values():
                path.mkdir(parents=True, exist_ok=True)
        else:
            category.mkdir(parents=True, exist_ok=True)
    
    # 创建训练和验证图片
    for split in ['train', 'val']:
        split_num = num_images if split == 'train' else num_images // 5
        
        for i in range(split_num):
            # 创建随机图片
            img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            
            # 保存图片
            img_name = f"{split}_{i:06d}.jpg"
            img_path = paths['images'][split] / img_name
            img_pil.save(img_path)
            
            # 创建对应的标签文件（BDD100K JSON格式）
            label_path = paths['labels'][split] / f"{split}_{i:06d}.json"
            
            # 随机生成1-3个边界框
            num_boxes = np.random.randint(1, 4)
            objects = []
            
            for _ in range(num_boxes):
                # 生成边界框坐标
                x1 = np.random.randint(100, 1000)
                y1 = np.random.randint(100, 600)
                width = np.random.randint(50, 200)
                height = np.random.randint(50, 200)
                x2 = min(x1 + width, 1279)
                y2 = min(y1 + height, 719)
                
                obj = {
                    'category': 'car',
                    'box2d': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }
                objects.append(obj)
            
            # 创建BDD100K格式的JSON标签
            label_data = {
                'frames': [{
                    'objects': objects
                }]
            }
            
            with open(label_path, 'w') as f:
                json.dump(label_data, f)
            
            # 创建分割掩码（可行驶区域）
            da_mask = np.zeros((720, 1280), dtype=np.uint8)
            # 简单的可行驶区域：图片下半部分
            da_mask[360:, :] = 1
            da_mask_pil = Image.fromarray(da_mask * 255)
            da_mask_path = paths['da_seg_masks'] / f"{split}_{i:06d}.png"
            da_mask_pil.save(da_mask_path)
            
            # 创建车道线掩码
            ll_mask = np.zeros((720, 1280), dtype=np.uint8)
            # 简单的车道线：几条垂直线
            for x in [320, 640, 960]:
                ll_mask[400:, x-5:x+5] = 1
            ll_mask_pil = Image.fromarray(ll_mask * 255)
            ll_mask_path = paths['ll_seg_masks'] / f"{split}_{i:06d}.png"
            ll_mask_pil.save(ll_mask_path)
    
    # 创建配置文件
    config = {
        'base_path': str(base_path),
        'num_train': num_images,
        'num_val': num_images // 5,
        'image_size': [720, 1280],
        'classes': ['car']
    }
    
    config_path = Path(base_path) / 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 虚拟数据集创建完成: {base_path}")
    print(f"   训练图片: {num_images}")
    print(f"   验证图片: {num_images // 5}")
    
    # 创建对应的yaml配置
    yaml_config = f"""# Dummy BDD100k Dataset Configuration

DATASET:
  DATAROOT: '{base_path}/images'
  LABELROOT: '{base_path}/labels'
  MASKROOT: '{base_path}/da_seg_masks'
  LANEROOT: '{base_path}/ll_seg_masks'
  DATASET: BddDataset
  TRAIN_SET: train
  TEST_SET: val
  DATA_FORMAT: jpg
  SELECT_DATA: false
  ORG_IMG_SIZE: [720, 1280]
  IMAGE_SIZE: [384, 640] # h, w
  NUM_SEG_CLASS: 2
  NUMBER_IMAGE: -1
  NC: 1  # Number of classes for detection
  NAMES: ['car']

AUGMENTATION:
  FLIP: true
  SCALE_FACTOR: 0.25
  ROT_FACTOR: 10
  TRANSLATE: 0.1
  SHEAR: 0.0
  COLOR_RGB: false
  HSV_H: 0.015
  HSV_S: 0.7
  HSV_V: 0.4
  MOSAIC_RATE: 0.0
  MIXUP_RATE: 0.15
"""
    
    yaml_path = Path('/workspace/YOLOPX/yolop_series/cfgs/data/dummy_bdd100k.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_config)
    
    print(f"✅ YAML配置文件创建: {yaml_path}")
    
    return base_path

if __name__ == '__main__':
    # 创建小型测试数据集
    create_dummy_bdd100k(num_images=20)