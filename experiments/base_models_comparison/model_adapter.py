#!/usr/bin/env python3
"""
统一的模型接口适配器
处理不同YOLOP模型的配置差异，保持原有模型的完整性
"""

import yaml
from pathlib import Path
from easydict import EasyDict as edict
import sys

# 添加yolop_series到Python路径
YOLOP_ROOT = Path(__file__).resolve().parent.parent.parent / 'yolop_series'
sys.path.insert(0, str(YOLOP_ROOT))


class ModelAdapter:
    """统一的模型适配器，处理不同模型的配置差异"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.yolop_root = YOLOP_ROOT
        
        # 模型特定的默认参数
        self.model_defaults = {
            'yolopv1': {
                'mosaic_rate': 0.0,
                'mixup_rate': 0.15,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
            },
            'yolopv2': {
                'mosaic_rate': 0.0,
                'mixup_rate': 0.15,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
            },
            'yolopv3': {
                'mosaic_rate': 0.0,
                'mixup_rate': 0.15,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
            },
            'yolopx': {
                'mosaic_rate': 0.0,
                'mixup_rate': 0.15,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
            }
        }
        
        # 模型配置文件映射
        self.model_configs = {
            'yolopv1': 'yolop_v1_official',
            'yolopv2': 'yolop',
            'yolopv3': 'yolop_v3_official',
            'yolopx': 'yolopx'
        }
    
    def load_configs(self):
        """加载并合并配置文件"""
        # 加载基础配置
        with open(self.yolop_root / 'cfgs/data/bdd100k_single_class.yaml', 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        with open(self.yolop_root / 'cfgs/train_default.yaml', 'r') as f:
            train_cfg = yaml.safe_load(f)
        
        model_cfg_name = self.model_configs[self.model_name]
        with open(self.yolop_root / f'cfgs/models/{model_cfg_name}.yaml', 'r') as f:
            model_cfg = yaml.safe_load(f)
        
        # 创建统一的配置对象
        cfg = edict()
        
        # 基础配置
        cfg.DATASET = edict(data_cfg['DATASET'])
        cfg.AUGMENTATION = edict(data_cfg['AUGMENTATION'])
        cfg.TRAIN = edict(train_cfg['TRAIN'])
        cfg.TEST = edict(train_cfg['TEST'])
        cfg.LOSS = edict(train_cfg['LOSS'])
        cfg.MODEL = edict(train_cfg['MODEL'])
        
        # 通用配置
        cfg.GPUS = train_cfg['GPUS']
        cfg.WORKERS = train_cfg['WORKERS']
        cfg.PIN_MEMORY = train_cfg['PIN_MEMORY']
        cfg.PRINT_FREQ = train_cfg['PRINT_FREQ']
        cfg.LOG_DIR = train_cfg['LOG_DIR']
        cfg.DEBUG = train_cfg['DEBUG']
        cfg.CUDNN = edict(train_cfg['CUDNN'])
        
        # 添加模型特定的默认参数
        defaults = self.model_defaults[self.model_name]
        
        # 根级别参数（AutoDriveDataset需要）
        cfg.mosaic_rate = defaults['mosaic_rate']
        cfg.mixup_rate = defaults['mixup_rate']
        cfg.num_seg_class = cfg.DATASET.NUM_SEG_CLASS
        
        # 确保DATASET包含所有必要的增强参数
        if not hasattr(cfg.DATASET, 'HSV_H'):
            cfg.DATASET.HSV_H = defaults['hsv_h']
            cfg.DATASET.HSV_S = defaults['hsv_s']
            cfg.DATASET.HSV_V = defaults['hsv_v']
        
        # 添加模型配置信息
        cfg.model_cfg_path = str(self.yolop_root / f'cfgs/models/{model_cfg_name}.yaml')
        cfg.model_name = self.model_name
        cfg.model_config = model_cfg
        
        # 确保MODEL.NC被设置
        cfg.MODEL.NC = cfg.DATASET.NC
        
        return cfg
    
    def get_dataset(self, cfg, is_train=True):
        """根据模型类型返回适配的数据集"""
        from data.bdd import BddDataset
        import torch
        import numpy as np
        
        # 创建transform函数，将numpy数组转换为张量
        def numpy_to_tensor(img):
            """将numpy图像转换为PyTorch张量"""
            if isinstance(img, np.ndarray):
                # HWC -> CHW
                img = img.transpose(2, 0, 1)
                # 归一化到[0, 1]
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img)
            return img
        
        # 创建数据集
        dataset = BddDataset(
            cfg=cfg,
            is_train=is_train,
            inputsize=cfg.DATASET.IMAGE_SIZE,
            transform=numpy_to_tensor
        )
        
        return dataset
    
    def get_model(self, cfg):
        """根据模型类型返回模型实例"""
        from models.builder import get_net_from_yaml
        
        model = get_net_from_yaml(cfg.model_cfg_path)
        return model
    
    def get_loss(self, cfg, device, model):
        """根据模型类型返回损失函数"""
        from core.loss import get_loss
        
        # 注意：get_loss需要cfg中的MODEL.NC
        if not hasattr(cfg.MODEL, 'NC'):
            cfg.MODEL.NC = cfg.DATASET.NC
        
        criterion = get_loss(cfg, device, model)
        return criterion
    
    def get_optimizer(self, cfg, model):
        """根据模型类型返回优化器"""
        import torch.nn as nn
        import torch.optim as optim
        
        # 分组参数
        pg0, pg1, pg2 = [], [], []
        
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        
        lr = cfg.TRAIN.LR0
        momentum = cfg.TRAIN.MOMENTUM
        weight_decay = cfg.TRAIN.WD
        
        if cfg.TRAIN.OPTIMIZER == 'adamw':
            optimizer = optim.AdamW(pg0, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(pg0, lr=lr, betas=(momentum, 0.999))
        else:
            optimizer = optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
        
        optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})
        optimizer.add_param_group({'params': pg2})
        
        # 保存初始学习率
        for g in optimizer.param_groups:
            g['initial_lr'] = g['lr']
        
        return optimizer