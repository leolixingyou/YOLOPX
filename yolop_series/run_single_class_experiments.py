#!/usr/bin/env python3
"""
YOLOP Series Single-Class Detection Experiments
包含YOLOPv1、YOLOPv2、YOLOPv3和YOLOPX的单类检测实验
可以直接在VSCode中使用F5运行
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
from easydict import EasyDict as edict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# 确保项目根目录在Python路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import init_weights, split_optimizer
from data.bdd_dataset import BddDataset  # 使用具体的BddDataset
from torch.utils.data import DataLoader
from core.loss import MultiHeadLoss, get_loss
from models.builder import get_net_from_yaml
from utils.logger import create_logger
import wandb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class YOLOPExperimentRunner:
    """YOLOP系列实验运行器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 实验配置
        self.experiments = {
            'yolopv1_official': {
                'model_cfg': 'cfgs/models/yolop_v1_official.yaml',
                'description': 'YOLOPv1官方版本 - CSP-Darknet骨干网络'
            },
            'yolopv2_anchor_based': {
                'model_cfg': 'cfgs/models/yolopx_v2_anchor_based.yaml', 
                'description': 'YOLOPv2基于锚框的版本'
            },
            'yolopv2_anchor_free': {
                'model_cfg': 'cfgs/models/yolopx_v2_anchor_free.yaml',
                'description': 'YOLOPv2无锚框版本'  
            },
            'yolopv3_official': {
                'model_cfg': 'cfgs/models/yolop_v3_official.yaml',
                'description': 'YOLOPv3官方版本 - ELANNet骨干网络'
            },
            'yolopx': {
                'model_cfg': 'cfgs/models/yolopx.yaml',
                'description': 'YOLOPX - 无锚框多任务网络'
            }
        }
    
    def load_configs(self, model_name: str) -> edict:
        """加载配置文件"""
        model_cfg_path = self.experiments[model_name]['model_cfg']
        data_cfg_path = 'cfgs/data/bdd100k_single_class.yaml'
        train_cfg_path = 'cfgs/train_default.yaml'
        
        # 加载配置
        configs = {}
        for cfg_name, cfg_path in [
            ('model', model_cfg_path),
            ('data', data_cfg_path), 
            ('train', train_cfg_path)
        ]:
            full_path = PROJECT_ROOT / cfg_path
            if not full_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                configs[cfg_name] = edict(yaml.safe_load(f))
        
        # 合并配置
        cfg = edict({**configs['train'], **configs['data'], **configs['model']})
        
        # 设置模型名称
        cfg.model_name = model_name
        cfg.model_description = self.experiments[model_name]['description']
        
        return cfg
    
    def setup_data_loader(self, cfg: edict) -> DataLoader:
        """设置数据加载器"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        train_dataset = BddDataset(cfg=cfg, is_train=True, inputsize=cfg.DATASET.IMAGE_SIZE, transform=transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=train_dataset.collate_fn
        )
        
        logger.info(f"训练数据集加载完成，共有 {len(train_dataset)} 张图片")
        return train_loader
    
    def setup_model_and_optimizer(self, cfg: edict, model_cfg_path: str):
        """设置模型和优化器"""
        # 创建模型
        model = get_net_from_yaml(PROJECT_ROOT / model_cfg_path).to(self.device)
        
        # 创建损失函数
        criterion = get_loss(cfg, self.device)
        
        # 创建优化器
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, torch.nn.BatchNorm2d) or 'bn' in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)  # apply decay
        
        if cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(pg0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            optimizer = optim.AdamW(pg0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999), weight_decay=0.0)
        else:
            optimizer = optim.SGD(pg0, lr=cfg.TRAIN.LR0, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
            
        optimizer.add_param_group({'params': pg1, 'weight_decay': cfg.TRAIN.WD})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type != 'cpu'))
        
        return model, criterion, optimizer, scaler
    
    def run_training_epoch(self, model, criterion, optimizer, scaler, train_loader, cfg: edict):
        """运行一个训练周期"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i, (input_data, target, _, _) in enumerate(train_loader):
            # 数据移到设备
            input_data = input_data.to(self.device, non_blocking=True)
            target = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in target]
            
            # 前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device.type != 'cpu')):
                outputs = model(input_data)
                shapes = None  # shapes not used in this dataset
                loss, loss_dict = criterion(outputs, target, shapes, model, input_data)
            
            # 跳过无效损失
            if loss == 0 or not torch.isfinite(loss):
                logger.warning(f"跳过无效损失: {loss.item()}")
                continue
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印训练信息
            if i % cfg.PRINT_FREQ == 0:
                logger.info(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def run_experiment(self, model_name: str):
        """运行单个实验"""
        logger.info(f"开始运行实验: {model_name}")
        logger.info(f"描述: {self.experiments[model_name]['description']}")
        
        try:
            # 加载配置
            cfg = self.load_configs(model_name)
            logger.info("配置加载完成")
            
            # 设置CUDNN
            cudnn.benchmark = cfg.CUDNN.BENCHMARK
            cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
            cudnn.enabled = cfg.CUDNN.ENABLED
            
            # 设置数据加载器
            train_loader = self.setup_data_loader(cfg)
            
            # 设置模型和优化器
            model_cfg_path = self.experiments[model_name]['model_cfg']
            model, criterion, optimizer, scaler = self.setup_model_and_optimizer(cfg, model_cfg_path)
            logger.info(f"模型 '{model_name}' 创建成功")
            
            # 训练循环
            logger.info("开始训练...")
            for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
                start_time = time.time()
                avg_loss = self.run_training_epoch(model, criterion, optimizer, scaler, train_loader, cfg)
                epoch_time = time.time() - start_time
                
                logger.info(f"Epoch {epoch+1}/{cfg.TRAIN.END_EPOCH} 完成, "
                          f"平均损失: {avg_loss:.4f}, 用时: {epoch_time:.2f}s")
            
            logger.info(f"实验 '{model_name}' 完成!")
            return True
            
        except Exception as e:
            logger.error(f"实验 '{model_name}' 失败: {str(e)}")
            return False
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("开始运行YOLOP系列单类检测实验")
        logger.info("=" * 60)
        
        results = {}
        for model_name in self.experiments.keys():
            logger.info(f"\n{'='*20} {model_name.upper()} {'='*20}")
            success = self.run_experiment(model_name)
            results[model_name] = success
            logger.info(f"实验结果: {'成功' if success else '失败'}")
        
        # 总结
        logger.info("\n" + "="*60)
        logger.info("实验总结:")
        for model_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            logger.info(f"  {model_name}: {status}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOP系列单类检测实验')
    parser.add_argument('--model', type=str, choices=[
        'yolopv1_official', 'yolopv2_anchor_based', 'yolopv2_anchor_free', 
        'yolopv3_official', 'yolopx', 'all'
    ], default='all', help='选择要运行的模型实验')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = YOLOPExperimentRunner()
    
    if args.model == 'all':
        runner.run_all_experiments()
    else:
        runner.run_experiment(args.model)

if __name__ == '__main__':
    main()