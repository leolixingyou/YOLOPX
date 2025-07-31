import argparse
import os
import sys
import time
from pathlib import Path
import yaml
from easydict import EasyDict as edict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import get_optimizer
from data.autodrive_dataset import AutoDriveDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from models.builder import get_net_from_yaml

def main():
    # --- 1. Parse Arguments and Load Configs ---
    args = parse_args()
    
    with open(args.model_cfg, 'r') as f:
        model_cfg = edict(yaml.safe_load(f))
    with open(args.data_cfg, 'r') as f:
        data_cfg = edict(yaml.safe_load(f))
    with open(args.train_cfg, 'r') as f:
        train_cfg = edict(yaml.safe_load(f))

    # Merge configs into a single master config object
    cfg = edict({**train_cfg, **data_cfg, **model_cfg})
    
    # --- 2. Setup Environment ---
    logger = ConsoleLogger()
    logger.info("Configurations loaded successfully.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # --- 3. Data Loading ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    train_dataset = AutoDriveDataset(cfg=cfg, is_train=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
        shuffle=cfg.TRAIN.SHUFFLE, 
        num_workers=cfg.WORKERS, 
        pin_memory=cfg.PIN_MEMORY, 
        collate_fn=AutoDriveDataset.collate_fn
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} images.")

    # --- 4. Model, Loss, and Optimizer Setup ---
    model = get_net_from_yaml(args.model_cfg).to(device)
    criterion = get_loss(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type != 'cpu'))
    
    logger.info(f"Model '{cfg.model_family}' created successfully.")

    # --- 5. Training Loop ---
    logger.info("Starting training...")
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        model.train()
        for i, (input_data, target, _, _) in enumerate(train_loader):
            input_data = input_data.to(device, non_blocking=True)
            # Ensure target is a list of tensors on the correct device
            target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type != 'cpu')):
                outputs = model(input_data)
                total_loss, _ = criterion(outputs, target, model=model)

            if total_loss == 0 or not torch.isfinite(total_loss):
                logger.warning(f"Skipping step with non-finite/zero loss: {total_loss.item()}")
                continue

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % cfg.PRINT_FREQ == 0:
                logger.info(f"Epoch: {epoch+1}/{cfg.TRAIN.END_EPOCH}, Iter: {i}/{len(train_loader)}, Loss: {total_loss.item():.4f}")

    logger.info("Training finished.")

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOPX-V2 Refactored Trainer')
    parser.add_argument('--model_cfg', required=True, help='Path to model config yaml', type=str)
    parser.add_argument('--data_cfg', required=True, help='Path to data config yaml', type=str)
    parser.add_argument('--train_cfg', required=True, help='Path to training config yaml', type=str)
    return parser.parse_args()

class ConsoleLogger:
    def info(self, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] INFO: {msg}")
    def warning(self, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] WARNING: {msg}")

if __name__ == '__main__':
    main()