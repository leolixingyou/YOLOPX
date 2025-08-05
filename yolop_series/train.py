"""Unified training script for YOLOP series models."""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import get_optimizer
from data.autodrive_dataset import AutoDriveDataset
from data.bdd import BddDataset
from torch.utils.data import DataLoader
from core.loss import get_loss
from models.builder import get_net_from_yaml


def calculate_task_conflict(gradients):
    """计算任务冲突强度(TCI)"""
    task_names = list(gradients.keys())
    if len(task_names) < 2:
        return 0.0
    
    conflicts = []
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            task1_grads = gradients[task_names[i]]
            task2_grads = gradients[task_names[j]]
            
            conflict = 0.0
            count = 0
            
            for g1, g2 in zip(task1_grads, task2_grads):
                if g1 is not None and g2 is not None:
                    g1_flat = g1.view(-1)
                    g2_flat = g2.view(-1)
                    
                    if g1_flat.shape == g2_flat.shape and g1_flat.numel() > 0:
                        dot_product = torch.dot(g1_flat, g2_flat)
                        norm1 = torch.norm(g1_flat)
                        norm2 = torch.norm(g2_flat)
                        
                        if norm1 > 0 and norm2 > 0:
                            cos_sim = dot_product / (norm1 * norm2)
                            if cos_sim < 0:
                                conflict += abs(cos_sim.item())
                                count += 1
            
            if count > 0:
                conflicts.append(conflict / count)
    
    return np.mean(conflicts) if conflicts else 0.0

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
    
    # Add necessary attributes for BddDataset
    if args.use_bdd:
        # Ensure BddDataset has all required attributes
        if 'AUGMENTATION' in data_cfg:
            cfg.mosaic_rate = data_cfg['AUGMENTATION'].get('MOSAIC_RATE', 0.0)
            cfg.mixup_rate = data_cfg['AUGMENTATION'].get('MIXUP_RATE', 0.0)
            cfg.DATASET.HSV_H = data_cfg['AUGMENTATION'].get('HSV_H', 0.1)
            cfg.DATASET.HSV_S = data_cfg['AUGMENTATION'].get('HSV_S', 0.7)
            cfg.DATASET.HSV_V = data_cfg['AUGMENTATION'].get('HSV_V', 0.4)
        cfg.num_seg_class = cfg.DATASET.get('NUM_SEG_CLASS', 2)
        cfg.MODEL.NC = cfg.DATASET.get('NC', 1)
    
    # --- 2. Setup Environment ---
    logger = ConsoleLogger()
    logger.info("Configurations loaded successfully.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # --- 3. Data Loading ---
    if args.use_bdd:
        # BddDataset specific transform
        def numpy_to_tensor(img):
            if isinstance(img, np.ndarray):
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img)
            return img
        
        train_dataset = BddDataset(cfg=cfg, is_train=True, inputsize=cfg.DATASET.IMAGE_SIZE, transform=numpy_to_tensor)
        collate_fn = train_dataset.collate_fn
    else:
        # AutoDriveDataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        train_dataset = AutoDriveDataset(cfg=cfg, is_train=True, transform=transform)
        collate_fn = AutoDriveDataset.collate_fn
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, 
        shuffle=cfg.TRAIN.SHUFFLE, 
        num_workers=cfg.WORKERS, 
        pin_memory=cfg.PIN_MEMORY, 
        collate_fn=collate_fn
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} images using {'BddDataset' if args.use_bdd else 'AutoDriveDataset'}.")

    # --- 4. Model, Loss, and Optimizer Setup ---
    model = get_net_from_yaml(args.model_cfg).to(device)
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type != 'cpu'))
    
    logger.info(f"Model created successfully.")

    # --- 5. Training Loop ---
    logger.info("Starting training...")
    
    # TCI tracking
    tci_history = [] if args.enable_tci else None
    
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        model.train()
        epoch_tci = [] if args.enable_tci else None
        
        for i, (input_data, target, _, _) in enumerate(train_loader):
            input_data = input_data.to(device, non_blocking=True)
            # Ensure target is a list of tensors on the correct device
            target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type != 'cpu')):
                outputs = model(input_data)
                total_loss, head_losses = criterion(outputs, target, model=model)

            if total_loss == 0 or not torch.isfinite(total_loss):
                logger.warning(f"Skipping step with non-finite/zero loss: {total_loss.item()}")
                continue

            # Task conflict detection
            if args.enable_tci and i % args.tci_freq == 0 and len(head_losses) >= 3:
                task_gradients = {}
                
                # Get shared parameters (assuming backbone is shared)
                shared_params = []
                for name, param in model.named_parameters():
                    if 'backbone' in name or 'features' in name or 'base' in name:
                        shared_params.append(param)
                
                if shared_params:
                    # Calculate gradients for each task
                    det_loss, da_loss, ll_loss = head_losses[0], head_losses[1], head_losses[2]
                    
                    if torch.isfinite(det_loss) and det_loss > 0:
                        grads = torch.autograd.grad(det_loss, shared_params, retain_graph=True, allow_unused=True)
                        task_gradients['detection'] = [g.detach().clone() if g is not None else None for g in grads]
                    
                    if torch.isfinite(da_loss) and da_loss > 0:
                        grads = torch.autograd.grad(da_loss, shared_params, retain_graph=True, allow_unused=True)
                        task_gradients['da_seg'] = [g.detach().clone() if g is not None else None for g in grads]
                    
                    if torch.isfinite(ll_loss) and ll_loss > 0:
                        grads = torch.autograd.grad(ll_loss, shared_params, retain_graph=True, allow_unused=True)
                        task_gradients['ll_seg'] = [g.detach().clone() if g is not None else None for g in grads]
                    
                    if len(task_gradients) >= 2:
                        tci = calculate_task_conflict(task_gradients)
                        epoch_tci.append(tci)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % cfg.PRINT_FREQ == 0:
                log_msg = f"Epoch: {epoch+1}/{cfg.TRAIN.END_EPOCH}, Iter: {i}/{len(train_loader)}, Loss: {total_loss.item():.4f}"
                if args.enable_tci and epoch_tci:
                    log_msg += f", TCI: {np.mean(epoch_tci):.4f}"
                logger.info(log_msg)
        
        # End of epoch TCI summary
        if args.enable_tci and epoch_tci:
            avg_tci = np.mean(epoch_tci)
            tci_history.append(avg_tci)
            logger.info(f"Epoch {epoch+1} - Average TCI: {avg_tci:.4f}")

    logger.info("Training finished.")
    
    # Save TCI results if requested
    if args.enable_tci and args.save_tci and tci_history:
        tci_results = {
            'model': os.path.basename(args.model_cfg),
            'final_tci': float(np.mean(tci_history)),
            'tci_history': [float(x) for x in tci_history],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(args.save_tci, 'w') as f:
            json.dump(tci_results, f, indent=2)
        logger.info(f"TCI results saved to {args.save_tci}")

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOPX-V2 Refactored Trainer')
    parser.add_argument('--model_cfg', required=True, help='Path to model config yaml', type=str)
    parser.add_argument('--data_cfg', required=True, help='Path to data config yaml', type=str)
    parser.add_argument('--train_cfg', required=True, help='Path to training config yaml', type=str)
    parser.add_argument('--enable_tci', action='store_true', help='Enable task conflict intensity calculation')
    parser.add_argument('--tci_freq', type=int, default=10, help='Calculate TCI every N batches')
    parser.add_argument('--use_bdd', action='store_true', help='Use BddDataset instead of AutoDriveDataset')
    parser.add_argument('--save_tci', type=str, default=None, help='Path to save TCI results')
    return parser.parse_args()

class ConsoleLogger:
    def info(self, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] INFO: {msg}")
    def warning(self, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] WARNING: {msg}")

if __name__ == '__main__':
    main()