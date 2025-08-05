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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from utils.conflict_detector import ConflictDetector
from models.model_factory import ModelFactory
from data.dataset import YOLOPDataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOP series models')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Path to model-specific config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                       help='Modify config options using the command-line')
    return parser.parse_args()

def create_optimizer(model, cfg):
    """Create optimizer based on configuration."""
    opt_type = cfg.TRAIN.OPTIMIZER.lower()
    lr = cfg.TRAIN.LR0
    weight_decay = cfg.TRAIN.WEIGHT_DECAY if hasattr(cfg.TRAIN, 'WEIGHT_DECAY') else cfg.TRAIN.WD
    
    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, 
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=weight_decay,
                            nesterov=cfg.TRAIN.NESTEROV)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
        
    return optimizer

def create_scheduler(optimizer, cfg, steps_per_epoch):
    """Create learning rate scheduler."""
    # Simple cosine annealing scheduler
    total_steps = cfg.TRAIN.END_EPOCH * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.TRAIN.LR0 * cfg.TRAIN.LRF)
    return scheduler

def compute_loss(outputs, targets, cfg):
    """Compute losses for all tasks.
    
    This is a simplified loss computation. 
    In practice, each model variant has its own loss implementation.
    """
    losses = {}
    
    # Detection loss (simplified)
    if 'detection' in outputs:
        # This would normally use the model-specific detection loss
        det_loss = torch.tensor(0.0, device=outputs['detection'].device)
        losses['detection'] = det_loss * cfg.LOSS.OBJ_GAIN
    
    # Driving area segmentation loss
    if 'da_seg' in outputs and 'da_seg_masks' in targets:
        da_seg_loss = nn.CrossEntropyLoss()(
            outputs['da_seg'], targets['da_seg_masks'])
        losses['da_segmentation'] = da_seg_loss * cfg.LOSS.DA_SEG_GAIN
        
    # Lane line segmentation loss  
    if 'll_seg' in outputs and 'll_seg_masks' in targets:
        ll_seg_loss = nn.CrossEntropyLoss()(
            outputs['ll_seg'], targets['ll_seg_masks'])
        losses['ll_segmentation'] = ll_seg_loss * cfg.LOSS.LL_SEG_GAIN
        
    return losses

def train_one_epoch(model, train_loader, optimizer, scheduler, 
                   conflict_detector, cfg, epoch, device):
    """Train for one epoch."""
    model.train()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.TRAIN.END_EPOCH}')
    
    # Metrics
    total_loss = 0
    task_losses = {task: 0 for task in ['detection', 'da_segmentation', 'll_segmentation']}
    
    for i, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].to(device)
        targets = {k: v.to(device) if torch.is_tensor(v) else v 
                  for k, v in batch.items()}
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        losses = compute_loss(outputs, targets, cfg)
        
        # Detect task conflicts
        iteration = epoch * len(train_loader) + i
        conflict_info = conflict_detector.detect_conflicts(model, losses, iteration)
        
        # Total loss
        total_batch_loss = sum(losses.values())
        
        # Backward pass
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += total_batch_loss.item()
        for task, loss in losses.items():
            task_losses[task] += loss.item()
            
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
    # Return epoch metrics
    n_batches = len(train_loader)
    metrics = {
        'total_loss': total_loss / n_batches,
        **{f'{task}_loss': loss / n_batches for task, loss in task_losses.items()}
    }
    
    return metrics

def validate(model, val_loader, cfg, device):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    task_losses = {task: 0 for task in ['detection', 'da_segmentation', 'll_segmentation']}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['images'].to(device)
            targets = {k: v.to(device) if torch.is_tensor(v) else v 
                      for k, v in batch.items()}
            
            outputs = model(images)
            losses = compute_loss(outputs, targets, cfg)
            
            total_loss += sum(losses.values()).item()
            for task, loss in losses.items():
                task_losses[task] += loss.item()
                
    n_batches = len(val_loader)
    metrics = {
        'val_total_loss': total_loss / n_batches,
        **{f'val_{task}_loss': loss / n_batches for task, loss in task_losses.items()}
    }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, cfg, metrics, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': cfg.cfg,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    cfg = Config(args.config)
    if args.model_config:
        model_cfg = Config(args.model_config)
        cfg.cfg.update(model_cfg.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
        
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {cfg.MODEL.TYPE}")
    try:
        model = ModelFactory.create_model(cfg)
        model = model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Note: Full model integration requires proper setup of original codebases")
        print("Using a placeholder model for demonstration")
        # Create a simple placeholder model for demonstration
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)  # Simplified output
        ).to(device)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = YOLOPDataset(cfg, split='train')
    val_dataset = YOLOPDataset(cfg, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.DEVICE.WORKERS,
        pin_memory=cfg.DEVICE.PIN_MEMORY,
        collate_fn=YOLOPDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DEVICE.WORKERS,
        pin_memory=cfg.DEVICE.PIN_MEMORY,
        collate_fn=YOLOPDataset.collate_fn
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    
    # Create conflict detector
    conflict_detector = ConflictDetector(cfg)
    
    # Create output directory
    os.makedirs(cfg.LOG.DIR, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    # Main training loop
    for epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            conflict_detector, cfg, epoch, device)
        
        # Validate
        if epoch % cfg.VAL.VAL_FREQ == 0:
            val_metrics = validate(model, val_loader, cfg, device)
            train_metrics.update(val_metrics)
            
        # Print metrics
        print(f"\nEpoch {epoch} metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
            
        # Save checkpoint
        if epoch % cfg.LOG.SAVE_FREQ == 0:
            save_path = os.path.join(cfg.LOG.DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, 
                          train_metrics, save_path)
            
    # Print final conflict summary
    if cfg.CONFLICT.ENABLED:
        print("\nFinal Task Conflict Summary:")
        conflict_summary = conflict_detector.get_conflict_summary()
        for key, value in conflict_summary.items():
            print(f"  {key}: {value:.4f}")
            
    print("Training completed!")

if __name__ == '__main__':
    main()