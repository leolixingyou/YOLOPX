#!/usr/bin/env python3
"""Unified training script for YOLOP series models."""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import cv2
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config
from utils.conflict_pcgrad import TaskConflictDetectorPCGrad as TaskConflictDetector
from utils.visualization_dual import save_dual_visualization
from models.factory import ModelFactory
from data.dataset import UnifiedYOLOPDataset
from core.loss import UnifiedLoss
from core.metrics import Metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOP series models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='yolopx',
                       choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                       help='Model to train')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of data loading workers')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (e.g., cuda:0, cpu)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Experiment name
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name for logging')
    
    return parser.parse_args()

def create_data_loaders(cfg):
    """Create data loaders for training and validation."""
    # Create datasets
    train_dataset = UnifiedYOLOPDataset(cfg, split='train')
    val_dataset = UnifiedYOLOPDataset(cfg, split='val')
    
    # Get number of workers (use 0 to avoid multiprocessing issues)
    num_workers = cfg.get('TRAIN.WORKERS', 0)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('TRAIN.BATCH_SIZE', 16),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        collate_fn=UnifiedYOLOPDataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get('VAL.BATCH_SIZE', 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        collate_fn=UnifiedYOLOPDataset.collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader

def create_optimizer(model, cfg):
    """Create optimizer based on configuration."""
    opt_type = cfg.get('TRAIN.OPTIMIZER', 'adamw').lower()
    lr = cfg.get('TRAIN.LR', 0.001)
    weight_decay = cfg.get('TRAIN.WEIGHT_DECAY', 0.0005)
    
    if opt_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get('TRAIN.MOMENTUM', 0.937),
            weight_decay=weight_decay,
            nesterov=True
        )
    elif opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
        
    return optimizer

def create_scheduler(optimizer, cfg, steps_per_epoch):
    """Create learning rate scheduler."""
    schedule_type = cfg.get('TRAIN.LR_SCHEDULE', 'cosine').lower()
    epochs = cfg.get('TRAIN.EPOCHS', 100)
    lr = cfg.get('TRAIN.LR', 0.001)
    
    if schedule_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * steps_per_epoch,
            eta_min=lr * 0.1
        )
    elif schedule_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30 * steps_per_epoch,
            gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: 1.0  # Constant LR
        )
        
    return scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                   conflict_detector, device, epoch, cfg, writer=None, wandb_run=None):
    """Train for one epoch."""
    model.train()
    
    # Initialize metrics
    epoch_losses = {'total': 0, 'detection': 0, 'da_seg': 0, 'll_seg': 0}
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for i, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].to(device)
        targets = {
            'det_labels': batch['det_labels'].to(device),
            'da_seg_masks': batch['da_seg_masks'].to(device),
            'll_seg_masks': batch['ll_seg_masks'].to(device)
        }
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        losses = criterion(outputs, targets)
        
        # Detect conflicts
        conflict_metrics = conflict_detector.detect_conflicts(model, losses)
        
        # Get total loss (with conflict resolution)
        total_loss = conflict_detector.resolve_conflicts(losses)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        epoch_losses['total'] += total_loss.item()
        for task in ['detection', 'da_seg', 'll_seg']:
            if task in losses:
                epoch_losses[task] += losses[task].item()
                
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # Log to tensorboard
        if writer and i % cfg.get('LOG.LOG_INTERVAL', 20) == 0:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('train/lr', current_lr, global_step)
            
            # Log individual losses
            for task, loss in losses.items():
                if loss is not None:
                    writer.add_scalar(f'train/{task}_loss', loss.item(), global_step)
                    
            # Log conflict metrics
            for key, value in conflict_metrics.items():
                writer.add_scalar(f'conflict/{key}', value, global_step)
                
            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    'train/total_loss': total_loss.item(),
                    'train/lr': current_lr,
                    **{f'train/{k}_loss': v.item() for k, v in losses.items() if v is not None},
                    **{f'conflict/{k}': v for k, v in conflict_metrics.items()},
                    'epoch': epoch,
                    'iteration': global_step
                })
                
    # Return average losses
    n_batches = len(train_loader)
    avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
    
    return avg_losses

def validate(model, val_loader, criterion, device, cfg, writer=None, epoch=None, 
            wandb_run=None, save_dir=None):
    """Validate the model."""
    model.eval()
    
    # Initialize metrics
    metrics = Metrics(cfg)
    val_losses = {'total': 0, 'detection': 0, 'da_seg': 0, 'll_seg': 0}
    
    # Create save directory for validation images
    if save_dir and epoch is not None:
        img_save_dir = save_dir / f'val_epoch_{epoch}'
        img_save_dir.mkdir(exist_ok=True)
    else:
        img_save_dir = None
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # Move data to device
            images = batch['images'].to(device)
            targets = {
                'det_labels': batch['det_labels'].to(device),
                'da_seg_masks': batch['da_seg_masks'].to(device),
                'll_seg_masks': batch['ll_seg_masks'].to(device)
            }
            
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            losses = criterion(outputs, targets)
            total_loss = sum(losses.values())
            
            # Update metrics
            val_losses['total'] += total_loss.item()
            for task in ['detection', 'da_seg', 'll_seg']:
                if task in losses:
                    val_losses[task] += losses[task].item()
                    
            # Update task-specific metrics
            metrics.update(outputs, targets)
            
            # Save sample images with dual visualization
            if img_save_dir and i < 10:  # Save first 10 batches
                save_dual_visualization(images, outputs, targets, 
                                      img_save_dir, i)
            
    # Compute average losses
    n_batches = len(val_loader)
    avg_losses = {k: v / n_batches for k, v in val_losses.items()}
    
    # Get final metrics
    final_metrics = metrics.compute()
    
    # Log to tensorboard
    if writer and epoch is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f'val/{key}', value, epoch)
            
        for key, value in final_metrics.items():
            writer.add_scalar(f'metrics/{key}', value, epoch)
            
    # Log to wandb
    if wandb_run and epoch is not None:
        wandb_run.log({
            **{f'val/{k}': v for k, v in avg_losses.items()},
            **{f'metrics/{k}': v for k, v in final_metrics.items()},
            'epoch': epoch
        })
            
    return avg_losses, final_metrics

# Removed old save_validation_images function - now using save_dual_visualization

def save_checkpoint(state, save_path):
    """Save training checkpoint."""
    torch.save(state, save_path)
    print(f"Saved checkpoint to {save_path}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Get configuration
    cfg = get_config(args)
    
    # Set device
    device = torch.device(cfg.get('SYSTEM.DEVICE', 'cuda:0'))
    if not torch.cuda.is_available() and 'cuda' in str(device):
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # Set random seed
    seed = cfg.get('SYSTEM.SEED', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    exp_name = args.name or f"{cfg.get('MODEL.NAME')}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(cfg.get('LOG.DIR', './runs')) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    cfg.save(output_dir / 'config.yaml')
    
    # Create tensorboard writer
    writer = SummaryWriter(output_dir) if cfg.get('LOG.TENSORBOARD', True) else None
    
    # Initialize wandb
    wandb_run = None
    if WANDB_AVAILABLE and cfg.get('LOG.WANDB', False):
        wandb_run = wandb.init(
            project="unified-yolop",
            name=exp_name,
            config=cfg.config,
            dir=output_dir
        )
    
    # Create model
    print(f"\nCreating model: {cfg.get('MODEL.NAME')}")
    model = ModelFactory.create_model(cfg)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(cfg)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create loss function
    criterion = UnifiedLoss(cfg)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    
    # Create conflict detector
    conflict_detector = TaskConflictDetector(cfg)
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', 0)
        
    # Training loop
    print("\nStarting training...")
    epochs = cfg.get('TRAIN.EPOCHS', 100)
    
    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            conflict_detector, device, epoch, cfg, writer, wandb_run
        )
        
        # Print training statistics
        print(f"\nEpoch [{epoch+1}/{epochs}] Training:")
        for key, value in train_losses.items():
            print(f"  {key}_loss: {value:.4f}")
            
        # Validate
        if (epoch + 1) % cfg.get('VAL.INTERVAL', 1) == 0:
            val_losses, val_metrics = validate(
                model, val_loader, criterion, device, cfg, writer, epoch,
                wandb_run, output_dir
            )
            
            # Print validation statistics
            print(f"\nEpoch [{epoch+1}/{epochs}] Validation:")
            for key, value in val_losses.items():
                print(f"  {key}_loss: {value:.4f}")
            print("\nMetrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
                
            # Save best model
            current_metric = val_metrics.get('overall_score', 0)
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'config': cfg.config
                }, output_dir / 'best.pth')
                
        # Save checkpoint
        if (epoch + 1) % cfg.get('LOG.SAVE_INTERVAL', 5) == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'config': cfg.config
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
    # Print and save final conflict summary
    if cfg.get('CONFLICT.ENABLED', True):
        print("\nFinal Task Conflict Summary:")
        conflict_summary = conflict_detector.get_summary()
        for key, value in conflict_summary.items():
            print(f"  {key}: {value:.4f}")
            
        # Save conflict summary to file
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        conflict_results = {
            'model': cfg.get('MODEL.NAME'),
            'epochs': epochs,
            'conflict_summary': convert_to_python_types(conflict_summary),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_dir / 'conflict_results.json', 'w') as f:
            json.dump(conflict_results, f, indent=2)
            
        # Log to wandb
        if wandb_run:
            wandb_run.log({f'final/{k}': v for k, v in conflict_summary.items()})
            wandb_run.summary.update(conflict_summary)
            
    # Save final model
    save_checkpoint({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'config': cfg.config
    }, output_dir / 'final.pth')
    
    print(f"\nTraining completed! Results saved to: {output_dir}")
    
    if writer:
        writer.close()
        
    if wandb_run:
        wandb_run.finish()

if __name__ == '__main__':
    main()