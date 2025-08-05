#!/usr/bin/env python3
"""Unified training script with centralized project management."""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.conflict_pcgrad import TaskConflictDetectorPCGrad as TaskConflictDetector
from utils.visualization_dual import save_dual_visualization
from utils.project_manager import ProjectManager
from models.factory import ModelFactory
from data.dataset import UnifiedYOLOPDataset
from core.loss import UnifiedLoss
from core.metrics import Metrics
from mtl_strategies import create_mtl_strategy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOP models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                      help='Model to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--workers', type=int, default=0,
                      help='Number of data loading workers')
    
    # System
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Project management
    parser.add_argument('--project-manager', type=str, default=None,
                      help='Existing project manager directory')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                      help='Log interval')
    parser.add_argument('--save-interval', type=int, default=5,
                      help='Save interval')
    parser.add_argument('--wandb', action='store_true',
                      help='Use wandb logging')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                      help='Quick test with limited data')
    
    # Dataset size control
    parser.add_argument('--train-samples', type=int, default=None,
                      help='Number of training samples to use (None for all)')
    parser.add_argument('--val-samples', type=int, default=None,
                      help='Number of validation samples to use (None for all)')
    
    # MTL strategy
    parser.add_argument('--mtl-strategy', type=str, default='original',
                      choices=['original', 'pcgrad', 'cagrad', 'gradnorm'],
                      help='Multi-task learning strategy')
    parser.add_argument('--mtl-alpha', type=float, default=1.5,
                      help='Alpha parameter for GradNorm')
    parser.add_argument('--mtl-c', type=float, default=0.5,
                      help='C parameter for CAGrad')
    
    return parser.parse_args()

def setup_logging(log_path: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, 
                conflict_detector, epoch, logger, writer=None, mtl_strategy=None):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    task_losses = {'detection': 0, 'da_seg': 0, 'll_seg': 0}
    mtl_info_history = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        targets = {
            'det_labels': batch['det_labels'].to(device),
            'da_seg_masks': batch['da_seg_mask'].to(device),
            'll_seg_masks': batch['ll_seg_mask'].to(device)
        }
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        losses = criterion(outputs, targets)
        
        # Detect conflicts
        conflicts = conflict_detector.detect_conflicts(model, losses)
        
        # Backward pass with MTL strategy
        if mtl_strategy is not None:
            # Use MTL strategy to handle gradients
            mtl_strategy.zero_grad()
            mtl_info = mtl_strategy.backward(losses)
            mtl_info_history.append(mtl_info)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Optimizer step
            mtl_strategy.step()
            
            # Use the total loss from MTL strategy for logging
            total_loss_batch = torch.tensor(mtl_info.get('total_loss', sum(losses.values()).item()))
        else:
            # Original backward pass
            total_loss_batch = sum(losses.values())
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
        
        # Update statistics
        total_loss += total_loss_batch.item()
        for task, loss in losses.items():
            if task in task_losses:
                task_losses[task] += loss.item()
                
        # Update progress bar
        postfix_dict = {
            'loss': total_loss_batch.item(),
            'det': losses.get('detection', 0).item(),
            'da': losses.get('da_seg', 0).item(),
            'll': losses.get('ll_seg', 0).item()
        }
        
        # Add MTL info to progress bar if available
        if mtl_strategy and mtl_info_history:
            latest_info = mtl_info_history[-1]
            if 'task_weights' in latest_info:
                postfix_dict.update({
                    f'w_{k[:3]}': f"{v:.2f}" 
                    for k, v in latest_info['task_weights'].items()
                })
        
        pbar.set_postfix(postfix_dict)
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/total_loss', total_loss_batch.item(), global_step)
            for task, loss in losses.items():
                writer.add_scalar(f'train/{task}_loss', loss.item(), global_step)
            for name, value in conflicts.items():
                writer.add_scalar(f'conflicts/{name}', value, global_step)
                
    # Average losses
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_task_losses = {k: v / n_batches for k, v in task_losses.items()}
    
    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    for task, loss in avg_task_losses.items():
        logger.info(f"  {task}: {loss:.4f}")
    
    # Log MTL strategy info if available
    if mtl_info_history:
        # Average task weights over the epoch
        avg_weights = {}
        for info in mtl_info_history:
            if 'task_weights' in info:
                for task, weight in info['task_weights'].items():
                    if task not in avg_weights:
                        avg_weights[task] = 0
                    avg_weights[task] += weight
        
        if avg_weights:
            n_updates = len(mtl_info_history)
            avg_weights = {k: v/n_updates for k, v in avg_weights.items()}
            logger.info("  Average task weights:")
            for task, weight in avg_weights.items():
                logger.info(f"    {task}: {weight:.3f}")
        
    return avg_loss, avg_task_losses

def validate(model, val_loader, criterion, metrics, device, epoch, 
            vis_dir=None, logger=None, writer=None):
    """Validate the model."""
    model.eval()
    
    val_losses = {'total': 0, 'detection': 0, 'da_seg': 0, 'll_seg': 0}
    metrics.reset()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # Move data to device
            images = batch['image'].to(device)
            targets = {
                'det_labels': batch['det_labels'].to(device),
                'da_seg_masks': batch['da_seg_mask'].to(device),
                'll_seg_masks': batch['ll_seg_mask'].to(device)
            }
            
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            losses = criterion(outputs, targets)
            total_loss = sum(losses.values())
            
            # Update losses
            val_losses['total'] += total_loss.item()
            for task in ['detection', 'da_seg', 'll_seg']:
                if task in losses:
                    val_losses[task] += losses[task].item()
                    
            # Update metrics
            metrics.update(outputs, targets)
            
            # Save visualizations (only first 3 batches per epoch)
            if vis_dir and i < 3:
                save_dual_visualization(images, outputs, targets, vis_dir, i)
                
    # Average losses
    n_batches = len(val_loader)
    avg_losses = {k: v / n_batches for k, v in val_losses.items()}
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Log results
    if logger:
        logger.info(f"Validation - Loss: {avg_losses['total']:.4f}")
        for task, loss in avg_losses.items():
            if task != 'total':
                logger.info(f"  {task}: {loss:.4f}")
        logger.info("Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
    # Log to tensorboard
    if writer:
        for key, value in avg_losses.items():
            writer.add_scalar(f'val/{key}_loss', value, epoch)
        for key, value in final_metrics.items():
            writer.add_scalar(f'metrics/{key}', value, epoch)
            
    return avg_losses, final_metrics

def main():
    """Main training function."""
    args = parse_args()
    
    # Get configuration
    cfg = get_config(args)
    
    # Quick test mode settings
    if args.quick_test:
        cfg.set('DATASET.MAX_SAMPLES', 100)  # Use only 100 samples
        print("Quick test mode: Using only 100 samples")
    
    # Apply custom sample limits if specified
    if args.train_samples:
        cfg.set('DATASET.TRAIN_SAMPLES', args.train_samples)
        print(f"Using {args.train_samples} training samples")
    
    if args.val_samples:
        cfg.set('DATASET.VAL_SAMPLES', args.val_samples)
        print(f"Using {args.val_samples} validation samples")
    
    # Create or load project manager
    if args.project_manager:
        # Use existing project directory
        project_dir = Path(args.project_manager)
        
        # Import ProjectManager
        from utils.project_manager import ProjectManager
        
        # Load existing metadata
        metadata_path = project_dir / "project_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Create minimal metadata if missing
            metadata = {
                'models': [args.model],
                'project_name': project_dir.name,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'in_progress'
            }
            
        # Create project manager instance without creating directories
        # Use empty models list and create_directories=False
        project_manager = ProjectManager([], str(project_dir.parent), create_directories=False)
        project_manager.project_dir = project_dir
        project_manager.project_name = project_dir.name
        project_manager.metadata = metadata
        project_manager.models = metadata.get('models', [args.model])
        
        # Create model directories if they don't exist
        project_manager.model_dirs = {}
        for model in project_manager.models:
            model_dir = project_manager.project_dir / model
            model_dir.mkdir(exist_ok=True)
            project_manager.model_dirs[model] = model_dir
            
        # Update metadata to include current model if not already there
        if args.model not in project_manager.models:
            project_manager.models.append(args.model)
            project_manager.metadata['models'] = project_manager.models
            project_manager.save_metadata()
            
        # Ensure current model directory exists
        model_dir = project_manager.project_dir / args.model
        model_dir.mkdir(exist_ok=True)
        project_manager.model_dirs[args.model] = model_dir
            
    else:
        # Check if we're being called from run_experiment.py via environment variable
        if 'YOLOP_PROJECT_DIR' in os.environ:
            # We're being called from run_experiment.py, don't create new project
            project_dir = Path(os.environ['YOLOP_PROJECT_DIR'])
            
            # Import ProjectManager
            from utils.project_manager import ProjectManager
            
            # Load existing metadata
            metadata_path = project_dir / "project_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
            # Create project manager instance without creating directories
            project_manager = ProjectManager([], str(project_dir.parent), create_directories=False)
            project_manager.project_dir = project_dir
            project_manager.project_name = project_dir.name
            project_manager.metadata = metadata
            project_manager.models = metadata.get('models', [args.model])
            
            # Ensure current model directory exists
            model_dir = project_manager.project_dir / args.model
            model_dir.mkdir(exist_ok=True)
            project_manager.model_dirs[args.model] = model_dir
        else:
            # Only create new project if running standalone
            from utils.project_manager import ProjectManager
            project_manager = ProjectManager([args.model], main_script=__file__)
        
    # Get model directory
    model_dir = project_manager.get_model_dir(args.model)
    
    # Setup logging
    logger = setup_logging(model_dir / "training.log")
    logger.info(f"Project: {project_manager.project_name}")
    logger.info(f"Model: {args.model}")
    
    # Save configuration
    project_manager.save_model_config(args.model, cfg.config)
    
    # Set device
    device = torch.device(args.device)
    if not torch.cuda.is_available() and 'cuda' in str(device):
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    logger.info("Creating model...")
    model = ModelFactory.create_model(cfg)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = UnifiedYOLOPDataset(cfg, split='train')
    val_dataset = UnifiedYOLOPDataset(cfg, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create loss and metrics
    criterion = UnifiedLoss(cfg)
    metrics = Metrics(cfg)
    conflict_detector = TaskConflictDetector(cfg)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup tensorboard
    writer = SummaryWriter(project_manager.get_log_dir(args.model))
    
    # Setup wandb
    wandb_run = None
    wandb_url = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=f"yolop-unified",
            name=f"{args.model}_{project_manager.project_name}",
            config=cfg.config
        )
        wandb_url = wandb_run.get_url()
        logger.info(f"WandB run URL: {wandb_url}")
        
    # Resume from checkpoint
    start_epoch = 0
    best_score = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', 0)
        
    # Training loop
    logger.info("Starting training...")
    training_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            conflict_detector, epoch, logger, writer
        )
        
        # Validate
        # Only save visualizations every 10 epochs or on first/last epoch
        save_vis = (epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % 10 == 0)
        vis_dir = project_manager.get_visualization_dir(args.model, epoch) if save_vis else None
        val_losses, val_metrics = validate(
            model, val_loader, criterion, metrics, device,
            epoch, vis_dir, logger, writer
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = project_manager.get_checkpoint_dir(args.model) / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': cfg.config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        # Save best model
        current_score = val_metrics.get('overall_score', 0)
        if current_score > best_score:
            best_score = current_score
            best_path = project_manager.get_checkpoint_dir(args.model) / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': cfg.config
            }, best_path)
            logger.info(f"New best model saved: {best_path} (score: {best_score:.4f})")
            
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                'epoch': epoch,
                'train/loss': train_loss,
                **{f'train/{k}': v for k, v in train_task_losses.items()},
                **{f'val/{k}': v for k, v in val_losses.items()},
                **{f'metrics/{k}': v for k, v in val_metrics.items()},
                'lr': optimizer.param_groups[0]['lr']
            })
            
    # Training completed
    training_time = time.time() - training_start
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    # Get final conflict summary
    conflict_summary = conflict_detector.get_summary()
    
    # Helper function to convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save final results
    results = {
        'model': args.model,
        'epochs': args.epochs,
        'training_time': training_time,
        'final_metrics': convert_numpy_types(val_metrics),
        'conflict_summary': convert_numpy_types(conflict_summary),
        'best_score': float(best_score),
        'config': cfg.config,
        'wandb_url': wandb_url if wandb_url else None
    }
    
    project_manager.save_model_results(args.model, results)
    
    # Update project metadata
    project_manager.update_metadata({
        'status': 'completed',
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Cleanup
    writer.close()
    if wandb_run:
        wandb_run.finish()
        
    logger.info(f"Results saved to: {model_dir}")
    
    return project_manager

if __name__ == '__main__':
    project_manager = main()
    print(f"\nTraining completed. Project directory: {project_manager.project_dir}")