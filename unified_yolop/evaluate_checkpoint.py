#!/usr/bin/env python3
"""Evaluate a saved checkpoint and generate results.json"""

import torch
import json
import argparse
from pathlib import Path
import sys
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models.factory import create_model
from data.dataset import UnifiedYOLOPDataset
from torch.utils.data import DataLoader
from utils.loss import UnifiedYOLOPLoss
from utils.metrics import UnifiedMetrics
from utils.config import Config

def evaluate_checkpoint(checkpoint_path, model_name, project_dir):
    """Evaluate a checkpoint and save results."""
    
    # Load config
    cfg = Config(model_name)
    cfg.set('DATASET.VAL_SAMPLES', 2000)
    
    # Create model
    print(f"Creating model: {model_name}")
    model = create_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Create dataset
    print("Loading validation dataset...")
    val_dataset = UnifiedYOLOPDataset(cfg, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create loss and metrics
    criterion = UnifiedYOLOPLoss(cfg)
    metrics = UnifiedMetrics(cfg)
    
    # Evaluate
    print("Evaluating...")
    val_losses = {'total': 0, 'detection': 0, 'da_seg': 0, 'll_seg': 0}
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(val_loader)}")
                
            images = images.to(device)
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device)
                    
            # Forward pass
            outputs = model(images)
            
            # Compute losses
            losses, total_loss = criterion(outputs, targets)
            
            val_losses['total'] += total_loss.item()
            for task in ['detection', 'da_seg', 'll_seg']:
                if task in losses:
                    val_losses[task] += losses[task].item()
                    
            # Update metrics
            metrics.update(outputs, targets)
            
    # Average losses
    n_batches = len(val_loader)
    avg_losses = {k: v / n_batches for k, v in val_losses.items()}
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Prepare results
    results = {
        'model': model_name,
        'epochs': 35,  # From the log
        'training_time': 0,  # Unknown
        'final_metrics': final_metrics,
        'conflict_summary': {
            'overall_tci': 0,
            'detection_da_seg_tci_mean': 0,
            'detection_ll_seg_tci_mean': 0,
            'da_seg_ll_seg_tci_mean': 0
        },
        'best_score': final_metrics.get('overall_score', 0),
        'validation_losses': avg_losses
    }
    
    # Save results
    results_path = Path(project_dir) / model_name / 'results.json'
    print(f"Saving results to: {results_path}")
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
            
    results = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved. Overall score: {results['best_score']:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--project-dir', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.model, args.project_dir)

if __name__ == '__main__':
    main()