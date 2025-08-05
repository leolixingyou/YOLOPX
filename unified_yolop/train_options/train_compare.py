#!/usr/bin/env python3
"""Batch training script for comparing multiple YOLOP models."""

import argparse
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import logging
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.project_manager import ProjectManager

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and compare multiple YOLOP models')
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx', 'all'],
                      default=['all'],
                      help='Models to train')
    
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
    parser.add_argument('--parallel', action='store_true',
                      help='Train models in parallel (requires multiple GPUs)')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                      help='Use wandb logging')
    
    # Dataset size control
    parser.add_argument('--train-samples', type=int, default=None,
                      help='Number of training samples to use (None for all)')
    parser.add_argument('--val-samples', type=int, default=None,
                      help='Number of validation samples to use (None for all)')
    
    # Project management
    parser.add_argument('--project-manager', type=str, default=None,
                      help='Existing project manager directory')
    
    return parser.parse_args()

def train_model(model_name: str, args: argparse.Namespace, 
                project_manager: ProjectManager, gpu_id: int = 0):
    """Train a single model.
    
    Args:
        model_name: Name of the model to train
        args: Training arguments
        project_manager: Project manager instance
        gpu_id: GPU ID for parallel training
    """
    # Prepare command
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), 'train_unified.py'),
        '--model', model_name,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--workers', str(args.workers),
        '--device', f'cuda:{gpu_id}' if 'cuda' in args.device else 'cpu',
        '--project-manager', str(project_manager.project_dir)
    ]
    
    if args.wandb:
        cmd.append('--wandb')
        
    if args.train_samples:
        cmd.extend(['--train-samples', str(args.train_samples)])
        
    if args.val_samples:
        cmd.extend(['--val-samples', str(args.val_samples)])
        
    # Log file for this model
    log_file = project_manager.get_model_dir(model_name) / 'train_output.log'
    
    print(f"\nStarting training for {model_name}...")
    print(f"Log file: {log_file}")
    
    # Run training
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for completion
        return_code = process.wait()
        
    elapsed_time = time.time() - start_time
    
    if return_code == 0:
        print(f"✓ {model_name} training completed in {elapsed_time/60:.1f} minutes")
    else:
        print(f"✗ {model_name} training failed with code {return_code}")
        print(f"  Check log file: {log_file}")
        
    return return_code, elapsed_time

def main():
    """Main function."""
    args = parse_args()
    
    # Determine models to train
    if 'all' in args.models:
        models = ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx']
    else:
        models = args.models
        
    print(f"Training models: {', '.join(models)}")
    
    # Create or load project manager
    if args.project_manager:
        # Use existing project directory
        project_dir = Path(args.project_manager)
        
        # Load existing metadata
        metadata_path = project_dir / "project_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'models': models,
                'project_name': project_dir.name,
                'created_at': datetime.now().isoformat(),
                'status': 'in_progress'
            }
            
        # Create project manager instance without creating directories
        project_manager = ProjectManager([], str(project_dir.parent), create_directories=False)
        project_manager.project_dir = project_dir
        project_manager.project_name = project_dir.name
        project_manager.metadata = metadata
        project_manager.models = models
        
        # Create model directories if they don't exist
        project_manager.model_dirs = {}
        for model in models:
            model_dir = project_manager.project_dir / model
            model_dir.mkdir(exist_ok=True)
            project_manager.model_dirs[model] = model_dir
            
        # Update metadata with all models
        project_manager.metadata['models'] = models
        project_manager.save_metadata()
    else:
        # Check if we're being called from run_experiment.py via environment variable
        if 'YOLOP_PROJECT_DIR' in os.environ:
            # We're being called from run_experiment.py, don't create new project
            project_dir = Path(os.environ['YOLOP_PROJECT_DIR'])
            
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
            project_manager.models = models
            
            # Create model directories if they don't exist
            project_manager.model_dirs = {}
            for model in models:
                model_dir = project_manager.project_dir / model
                model_dir.mkdir(exist_ok=True)
                project_manager.model_dirs[model] = model_dir
        else:
            # Only create new project if running standalone
            project_manager = ProjectManager(models, main_script=__file__)
    
    print(f"\nProject directory: {project_manager.project_dir}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(project_manager.project_dir / 'experiment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {project_manager.project_name}")
    logger.info(f"Models: {models}")
    logger.info(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    # Train models
    experiment_start = time.time()
    results = {}
    
    if args.parallel and len(models) > 1:
        # Parallel training (requires multiple GPUs)
        import multiprocessing
        from functools import partial
        import torch
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Parallel training on {num_gpus} GPUs")
        
        # Create pool
        pool = multiprocessing.Pool(min(len(models), num_gpus))
        
        # Train in parallel
        train_func = partial(train_model, args=args, project_manager=project_manager)
        gpu_assignments = [(model, i % num_gpus) for i, model in enumerate(models)]
        
        results_list = pool.starmap(train_func, gpu_assignments)
        pool.close()
        pool.join()
        
        # Collect results
        for model, (return_code, elapsed_time) in zip(models, results_list):
            results[model] = {
                'success': return_code == 0,
                'training_time': elapsed_time
            }
    else:
        # Sequential training
        for model in models:
            return_code, elapsed_time = train_model(model, args, project_manager)
            results[model] = {
                'success': return_code == 0,
                'training_time': elapsed_time
            }
            
    experiment_time = time.time() - experiment_start
    
    # Update project metadata
    project_manager.update_metadata({
        'status': 'completed',
        'experiment_time': experiment_time,
        'training_results': results
    })
    
    # Generate comparison report
    logger.info("\nGenerating comparison report...")
    try:
        project_manager.generate_comparison_report()
        logger.info(f"Comparison report saved to: {project_manager.project_dir}/comparison_report.md")
    except Exception as e:
        logger.error(f"Failed to generate comparison report: {e}")
        
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Project: {project_manager.project_name}")
    logger.info(f"Total time: {experiment_time/3600:.2f} hours")
    logger.info("\nModel Results:")
    
    for model, result in results.items():
        status = "✓" if result['success'] else "✗"
        time_str = f"{result['training_time']/60:.1f} min" if result['success'] else "FAILED"
        logger.info(f"  {status} {model}: {time_str}")
        
    logger.info(f"\nResults directory: {project_manager.project_dir}")
    
    # Create a simple run script for viewing results
    run_script = project_manager.project_dir / "view_results.sh"
    with open(run_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# View experiment results for {project_manager.project_name}\n\n")
        f.write("echo 'Comparison Report:'\n")
        f.write("echo '=================='\n")
        f.write("cat comparison_report.md\n\n")
        f.write("echo -e '\\n\\nTensorBoard logs:'\n")
        f.write("echo '=================='\n")
        f.write("echo 'Run: tensorboard --logdir=. --port=6006'\n")
        
    run_script.chmod(0o755)
    
    print(f"\nTo view results:")
    print(f"  cd {project_manager.project_dir}")
    print(f"  ./view_results.sh")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={project_manager.project_dir}")

if __name__ == '__main__':
    main()