#!/usr/bin/env python3
"""Unified experiment runner for YOLOP models."""

import argparse
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.project_manager import ProjectManager
from utils.experiment_config import ExperimentConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run YOLOP experiments')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                      help='Path to YAML configuration file')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=False,
                      choices=['single', 'compare', 'all', 'generate-scripts'],
                      help='Experiment mode (can be specified in config)')
    
    # Model selection
    parser.add_argument('--model', type=str,
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                      help='Model to train (for single mode)')
    parser.add_argument('--models', nargs='+',
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                      help='Models to compare (for compare mode)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
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
    
    # Logging
    parser.add_argument('--wandb', action='store_true', default= True,
                      help='Use wandb logging')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true', default= True,
                      help='Quick test with limited data')
    
    # Dataset size control
    parser.add_argument('--train-samples', type=int, default=200,
                      help='Number of training samples to use (None for all)')
    parser.add_argument('--val-samples', type=int, default=100,
                      help='Number of validation samples to use (None for all)')
    
    return parser.parse_args()

def run_single_model(args):
    """Run training for a single model."""
    if not args.model:
        print("Error: No model specified for single mode")
        sys.exit(1)
        
    # Create project manager with main script name
    project_manager = ProjectManager(
        [args.model],
        main_script=__file__,
        use_shared_project=True
    )
    
    # Set environment variable for sub-processes
    os.environ['YOLOP_PROJECT_DIR'] = str(project_manager.project_dir)
    
    # Prepare command
    cmd = [
        sys.executable, 'train_options/train_unified.py',
        '--model', args.model,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', args.device,
        '--workers', str(args.workers),
        '--seed', str(args.seed),
        '--project-manager', str(project_manager.project_dir)
    ]
    
    if args.wandb:
        cmd.append('--wandb')
        
    if args.quick_test:
        cmd.append('--quick-test')
        
    if args.train_samples:
        cmd.extend(['--train-samples', str(args.train_samples)])
        
    if args.val_samples:
        cmd.extend(['--val-samples', str(args.val_samples)])
        
    print(f"Starting training for {args.model}...")
    print(f"Project directory: {project_manager.project_dir}")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Training completed in {elapsed_time/60:.1f} minutes")
        print(f"Results saved to: {project_manager.project_dir}")
    else:
        print(f"\n✗ Training failed with code {result.returncode}")
        
    return result.returncode

def run_compare_models(args):
    """Run comparison experiment for multiple models."""
    models = args.models if args.models else ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx']
    
    # Create project manager with main script name
    project_manager = ProjectManager(
        models,
        main_script=__file__,
        use_shared_project=True
    )
    
    # Set environment variable for sub-processes
    os.environ['YOLOP_PROJECT_DIR'] = str(project_manager.project_dir)
    
    # Run comparison script
    cmd = [
        sys.executable, 'train_options/train_compare.py',
        '--models'] + models + [
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--workers', str(args.workers),
        '--device', args.device,
        '--project-manager', str(project_manager.project_dir)
    ]
    
    if args.wandb:
        cmd.append('--wandb')
        
    if args.train_samples:
        cmd.extend(['--train-samples', str(args.train_samples)])
        
    if args.val_samples:
        cmd.extend(['--val-samples', str(args.val_samples)])
        
    print(f"Starting comparison experiment for: {', '.join(models)}")
    print(f"Project directory: {project_manager.project_dir}")
    
    result = subprocess.run(cmd)
    return result.returncode

def run_all_models(args):
    """Run all models comparison."""
    args.models = ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx']
    return run_compare_models(args)

def generate_scripts(args):
    """Generate shell scripts for different experiments."""
    scripts_dir = Path('scripts')
    scripts_dir.mkdir(exist_ok=True)
    
    # Common parameters
    common_params = f"--epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr}"
    
    # Generate individual model scripts
    for model in ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx']:
        script_path = scripts_dir / f'train_{model}.sh'
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Train {model.upper()}\n\n")
            f.write(f"python3 run_experiment.py --mode single --model {model} {common_params}\n")
        script_path.chmod(0o755)
        
    # Generate comparison script
    script_path = scripts_dir / 'train_compare_all.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Compare all YOLOP models\n\n")
        f.write(f"python3 run_experiment.py --mode all {common_params}\n")
    script_path.chmod(0o755)
    
    # Generate custom comparison script
    script_path = scripts_dir / 'train_compare_custom.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Compare specific YOLOP models\n")
        f.write("# Usage: ./train_compare_custom.sh yolop_v1 yolopx\n\n")
        f.write(f'python3 run_experiment.py --mode compare --models "$@" {common_params}\n')
    script_path.chmod(0o755)
    
    # Generate quick test script
    script_path = scripts_dir / 'quick_test.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Quick test with 1 epoch\n\n")
        f.write("python3 run_experiment.py --mode single --model yolopx --epochs 1 --batch-size 4\n")
    script_path.chmod(0o755)
    
    print(f"Scripts generated in {scripts_dir}/:")
    for script in scripts_dir.glob('*.sh'):
        print(f"  - {script.name}")
        
    return 0

def main():
    """Main function."""
    args = parse_args()
    
    # Run based on mode
    if args.mode == 'single':
        return run_single_model(args)
    elif args.mode == 'compare':
        return run_compare_models(args)
    elif args.mode == 'all':
        return run_all_models(args)
    elif args.mode == 'generate-scripts':
        return generate_scripts(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

if __name__ == '__main__':
    sys.exit(main())