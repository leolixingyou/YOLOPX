#!/usr/bin/env python3
"""Unified experiment runner for YOLOP models with YAML configuration support."""

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
    parser.add_argument('--mode', type=str,
                      choices=['single', 'compare', 'all', 'generate-scripts'],
                      help='Experiment mode (overrides config)')
    
    # Model selection
    parser.add_argument('--model', type=str,
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                      help='Model to train (for single mode)')
    parser.add_argument('--models', nargs='+',
                      choices=['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'],
                      help='Models to compare (for compare mode)')
    
    # Training parameters (override config)
    parser.add_argument('--epochs', type=int,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int,
                      help='Batch size')
    parser.add_argument('--lr', type=float,
                      help='Learning rate')
    parser.add_argument('--workers', type=int,
                      help='Number of data loading workers')
    
    # System
    parser.add_argument('--device', type=str,
                      help='Device to use')
    parser.add_argument('--seed', type=int,
                      help='Random seed')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                      help='Use wandb logging')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable wandb logging')
    
    # Dataset
    parser.add_argument('--train-samples', type=int,
                      help='Number of training samples')
    parser.add_argument('--val-samples', type=int,
                      help='Number of validation samples')
    parser.add_argument('--quick-test', action='store_true',
                      help='Quick test with limited data')
    
    return parser.parse_args()

def run_single_model(config, project_manager):
    """Run training for a single model."""
    models = config.get('models', [])
    if not models:
        print("Error: No models specified in config")
        sys.exit(1)
    
    model = models[0] if isinstance(models, list) else models
    
    # Prepare command
    cmd = [
        sys.executable, 'train_options/train_unified.py',
        '--model', model,
        '--epochs', str(config.get('training.epochs')),
        '--batch-size', str(config.get('training.batch_size')),
        '--lr', str(config.get('training.learning_rate')),
        '--device', config.get('training.device'),
        '--workers', str(config.get('training.workers')),
        '--seed', str(config.get('training.seed')),
        '--project-manager', str(project_manager.project_dir)
    ]
    
    if config.get('logging.wandb'):
        cmd.append('--wandb')
        
    if config.get('dataset.quick_test'):
        cmd.append('--quick-test')
        
    train_samples = config.get('dataset.train_samples')
    if train_samples:
        cmd.extend(['--train-samples', str(train_samples)])
        
    val_samples = config.get('dataset.val_samples')
    if val_samples:
        cmd.extend(['--val-samples', str(val_samples)])
        
    print(f"Starting training for {model}...")
    print(f"Project directory: {project_manager.project_dir}")
    
    # Save configuration
    config.save(project_manager.project_dir / "experiment_config.yaml")
    
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

def run_compare_models(config, project_manager):
    """Run comparison experiment for multiple models."""
    models = config.get('models', ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx'])
    
    # Run comparison script
    cmd = [
        sys.executable, 'train_options/train_compare.py',
        '--models'] + models + [
        '--epochs', str(config.get('training.epochs')),
        '--batch-size', str(config.get('training.batch_size')),
        '--lr', str(config.get('training.learning_rate')),
        '--workers', str(config.get('training.workers')),
        '--device', config.get('training.device')
    ]
    
    if config.get('logging.wandb'):
        cmd.append('--wandb')
        
    train_samples = config.get('dataset.train_samples')
    if train_samples:
        cmd.extend(['--train-samples', str(train_samples)])
        
    val_samples = config.get('dataset.val_samples')
    if val_samples:
        cmd.extend(['--val-samples', str(val_samples)])
        
    print(f"Starting comparison experiment for: {', '.join(models)}")
    
    # Save configuration
    config.save(project_manager.project_dir / "experiment_config.yaml")
    
    result = subprocess.run(cmd)
    return result.returncode

def generate_scripts(config):
    """Generate shell scripts for different experiments."""
    scripts_dir = Path('scripts')
    scripts_dir.mkdir(exist_ok=True)
    
    # Common parameters from config
    epochs = config.get('training.epochs', 20)
    batch_size = config.get('training.batch_size', 8)
    lr = config.get('training.learning_rate', 0.001)
    
    common_params = f"--epochs {epochs} --batch-size {batch_size} --lr {lr}"
    
    # Generate individual model scripts
    for model in ['yolop_v1', 'yolop_v2', 'yolop_v3', 'yolopx']:
        script_path = scripts_dir / f'train_{model}.sh'
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Train {model.upper()}\n\n")
            f.write(f"python3 run_experiment_v2.py --config experiments/base_config.yaml --mode single --model {model} {common_params}\n")
        script_path.chmod(0o755)
        
    # Generate comparison script
    script_path = scripts_dir / 'train_compare_all.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Compare all YOLOP models\n\n")
        f.write(f"python3 run_experiment_v2.py --config experiments/compare_all_models.yaml --mode all\n")
    script_path.chmod(0o755)
    
    # Generate quick test script
    script_path = scripts_dir / 'quick_test.sh'
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Quick test with limited data\n\n")
        f.write("python3 run_experiment_v2.py --config experiments/quick_test.yaml --mode single\n")
    script_path.chmod(0o755)
    
    print(f"Scripts generated in {scripts_dir}/:")
    for script in scripts_dir.glob('*.sh'):
        print(f"  - {script.name}")
        
    return 0

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = ExperimentConfig(args.config)
    
    # Update config with command-line arguments
    config.update_from_args(args)
    
    # Determine mode
    mode = args.mode
    if not mode:
        # Try to infer mode from config
        models = config.get('models', [])
        if len(models) == 1:
            mode = 'single'
        elif len(models) > 1:
            mode = 'compare'
        else:
            print("Error: No mode specified and cannot infer from config")
            return 1
            
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create project manager with new naming scheme
    project_name = config.get_project_name(__file__)
    models = config.get('models', [])
    
    # Check for shared project setting
    use_shared = config.get('system.shared_project', True)
    
    # If using shared project and this is a sub-experiment, try to find existing project
    if use_shared and 'YOLOP_PROJECT_DIR' in os.environ:
        project_manager = ProjectManager(
            models, 
            project_name=None,
            main_script=None
        )
        project_manager.project_dir = Path(os.environ['YOLOP_PROJECT_DIR'])
        project_manager.project_name = project_manager.project_dir.name
    else:
        project_manager = ProjectManager(
            models,
            project_name=project_name,
            main_script=__file__,
            use_shared_project=use_shared
        )
        
        # Set environment variable for sub-processes
        if use_shared:
            os.environ['YOLOP_PROJECT_DIR'] = str(project_manager.project_dir)
    
    # Log experiment start
    logging.info(f"Starting experiment: {project_manager.project_name}")
    logging.info(f"Models: {models}")
    logging.info(f"Configuration: epochs={config.get('training.epochs')}, "
                f"batch_size={config.get('training.batch_size')}, "
                f"lr={config.get('training.learning_rate')}")
    
    # Create experiment log
    log_path = project_manager.project_dir / "experiment.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Run based on mode
    if mode == 'single':
        return run_single_model(config, project_manager)
    elif mode == 'compare':
        return run_compare_models(config, project_manager)
    elif mode == 'all':
        return run_compare_models(config, project_manager)
    elif mode == 'generate-scripts':
        return generate_scripts(config)
    else:
        print(f"Unknown mode: {mode}")
        return 1

if __name__ == '__main__':
    sys.exit(main())