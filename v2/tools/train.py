"""
Main training script for the YOLOPX project (Refactored).

This script orchestrates the training and validation process.
It is designed to be highly configurable via command-line arguments
and YAML configuration files.
"""
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

# --- Add project root to sys.path ---
# This allows us to import modules from the `v2` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'v2'))

# --- Refactored Imports ---
from data.bdd_dataset import BddDataset
from data.autodrive_dataset import AutoDriveDataset
from models.builder import get_net_from_yaml
from models.losses.loss import get_loss
from engine.trainer import Trainer
from engine.validator import Validator
from utils.logger import WandBLogger, LocalFileLogger, ConsoleLogger, setup_program_logger
from mtl.grad_solvers import FixedGradientConflictSolver
from mtl.heuristic_solver import MDO_Optimizer
from utils.general import get_optimizer, is_parallel, DataLoaderX

def load_config(file_path):
    """Loads a YAML file into an EasyDict."""
    with open(file_path, 'r') as f:
        return edict(yaml.safe_load(f))

def main():
    """Main function to drive the training process."""
    parser = argparse.ArgumentParser(description='YOLOPX Training')
    parser.add_argument('--model-cfg', type=str, default='cfgs/models/yolopx.yaml', help='Path to model.yaml')
    parser.add_argument('--data-cfg', type=str, default='cfgs/data/bdd100k.yaml', help='Path to data.yaml')
    parser.add_argument('--train-cfg', type=str, default='cfgs/train_default.yaml', help='Path to train.yaml')
    parser.add_argument('--mtl-method', type=str, default='original',
                        choices=['original', 'gradnorm', 'pcgrad', 'cagrad', 'tag', 'mdo_heuristic'],
                        help='Multi-task learning method')
    parser.add_argument('--weights', type=str, default='', help='Path to pretrained weights')
    parser.add_argument('--run-name', type=str, default=None, help='Custom name for the run')
    args = parser.parse_args()

    # --- Load Configurations ---
    cfg = edict()
    cfg.update(load_config(args.data_cfg))
    cfg.update(load_config(args.train_cfg))
    cfg.MODEL.CONFIG = args.model_cfg # Add model config path to cfg
    cfg.MODEL.PRETRAINED = args.weights

    # --- Setup Environment ---
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.DEBUG else 'cpu')
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    # --- Setup Logging ---
    run_name = args.run_name or f"{Path(args.model_cfg).stem}-{args.mtl_method}-{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(cfg.LOG_DIR) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    program_logger = setup_program_logger(log_dir)
    loggers = setup_loggers(cfg, log_dir, run_name)
    
    program_logger.info(f"Starting experiment: {run_name}")
    program_logger.info(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")
    program_logger.info(f"Config files: Model={args.model_cfg}, Data={args.data_cfg}, Train={args.train_cfg}")

    # --- Build Components ---
    train_loader, valid_loader, valid_dataset = build_dataloaders(cfg)
    model, criterion, optimizer, scaler, solver = build_experiment_components(cfg, device, args.mtl_method)
    
    program_logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(valid_dataset)}")

    # --- Load Pretrained Model ---
    begin_epoch = load_pretrained_model(model, optimizer, cfg, loggers['console'].get_logger(), args.mtl_method)

    # --- Initialize Engines ---
    trainer = Trainer(cfg, train_loader, model, criterion, optimizer, scaler, loggers['console'], device, loggers['wandb'], solver)
    validator = Validator(cfg, valid_loader, valid_dataset, model, criterion, loggers['local'], loggers['wandb'])

    # --- Training Loop ---
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        epoch_metrics = trainer.train_epoch(epoch, cfg.TRAIN.END_EPOCH)
        loggers['wandb'].log_epoch_metrics(epoch_metrics, epoch)

        # Skip validation for now to focus on training
        # if epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH:
        #     val_results = validator.validate(epoch)
        #     # TODO: Add logging for validation results

    program_logger.info(f"Experiment {run_name} completed successfully.")
    program_logger.info(f"Results saved in: {log_dir}")

def build_dataloaders(cfg):
    """Builds and returns the training and validation dataloaders."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.DATASET.IMAGE_SIZE),
        transforms.ToTensor(), 
        normalize
    ])
    
    # Use a factory pattern or eval safely
    dataset_map = {'BddDataset': BddDataset}
    dataset_class = dataset_map[cfg.DATASET.DATASET]

    train_dataset = dataset_class(cfg=cfg, is_train=True, transform=transform)
    train_loader = DataLoaderX(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=cfg.TRAIN.SHUFFLE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, collate_fn=AutoDriveDataset.collate_fn)
    
    valid_dataset = dataset_class(cfg=cfg, is_train=False, transform=transform)
    valid_loader = DataLoaderX(valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, collate_fn=AutoDriveDataset.collate_fn)
    
    return train_loader, valid_loader, valid_dataset

def build_experiment_components(cfg, device, method):
    """Builds and returns the core components for an experiment."""
    # --- Model ---
    model_config_path = cfg.MODEL.CONFIG
    if method == 'tag':
        # Use the paper-based TAG configuration
        model_config_path = str(Path(cfg.MODEL.CONFIG).parent / 'yolopx_tag_paper.yaml')
    model = get_net_from_yaml(model_config_path).to(device)

    # --- Loss, Optimizer, Scaler ---
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type != 'cpu')

    # --- MTL Solver ---
    solver_map = {
        'gradnorm': FixedGradientConflictSolver(method='gradnorm', num_tasks=3, device=device, alpha=1.5, update_freq=20),
        'pcgrad': FixedGradientConflictSolver(method='pcgrad', num_tasks=3, device=device),
        'cagrad': FixedGradientConflictSolver(method='cagrad', num_tasks=3, device=device, c=0.5),
        'mdo_heuristic': MDO_Optimizer(model, num_tasks=3, device=device, update_freq=20),
        'tag': FixedGradientConflictSolver(method='tag', num_tasks=3, device=device, update_freq=20)
    }
    solver = solver_map.get(method)

    return model, criterion, optimizer, scaler, solver

def setup_loggers(cfg, log_dir, run_name):
    """Initializes all loggers for the run."""
    console_logger = ConsoleLogger(log_dir, run_name)
    local_file_logger = LocalFileLogger(log_dir)
    wandb_logger = WandBLogger(cfg, run_name, project_name=f"Phoenix-YOLOPX")
    
    return {'console': console_logger, 'wandb': wandb_logger, 'local': local_file_logger}

def load_pretrained_model(model, optimizer, cfg, logger, method=None):
    """Loads a pretrained model with smart handling for architecture mismatches."""
    begin_epoch = 0
    if cfg.MODEL.PRETRAINED and os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"Loading pretrained model: {cfg.MODEL.PRETRAINED}")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        
        # Smart state_dict loading
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            # Adjust for DataParallel prefix
            k_no_module = k.replace("module.", "")
            if k_no_module in model_state_dict and model_state_dict[k_no_module].shape == v.shape:
                new_state_dict[k_no_module] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded {len(new_state_dict)} keys from checkpoint.")

        # Load optimizer state if architectures match
        if 'optimizer' in checkpoint and method != 'tag':
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("Loaded optimizer state.")
            except ValueError:
                logger.warning("Could not load optimizer state, likely due to architecture mismatch.")

        begin_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resuming from epoch {begin_epoch}")
    else:
        logger.info("No pretrained model found, starting from scratch.")
    
    return begin_epoch

if __name__ == '__main__':
    main()