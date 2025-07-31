import argparse, os , sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import wandb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.models import get_net_from_yaml
from lib.core.loss import get_loss
from lib.utils.utils import get_optimizer, is_parallel

from xy_conflict_solver_gemini import FixedGradientConflictSolver
from mdo_optimizer import MDO_Optimizer
from log_manager import WandBLogger, LocalFileLogger, ConsoleLogger
from validator import Validator
from trainer import Trainer

def main():
    args = parse_args()
    update_config(cfg, args)

    # --- Start of Gemini Modification ---
    # Force settings for baseline validation
    cfg.defrost()
    cfg.TRAIN.END_EPOCH = 20
    cfg.freeze()
    # --- End of Gemini Modification ---

    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.DEBUG else 'cpu')
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    train_loader, valid_loader, valid_dataset = create_data_loaders(cfg)

    # --- Start of Gemini Modification ---
    # Only run the 'original' method for this baseline test
    methods = [None] 
    # --- End of Gemini Modification ---
    
    all_experiment_metrics = {}

    for method in methods:
        method_name = method or 'original'
        print(f"\n{'='*50}\nStarting experiment with method: {method_name}\n{'='*50}")

        loggers = setup_loggers(cfg, method_name) 
        model, criterion, optimizer, scaler, solver = setup_experiment(cfg, device, method)

        begin_epoch = load_pretrained_model(model, optimizer, cfg, loggers['console'].get_logger(), method)

        trainer = Trainer(cfg, train_loader, model, criterion, optimizer, scaler, loggers['console'], device, loggers['wandb'], None, solver)
        validator = Validator(cfg, valid_loader, valid_dataset, model, criterion, loggers['local'].log_dir, loggers['console'], loggers['wandb'])

        metrics_for_plotting = {'train_total_loss_avg': [], 'task_conflict_intensity_avg': [], 'epoch_num': []}
        for_epoch = begin_epoch + cfg.TRAIN.END_EPOCH
        for epoch in range(begin_epoch + 1, for_epoch + 1):
            epoch_metrics = trainer.train_epoch(epoch, for_epoch)
            
            metrics_for_plotting['train_total_loss_avg'].append(epoch_metrics.get('total_loss', float('nan')))
            metrics_for_plotting['task_conflict_intensity_avg'].append(epoch_metrics.get('tci', float('nan')))
            metrics_for_plotting['epoch_num'].append(epoch)

            loggers['wandb'].log_epoch_metrics(epoch_metrics, epoch)

            if epoch >= cfg.TRAIN.END_EPOCH - 1:
                val_results = validator.validate(epoch)
                log_validation_results(loggers['wandb'], val_results, epoch)

        all_experiment_metrics[method_name] = metrics_for_plotting

    generate_comparison_plots(all_experiment_metrics, cfg, loggers['local'], loggers['wandb'])

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--eval_interval', type=int, default=1, help='Epoch interval for validation')
    return parser.parse_args()

def create_data_loaders(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg=cfg, is_train=True, inputsize=cfg.MODEL.IMAGE_SIZE, transform=transform)
    train_loader = DataLoaderX(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=cfg.TRAIN.SHUFFLE, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, collate_fn=dataset.AutoDriveDataset.collate_fn)
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE, transform=transform)
    valid_loader = DataLoaderX(valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY, collate_fn=dataset.AutoDriveDataset.collate_fn)
    return train_loader, valid_loader, valid_dataset

def setup_loggers(cfg, method_name):
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{hash(time.time()) % 10000:04d}"
    log_dir_path = Path(cfg.LOG_DIR) / cfg.DATASET.DATASET / f'{run_id}_{method_name}'
    log_dir_path.mkdir(parents=True, exist_ok=True)

    console_logger = ConsoleLogger(str(log_dir_path), f'{run_id}_{method_name}')
    wandb_logger = WandBLogger(cfg, f"{run_id}_{method_name}", project_name=f"multitask-training-{cfg.DATASET.DATASET}_refactor", reinit=True)
    local_file_logger = LocalFileLogger(str(log_dir_path))

    wandb_logger.set_logger(console_logger.get_logger())
    local_file_logger.set_logger(console_logger.get_logger())

    return {'console': console_logger, 'wandb': wandb_logger, 'local': local_file_logger}

def setup_experiment(cfg, device, method):
    # Use optimized model selection logic
    if method == 'tag':
        model_config_path = Path(__file__).parent.parent / 'lib' / 'config' / 'yolopx-tag-optimized.yaml'
    else:
        model_config_path = cfg.MODEL.CONFIG  # Use config-specified model path
    
    model = get_net_from_yaml(str(model_config_path)).to(device)
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type != 'cpu')

    solver_map = {
        'gradnorm': FixedGradientConflictSolver(method='gradnorm', num_tasks=3, device=device, alpha=1.5, update_freq=20),
        'pcgrad': FixedGradientConflictSolver(method='pcgrad', num_tasks=3, device=device),
        'cagrad': FixedGradientConflictSolver(method='cagrad', num_tasks=3, device=device, c=0.5),
        'mdo': MDO_Optimizer(model, num_tasks=3, device=device, update_freq=20),
        'tag': FixedGradientConflictSolver(method='tag', num_tasks=3, device=device, update_freq=20)
    }
    solver = solver_map.get(method)

    return model, criterion, optimizer, scaler, solver

def load_pretrained_model(model, optimizer, cfg, logger, method=None):
    """Loads a pretrained model - restored original logic with smart loading for TAG"""
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    if os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"Loading pretrained model: {cfg.MODEL.PRETRAINED}")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        begin_epoch = checkpoint['epoch']
        
        state_dict = checkpoint['state_dict']
        
        # For TAG method, use smart loading to handle architecture differences
        if method == 'tag':
            model_state_dict = model.state_dict()
            new_state_dict = {}
            skipped_keys = []
            
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    new_state_dict[k] = v
                else:
                    skipped_keys.append(k)
            
            # Handle DataParallel prefix if necessary
            if isinstance(model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in new_state_dict.keys()):
                new_state_dict = {'module.' + k: v for k, v in new_state_dict.items()}
            elif not isinstance(model, torch.nn.DataParallel) and any(k.startswith('module.') for k in new_state_dict.keys()):
                new_state_dict = {k.replace("module.", ""): v for k, v in new_state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"TAG model: Loaded {len(new_state_dict)} keys from checkpoint. Skipped {len(skipped_keys)} mismatched keys.")
            if len(skipped_keys) < 20:  # Only log if not too many
                for key in skipped_keys[:10]:  # Show first 10
                    logger.info(f"  Skipped: {key}")
        else:
            # Original strict loading for non-TAG methods
            if isinstance(model, torch.nn.DataParallel) and not 'module.' in list(state_dict.keys())[0]:
                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            elif not isinstance(model, torch.nn.DataParallel) and 'module.' in list(state_dict.keys())[0]:
                new_state_dict = {k.replace("module.", ""): v for k, v in new_state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

        if optimizer is not None and method != 'tag':
            # Skip optimizer loading for TAG since parameter count differs
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"Loaded optimizer state from epoch {checkpoint['epoch']}")
        elif method == 'tag':
            logger.info("Skipping optimizer state loading for TAG method due to architecture differences")
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        logger.info(f"No pretrained model found at {cfg.MODEL.PRETRAINED}, starting from scratch")
    
    return begin_epoch

def log_validation_results(wandb_logger, results, epoch):
    da_results, ll_results, detect_results, total_loss, _, times = results
    log_dict = {
        'val_loss': total_loss,
        'val_da_acc': da_results[0], 'val_da_iou': da_results[1], 'val_da_miou': da_results[2],
        'val_ll_acc': ll_results[0], 'val_ll_iou': ll_results[1], 'val_ll_miou': ll_results[2],
        'val_det_precision': detect_results[0], 'val_det_recall': detect_results[1], 'val_det_map50': detect_results[2], 'val_det_map': detect_results[3],
        'inference_time': times[0], 'nms_time': times[1],
        'epoch': epoch
    }
    wandb_logger.log_epoch_metrics(log_dict, epoch)

def generate_comparison_plots(all_metrics, cfg, local_logger, wandb_logger):
    metrics_to_plot = {
        'task_conflict_intensity_avg': 'Inter-Task Correlation (ITC) / Task Conflict Intensity',
        'train_total_loss_avg': 'Total Training Loss'
    }
    for metric_key, metric_title in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        has_data = False
        for method, metrics in all_metrics.items():
            if metrics.get(metric_key) and any(~np.isnan(v) for v in metrics[metric_key]):
                has_data = True
                valid_indices = ~np.isnan(metrics[metric_key])
                ax.plot(np.array(metrics['epoch_num'])[valid_indices], np.array(metrics[metric_key])[valid_indices], label=method, linewidth=1.5)
        
        if has_data:
            ax.set_title(f'{metric_title} Comparison', fontsize=16)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_title, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            fig.tight_layout()
            local_logger.save_comparison_plot(fig, f"MTL_comparison_{metric_key}")
            if wandb_logger.wandb_run:
                wandb_logger.wandb_run.log({f"comparative_plots/{metric_key}": wandb.Image(fig)})
        plt.close(fig)

if __name__ == '__main__':
    main()
