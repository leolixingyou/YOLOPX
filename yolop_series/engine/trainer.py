"""
The core training engine for the YOLOPX project.
"""
import torch
import sys
import math
import numpy as np
from tqdm import tqdm
from torch.cuda import amp

# --- Refactored Imports ---
from utils.general import is_parallel, AverageMeter

class Trainer:
    """
    The Trainer class encapsulates the logic for a single training epoch.
    """
    def __init__(self, cfg, train_loader, model, criterion, optimizer, scaler, console_logger, device, wandb_logger=None, solver=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.console_logger = console_logger
        self.device = device
        self.wandb_logger = wandb_logger
        self.solver = solver

    def train_epoch(self, epoch, max_epoch):
        """Runs a single epoch of training."""
        logger = self.console_logger.get_logger()
        num_batch = len(self.train_loader)
        num_warmup = max(round(self.cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
        
        epoch_metrics = {m: AverageMeter() for m in ['total_loss', 'det_loss', 'da_seg_loss', 'll_seg_loss', 'tci']}

        self.model.train()
        
        pbar_desc = f'Epoch {epoch}/{max_epoch}'
        train_pbar = tqdm(self.train_loader, desc=pbar_desc, file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}')

        for i, (input_data, det_labels, seg_labels, lane_labels, paths) in enumerate(train_pbar):
            # Pack labels for multi-task training
            target = [det_labels, seg_labels, lane_labels]
            shapes = None  # Shapes not provided by this dataset format
            num_iter = i + num_batch * (epoch - 1)
            self._adjust_lr(num_iter, num_warmup, epoch)

            input_data = input_data.to(self.device, non_blocking=True)
            target = [t.to(self.device) for t in target]

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.device.type != 'cpu'):
                outputs = self.model(input_data)
                total_loss, head_losses = self.criterion(outputs, target, shapes, self.model, input_data)

                if self.solver:
                    # The solver will handle gradient calculation and weighting internally
                    total_loss = self.solver.compute_weighted_loss_with_gradients(
                        head_losses, self._get_shared_params(), self.scaler, num_iter
                    )

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._update_metrics(epoch_metrics, total_loss, head_losses, input_data.size(0))
            train_pbar.set_postfix_str(f'Loss {epoch_metrics["total_loss"].avg:.4f}')

            if self.wandb_logger:
                self._log_batch_to_wandb(epoch, i, epoch_metrics)

        logger.info(f'Epoch {epoch} finished. Avg Loss: {epoch_metrics["total_loss"].avg:.4f}')
        return {k: v.avg for k, v in epoch_metrics.items()}

    def _get_shared_params(self):
        """Identifies parameters shared across task heads."""
        model_to_inspect = self.model.module if is_parallel(self.model) else self.model
        # A more robust way to identify shared params would be needed if arch changes.
        # For now, assuming non-head params are shared.
        return [p for n, p in model_to_inspect.named_parameters() if p.requires_grad and 'head' not in n]

    def _adjust_lr(self, num_iter, num_warmup, epoch):
        """Adjusts learning rate based on warmup schedule."""
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / self.cfg.TRAIN.END_EPOCH)) / 2) * (1 - self.cfg.TRAIN.LRF) + self.cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(self.optimizer.param_groups):
                # Bias lr warmup is handled differently
                x['lr'] = np.interp(num_iter, xi, [self.cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [self.cfg.TRAIN.WARMUP_MOMENTUM, self.cfg.TRAIN.MOMENTUM])

    def _update_metrics(self, metrics, total_loss, head_losses, batch_size):
        """Updates the running average of metrics for the epoch."""
        metrics['total_loss'].update(total_loss.item(), batch_size)
        if len(head_losses) >= 3:
            det_loss, da_loss, ll_loss = head_losses[0], head_losses[1], head_losses[2]
            if det_loss is not None: metrics['det_loss'].update(det_loss.item(), batch_size)
            if da_loss is not None: metrics['da_seg_loss'].update(da_loss.item(), batch_size)
            if ll_loss is not None: metrics['ll_seg_loss'].update(ll_loss.item(), batch_size)
            
            tci = self._calculate_tci([h.item() for h in head_losses if h is not None])
            metrics['tci'].update(tci, batch_size)

    def _calculate_tci(self, losses):
        """Calculates Task Conflict Intensity based on normalized loss variance."""
        if len(losses) < 2:
            return 0.0
        losses_array = np.array(losses)
        loss_mean = np.mean(losses_array)
        loss_var = np.var(losses_array)
        # Normalize by mean to avoid scale dependency
        return loss_var / (loss_mean + 1e-8) if loss_mean > 1e-8 else 0.0

    def _log_batch_to_wandb(self, epoch, batch_idx, metrics):
        """Logs batch-level metrics to Weights & Biases."""
        log_dict = {
            'batch/train_total_loss': metrics['total_loss'].val,
            'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch/epoch': epoch,
            'batch/batch_idx': batch_idx,
            'batch/det_loss': metrics['det_loss'].val,
            'batch/da_seg_loss': metrics['da_seg_loss'].val,
            'batch/ll_seg_loss': metrics['ll_seg_loss'].val,
            'batch/tci': metrics['tci'].val,
        }
        self.wandb_logger.log_batch_metrics(log_dict)