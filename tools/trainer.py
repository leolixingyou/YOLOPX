import torch
import math
import time
import numpy as np
from tqdm import tqdm
from torch.cuda import amp

from lib.utils.utils import is_parallel

class Trainer:
    def __init__(self, cfg, train_loader, model, criterion, optimizer, scaler, console_logger_instance, device, wandb_logger_instance=None, conflict_detector=None, conflict_solver=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.console_logger_instance = console_logger_instance
        self.device = device
        self.wandb_logger_instance = wandb_logger_instance
        self.conflict_detector = conflict_detector
        self.conflict_solver = conflict_solver

    def train_epoch(self, epoch, max_epoch):
        logger = self.console_logger_instance.get_logger()
        num_batch = len(self.train_loader)
        num_warmup = max(round(self.cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
        
        epoch_metrics = {m: AverageMeter() for m in ['total_loss', 'det_loss', 'da_seg_loss', 'll_seg_loss', 'tci']}

        self.model.train()
        self.console_logger_instance.remove_console_handler()

        train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{max_epoch}')

        for i, (input, target, paths, shapes) in enumerate(train_pbar):
            num_iter = i + num_batch * (epoch - 1)
            self._adjust_lr(num_iter, num_warmup, epoch)

            input = input.to(self.device, non_blocking=True)
            target = [t.to(self.device) for t in target]

            self.optimizer.zero_grad()

            with amp.autocast(enabled=self.device.type != 'cpu'):
                outputs = self.model(input)
                total_loss, head_losses = self.criterion(outputs, target, shapes, self.model, input)

                if self.conflict_solver:
                    total_loss = self.conflict_solver.compute_weighted_loss_with_gradients(head_losses, self._get_shared_params(), self.scaler, num_iter)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._update_metrics(epoch_metrics, total_loss, head_losses, input.size(0))
            train_pbar.set_postfix_str(f'Loss {epoch_metrics["total_loss"].avg:.4f}')

            if self.wandb_logger_instance:
                self._log_batch_to_wandb(epoch, i, total_loss, head_losses)

        self.console_logger_instance.add_console_handler()
        logger.info(f'Epoch {epoch} finished. Avg Loss: {epoch_metrics["total_loss"].avg:.4f}')

        return {k: v.avg for k, v in epoch_metrics.items()}

    def _get_shared_params(self):
        model_to_inspect = self.model.module if is_parallel(self.model) else self.model
        return [p for n, p in model_to_inspect.named_parameters() if p.requires_grad and 'head' not in n]

    def _adjust_lr(self, num_iter, num_warmup, epoch):
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / self.cfg.TRAIN.END_EPOCH)) / 2) * (1 - self.cfg.TRAIN.LRF) + self.cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi, [self.cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [self.cfg.TRAIN.WARMUP_MOMENTUM, self.cfg.TRAIN.MOMENTUM])

    def _update_metrics(self, metrics, total_loss, head_losses, batch_size):
        metrics['total_loss'].update(total_loss.item(), batch_size)
        if len(head_losses) >= 3:
            if head_losses[0] is not None: metrics['det_loss'].update(head_losses[0].item(), batch_size)
            if head_losses[1] is not None: metrics['da_seg_loss'].update(head_losses[1].item(), batch_size)
            if head_losses[2] is not None: metrics['ll_seg_loss'].update(head_losses[2].item(), batch_size)
            
            # Calculate Task Conflict Intensity (TCI)
            tci = self._calculate_tci(head_losses)
            metrics['tci'].update(tci, batch_size)

    def _calculate_tci(self, head_losses):
        """Calculate Task Conflict Intensity based on loss variance"""
        losses = []
        for loss in head_losses[:3]:  # Only consider first 3 losses
            if loss is not None and not torch.isnan(loss):
                losses.append(loss.item())
        
        if len(losses) < 2:
            return 0.0
            
        import numpy as np
        losses_array = np.array(losses)
        loss_mean = np.mean(losses_array)
        loss_var = np.var(losses_array)
        
        # Normalize by mean to avoid scale dependency
        tci = loss_var / (loss_mean + 1e-8) if loss_mean > 1e-8 else 0.0
        return tci

    def _log_batch_to_wandb(self, epoch, batch_idx, total_loss, head_losses):
        log_dict = {
            'batch/train_total_loss': total_loss.item(),
            'batch/learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch/epoch': epoch,
            'batch/batch_idx': batch_idx
        }
        if len(head_losses) >= 3:
            log_dict.update({
                'batch/train_det_loss': head_losses[0].item() if head_losses[0] is not None else float('nan'),
                'batch/train_da_seg_loss': head_losses[1].item() if head_losses[1] is not None else float('nan'),
                'batch/train_ll_seg_loss': head_losses[2].item() if head_losses[2] is not None else float('nan'),
            })
        self.wandb_logger_instance.log_batch_metrics(log_dict)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
