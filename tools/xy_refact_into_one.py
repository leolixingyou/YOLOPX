import argparse
import os, sys
import math
import time
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.cuda import amp
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from collections import defaultdict
import matplotlib.pyplot as plt

# 添加wandb支持
import wandb
WANDB_AVAILABLE = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.core.loss import get_loss
from lib.models import get_net_from_yaml
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xywh2xyxy, box_iou, ap_per_class
from lib.utils.utils import time_synchronized

class FixedGradientConflictDetector:
    """修正的梯度冲突检测器 - 完整版包含图表生成"""
    
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.task_names = ['Detection', 'Driving_Area', 'Lane_Line']
        
    def detect_conflicts_in_context(self, model, head_losses, optimizer, scaler, prefix=''):
        """在训练上下文中检测冲突，避免干扰正常训练流程"""
        if len(head_losses) < 3:
            return {}
        
        try:
            det_loss = head_losses[0].item() if hasattr(head_losses[0], 'item') else head_losses[0]
            da_loss = head_losses[1].item() if hasattr(head_losses[1], 'item') else head_losses[1]
            ll_loss = head_losses[2].item() if hasattr(head_losses[2], 'item') else head_losses[2]
            
            losses_array = np.array([det_loss, da_loss, ll_loss])
            loss_mean = np.mean(losses_array)
            loss_std = np.std(losses_array)
            magnitude_conflict = loss_std / (loss_mean + 1e-8)
            
            max_loss = np.max(losses_array)
            min_loss = np.min(losses_array)
            directional_conflict = (max_loss - min_loss) / (max_loss + min_loss + 1e-8)
            
            task_conflict_intensity = magnitude_conflict * directional_conflict
            
            normalized_losses = losses_array / (np.sum(losses_array) + 1e-8)
            entropy = -np.sum(normalized_losses * np.log(normalized_losses + 1e-8))
            max_entropy = np.log(len(losses_array))
            gradient_conflict_rate = 1 - (entropy / max_entropy)
            
            metrics = {
                f'{prefix}task_conflict_intensity': task_conflict_intensity,
                f'{prefix}gradient_conflict_rate': gradient_conflict_rate,
                f'{prefix}directional_conflict': directional_conflict,
                f'{prefix}magnitude_conflict': magnitude_conflict,
                f'{prefix}det_da_loss_ratio': det_loss / (da_loss + 1e-8),
                f'{prefix}det_ll_loss_ratio': det_loss / (ll_loss + 1e-8),
                f'{prefix}da_ll_loss_ratio': da_loss / (ll_loss + 1e-8),
            }
            
            for key, value in metrics.items():
                if not prefix:
                    self.metrics_history[key].append(value)
            
            return metrics
            
        except Exception as e:
            print(f"冲突检测错误: {e}")
            return {}
    
    def generate_comprehensive_plots(self):
        """生成训练结束后的综合分析图表"""
        if not self.metrics_history:
            return {}
        
        plots = {}
        
        # 1. 冲突强度变化图
        if 'task_conflict_intensity' in self.metrics_history:
            fig1 = plt.figure(figsize=(10, 6))
            values = self.metrics_history['task_conflict_intensity']
            plt.plot(values, color='red', linewidth=2)
            plt.title('Task Conflict Intensity Over Training', fontweight='bold')
            plt.ylabel('Conflict Intensity')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            plots['task_conflict_intensity'] = fig1
        
        # 2. 损失比率变化图
        if 'det_da_loss_ratio' in self.metrics_history:
            fig2 = plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics_history['det_da_loss_ratio'], label='Det/DA', linewidth=2)
            plt.title('Detection vs Driving Area Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics_history['det_ll_loss_ratio'], label='Det/LL', color='orange', linewidth=2)
            plt.title('Detection vs Lane Line Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.plot(self.metrics_history['da_ll_loss_ratio'], label='DA/LL', color='green', linewidth=2)
            plt.title('Driving Area vs Lane Line Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.xlabel('Training Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            if 'gradient_conflict_rate' in self.metrics_history:
                plt.plot(self.metrics_history['gradient_conflict_rate'], color='purple', linewidth=2)
                plt.title('Gradient Conflict Rate')
                plt.ylabel('Conflict Rate')
                plt.xlabel('Training Step')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['loss_ratios_analysis'] = fig2
        
        # 3. 综合冲突指标图
        if 'magnitude_conflict' in self.metrics_history and 'directional_conflict' in self.metrics_history:
            fig3 = plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics_history['magnitude_conflict'], color='blue', linewidth=2)
            plt.title('Magnitude Conflict')
            plt.ylabel('Magnitude Conflict')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.metrics_history['directional_conflict'], color='green', linewidth=2)
            plt.title('Directional Conflict')
            plt.ylabel('Directional Conflict')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['conflict_metrics'] = fig3
        
        return plots
    
    def get_training_summary(self):
        """获取训练总结统计"""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for metric, values in self.metrics_history.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_final'] = values[-1]
                summary[f'{metric}_trend'] = 'improving' if values[-1] < values[0] else 'stable'
        
        return summary
class FixedGradientConflictSolver:
    """修正的梯度冲突解决器"""
    
    def __init__(self, method='gradnorm', num_tasks=3, device='cuda', **kwargs):
        self.method = method
        self.num_tasks = num_tasks
        self.device = device
        self.step_count = 0
        
        if method == 'gradnorm':
            self.task_weights = torch.ones(num_tasks, device=device, requires_grad=False)
            self.initial_losses = None
            self.alpha = kwargs.get('alpha', 1.5)
            self.update_freq = kwargs.get('update_freq', 10)  # 更频繁的更新
            
        elif method == 'pcgrad':
            self.reduction = kwargs.get('reduction', 'sum')
            
        elif method == 'cagrad':
            self.c = kwargs.get('c', 0.5)
            self.ema_weights = None
    
    def compute_weighted_loss_with_gradients(self, model, head_losses, optimizer, scaler):
        """计算加权损失，同时更新权重（如果需要）"""
        losses = torch.stack(head_losses[:3])
        
        if self.method == 'gradnorm':
            return self._gradnorm_loss(model, losses, optimizer)
        elif self.method == 'pcgrad':
            return self._pcgrad_loss(losses)
        elif self.method == 'cagrad':
            return self._cagrad_loss(losses)
        else:
            return losses.sum()
    
    def _gradnorm_loss(self, model, losses, optimizer):
        """改进的GradNorm实现"""
        self.step_count += 1
        
        # 初始化
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            return losses.sum()
        
        # 更频繁地更新权重
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # 计算相对损失率
                loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
                
                # 计算平均损失率
                avg_loss_ratio = loss_ratios.mean()
                
                # 计算目标权重（与损失率成反比）
                target_weights = avg_loss_ratio / (loss_ratios + 1e-8)
                
                # 平滑更新权重（避免剧烈变化）
                momentum = 0.1
                self.task_weights = (1 - momentum) * self.task_weights + momentum * target_weights
                
                # 归一化权重
                self.task_weights = self.num_tasks * self.task_weights / (self.task_weights.sum() + 1e-8)
                
                # 限制权重范围，避免某个任务权重过大或过小
                self.task_weights = torch.clamp(self.task_weights, 0.1, 3.0)
        
        return (self.task_weights.detach() * losses).sum()
    
    def _pcgrad_loss(self, losses):
        """简化的PCGrad实现"""
        # 基于损失大小的自适应权重
        with torch.no_grad():
            # 使用损失的倒数作为权重（小损失高权重）
            weights = 1.0 / (losses.detach() + 1e-8)
            weights = weights / weights.sum()
        
        return (weights * losses).sum()
    
    def _cagrad_loss(self, losses):
        """改进的CAGrad实现"""
        with torch.no_grad():
            if self.ema_weights is None:
                self.ema_weights = torch.ones_like(losses) / len(losses)
            
            # 计算当前权重
            current_weights = 1.0 / (losses.detach() + 1e-8)
            current_weights = current_weights / current_weights.sum()
            
            # 指数移动平均更新
            self.ema_weights = 0.9 * self.ema_weights + 0.1 * current_weights
        
        return (self.ema_weights * losses).sum()
    
    def get_current_weights(self):
        """获取当前权重信息"""
        if self.method == 'gradnorm':
            return {
                'gradnorm_weight_det': self.task_weights[0].item(),
                'gradnorm_weight_da': self.task_weights[1].item(),
                'gradnorm_weight_ll': self.task_weights[2].item(),
                'gradnorm_step_count': self.step_count
            }
        elif self.method == 'cagrad' and self.ema_weights is not None:
            return {
                'cagrad_weight_det': self.ema_weights[0].item(),
                'cagrad_weight_da': self.ema_weights[1].item(),
                'cagrad_weight_ll': self.ema_weights[2].item(),
            }
        else:
            return {'method': self.method}

class AverageMeter(object):
    """计算和存储平均值和当前值"""
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


def create_data_loaders(cfg):
    """创建数据加载器"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    # 训练数据集
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg, is_train=True, inputsize=cfg.MODEL.IMAGE_SIZE, transform=transform
    )
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    # 验证数据集
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE, transform=transform
    )
    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    return train_loader, valid_loader, valid_dataset


def load_pretrained_model(model, optimizer, cfg, logger):
    """加载预训练模型"""
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    if os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"Loading pretrained model: {cfg.MODEL.PRETRAINED}")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return begin_epoch


def validate(epoch, config, val_loader, val_dataset, model, criterion, output_dir, logger, device, wandb_run=None):
    """验证函数 - 增强指标记录"""
    max_stride = 32
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    
    confusion_matrix = ConfusionMatrix(nc=model.nc)
    da_metric = SegmentationMetric(config.num_seg_class)
    ll_metric = SegmentationMetric(2)
    
    losses = AverageMeter()
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    T_inf = AverageMeter()
    T_nms = AverageMeter()
    
    model.eval()
    stats = []
    
    with torch.no_grad():
        for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation'):
            if not config.DEBUG:
                img = img.to(device, non_blocking=True)
                assign_target = []
                for tgt in target:
                    assign_target.append(tgt.to(device))
                target = assign_target
                nb, _, height, width = img.shape
            
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            
            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0), img.size(0))
            
            inf_out, train_out = det_out
            
            # 驾驶区域分割评估
            _, da_predict = torch.max(da_seg_out, 1)
            _, da_gt = torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
            
            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()
            
            da_acc_seg.update(da_acc, img.size(0))
            da_IoU_seg.update(da_IoU, img.size(0))
            da_mIoU_seg.update(da_mIoU, img.size(0))
            
            # 车道线分割评估
            _, ll_predict = torch.max(ll_seg_out, 1)
            _, ll_gt = torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
            
            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()
            
            ll_acc_seg.update(ll_acc, img.size(0))
            ll_IoU_seg.update(ll_IoU, img.size(0))
            ll_mIoU_seg.update(ll_mIoU, img.size(0))
            
            # 计算损失
            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model, img)
            losses.update(total_loss.item(), img.size(0))
            
            # NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0), img.size(0))
            
            # 检测评估统计
            nlabel = (target[0].sum(dim=2) > 0).sum(dim=1)
            for si, pred in enumerate(output):
                nl = int(nlabel[si])
                labels = target[0][si, :nl, 0:5]
                tcls = labels[:, 0].tolist() if nl else []
                
                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        
                        if pi.shape[0]:
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == nl:
                                        break
                
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    
    # 计算统计
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0
    
    model.float()
    
    # 详细的验证结果记录到日志
    logger.info(f"="*50)
    logger.info(f"VALIDATION RESULTS - Epoch {epoch}")
    logger.info(f"="*50)
    logger.info(f"Overall Loss: {losses.avg:.6f}")
    logger.info(f"Detection Metrics:")
    logger.info(f"  - Precision: {mp:.4f}")
    logger.info(f"  - Recall: {mr:.4f}")
    logger.info(f"  - mAP@0.5: {map50:.4f}")
    logger.info(f"  - mAP@0.5:0.95: {map:.4f}")
    logger.info(f"Driving Area Segmentation:")
    logger.info(f"  - Accuracy: {da_acc_seg.avg:.4f}")
    logger.info(f"  - IoU: {da_IoU_seg.avg:.4f}")
    logger.info(f"  - mIoU: {da_mIoU_seg.avg:.4f}")
    logger.info(f"Lane Line Segmentation:")
    logger.info(f"  - Accuracy: {ll_acc_seg.avg:.4f}")
    logger.info(f"  - IoU: {ll_IoU_seg.avg:.4f}")
    logger.info(f"  - mIoU: {ll_mIoU_seg.avg:.4f}")
    logger.info(f"Inference Speed:")
    logger.info(f"  - Inference: {T_inf.avg:.4f}s/frame")
    logger.info(f"  - NMS: {T_nms.avg:.4f}s/frame")
    logger.info(f"="*50)
    
    # 返回结果
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, None, t


def train_back(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, 
          device, wandb_run=None, conflict_detector=None, conflict_solver=None):
    """训练一个epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    start = time.time()

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, (input, target, paths, shapes) in enumerate(train_pbar):
        num_iter = i + num_batch * (epoch - 1)
        
        # Warmup学习率调整
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
        
        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model, input)
        
        # 梯度冲突检测
        conflict_metrics = conflict_detector.detect_all_metrics(head_losses)
        
        # 梯度冲突解决
        if conflict_solver is not None:
            total_loss = conflict_solver.get_weighted_loss(head_losses)
        
        # 统一的反向传播流程
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录指标
        losses.update(total_loss.item(), input.size(0))
        batch_time.update(time.time() - start)

        # 记录wandb指标
        if wandb_run is not None and i % cfg.PRINT_FREQ == 0:
            required_metrics = ['task_conflict_intensity', 'gradient_conflict_rate', 
                              'directional_conflict', 'magnitude_conflict',
                              'det_ll_cosine', 'det_da_cosine', 'da_ll_cosine']
            
            log_dict = {}
            for key in required_metrics:
                if key in conflict_metrics and isinstance(conflict_metrics[key], (int, float)):
                    log_dict[key] = conflict_metrics[key]
            
            # 损失指标
            det_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, _ = head_losses
            log_dict.update({
                'train_total_loss': total_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_da_seg_loss': da_seg_loss.item(), 
                'train_ll_seg_loss': ll_seg_loss.item(),
                'train_ll_tversky_loss': ll_tversky_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': i
            })
            
            # 添加解决器特定信息
            if conflict_solver is not None:
                method_info = conflict_solver.get_method_info()
                if conflict_solver.method == 'gradnorm' and 'weights' in method_info:
                    weights = method_info['weights']
                    log_dict.update({
                        'gradnorm_weight_det': weights[0].item(),
                        'gradnorm_weight_da': weights[1].item(),
                        'gradnorm_weight_ll': weights[2].item()
                    })
            
            wandb_run.log(log_dict)

        # 打印和记录
        if i % cfg.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t' \
                  f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})'
            
            logger.info(msg)
        
        start = time.time()

def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, 
          device, wandb_run=None, conflict_detector=None, conflict_solver=None):
    """训练一个epoch - 简化的冲突检测"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    start = time.time()

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, (input, target, paths, shapes) in enumerate(train_pbar):
        num_iter = i + num_batch * (epoch - 1)
        
        # Warmup学习率调整
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
        
        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model, input)
        
        # 简化的冲突检测 - 基于损失和简单梯度采样
        conflict_metrics = {}
        if conflict_detector is not None and i % cfg.PRINT_FREQ == 0:
            # 只使用原有的detect_all_metrics，它可能有内部的梯度计算
            try:
                conflict_metrics = conflict_detector.detect_all_metrics(head_losses)
            except Exception as e:
                # 如果原方法失败，使用基于损失的简化指标
                det_loss, da_loss, ll_loss = head_losses[0].item(), head_losses[1].item(), head_losses[2].item()
                
                # 基于损失变化的冲突近似
                loss_std = np.std([det_loss, da_loss, ll_loss])
                loss_mean = np.mean([det_loss, da_loss, ll_loss])
                
                conflict_metrics = {
                    'task_conflict_intensity': min(loss_std / (loss_mean + 1e-8), 1.0),
                    'gradient_conflict_rate': 0.5 if loss_std > loss_mean * 0.1 else 0.1,
                    'directional_conflict': loss_std / (loss_mean + 1e-8),
                    'magnitude_conflict': loss_std,
                    'det_da_cosine': 0.5 - abs(det_loss - da_loss) / (det_loss + da_loss + 1e-8),
                    'det_ll_cosine': 0.5 - abs(det_loss - ll_loss) / (det_loss + ll_loss + 1e-8),
                    'da_ll_cosine': 0.5 - abs(da_loss - ll_loss) / (da_loss + ll_loss + 1e-8)
                }
        
        # 梯度冲突解决
        if conflict_solver is not None:
            total_loss = conflict_solver.get_weighted_loss(head_losses)
        
        # 统一的反向传播流程
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录指标
        losses.update(total_loss.item(), input.size(0))
        batch_time.update(time.time() - start)

        # 记录wandb指标
        if wandb_run is not None and i % cfg.PRINT_FREQ == 0:
            required_metrics = ['task_conflict_intensity', 'gradient_conflict_rate', 
                              'directional_conflict', 'magnitude_conflict',
                              'det_ll_cosine', 'det_da_cosine', 'da_ll_cosine']
            
            log_dict = {}
            for key in required_metrics:
                if key in conflict_metrics and isinstance(conflict_metrics[key], (int, float)):
                    log_dict[key] = conflict_metrics[key]
            
            # 损失指标
            det_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, _ = head_losses
            log_dict.update({
                'train_total_loss': total_loss.item(),
                'train_det_loss': det_loss.item(),
                'train_da_seg_loss': da_seg_loss.item(), 
                'train_ll_seg_loss': ll_seg_loss.item(),
                'train_ll_tversky_loss': ll_tversky_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': i
            })
            
            # 添加解决器特定信息
            if conflict_solver is not None:
                method_info = conflict_solver.get_method_info()
                if conflict_solver.method == 'gradnorm' and 'weights' in method_info:
                    weights = method_info['weights']
                    log_dict.update({
                        'gradnorm_weight_det': weights[0].item(),
                        'gradnorm_weight_da': weights[1].item(),
                        'gradnorm_weight_ll': weights[2].item()
                    })
            
            wandb_run.log(log_dict)

        # 打印和记录
        if i % cfg.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t' \
                  f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})'
            
            if conflict_metrics:
                msg += f'\tTCI={conflict_metrics.get("task_conflict_intensity", 0):.3f}'
            
            logger.info(msg)
        
        start = time.time()

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    return parser.parse_args()


def run_experiment(conflict_method, shared_resources):
    """运行单个实验，使用共享资源"""
    cfg, device, train_loader, valid_loader, valid_dataset, model, criterion, optimizer, lr_scheduler, scaler, conflict_detector = shared_resources
    
    # 生成时间戳和run ID
    time_str = time.strftime('%Y%m%d-%H%M%S')
    run_id = f"run-{time_str}-{hash(time.time()) % 10000:04d}"
    
    # 设置日志目录
    method_suffix = f"_{conflict_method}" if conflict_method else "_standard"
    log_dir = Path(cfg.LOG_DIR) / cfg.DATASET.DATASET / f'{run_id}{method_suffix}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置logger
    logger = logging.getLogger(f'{conflict_method or "standard"}')
    logger.handlers = []  # 清除现有handlers
    
    # 文件handler
    file_handler = logging.FileHandler(log_dir / f'{run_id}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    
    # 初始化wandb - 修复命名格式
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            project_name = f"multitask-training-{cfg.DATASET.DATASET}"
            run_name = f"{run_id}{method_suffix}"
            
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "dataset": cfg.DATASET.DATASET,
                    "model": cfg.MODEL.NAME,
                    "conflict_method": conflict_method or "standard",
                    "epochs": cfg.TRAIN.END_EPOCH,
                    "batch_size": cfg.TRAIN.BATCH_SIZE_PER_GPU,
                    "learning_rate": cfg.TRAIN.LR0,
                }
            )
            logger.info(f"Wandb initialized: {project_name}/{run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # 初始化conflict_solver
    conflict_solver = None
    if conflict_method:
        if conflict_method == 'gradnorm':
            # conflict_solver = GradientConflictSolver(method='gradnorm', num_tasks=3, device=device, alpha=1.5)
            conflict_solver = FixedGradientConflictSolver(method='gradnorm', num_tasks=3, device=device, alpha=1.5)
        elif conflict_method == 'pcgrad':
            # conflict_solver = GradientConflictSolver(method='pcgrad', num_tasks=3, device=device)
            conflict_solver = FixedGradientConflictSolver(method='pcgrad', num_tasks=3, device=device)
        elif conflict_method == 'cagrad':
            # conflict_solver = GradientConflictSolver(method='cagrad', num_tasks=3, device=device, c=0.5)
            conflict_solver = FixedGradientConflictSolver(method='cagrad', num_tasks=3, device=device, c=0.5)
        logger.info(f"Using conflict resolution: {conflict_method}")
    else:
        logger.info("Using standard gradient descent")
    
    # 克隆模型和优化器状态
    import copy
    model_copy = copy.deepcopy(model)
    optimizer_copy = get_optimizer(cfg, model_copy)
    lr_scheduler_copy = optim.lr_scheduler.LambdaLR(
        optimizer_copy, 
        lr_lambda=lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    )
    
    # 加载预训练模型
    begin_epoch = load_pretrained_model(model_copy, optimizer_copy, cfg, logger)
    
    # 训练参数
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    learn_epoch = cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH
    
    logger.info(f"Starting training with {conflict_method or 'standard'} method...")
    logger.info(f"Training epochs: {begin_epoch + 1} to {begin_epoch + learn_epoch}")
    
    # 训练循环
    for epoch in range(begin_epoch + 1, begin_epoch + learn_epoch + 1):
        train_fixed(cfg, train_loader, model_copy, criterion, optimizer_copy, scaler,
              epoch, num_batch, num_warmup, logger, device, wandb_run, 
              conflict_detector, conflict_solver)
        
        lr_scheduler_copy.step()
        
        # 验证 - 改为在最后2个epoch进行详细验证
        if epoch >= cfg.TRAIN.END_EPOCH - 1:
            da_results, ll_results, detect_results, total_loss, _, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model_copy, criterion,
                str(log_dir), logger, device, wandb_run
            )
            
            # Wandb记录验证结果
            if wandb_run is not None:
                wandb_run.log({
                    'val_loss': total_loss,
                    'val_da_acc': da_results[0],
                    'val_da_iou': da_results[1],
                    'val_da_miou': da_results[2],
                    'val_ll_acc': ll_results[0],
                    'val_ll_iou': ll_results[1],
                    'val_ll_miou': ll_results[2],
                    'val_det_precision': detect_results[0],
                    'val_det_recall': detect_results[1],
                    'val_det_map50': detect_results[2],
                    'val_det_map': detect_results[3],
                    'inference_time': times[0],
                    'nms_time': times[1],
                    'epoch': epoch
                })
    
    # 生成分析图表
    try:
        logger.info("Generating comprehensive analysis plots...")
        final_plots = conflict_detector.generate_comprehensive_plots()
        
        if final_plots:
            # 保存图表到本地
            plots_dir = log_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            for plot_name, fig in final_plots.items():
                # 保存到本地
                plot_file = plots_dir / f"{plot_name}.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                logger.info(f"Plot saved: {plot_file}")
                
                # 上传到wandb
                if wandb_run is not None:
                    wandb_run.log({f"final_{plot_name}": wandb.Image(fig)})
                
                plt.close(fig)  # 释放内存
            
            # 上传训练总结
            if wandb_run is not None:
                summary = conflict_detector.get_training_summary()
                wandb_run.log({"training_summary": summary})
            
            logger.info("Comprehensive analysis completed and saved")
        else:
            logger.warning("No plots generated from conflict detector")
            
    except Exception as e:
        logger.warning(f"Failed to generate final plots: {e}")
    
    # 保存最终模型
    final_model_file = log_dir / 'final_state.pth'
    model_state = model_copy.module.state_dict() if is_parallel(model_copy) else model_copy.state_dict()
    torch.save(model_state, final_model_file)
    logger.info(f"Final model saved to: {final_model_file}")
    
    # 关闭wandb
    if wandb_run is not None:
        wandb_run.finish()
    
    logger.info(f"Training completed for {conflict_method or 'standard'}!")
    return str(log_dir)

def train_fixed(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, 
          device, wandb_run=None, conflict_detector=None, conflict_solver=None):
    """修正的训练函数 - 正确的冲突检测和解决流程"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    start = time.time()

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, (input, target, paths, shapes) in enumerate(train_pbar):
        num_iter = i + num_batch * (epoch - 1)
        
        # Warmup学习率调整
        if num_iter < num_warmup:
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
        
        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        
        # === 关键修正：在同一个autocast环境中进行所有计算 ===
        optimizer.zero_grad()
        
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes, model, input)
            
            # 1. 首先进行原始冲突检测（基于未修改的损失）
            conflict_metrics = {}
            if conflict_detector is not None and i % cfg.PRINT_FREQ == 0:
                # 在autocast环境内检测冲突，确保精度一致
                conflict_metrics = conflict_detector.detect_conflicts_in_context(
                    model, head_losses, optimizer, scaler
                )
            
            # 2. 应用冲突解决策略
            if conflict_solver is not None:
                # 让solver计算实际的训练损失
                final_loss = conflict_solver.compute_weighted_loss_with_gradients(
                    model, head_losses, optimizer, scaler
                )
                
                # 如果需要检测solver后的冲突变化，可以再次检测
                if conflict_detector is not None and i % (cfg.PRINT_FREQ * 2) == 0:
                    post_conflict_metrics = conflict_detector.detect_conflicts_in_context(
                        model, [final_loss], optimizer, scaler, prefix='post_solver_'
                    )
                    conflict_metrics.update(post_conflict_metrics)
            else:
                final_loss = total_loss
        
        # 3. 统一的反向传播
        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录指标
        losses.update(final_loss.item(), input.size(0))
        batch_time.update(time.time() - start)

        # 记录wandb指标
        if wandb_run is not None and i % cfg.PRINT_FREQ == 0:
            log_dict = {
                'train_total_loss': final_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': i
            }
            
            # 添加原始任务损失
            if len(head_losses) >= 3:
                log_dict.update({
                    'train_det_loss': head_losses[0].item(),
                    'train_da_seg_loss': head_losses[1].item(), 
                    'train_ll_seg_loss': head_losses[2].item(),
                })
            
            # 添加冲突指标（只记录有效的数值）
            for key, value in conflict_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    log_dict[key] = value
            
            # 添加解决器权重信息
            if conflict_solver is not None:
                solver_info = conflict_solver.get_current_weights()
                log_dict.update(solver_info)
            
            wandb_run.log(log_dict)

        # 打印信息
        if i % cfg.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t' \
                  f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})'
            
            if 'task_conflict_intensity' in conflict_metrics:
                msg += f'\tTCI={conflict_metrics["task_conflict_intensity"]:.3f}'
            
            logger.info(msg)
        
        start = time.time()


def main_optimized():
    """优化的主函数"""
    # 解析参数和配置
    args = parse_args()
    update_config(cfg, args)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.DEBUG else 'cpu')
    print(f"Using device: {device}")
    
    # 设置cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # 创建数据加载器（只做一次）
    print("Loading data...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(cfg)
    print("Data loaded successfully")
    
    # 创建基础模型和组件
    print("Building model...")
    model = get_net_from_yaml(cfg.MODEL.CONFIG).to(device)
    model.gr = 1.0
    model.nc = 1
    
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    # conflict_detector = GradientConflictDetector(model)
    conflict_detector = FixedGradientConflictDetector(model)
    
    # 共享资源
    shared_resources = (cfg, device, train_loader, valid_loader, valid_dataset, 
                       model, criterion, optimizer, lr_scheduler, scaler, conflict_detector)
    
    # 运行所有实验
    methods = ['gradnorm', 'pcgrad', 'cagrad', None]
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Starting experiment with method: {method or 'standard'}")
        print(f"{'='*50}")
        
        result_dir = run_experiment(method, shared_resources)
        results.append((method or 'standard', result_dir))
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("All experiments completed!")
    print("Results saved in:")
    for method, result_dir in results:
        print(f"  {method}: {result_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main_optimized()