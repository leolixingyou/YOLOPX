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

# 添加wandb支持
import wandb
WANDB_AVAILABLE = True


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from gradient_detect import GradientConflictDetector
from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.core.loss import get_loss
from lib.models import get_net_from_yaml
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer, save_checkpoint

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xywh2xyxy, box_iou, ap_per_class
from lib.utils.utils import time_synchronized


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


def setup_logging(cfg):
    """设置日志和设备"""
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.DEBUG else 'cpu')
    
    # 日志目录设置 - 直接使用 LOG_DIR/DATASET.DATASET
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_dir = Path(cfg.LOG_DIR) / cfg.DATASET.DATASET / (f'train_{time_str}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置本地日志
    log_file = log_dir / f'train_{time_str}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # 初始化wandb
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=f"multitask-training-{cfg.DATASET.DATASET}",
                name=f"train_{time_str}",
                config={
                    "dataset": cfg.DATASET.DATASET,
                    "model": cfg.MODEL.NAME,
                    "epochs": cfg.TRAIN.END_EPOCH, 
                    "batch_size": cfg.TRAIN.BATCH_SIZE_PER_GPU,
                    "learning_rate": cfg.TRAIN.LR0,
                    "optimizer": cfg.TRAIN.OPTIMIZER,
                }
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None
    
    return device, logger, str(log_dir), wandb_run


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
    """验证函数"""
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
    
    # 返回结果
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, None, t

def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, device, wandb_run=None, conflict_detector=None):
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
        
        # Warmup学习率调整 (保持原有逻辑)
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
        
        # 梯度冲突检测 - 只记录指定指标
        if conflict_detector is not None:
            metrics = conflict_detector.detect_all_metrics(head_losses)
            if wandb_run is not None and i % cfg.PRINT_FREQ == 0:
                # 只记录需要的指标
                wandb_metrics = {}
                required_metrics = ['task_conflict_intensity', 'gradient_conflict_rate', 
                                  'directional_conflict', 'magnitude_conflict',
                                  'det_ll_cosine', 'det_da_cosine', 'da_ll_cosine']
                
                for key in required_metrics:
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        wandb_metrics[key] = metrics[key]
                
                wandb_metrics.update({'epoch': epoch, 'batch': i})
                wandb_run.log(wandb_metrics)
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 记录指标
        losses.update(total_loss.item(), input.size(0))
        batch_time.update(time.time() - start)
        
        # 打印和记录
        if i % cfg.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' \
                  f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  f'Speed {input.size(0)/batch_time.val:.1f} samples/s\t' \
                  f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  f'Loss {losses.val:.5f} ({losses.avg:.5f})'
            
            logger.info(msg)
            
            # Wandb记录
            if wandb_run is not None:
                det_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, _ = head_losses
                wandb_run.log({
                    'train_total_loss': total_loss.item(),
                    'train_det_loss': det_loss.item(),
                    'train_da_seg_loss': da_seg_loss.item(), 
                    'train_ll_seg_loss': ll_seg_loss.item(),
                    'train_ll_tversky_loss': ll_tversky_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'batch': i
                })
        
        start = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    return parser.parse_args()


def main():
    # 解析参数和配置
    args = parse_args()
    update_config(cfg, args)


    # 设置日志和wandb
    device, logger, output_dir, wandb_run = setup_logging(cfg)
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # 设置cudnn
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # 创建模型
    logger.info("Building model...")
    model = get_net_from_yaml(cfg.MODEL.CONFIG).to(device)

    model.gr = 1.0
    model.nc = 1
    
    # 创建数据加载器
    logger.info("Loading data...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(cfg)
    logger.info("Data loaded successfully")

    # 损失函数和优化器
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    
    # 学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 加载预训练模型
    begin_epoch = load_pretrained_model(model, optimizer, cfg, logger)

    # 训练设置
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    conflict_detector = GradientConflictDetector(model)
    logger.info("Starting training...")
    learn_epoch = cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH
    
    # 训练循环
    for epoch in range(begin_epoch + 1, begin_epoch + learn_epoch + 1):
        # 训练一个epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, logger, device, wandb_run, conflict_detector)
        
        lr_scheduler.step()
        
        # 验证和保存
        if epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH:
            # 验证
            da_results, ll_results, detect_results, total_loss, _, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                output_dir, logger, device, wandb_run
            )
            
            # 记录结果
            msg = (f'Epoch: [{epoch}] Loss({total_loss:.3f})\n'
                   f'Driving area Segment: Acc({da_results[0]:.3f}) IOU({da_results[1]:.3f}) mIOU({da_results[2]:.3f})\n'
                   f'Lane line Segment: Acc({ll_results[0]:.3f}) IOU({ll_results[1]:.3f}) mIOU({ll_results[2]:.3f})\n'
                   f'Detect: P({detect_results[0]:.3f}) R({detect_results[1]:.3f}) mAP@0.5({detect_results[2]:.3f}) mAP@0.5:0.95({detect_results[3]:.3f})\n'
                   f'Time: inference({times[0]:.4f}s/frame) nms({times[1]:.4f}s/frame)')
            logger.info(msg)
            
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
            
            # 保存模型
            save_checkpoint(
                epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                output_dir=output_dir, filename=f'epoch-{epoch}.pth'
            )
    

    try:
        final_plots = conflict_detector.generate_comprehensive_plots()
        
        if wandb_run is not None and final_plots:
            # 上传最终的综合分析图表
            for plot_name, fig in final_plots.items():
                wandb_run.log({f"final_{plot_name}": wandb.Image(fig)})
                plt.close(fig)  # 释放内存
            
            # 上传训练总结
            summary = conflict_detector.get_training_summary()
            wandb_run.log({"training_summary": summary})
        
        logger.info("Comprehensive analysis completed and uploaded to wandb")
        
    except Exception as e:
        logger.warning(f"Failed to generate final plots: {e}")


    # 保存最终模型
    final_model_file = os.path.join(output_dir, 'final_state.pth')
    logger.info(f"Saving final model to {final_model_file}")
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    torch.save(model_state, final_model_file)
    
    # 关闭wandb
    if wandb_run is not None:
        wandb_run.finish()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()