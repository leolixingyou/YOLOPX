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

# Add wandb support
import wandb
WANDB_AVAILABLE = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Assume these lib directory files exist and are correctly configured
from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.core.loss import get_loss
from lib.models import get_net_from_yaml # Used to load the model, might need modification for TAG
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer

# Corrected import paths for custom modules
from xy_conflict_detect_gemini import FixedGradientConflictDetector
from xy_conflict_solver_gemini import FixedGradientConflictSolver
from mdo_optimizer import MDO_Optimizer

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xywh2xyxy, box_iou, ap_per_class
from lib.utils.utils import time_synchronized

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Creates data loaders"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    # Train dataset
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
    
    # Validation dataset
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
    """Loads a pretrained model"""
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    if os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"Loading pretrained model: {cfg.MODEL.PRETRAINED}")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED)
        begin_epoch = checkpoint['epoch']
        
        # Compatibility for loading parallel models
        state_dict = checkpoint['state_dict']
        if isinstance(model, torch.nn.DataParallel) and not 'module.' in list(state_dict.keys())[0]:
            # Loading from non-parallel model to parallel model
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        elif not isinstance(model, torch.nn.DataParallel) and 'module.' in list(state_dict.keys())[0]:
            # Loading from parallel model to non-parallel model
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            # Direct loading
            model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return begin_epoch


def validate(epoch, config, val_loader, val_dataset, model, criterion, output_dir, logger, device, wandb_run=None):
    """Validation function - enhanced metric logging"""
    max_stride = 32
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    
    # Ensure nc and num_seg_class are correctly set after model loading
    detection_classes = getattr(model, 'nc', 1) 
    da_seg_classes = getattr(config, 'num_seg_class', 2) 
    ll_seg_classes = 2 

    confusion_matrix = ConfusionMatrix(nc=detection_classes)
    da_metric = SegmentationMetric(da_seg_classes)
    ll_metric = SegmentationMetric(ll_seg_classes)
    
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
            
            # --- FIX for shapes parsing ---
            pad_h, pad_w = 0, 0 
            if len(shapes) > 0 and len(shapes[0]) > 1:
                padding_info = shapes[0][1] 
                
                if isinstance(padding_info, (tuple, list)) and len(padding_info) == 2:
                    if isinstance(padding_info[0], (list, torch.Tensor)) and isinstance(padding_info[1], (list, torch.Tensor)):
                        pad_h = int(padding_info[0][0]) 
                        pad_w = int(padding_info[1][0])
                    elif isinstance(padding_info[0], (int, float)) and isinstance(padding_info[1], (int, float)):
                        pad_h = int(padding_info[0])
                        pad_w = int(padding_info[1])
                    else:
                        logger.warning(f"Unexpected inner padding_info format: {padding_info}. Defaulting to 0 padding.")
                        pad_h, pad_w = 0, 0
                else:
                    logger.warning(f"Unexpected shapes[0][1] format: {shapes[0][1]}. Defaulting to 0 padding.")
                    pad_h, pad_w = 0, 0
            else:
                logger.warning("Shapes list is empty or malformed. Defaulting to 0 padding.")
                pad_h, pad_w = 0, 0
            # --- END FIX for shapes parsing ---

            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0: 
                T_inf.update(t_inf/img.size(0), img.size(0))
            
            inf_out, train_out = det_out
            
            # Driving Area Segmentation Evaluation
            if da_seg_out is not None and target[1] is not None:
                _, da_predict = torch.max(da_seg_out, 1)
                _, da_gt = torch.max(target[1], 1)
                if height > 2 * pad_h and width > 2 * pad_w: 
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
            else:
                pass
            
            # Lane Line Segmentation Evaluation
            if ll_seg_out is not None and target[2] is not None:
                _, ll_predict = torch.max(ll_seg_out, 1)
                _, ll_gt = torch.max(target[2], 1)
                if height > 2 * pad_h and width > 2 * pad_w: 
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
            else:
                pass
            
            # Calculate Total Loss and Head Losses
            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model, img)
            losses.update(total_loss.item(), img.size(0))
            
            # NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0), img.size(0))
            
            # Detection Evaluation Stats
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
                confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], xywh2xyxy(labels[:, 1:5])), 1))

                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    
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
    
    # Calculate statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        if hasattr(val_dataset, 'names') and len(val_dataset.names) > 0:
            p, r, ap, f1, ap_class = ap_per_class(*stats, names=val_dataset.names)
        else: 
            p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0
    
    model.float() 
    
    # Log detailed validation results
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
    
    # Return results
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, None, t

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--eval_interval', type=int, default=1, help='Epoch interval for validation')
    return parser.parse_args()

# Global dictionary to store training metrics for comparison plots
all_experiment_metrics = {}

def run_experiment(conflict_method, shared_resources):
    """Runs a single experiment using shared resources"""
    cfg, device, train_loader, valid_loader, valid_dataset, model_proto, criterion_proto, _, _, _, conflict_detector_proto = shared_resources

    time_str = time.strftime('%Y%m%d-%H%M%S')
    run_id = f"run-{time_str}-{hash(time.time()) % 10000:04d}"
    
    method_suffix = f"_{conflict_method}" if conflict_method else "_original"
    log_dir = Path(cfg.LOG_DIR) / cfg.DATASET.DATASET / f'{run_id}{method_suffix}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f'{conflict_method or "original"}')
    logger.handlers = []
    logger.propagate = False
    
    file_handler = logging.FileHandler(log_dir / f'{run_id}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.INFO)
    
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            project_name = f"multitask-training-{cfg.DATASET.DATASET}_refactor" 
            run_name = f"{run_id}{method_suffix}"
            
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "dataset": cfg.DATASET.DATASET,
                    "model": cfg.MODEL.NAME,
                    "conflict_method": conflict_method or "original",
                    "epochs": cfg.TRAIN.END_EPOCH,
                    "batch_size": cfg.TRAIN.BATCH_SIZE_PER_GPU,
                    "learning_rate": cfg.TRAIN.LR0,
                },
                reinit=True
            )
            logger.info(f"Wandb initialized: {project_name}/{run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Wandb will be disabled for this run.")
            wandb_run = None
    
    import copy
    model_copy = copy.deepcopy(model_proto).to(device)
    
    if isinstance(model_copy, torch.nn.DataParallel):
        model_copy = model_copy.module 
    
    model_copy = model_copy.to(device) 
    
    if str(device).startswith('cuda') and torch.cuda.device_count() > 0:
        torch.cuda.set_device(0) 
        model_copy = model_copy.cuda(0)

    optimizer_copy = get_optimizer(cfg, model_copy)
    lr_scheduler_copy = optim.lr_scheduler.LambdaLR(
        optimizer_copy, 
        lr_lambda=lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    )
    scaler_copy = amp.GradScaler(enabled=device.type != 'cpu')
    conflict_detector_exp = FixedGradientConflictDetector(model_copy)

    solver_map = {
        'gradnorm': FixedGradientConflictSolver(method='gradnorm', num_tasks=3, device=device, alpha=1.5, update_freq=20),
        'pcgrad': FixedGradientConflictSolver(method='pcgrad', num_tasks=3, device=device),
        'cagrad': FixedGradientConflictSolver(method='cagrad', num_tasks=3, device=device, c=0.5),
        'mdo': MDO_Optimizer(model_copy, num_tasks=3, device=device, update_freq=20), 
        'tag': FixedGradientConflictSolver(method='tag', num_tasks=3, device=device, update_freq=20) 
    }
    conflict_solver_exp = solver_map.get(conflict_method)
    
    begin_epoch = load_pretrained_model(model_copy, optimizer_copy, cfg, logger)
    
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    
    learn_epoch = cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH
    
    logger.info(f"Starting training with {conflict_method or 'original'} method...")
    logger.info(f"Training epochs: {begin_epoch + 1} to {begin_epoch + learn_epoch}")
    
    # Store metrics for this experiment
    metrics_for_plotting = {
        'total_loss': [],
        'task_conflict_intensity': [],
        'epoch_num': []
    }

    for epoch in range(begin_epoch + 1, begin_epoch + learn_epoch + 1):
        epoch_metrics = train_fixed(cfg, train_loader, model_copy, criterion_proto, optimizer_copy, scaler_copy,
              epoch, num_batch, num_warmup, logger, device, wandb_run, 
              conflict_detector_exp, conflict_solver_exp, max_epch=begin_epoch + learn_epoch + 1)
        
        lr_scheduler_copy.step()
        
        # Collect metrics for plotting
        metrics_for_plotting['total_loss'].append(epoch_metrics.get('train_total_loss_avg', float('nan')))
        metrics_for_plotting['task_conflict_intensity'].append(epoch_metrics.get('task_conflict_intensity_avg', float('nan')))
        metrics_for_plotting['epoch_num'].append(epoch)

        if epoch >= cfg.TRAIN.END_EPOCH - 1: # Perform validation only at the end
            da_results, ll_results, detect_results, total_loss, _, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model_copy, criterion_proto,
                str(log_dir), logger, device, wandb_run
            )
            
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
    
    try:
        logger.info("Generating comprehensive analysis plots...")
        final_plots = conflict_detector_exp.generate_comprehensive_plots()
        
        if final_plots:
            plots_dir = log_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            for plot_name, fig in final_plots.items():
                plot_file = plots_dir / f"{plot_name}.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                logger.info(f"Plot saved: {plot_file}")
                
                if wandb_run is not None:
                    wandb_run.log({f"final_{plot_name}": wandb.Image(fig)})
                
                plt.close(fig)
            
            if wandb_run is not None:
                summary = conflict_detector_exp.get_training_summary()
                wandb_run.log({"training_summary": summary})
            
            logger.info("Comprehensive analysis completed and saved")
        else:
            logger.warning("No plots generated from conflict detector.")
            
    except Exception as e:
        logger.warning(f"Failed to generate final plots: {e}")
    
    final_model_file = log_dir / 'final_state.pth'
    model_state = model_copy.module.state_dict() if is_parallel(model_copy) else model_copy.state_dict()
    torch.save(model_state, final_model_file)
    logger.info(f"Final model saved to: {final_model_file}")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    logger.info(f"Training completed for {conflict_method or 'original'}!")
    
    # Return collected metrics for main_optimized to plot
    return metrics_for_plotting

def train_fixed(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup, logger, 
          device, wandb_run=None, conflict_detector=None, conflict_solver=None, max_epch=0):
    """Fixed training function - correct conflict detection and resolution process"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Metrics to return for epoch-level plotting
    epoch_metrics_accumulator = {
        'train_total_loss': AverageMeter(),
        'task_conflict_intensity': AverageMeter(),
        'train_det_loss': AverageMeter(),
        'train_da_seg_loss': AverageMeter(),
        'train_ll_seg_loss': AverageMeter(),
    }

    model.train()
    start = time.time()

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} / {max_epch}', 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    model_to_inspect = model.module if is_parallel(model) else model
    
    shared_params = [
        p for n, p in model_to_inspect.named_parameters() 
        if p.requires_grad and 
           not any(head_name in n for head_name in ['head', 'det_head', 'da_seg_head', 'll_seg_head', 'seg_head']) 
    ]
    
    if not shared_params:
        logger.warning("No shared parameters found based on 'head' exclusion. This might affect gradient-based conflict resolution methods. Using all trainable parameters as shared.")
        shared_params = [p for p in model_to_inspect.parameters() if p.requires_grad]


    for i, (input, target, paths, shapes) in enumerate(train_pbar):
        num_iter = i + num_batch * (epoch - 1)
        
        # Warmup learning rate adjustment
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
        
        optimizer.zero_grad() 
        
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            
            train_out, da_seg_out, ll_seg_out = outputs[0], outputs[1], outputs[2]

            if da_seg_out is not None:
                da_seg_out = torch.clamp(da_seg_out, min=-1e8, max=1e8) 
            if ll_seg_out is not None:
                ll_seg_out = torch.clamp(ll_seg_out, min=-1e8, max=1e8) 

            total_loss_raw, head_losses_raw = None, None
            try:
                total_loss_raw, head_losses_raw = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model, input)
            except RuntimeError as e:
                if "k out of range" in str(e) or "Assertion `input_val >= zero && input_val <= one` failed" in str(e):
                    logger.warning(f"Caught RuntimeError in criterion: {e}. Attempting to handle by setting detection loss to 0 for this batch.")
                    
                    det_loss_val = torch.tensor(0.0, device=device, requires_grad=True) 
                    
                    head_losses_raw = [
                        det_loss_val,
                        head_losses_raw[1] if len(head_losses_raw) > 1 and head_losses_raw[1] is not None else torch.tensor(0.0, device=device, requires_grad=True), 
                        head_losses_raw[2] if len(head_losses_raw) > 2 and head_losses_raw[2] is not None else torch.tensor(0.0, device=device, requires_grad=True)  
                    ]
                    
                    total_loss_raw = sum(l for l in head_losses_raw if l is not None)
                    
                    if total_loss_raw.item() == 0: 
                        total_loss_raw = torch.tensor(1e-5, device=device, requires_grad=True) 
                    
                else:
                    raise e
            
            if total_loss_raw is None or head_losses_raw is None: # Safety check
                logger.warning("Loss computation failed for this batch. Skipping backward pass.")
                continue

            conflict_metrics = {}
            # Conflict detection should use the raw head losses to observe actual task behavior
            if conflict_detector is not None: 
                conflict_metrics = conflict_detector.detect_conflicts_in_context(
                    model, head_losses_raw, optimizer, scaler 
                )
            
            final_loss_for_backward = total_loss_raw 

            if conflict_solver is not None:
                final_loss_for_backward = conflict_solver.compute_weighted_loss_with_gradients(
                head_losses_raw, shared_params, scaler, num_iter # 传入 num_iter 作为 step_count
            )
            
            scaler.scale(final_loss_for_backward).backward()
            scaler.step(optimizer)
            scaler.update()
            
        losses.update(final_loss_for_backward.item(), input.size(0))
        epoch_metrics_accumulator['train_total_loss'].update(final_loss_for_backward.item(), input.size(0))
        
        if len(head_losses_raw) >= 3:
            if head_losses_raw[0] is not None: epoch_metrics_accumulator['train_det_loss'].update(head_losses_raw[0].item(), input.size(0))
            if head_losses_raw[1] is not None: epoch_metrics_accumulator['train_da_seg_loss'].update(head_losses_raw[1].item(), input.size(0))
            if head_losses_raw[2] is not None: epoch_metrics_accumulator['train_ll_seg_loss'].update(head_losses_raw[2].item(), input.size(0))

        if 'task_conflict_intensity' in conflict_metrics:
            epoch_metrics_accumulator['task_conflict_intensity'].update(conflict_metrics['task_conflict_intensity'], input.size(0))

        batch_time.update(time.time() - start)

        if wandb_run is not None :
            log_dict = {
                'train_total_loss': final_loss_for_backward.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'batch': i
            }
            
            if len(head_losses_raw) >= 3:
                log_dict.update({
                    'train_det_loss': head_losses_raw[0].item() if head_losses_raw[0] is not None else float('nan'),
                    'train_da_seg_loss': head_losses_raw[1].item() if head_losses_raw[1] is not None else float('nan'),
                    'train_ll_seg_loss': head_losses_raw[2].item() if head_losses_raw[2] is not None else float('nan'),
                })
            
            for key, value in conflict_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    log_dict[key] = value
            
            if conflict_solver is not None:
                solver_info = conflict_solver.get_current_weights()
                log_dict.update(solver_info)
            
            wandb_run.log(log_dict)

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
    
    # Return aggregated epoch metrics
    return {
        'train_total_loss_avg': epoch_metrics_accumulator['train_total_loss'].avg,
        'task_conflict_intensity_avg': epoch_metrics_accumulator['task_conflict_intensity'].avg if epoch_metrics_accumulator['task_conflict_intensity'].count > 0 else float('nan'),
        'train_det_loss_avg': epoch_metrics_accumulator['train_det_loss'].avg if epoch_metrics_accumulator['train_det_loss'].count > 0 else float('nan'),
        'train_da_seg_loss_avg': epoch_metrics_accumulator['train_da_seg_loss'].avg if epoch_metrics_accumulator['train_da_seg_loss'].count > 0 else float('nan'),
        'train_ll_seg_loss_avg': epoch_metrics_accumulator['train_ll_seg_loss'].avg if epoch_metrics_accumulator['train_ll_seg_loss'].count > 0 else float('nan'),
    }


def main_optimized():
    """Optimized main function"""
    args = parse_args()
    update_config(cfg, args)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.DEBUG else 'cpu')
    print(f"Using device: {device}")
    
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    print("Loading data...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(cfg)
    print("Data loaded successfully")
    
    print("Building model prototype...")
    model_proto = get_net_from_yaml(cfg.MODEL.CONFIG).to(device)
    model_proto.gr = 1.0 
    model_proto.nc = 1 

    if isinstance(model_proto, torch.nn.DataParallel):
        model_proto = model_proto.module 
    
    if str(device).startswith('cuda') and torch.cuda.device_count() > 0:
        model_proto = model_proto.cuda(0)
        device = torch.device('cuda:0') 
        print(f"Forcing single GPU usage on {device}")

    criterion_proto = get_loss(cfg, device, model_proto)
    conflict_detector_proto = FixedGradientConflictDetector(model_proto)

    shared_resources = (cfg, device, train_loader, valid_loader, valid_dataset, 
                       model_proto, criterion_proto, None, None, None, conflict_detector_proto)
    
    methods = ['gradnorm', 'pcgrad', 'cagrad', 'mdo', None] # None for original
    # methods = ['cagrad'] # None for original
    global all_experiment_metrics
    all_experiment_metrics = {} # Reset global metrics for each run of main_optimized
    
    for method in methods:
        method_name = method or 'original'
        print(f"\n{'='*50}")
        print(f"Starting experiment with method: {method_name}")
        print(f"{'='*50}")
        
        collected_metrics = run_experiment(method, shared_resources)
        all_experiment_metrics[method_name] = collected_metrics
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("All experiments completed!")




    metrics_to_plot = {
        'task_conflict_intensity': 'Task Conflict Intensity',
        'directional_conflict': 'Directional Conflict',
        'gradient_conflict_rate': 'Gradient Conflict Rate',
        'magnitude_conflict': 'Magnitude Conflict',
        'total_loss': 'Total Training Loss' # 使用 train_total_loss_avg
    }

    # 创建一个子目录来存放所有比较图
    comparison_plots_dir = Path(cfg.LOG_DIR) / cfg.DATASET.DATASET / 'comparisons'
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_title in metrics_to_plot.items():
        print(f"Generating comparative plot for: {metric_title}...")
        plt.figure(figsize=(12, 7))
        
        # 检查是否有任何方法为此指标提供了有效数据
        has_valid_data = False
        for method_name, metrics in all_experiment_metrics.items():
            # 对于 'total_loss'，实际键是 'train_total_loss_avg'
            # 对于其他冲突指标，实际键是 'task_conflict_intensity_avg' 等 (根据 FixedGradientConflictDetector)
            # 在 train_fixed 中，我们已经将它们映射到了 metrics_for_plotting
            
            # 确保这里获取的键与 train_fixed 返回的实际平均值键一致
            # 如果metric_key是'total_loss'，我们实际要取'train_total_loss'
            # 如果metric_key是'task_conflict_intensity'，我们实际要取'task_conflict_intensity'
            # ... 等等
            
            # 简化：直接使用 metric_key 作为字典中的键
            values = metrics.get(metric_key) # Attempt to get the averaged metric

            if values is None or not isinstance(values, list) or not any(~np.isnan(v) for v in values):
                print(f"No valid data for {metric_title} for method {method_name}. Skipping plot for this method on this metric.")
                continue
            
            has_valid_data = True
            epochs = metrics['epoch_num'] # Epochs should be consistent for all plots
            valid_indices = ~np.isnan(values)
            
            if np.any(valid_indices):
                plt.plot(np.array(epochs)[valid_indices], np.array(values)[valid_indices], label=method_name, linewidth=1.5)
            else:
                print(f"All values are NaN for {metric_title} for method {method_name}. Skipping plot for this method on this metric.")

        if has_valid_data:
            plt.title(f'{metric_title} Comparison Across Methods', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(metric_title, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
            
            plot_filename = comparison_plots_dir / f"MTL_comparison_{metric_key}.png"
            plt.savefig(plot_filename, dpi=300)
            print(f"Comparative plot saved to: {plot_filename}")
            
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    # Log the plot to the current WandB run (if it's the last one)
                    # For comprehensive logging across all methods, this might need a separate artifact upload or a summary run
                    wandb.log({f"comparative_plots/{metric_key}": wandb.Image(str(plot_filename))})
                    print(f"Comparative plot also logged to WandB: {metric_key}")
                except Exception as e:
                    print(f"Failed to log comparative plot to WandB for {metric_key}: {e}")
            plt.close() # Close the plot to free memory
        else:
            print(f"No valid data to generate comparative plot for {metric_title}.")

  

    print(f"{'='*50}")


if __name__ == '__main__':
    main_optimized()