"""
The core validation engine for the YOLOPX project.
"""
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import sys

# --- Refactored Imports ---
from utils.metrics import ConfusionMatrix, SegmentationMetric, ap_per_class
from utils.general import non_max_suppression, xywh2xyxy, scale_coords, time_synchronized, AverageMeter, box_iou
from utils.plots import _denormalize_img_tensor, _overlay_mask, _draw_boxes, _add_detection_legend, _add_segmentation_legend, _get_class_colors

class Validator:
    """
    The Validator class encapsulates the logic for evaluating the model's performance
    on the validation dataset.
    """
    def __init__(self, cfg, val_loader, val_dataset, model, criterion, local_logger, wandb_logger):
        self.cfg = cfg
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.model = model
        self.criterion = criterion
        self.local_logger = local_logger
        self.wandb_logger = wandb_logger
        self.device = next(model.parameters()).device

    def validate(self, epoch):
        """Runs a full validation cycle."""
        logger = self.local_logger.get_logger() if hasattr(self.local_logger, 'get_logger') else None
        self.model.eval()
        
        # Initialize metrics
        da_metric = SegmentationMetric(self.cfg.DATASET.NUM_SEG_CLASS)
        ll_metric = SegmentationMetric(2) # Lane line is always binary
        losses, da_acc, da_iou, da_miou, ll_acc, ll_iou, ll_miou, t_inf, t_nms = [AverageMeter() for _ in range(9)]
        
        stats = []
        # Simple fallback names for classes
        names = {0: 'object'}

        pbar_desc = f'Validating Epoch {epoch}'
        val_pbar = tqdm(self.val_loader, desc=pbar_desc, file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}')

        with torch.no_grad():
            for i, (img, det_labels, seg_labels, lane_labels, paths) in enumerate(val_pbar):
                target = [det_labels, seg_labels, lane_labels]
                shapes = None
                img = img.to(self.device, non_blocking=True)
                target = [t.to(self.device) for t in target]
                nb, _, height, width = img.shape

                # Inference
                t = time_synchronized()
                det_out, da_seg_out, ll_seg_out = self.model(img)
                t_inf.update((time_synchronized() - t) / nb, nb)

                # Loss
                total_loss, _ = self.criterion((det_out, da_seg_out, ll_seg_out), target, shapes, self.model, img)
                losses.update(total_loss.item(), nb)

                # NMS
                t = time_synchronized()
                # In eval mode, model builder might return (processed_detections, raw_features)
                det_for_nms = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
                output = non_max_suppression(det_for_nms, conf_thres=self.cfg.TEST.NMS_CONF_THRESHOLD, iou_thres=self.cfg.TEST.NMS_IOU_THRESHOLD)
                t_nms.update((time_synchronized() - t) / nb, nb)

                # Segmentation Metrics
                self._update_seg_metrics(da_seg_out, target[1], da_metric, da_acc, da_iou, da_miou, nb)
                self._update_seg_metrics(ll_seg_out, target[2], ll_metric, ll_acc, ll_iou, ll_miou, nb)

                # Detection Metrics
                for si, pred in enumerate(output):
                    labels = target[0][target[0][:, 0] == si, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []  # target class
                    
                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, 7, device=self.device), torch.Tensor(), torch.Tensor(), tcls))
                        continue

                    # Correct predictions
                    predn = pred.clone()
                    scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = self._process_batch(predn, labelsn)
                    else:
                        correct = torch.zeros(pred.shape[0], 5, device=self.device)
                    stats.append((correct, pred[:, 4], pred[:, 5], tcls))

                # Visualization - Cache first image of the first 2 batches
                if i < 2 and si == 0:
                    gt_boxes_for_log = []
                    gt_labels_for_img = target[0][target[0][:, 0] == si]
                    if len(gt_labels_for_img) > 0:
                        gt_boxes_for_log = xywh2xyxy(gt_labels_for_img[:, 2:6])
                        scale_coords(img[si].shape[1:], gt_boxes_for_log, shapes[si][0], shapes[si][1])
                        gt_boxes_for_log = torch.cat((gt_labels_for_img[:, 1:2], gt_boxes_for_log), 1)

                    predictions_log = {
                        'det': predn.cpu().numpy(),
                        'da_seg': torch.argmax(da_seg_out[si], 0).cpu().numpy(),
                        'll_seg': torch.argmax(ll_seg_out[si], 0).cpu().numpy()
                    }
                    ground_truths_log = {
                        'det': gt_boxes_for_log.cpu().numpy() if len(gt_boxes_for_log) > 0 else [],
                        'da_seg': target[1][si].cpu().numpy(),
                        'll_seg': target[2][si].cpu().numpy()
                    }
                    
                    self.wandb_logger.cache_multitask_image(
                        img[si], 
                        predictions_log, 
                        ground_truths_log, 
                        names, 
                        f"Epoch_{epoch}_Batch_{i}"
                    )
        
        # Log all cached images at the end of the validation loop
        self.wandb_logger.log_validation_images(epoch, self.local_logger)
        
        # Compute detection stats
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=self.local_logger.log_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        else:
            mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        
        # Log and print results
        msg = f'Validation Epoch {epoch}: Loss={losses.avg:.4f}, mAP50={map50:.4f}, DA_mIoU={da_miou.avg:.4f}, LL_mIoU={ll_miou.avg:.4f}'
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        return {
            "val_loss": losses.avg,
            "map": map, "map50": map50, "precision": mp, "recall": mr,
            "da_miou": da_miou.avg, "da_iou": da_iou.avg, "da_acc": da_acc.avg,
            "ll_miou": ll_miou.avg, "ll_iou": ll_iou.avg, "ll_acc": ll_acc.avg,
        }

    def _update_seg_metrics(self, pred_seg, gt_seg, metric, acc, iou, miou, batch_size):
        """Helper to compute segmentation metrics for a single head."""
        pred = torch.argmax(pred_seg, 1)
        # Ensure gt_seg is single-channel
        gt = gt_seg if gt_seg.dim() == 3 else torch.argmax(gt_seg, 1)
        
        metric.reset()
        metric.addBatch(pred.cpu(), gt.cpu())
        
        acc.update(metric.pixelAccuracy(), batch_size)
        iou_val = metric.IntersectionOverUnion()
        iou.update(iou_val[1] if isinstance(iou_val, (list, tuple, np.ndarray)) and len(iou_val) > 1 else iou_val, batch_size)
        miou.update(metric.meanIntersectionOverUnion(), batch_size)
        
    def _process_batch(self, detections, labels):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU thresholds
        """
        correct = torch.zeros(detections.shape[0], 10, device=self.device, dtype=torch.bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(iou > 0.5)  # IoU > 0.5
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            t_iou = torch.linspace(0.5, 0.95, 10, device=self.device)
            for m in matches:
                if int(detections[int(m[1]), 5]) == int(labels[int(m[0]), 0]):
                    correct[int(m[1])] = torch.tensor(m[2], device=self.device) > t_iou
        return correct
