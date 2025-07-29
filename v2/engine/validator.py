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

                # Visualization
                if i < 2: # Visualize first 2 batches
                    for si, (pred, path, shape) in enumerate(zip(output, paths, shapes)):
                        if si < 2: # Visualize first 2 images of the batch
                            self._create_and_log_visualizations(img[si], pred, target, si, da_seg_out, ll_seg_out, names, epoch, path, shape)
        
        # Log combined validation image grid
        self.wandb_logger.log_validation_images(epoch, self.local_logger)
        
        # Compute detection stats
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map50, map = p.mean(), r.mean(), ap[:, 0].mean(), ap.mean()
        
        # Log and print results
        if logger:
            logger.info(f'Validation Epoch {epoch}: Loss={losses.avg:.4f}, mAP50={map50:.4f}, DA_mIoU={da_miou.avg:.4f}, LL_mIoU={ll_miou.avg:.4f}')
        else:
            print(f'Validation Epoch {epoch}: Loss={losses.avg:.4f}, mAP50={map50:.4f}, DA_mIoU={da_miou.avg:.4f}, LL_mIoU={ll_miou.avg:.4f}')
        
        return {
            "val_loss": losses.avg,
            "map": map, "map50": map50, "precision": mp, "recall": mr,
            "da_miou": da_miou.avg, "da_iou": da_iou.avg, "da_acc": da_acc.avg,
            "ll_miou": ll_miou.avg, "ll_iou": ll_iou.avg, "ll_acc": ll_acc.avg,
        }

    def _update_seg_metrics(self, pred_seg, gt_seg, metric, acc, iou, miou, batch_size):
        """Helper to compute segmentation metrics for a single head."""
        pred = torch.argmax(pred_seg, 1)
        gt = torch.argmax(gt_seg, 1) if gt_seg.dim() == 4 and gt_seg.shape[1] > 1 else gt_seg
        
        metric.reset()
        metric.addBatch(pred.cpu(), gt.cpu())
        
        acc.update(metric.pixelAccuracy(), batch_size)
        iou_val = metric.IntersectionOverUnion()
        # Handle case where a class might be missing in the batch
        iou.update(iou_val[1] if isinstance(iou_val, (list, tuple, torch.Tensor)) and len(iou_val) > 1 else iou_val, batch_size)
        miou.update(metric.meanIntersectionOverUnion(), batch_size)

    def _create_and_log_visualizations(self, img_tensor, pred_det, targets, batch_idx, da_seg_out, ll_seg_out, names, epoch, path, shape):
        """Creates and caches a single visualization grid for one image."""
        try:
            img_np = _denormalize_img_tensor(img_tensor)
            
            # Create a 2x2 grid for visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
            plt.subplots_adjust(wspace=0.05, hspace=0.15)
            fig.suptitle(f'Epoch {epoch} - {Path(path).name}', fontsize=16)

            # 1. Original Image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')

            # 2. Detection
            det_img = self._visualize_detection(img_np.copy(), pred_det, targets[0], batch_idx, names, shape)
            axes[0, 1].imshow(det_img)
            axes[0, 1].set_title('Object Detection')
            axes[0, 1].axis('off')

            # 3. Drivable Area
            da_img = self._visualize_segmentation(img_np.copy(), da_seg_out[batch_idx], targets[1][batch_idx])
            axes[1, 0].imshow(da_img)
            axes[1, 0].set_title('Drivable Area Segmentation')
            axes[1, 0].axis('off')

            # 4. Lane Line
            ll_img = self._visualize_segmentation(img_np.copy(), ll_seg_out[batch_idx], targets[2][batch_idx])
            axes[1, 1].imshow(ll_img)
            axes[1, 1].set_title('Lane Line Segmentation')
            axes[1, 1].axis('off')

            # Cache the figure for later logging
            self.wandb_logger.cache_validation_image(fig, f"{Path(path).stem}")
            plt.close(fig)

        except Exception as e:
            print(f"Error creating visualization for {path}: {e}")
            plt.close('all') # Close any dangling figures

    def _visualize_detection(self, img_np, pred, gt_all, batch_idx, names, shape):
        # GT boxes
        gt_boxes_xyxy = []
        gt_cls = []
        gt_for_img = gt_all[gt_all[:, 0] == batch_idx]
        if len(gt_for_img) > 0:
            gt_boxes_xyxy = xywh2xyxy(gt_for_img[:, 2:6])
            scale_coords(img_np.shape[:2], gt_boxes_xyxy, shape[0], shape[1])
            gt_cls = gt_for_img[:, 1]
        
        # Pred boxes
        pred_boxes_xyxy = pred[:, :4] if len(pred) > 0 else []
        pred_cls = pred[:, 5] if len(pred) > 0 else []

        img_np = _draw_boxes(img_np, gt_boxes_xyxy, gt_cls, names, is_gt=True)
        img_np = _draw_boxes(img_np, pred_boxes_xyxy, pred_cls, names, is_gt=False)
        return img_np

    def _visualize_segmentation(self, img_np, pred_seg, gt_seg):
        pred_mask = torch.argmax(pred_seg, 0).cpu().numpy()
        gt_mask = torch.argmax(gt_seg, 0).cpu().numpy() if gt_seg.dim() == 3 else gt_seg.cpu().numpy()
        
        img_np = _overlay_mask(img_np, gt_mask, (255, 0, 0), alpha=0.4) # Red for GT
        img_np = _overlay_mask(img_np, pred_mask, (0, 255, 0), alpha=0.4) # Green for Pred
        return img_np
        
    def _process_batch(self, detections, labels):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 5]), true_class, conf, p, r, iou
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = torch.zeros(detections.shape[0], 5, device=self.device)
        # Corrected IoU threshold usage
        correct_idx = torch.where(iou > self.cfg.TEST.NMS_IOU_THRESHOLD)
        
        # Simplified matching logic
        if correct_idx[0].shape[0]:
            matches = torch.cat((torch.stack(correct_idx, 1), iou[correct_idx].unsqueeze(1)), 1)
            if correct_idx[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort(descending=True)]
                matches = matches[torch.unique(matches[:, 1], return_inverse=False)[1]]
                matches = matches[torch.unique(matches[:, 0], return_inverse=False)[1]]
            
            correct[matches[:, 1].long()] = matches[:, 2].unsqueeze(1) > self.cfg.TEST.NMS_IOU_THRESHOLD
        return correct
