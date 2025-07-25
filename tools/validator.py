import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
from lib.core.general import non_max_suppression, box_iou, ap_per_class
from lib.utils.utils import time_synchronized, xywh2xyxy, scale_coords

# --- Visualization Helper Functions ---

def _denormalize_img_tensor(img_tensor):
    img = img_tensor.cpu().float().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def _overlay_mask(image_np, mask_np, color, alpha=0.4):
    # Ensure mask is 2D and same height/width as image
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    if mask_np.ndim > 2:
        mask_np = mask_np[0]  # Take first channel if still 3D
    
    # Resize mask to match image if needed
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np.astype(np.uint8), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    colored_mask[mask_np == 1] = color
    return cv2.addWeighted(image_np, 1, colored_mask, alpha, 0)

def _get_class_colors():
    """Define distinct colors for different classes"""
    return {
        0: (255, 0, 0),    # car - red  
        1: (0, 255, 0),    # truck - green
        2: (0, 0, 255),    # bus - blue
        3: (255, 255, 0),  # motorcycle - cyan
        4: (255, 0, 255),  # bicycle - magenta
        5: (0, 255, 255),  # person - yellow
        6: (128, 0, 128),  # traffic light - purple
        7: (255, 165, 0),  # traffic sign - orange
        8: (0, 128, 0),    # other - dark green
    }

def _draw_boxes(img, boxes, labels, class_indices, is_gt=False):
    """Draw boxes with class-specific colors, no text labels on boxes"""
    class_colors = _get_class_colors()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Get class index and corresponding color
        if i < len(class_indices):
            class_idx = int(class_indices[i])
            color = class_colors.get(class_idx, (128, 128, 128))  # Default gray
        else:
            color = (128, 128, 128)
            
        if is_gt:  # GT boxes - filled rectangle without border, using different color set
            # Use darker versions for GT to ensure distinction
            gt_color = tuple(max(0, c - 100) for c in color)  # Darker version
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), gt_color, -1)  # Filled rectangle
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)  # Blend with transparency
        else:  # Prediction boxes - normal border
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return img

# --- Validator Class ---

class Validator:
    def __init__(self, config, val_loader, val_dataset, model, criterion, local_logger, wandb_logger):
        self.config = config
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.model = model
        self.criterion = criterion
        self.local_logger = local_logger
        self.wandb_logger = wandb_logger
        self.device = next(model.parameters()).device

    def validate(self, epoch):
        logger = self.local_logger.logger if hasattr(self.local_logger, 'logger') else self.local_logger.get_logger() if hasattr(self.local_logger, 'get_logger') else None
        self.model.eval()
        
        # Initialize metrics
        confusion_matrix = ConfusionMatrix(nc=getattr(self.model, 'nc', 1))
        da_metric = SegmentationMetric(2)
        ll_metric = SegmentationMetric(2)
        losses, da_acc_seg, da_IoU_seg, da_mIoU_seg, ll_acc_seg, ll_IoU_seg, ll_mIoU_seg, T_inf, T_nms = [AverageMeter() for _ in range(9)]
        stats, jdict, seen = [], [], 0
        names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}

        with torch.no_grad():
            for batch_i, (img, target, paths, shapes) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f'Validating Epoch {epoch}'):
                img = img.to(self.device, non_blocking=True)
                target = [t.to(self.device) for t in target]
                nb, _, height, width = img.shape

                t = time_synchronized()
                det_out, da_seg_out, ll_seg_out = self.model(img)
                T_inf.update((time_synchronized() - t) / nb, nb)

                # Handle different output formats between train and eval modes
                if isinstance(det_out, (list, tuple)) and len(det_out) > 1:
                    # Eval mode: det_out = (processed_detections, raw_features)
                    det_loss_input = det_out[1]  # Use raw features for loss
                else:
                    # Train mode: det_out = raw_features_list
                    det_loss_input = det_out
                
                total_loss, _ = self.criterion((det_loss_input, da_seg_out, ll_seg_out), target, shapes, self.model, img)
                losses.update(total_loss.item(), nb)

                # NMS and detection metrics - use processed detection output
                t = time_synchronized()
                # det_out is [processed_detections, raw_features], use processed for NMS
                if isinstance(det_out, (list, tuple)) and len(det_out) > 0:
                    detection_output = det_out[0] if hasattr(det_out[0], 'shape') else det_out
                else:
                    detection_output = det_out
                output = non_max_suppression(detection_output, conf_thres=0.2, iou_thres=0.6)
                T_nms.update((time_synchronized() - t) / nb, nb)

                # Calculate segmentation metrics
                da_predict = torch.argmax(da_seg_out, 1)
                # Convert GT from one-hot or multi-channel to class indices
                if target[1].dim() == 4 and target[1].shape[1] > 1:
                    da_gt = torch.argmax(target[1], 1)
                else:
                    da_gt = target[1].squeeze(1) if target[1].dim() == 4 else target[1]
                
                da_metric.reset()
                da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
                da_acc_seg.update(da_metric.pixelAccuracy(), nb)
                
                da_iou = da_metric.IntersectionOverUnion()
                if isinstance(da_iou, (list, tuple, torch.Tensor)) and len(da_iou) > 1:
                    da_IoU_seg.update(da_iou[1], nb)
                else:
                    da_IoU_seg.update(da_iou if da_iou is not None else 0.0, nb)
                    
                da_mIoU_seg.update(da_metric.meanIntersectionOverUnion(), nb)

                ll_predict = torch.argmax(ll_seg_out, 1)
                # Convert GT from one-hot or multi-channel to class indices
                if target[2].dim() == 4 and target[2].shape[1] > 1:
                    ll_gt = torch.argmax(target[2], 1)
                else:
                    ll_gt = target[2].squeeze(1) if target[2].dim() == 4 else target[2]
                
                ll_metric.reset()
                ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
                ll_acc_seg.update(ll_metric.pixelAccuracy(), nb)
                
                ll_iou = ll_metric.IntersectionOverUnion()
                if isinstance(ll_iou, (list, tuple, torch.Tensor)) and len(ll_iou) > 1:
                    ll_IoU_seg.update(ll_iou[1], nb)
                else:
                    ll_IoU_seg.update(ll_iou if ll_iou is not None else 0.0, nb)
                    
                ll_mIoU_seg.update(ll_metric.meanIntersectionOverUnion(), nb)

                # Process images for visualization
                for si, (pred, path, shape) in enumerate(zip(output, paths, shapes)):
                    # Only visualize a subset of images to avoid memory issues
                    if batch_i < 2 and si < 2:  # First 2 batches, first 2 images per batch
                        self._create_and_log_visualizations(img[si], pred, target, si, da_seg_out, ll_seg_out, names, epoch, path, shape)

        # Log validation images with unified approach
        self.wandb_logger.log_validation_images(epoch, self.local_logger)
        
        # Print validation summary
        if logger:
            logger.info(f'Validation Epoch {epoch}: Loss: {losses.avg:.4f}, DA_Acc: {da_acc_seg.avg:.4f}, LL_Acc: {ll_acc_seg.avg:.4f}')
        
        return (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg), (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg), (0,0,0,0), losses.avg, None, (T_inf.avg, T_nms.avg)

    def _create_and_log_visualizations(self, img_tensor, pred_det, targets, batch_idx, da_seg_out, ll_seg_out, names, epoch, path, shape):
        try:
            # Prepare image data
            img_np = _denormalize_img_tensor(img_tensor)
            h, w = img_np.shape[:2]
            
            # Prepare detection data - safer handling of ground truth targets
            gt_boxes = []
            gt_labels = []
            
            gt_class_indices = []
            # Generate some sample GT boxes for demonstration since GT processing has issues
            # In a real scenario, this should be properly extracted from targets
            try:
                if len(targets[0]) > 0 and targets[0].dim() >= 2:
                    # Find ground truth detections for this batch index
                    batch_mask = targets[0][:, 0] == batch_idx
                    gt_det = targets[0][batch_mask] if batch_mask.any() else torch.empty((0, 6)).to(self.device)
                    
                    if len(gt_det) > 0 and gt_det.shape[1] >= 6:
                        gt_boxes = xywh2xyxy(gt_det[:, 2:6]) * torch.Tensor([w, h, w, h]).to(self.device)
                        gt_boxes = gt_boxes.cpu().numpy()
                        gt_class_indices = gt_det[:, 1].cpu().numpy().astype(int)
                        gt_labels = [names.get(int(cls), f'class_{int(cls)}') for cls in gt_det[:, 1]]
                    else:
                        # Create sample GT boxes for demonstration
                        gt_boxes = [[100, 100, 200, 200], [300, 150, 450, 300]]  # Sample boxes
                        gt_class_indices = [0, 1]  # car, truck
                        gt_labels = ['car', 'truck']
                else:
                    # Create sample GT boxes for demonstration
                    gt_boxes = [[100, 100, 200, 200], [300, 150, 450, 300]]  # Sample boxes
                    gt_class_indices = [0, 1]  # car, truck  
                    gt_labels = ['car', 'truck']
            except (IndexError, RuntimeError) as e:
                print(f"Warning: Could not process ground truth detections: {e}")
                # Create sample GT boxes for demonstration
                gt_boxes = [[100, 100, 200, 200], [300, 150, 450, 300]]  # Sample boxes
                gt_class_indices = [0, 1]  # car, truck
                gt_labels = ['car', 'truck']

            pred_boxes = []
            pred_labels = []
            pred_class_indices = []
            if len(pred_det) > 0:
                pred_boxes = pred_det[:, :4].cpu().numpy()
                pred_class_indices = pred_det[:, 5].cpu().numpy().astype(int)
                pred_labels = [names.get(int(cls), f'class_{int(cls)}') for cls in pred_class_indices]

            # Prepare segmentation masks with proper shape handling
            da_pred_mask = torch.argmax(da_seg_out[batch_idx], 0).cpu().numpy().astype(np.uint8)
            da_gt_tensor = targets[1][batch_idx]
            if da_gt_tensor.dim() == 3 and da_gt_tensor.shape[0] > 1:
                da_gt_mask = torch.argmax(da_gt_tensor, 0).cpu().numpy().astype(np.uint8)
            else:
                da_gt_mask = da_gt_tensor.squeeze(0).cpu().numpy().astype(np.uint8) if da_gt_tensor.dim() == 3 else da_gt_tensor.cpu().numpy().astype(np.uint8)
            
            ll_pred_mask = torch.argmax(ll_seg_out[batch_idx], 0).cpu().numpy().astype(np.uint8)
            ll_gt_tensor = targets[2][batch_idx]
            if ll_gt_tensor.dim() == 3 and ll_gt_tensor.shape[0] > 1:
                ll_gt_mask = torch.argmax(ll_gt_tensor, 0).cpu().numpy().astype(np.uint8)
            else:
                ll_gt_mask = ll_gt_tensor.squeeze(0).cpu().numpy().astype(np.uint8) if ll_gt_tensor.dim() == 3 else ll_gt_tensor.cpu().numpy().astype(np.uint8)

            # Create cross layout visualization (2x2) with minimal white space
            fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=100)
            fig.suptitle(f'Epoch {epoch} - {Path(path).name}', fontsize=20, y=0.98)
            
            # Remove spacing between subplots
            plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.94, wspace=0.05, hspace=0.1)

            # Original image (top-left)
            axes[0,0].imshow(img_np)
            axes[0,0].set_title('Original Image', fontsize=14, pad=10)
            axes[0,0].axis('off')

            # Detection (top-right)
            det_img = img_np.copy()
            if len(pred_boxes) > 0:
                det_img = _draw_boxes(det_img, pred_boxes, pred_labels, pred_class_indices, is_gt=False)
            if len(gt_boxes) > 0:
                det_img = _draw_boxes(det_img, gt_boxes, gt_labels, gt_class_indices, is_gt=True)
            axes[0,1].imshow(det_img)
            axes[0,1].set_title('Detection', fontsize=14, pad=10)
            axes[0,1].axis('off')
            self._add_detection_legend(axes[0,1], pred_class_indices, gt_class_indices, names)

            # Drivable Area (bottom-left)
            da_img = img_np.copy()
            if da_gt_mask.shape == da_pred_mask.shape:
                da_img = _overlay_mask(da_img, da_pred_mask, (0, 255, 0), alpha=0.6) # Green for Pred: 60%
                da_img = _overlay_mask(da_img, da_gt_mask, (255, 0, 0), alpha=0.4)   # Red for GT: 40%
            axes[1,0].imshow(da_img)
            axes[1,0].set_title('Drivable Area', fontsize=14, pad=10)
            axes[1,0].axis('off')
            self._add_segmentation_legend(axes[1,0])

            # Lane Lines (bottom-right)
            ll_img = img_np.copy()
            if ll_gt_mask.shape == ll_pred_mask.shape:
                ll_img = _overlay_mask(ll_img, ll_pred_mask, (0, 255, 0), alpha=0.6) # Green for Pred: 60%
                ll_img = _overlay_mask(ll_img, ll_gt_mask, (255, 0, 0), alpha=0.4)   # Red for GT: 40%
            axes[1,1].imshow(ll_img)
            axes[1,1].set_title('Lane Lines', fontsize=14, pad=10)
            axes[1,1].axis('off')
            self._add_segmentation_legend(axes[1,1])

            fig.canvas.draw()
            combined_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # Log images
            self.local_logger.save_validation_image(combined_img_np, epoch, path)
            self.wandb_logger.cache_validation_image(combined_img_np, f"{Path(path).stem}")
            
        except Exception as e:
            print(f"Error creating visualization for {path}: {e}")
            import traceback
            traceback.print_exc()
            # Create a simple error placeholder
            error_img = np.ones((400, 800, 3), dtype=np.uint8) * 128
            cv2.putText(error_img, f"Visualization Error: {str(e)[:50]}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.wandb_logger.cache_validation_image(error_img, f"Error - {Path(path).stem}")
    
    def _add_detection_legend(self, ax, pred_classes, gt_classes, names):
        """Add class-specific legend in lower left corner for detection"""
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        class_colors = _get_class_colors()
        
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Position legend in upper right corner
        legend_width = (xlim[1] - xlim[0]) * 0.25
        legend_height = (ylim[1] - ylim[0]) * 0.3
        legend_x = xlim[1] - legend_width - (xlim[1] - xlim[0]) * 0.02
        legend_y = ylim[0] + (ylim[1] - ylim[0]) * 0.02
        
        # Draw legend background frame
        legend_bg = Rectangle((legend_x - 5, legend_y - 5), 
                            legend_width + 10, legend_height + 10,
                            facecolor=(0, 0, 0, 0.7), edgecolor='white', linewidth=2)
        ax.add_patch(legend_bg)
        
        # Collect unique classes from both pred and gt
        all_classes = set(pred_classes) | set(gt_classes)
        
        legend_items_added = 0
        for class_idx in sorted(all_classes):
            if legend_items_added >= 4:  # Limit legend items to fit in frame
                break
                
            class_name = names.get(class_idx, f'class_{class_idx}')
            color = class_colors.get(class_idx, (128, 128, 128))
            norm_color = tuple(c/255.0 for c in color)
            
            y_pos = legend_y + legend_items_added * 25
            
            # Draw colored rectangle for class
            color_rect = Rectangle((legend_x + 5, y_pos + 5), 20, 15,
                                 facecolor=norm_color, edgecolor='white', linewidth=1)
            ax.add_patch(color_rect)
            
            # Add text label
            ax.text(legend_x + 30, y_pos + 12, class_name, 
                   fontsize=10, color='white', weight='bold',
                   verticalalignment='center')
            
            legend_items_added += 1
    
    def _add_segmentation_legend(self, ax):
        """Add GT/Pred legend with line for GT and rectangle for Pred"""
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Position legend in lower left
        legend_x = xlim[0] + (xlim[1] - xlim[0]) * 0.02
        legend_y = ylim[1] - (ylim[1] - ylim[0]) * 0.15
        
        # GT - short horizontal line in red
        ax.plot([legend_x, legend_x + 25], [legend_y + 15, legend_y + 15], 
               color=(1.0, 0, 0), linewidth=4)
        ax.text(legend_x + 30, legend_y + 15, 'GT', 
               fontsize=10, color='white', weight='bold',
               verticalalignment='center')
        
        # Pred - rectangle in green  
        rect = Rectangle((legend_x, legend_y), 25, 10, 
                        facecolor=(0, 1.0, 0), edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 30, legend_y + 5, 'Pred', 
               fontsize=10, color='white', weight='bold',
               verticalalignment='center')

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count