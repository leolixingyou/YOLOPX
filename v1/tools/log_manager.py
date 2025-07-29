import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. Metrics will not be logged to wandb.")

from lib.utils.utils import xywh2xyxy, scale_coords, clip_coords, _coco80_to_coco91_class

def _denormalize_img_tensor(img_tensor):
    """Denormalizes a single image tensor and converts to HWC numpy array (0-255)."""
    img_tensor_cpu = img_tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = img_tensor_cpu * std + mean
    img = img.permute(1, 2, 0).numpy() * 255
    return img.astype(np.uint8)

def _overlay_mask_on_image(image_np, mask_np, color=(0, 255, 0), alpha=0.5):
    """
    Overlays a binary mask onto an image.
    image_np: HWC numpy array (0-255, RGB).
    mask_np: HW numpy array (binary, 0 or 1).
    color: BGR tuple for the mask color.
    alpha: Transparency of the overlay.
    """
    if image_np.shape[:2] != mask_np.shape[:2]:
        raise ValueError("Image and mask must have the same HxW dimensions.")
    
    overlay = image_np.copy()
    colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    
    mask_bool = mask_np.astype(bool)
    
    colored_mask[mask_bool] = color
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def _draw_box(img, xyxy, label, color):
    x1, y1, x2, y2 = [int(c) for c in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # Add label text
    tf = max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2))
    t_size = cv2.getTextSize(label, 0, fontScale=tf / 3, thickness=tf)[0]
    c2 = x1 + t_size[0], y1 - t_size[1] - 3
    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (x1, y1 - 2), 0, tf / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Placeholder for coco80_to_coco91_class if not directly imported or accessible
# In a real scenario, you'd import this from lib.core.general or define it if standalone.
def _coco80_to_coco91_class():
    return list(range(80)) # Simplified for example, real mapping is more complex

class WandBLogger:
    def __init__(self, cfg, run_name, project_name="multitask-training-refactor", reinit=True):
        self.cfg = cfg
        self.wandb_run = None
        self.logger = logging.getLogger(self.__class__.__name__)
        if WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config={
                        "dataset": cfg.DATASET.DATASET,
                        "model": cfg.MODEL.NAME,
                        # "conflict_method": (set by run_experiment),
                        "epochs": cfg.TRAIN.END_EPOCH,
                        "batch_size": cfg.TRAIN.BATCH_SIZE_PER_GPU,
                        "learning_rate": cfg.TRAIN.LR0,
                    },
                    reinit=reinit
                )
                self.logger.info(f"Wandb initialized: {project_name}/{run_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}. Wandb will be disabled for this run.")
                self.wandb_run = None
        
        self.current_epoch_metrics = {} # To accumulate metrics over an epoch for epoch-level log
        self.log_images_cache = { # Cache for images before logging at epoch end
            'valid_img_detection': [],
            'valid_img_da_seg': [],
            'valid_img_ll_seg': []
        }
        self.log_imgs_limit = min(16, 100) # Max images to log per epoch for samples

        # Setup a dummy logger for init if no logger is passed
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def set_logger(self, logger_instance):
        self.logger = logger_instance

    def log_batch_metrics(self, metrics_dict):
        if self.wandb_run:
            self.wandb_run.log(metrics_dict)

    def log_epoch_metrics(self, epoch_metrics, epoch):
        if self.wandb_run:
            epoch_metrics['epoch'] = epoch
            self.wandb_run.log(epoch_metrics)

    def cache_valid_detection_image(self, img_tensor, pred_data, gt_data, names, epoch, path_name):
        if self.wandb_run and len(self.log_images_cache['valid_img_detection']) < self.log_imgs_limit:
            # Convert tensor to a writable numpy array
            img_np = _denormalize_img_tensor(img_tensor)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

            # Draw prediction boxes (Green)
            for *xyxy, conf, cls in pred_data:
                label = f'{names.get(int(cls), "unknown")} {conf:.2f}'
                _draw_box(img_vis, xyxy, label, color=(0, 255, 0))

            # Draw ground truth boxes (Blue)
            for *xyxy, cls in gt_data:
                label = f'{names.get(int(cls), "unknown")} (GT)'
                _draw_box(img_vis, xyxy, label, color=(255, 0, 0))

            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            self.log_images_cache['valid_img_detection'].append(wandb.Image(img_vis_rgb, caption=f"Epoch {epoch} - Det - {path_name}"))

    def cache_valid_seg_image(self, img_tensor, pred_mask_np, gt_mask_np, task_type, epoch, path_name, padding_info=None):
        if self.wandb_run and len(self.log_images_cache[f'valid_img_{task_type}']) < self.log_imgs_limit:
            original_img_np = _denormalize_img_tensor(img_tensor)

            # Crop padding if info is provided
            if padding_info:
                (original_h, original_w), (pad_h, pad_w) = padding_info
                if original_h > 2 * pad_h and original_w > 2 * pad_w:
                    cropped_img_np = original_img_np[pad_h:original_h-pad_h, pad_w:original_w-pad_w, :]
                else:
                    cropped_img_np = original_img_np # Avoid cropping if padding is larger than image
            else:
                cropped_img_np = original_img_np

            # Overlay Prediction Mask (Green)
            vis_img = _overlay_mask_on_image(cropped_img_np, pred_mask_np, color=(0, 255, 0), alpha=0.5)
            # Overlay Ground Truth Mask (Blue)
            vis_img = _overlay_mask_on_image(vis_img, gt_mask_np, color=(255, 0, 0), alpha=0.5)

            self.log_images_cache[f'valid_img_{task_type}'].append(wandb.Image(vis_img, caption=f"Epoch {epoch} - {task_type.upper().replace('_',' ')} - {path_name}"))

    def log_validation_images(self, epoch):
        if self.wandb_run:
            for key, images_list in self.log_images_cache.items():
                if images_list:
                    self.wandb_run.log({f"Validation/{key.replace('_',' ').title()} Samples": images_list, "epoch": epoch})
                    self.logger.info(f"Logged {len(images_list)} {key.replace('_',' ')} sample images to WandB.")
            self.clear_image_cache()

    def clear_image_cache(self):
        for key in self.log_images_cache:
            self.log_images_cache[key] = []

    def log_final_plots(self, plot_name, fig):
        if self.wandb_run:
            self.wandb_run.log({f"final_analysis_plots/{plot_name}": wandb.Image(fig)})

    def log_training_summary(self, summary_data):
        if self.wandb_run:
            self.wandb_run.log({"training_summary": summary_data})

    def finish_run(self):
        if self.wandb_run:
            self.wandb_run.finish()

class LocalFileLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.save_dir = self.log_dir / 'visualization' # For plots and optionally txt/json
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'labels').mkdir(parents=True, exist_ok=True) # For .txt labels

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def set_logger(self, logger_instance):
        self.logger = logger_instance

    def save_confusion_matrix_plot(self, confusion_matrix_instance, names):
        # Assuming ConfusionMatrix has a .plot() method that saves to disk
        confusion_matrix_instance.plot(save_dir=self.save_dir, names=list(names.values()))
        self.logger.info(f"Confusion matrix plot saved to {self.save_dir}")

    def save_detection_txt(self, predn_list, shapes_si, path_stem):
        # predn_list: list of [xyxy, conf, cls] for a single image, already scaled
        # shapes_si: (original_shape, (ratio, (pad_h, pad_w)))
        gn = torch.tensor(shapes_si[0])[[1, 0, 1, 0]] # normalization gain whwh
        labels_dir = self.save_dir / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True) # Ensure labels directory exists
        with open(labels_dir / f"{path_stem}.txt", 'a') as f:
            for *xyxy, conf, cls in predn_list:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
                line = (int(cls), *xywh, float(conf)) # label format (cls, x, y, w, h, conf)
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def save_detection_json(self, jdict_list, filename="predictions.json"):
        # jdict_list: list of COCO format dicts for predictions
        # This will save all jdict entries collected over the entire validation.
        output_json_path = self.save_dir / filename
        with open(output_json_path, 'w') as f:
            import json # Ensure json is imported
            json.dump(jdict_list, f)
        self.logger.info(f"Detection JSON saved to {output_json_path}")
    
    def save_comparison_plot(self, fig, plot_name):
        plot_file = self.log_dir / 'comparisons' / f"{plot_name}.png"
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the plot to free memory
        self.logger.info(f"Comparative plot saved to: {plot_file}")

class ConsoleLogger:
    def __init__(self, log_dir, run_id_suffix):
        self.logger = logging.getLogger(run_id_suffix)
        self.logger.handlers = []
        self.logger.propagate = False # Prevent multiple logs if root logger is configured

        file_handler = logging.FileHandler(Path(log_dir) / f'{run_id_suffix}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        self.logger.setLevel(logging.INFO)

    def get_logger(self):
        return self.logger

    def remove_console_handler(self):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)
                return
    
    def add_console_handler(self):
        # Re-add if it was removed
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(console_handler)

