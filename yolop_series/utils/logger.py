"""
Logging utilities for the YOLOPX project.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# --- Drawing Utilities (Adapted from V1) ---

def _denormalize_img_tensor(img_tensor):
    """Denormalizes a single image tensor and converts to HWC numpy array (0-255)."""
    img_tensor_cpu = img_tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor_cpu.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor_cpu.device).view(3, 1, 1)
    
    img = img_tensor_cpu * std + mean
    img = img.permute(1, 2, 0).cpu().numpy() * 255
    return img.astype(np.uint8)

def _overlay_mask(image_np, mask_np, color, alpha=0.5):
    """Overlays a binary mask onto an image."""
    overlay = image_np.copy()
    colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    colored_mask[mask_np > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def _draw_box(img, xyxy, label, color):
    """Draws a bounding box with a label on an image."""
    try:
        coords = np.array(xyxy).flatten().tolist()
        x1, y1, x2, y2 = map(int, coords)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tf = max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2))
        t_size = cv2.getTextSize(label, 0, fontScale=tf / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, tf / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    except (ValueError, TypeError) as e:
        print(f"Skipping drawing box due to invalid coordinate format: {xyxy}. Error: {e}")

# --- Loggers ---

def setup_program_logger(log_dir):
    """Sets up the main program logger."""
    program_logger = logging.getLogger('YOLOPX')
    program_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    file_handler = logging.FileHandler(log_dir / 'program.log')
    file_handler.setFormatter(formatter)
    program_logger.addHandler(file_handler)
    
    return program_logger

class WandBLogger:
    """Logger for Weights & Biases integration."""
    def __init__(self, cfg, run_name, project_name="Phoenix-YOLOPX"):
        self.wandb_run = None
        if WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=dict(cfg)
                )
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
        
        self.validation_images_cache = []

    def log_epoch_metrics(self, epoch_metrics, epoch):
        if self.wandb_run:
            self.wandb_run.log({"epoch": epoch, **epoch_metrics})
    
    def log_batch_metrics(self, batch_metrics):
        if self.wandb_run:
            self.wandb_run.log(batch_metrics)

    def cache_multitask_image(self, img_tensor, predictions, ground_truths, class_names, caption):
        if self.wandb_run and len(self.validation_images_cache) < 16:
            img_np = _denormalize_img_tensor(img_tensor)
            vis_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Draw Ground Truths (Blue)
            vis_img = _overlay_mask(vis_img, ground_truths['da_seg'], color=(255, 0, 0))
            vis_img = _overlay_mask(vis_img, ground_truths['ll_seg'], color=(255, 0, 0))
            for *xyxy, cls in ground_truths['det']:
                label = f'{class_names[int(cls)]} (GT)'
                _draw_box(vis_img, xyxy, label, color=(255, 0, 0))

            # Draw Predictions (Green)
            vis_img = _overlay_mask(vis_img, predictions['da_seg'], color=(0, 255, 0))
            vis_img = _overlay_mask(vis_img, predictions['ll_seg'], color=(0, 255, 0))
            for *xyxy, conf, cls in predictions['det']:
                label = f'{class_names[int(cls)]} {conf:.2f}'
                _draw_box(vis_img, xyxy, label, color=(0, 255, 0))

            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            self.validation_images_cache.append(wandb.Image(vis_img_rgb, caption=caption))

    def log_validation_images(self, epoch, local_logger):
        if self.wandb_run and self.validation_images_cache:
            self.wandb_run.log({"validation_results": self.validation_images_cache})
        
        for i, img in enumerate(self.validation_images_cache):
            local_logger.save_validation_image(img.image, epoch, f"batch_{i}")
        self.validation_images_cache.clear()

class LocalFileLogger:
    """Logger for saving results to local files."""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.vis_dir = self.log_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def save_validation_image(self, image_np, epoch, name):
        save_path = self.vis_dir / f"epoch_{epoch}_{name}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

class ConsoleLogger:
    """Logger for console output."""
    def __init__(self, log_dir, run_name):
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_dir / f'{run_name}.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
