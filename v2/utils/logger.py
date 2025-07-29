"""
Logging utilities for the YOLOPX project.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

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

    def cache_validation_image(self, fig, caption):
        if len(self.validation_images_cache) < 16: # Limit cache size
            fig.canvas.draw()
            img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.validation_images_cache.append(wandb.Image(img_np, caption=caption))

    def log_validation_images(self, epoch, local_logger):
        if self.wandb_run and self.validation_images_cache:
            self.wandb_run.log({"validation_results": self.validation_images_cache})
        # Optionally save locally as well
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
