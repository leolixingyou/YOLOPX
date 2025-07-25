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
                        "epochs": cfg.TRAIN.END_EPOCH,
                        "batch_size": cfg.TRAIN.BATCH_SIZE_PER_GPU,
                        "learning_rate": cfg.TRAIN.LR0,
                    },
                    reinit=reinit
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_run = None
        
        self.validation_images_cache = []
        self.log_imgs_limit = 16

    def set_logger(self, logger_instance):
        self.logger = logger_instance

    def log_epoch_metrics(self, epoch_metrics, epoch):
        if self.wandb_run:
            self.wandb_run.log({"epoch": epoch, **epoch_metrics})
    
    def log_batch_metrics(self, batch_metrics):
        """Log batch-level metrics to wandb"""
        if self.wandb_run:
            self.wandb_run.log(batch_metrics)

    def cache_validation_image(self, image_np, caption):
        """Cache individual validation images for creating a combined visualization"""
        if len(self.validation_images_cache) < self.log_imgs_limit:
            self.validation_images_cache.append({
                'image': image_np,
                'caption': caption
            })

    def log_validation_images(self, epoch, local_logger=None):
        """Create and log a unified grid of validation images to wandb and optionally save locally"""
        if not self.validation_images_cache:
            return
            
        # Create a grid of validation images
        n_images = len(self.validation_images_cache)
        if n_images == 0:
            return
            
        # Calculate grid dimensions (optimized for wandb horizontal display)
        cols = min(4, n_images)  # Max 4 columns
        rows = (n_images + cols - 1) // cols
        
        # Create the combined figure - optimized for wandb
        fig_width = cols * 5  # Smaller width for better horizontal display
        fig_height = rows * 3  # Compact height
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=120)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        fig.suptitle(f'Validation Results - Epoch {epoch}', fontsize=14, y=0.95)
        
        # Tight spacing for wandb display
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.88, wspace=0.05, hspace=0.15)
        
        # Plot each image
        for i, img_data in enumerate(self.validation_images_cache):
            if i < len(axes):
                axes[i].imshow(img_data['image'])
                axes[i].set_title(img_data['caption'], fontsize=9, pad=3)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(self.validation_images_cache), len(axes)):
            axes[i].axis('off')
        
        # Convert to numpy array
        fig.canvas.draw()
        combined_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        combined_img_np = combined_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Save locally if local logger is provided
        if local_logger:
            local_logger.save_combined_validation_image(combined_img_np, epoch)
        
        # Log to wandb if available
        if self.wandb_run:
            self.wandb_run.log({
                "validation/combined_results": wandb.Image(combined_img_np, caption=f"Validation Results - Epoch {epoch}"),
                "epoch": epoch
            })
        
        plt.close(fig)
        self.validation_images_cache.clear()

    def finish_run(self):
        if self.wandb_run:
            self.wandb_run.finish()

class LocalFileLogger:
    def __init__(self, log_dir, create_visualization_dirs=True):
        self.log_dir = Path(log_dir)
        
        if create_visualization_dirs:
            self.vis_dir = self.log_dir / 'visualizations'
            self.individual_dir = self.vis_dir / 'individual'
            self.combined_dir = self.vis_dir / 'combined'
            
            # Create all visualization directories
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            self.individual_dir.mkdir(parents=True, exist_ok=True)
            self.combined_dir.mkdir(parents=True, exist_ok=True)
        else:
            # For main directory logger - no visualization dirs
            self.vis_dir = None
            self.individual_dir = None
            self.combined_dir = None

    def save_validation_image(self, image_np, epoch, path_name):
        """Save individual validation image"""
        if self.individual_dir is None:
            return None  # No visualization directories
        epoch_dir = self.individual_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        save_path = epoch_dir / f"{Path(path_name).stem}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return save_path

    def save_combined_validation_image(self, image_np, epoch):
        """Save combined validation grid image"""
        if self.combined_dir is None:
            return None  # No visualization directories
        save_path = self.combined_dir / f"validation_combined_epoch_{epoch}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return save_path

    def save_comparison_plot(self, fig, plot_name):
        """Save comparison plots"""
        plot_file = self.log_dir / f"{plot_name}.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        return plot_file
    
    def get_visualization_paths(self):
        """Return all visualization directory paths for reference"""
        if self.vis_dir is None:
            return None
        return {
            'root': self.vis_dir,
            'individual': self.individual_dir,
            'combined': self.combined_dir
        }

class ConsoleLogger:
    def __init__(self, log_dir, run_id_suffix):
        self.logger = logging.getLogger(run_id_suffix)
        self.logger.handlers = []
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        self.file_handler = logging.FileHandler(Path(log_dir) / f'{run_id_suffix}.log')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)
        
        self.logger.setLevel(logging.INFO)

    def get_logger(self):
        return self.logger
    
    def remove_console_handler(self):
        """Remove console handler to reduce output during training"""
        if self.console_handler in self.logger.handlers:
            self.logger.removeHandler(self.console_handler)
    
    def add_console_handler(self):
        """Add back console handler"""
        if self.console_handler not in self.logger.handlers:
            self.logger.addHandler(self.console_handler)