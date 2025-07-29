import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Optional, Tuple

class GradNormBalancer:
    """
    GradNorm implementation for adaptive loss balancing in multi-task learning
    Based on: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    """
    
    def __init__(self, 
                 task_names: List[str], 
                 alpha: float = 1.5, 
                 update_freq: int = 1,
                 log_dir: str = "./gradnorm_logs",
                 device: torch.device = torch.device('cpu')):
        """
        Initialize GradNorm balancer
        
        Args:
            task_names: List of task names ['detection', 'lane_segment', 'drivable_segment']
            alpha: Controls the strength of restoring force (higher = more aggressive)
            update_freq: How often to update weights (every N steps)
            log_dir: Directory to save logs and visualizations
            device: torch device
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.alpha = alpha
        self.update_freq = update_freq
        self.device = device
        self.log_dir = log_dir
        
        # Initialize task weights (equal weights initially)
        self.task_weights = torch.ones(self.num_tasks, device=device, requires_grad=True)
        
        # Storage for tracking
        self.initial_losses = None
        self.loss_history = defaultdict(list)
        self.weight_history = defaultdict(list)
        self.gradient_norm_history = defaultdict(list)
        self.relative_loss_history = defaultdict(list)
        
        # Training state
        self.step_count = 0
        self.initialized = False
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # For gradient computation
        self.last_shared_layer = None
        
    def initialize_loss_weights(self, initial_losses: Dict[str, float], shared_layer: torch.nn.Module):
        """
        Initialize with first epoch losses and shared layer reference
        
        Args:
            initial_losses: Dict mapping task names to initial loss values
            shared_layer: Reference to shared backbone layer for gradient computation
        """
        self.initial_losses = {task: loss for task, loss in initial_losses.items()}
        self.last_shared_layer = shared_layer
        self.initialized = True
        
        print(f"GradNorm initialized with losses: {self.initial_losses}")
        print(f"Initial task weights: {self.task_weights.detach().cpu().numpy()}")
    
    def compute_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for each task with respect to the shared layer
        
        Args:
            losses: Dict mapping task names to loss tensors
            
        Returns:
            Dict mapping task names to gradient norms
        """
        if not self.initialized or self.last_shared_layer is None:
            return {}
        
        gradients = {}
        
        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                # Compute gradient of weighted loss w.r.t. shared parameters
                weighted_loss = self.task_weights[i] * losses[task_name]
                
                # Get gradients w.r.t. last shared layer
                grad = torch.autograd.grad(
                    weighted_loss, 
                    self.last_shared_layer.parameters(), 
                    retain_graph=True, 
                    create_graph=True
                )[0]
                
                # Compute gradient norm
                grad_norm = torch.norm(grad)
                gradients[task_name] = grad_norm
        
        return gradients
    
    def update_weights(self, 
                      current_losses: Dict[str, float], 
                      epoch: int,
                      step: int) -> Dict[str, float]:
        """
        Update task weights based on GradNorm algorithm
        
        Args:
            current_losses: Current loss values for each task
            epoch: Current training epoch
            step: Current training step
            
        Returns:
            Updated task weights as dict
        """
        if not self.initialized:
            return {task: 1.0 for task in self.task_names}
        
        self.step_count += 1
        
        # Convert losses to tensors
        loss_tensors = {name: torch.tensor(loss, device=self.device, requires_grad=True) 
                       for name, loss in current_losses.items()}
        
        # Compute gradients
        gradients = self.compute_gradients(loss_tensors)
        
        if len(gradients) < self.num_tasks:
            # Fallback if gradient computation fails
            return {self.task_names[i]: self.task_weights[i].item() for i in range(self.num_tasks)}
        
        # Update weights every update_freq steps
        if self.step_count % self.update_freq == 0:
            self._update_task_weights(current_losses, gradients)
        
        # Log current state
        self._log_current_state(current_losses, gradients, epoch, step)
        
        # Return current weights as dict
        return {self.task_names[i]: self.task_weights[i].item() for i in range(self.num_tasks)}
    
    def _update_task_weights(self, current_losses: Dict[str, float], gradients: Dict[str, torch.Tensor]):
        """Core GradNorm weight update algorithm"""
        
        # Compute relative inverse training rates
        relative_losses = {}
        for task in self.task_names:
            if task in current_losses and task in self.initial_losses:
                r_i = current_losses[task] / self.initial_losses[task]
                relative_losses[task] = r_i
        
        # Average relative loss
        r_avg = np.mean(list(relative_losses.values()))
        
        # Target gradient norms
        G_avg = np.mean([grad.item() for grad in gradients.values()])
        
        # Update weights
        with torch.no_grad():
            for i, task in enumerate(self.task_names):
                if task in relative_losses and task in gradients:
                    # Target gradient norm for this task
                    target_grad = G_avg * (relative_losses[task] / r_avg) ** self.alpha
                    
                    # Current gradient norm
                    current_grad = gradients[task].item()
                    
                    # Update weight (simple proportional adjustment)
                    if current_grad > 0:
                        adjustment = target_grad / current_grad
                        self.task_weights[i] *= adjustment
            
            # Renormalize weights to sum to num_tasks (keep average weight = 1)
            self.task_weights *= self.num_tasks / self.task_weights.sum()
    
    def _log_current_state(self, current_losses: Dict[str, float], 
                          gradients: Dict[str, torch.Tensor], 
                          epoch: int, step: int):
        """Log current training state"""
        
        # Store losses
        for task, loss in current_losses.items():
            self.loss_history[task].append(loss)
            if self.initial_losses and task in self.initial_losses:
                relative_loss = loss / self.initial_losses[task]
                self.relative_loss_history[task].append(relative_loss)
        
        # Store weights
        for i, task in enumerate(self.task_names):
            self.weight_history[task].append(self.task_weights[i].item())
        
        # Store gradient norms
        for task, grad_norm in gradients.items():
            self.gradient_norm_history[task].append(grad_norm.item())
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current task weights"""
        return {self.task_names[i]: self.task_weights[i].item() for i in range(self.num_tasks)}
    
    def log_status(self, logger):
        """Log current status to logger"""
        weights = self.get_current_weights()
        weight_str = ", ".join([f"{task}: {weight:.4f}" for task, weight in weights.items()])
        logger.info(f"GradNorm weights - {weight_str}")
    
    def save_training_curves(self, save_path: Optional[str] = None):
        """
        Save comprehensive training curves and data
        
        Args:
            save_path: Optional custom save path
        """
        if save_path is None:
            save_path = self.log_dir
        
        # Create comprehensive plots
        self._create_loss_curves(save_path)
        self._create_weight_curves(save_path)
        self._create_gradient_curves(save_path)
        self._create_combined_dashboard(save_path)
        
        # Save raw data as CSV
        self._save_csv_data(save_path)
        
        print(f"Training curves and data saved to {save_path}")
    
    def _create_loss_curves(self, save_path: str):
        """Create loss evolution plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Absolute losses
        for task in self.task_names:
            if task in self.loss_history:
                ax1.plot(self.loss_history[task], label=f'{task}', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Absolute Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative losses
        for task in self.task_names:
            if task in self.relative_loss_history:
                ax2.plot(self.relative_loss_history[task], label=f'{task}', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Relative Loss (L_i(t)/L_i(0))')
        ax2.set_title('Relative Loss Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_weight_curves(self, save_path: str):
        """Create task weight evolution plots"""
        plt.figure(figsize=(12, 6))
        
        for task in self.task_names:
            if task in self.weight_history:
                plt.plot(self.weight_history[task], label=f'{task}', linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Task Weight')
        plt.title('GradNorm Task Weight Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal weight')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weight_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gradient_curves(self, save_path: str):
        """Create gradient norm evolution plots"""
        plt.figure(figsize=(12, 6))
        
        for task in self.task_names:
            if task in self.gradient_norm_history:
                plt.plot(self.gradient_norm_history[task], label=f'{task}', linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for gradient norms
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'gradient_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_combined_dashboard(self, save_path: str):
        """Create comprehensive dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss evolution
        for task in self.task_names:
            if task in self.loss_history:
                axes[0,0].plot(self.loss_history[task], label=f'{task}', linewidth=2)
        axes[0,0].set_title('Loss Evolution')
        axes[0,0].set_xlabel('Steps')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Weight evolution
        for task in self.task_names:
            if task in self.weight_history:
                axes[0,1].plot(self.weight_history[task], label=f'{task}', linewidth=2)
        axes[0,1].set_title('Task Weight Evolution')
        axes[0,1].set_xlabel('Steps')
        axes[0,1].set_ylabel('Weight')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Gradient norms
        for task in self.task_names:
            if task in self.gradient_norm_history:
                axes[1,0].semilogy(self.gradient_norm_history[task], label=f'{task}', linewidth=2)
        axes[1,0].set_title('Gradient Norm Evolution')
        axes[1,0].set_xlabel('Steps')
        axes[1,0].set_ylabel('Gradient Norm (log scale)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Relative losses
        for task in self.task_names:
            if task in self.relative_loss_history:
                axes[1,1].plot(self.relative_loss_history[task], label=f'{task}', linewidth=2)
        axes[1,1].set_title('Relative Loss Evolution')
        axes[1,1].set_xlabel('Steps')
        axes[1,1].set_ylabel('L_i(t)/L_i(0)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'gradnorm_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_csv_data(self, save_path: str):
        """Save all tracking data as CSV files"""
        
        # Prepare data for CSV
        max_length = max([
            len(self.loss_history.get(task, [])) for task in self.task_names
        ] + [
            len(self.weight_history.get(task, [])) for task in self.task_names
        ] + [
            len(self.gradient_norm_history.get(task, [])) for task in self.task_names
        ])
        
        if max_length == 0:
            return
        
        # Create comprehensive dataframe
        data = {'step': list(range(max_length))}
        
        # Add loss data
        for task in self.task_names:
            if task in self.loss_history:
                # Pad with NaN if necessary
                losses = self.loss_history[task] + [np.nan] * (max_length - len(self.loss_history[task]))
                data[f'loss_{task}'] = losses
        
        # Add weight data
        for task in self.task_names:
            if task in self.weight_history:
                weights = self.weight_history[task] + [np.nan] * (max_length - len(self.weight_history[task]))
                data[f'weight_{task}'] = weights
        
        # Add gradient norm data
        for task in self.task_names:
            if task in self.gradient_norm_history:
                grads = self.gradient_norm_history[task] + [np.nan] * (max_length - len(self.gradient_norm_history[task]))
                data[f'grad_norm_{task}'] = grads
        
        # Add relative loss data
        for task in self.task_names:
            if task in self.relative_loss_history:
                rel_losses = self.relative_loss_history[task] + [np.nan] * (max_length - len(self.relative_loss_history[task]))
                data[f'relative_loss_{task}'] = rel_losses
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, 'gradnorm_training_data.csv'), index=False)
        
        # Save summary statistics
        summary_data = {
            'task': self.task_names,
            'initial_loss': [self.initial_losses.get(task, np.nan) for task in self.task_names],
            'final_weight': [self.weight_history[task][-1] if task in self.weight_history and self.weight_history[task] else np.nan for task in self.task_names],
            'avg_gradient_norm': [np.mean(self.gradient_norm_history[task]) if task in self.gradient_norm_history else np.nan for task in self.task_names]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_path, 'gradnorm_summary.csv'), index=False)


# Integration helper functions for your existing codebase

def integrate_gradnorm_with_training(cfg, model, device, log_dir):
    """
    Helper function to integrate GradNorm with existing training code
    
    Args:
        cfg: Your config object
        model: Your YOLOPX model
        device: Training device
        log_dir: Directory for logs
        
    Returns:
        GradNormBalancer instance
    """
    task_names = ['detection', 'lane_segment', 'drivable_segment']
    
    # Create GradNorm balancer
    gradnorm_balancer = GradNormBalancer(
        task_names=task_names,
        alpha=cfg.get('GRADNORM_ALPHA', 1.5),  # Add to your config
        update_freq=cfg.get('GRADNORM_UPDATE_FREQ', 1),
        log_dir=os.path.join(log_dir, 'gradnorm'),
        device=device
    )
    
    # Find shared layer (typically the backbone's last layer)
    shared_layer = None
    if hasattr(model, 'backbone'):
        shared_layer = model.backbone
    elif hasattr(model, 'module') and hasattr(model.module, 'backbone'):
        shared_layer = model.module.backbone
    else:
        # Fallback: use first parameter-containing layer
        for module in model.modules():
            if list(module.parameters()):
                shared_layer = module
                break
    
    gradnorm_balancer.last_shared_layer = shared_layer
    
    return gradnorm_balancer

def compute_weighted_loss(losses_dict, weights_dict):
    """
    Compute weighted multi-task loss
    
    Args:
        losses_dict: Dict of individual task losses
        weights_dict: Dict of task weights from GradNorm
        
    Returns:
        Weighted total loss
    """
    total_loss = 0
    for task, loss in losses_dict.items():
        if task in weights_dict:
            total_loss += weights_dict[task] * loss
        else:
            total_loss += loss  # Default weight of 1.0
    
    return total_loss