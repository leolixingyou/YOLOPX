"""Task conflict detection module for multi-task learning."""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class ConflictDetector:
    """Detect and analyze task conflicts in multi-task learning."""
    
    def __init__(self, cfg):
        """Initialize conflict detector.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.enabled = cfg.CONFLICT.ENABLED
        self.method = cfg.CONFLICT.METHOD
        self.log_freq = cfg.CONFLICT.LOG_FREQ
        
        # Storage for conflict metrics
        self.conflict_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.loss_ratio_history = defaultdict(list)
        
        # Task names
        self.tasks = ['detection', 'da_segmentation', 'll_segmentation']
        
    def compute_gradient_conflict(self, model, losses):
        """Compute gradient conflicts between tasks.
        
        Args:
            model: PyTorch model
            losses: Dictionary of task losses
            
        Returns:
            conflict_metrics: Dictionary of conflict metrics
        """
        if not self.enabled or self.method not in ['gradient', 'both']:
            return {}
            
        # Store gradients for each task
        task_gradients = {}
        
        for task_name, loss in losses.items():
            # Zero gradients
            model.zero_grad()
            
            # Backward pass for this task only
            loss.backward(retain_graph=True)
            
            # Collect gradients
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.clone().flatten())
                    
            if grads:
                task_gradients[task_name] = torch.cat(grads)
                
        # Compute pairwise cosine similarities
        conflict_metrics = {}
        task_names = list(task_gradients.keys())
        
        for i in range(len(task_names)):
            for j in range(i + 1, len(task_names)):
                task1, task2 = task_names[i], task_names[j]
                grad1 = task_gradients[task1]
                grad2 = task_gradients[task2]
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad1.unsqueeze(0), grad2.unsqueeze(0)
                ).item()
                
                # Conflict measure (negative cosine similarity indicates conflict)
                conflict = -cos_sim if cos_sim < 0 else 0
                
                key = f"{task1}_{task2}_gradient_conflict"
                conflict_metrics[key] = conflict
                self.gradient_history[key].append(conflict)
                
        return conflict_metrics
        
    def compute_loss_ratio_conflict(self, losses):
        """Compute task conflict based on loss ratios.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            conflict_metrics: Dictionary of conflict metrics
        """
        if not self.enabled or self.method not in ['loss_ratio', 'both']:
            return {}
            
        conflict_metrics = {}
        loss_values = {k: v.item() for k, v in losses.items()}
        
        # Compute loss ratios
        total_loss = sum(loss_values.values())
        if total_loss > 0:
            loss_ratios = {k: v / total_loss for k, v in loss_values.items()}
            
            # Compute standard deviation of loss ratios as conflict measure
            ratio_values = list(loss_ratios.values())
            ratio_std = np.std(ratio_values)
            
            conflict_metrics['loss_ratio_std'] = ratio_std
            self.loss_ratio_history['loss_ratio_std'].append(ratio_std)
            
            # Store individual ratios
            for task, ratio in loss_ratios.items():
                key = f"{task}_loss_ratio"
                conflict_metrics[key] = ratio
                self.loss_ratio_history[key].append(ratio)
                
        return conflict_metrics
        
    def detect_conflicts(self, model, losses, iteration):
        """Main method to detect task conflicts.
        
        Args:
            model: PyTorch model
            losses: Dictionary of task losses
            iteration: Current training iteration
            
        Returns:
            conflict_info: Dictionary containing conflict information
        """
        if not self.enabled:
            return {}
            
        conflict_info = {}
        
        # Gradient-based conflict detection
        if self.method in ['gradient', 'both']:
            gradient_conflicts = self.compute_gradient_conflict(model, losses)
            conflict_info.update(gradient_conflicts)
            
        # Loss ratio-based conflict detection
        if self.method in ['loss_ratio', 'both']:
            loss_conflicts = self.compute_loss_ratio_conflict(losses)
            conflict_info.update(loss_conflicts)
            
        # Compute overall conflict score
        if conflict_info:
            # Average of all conflict measures
            conflict_values = [v for k, v in conflict_info.items() 
                             if 'conflict' in k or 'std' in k]
            if conflict_values:
                conflict_info['overall_conflict'] = np.mean(conflict_values)
                
        # Log if needed
        if iteration % self.log_freq == 0:
            self._log_conflicts(conflict_info, iteration)
            
        return conflict_info
        
    def _log_conflicts(self, conflict_info, iteration):
        """Log conflict information."""
        print(f"\n[Iteration {iteration}] Task Conflict Analysis:")
        for key, value in conflict_info.items():
            print(f"  {key}: {value:.4f}")
            
    def get_conflict_summary(self):
        """Get summary statistics of conflicts."""
        summary = {}
        
        # Gradient conflicts
        for key, values in self.gradient_history.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_max"] = np.max(values)
                
        # Loss ratio conflicts  
        for key, values in self.loss_ratio_history.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                
        return summary