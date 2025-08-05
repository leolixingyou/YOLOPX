"""Task conflict detection and resolution for multi-task learning."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class TaskConflictDetector:
    """Detect and analyze conflicts between multiple tasks."""
    
    def __init__(self, cfg):
        """Initialize conflict detector.
        
        Args:
            cfg: Configuration object
        """
        self.enabled = cfg.get('CONFLICT.ENABLED', True)
        self.method = cfg.get('CONFLICT.METHOD', 'gradient')
        self.log_interval = cfg.get('CONFLICT.LOG_INTERVAL', 50)
        self.resolution = cfg.get('CONFLICT.RESOLUTION', 'none')
        
        # Task names
        self.tasks = ['detection', 'da_seg', 'll_seg']
        
        # Storage for metrics
        self.gradient_conflicts = defaultdict(list)
        self.loss_ratios = defaultdict(list)
        self.iteration = 0
        
    def compute_gradient_conflicts(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute pairwise gradient conflicts between tasks.
        
        Args:
            model: PyTorch model
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of conflict metrics
        """
        if not self.enabled or self.method not in ['gradient', 'both']:
            return {}
            
        conflicts = {}
        gradients = {}
        
        # Compute gradients for each task
        for task, loss in losses.items():
            if loss is None:
                continue
            # Check if loss is a valid tensor with gradients
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            if loss.item() == 0:
                continue
                
            # Zero gradients
            model.zero_grad()
            
            # Backward for this task
            loss.backward(retain_graph=True)
            
            # Collect gradients
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.data.view(-1))
                    
            if grad_vec:
                gradients[task] = torch.cat(grad_vec)
                
        # Compute pairwise cosine similarities
        task_list = list(gradients.keys())
        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                task1, task2 = task_list[i], task_list[j]
                
                # Skip if gradients have different sizes (different architectures)
                if gradients[task1].shape != gradients[task2].shape:
                    # Use a default conflict value or skip
                    conflict_key = f'{task1}_{task2}_conflict'
                    conflicts[conflict_key] = 0.0  # No conflict if different architectures
                    self.gradient_conflicts[conflict_key].append(0.0)
                    continue
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    gradients[task1].unsqueeze(0),
                    gradients[task2].unsqueeze(0)
                ).item()
                
                # Store conflict (negative cosine = conflict)
                conflict_key = f'{task1}_{task2}_conflict'
                conflicts[conflict_key] = max(0, -cos_sim)
                self.gradient_conflicts[conflict_key].append(conflicts[conflict_key])
                
        return conflicts
        
    def compute_loss_ratios(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute loss ratio imbalances.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of loss ratio metrics
        """
        if not self.enabled or self.method not in ['loss_ratio', 'both']:
            return {}
            
        metrics = {}
        
        # Get loss values
        loss_values = {}
        for task, loss in losses.items():
            if loss is not None:
                loss_values[task] = loss.item()
                
        # Compute total loss
        total_loss = sum(loss_values.values())
        
        if total_loss > 0:
            # Compute ratios
            ratios = {task: val / total_loss for task, val in loss_values.items()}
            
            # Store ratios
            for task, ratio in ratios.items():
                key = f'{task}_ratio'
                metrics[key] = ratio
                self.loss_ratios[key].append(ratio)
                
            # Compute standard deviation as imbalance metric
            ratio_values = list(ratios.values())
            if len(ratio_values) > 1:
                metrics['ratio_imbalance'] = np.std(ratio_values)
                
        return metrics
        
    def detect_conflicts(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Main method to detect task conflicts.
        
        Args:
            model: PyTorch model
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of conflict metrics
        """
        if not self.enabled:
            return {}
            
        self.iteration += 1
        metrics = {}
        
        # Gradient-based conflicts
        if self.method in ['gradient', 'both']:
            grad_conflicts = self.compute_gradient_conflicts(model, losses)
            metrics.update(grad_conflicts)
            
        # Loss ratio analysis
        if self.method in ['loss_ratio', 'both']:
            loss_metrics = self.compute_loss_ratios(losses)
            metrics.update(loss_metrics)
            
        # Compute overall conflict score
        conflict_values = [v for k, v in metrics.items() if 'conflict' in k]
        if conflict_values:
            metrics['overall_conflict'] = np.mean(conflict_values)
            
        # Log if needed
        if self.iteration % self.log_interval == 0:
            self._log_metrics(metrics)
            
        return metrics
        
    def resolve_conflicts(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply conflict resolution method to losses.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            Combined loss after conflict resolution
        """
        if self.resolution == 'none':
            # Simple sum
            return sum(losses.values())
            
        elif self.resolution == 'pcgrad':
            # PCGrad: Project conflicting gradients
            # This is a simplified version
            return self._pcgrad(losses)
            
        elif self.resolution == 'cagrad':
            # CAGrad: Conflict-Averse Gradient
            return self._cagrad(losses)
            
        else:
            return sum(losses.values())
            
    def _pcgrad(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simplified PCGrad implementation."""
        # For now, just return weighted sum
        # Full implementation would project conflicting gradients
        weights = {'detection': 1.0, 'da_seg': 0.5, 'll_seg': 0.5}
        total = 0
        for task, loss in losses.items():
            if loss is not None:
                total = total + weights.get(task, 1.0) * loss
        return total
        
    def _cagrad(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simplified CAGrad implementation."""
        # For now, just return weighted sum based on loss magnitudes
        # Full implementation would use conflict-averse weighting
        total_magnitude = sum(loss.item() for loss in losses.values() if loss is not None)
        total = 0
        for loss in losses.values():
            if loss is not None:
                weight = 1.0 / (1.0 + loss.item() / (total_magnitude + 1e-8))
                total = total + weight * loss
        return total
        
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log conflict metrics."""
        print(f"\n[Iter {self.iteration}] Task Conflict Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of conflicts."""
        summary = {}
        
        # Gradient conflict summary
        for key, values in self.gradient_conflicts.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_max'] = np.max(values)
                
        # Loss ratio summary
        for key, values in self.loss_ratios.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                
        return summary