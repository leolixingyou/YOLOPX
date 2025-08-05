"""Task conflict detection with PCGrad-style metrics including TCI."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class TaskConflictDetectorPCGrad:
    """Conflict detector with PCGrad paper metrics including TCI (Task Conflict Index)."""
    
    def __init__(self, cfg):
        """Initialize conflict detector.
        
        Args:
            cfg: Configuration object
        """
        self.enabled = cfg.get('CONFLICT.ENABLED', True)
        self.method = cfg.get('CONFLICT.METHOD', 'gradient')
        self.log_interval = cfg.get('CONFLICT.LOG_INTERVAL', 50)
        
        # Task names
        self.tasks = ['detection', 'da_seg', 'll_seg', 'll_iou']
        
        # Storage for metrics
        self.gradient_conflicts = defaultdict(list)
        self.cosine_similarities = defaultdict(list)
        self.tci_values = defaultdict(list)  # Task Conflict Index
        self.iteration = 0
        
    def compute_gradient_conflicts(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute pairwise gradient conflicts between tasks.
        
        Args:
            model: PyTorch model
            losses: Dictionary of task losses
            
        Returns:
            Dictionary of conflict metrics including cosine similarity and TCI
        """
        if not self.enabled:
            return {}
            
        conflicts = {}
        gradients = {}
        
        # Filter valid losses
        valid_losses = {}
        for task, loss in losses.items():
            if loss is None or not isinstance(loss, torch.Tensor):
                continue
            if not loss.requires_grad:
                continue
            valid_losses[task] = loss
        
        # Compute gradients for each task
        for task, loss in valid_losses.items():
            # Zero gradients
            model.zero_grad()
            
            # Backward for this task
            try:
                loss.backward(retain_graph=True)
            except RuntimeError:
                continue
            
            # Collect gradients from shared parameters
            grad_vec = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Only collect gradients from shared layers (backbone)
                    if not any(head in name for head in ['det_head', 'seg_head', 'head', 'final']):
                        grad_vec.append(param.grad.data.view(-1))
                    
            if grad_vec:
                gradients[task] = torch.cat(grad_vec)
                
        # Compute pairwise metrics
        task_list = list(gradients.keys())
        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                task1, task2 = task_list[i], task_list[j]
                
                # Skip if gradients have different sizes
                if gradients[task1].shape != gradients[task2].shape:
                    continue
                
                # Skip if gradients are too small
                norm1 = gradients[task1].norm()
                norm2 = gradients[task2].norm()
                if norm1 < 1e-8 or norm2 < 1e-8:
                    # Still record zero values
                    cos_key = f'{task1}_{task2}_cos_sim'
                    conflict_key = f'{task1}_{task2}_conflict'
                    tci_key = f'{task1}_{task2}_tci'
                    
                    conflicts[cos_key] = 0.0
                    conflicts[conflict_key] = 0.0
                    conflicts[tci_key] = 1.0  # TCI = 1 - cos_sim when cos_sim = 0
                    
                    self.cosine_similarities[cos_key].append(0.0)
                    self.gradient_conflicts[conflict_key].append(0.0)
                    self.tci_values[tci_key].append(1.0)
                    continue
                
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    gradients[task1].unsqueeze(0),
                    gradients[task2].unsqueeze(0)
                ).item()
                
                # Store all metrics
                cos_key = f'{task1}_{task2}_cos_sim'
                conflict_key = f'{task1}_{task2}_conflict'
                tci_key = f'{task1}_{task2}_tci'
                
                # Cosine similarity (can be negative)
                conflicts[cos_key] = cos_sim
                self.cosine_similarities[cos_key].append(cos_sim)
                
                # Conflict (negative cosine, clipped at 0)
                conflict_value = max(0, -cos_sim)
                conflicts[conflict_key] = conflict_value
                self.gradient_conflicts[conflict_key].append(conflict_value)
                
                # TCI (Task Conflict Index) = 1 - cos_sim (from PCGrad paper)
                tci_value = 1 - cos_sim
                conflicts[tci_key] = tci_value
                self.tci_values[tci_key].append(tci_value)
                
        return conflicts
        
    def detect_conflicts(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Main method to detect conflicts between tasks."""
        metrics = self.compute_gradient_conflicts(model, losses)
        self.iteration += 1
        
        # Log detailed metrics periodically
        if self.iteration % self.log_interval == 0:
            self._log_detailed_metrics()
            
        return metrics
        
    def _log_detailed_metrics(self):
        """Log detailed conflict metrics."""
        print(f"\n=== Task Conflict Metrics (Iteration {self.iteration}) ===")
        
        # Log cosine similarities
        print("\nCosine Similarities:")
        for key, values in self.cosine_similarities.items():
            if values:
                recent = values[-10:]  # Last 10 values
                print(f"  {key}: mean={np.mean(recent):.4f}, "
                      f"min={np.min(recent):.4f}, max={np.max(recent):.4f}")
                
        # Log TCI values
        print("\nTask Conflict Index (TCI):")
        for key, values in self.tci_values.items():
            if values:
                recent = values[-10:]
                print(f"  {key}: mean={np.mean(recent):.4f}, "
                      f"min={np.min(recent):.4f}, max={np.max(recent):.4f}")
                
    def get_summary(self) -> Dict[str, float]:
        """Get summary of conflict metrics."""
        summary = {}
        
        # Average cosine similarities
        for key, values in self.cosine_similarities.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                
        # Average conflicts
        for key, values in self.gradient_conflicts.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_max'] = np.max(values)
                
        # Average TCI values
        for key, values in self.tci_values.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_max'] = np.max(values)
                
        # Overall metrics
        tci_means = [summary[key] for key in summary if 'tci_mean' in key]
        if tci_means:
            summary['overall_tci'] = np.mean(tci_means)
            
        conflict_means = [summary[key] for key in summary if 'conflict_mean' in key]
        if conflict_means:
            summary['overall_conflict'] = np.mean(conflict_means)
            
        return summary