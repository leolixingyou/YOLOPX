"""
Original MTL strategy - Simple loss summation without any gradient manipulation.

This serves as the baseline method where gradients from all tasks are simply
added together. No conflict resolution or gradient modification is performed.
"""

import torch
from typing import Dict, Any
from .base import MTLStrategy


class OriginalStrategy(MTLStrategy):
    """
    The baseline MTL strategy that simply sums all task losses.
    
    This is equivalent to the standard multi-task learning approach where:
    L_total = Σ_i L_i
    
    No gradient manipulation or conflict resolution is performed.
    """
    
    def __init__(self, model, optimizer, num_tasks=3, task_names=None):
        """
        Initialize the Original strategy.
        
        Args:
            model: The multi-task model
            optimizer: The optimizer for model parameters
            num_tasks: Number of tasks (default: 3 for YOLOP)
            task_names: Names of tasks (default: ['detection', 'da_seg', 'll_seg'])
        """
        super().__init__(model, optimizer, num_tasks, task_names)
        
    def backward(self, losses: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Perform standard backward pass with summed losses.
        
        Args:
            losses: Dictionary mapping task names to their loss values
            **kwargs: Unused in this strategy
            
        Returns:
            Dictionary containing:
            - total_loss: The sum of all task losses
            - task_weights: Equal weights (1.0) for all tasks
        """
        # Simply sum all losses
        total_loss = sum(losses.values())
        
        # Perform standard backward pass
        total_loss.backward()
        
        # Return info for logging
        info = {
            'total_loss': total_loss.item(),
            'task_weights': {name: 1.0 for name in self.task_names},
            'strategy': 'original'
        }
        
        return info