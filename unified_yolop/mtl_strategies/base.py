"""
Base class for Multi-Task Learning (MTL) strategies.

This module defines the abstract interface that all MTL gradient manipulation
strategies must implement. The strategies operate during the backward pass to
modify gradients before they are applied to the model parameters.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any


class MTLStrategy(ABC):
    """
    Abstract base class for Multi-Task Learning gradient strategies.
    
    All MTL strategies should inherit from this class and implement the
    backward() method to define how gradients are computed and manipulated.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 num_tasks: int = 3, task_names: Optional[List[str]] = None):
        """
        Initialize the MTL strategy.
        
        Args:
            model: The multi-task model
            optimizer: The optimizer for model parameters
            num_tasks: Number of tasks (default: 3 for YOLOP)
            task_names: Names of tasks (default: ['detection', 'da_seg', 'll_seg'])
        """
        self.model = model
        self.optimizer = optimizer
        self.num_tasks = num_tasks
        self.task_names = task_names or ['detection', 'da_seg', 'll_seg']
        
        # Get shared parameters (backbone parameters)
        self.shared_params = self._get_shared_parameters()
        
    def _get_shared_parameters(self) -> List[nn.Parameter]:
        """
        Get the shared parameters of the model (typically the backbone).
        
        Returns:
            List of shared parameters
        """
        shared_params = []
        
        # For YOLOP models, shared parameters are typically in the backbone
        for name, param in self.model.named_parameters():
            # Common backbone module names across different YOLOP variants
            if any(keyword in name.lower() for keyword in ['backbone', 'encoder', 'shared', 'stem']):
                if param.requires_grad:
                    shared_params.append(param)
            # Also include early layers that are typically shared
            elif any(name.startswith(prefix) for prefix in ['conv', 'layer', 'stage']):
                # But exclude task-specific heads
                if not any(head in name.lower() for head in ['head', 'seg', 'detect', 'da_', 'll_']):
                    if param.requires_grad:
                        shared_params.append(param)
        
        # If no shared parameters found, assume all parameters are shared
        # (this is a fallback for models with different naming conventions)
        if not shared_params:
            shared_params = [p for p in self.model.parameters() if p.requires_grad]
            
        return shared_params
    
    @abstractmethod
    def backward(self, losses: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Compute and apply gradients based on the MTL strategy.
        
        This method should:
        1. Calculate gradients for each task
        2. Apply the strategy-specific gradient manipulation
        3. Apply the final gradients to the model
        
        Args:
            losses: Dictionary mapping task names to their loss values
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary containing strategy-specific information (e.g., task weights,
            conflict metrics, etc.) for logging purposes
        """
        pass
    
    def step(self):
        """
        Perform the optimizer step.
        
        This is typically called after backward() to update the model parameters.
        Most strategies will just call the optimizer's step() method, but some
        (like GradNorm) may need custom logic here.
        """
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero the gradients of the optimizer."""
        self.optimizer.zero_grad()
    
    def compute_task_gradients(self, losses: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Compute gradients for each task separately.
        
        Args:
            losses: Dictionary mapping task names to their loss values
            
        Returns:
            Dictionary mapping task names to lists of gradient tensors
        """
        task_gradients = {}
        
        for task_name, loss in losses.items():
            # Zero gradients before computing for this task
            self.optimizer.zero_grad()
            
            # Compute gradients for this task
            loss.backward(retain_graph=True)
            
            # Store gradients for shared parameters
            task_grads = []
            for param in self.shared_params:
                if param.grad is not None:
                    task_grads.append(param.grad.clone())
                else:
                    # If no gradient, use zeros
                    task_grads.append(torch.zeros_like(param))
                    
            task_gradients[task_name] = task_grads
            
        # Clear gradients after collecting them
        self.optimizer.zero_grad()
        
        return task_gradients
    
    def flatten_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten a list of gradient tensors into a single vector.
        
        Args:
            gradients: List of gradient tensors
            
        Returns:
            Flattened gradient vector
        """
        return torch.cat([g.flatten() for g in gradients])
    
    def unflatten_gradients(self, flattened: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        """
        Unflatten a gradient vector back to the original shapes.
        
        Args:
            flattened: Flattened gradient vector
            shapes: List of original tensor shapes
            
        Returns:
            List of gradient tensors with original shapes
        """
        gradients = []
        offset = 0
        
        for shape in shapes:
            numel = shape.numel()
            gradient = flattened[offset:offset + numel].reshape(shape)
            gradients.append(gradient)
            offset += numel
            
        return gradients
    
    def apply_gradients(self, gradients: List[torch.Tensor]):
        """
        Apply gradients to the shared parameters.
        
        Args:
            gradients: List of gradient tensors to apply
        """
        for param, grad in zip(self.shared_params, gradients):
            param.grad = grad.clone()