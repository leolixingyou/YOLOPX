"""
GradNorm (Gradient Normalization) MTL strategy.

Based on the paper "GradNorm: Gradient Normalization for Adaptive Loss Balancing 
in Deep Multitask Networks" (Chen et al., ICML 2018). This strategy dynamically 
learns task weights to balance gradient magnitudes and training rates across tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from .base import MTLStrategy


class GradNormStrategy(MTLStrategy):
    """
    GradNorm strategy that dynamically adjusts task weights.
    
    The strategy learns weights for each task loss to ensure:
    1. All tasks have similar gradient magnitudes
    2. All tasks train at similar rates (based on loss ratios)
    """
    
    def __init__(self, model, optimizer, num_tasks=3, task_names=None, 
                 alpha=1.5, update_freq=20, weight_lr=0.025):
        """
        Initialize the GradNorm strategy.
        
        Args:
            model: The multi-task model
            optimizer: The optimizer for model parameters
            num_tasks: Number of tasks (default: 3 for YOLOP)
            task_names: Names of tasks (default: ['detection', 'da_seg', 'll_seg'])
            alpha: Restoring force strength (default: 1.5)
                   - alpha > 1: Prioritizes tasks that train slowly
                   - alpha < 1: Prioritizes tasks that decrease quickly
                   - alpha = 0: Only balances gradient magnitudes
            update_freq: How often to update weights (default: 20 steps)
            weight_lr: Learning rate for weight updates (default: 0.025)
        """
        super().__init__(model, optimizer, num_tasks, task_names)
        
        self.alpha = alpha
        self.update_freq = update_freq
        
        # Initialize learnable task weights
        self.weights = nn.Parameter(torch.ones(num_tasks, device=next(model.parameters()).device))
        self.weights.requires_grad = True
        
        # Create optimizer for weights
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=weight_lr)
        
        # Track initial losses and training progress
        self.initial_losses = None
        self.loss_history = {name: [] for name in self.task_names}
        self.step_count = 0
        
        # Get the last shared layer for gradient computation
        self.last_shared_layer = self._get_last_shared_layer()
        
    def _get_last_shared_layer(self) -> Optional[nn.Module]:
        """
        Find the last shared layer in the model.
        
        Returns:
            The last shared layer module, or None if not found
        """
        # For YOLOP models, the last shared layer is typically before the task heads
        # We'll look for the last layer that's part of the backbone
        last_layer = None
        
        for name, module in self.model.named_modules():
            # Skip task-specific modules
            if any(head in name.lower() for head in ['head', 'seg', 'detect', 'da_', 'll_']):
                continue
            
            # Look for convolutional or linear layers in the backbone
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if any(keyword in name.lower() for keyword in ['backbone', 'encoder', 'shared', 'conv', 'layer']):
                    last_layer = module
        
        return last_layer
    
    def backward(self, losses: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Perform GradNorm backward pass with dynamic weight adjustment.
        
        Args:
            losses: Dictionary mapping task names to their loss values
            **kwargs: Additional parameters (e.g., 'alpha' to override default)
            
        Returns:
            Dictionary containing:
            - total_loss: The weighted sum of task losses
            - task_weights: Current task weights
            - loss_ratios: Current loss ratios relative to initial
            - grad_norms: Gradient norms for each task
        """
        alpha = kwargs.get('alpha', self.alpha)
        
        # Convert losses dict to ordered list
        loss_list = []
        for task_name in self.task_names:
            if task_name in losses:
                loss_list.append(losses[task_name])
        
        if not loss_list:
            # No losses to process
            return {
                'total_loss': 0.0,
                'task_weights': {},
                'loss_ratios': {},
                'grad_norms': {},
                'strategy': 'gradnorm'
            }
        
        # Record initial losses
        if self.initial_losses is None:
            with torch.no_grad():
                self.initial_losses = [loss.item() for loss in loss_list]
        
        # Update loss history
        for i, (task_name, loss) in enumerate(zip(self.task_names, loss_list)):
            self.loss_history[task_name].append(loss.item())
        
        # Normalize weights to sum to num_tasks
        with torch.no_grad():
            self.weights.data = self.weights.data * self.num_tasks / self.weights.data.sum()
        
        # Compute weighted loss
        weighted_losses = []
        for i, loss in enumerate(loss_list):
            weighted_losses.append(self.weights[i] * loss)
        
        total_loss = sum(weighted_losses)
        
        # Standard backward for model parameters
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        
        # Update weights using GradNorm algorithm
        if self.step_count % self.update_freq == 0 and self.step_count > 0:
            self._update_weights(loss_list, alpha)
        
        self.step_count += 1
        
        # Collect info for logging
        with torch.no_grad():
            weight_dict = {name: self.weights[i].item() 
                          for i, name in enumerate(self.task_names[:len(loss_list)])}
            
            # Compute loss ratios
            loss_ratios = {}
            for i, task_name in enumerate(self.task_names[:len(loss_list)]):
                if self.initial_losses[i] > 0:
                    ratio = loss_list[i].item() / self.initial_losses[i]
                    loss_ratios[task_name] = ratio
                else:
                    loss_ratios[task_name] = 1.0
        
        info = {
            'total_loss': total_loss.item(),
            'task_weights': weight_dict,
            'loss_ratios': loss_ratios,
            'strategy': 'gradnorm'
        }
        
        return info
    
    def _update_weights(self, losses: List[torch.Tensor], alpha: float):
        """
        Update task weights using the GradNorm algorithm.
        
        Args:
            losses: List of task losses
            alpha: Restoring force strength
        """
        if self.last_shared_layer is None:
            # Cannot compute GradNorm without shared layer
            return
        
        # Get the parameters of the last shared layer
        W = list(self.last_shared_layer.parameters())
        if not W:
            return
        W = W[0]  # Use the first parameter (usually weight)
        
        # Compute gradients for each task w.r.t. shared parameters
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute gradient of weighted loss w.r.t. W
            self.optimizer.zero_grad()
            weighted_loss = self.weights[i] * loss
            grads = torch.autograd.grad(weighted_loss, W, retain_graph=True, create_graph=True)[0]
            
            # Compute L2 norm of gradients
            grad_norm = torch.norm(grads, p=2)
            grad_norms.append(grad_norm)
        
        # Stack gradient norms
        grad_norms = torch.stack(grad_norms)
        
        # Compute average gradient norm
        mean_grad_norm = grad_norms.mean()
        
        # Compute relative inverse training rates
        loss_ratios = []
        for i in range(len(losses)):
            if self.initial_losses[i] > 0:
                ratio = losses[i].item() / self.initial_losses[i]
            else:
                ratio = 1.0
            loss_ratios.append(ratio)
        
        loss_ratios = torch.tensor(loss_ratios, device=grad_norms.device)
        relative_inverse_rates = loss_ratios / loss_ratios.mean()
        
        # Compute target gradient norms
        target_grad_norms = mean_grad_norm * (relative_inverse_rates ** alpha)
        
        # Compute GradNorm loss
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms.detach()).sum()
        
        # Update weights
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()
        
        # Ensure weights are positive and normalized
        with torch.no_grad():
            self.weights.data = torch.relu(self.weights.data)
            self.weights.data = self.weights.data * self.num_tasks / self.weights.data.sum()