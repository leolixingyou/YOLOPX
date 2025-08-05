"""
CAGrad (Conflict-Averse Gradient) MTL strategy.

Based on the paper "Conflict-Averse Gradient descent for Multi-task learning" 
(Liu et al., NeurIPS 2021). This strategy finds a gradient that maintains 
non-negative inner products with all task gradients while staying close to 
the average gradient.
"""

import torch
from typing import Dict, List, Any
from .base import MTLStrategy


class CAGradStrategy(MTLStrategy):
    """
    CAGrad strategy that finds conflict-averse gradients.
    
    The strategy solves an optimization problem to find a new gradient that:
    1. Has non-negative inner product with all task gradients
    2. Is as close as possible to the average gradient
    """
    
    def __init__(self, model, optimizer, num_tasks=3, task_names=None, c=0.5):
        """
        Initialize the CAGrad strategy.
        
        Args:
            model: The multi-task model
            optimizer: The optimizer for model parameters
            num_tasks: Number of tasks (default: 3 for YOLOP)
            task_names: Names of tasks (default: ['detection', 'da_seg', 'll_seg'])
            c: Conflict aversion parameter (default: 0.5)
               - Higher c means stronger conflict aversion
               - c in [0, 1] typically works well
        """
        super().__init__(model, optimizer, num_tasks, task_names)
        self.c = c
        
    def backward(self, losses: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Perform CAGrad backward pass with conflict-averse gradient computation.
        
        Args:
            losses: Dictionary mapping task names to their loss values
            **kwargs: Additional parameters (e.g., 'c' to override default)
            
        Returns:
            Dictionary containing:
            - total_loss: The sum of all task losses
            - num_conflicts: Number of conflicting task pairs
            - avg_conflict_magnitude: Average magnitude of conflicts
            - cagrad_c: The c parameter used
        """
        # Override c if provided
        c = kwargs.get('c', self.c)
        
        # Compute gradients for each task
        task_gradients = self.compute_task_gradients(losses)
        
        # Get shapes for unflattening later
        shapes = [g.shape for g in task_gradients[self.task_names[0]]]
        
        # Flatten gradients for each task
        flat_grads = []
        task_order = []
        for task_name in self.task_names:
            if task_name in task_gradients:
                flat_grads.append(self.flatten_gradients(task_gradients[task_name]))
                task_order.append(task_name)
        
        if not flat_grads:
            # No gradients to process
            total_loss = sum(losses.values())
            return {
                'total_loss': total_loss.item(),
                'num_conflicts': 0,
                'avg_conflict_magnitude': 0.0,
                'cagrad_c': c,
                'strategy': 'cagrad'
            }
        
        # Stack gradients into matrix (tasks x parameters)
        G = torch.stack(flat_grads)  # Shape: (num_tasks, num_params)
        
        # Compute average gradient
        g_avg = G.mean(dim=0)
        
        # Check for conflicts
        num_conflicts = 0
        total_conflict_magnitude = 0.0
        
        # Compute inner products between average gradient and task gradients
        inner_products = torch.matmul(G, g_avg)
        
        # Check which tasks conflict with average
        conflicting_tasks = []
        for i, inner_prod in enumerate(inner_products):
            if inner_prod < 0:
                num_conflicts += 1
                total_conflict_magnitude += abs(inner_prod.item())
                conflicting_tasks.append(i)
        
        # If no conflicts, use average gradient
        if num_conflicts == 0:
            final_grad = g_avg
        else:
            # Apply CAGrad algorithm to find conflict-averse gradient
            final_grad = self._cagrad_optimization(G, g_avg, c)
        
        # Unflatten and apply the final gradient
        final_grads_unflat = self.unflatten_gradients(final_grad, shapes)
        self.apply_gradients(final_grads_unflat)
        
        # Compute total loss for logging
        total_loss = sum(losses.values())
        
        # Return info for logging
        avg_conflict = total_conflict_magnitude / max(num_conflicts, 1)
        info = {
            'total_loss': total_loss.item(),
            'num_conflicts': num_conflicts,
            'avg_conflict_magnitude': avg_conflict,
            'cagrad_c': c,
            'strategy': 'cagrad'
        }
        
        return info
    
    def _cagrad_optimization(self, G: torch.Tensor, g_avg: torch.Tensor, 
                            c: float, max_iter: int = 100, lr: float = 0.1) -> torch.Tensor:
        """
        Solve the CAGrad optimization problem using gradient descent.
        
        The problem is formulated as finding alpha that minimizes:
        ||sum_i alpha_i * g_i - g_avg||^2
        subject to: alpha_i >= 0 for all i
        
        Args:
            G: Matrix of task gradients (num_tasks x num_params)
            g_avg: Average gradient
            c: Conflict aversion parameter
            max_iter: Maximum iterations for optimization
            lr: Learning rate for optimization
            
        Returns:
            The conflict-averse gradient
        """
        num_tasks = G.shape[0]
        device = G.device
        
        # Initialize alpha (weights for each task gradient)
        alpha = torch.ones(num_tasks, device=device) / num_tasks
        alpha.requires_grad_(True)
        
        # Compute G @ G^T for efficiency
        GG = torch.matmul(G, G.t())
        Gg_avg = torch.matmul(G, g_avg)
        
        # Optimization loop
        optimizer = torch.optim.Adam([alpha], lr=lr)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            
            # Compute weighted gradient: sum_i alpha_i * g_i
            weighted_grad = torch.matmul(alpha, G)
            
            # Compute loss: ||weighted_grad - g_avg||^2 + c * constraint
            diff = weighted_grad - g_avg
            loss = torch.dot(diff, diff)
            
            # Add penalty for negative inner products
            # This encourages the solution to have non-negative inner products
            # with all task gradients
            inner_prods = torch.matmul(G, weighted_grad)
            penalty = torch.sum(torch.relu(-inner_prods))
            loss = loss + c * penalty
            
            # Backward and update
            loss.backward()
            optimizer.step()
            
            # Project alpha to simplex (ensure sum to 1 and non-negative)
            with torch.no_grad():
                # Ensure non-negative
                alpha.clamp_(min=0)
                # Normalize to sum to 1
                alpha_sum = alpha.sum()
                if alpha_sum > 0:
                    alpha /= alpha_sum
                else:
                    # If all zeros, reset to uniform
                    alpha.fill_(1.0 / num_tasks)
        
        # Compute final gradient
        with torch.no_grad():
            final_grad = torch.matmul(alpha.detach(), G)
        
        return final_grad