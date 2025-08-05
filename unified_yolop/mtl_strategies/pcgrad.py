"""
PCGrad (Projected Gradient) MTL strategy.

Based on the paper "Gradient Surgery for Multi-Task Learning" (Yu et al., NeurIPS 2020).
This strategy projects conflicting gradients to eliminate negative interference between tasks.
"""

import torch
from typing import Dict, List, Any
from .base import MTLStrategy


class PCGradStrategy(MTLStrategy):
    """
    PCGrad strategy that performs gradient surgery to resolve conflicts.
    
    When two task gradients have negative cosine similarity (conflicting),
    PCGrad projects one gradient onto the normal plane of the other to
    remove the conflicting component.
    """
    
    def __init__(self, model, optimizer, num_tasks=3, task_names=None):
        """
        Initialize the PCGrad strategy.
        
        Args:
            model: The multi-task model
            optimizer: The optimizer for model parameters
            num_tasks: Number of tasks (default: 3 for YOLOP)
            task_names: Names of tasks (default: ['detection', 'da_seg', 'll_seg'])
        """
        super().__init__(model, optimizer, num_tasks, task_names)
        
    def backward(self, losses: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Perform PCGrad backward pass with gradient projection.
        
        Args:
            losses: Dictionary mapping task names to their loss values
            **kwargs: Unused in this strategy
            
        Returns:
            Dictionary containing:
            - total_loss: The sum of all task losses
            - num_conflicts: Number of gradient conflicts detected
            - avg_conflict_magnitude: Average magnitude of conflicts
        """
        # Compute gradients for each task
        task_gradients = self.compute_task_gradients(losses)
        
        # Get shapes for unflattening later
        shapes = [g.shape for g in task_gradients[self.task_names[0]]]
        
        # Flatten gradients for each task
        flat_grads = {}
        for task_name in self.task_names:
            if task_name in task_gradients:
                flat_grads[task_name] = self.flatten_gradients(task_gradients[task_name])
        
        # Perform PCGrad algorithm
        num_conflicts = 0
        total_conflict_magnitude = 0.0
        
        # Create a copy of gradients that will be modified
        pc_grads = {name: grad.clone() for name, grad in flat_grads.items()}
        
        # Compare all pairs of tasks
        task_list = list(pc_grads.keys())
        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                task_i, task_j = task_list[i], task_list[j]
                grad_i = pc_grads[task_i]
                grad_j = pc_grads[task_j]
                
                # Compute cosine similarity
                dot_product = torch.dot(grad_i, grad_j)
                norm_i = torch.norm(grad_i, p=2)
                norm_j = torch.norm(grad_j, p=2)
                
                # Avoid division by zero
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = dot_product / (norm_i * norm_j)
                    
                    # If gradients conflict (negative cosine similarity)
                    if cos_sim < 0:
                        num_conflicts += 1
                        total_conflict_magnitude += abs(cos_sim.item())
                        
                        # Project grad_i onto the normal plane of grad_j
                        # g_i_new = g_i - (g_i · g_j / ||g_j||^2) * g_j
                        proj_i = dot_product / (norm_j ** 2) * grad_j
                        pc_grads[task_i] = grad_i - proj_i
                        
                        # For symmetric treatment, also project grad_j
                        # Note: In the original paper, they use random order,
                        # but symmetric projection often works better
                        grad_i_new = pc_grads[task_i]
                        dot_product_new = torch.dot(grad_i_new, grad_j)
                        if dot_product_new < 0:
                            proj_j = dot_product_new / torch.norm(grad_i_new, p=2) ** 2 * grad_i_new
                            pc_grads[task_j] = grad_j - proj_j
        
        # Sum the projected gradients
        final_grad = sum(pc_grads.values())
        
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
            'strategy': 'pcgrad'
        }
        
        return info