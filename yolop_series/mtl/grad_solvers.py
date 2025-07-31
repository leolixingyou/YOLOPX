"""
Gradient-level Multi-Task Learning Solvers.
Contains implementations of GradNorm, PCGrad, and CAGrad.
"""
import torch
import torch.nn.functional as F
import numpy as np

class FixedGradientConflictSolver:
    """
    A unified solver for various gradient-based MTL methods.
    This class can either return a weighted loss (GradNorm) or directly
    manipulate gradients of shared parameters (PCGrad, CAGrad).
    """
    def __init__(self, method='gradnorm', num_tasks=3, device='cuda', **kwargs):
        self.method = method
        self.num_tasks = num_tasks
        self.device = device
        self.step_count = 0
        
        if method == 'gradnorm':
            self.task_weights = torch.ones(num_tasks, device=device)
            self.initial_losses = None
            self.alpha = kwargs.get('alpha', 1.5)
            self.update_freq = kwargs.get('update_freq', 50)
        elif method == 'cagrad':
            self.c = kwargs.get('c', 0.5)
        elif method not in ['pcgrad', 'tag', 'original']:
             raise ValueError(f"Unknown conflict resolution method: {method}")

    def compute_weighted_loss_with_gradients(self, head_losses, shared_params, scaler, step_count):
        """
        Main entry point for the solver.
        """
        self.step_count = step_count
        losses = torch.stack([l for l in head_losses if l is not None])

        if self.method == 'gradnorm':
            return self._gradnorm_loss(losses)
        
        # For methods that manipulate gradients directly
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()

        if self.method == 'pcgrad':
            self._pcgrad_grads(losses, shared_params, scaler)
        elif self.method == 'cagrad':
            self._cagrad_grads(losses, shared_params, scaler)
        elif self.method == 'tag':
            # For TAG, the primary mechanism is architectural.
            # The gradient strategy is a simple sum of gradients from task-specific features.
            self._sum_grads(losses, shared_params, scaler)
        
        return losses.sum()

    def _get_per_task_grads(self, losses, shared_params, scaler):
        """Helper to compute unscaled gradients for each task."""
        per_task_grads = []
        for loss in losses:
            scaled_loss = scaler.scale(loss)
            grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
            unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            per_task_grads.append(unscaled_grads)
        return per_task_grads

    def _sum_grads(self, losses, shared_params, scaler):
        """Simple gradient summation, used by TAG."""
        per_task_grads = self._get_per_task_grads(losses, shared_params, scaler)
        # Sum gradients across tasks for each parameter
        combined_grads = [torch.sum(torch.stack(grads), dim=0) for grads in zip(*per_task_grads)]
        for p, grad in zip(shared_params, combined_grads):
            p.grad = grad * scaler.get_scale()

    def _gradnorm_loss(self, losses):
        """GradNorm: Dynamically adjusts loss weights."""
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
        
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                loss_ratios = losses.detach() / self.initial_losses
                avg_loss_ratio = loss_ratios.mean()
                target_weights = avg_loss_ratio / loss_ratios
                self.task_weights = F.softmax(target_weights, dim=0) * self.num_tasks
        
        return (self.task_weights * losses).sum()

    def _pcgrad_grads(self, losses, shared_params, scaler):
        """PCGrad: Projects conflicting gradients."""
        per_task_grads = self._get_per_task_grads(losses, shared_params, scaler)
        
        # Project gradients for each parameter
        for j in range(len(shared_params)):
            # Get all task gradients for the j-th parameter
            param_grads = torch.stack([per_task_grads[i][j] for i in range(self.num_tasks)])
            
            for i in range(self.num_tasks):
                g_i = param_grads[i]
                for k in range(self.num_tasks):
                    if i != k:
                        g_k = param_grads[k]
                        dot_product = torch.sum(g_i * g_k)
                        if dot_product < 0:
                            # Project g_i onto the normal plane of g_k
                            g_i -= (dot_product / torch.sum(g_k * g_k)) * g_k
            
            # Sum the projected gradients and assign back
            shared_params[j].grad = torch.sum(param_grads, dim=0) * scaler.get_scale()

    def _cagrad_grads(self, losses, shared_params, scaler):
        """CAGrad: Finds a gradient that minimally interferes with others."""
        per_task_grads = self._get_per_task_grads(losses, shared_params, scaler)
        
        # Flatten gradients
        G_matrix = torch.stack([torch.cat([g.view(-1) for g in task_grads]) for task_grads in per_task_grads])
        
        g0 = G_matrix.mean(dim=0)
        g0_norm_sq = torch.dot(g0, g0)

        x_start = torch.zeros(self.num_tasks, device=self.device)
        
        # Solve the dual problem
        A = G_matrix @ G_matrix.T
        b = G_matrix @ g0
        
        # Simple gradient descent to solve for alphas
        alphas = F.softmax(x_start, dim=0)
        for _ in range(5): # Few steps are usually enough
            grad = 2 * A @ alphas - 2 * b + 2 * self.c * g0_norm_sq * (alphas.sum() - 1)
            alphas = alphas - 0.1 * grad
            alphas = F.softmax(alphas, dim=0)

        final_grad_flat = alphas.unsqueeze(0) @ G_matrix
        
        # Un-flatten and assign back
        start_idx = 0
        for p in shared_params:
            param_size = p.numel()
            grad_chunk = final_grad_flat[0, start_idx : start_idx + param_size].view(p.shape)
            p.grad = grad_chunk * scaler.get_scale()
            start_idx += param_size
