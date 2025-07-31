"""
Heuristic Multi-Task Learning Solver.
This is a custom implementation inspired by MDO principles but
operates as a dynamic, gradient-based weighting scheme.
"""
import torch
import torch.nn.functional as F

class MDO_Optimizer:
    """
    A heuristic optimizer that dynamically adjusts task weights based on
    a combination of loss ratios and inter-task gradient similarity (ITC).
    """
    def __init__(self, model, num_tasks=3, device='cuda', update_freq=50, alpha_itc=0.5):
        self.model = model
        self.num_tasks = num_tasks
        self.device = device
        self.update_freq = update_freq
        self.alpha_itc = alpha_itc # Balance between loss-based and ITC-based weighting
        
        self.task_weights = torch.ones(num_tasks, device=device)
        self.initial_losses = None

    def compute_weighted_loss_with_gradients(self, head_losses, shared_params, scaler, step_count):
        """
        Calculates and applies a combined gradient to shared parameters.
        """
        losses = torch.stack([l for l in head_losses if l is not None])

        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()

        # --- Get per-task gradients ---
        per_task_grads_flat = []
        for loss in losses:
            scaled_loss = scaler.scale(loss)
            grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
            unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            per_task_grads_flat.append(torch.cat([g.view(-1) for g in unscaled_grads]))
        
        G_matrix = torch.stack(per_task_grads_flat)

        # --- Dynamically adjust task weights ---
        if step_count % self.update_freq == 0:
            with torch.no_grad():
                # 1. Loss-based weight component (similar to GradNorm)
                loss_ratios = losses / self.initial_losses
                loss_weights = loss_ratios.mean() / loss_ratios

                # 2. ITC-based weight component
                cos_sim = F.cosine_similarity(G_matrix.unsqueeze(1), G_matrix.unsqueeze(0), dim=2)
                # We want to give higher weight to tasks that agree with others (positive correlation)
                itc_influence = F.relu(cos_sim).mean(dim=1)

                # 3. Combine weights
                combined_weights = (1 - self.alpha_itc) * loss_weights + self.alpha_itc * itc_influence
                self.task_weights = F.softmax(combined_weights, dim=0) * self.num_tasks
        
        # --- Apply weights and set gradients ---
        final_grad_flat = self.task_weights.unsqueeze(0) @ G_matrix
        
        start_idx = 0
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()
            param_size = p.numel()
            grad_chunk = final_grad_flat[0, start_idx : start_idx + param_size].view(p.shape)
            p.grad = grad_chunk * scaler.get_scale()
            start_idx += param_size
            
        return losses.sum()
