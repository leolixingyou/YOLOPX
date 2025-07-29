# mdo_optimizer.py 文件内容

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MDO_Optimizer:
    def __init__(self, model, num_tasks=3, device='cuda', alpha_itc=0.5, update_freq=20, min_weight=0.1, max_weight=3.0):
        self.model = model
        self.num_tasks = num_tasks
        self.device = device
        self.alpha_itc = alpha_itc # Weight for ITC influence on task weights
        self.update_freq = update_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        self.task_weights = torch.ones(self.num_tasks, device=self.device, requires_grad=False)
        self.initial_losses = None
        self.metrics_history = defaultdict(list)
        self.task_names = ['Detection', 'Driving_Area', 'Lane_Line'] # Assuming these tasks from YOLOPx

    def _get_shared_params(self):
        # Identify shared parameters (typically backbone, excluding task-specific heads)
        model_to_inspect = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        shared_params = [
            p for n, p in model_to_inspect.named_parameters() 
            if p.requires_grad and 
               not any(head_name in n for head_name in ['head', 'det_head', 'da_seg_head', 'll_seg_head', 'seg_head']) 
        ]
        if not shared_params:
            print("Warning: No shared parameters found. Using all trainable parameters as shared.")
            shared_params = [p for p in model_to_inspect.parameters() if p.requires_grad]
        return shared_params

    def compute_weighted_loss_with_gradients(self, head_losses, shared_params, scaler, step_count): # Added shared_params, scaler
        """
        Modified to be compatible with train_fixed's original call signature.
        This method will call the internal MDO logic.
        """
        self.step_count = step_count # Use external step count for consistency
        
        # Ensure losses are tensors and handle None
        losses = [l if l is not None and l.requires_grad else torch.tensor(0.0, device=self.device, requires_grad=True) for l in head_losses[:self.num_tasks]]
        losses = torch.stack(losses)

        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            self.task_weights = torch.ones_like(losses, device=self.device, requires_grad=False)
        
        # We need to zero out gradients here if this is the entry point for the optimizer.
        # However, xy_train_gemini.py already calls optimizer.zero_grad() globally.
        # Per-task gradients are calculated with retain_graph=True, so shared_params.grad isn't accumulated by autograd.grad yet.
        # We also need to zero shared_params.grad before we assign combined_grad to it.
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()

        # 1. Get per-task flattened gradients
        per_task_flattened_grads = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0 and l_i.requires_grad:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            
            flat_g = torch.cat([g.view(-1) for g in unscaled_grads])
            per_task_flattened_grads.append(flat_g)
        
        G_matrix = torch.stack(per_task_flattened_grads) # Shape: (num_tasks, total_param_dim)

        # 2. Compute Inter-Task Correlation (ITC) based on gradients
        correlation_matrix = torch.eye(self.num_tasks, device=self.device)
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                norm_i = torch.norm(G_matrix[i]) + 1e-8
                norm_j = torch.norm(G_matrix[j]) + 1e-8
                
                cosine_sim = torch.dot(G_matrix[i], G_matrix[j]) / (norm_i * norm_j)
                correlation_matrix[i, j] = cosine_sim
                correlation_matrix[j, i] = cosine_sim
        
        # Store for visualization
        self.metrics_history['itc_correlation_matrix'].append(correlation_matrix.cpu().numpy())

        # 3. Dynamically adjust task weights based on losses AND ITC (MDO's spirit)
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                base_weights_loss = 1.0 / (losses.detach() / (self.initial_losses + 1e-8) + 1e-8)
                itc_influence = torch.mean(F.relu(correlation_matrix), dim=1) 
                
                combined_raw_weights = (1 - self.alpha_itc) * base_weights_loss + self.alpha_itc * itc_influence
                
                self.task_weights.data = combined_raw_weights / (combined_raw_weights.sum() + 1e-8) * self.num_tasks
                self.task_weights.data = torch.clamp(self.task_weights.data, self.min_weight, self.max_weight)
        
        # Log current task weights for each step (not just update_freq steps)
        self.metrics_history['mdo_weight_det'].append(self.task_weights[0].item())
        self.metrics_history['mdo_weight_da'].append(self.task_weights[1].item())
        self.metrics_history['mdo_weight_ll'].append(self.task_weights[2].item())


        # 4. Apply adjusted weights to aggregate gradients
        final_combined_grad_flat = torch.sum(self.task_weights.unsqueeze(1) * G_matrix.detach(), dim=0)

        # 5. Reshape and assign the combined gradient back to each shared parameter
        start_idx = 0
        for j, p in enumerate(shared_params):
            param_size = p.numel()
            grad_chunk = final_combined_grad_flat[start_idx : start_idx + param_size].view(p.shape)
            shared_params[j].grad = grad_chunk * scaler.get_scale()
            start_idx += param_size
        
        return losses.sum()

    def visualize_itc_history(self):
        if not self.metrics_history['itc_correlation_matrix']:
            print("No ITC history to visualize.")
            return None # Return None if no plot generated

        # Visualize average ITC over time for each pair
        avg_itc_pairs = defaultdict(list)
        for matrix in self.metrics_history['itc_correlation_matrix']:
            for i in range(self.num_tasks):
                for j in range(i + 1, self.num_tasks):
                    task_pair_name = f"{self.task_names[i]}-{self.task_names[j]}"
                    avg_itc_pairs[task_pair_name].append(matrix[i, j])

        fig, ax = plt.subplots(figsize=(10, 6))
        for pair_name, values in avg_itc_pairs.items():
            ax.plot(values, label=pair_name)

        ax.set_title('Inter-Task Correlation (ITC) Over Time')
        ax.set_xlabel('Training Step (x_update_freq)')
        ax.set_ylabel('Cosine Similarity of Gradients')
        ax.legend()
        ax.grid(True)
        return fig
    
    # Renamed to be compatible with xy_train_gemini.py's expectation
    def get_current_weights(self): 
        # Provide current MDO task weights for logging
        return {
            'mdo_weight_det': self.task_weights[0].item() if self.task_weights.numel() > 0 else float('nan'),
            'mdo_weight_da': self.task_weights[1].item() if self.task_weights.numel() > 1 else float('nan'),
            'mdo_weight_ll': self.task_weights[2].item() if self.task_weights.numel() > 2 else float('nan'),
            'mdo_step_count': self.step_count # Add step count for better context
        }