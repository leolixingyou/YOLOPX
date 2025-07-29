import torch
import torch.nn.functional as F
import numpy as np

class FixedGradientConflictSolver:
    """修正的梯度冲突解决器"""
    
    def __init__(self, method='gradnorm', num_tasks=3, device='cuda', **kwargs):
        self.method = method
        self.num_tasks = num_tasks
        self.device = device
        self.step_count = 0
        
        if method == 'gradnorm':
            self.task_weights = torch.ones(num_tasks, device=device, requires_grad=False)
            self.initial_losses = None
            self.alpha = kwargs.get('alpha', 1.5)
            self.update_freq = kwargs.get('update_freq', 10)
            
        elif method == 'pcgrad':
            pass 
            
        elif method == 'cagrad':
            self.c = kwargs.get('c', 0.5) 
            
        elif method == 'mdo':
            self.initial_losses = None
            self.task_weights = torch.ones(num_tasks, device=device, requires_grad=False)
            self.correlation_history = []
            self.update_freq = kwargs.get('update_freq', 20) # MDO specific update_freq
            
        elif method == 'tag':
            self.update_freq = kwargs.get('update_freq', 20) # TAG specific update_freq, though its primary mechanism is architectural
            pass
        else:
            raise ValueError(f"Unknown conflict resolution method: {method}")
    
    def compute_weighted_loss_with_gradients(self, head_losses, shared_params, scaler, step_count):
        """
        根据冲突解决策略计算损失或直接修改梯度。
        对于PCGrad, CAGrad, MDO, TAG，本方法将直接在 shared_params 上设置梯度。
        
        Args:
            head_losses: 各个任务的原始损失 (列表或张量)。
            shared_params: 共享参数的列表，用于计算每个任务的独立梯度。
            scaler: GradScaler (用于AMP)。
        Returns:
            返回一个标量损失，用于后续的 scaler.scale().backward()。
            对于 PCGrad, CAGrad, MDO, TAG，这个返回值主要是为了兼容外部流程，
            实际的梯度已经在此方法内部处理。
        """
        # Filter out None losses and ensure they are tensors with requires_grad
        # If a loss is 0.0 or None, its gradient won't contribute, which is fine.
        # Ensure losses used for stack have requires_grad=True
        self.step_count = step_count  # Use external step count for consistency
        losses = [l if l is not None and l.requires_grad else torch.tensor(0.0, device=self.device, requires_grad=True) for l in head_losses[:self.num_tasks]]
        losses = torch.stack(losses)

        # For GradNorm, directly return weighted loss.
        if self.method == 'gradnorm':
            return self._gradnorm_loss(losses)
        
        # For PCGrad, CAGrad, MDO, TAG, they directly modify gradients, then return a sum loss.
        # Ensure that shared_params gradients are zeroed before we manually set them.
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()

        # Select processing logic based on method
        if self.method == 'pcgrad':
            self._pcgrad_grads(losses, shared_params, scaler)
        elif self.method == 'cagrad':
            self._cagrad_grads(losses, shared_params, scaler)
        elif self.method == 'mdo':
            self._mdo_logic(losses, shared_params, scaler)
        elif self.method == 'tag':
            self._tag_logic(losses, shared_params, scaler)
        
        # Return the sum of raw losses for the external scaler.scale().backward() to be called.
        # At this point, the gradients for shared parameters have already been set internally.
        return losses.sum()

    def _gradnorm_loss(self, losses):
        """改进的GradNorm实现"""
        self.step_count += 1
        
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            self.task_weights = torch.ones_like(losses, device=self.device, requires_grad=False)
            return losses.sum()
        
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # Avoid division by zero and log(0) issues
                loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
                avg_loss_ratio = loss_ratios.mean()
                target_weights = avg_loss_ratio / (loss_ratios + 1e-8)
                
                momentum = 0.1
                self.task_weights.data = (1 - momentum) * self.task_weights.data + momentum * target_weights
                self.task_weights.data = self.num_tasks * self.task_weights.data / (self.task_weights.sum().item() + 1e-8)
                self.task_weights.data = torch.clamp(self.task_weights.data, 0.1, 3.0) # Clamp to reasonable range
        
        return (self.task_weights * losses).sum()

    def _pcgrad_grads(self, losses, shared_params, scaler):
        """PCGrad: Projecting Conflicting Gradients."""
        per_task_grads_list = [] 
        for i, l_i in enumerate(losses):
            if l_i.item() != 0 and l_i.requires_grad:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            
            per_task_grads_list.append(unscaled_grads)

        grads_by_param = [[] for _ in range(len(shared_params))]
        for task_grads in per_task_grads_list:
            for j, p_grad in enumerate(task_grads):
                grads_by_param[j].append(p_grad)

        for j, param_grads in enumerate(grads_by_param):
            if not param_grads: continue
            
            stacked_param_grads = torch.stack(param_grads)
            
            for i in range(self.num_tasks):
                g_i = stacked_param_grads[i]
                for k in range(self.num_tasks):
                    if i != k:
                        g_k = stacked_param_grads[k]
                        
                        dot_product = torch.sum(g_i * g_k)
                        norm_gk_sq = torch.sum(g_k * g_k)
                        
                        if norm_gk_sq != 0 and dot_product < 0:
                            g_i_projected = g_i - (dot_product / norm_gk_sq) * g_k
                            stacked_param_grads[i] = g_i_projected 
            
            combined_grad = torch.sum(stacked_param_grads, dim=0)
            shared_params[j].grad = combined_grad * scaler.get_scale()

    def _cagrad_grads(self, losses, shared_params, scaler):
        """CAGrad: Conflict-Averse Gradient Descent."""
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0 and l_i.requires_grad:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            per_task_grads_list.append(unscaled_grads)

        flattened_grads = []
        for task_grads in per_task_grads_list:
            flat_g = torch.cat([g.view(-1) for g in task_grads])
            flattened_grads.append(flat_g)
        
        G_matrix = torch.stack(flattened_grads) 
        g_hat = G_matrix.mean(dim=0) 

        alphas = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        alphas.requires_grad_(True)
        
        optimizer_alphas = torch.optim.Adam([alphas], lr=0.01)

        for _ in range(50): 
            optimizer_alphas.zero_grad()
            combined_grad = torch.sum(alphas.unsqueeze(1) * G_matrix, dim=0)
            loss_qp = torch.sum(combined_grad * combined_grad)
            
            g_hat_norm_sq = torch.sum(g_hat * g_hat)
            dot_prod_constraint = torch.sum(combined_grad * g_hat)
            
            penalty = 0.0
            if dot_prod_constraint < self.c * g_hat_norm_sq:
                penalty = (self.c * g_hat_norm_sq - dot_prod_constraint) ** 2 
            
            loss_qp_total = loss_qp + 1000 * penalty 
            loss_qp_total.backward()
            optimizer_alphas.step()
            
            with torch.no_grad():
                alphas.data = F.relu(alphas.data)
                alphas.data = alphas.data / (alphas.data.sum() + 1e-8)

        final_combined_grad = torch.sum(alphas.unsqueeze(1) * G_matrix.detach(), dim=0)
        
        start_idx = 0
        for j, p in enumerate(shared_params):
            param_size = p.numel()
            grad_chunk = final_combined_grad[start_idx : start_idx + param_size].view(p.shape)
            shared_params[j].grad = grad_chunk * scaler.get_scale() 
            start_idx += param_size

# xy_conflict_solver_gemini.py 中 _mdo_logic 的修改

    def _mdo_logic(self, losses, shared_params, scaler):
        """MDO: Dynamically adjusts task weights based on losses."""
        self.step_count += 1
        
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            self.task_weights = torch.ones_like(losses, device=self.device, requires_grad=False)
            
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # A heuristic for dynamic weighting: inversely proportional to current loss
                raw_weights = 1.0 / (losses.detach() + 1e-8)
                self.task_weights.data = raw_weights / (raw_weights.sum() + 1e-8) * self.num_tasks
                self.task_weights.data = torch.clamp(self.task_weights.data, 0.1, 3.0) 

        # 1. Collect per-task gradients and flatten them
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0 and l_i.requires_grad:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                # Unscale gradients immediately after obtaining them
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params] # Ensure all entries are tensors
            per_task_grads_list.append(unscaled_grads)

        # Flatten and stack gradients into a matrix G_matrix: (num_tasks, total_param_dim)
        flattened_grads_for_G_matrix = []
        for task_grads in per_task_grads_list:
            flat_g = torch.cat([g.view(-1) for g in task_grads])
            flattened_grads_for_G_matrix.append(flat_g)
        
        G_matrix = torch.stack(flattened_grads_for_G_matrix) 

        # 2. Apply adjusted weights to aggregate gradients
        final_combined_grad_flat = torch.sum(self.task_weights.unsqueeze(1) * G_matrix.detach(), dim=0) # detach G_matrix because we don't need its gradients

        # 3. Reshape and assign the combined gradient back to each shared parameter
        start_idx = 0
        for j, p in enumerate(shared_params):
            param_size = p.numel()
            grad_chunk = final_combined_grad_flat[start_idx : start_idx + param_size].view(p.shape)
            shared_params[j].grad = grad_chunk * scaler.get_scale() # Assign, not add
            start_idx += param_size

    def _tag_logic(self, losses, shared_params, scaler):
        """TAG: Primarily architectural; solver sums gradients from task-adaptive features."""
        # TAG's primary mechanism is the attention generator within the model architecture.
        # This solver method will ensure gradients from each task's path (after TAG) are combined.
        # Assuming the external 'criterion' handles any fixed loss weighting defined in TAG paper.
        
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0 and l_i.requires_grad:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            per_task_grads_list.append(unscaled_grads)

        # Sum up all unscaled gradients for shared parameters
        combined_grad = [torch.sum(torch.stack([g for g in grads]), dim=0) for grads in zip(*per_task_grads_list)]

        # Assign combined gradient to shared_params
        for j, p in enumerate(shared_params):
            p.grad = combined_grad[j] * scaler.get_scale() 

    def get_current_weights(self):
        """获取当前权重信息"""
        if self.method == 'gradnorm':
            return {
                'gradnorm_weight_det': self.task_weights[0].item(),
                'gradnorm_weight_da': self.task_weights[1].item(),
                'gradnorm_weight_ll': self.task_weights[2].item(),
                'gradnorm_step_count': self.step_count
            }
        elif self.method == 'pcgrad':
            return {'method': 'pcgrad', 'note': 'PCGrad operates on gradients, no explicit task weights.'}
        elif self.method == 'cagrad':
            return {'method': 'cagrad', 'note': 'CAGrad operates on gradients, no explicit task weights.'}
        elif self.method == 'mdo':
            return {
                'method': 'mdo',
                'mdo_weight_det': self.task_weights[0].item() if hasattr(self, 'task_weights') and len(self.task_weights) > 0 else float('nan'),
                'mdo_weight_da': self.task_weights[1].item() if hasattr(self, 'task_weights') and len(self.task_weights) > 1 else float('nan'),
                'mdo_weight_ll': self.task_weights[2].item() if hasattr(self, 'task_weights') and len(self.task_weights) > 2 else float('nan'),
                'mdo_step_count': self.step_count
            }
        elif self.method == 'tag':
            return {'method': 'tag', 'note': 'TAG operates on attention/features, explicit task weights depend on implementation.'}
        else:
            return {'method': self.method}