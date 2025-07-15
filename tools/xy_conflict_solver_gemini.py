import torch
import torch.nn.functional as F

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
            self.update_freq = kwargs.get('update_freq', 10) # FIX: Initialize update_freq for MDO
            # TODO: MDO特有的初始化参数
            
        elif method == 'tag':
            self.update_freq = kwargs.get('update_freq', 10) # FIX: Also initialize for TAG if it uses it
            pass
        else:
            raise ValueError(f"Unknown conflict resolution method: {method}")
    
    def compute_weighted_loss_with_gradients(self, head_losses, shared_params, scaler):
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
        losses = torch.stack(head_losses[:self.num_tasks]) # 确保只取前num_tasks个损失

        # 对于 GradNorm，直接返回加权损失。
        if self.method == 'gradnorm':
            return self._gradnorm_loss(losses)
        
        # 对于 PCGrad, CAGrad, MDO, TAG，它们直接修改梯度，然后返回一个求和损失。
        # 确保在这些方法内部，对共享参数的梯度操作是 **赋值** 而非累加。
        
        # 清零共享参数的梯度，以便我们手动设置它们。
        # 这里清零，确保后续的梯度赋值是唯一的。
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()

        # 根据方法选择处理逻辑
        if self.method == 'pcgrad':
            self._pcgrad_grads(losses, shared_params, scaler)
        elif self.method == 'cagrad':
            self._cagrad_grads(losses, shared_params, scaler)
        elif self.method == 'mdo':
            self._mdo_logic(losses, shared_params, scaler)
        elif self.method == 'tag':
            self._tag_logic(losses, shared_params, scaler)
        
        # 返回原始损失的总和，以便外部的 scaler.scale().backward() 可以被调用。
        # 此时，共享参数的梯度已经被本方法内部设置，外部的 backward 会累加头部参数的梯度。
        # 这是一个折衷方案，确保流程兼容。
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
                loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
                avg_loss_ratio = loss_ratios.mean()
                target_weights = avg_loss_ratio / (loss_ratios + 1e-8)
                
                momentum = 0.1
                self.task_weights.data = (1 - momentum) * self.task_weights.data + momentum * target_weights
                self.task_weights.data = self.num_tasks * self.task_weights.data / (self.task_weights.sum().item() + 1e-8)
                self.task_weights.data = torch.clamp(self.task_weights.data, 0.1, 3.0)
        
        return (self.task_weights * losses).sum()

    def _pcgrad_grads(self, losses, shared_params, scaler):
        """
        PCGrad: Projecting Conflicting Gradients.
        直接修改 shared_params 的 .grad 属性。
        使用 torch.autograd.grad 获取梯度以避免 graph 释放问题。
        """
        # Collect per-task gradients for shared parameters
        per_task_grads_list = [] # List of lists of gradients for each task
        for i, l_i in enumerate(losses):
            if l_i.item() != 0:
                # Use torch.autograd.grad to get gradients without modifying .grad or freeing graph
                # Ensure the loss is scaled before passing to autograd.grad if AMP is enabled
                # Note: autograd.grad with scaled loss returns scaled gradients.
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                # Unscale gradients immediately after obtaining them
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            
            per_task_grads_list.append(unscaled_grads)

        # Convert to (num_params, num_tasks, ?) for easier processing per parameter
        grads_by_param = [[] for _ in range(len(shared_params))]
        for task_grads in per_task_grads_list:
            for j, p_grad in enumerate(task_grads):
                grads_by_param[j].append(p_grad)

        # Apply PCGrad projection for each shared parameter
        for j, param_grads in enumerate(grads_by_param):
            if not param_grads: continue

            stacked_param_grads = torch.stack(param_grads) # shape: (num_tasks, param_shape)
            
            # Compute projection for each task's gradient
            for i in range(self.num_tasks):
                g_i = stacked_param_grads[i]
                for k in range(self.num_tasks):
                    if i != k:
                        g_k = stacked_param_grads[k]
                        
                        dot_product = torch.sum(g_i * g_k)
                        norm_gk_sq = torch.sum(g_k * g_k)
                        
                        if norm_gk_sq == 0:
                            continue

                        if dot_product < 0: # If gradients conflict
                            g_i_projected = g_i - (dot_product / norm_gk_sq) * g_k
                            stacked_param_grads[i] = g_i_projected # Update in-place
            
            # Combine projected gradients (e.g., by summing)
            combined_grad = torch.sum(stacked_param_grads, dim=0)

            # Assign the combined and projected gradient to the shared parameter's .grad attribute
            # Ensure it's scaled back for AMP
            shared_params[j].grad = combined_grad * scaler.get_scale() # Assign, not add


    def _cagrad_grads(self, losses, shared_params, scaler):
        """
        CAGrad: Conflict-Averse Gradient Descent.
        通过解决一个二次规划问题来找到最优的梯度加权。
        同样直接修改 shared_params 的 .grad 属性。
        """
        
        # Collect per-task gradients
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            per_task_grads_list.append(unscaled_grads)

        # Flatten and stack gradients
        flattened_grads = []
        for task_grads in per_task_grads_list:
            flat_g = torch.cat([g.view(-1) for g in task_grads])
            flattened_grads.append(flat_g)
        
        G_matrix = torch.stack(flattened_grads) # (num_tasks, total_param_dim)
        g_hat = G_matrix.mean(dim=0) # (total_param_dim,)

        # CAGrad QP formulation (simplified iterative projection)
        # This is an approximation. For exact QP, consider dedicated solvers.
        alphas = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        alphas.requires_grad_(True)
        
        optimizer_alphas = torch.optim.Adam([alphas], lr=0.01)

        for _ in range(50): 
            optimizer_alphas.zero_grad()
            
            combined_grad = torch.sum(alphas.unsqueeze(1) * G_matrix, dim=0)
            
            # Objective: minimize || Combined Grad ||^2
            loss_qp = torch.sum(combined_grad * combined_grad)
            
            # Constraint: Σ α_i * g_i . g_hat >= c * ||g_hat||^2
            g_hat_norm_sq = torch.sum(g_hat * g_hat)
            dot_prod_constraint = torch.sum(combined_grad * g_hat)
            
            penalty = 0.0
            if dot_prod_constraint < self.c * g_hat_norm_sq:
                penalty = (self.c * g_hat_norm_sq - dot_prod_constraint) ** 2 # Quadratic penalty
            
            loss_qp_total = loss_qp + 1000 * penalty # High penalty weight
            loss_qp_total.backward()
            optimizer_alphas.step()
            
            with torch.no_grad():
                alphas.data = F.relu(alphas.data)
                alphas.data = alphas.data / (alphas.data.sum() + 1e-8)

        # Assign final combined gradient
        final_combined_grad = torch.sum(alphas.unsqueeze(1) * G_matrix.detach(), dim=0)
        
        start_idx = 0
        for j, p in enumerate(shared_params):
            param_size = p.numel()
            grad_chunk = final_combined_grad[start_idx : start_idx + param_size].view(p.shape)
            shared_params[j].grad = grad_chunk * scaler.get_scale() # Assign, not add
            start_idx += param_size

    def _mdo_logic(self, losses, shared_params, scaler):
        """
        MDO (Optimal Configuration of Multi-Task Learning for Autonomous Driving) 框架。
        核心思想：Inter-Task Correlation (ITC) 驱动的权重调整。
        
        TODO: 您需要根据 MDO 论文的 Inter-Task Correlation (ITC) 和权重更新策略来填充此处。
        """
        self.step_count += 1
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            self.task_weights = torch.ones_like(losses, device=self.device, requires_grad=False)
            
        # 1. 获取每个任务的梯度
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0:
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
        
        G_matrix = torch.stack(flattened_grads) # (num_tasks, total_param_dim)
        
        # 2. 计算 Inter-Task Correlation (ITC)
        # 例如，可以使用梯度之间的余弦相似度作为相关性
        correlation_matrix = torch.eye(self.num_tasks, device=self.device)
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                norm_i = torch.norm(G_matrix[i]) + 1e-8
                norm_j = torch.norm(G_matrix[j]) + 1e-8
                
                cosine_sim = torch.dot(G_matrix[i], G_matrix[j]) / (norm_i * norm_j)
                correlation_matrix[i, j] = cosine_sim
                correlation_matrix[j, i] = cosine_sim
        
        if self.step_count % self.update_freq == 0:
            self.correlation_history.append(correlation_matrix.cpu().numpy())

        # 3. 基于 ITC 和损失动态调整任务权重
        # TODO: 填充MDO的权重更新逻辑。这可能是根据论文定义的复杂公式。
        # 示例 (这是一个简化的占位符):
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # 这是一个启发式的权重调整示例，您需要根据MDO论文替换为实际公式
                # 例如：权重可以与损失的倒数成正比，并根据任务相关性进行微调
                base_weights = 1.0 / (losses.detach() + 1e-8)
                
                # MDO论文的权重公式会更具体，请务必参考原文实现
                # 假设 MDO 会根据相关性调整权重，例如，如果两个任务正相关，它们的权重可能被鼓励同步
                # 如果负相关，可能会调整权重以减少冲突
                # self.task_weights.data = ... (根据ITC和损失计算)
                self.task_weights.data = base_weights / (base_weights.sum() + 1e-8) * self.num_tasks
                self.task_weights.data = torch.clamp(self.task_weights.data, 0.1, 3.0)


        # 4. 应用调整后的权重来聚合梯度
        final_combined_grad = torch.sum(self.task_weights.unsqueeze(1) * G_matrix.detach(), dim=0)
        
        start_idx = 0
        for j, p in enumerate(shared_params):
            param_size = p.numel()
            grad_chunk = final_combined_grad[start_idx : start_idx + param_size].view(p.shape)
            shared_params[j].grad = grad_chunk * scaler.get_scale() # Assign, not add
            start_idx += param_size

    def _tag_logic(self, losses, shared_params, scaler):
        """
        TAG (Task-adaptive attention generator) 框架。
        核心思想：在模型结构中引入 Task-adaptive Attention Generator。
        
        TODO: TAG 主要是通过修改 YOLOPx 的模型结构 (lib/models) 来实现，
        在共享骨干和任务头之间插入注意力模块。
        如果 TAG 论文中还包含额外的损失项或梯度操作，请在此处补充。

        如果 TAG 仅仅是模型结构上的改变，那么此处可能只需要简单地聚合梯度。
        """
        print("TAG method selected. Core logic for Task-adaptive Attention Generator should be implemented in model architecture (lib/models).")
        print("If TAG involves additional loss terms or gradient manipulation, implement it here.")

        # 默认：获取所有任务的梯度，然后求和并设置
        # 如果TAG有自己的特定梯度处理逻辑，请在此处替换
        per_task_grads_list = []
        for i, l_i in enumerate(losses):
            if l_i.item() != 0:
                scaled_loss = scaler.scale(l_i)
                grads = torch.autograd.grad(scaled_loss, shared_params, retain_graph=True, allow_unused=True)
                unscaled_grads = [g / scaler.get_scale() if g is not None else torch.zeros_like(p) for g, p in zip(grads, shared_params)]
            else:
                unscaled_grads = [torch.zeros_like(p) for p in shared_params]
            per_task_grads_list.append(unscaled_grads)

        # Sum up all unscaled gradients
        combined_grad = [torch.sum(torch.stack([g for g in grads if g is not None]), dim=0) for grads in zip(*per_task_grads_list)]

        # Assign combined gradient to shared_params
        for j, p in enumerate(shared_params):
            if p.grad is None:
                p.grad = combined_grad[j] * scaler.get_scale() # Assign, not add
            else: # Should not happen if zero_grad() is called correctly before
                p.grad.add_(combined_grad[j] * scaler.get_scale())
        
        pass # TODO: Implement actual TAG specific gradient manipulation or loss weighting here

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
                # TODO: Add MDO specific logs, e.g., average correlation from self.correlation_history
            }
        elif self.method == 'tag':
            return {'method': 'tag', 'note': 'TAG operates on attention/features, explicit task weights depend on implementation.'}
        else:
            return {'method': self.method}