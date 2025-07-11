import torch


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
            self.update_freq = kwargs.get('update_freq', 10)  # 更频繁的更新
            
        elif method == 'pcgrad':
            self.reduction = kwargs.get('reduction', 'sum')
            
        elif method == 'cagrad':
            self.c = kwargs.get('c', 0.5)
            self.ema_weights = None
    
    def compute_weighted_loss_with_gradients(self, model, head_losses, optimizer, scaler):
        """计算加权损失，同时更新权重（如果需要）"""
        losses = torch.stack(head_losses[:3])
        
        if self.method == 'gradnorm':
            return self._gradnorm_loss(model, losses, optimizer)
        elif self.method == 'pcgrad':
            return self._pcgrad_loss(losses)
        elif self.method == 'cagrad':
            return self._cagrad_loss(losses)
        else:
            return losses.sum()
    
    def _gradnorm_loss(self, model, losses, optimizer):
        """改进的GradNorm实现"""
        self.step_count += 1
        
        # 初始化
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
            return losses.sum()
        
        # 更频繁地更新权重
        if self.step_count % self.update_freq == 0:
            with torch.no_grad():
                # 计算相对损失率
                loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
                
                # 计算平均损失率
                avg_loss_ratio = loss_ratios.mean()
                
                # 计算目标权重（与损失率成反比）
                target_weights = avg_loss_ratio / (loss_ratios + 1e-8)
                
                # 平滑更新权重（避免剧烈变化）
                momentum = 0.1
                self.task_weights = (1 - momentum) * self.task_weights + momentum * target_weights
                
                # 归一化权重
                self.task_weights = self.num_tasks * self.task_weights / (self.task_weights.sum() + 1e-8)
                
                # 限制权重范围，避免某个任务权重过大或过小
                self.task_weights = torch.clamp(self.task_weights, 0.1, 3.0)
        
        return (self.task_weights.detach() * losses).sum()
    
    def _pcgrad_loss(self, losses):
        """简化的PCGrad实现"""
        # 基于损失大小的自适应权重
        with torch.no_grad():
            # 使用损失的倒数作为权重（小损失高权重）
            weights = 1.0 / (losses.detach() + 1e-8)
            weights = weights / weights.sum()
        
        return (weights * losses).sum()
    
    def _cagrad_loss(self, losses):
        """改进的CAGrad实现"""
        with torch.no_grad():
            if self.ema_weights is None:
                self.ema_weights = torch.ones_like(losses) / len(losses)
            
            # 计算当前权重
            current_weights = 1.0 / (losses.detach() + 1e-8)
            current_weights = current_weights / current_weights.sum()
            
            # 指数移动平均更新
            self.ema_weights = 0.9 * self.ema_weights + 0.1 * current_weights
        
        return (self.ema_weights * losses).sum()
    
    def get_current_weights(self):
        """获取当前权重信息"""
        if self.method == 'gradnorm':
            return {
                'gradnorm_weight_det': self.task_weights[0].item(),
                'gradnorm_weight_da': self.task_weights[1].item(),
                'gradnorm_weight_ll': self.task_weights[2].item(),
                'gradnorm_step_count': self.step_count
            }
        elif self.method == 'cagrad' and self.ema_weights is not None:
            return {
                'cagrad_weight_det': self.ema_weights[0].item(),
                'cagrad_weight_da': self.ema_weights[1].item(),
                'cagrad_weight_ll': self.ema_weights[2].item(),
            }
        else:
            return {'method': self.method}
