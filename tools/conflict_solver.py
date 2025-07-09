import torch
import torch.nn as nn
import numpy as np

class GradientConflictSolver:
    """极简梯度冲突解决器 - 统一处理AMP scaler"""
    
    def __init__(self, method='gradnorm', num_tasks=3, device='cuda', **kwargs):
        self.method = method
        self.num_tasks = num_tasks
        self.device = device
        
        if method == 'gradnorm':
            self.task_weights = torch.ones(num_tasks, device=device)
            self.initial_losses = None
            self.alpha = kwargs.get('alpha', 1.5)
            self.step_count = 0
            
        elif method == 'pcgrad':
            self.reduction = kwargs.get('reduction', 'mean')
            
        elif method == 'cagrad':
            self.c = kwargs.get('c', 0.5)
    
    def get_weighted_loss(self, head_losses):
        """统一接口：返回加权损失，让外部统一处理backward"""
        losses = torch.stack(head_losses[:3])
        
        if self.method == 'gradnorm':
            return self._gradnorm_weight(losses)
        elif self.method == 'pcgrad':
            return self._pcgrad_weight(losses)
        elif self.method == 'cagrad':
            return self._cagrad_weight(losses)
        else:
            return losses.sum()
    
    def _gradnorm_weight(self, losses):
        """GradNorm权重计算 - 周期性更新"""
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()
        
        self.step_count += 1
        # 每50步更新一次权重，避免复杂计算
        if self.step_count % 50 == 0:
            with torch.no_grad():
                # 基于损失比例简单调整权重
                loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
                avg_ratio = loss_ratios.mean()
                # 权重与损失比例成反比
                self.task_weights = avg_ratio / (loss_ratios + 1e-8)
                # 归一化
                self.task_weights = self.num_tasks * self.task_weights / self.task_weights.sum()
        
        return (self.task_weights.detach() * losses).sum()
    
    def _pcgrad_weight(self, losses):
        """PCGrad权重 - 简化为自适应权重"""
        with torch.no_grad():
            # 简化：根据损失大小自适应调整权重
            inv_losses = 1.0 / (losses.detach() + 1e-8)
            weights = inv_losses / inv_losses.sum()
        return (weights * losses).sum()
    
    def _cagrad_weight(self, losses):
        """CAGrad权重 - 简化为平滑权重"""
        with torch.no_grad():
            # 简化：使用指数移动平均权重
            if not hasattr(self, 'ema_weights'):
                self.ema_weights = torch.ones_like(losses)
            
            current_weights = 1.0 / (losses.detach() + 1e-8)
            current_weights = current_weights / current_weights.sum()
            
            # EMA更新
            self.ema_weights = 0.9 * self.ema_weights + 0.1 * current_weights
            
        return (self.ema_weights * losses).sum()
    
    def get_method_info(self):
        """获取方法信息"""
        if self.method == 'gradnorm':
            return {'method': 'GradNorm', 'weights': self.task_weights}
        else:
            return {'method': self.method}