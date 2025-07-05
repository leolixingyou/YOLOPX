import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List
import wandb


class MTLConflictResolver(ABC):
    """多任务学习冲突处理基类"""
    
    def __init__(self, task_names: List[str], device: str = 'cuda'):
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.device = device
        self.initialized = False
        
    @abstractmethod
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass


class Original(MTLConflictResolver):
    """原始方法 - 均等权重"""
    
    def __init__(self, task_names: List[str], device: str = 'cuda'):
        super().__init__(task_names, device)
        self.loss_history = []
        self.initialized = True
        
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        total_loss = sum(loss.item() for loss in losses.values())
        self.loss_history.append(total_loss)
        return {task: 1.0 for task in self.task_names}
    
    def get_metrics(self) -> Dict[str, float]:
        if len(self.loss_history) > 5:
            recent = self.loss_history[-5:]
            trend = (recent[-1] - recent[0]) / len(recent)
            return {"loss_trend": trend, "loss_stability": np.std(recent)}
        return {}


class GradNorm(MTLConflictResolver):
    """GradNorm动态权重平衡"""
    
    def __init__(self, task_names: List[str], alpha: float = 1.5, device: str = 'cuda'):
        super().__init__(task_names, device)
        self.alpha = alpha
        self.task_weights = torch.ones(self.num_tasks, device=device, requires_grad=True)
        self.initial_losses = None
        self.shared_layer = None
        self.optimizer = None
        self.training_rate_variance = []
        self.weight_adjustments = []
        
    def initialize(self, initial_losses: Dict[str, torch.Tensor], shared_layer: nn.Module):
        self.initial_losses = torch.stack([initial_losses[task].detach() for task in self.task_names])
        self.shared_layer = shared_layer
        self.optimizer = torch.optim.Adam([self.task_weights], lr=0.025)
        self.initialized = True
        
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        if not self.initialized:
            return {task: 1.0 for task in self.task_names}
            
        current_losses = torch.stack([losses[task].detach() for task in self.task_names])
        
        # 计算训练速率差异
        loss_ratios = current_losses / (self.initial_losses + 1e-8)
        avg_loss_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (avg_loss_ratio + 1e-8)
        variance = torch.var(relative_rates).item()
        self.training_rate_variance.append(variance)
        
        # 计算梯度范数
        grad_norms = []
        for i, task in enumerate(self.task_names):
            try:
                grad = torch.autograd.grad(losses[task], self.shared_layer.parameters(), 
                                        retain_graph=True, create_graph=True, allow_unused=True)
                valid_grads = [g for g in grad if g is not None]
                if valid_grads:
                    grad_norm = torch.norm(torch.cat([g.flatten() for g in valid_grads]))
                else:
                    grad_norm = torch.tensor(1e-8, device=self.device)  # 避免零梯度
                grad_norms.append(grad_norm)
            except Exception as e:
                print(f"Warning: GradNorm gradient computation failed for {task}: {e}")
                grad_norms.append(torch.tensor(1e-8, device=self.device))
        
        if len(grad_norms) == 0:
            return {task: 1.0 for task in self.task_names}
        
        grad_norms = torch.stack(grad_norms)
        
        # 更新权重
        old_weights = self.task_weights.clone()
        avg_grad_norm = grad_norms.mean()
        target_norms = avg_grad_norm * (relative_rates ** self.alpha)
        gradnorm_loss = torch.abs(grad_norms - target_norms).sum()
        
        try:
            self.optimizer.zero_grad()
            gradnorm_loss.backward(retain_graph=True)
            self.optimizer.step()
            
            with torch.no_grad():
                self.task_weights.data = torch.clamp(self.task_weights.data, min=0.1)
                self.task_weights.data = self.num_tasks * self.task_weights.data / self.task_weights.data.sum()
        except Exception as e:
            print(f"Warning: GradNorm weight update failed: {e}")
        
        adjustment = torch.norm(self.task_weights - old_weights).item()
        self.weight_adjustments.append(adjustment)
        
        # 修复：直接返回原始权重而不是softmax
        return {task: self.task_weights[i].item() for i, task in enumerate(self.task_names)}
    
    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.training_rate_variance:
            metrics["detect_training_rate_variance"] = self.training_rate_variance[-1]
        if self.weight_adjustments:
            metrics["operation_weight_adjustment"] = self.weight_adjustments[-1]
        return metrics


class PCGrad(MTLConflictResolver):
    """PCGrad梯度投影"""
    
    def __init__(self, task_names: List[str], device: str = 'cuda'):
        super().__init__(task_names, device)
        self.conflict_count = 0
        self.total_pairs = 0
        self.cosine_similarities = []
        self.conflict_detections = []
        self.projection_magnitudes = []
        
    def initialize(self, shared_layer: nn.Module):
        self.shared_layer = shared_layer
        self.initialized = True
        
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        return {task: 1.0 for task in self.task_names}
    
    def apply_pcgrad_to_gradients(self, losses: Dict[str, torch.Tensor], model):
        """应用PCGrad梯度投影 - 修复版本"""
        if not self.initialized:
            return
        
        # 修复：统一收集所有参数的梯度
        all_params = list(model.parameters())
        task_gradients = {}
        
        for task, loss in losses.items():
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            # 收集所有参数的梯度，确保一致性
            grads = []
            for param in all_params:
                if param.grad is not None:
                    grads.append(param.grad.data.clone().flatten())
                else:
                    # 重要修复：为没有梯度的参数填充零
                    grads.append(torch.zeros(param.numel(), device=self.device))
            
            task_gradients[task] = torch.cat(grads)
        
        # 验证梯度尺寸一致性
        grad_sizes = [grad.numel() for grad in task_gradients.values()]
        if len(set(grad_sizes)) > 1:
            print(f"Warning: Gradient size mismatch: {grad_sizes}")
            return  # 跳过这次投影
        
        # PCGrad投影
        projected_grad = self._apply_pcgrad_projection(task_gradients)
        
        # 设置投影后的梯度
        model.zero_grad()
        start_idx = 0
        for param in all_params:
            param_numel = param.numel()
            if param_numel > 0:
                param_grad = projected_grad[start_idx:start_idx + param_numel]
                param.grad = param_grad.reshape(param.shape)
                start_idx += param_numel
    
    def _apply_pcgrad_projection(self, task_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """PCGrad核心算法"""
        grad_list = [task_gradients[task] for task in self.task_names]
        projected_grads = [grad.clone() for grad in grad_list]
        
        conflicts = 0
        projection_magnitude = 0.0
        similarities = []
        
        for i, grad_i in enumerate(grad_list):
            for j, grad_j in enumerate(grad_list):
                if i != j:
                    self.total_pairs += 1
                    dot_product = torch.dot(grad_i, grad_j)
                    
                    # 计算余弦相似度
                    norm_i, norm_j = torch.norm(grad_i), torch.norm(grad_j)
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        similarities.append(similarity.item())
                    
                    # 检测冲突并投影
                    if dot_product < 0:
                        conflicts += 1
                        grad_j_norm_sq = torch.norm(grad_j) ** 2
                        if grad_j_norm_sq > 1e-8:
                            proj_coeff = dot_product / grad_j_norm_sq
                            projection = proj_coeff * grad_j
                            projected_grads[i] = projected_grads[i] - projection
                            projection_magnitude += torch.norm(projection).item()
        
        self.conflict_count += conflicts
        self.conflict_detections.append(conflicts)
        self.projection_magnitudes.append(projection_magnitude)
        self.cosine_similarities.append(np.mean(similarities) if similarities else 1.0)
        
        return torch.stack(projected_grads).mean(dim=0)
    
    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.cosine_similarities:
            metrics["detect_avg_cosine_similarity"] = self.cosine_similarities[-1]
        if self.conflict_detections:
            metrics["detect_current_conflicts"] = self.conflict_detections[-1]
        if self.projection_magnitudes:
            metrics["operation_projection_magnitude"] = self.projection_magnitudes[-1]
        metrics["detect_conflict_ratio"] = self.conflict_count / max(self.total_pairs, 1)
        return metrics


class UncertaintyWeighting(MTLConflictResolver):
    """基于不确定性的权重调整"""
    
    def __init__(self, task_names: List[str], device: str = 'cuda'):
        super().__init__(task_names, device)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks, device=device))
        self.optimizer = None
        
    def initialize(self, learning_rate: float = 0.001):
        self.optimizer = torch.optim.Adam([self.log_vars], lr=learning_rate)
        self.initialized = True
        
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        if not self.initialized:
            self.initialize()
        
        weights = {}
        for i, task in enumerate(self.task_names):
            precision = torch.exp(-self.log_vars[i])
            weights[task] = precision.item()
        return weights
    
    def get_loss_with_uncertainty(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.initialized:
            return sum(losses.values())
        
        total_loss = torch.tensor(0.0, device=self.device)
        for i, task in enumerate(self.task_names):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * losses[task] + self.log_vars[i]
        return total_loss
    
    def update_uncertainty(self, losses: Dict[str, torch.Tensor]):
        if not self.initialized:
            return
        uncertainty_loss = self.get_loss_with_uncertainty(losses)
        self.optimizer.zero_grad()
        uncertainty_loss.backward(retain_graph=True)
        self.optimizer.step()
    
    def get_metrics(self) -> Dict[str, float]:
        if not self.initialized:
            return {}
        metrics = {}
        for i, task in enumerate(self.task_names):
            uncertainty = torch.exp(self.log_vars[i]).item()
            metrics[f"uncertainty_{task}"] = uncertainty
        return metrics


class MGDA(MTLConflictResolver):
    """多目标梯度下降算法"""
    
    def __init__(self, task_names: List[str], device: str = 'cuda'):
        super().__init__(task_names, device)
        self.weights = torch.ones(self.num_tasks, device=device) / self.num_tasks
        self.convergence_history = []
        
    def update_weights(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> Dict[str, float]:
        return {task: self.weights[i].item() for i, task in enumerate(self.task_names)}
    
    def compute_mgda_weights(self, losses: Dict[str, torch.Tensor], model_params) -> Dict[str, float]:
        """计算MGDA权重 - 修复版本"""
        # 计算梯度并验证有效性
        gradients = []
        all_params = list(model_params)
        
        for task in self.task_names:
            try:
                grad = torch.autograd.grad(losses[task], all_params, retain_graph=True, allow_unused=True)
                # 过滤掉None梯度并转换为扁平张量
                valid_grads = [g.flatten() for g in grad if g is not None]
                if len(valid_grads) == 0:
                    print(f"Warning: No valid gradients for task {task}, using zero gradient")
                    # 使用零梯度作为fallback
                    total_params = sum(p.numel() for p in all_params)
                    grad_flat = torch.zeros(total_params, device=self.device)
                else:
                    grad_flat = torch.cat(valid_grads)
                gradients.append(grad_flat)
            except Exception as e:
                print(f"Error computing gradient for task {task}: {e}")
                # 使用零梯度作为fallback
                total_params = sum(p.numel() for p in all_params)
                grad_flat = torch.zeros(total_params, device=self.device)
                gradients.append(grad_flat)
        
        # 验证梯度有效性
        if len(gradients) == 0:
            return {task: 1.0 / self.num_tasks for task in self.task_names}
        
        # 验证梯度尺寸一致性
        grad_sizes = [grad.numel() for grad in gradients]
        if len(set(grad_sizes)) > 1:
            print(f"Warning: MGDA gradient size mismatch: {grad_sizes}")
            return {task: 1.0 / self.num_tasks for task in self.task_names}
        
        # Frank-Wolfe求解
        weights = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        
        try:
            grad_matrix = torch.stack(gradients)
            
            for iteration in range(20):
                old_weights = weights.clone()
                weighted_grad = torch.sum(weights.unsqueeze(1) * grad_matrix, dim=0)
                dot_products = torch.mv(grad_matrix, weighted_grad)
                min_idx = torch.argmin(dot_products)
                
                step_size = 2.0 / (iteration + 2.0)
                new_weights = torch.zeros_like(weights)
                new_weights[min_idx] = 1.0
                weights = (1 - step_size) * weights + step_size * new_weights
                
                weight_change = torch.norm(weights - old_weights).item()
                self.convergence_history.append(weight_change)
                if weight_change < 1e-6:
                    break
        except Exception as e:
            print(f"Warning: MGDA Frank-Wolfe failed: {e}, using uniform weights")
            weights = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        
        self.weights = weights
        return {task: weights[i].item() for i, task in enumerate(self.task_names)}
    
    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        for i, task in enumerate(self.task_names):
            metrics[f"mgda_weight_{task}"] = self.weights[i].item()
        if self.convergence_history:
            metrics["mgda_convergence_rate"] = self.convergence_history[-1]
        return metrics


def create_mtl_resolver(method: str, task_names: List[str], device: str = 'cuda') -> MTLConflictResolver:
    """工厂函数"""
    resolvers = {
        'original': Original,
        'gradnorm': GradNorm,
        'pcgrad': PCGrad,
        'uncertainty': UncertaintyWeighting,
        'mgda': MGDA,
    }
    
    if method.lower() not in resolvers:
        raise ValueError(f"Unknown method: {method}. Available: {list(resolvers.keys())}")
    
    return resolvers[method.lower()](task_names, device=device)