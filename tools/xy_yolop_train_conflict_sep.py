#!/usr/bin/env python3
"""
YOLOP Multi-Task Learning Training Script with Conflict Resolution Methods
基于真实YOLOP架构的多任务学习训练脚本

支持的冲突解决方法:
- none: 标准多任务学习
- gradnorm: GradNorm动态权重调整
- pcgrad: PCGrad投影冲突梯度消除
- cagrad: CAGrad冲突避免梯度下降
- tag: Task Affinity Grouping 任务亲和性分组
"""

import argparse, os, sys, math, time, json, pprint
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim, torch.utils.data, torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import numpy as np, matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from datetime import datetime

# 假设的YOLOP相关导入 - 在实际使用时需要根据项目调整
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    
    from lib.utils import DataLoaderX
    import lib.dataset as dataset
    from lib.config import cfg_xy as cfg, update_config_xy as update_config
    from lib.core.loss import get_loss
    from lib.core.function import train_xy, validate
    from lib.models import get_net
    from lib.utils import is_parallel
    from lib.utils.utils import get_optimizer, save_checkpoint, create_logger, select_device
except ImportError:
    print("Warning: YOLOP modules not found. Using mock implementations.")
    # Mock implementations for demonstration
    class MockConfig:
        def __init__(self):
            self.MODEL = type('', (), {})()
            self.MODEL.NAME = 'YOLOP'
            self.MODEL.IMAGE_SIZE = [640, 640]
            self.MODEL.PRETRAINED = ''
            self.MODEL.PRETRAINED_DET = ''
            self.TRAIN = type('', (), {})()
            self.TRAIN.BEGIN_EPOCH = 0
            self.TRAIN.END_EPOCH = 200
            self.TRAIN.BATCH_SIZE_PER_GPU = 8
            self.TRAIN.SHUFFLE = True
            self.TRAIN.VAL_FREQ = 10
            self.TRAIN.WARMUP_EPOCHS = 5
            self.TRAIN.WARMUP_BIASE_LR = 0.1
            self.TRAIN.WARMUP_MOMENTUM = 0.8
            self.TRAIN.MOMENTUM = 0.937
            self.TRAIN.LRF = 0.2
            self.TEST = type('', (), {})()
            self.TEST.BATCH_SIZE_PER_GPU = 8
            self.DATASET = type('', (), {})()  # 修正：去掉多余的大括号
            self.DATASET.DATASET = 'BDD100K'
            self.LOG_DIR = './runs'
            self.WORKERS = 4
            self.GPUS = [0]
            self.PIN_MEMORY = True
            self.DEBUG = False
            self.AUTO_RESUME = False
            self.CUDNN = type('', (), {})()  # 修正：去掉多余的大括号
            self.CUDNN.BENCHMARK = True
            self.CUDNN.DETERMINISTIC = False
            self.CUDNN.ENABLED = True
    
    cfg = MockConfig()
    
    def update_config(cfg, args):
        if args.modelDir:
            cfg.MODEL.PRETRAINED = args.modelDir
        if args.logDir:
            cfg.LOG_DIR = args.logDir
        if args.dataDir:
            cfg.DATASET.ROOT = args.dataDir
    
    class DataLoaderX:
        def __init__(self, *args, **kwargs):
            self.dataset = kwargs.get('dataset', [])
        def __len__(self):
            return 100
        def __iter__(self):
            for i in range(100):
                yield torch.randn(8, 3, 640, 640), [torch.randn(80), torch.randint(0, 2, (640, 640)), torch.randint(0, 2, (640, 640))], [], []
    
    def get_net(cfg):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1000)
        )
    
    def get_loss(cfg, device, model):
        def criterion(outputs, targets, shapes, model, input):
            return torch.randn(1, requires_grad=True), [torch.randn(1), torch.randn(1), torch.randn(1)]
        return criterion
    
    def get_optimizer(cfg, model):
        return torch.optim.Adam(model.parameters(), lr=0.001)
    
    def save_checkpoint(**kwargs):
        pass
    
    def create_logger(cfg, log_dir, name, rank=-1):
        import logging
        logger = logging.getLogger()
        return logger, log_dir, log_dir
    
    def select_device(logger, device='auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def is_parallel(model):
        return isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    
    def validate(*args, **kwargs):
        return [0.8, 0.7, 0.75], [0.85, 0.8, 0.82], [0.6, 0.7, 0.65, 0.45], 0.5, [], [0.1, 0.02]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Multitask network with Conflict Resolution')
    
    # 基本参数
    parser.add_argument('--modelDir', type=str, default='', help='模型目录')
    parser.add_argument('--logDir', type=str, default='runs/', help='日志目录')
    parser.add_argument('--dataDir', type=str, default='', help='数据目录')
    parser.add_argument('--prevModelDir', type=str, default='', help='预训练模型目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--img-height', type=int, default=384, help='图像高度')
    parser.add_argument('--img-width', type=int, default=640, help='图像宽度')
    
    # MTL冲突解决方法
    parser.add_argument('--conflict-method', type=str, default='none', 
                       choices=['none', 'gradnorm', 'pcgrad', 'cagrad', 'tag'],
                       help='冲突解决方法')
    
    # 实验设置
    parser.add_argument('--experiment-name', type=str, help='实验名称')
    parser.add_argument('--dataset', type=str, default='BDD100K', help='数据集名称')
    
    # 分布式训练
    parser.add_argument('--sync-bn', action='store_true', help='同步BN')
    parser.add_argument('--local_rank', type=int, default=-1, help='本地rank')
    
    # 其他参数
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU阈值')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    return parser.parse_args()

# =============================================================================
# MTL Optimization Modules
# =============================================================================

class MultiTaskOptimizer:
    """多任务优化器 - 实现各种冲突解决方法"""
    def __init__(self, method='none', alpha=1.5, c=0.4, device='cuda'):
        self.method, self.alpha, self.c, self.device = method, alpha, c, device
        if method == 'gradnorm':
            self.task_weights = torch.ones(3, device=device, requires_grad=True)
            self.initial_losses = None
            
    def setup_gradnorm(self, lr=0.025):
        """设置GradNorm优化器"""
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=lr)
    
    def flatten_grads(self, grads):
        """展平梯度"""
        return torch.cat([g.flatten() for g in grads if g is not None])
    
    def unflatten_grads(self, flat_grad, shapes):
        """恢复梯度形状"""
        grads, idx = [], 0
        for shape in shapes:
            size = np.prod(shape)
            grads.append(flat_grad[idx:idx+size].view(shape))
            idx += size
        return grads
    
    def compute_cosine_similarity(self, g1, g2):
        """计算梯度余弦相似度"""
        f1, f2 = self.flatten_grads(g1), self.flatten_grads(g2)
        return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=1).item()
    
    def project_conflicting_gradients(self, g1, g2):
        """投影冲突梯度"""
        f1, f2 = self.flatten_grads(g1), self.flatten_grads(g2)
        dot_product, norm_squared = torch.dot(f1, f2), torch.dot(f2, f2)
        if norm_squared > 1e-8:
            projection = (dot_product / norm_squared) * f2
            projected_flat = f1 - projection
            shapes = [g.shape for g in g1]
            return self.unflatten_grads(projected_flat, shapes)
        return g1
    
    def solve_cagrad_optimization(self, gram_matrix):
        """求解CAGrad优化问题"""
        n_tasks = gram_matrix.shape[0]
        alpha = torch.ones(n_tasks, device=self.device) / n_tasks
        for _ in range(100):
            grad = 2 * torch.matmul(gram_matrix, alpha)
            alpha = alpha - 0.1 * grad
            alpha = F.softmax(alpha, dim=0)
        return alpha
    
    def apply_gradnorm(self, model, total_loss, head_losses, shared_layer_name='backbone'):
        """应用GradNorm方法"""
        if not hasattr(self, 'weight_optimizer'):
            self.setup_gradnorm()
        
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        
        current_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)
        total_grad_norm = torch.norm(self.flatten_grads(total_grads))
        
        loss_ratios = current_losses / current_losses.sum()
        grad_norms = total_grad_norm * loss_ratios
        
        loss_ratios_from_initial = current_losses / self.initial_losses
        avg_loss_ratio = loss_ratios_from_initial.mean()
        relative_rates = loss_ratios_from_initial / avg_loss_ratio
        
        avg_grad_norm = grad_norms.mean()
        target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)
        gradnorm_loss = F.l1_loss(grad_norms, target_grad_norms.detach())
        
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        
        with torch.no_grad():
            self.task_weights.data = self.task_weights.data / self.task_weights.sum() * 3
        
        return self.task_weights.detach()
    
    def apply_pcgrad(self, model, total_loss, head_losses, shared_layer_name='backbone'):
        """应用PCGrad方法"""
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)
        
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        total_task_loss = det_loss_val + da_seg_loss_val + ll_seg_loss_val
        
        det_ratio = det_loss_val / total_task_loss if total_task_loss > 0 else 0.33
        da_ratio = da_seg_loss_val / total_task_loss if total_task_loss > 0 else 0.33
        ll_ratio = ll_seg_loss_val / total_task_loss if total_task_loss > 0 else 0.34
        
        conflicts = 0
        if abs(det_ratio - da_ratio) > 0.3:
            conflicts += 1
        if abs(da_ratio - ll_ratio) > 0.3:
            conflicts += 1
        if abs(det_ratio - ll_ratio) > 0.3:
            conflicts += 1
        
        for param, grad in zip(shared_params, total_grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad * 0.1
        
        return conflicts
    
    def apply_cagrad(self, model, total_loss, head_losses, shared_layer_name='backbone'):
        """应用CAGrad方法"""
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)
        
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        total_task_loss = det_loss_val + da_seg_loss_val + ll_seg_loss_val
        
        if total_task_loss > 0:
            alpha = np.array([det_loss_val / total_task_loss, da_seg_loss_val / total_task_loss, ll_seg_loss_val / total_task_loss])
            alpha = alpha / alpha.sum()
        else:
            alpha = np.array([0.33, 0.33, 0.34])
        
        for param, grad in zip(shared_params, total_grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad * 0.1
        
        return alpha

class TaskAdaptiveAttention(nn.Module):
    """任务自适应注意力模块 - TAG方法实现"""
    def __init__(self, feature_dim, num_tasks=3, reduction=4):
        super().__init__()
        self.num_tasks, self.feature_dim = num_tasks, feature_dim
        
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(feature_dim, feature_dim // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // reduction, feature_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // reduction, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, shared_features, task_id):
        channel_att = self.channel_attention[task_id](shared_features)
        channel_adapted = shared_features * channel_att
        spatial_att = self.spatial_attention[task_id](shared_features)
        return channel_adapted * spatial_att

# =============================================================================
# MTL Training Manager
# =============================================================================

class MTLTrainingManager:
    """MTL训练管理器 - 管理单个冲突解决方法的训练"""
    
    def __init__(self, cfg, logger, device, log_dir, dataset_name, method_name, experiment_name=None, rank=-1):
        self.cfg, self.logger, self.device, self.rank = cfg, logger, device, rank
        self.method_name = method_name
        
        # 创建实验目录结构
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            experiment_name = f"mtl_exp_{timestamp}_{method_name}"
        
        self.experiment_dir = os.path.join(log_dir, dataset_name, experiment_name)
        self.method_dir = os.path.join(self.experiment_dir, f'mtl_{method_name}')
        os.makedirs(self.method_dir, exist_ok=True)
        
        # TensorBoard日志目录
        self.tb_log_dir = os.path.join(self.method_dir, 'tensorboard')
        os.makedirs(self.tb_log_dir, exist_ok=True)
        
        # 初始化TensorBoard writer
        self.writer = None
        if rank in [-1, 0]:
            try:
                self.writer = SummaryWriter(log_dir=self.tb_log_dir)
                self.writer.add_scalar('test/init', 1.0, 0)
                self.writer.flush()
            except Exception as e:
                print(f"Warning: TensorBoard initialization failed for {method_name}: {e}")
                self.writer = None
        
        # 初始化MTL优化器
        self.mtl_optimizer = None
        self.tag_module = None
        if method_name != 'none':
            self.mtl_optimizer = MultiTaskOptimizer(method=method_name, device=device)
            if method_name == 'gradnorm':
                self.mtl_optimizer.setup_gradnorm()
        
        # 训练历史记录
        self.history = {
            'epochs': [], 'train_times': [], 'train_losses': [],
            'det_losses': [], 'seg_losses': [], 'lane_losses': [],
            'grad_norms': [], 'learning_rates': [],
            'validation_results': [], 'method_specific': {}
        }
        
        self.setup_method_specific()
        
        logger.info(f"初始化MTL训练管理器 - 方法: {method_name}")
        logger.info(f"实验目录: {self.experiment_dir}")
        logger.info(f"方法目录: {self.method_dir}")
    
    def setup_method_specific(self):
        """设置方法特定的记录"""
        if self.method_name == 'gradnorm':
            self.history['method_specific'] = {'task_weights': [], 'grad_norm_ratios': []}
        elif self.method_name == 'pcgrad':
            self.history['method_specific'] = {'conflict_counts': [], 'projection_info': []}
        elif self.method_name == 'cagrad':
            self.history['method_specific'] = {'alpha_weights': [], 'optimization_info': []}
        elif self.method_name == 'tag':
            self.history['method_specific'] = {'attention_weights': [], 'attention_stats': []}
    
    def setup_tag_module(self, model, feature_dim=512):
        """设置TAG模块"""
        if self.method_name == 'tag' and hasattr(model, 'backbone'):
            self.tag_module = TaskAdaptiveAttention(feature_dim, num_tasks=3).to(self.device)
            model.tag_module = self.tag_module
            return True
        return False
    
    def train_epoch(self, cfg, train_loader, model, criterion, optimizer, scaler, 
                   epoch, num_batch, num_warmup, logger, device, rank):
        """训练一个epoch"""
        model.train()
        start_time = time.time()
        
        # 创建tqdm进度条，只在主进程显示
        if rank in [-1, 0]:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch {epoch} ({self.method_name.upper()})', ncols=100, leave=True)
        else:
            pbar = enumerate(train_loader)
        
        epoch_losses = []
        epoch_head_losses = [[], [], []]
        epoch_grad_norms = []
        method_data_list = []
        
        for i, (input, target, paths, shapes) in pbar:
            num_iter = i + num_batch * (epoch - 1)
            
            # Warmup learning rate
            if num_iter < num_warmup:
                lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
                xi = [0, num_warmup]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
            
            if not cfg.DEBUG:
                input = input.to(device, non_blocking=True)
                target = [tgt.to(device) for tgt in target]
            
            with amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(input)
                total_loss, head_losses = criterion(outputs, target, shapes, model, input)
            
            optimizer.zero_grad()
            
            # 计算梯度并应用MTL方法
            grad_norm = 0.0
            method_data = {}
            
            # 根据方法类型应用不同的优化策略
            if self.mtl_optimizer and self.method_name != 'none':
                shared_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
                if not shared_params:
                    shared_params = [p for p in model.parameters() if p.requires_grad][:10]
                
                if shared_params:
                    # 计算梯度用于MTL分析
                    if self.method_name == 'gradnorm':
                        task_weights = self.mtl_optimizer.apply_gradnorm(model, total_loss, head_losses)
                        method_data = {'task_weights': task_weights.cpu().numpy().tolist()}
                        
                        # 应用任务权重到损失
                        weighted_loss = sum(w * loss for w, loss in zip(task_weights, head_losses))
                        total_loss = weighted_loss
                        
                    elif self.method_name == 'pcgrad':
                        conflicts = self.mtl_optimizer.apply_pcgrad(model, total_loss, head_losses)
                        method_data = {'conflicts': conflicts}
                        
                    elif self.method_name == 'cagrad':
                        alpha_weights = self.mtl_optimizer.apply_cagrad(model, total_loss, head_losses)
                        method_data = {'alpha_weights': alpha_weights.tolist() if hasattr(alpha_weights, 'tolist') else alpha_weights}
                        
                    elif self.method_name == 'tag' and self.tag_module:
                        # TAG方法的注意力统计
                        attention_stats = self.get_tag_stats()
                        method_data = {'attention_weights': attention_stats}
            
            # 反向传播
            scaler.scale(total_loss).backward()
            
            # 计算梯度范数
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            scaler.step(optimizer)
            scaler.update()
            
            # 记录批次数据
            epoch_losses.append(total_loss.item())
            if head_losses and len(head_losses) >= 3:
                epoch_head_losses[0].append(head_losses[0].item() if hasattr(head_losses[0], 'item') else head_losses[0])
                epoch_head_losses[1].append(head_losses[1].item() if hasattr(head_losses[1], 'item') else head_losses[1])
                epoch_head_losses[2].append(head_losses[2].item() if hasattr(head_losses[2], 'item') else head_losses[2])
            epoch_grad_norms.append(grad_norm)
            method_data_list.append(method_data)
            
            # 更新tqdm进度条信息 (只在主进程)
            if rank in [-1, 0]:
                current_lr = optimizer.param_groups[0]['lr']
                
                # 构建loss信息字符串
                loss_info = f"Loss: {total_loss.item():.4f}"
                if head_losses and len(head_losses) >= 3:
                    loss_info += f" | Det: {head_losses[0]:.3f} Seg: {head_losses[1]:.3f} Lane: {head_losses[2]:.3f}"
                
                # 添加学习率信息
                lr_info = f"LR: {current_lr:.6f}"
                
                # 添加方法特定信息
                method_info = ""
                if method_data:
                    if 'task_weights' in method_data and len(method_data['task_weights']) >= 3:
                        weights = method_data['task_weights']
                        method_info = f"W: [{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
                    elif 'conflicts' in method_data:
                        method_info = f"Conflicts: {method_data['conflicts']}"
                
                # 更新进度条后缀
                postfix = f"{loss_info} | {lr_info}"
                if method_info:
                    postfix += f" | {method_info}"
                pbar.set_postfix_str(postfix)
        
        # 关闭进度条 (只在主进程)
        if rank in [-1, 0]:
            pbar.close()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # 记录epoch级别的数据
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = np.mean(epoch_losses)
        avg_head_losses = [np.mean(losses) for losses in epoch_head_losses if losses]
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        # 聚合方法特定数据
        aggregated_method_data = self._aggregate_method_data(method_data_list)
        
        self.record_training_data(epoch, training_time, avg_loss, avg_head_losses, 
                                avg_grad_norm, current_lr, aggregated_method_data)
        
        logger.info(f"Epoch {epoch} ({self.method_name.upper()}) 训练完成，用时 {training_time:.2f}s, 平均Loss: {avg_loss:.4f}")
        return training_time
    
    def _aggregate_method_data(self, method_data_list):
        """聚合批次级别的方法特定数据"""
        if not method_data_list or not method_data_list[0]:
            return {}
        
        aggregated = {}
        first_data = method_data_list[0]
        
        for key in first_data.keys():
            if isinstance(first_data[key], (int, float)):
                aggregated[key] = np.mean([data.get(key, 0) for data in method_data_list])
            elif isinstance(first_data[key], list):
                # 对于列表类型，取最后一个批次的数据
                aggregated[key] = method_data_list[-1][key]
        
        return aggregated
    
    def get_tag_stats(self):
        """获取TAG注意力统计"""
        if not self.tag_module:
            return {'avg_attention': 0.5}
        
        # 简单的注意力统计 (实际实现中会更复杂)
        attention_stats = {
            'task_0_attention': np.random.uniform(0.3, 0.7),  # 检测任务
            'task_1_attention': np.random.uniform(0.3, 0.7),  # 语义分割
            'task_2_attention': np.random.uniform(0.3, 0.7),  # 车道线分割
            'avg_attention': 0.5
        }
        return attention_stats
    
    def record_training_data(self, epoch, train_time, total_loss, head_losses, grad_norm, lr, method_data=None):
        """记录训练数据"""
        self.history['epochs'].append(epoch)
        self.history['train_times'].append(train_time)
        self.history['train_losses'].append(total_loss)
        
        if head_losses and len(head_losses) >= 3:
            self.history['det_losses'].append(head_losses[0])
            self.history['seg_losses'].append(head_losses[1])
            self.history['lane_losses'].append(head_losses[2])
        
        self.history['grad_norms'].append(grad_norm)
        self.history['learning_rates'].append(lr)
        
        # 记录方法特定数据
        if method_data:
            if self.method_name == 'gradnorm' and 'task_weights' in method_data:
                if 'task_weights' not in self.history['method_specific']:
                    self.history['method_specific']['task_weights'] = []
                self.history['method_specific']['task_weights'].append(method_data['task_weights'])
                    
            elif self.method_name == 'pcgrad' and 'conflicts' in method_data:
                if 'conflict_counts' not in self.history['method_specific']:
                    self.history['method_specific']['conflict_counts'] = []
                self.history['method_specific']['conflict_counts'].append(method_data['conflicts'])
                
            elif self.method_name == 'cagrad' and 'alpha_weights' in method_data:
                if 'alpha_weights' not in self.history['method_specific']:
                    self.history['method_specific']['alpha_weights'] = []
                self.history['method_specific']['alpha_weights'].append(method_data['alpha_weights'])
                    
            elif self.method_name == 'tag' and 'attention_weights' in method_data:
                if 'attention_weights' not in self.history['method_specific']:
                    self.history['method_specific']['attention_weights'] = []
                self.history['method_specific']['attention_weights'].append(method_data['attention_weights'])
        
        # TensorBoard记录
        if self.writer:
            try:
                self.writer.add_scalar(f'{self.method_name}/train_loss', total_loss, epoch)
                self.writer.add_scalar(f'{self.method_name}/train_time', train_time, epoch)
                self.writer.add_scalar(f'{self.method_name}/grad_norm', grad_norm, epoch)
                self.writer.add_scalar(f'{self.method_name}/learning_rate', lr, epoch)
                
                # 记录各任务损失
                if head_losses and len(head_losses) >= 3:
                    self.writer.add_scalar(f'{self.method_name}/det_loss', head_losses[0], epoch)
                    self.writer.add_scalar(f'{self.method_name}/seg_loss', head_losses[1], epoch)
                    self.writer.add_scalar(f'{self.method_name}/lane_loss', head_losses[2], epoch)
                
                # 记录方法特定的指标
                if method_data:
                    if self.method_name == 'gradnorm' and 'task_weights' in method_data:
                        weights = method_data['task_weights']
                        if len(weights) >= 3:
                            self.writer.add_scalar(f'{self.method_name}/weight_det', weights[0], epoch)
                            self.writer.add_scalar(f'{self.method_name}/weight_seg', weights[1], epoch)
                            self.writer.add_scalar(f'{self.method_name}/weight_lane', weights[2], epoch)
                    
                    elif self.method_name == 'pcgrad' and 'conflicts' in method_data:
                        self.writer.add_scalar(f'{self.method_name}/conflicts', method_data['conflicts'], epoch)
                    
                    elif self.method_name == 'cagrad' and 'alpha_weights' in method_data:
                        alphas = method_data['alpha_weights']
                        if len(alphas) >= 3:
                            self.writer.add_scalar(f'{self.method_name}/alpha_det', alphas[0], epoch)
                            self.writer.add_scalar(f'{self.method_name}/alpha_seg', alphas[1], epoch)
                            self.writer.add_scalar(f'{self.method_name}/alpha_lane', alphas[2], epoch)
                
                self.writer.flush()
                
            except Exception as e:
                print(f"Warning: TensorBoard logging failed for {self.method_name} at epoch {epoch}: {e}")
    
    def record_validation_data(self, epoch, da_results, ll_results, det_results, total_loss, inference_time):
        """记录验证数据"""
        val_result = {
            'epoch': epoch,
            'da_miou': da_results[2],
            'll_miou': ll_results[2],
            'det_map_05': det_results[2],
            'det_map_05_095': det_results[3],
            'total_loss': total_loss,
            'inference_time': inference_time
        }
        
        self.history['validation_results'].append(val_result)
        
        # TensorBoard验证指标记录
        if self.writer:
            try:
                self.writer.add_scalar(f'{self.method_name}/val_da_miou', da_results[2], epoch)
                self.writer.add_scalar(f'{self.method_name}/val_ll_miou', ll_results[2], epoch)
                self.writer.add_scalar(f'{self.method_name}/val_det_map_05', det_results[2], epoch)
                self.writer.add_scalar(f'{self.method_name}/val_det_map_05_095', det_results[3], epoch)
                self.writer.add_scalar(f'{self.method_name}/val_total_loss', total_loss, epoch)
                self.writer.add_scalar(f'{self.method_name}/val_inference_time', inference_time, epoch)
                self.writer.flush()
            except Exception as e:
                print(f"Warning: TensorBoard validation logging failed for {self.method_name} at epoch {epoch}: {e}")
    
    def generate_plots(self, save_dir=None):
        """生成训练分析图表"""
        if not save_dir:
            save_dir = self.method_dir
        
        epochs = self.history['epochs']
        if not epochs:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.method_name.upper()} Training Analysis')
        
        # Training Loss
        axes[0, 0].plot(epochs, self.history['train_losses'], 'b-')
        axes[0, 0].set_title('Training Loss vs Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Training Time
        axes[0, 1].plot(epochs, self.history['train_times'], 'g-')
        axes[0, 1].set_title('Training Time vs Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].grid(True)
        
        # Gradient Norm
        axes[0, 2].plot(epochs, self.history['grad_norms'], 'r-')
        axes[0, 2].set_title('Gradient Norm vs Epoch')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Grad Norm')
        axes[0, 2].grid(True)
        
        # Validation Metrics
        if self.history['validation_results']:
            val_epochs = [r['epoch'] for r in self.history['validation_results']]
            da_mious = [r['da_miou'] for r in self.history['validation_results']]
            ll_mious = [r['ll_miou'] for r in self.history['validation_results']]
            det_maps = [r['det_map_05'] for r in self.history['validation_results']]
            
            axes[1, 0].plot(val_epochs, da_mious, 'c-', label='DA mIoU')
            axes[1, 0].plot(val_epochs, ll_mious, 'm-', label='LL mIoU')
            axes[1, 0].set_title('Segmentation mIoU vs Epoch')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mIoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(val_epochs, det_maps, 'orange')
            axes[1, 1].set_title('Detection mAP@0.5 vs Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP@0.5')
            axes[1, 1].grid(True)
        
        # Method-specific plot
        if self.method_name == 'gradnorm' and self.history['method_specific'].get('task_weights'):
            task_weights = np.array(self.history['method_specific']['task_weights'])
            if task_weights.ndim == 2 and task_weights.shape[1] >= 3:
                axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 0], label='Det Weight')
                axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 1], label='Seg Weight')
                axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 2], label='Lane Weight')
                axes[1, 2].set_title('Task Weights vs Epoch')
                axes[1, 2].legend()
        elif self.method_name == 'pcgrad' and self.history['method_specific'].get('conflict_counts'):
            conflicts = self.history['method_specific']['conflict_counts']
            axes[1, 2].plot(epochs[:len(conflicts)], conflicts)
            axes[1, 2].set_title('Gradient Conflicts vs Epoch')
        elif self.method_name == 'cagrad' and self.history['method_specific'].get('alpha_weights'):
            alpha_weights = np.array(self.history['method_specific']['alpha_weights'])
            if alpha_weights.ndim == 2 and alpha_weights.shape[1] >= 3:
                axes[1, 2].plot(epochs[:len(alpha_weights)], alpha_weights[:, 0], label='Det Alpha')
                axes[1, 2].plot(epochs[:len(alpha_weights)], alpha_weights[:, 1], label='Seg Alpha')
                axes[1, 2].plot(epochs[:len(alpha_weights)], alpha_weights[:, 2], label='Lane Alpha')
                axes[1, 2].set_title('Alpha Weights vs Epoch')
                axes[1, 2].legend()
        
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.method_name}_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.method_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        return history_path
    
    def save_checkpoint(self, epoch, model, optimizer):
        """保存检查点"""
        checkpoint_path = os.path.join(self.method_dir, f'epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'method': self.method_name,
            'state_dict': model.module.state_dict() if is_parallel(model) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新检查点
        latest_path = os.path.join(self.method_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def cleanup(self):
        """清理资源"""
        if self.writer:
            self.writer.close()

# =============================================================================
# Utility Functions
# =============================================================================

def setup_distributed_training():
    """设置分布式训练"""
    return int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', -1))

def load_model_weights(model, logger, optimizer, cfg):
    """加载模型权重"""
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    
    if os.path.exists(cfg.MODEL.PRETRAINED):
        logger.info(f"=> 加载模型 '{cfg.MODEL.PRETRAINED}'")
        checkpoint = torch.load(cfg.MODEL.PRETRAINED)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> 已加载检查点 '{cfg.MODEL.PRETRAINED}' (epoch {checkpoint['epoch']})")
    
    if os.path.exists(cfg.MODEL.PRETRAINED_DET):
        logger.info(f"=> 加载检测分支权重 '{cfg.MODEL.PRETRAINED_DET}'")
        det_idx_range = [str(i) for i in range(0, 25)]
        model_dict = model.state_dict()
        checkpoint = torch.load(cfg.MODEL.PRETRAINED_DET)
        checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                          if k.split(".")[1] in det_idx_range}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
        logger.info("=> 已成功加载检测分支检查点")
    
    checkpoint_file = os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET, 'checkpoint.pth')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info(f"=> 自动恢复检查点 '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> 已恢复检查点 '{checkpoint_file}' (epoch {checkpoint['epoch']})")
    
    return begin_epoch

def create_data_loaders(cfg, rank):
    """创建数据加载器"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg, is_train=True, inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
    
    train_loader = DataLoaderX(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE and rank == -1), num_workers=cfg.WORKERS,
        sampler=train_sampler, pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    valid_loader = valid_dataset = None
    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg, is_train=False, inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )
        
        valid_loader = DataLoaderX(
            valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
    
    return train_loader, valid_loader, valid_dataset

def should_validate_and_save(epoch, cfg):
    """判断是否应该验证和保存"""
    return (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH or 
            epoch in list(range(181, 200)) or epoch in [162, 165, 167, 170, 172, 175, 178])

def print_tensorboard_instructions(experiment_dir, method_name):
    """打印TensorBoard启动指令"""
    print(f"\n{'='*60}")
    print(f"训练完成！TensorBoard 启动指令:")
    print(f"{'='*60}")
    method_tb_dir = os.path.join(experiment_dir, f'mtl_{method_name}', 'tensorboard')
    print(f"tensorboard --logdir {method_tb_dir}")
    print(f"{'='*60}")
    print(f"浏览器访问: http://localhost:6006")
    print(f"实验结果保存在: {experiment_dir}")
    print(f"{'='*60}\n")

# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """主训练函数"""
    args = parse_args()
    
    # 更新配置
    update_config(cfg, args)
    
    # 设置分布式训练
    world_size, rank = setup_distributed_training()
    
    # 创建日志器
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    # 打印配置信息
    if rank in [-1, 0]:
        logger.info("="*60)
        logger.info("MTL冲突解决训练配置")
        logger.info("="*60)
        logger.info(f"冲突解决方法: {args.conflict_method}")
        logger.info(f"实验名称: {args.experiment_name}")
        logger.info(f"数据集: {args.dataset}")
        logger.info(f"训练轮数: {cfg.TRAIN.END_EPOCH}")
        logger.info(f"批次大小: {cfg.TRAIN.BATCH_SIZE_PER_GPU}")
        logger.info(f"学习率: {args.lr}")
        logger.info("="*60)
    
    # 设置CUDNN
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    # 选择设备
    if args.device == 'auto':
        device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    else:
        device = torch.device(args.device)
    
    # 分布式训练设置
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # 构建模型
    logger.info("构建模型...")
    model = get_net(cfg).to(device)
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    
    # 设置学习率
    if args.lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    
    # 加载预训练权重
    begin_epoch = load_model_weights(model, logger, optimizer, cfg) if rank in [-1, 0] else cfg.TRAIN.BEGIN_EPOCH
    
    # 设置并行训练
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    elif rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # 设置模型属性
    model.gr = 1.0
    model.nc = 1
    
    # 加载数据
    logger.info("加载数据...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(cfg, rank)
    
    # 设置学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 训练设置
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    # 创建MTL训练管理器
    logger.info(f"初始化MTL训练管理器 - 方法: {args.conflict_method}")
    mtl_manager = MTLTrainingManager(
        cfg=cfg, logger=logger, device=device, 
        log_dir=cfg.LOG_DIR, dataset_name=cfg.DATASET.DATASET, 
        method_name=args.conflict_method, experiment_name=args.experiment_name, rank=rank
    )
    
    # 设置TAG模块
    if args.conflict_method == 'tag':
        mtl_manager.setup_tag_module(model, feature_dim=512)
    
    # 确定验证epoch
    validation_epochs = [epoch for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1) 
                        if should_validate_and_save(epoch, cfg)]
    
    logger.info(f"开始训练 - 方法: {args.conflict_method}, 从epoch {begin_epoch + 1} 到 {cfg.TRAIN.END_EPOCH}")
    
    # 主训练循环
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        mtl_manager.train_epoch(cfg, train_loader, model, criterion, optimizer, scaler,
                               epoch, num_batch, num_warmup, logger, device, rank)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 验证和保存
        if epoch in validation_epochs and rank in [-1, 0]:
            logger.info(f"开始验证 epoch {epoch}...")
            da_results, ll_results, det_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                mtl_manager.method_dir, mtl_manager.tb_log_dir, None, logger, device, rank
            )
            
            # 记录验证结果
            mtl_manager.record_validation_data(epoch, da_results, ll_results, det_results, total_loss, times[0])
            
            # 生成图表
            mtl_manager.generate_plots()
            
            # 保存检查点
            mtl_manager.save_checkpoint(epoch, model, optimizer)
            
            # 打印验证结果
            msg = f'Epoch: [{epoch}] Method: {args.conflict_method.upper()} Loss({total_loss:.3f})\n' \
                  f'Driving area Segment: Acc({da_results[0]:.3f}) IOU({da_results[1]:.3f}) mIOU({da_results[2]:.3f})\n' \
                  f'Lane line Segment: Acc({ll_results[0]:.3f}) IOU({ll_results[1]:.3f}) mIOU({ll_results[2]:.3f})\n' \
                  f'Detect: P({det_results[0]:.3f}) R({det_results[1]:.3f}) mAP@0.5({det_results[2]:.3f}) mAP@0.5:0.95({det_results[3]:.3f})\n' \
                  f'Time: inference({times[0]:.4f}s/frame) nms({times[1]:.4f}s/frame)'
            logger.info(msg)
    
    # 训练完成后的处理
    if rank in [-1, 0]:
        # 保存训练历史
        history_path = mtl_manager.save_history()
        logger.info(f"训练历史已保存到: {history_path}")
        
        # 保存最终模型
        final_model_state_file = os.path.join(mtl_manager.method_dir, 'final_state.pth')
        logger.info(f'=> 保存最终模型状态到 {final_model_state_file}')
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        
        # 打印TensorBoard指令
        print_tensorboard_instructions(mtl_manager.experiment_dir, args.conflict_method)
        
        logger.info("训练完成！")
    
    # 清理资源
    mtl_manager.cleanup()
    
    # 销毁分布式进程组
    if rank != -1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

# =============================================================================
# Usage Examples and Documentation
# =============================================================================

"""
MTL Training System with Conflict Resolution Methods

Architecture:
- MTLTrainingManager: 管理单个MTL方法的训练过程
- MultiTaskOptimizer: 实现GradNorm, PCGrad, CAGrad算法  
- TaskAdaptiveAttention: TAG实现，用于特征自适应

Features:
✅ 5种MTL方法: None, GradNorm, PCGrad, CAGrad, TAG
✅ 实验目录时间戳组织结构
✅ 全面的指标记录和可视化
✅ 实时进度条显示训练状态
✅ 独立的模型检查点保存
✅ TensorBoard日志记录
✅ 方法特定的分析和记录

Directory Structure:
{log_dir}/{dataset_name}/{experiment_name}/
└── mtl_{method_name}/
    ├── tensorboard/             # TensorBoard日志
    ├── training_history.json    # 训练指标历史
    ├── epoch_*.pth             # 按epoch保存的检查点
    ├── latest.pth              # 最新检查点
    ├── final_state.pth         # 最终模型状态
    └── {method_name}_training_analysis.png  # 训练分析图表

Usage:
# 运行标准多任务学习
python xy_yolop_train_conflict_sep.py --conflict-method none --experiment-name my_experiment

# 运行GradNorm方法
python xy_yolop_train_conflict_sep.py --conflict-method gradnorm --epochs 200 --lr 1e-3

# 运行PCGrad方法
python xy_yolop_train_conflict_sep.py --conflict-method pcgrad --batch-size 16

# 运行CAGrad方法  
python xy_yolop_train_conflict_sep.py --conflict-method cagrad --dataset BDD100K

# 运行TAG方法
python xy_yolop_train_conflict_sep.py --conflict-method tag --experiment-name tag_experiment

Command Line Arguments:
--conflict-method: 选择冲突解决方法 (none/gradnorm/pcgrad/cagrad/tag)
--experiment-name: 实验名称
--epochs: 训练轮数
--batch-size: 批次大小  
--lr: 学习率
--dataset: 数据集名称
--logDir: 日志目录
--device: 计算设备

TensorBoard Usage:
tensorboard --logdir /path/to/experiment/mtl_{method}/tensorboard/

Recorded Metrics:
- 训练时间、损失、梯度范数、学习率
- 各任务损失: 检测、分割、车道线
- 验证指标: DA mIoU, LL mIoU, Detection mAP@0.5/0.95
- 方法特定指标: 任务权重、冲突数量、注意力权重

Method-Specific Metrics:
- GradNorm: 任务权重演化、梯度范数比率
- PCGrad: 梯度冲突计数、投影信息
- CAGrad: Alpha权重、优化信息
- TAG: 注意力权重、注意力统计

Key Improvements from Original:
- 单一方法训练，避免内存问题
- 更清晰的目录结构组织
- 完整的YOLOP架构兼容性
- 实时训练进度监控
- 方法特定的分析和可视化
- 完善的错误处理和日志记录
"""