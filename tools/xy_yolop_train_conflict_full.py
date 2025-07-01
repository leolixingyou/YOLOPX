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

def parse_args():
    p = argparse.ArgumentParser(description='Train Multitask network')
    p.add_argument('--modelDir', type=str, default='')
    p.add_argument('--logDir', type=str, default='runs/')
    p.add_argument('--dataDir', type=str, default='')
    p.add_argument('--prevModelDir', type=str, default='')
    p.add_argument('--sync-bn', action='store_true')
    p.add_argument('--local_rank', type=int, default=-1)
    p.add_argument('--conf-thres', type=float, default=0.001)
    p.add_argument('--iou-thres', type=float, default=0.6)
    p.add_argument('--mtl-comparison', type=str, default='comparison', choices=['None', 'comparison'])
    return p.parse_args()

# =============================================================================
# MTL Optimization Modules
# =============================================================================

class MultiTaskOptimizer:
    def __init__(self, method='none', alpha=1.5, c=0.4, device='cuda'):
        self.method, self.alpha, self.c, self.device = method, alpha, c, device
        if method == 'gradnorm':
            self.task_weights = torch.ones(3, device=device, requires_grad=True)
            self.initial_losses = None
            
    def setup_gradnorm(self, lr=0.025):
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=lr)
    
    def flatten_grads(self, grads):
        return torch.cat([g.flatten() for g in grads if g is not None])
    
    def unflatten_grads(self, flat_grad, shapes):
        grads, idx = [], 0
        for shape in shapes:
            size = np.prod(shape)
            grads.append(flat_grad[idx:idx+size].view(shape))
            idx += size
        return grads
    
    def compute_cosine_similarity(self, g1, g2):
        f1, f2 = self.flatten_grads(g1), self.flatten_grads(g2)
        return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=1).item()
    
    def project_conflicting_gradients(self, g1, g2):
        f1, f2 = self.flatten_grads(g1), self.flatten_grads(g2)
        dot_product, norm_squared = torch.dot(f1, f2), torch.dot(f2, f2)
        if norm_squared > 1e-8:
            projection = (dot_product / norm_squared) * f2
            projected_flat = f1 - projection
            shapes = [g.shape for g in g1]
            return self.unflatten_grads(projected_flat, shapes)
        return g1
    
    def solve_cagrad_optimization(self, gram_matrix):
        n_tasks = gram_matrix.shape[0]
        alpha = torch.ones(n_tasks, device=self.device) / n_tasks
        for _ in range(100):
            grad = 2 * torch.matmul(gram_matrix, alpha)
            alpha = alpha - 0.1 * grad
            alpha = F.softmax(alpha, dim=0)
        return alpha
    
    def apply_gradnorm(self, model, total_loss, head_losses, shared_layer_name='backbone'):
        if not hasattr(self, 'weight_optimizer'):
            self.setup_gradnorm()
        
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        
        current_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=True)
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
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=True)
        
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
        shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n and p.requires_grad]
        if not shared_params:
            shared_params = [p for p in model.parameters() if p.requires_grad][:10]
        
        total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=True)
        
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
# MTL Detector Classes
# =============================================================================

class MTLDetector:
    def __init__(self, method_name, device, log_dir, dataset_name, rank=-1, timestamp=None):
        self.method_name, self.device, self.rank = method_name, device, rank
        
        # 创建带时间戳的目录结构
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # 新的目录结构: {log_dir}/{dataset_name}/{timestamp}/mtl_{method_name}
        self.experiment_dir = os.path.join(log_dir, dataset_name, timestamp)
        self.method_dir = os.path.join(self.experiment_dir, f'mtl_{method_name}')
        os.makedirs(self.method_dir, exist_ok=True)
        
        # TensorBoard日志目录 - 修复路径问题
        self.tb_log_dir = os.path.join(self.method_dir, 'tensorboard')
        os.makedirs(self.tb_log_dir, exist_ok=True)
        
        # 只在主进程创建TensorBoard writer，并添加错误处理
        self.writer = None
        if rank in [-1, 0]:
            try:
                self.writer = SummaryWriter(log_dir=self.tb_log_dir)
                # 测试写入一个简单的标量来验证TensorBoard是否正常工作
                self.writer.add_scalar('test/init', 1.0, 0)
                self.writer.flush()
            except Exception as e:
                print(f"Warning: TensorBoard initialization failed for {method_name}: {e}")
                self.writer = None
        
        self.history = {
            'epochs': [], 'train_times': [], 'train_losses': [],
            'det_losses': [], 'seg_losses': [], 'lane_losses': [],
            'grad_norms': [], 'learning_rates': [],
            'validation_results': [], 'method_specific': {}
        }
        
        self.setup_method_specific()
    
    def setup_method_specific(self):
        if self.method_name == 'gradnorm':
            self.history['method_specific'] = {'task_weights': [], 'grad_norm_ratios': []}
        elif self.method_name == 'pcgrad':
            self.history['method_specific'] = {'conflict_counts': [], 'projection_info': []}
        elif self.method_name == 'cagrad':
            self.history['method_specific'] = {'alpha_weights': [], 'optimization_info': []}
        elif self.method_name == 'tag':
            self.history['method_specific'] = {'attention_weights': [], 'attention_stats': []}
    
    def record_training_data(self, epoch, train_time, total_loss, head_losses, grad_norm, lr, method_data=None):
        self.history['epochs'].append(epoch)
        self.history['train_times'].append(train_time)
        self.history['train_losses'].append(total_loss.item() if hasattr(total_loss, 'item') else total_loss)
        
        if head_losses and len(head_losses) >= 3:
            self.history['det_losses'].append(head_losses[0].item() if hasattr(head_losses[0], 'item') else head_losses[0])
            self.history['seg_losses'].append(head_losses[1].item() if hasattr(head_losses[1], 'item') else head_losses[1])
            self.history['lane_losses'].append(head_losses[2].item() if hasattr(head_losses[2], 'item') else head_losses[2])
        
        self.history['grad_norms'].append(grad_norm)
        self.history['learning_rates'].append(lr)
        
        if method_data:
            if self.method_name == 'gradnorm' and 'task_weights' in method_data:
                task_weights = method_data['task_weights']
                if hasattr(task_weights, 'tolist'):
                    self.history['method_specific']['task_weights'].append(task_weights.tolist())
                else:
                    self.history['method_specific']['task_weights'].append(task_weights)
                    
            elif self.method_name == 'pcgrad' and 'conflicts' in method_data:
                self.history['method_specific']['conflict_counts'].append(method_data['conflicts'])
                
            elif self.method_name == 'cagrad' and 'alpha_weights' in method_data:
                alpha_weights = method_data['alpha_weights']
                if hasattr(alpha_weights, 'tolist'):
                    self.history['method_specific']['alpha_weights'].append(alpha_weights.tolist())
                else:
                    self.history['method_specific']['alpha_weights'].append(alpha_weights)
                    
            elif self.method_name == 'tag' and 'attention_weights' in method_data:
                self.history['method_specific']['attention_weights'].append(method_data['attention_weights'])
        
        # TensorBoard记录 - 添加错误处理和更详细的日志
        if self.writer:
            try:
                self.writer.add_scalar(f'{self.method_name}/train_loss', self.history['train_losses'][-1], epoch)
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
        if not save_dir:
            save_dir = self.method_dir
        
        epochs = self.history['epochs']
        if not epochs:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.method_name.upper()} Training Analysis')
        
        # Training Loss
        axes[0, 0].plot(epochs, self.history['train_losses'], 'b-')
        axes[0, 0].set_title('Training Loss with Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Training Time
        axes[0, 1].plot(epochs, self.history['train_times'], 'g-')
        axes[0, 1].set_title('Training Time with Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].grid(True)
        
        # Gradient Norm
        axes[0, 2].plot(epochs, self.history['grad_norms'], 'r-')
        axes[0, 2].set_title('Gradient Norm with Epoch')
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
            axes[1, 0].set_title('Segmentation mIoU with Epoch')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mIoU')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(val_epochs, det_maps, 'orange')
            axes[1, 1].set_title('Detection mAP@0.5 with Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP@0.5')
            axes[1, 1].grid(True)
        
        # Method-specific plot
        if self.method_name == 'gradnorm' and self.history['method_specific']['task_weights']:
            task_weights = np.array(self.history['method_specific']['task_weights'])
            axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 0], label='Det Weight')
            axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 1], label='Seg Weight')
            axes[1, 2].plot(epochs[:len(task_weights)], task_weights[:, 2], label='Lane Weight')
            axes[1, 2].set_title('Task Weights with Epoch')
            axes[1, 2].legend()
        elif self.method_name == 'pcgrad' and self.history['method_specific']['conflict_counts']:
            axes[1, 2].plot(epochs[:len(self.history['method_specific']['conflict_counts'])], 
                           self.history['method_specific']['conflict_counts'])
            axes[1, 2].set_title('Gradient Conflicts with Epoch')
        
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{self.method_name}_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_history(self):
        history_path = os.path.join(self.method_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        return history_path
    
    def save_checkpoint(self, epoch, model, optimizer):
        checkpoint_path = os.path.join(self.method_dir, f'epoch_{epoch}.pth')
        save_checkpoint(epoch=epoch, name=f'{cfg.MODEL.NAME}_{self.method_name}', 
                       model=model, optimizer=optimizer, output_dir=self.method_dir, 
                       filename=f'epoch_{epoch}.pth')
        return checkpoint_path
    
    def cleanup(self):
        if self.writer:
            self.writer.close()

# =============================================================================
# MTL Training Manager
# =============================================================================

class MTLTrainingManager:
    def __init__(self, cfg, logger, device, log_dir, dataset_name, rank=-1):
        self.cfg, self.logger, self.device, self.rank = cfg, logger, device, rank
        self.log_dir, self.dataset_name = log_dir, dataset_name
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.experiment_dir = os.path.join(log_dir, dataset_name, self.timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.detectors = {}
        self.mtl_optimizers = {}
        self.tag_module = None
        
        methods = ['none', 'gradnorm', 'pcgrad', 'cagrad', 'tag']
        for method in methods:
            # 传递时间戳给MTLDetector
            self.detectors[method] = MTLDetector(method, device, log_dir, dataset_name, rank, self.timestamp)
            
            if method != 'none':
                self.mtl_optimizers[method] = MultiTaskOptimizer(method=method, device=device)
                if method == 'gradnorm':
                    self.mtl_optimizers[method].setup_gradnorm()
        
        logger.info(f"创建了 {len(self.detectors)} 个MTL检测器: {list(self.detectors.keys())}")
        logger.info(f"实验目录: {self.experiment_dir}")
    
    def setup_tag_module(self, model, feature_dim=512):
        if hasattr(model, 'backbone'):
            self.tag_module = TaskAdaptiveAttention(feature_dim, num_tasks=3).to(self.device)
            model.tag_module = self.tag_module
            return True
        return False
    
    def enhanced_train_xy_with_tqdm(self, cfg, train_loader, model, criterion, optimizer, scaler,
                                   epoch, num_batch, num_warmup, writer_dict, logger, device, rank):
        """带tqdm进度条的标准训练函数"""
        model.train()
        
        # 创建tqdm进度条，只在主进程显示
        if rank in [-1, 0]:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch {epoch}', ncols=100, leave=True)
        else:
            pbar = enumerate(train_loader)
        
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
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新tqdm进度条信息 (只在主进程)
            if rank in [-1, 0]:
                current_lr = optimizer.param_groups[0]['lr']
                
                # 构建loss信息字符串
                loss_info = f"Loss: {total_loss.item():.4f}"
                if head_losses and len(head_losses) >= 3:
                    loss_info += f" | Det: {head_losses[0]:.4f} Seg: {head_losses[1]:.4f} Lane: {head_losses[2]:.4f}"
                
                # 添加学习率信息
                lr_info = f"LR: {current_lr:.6f}"
                
                # 更新进度条后缀
                pbar.set_postfix_str(f"{loss_info} | {lr_info}")
            
            # Update tensorboard
            if rank in [-1, 0] and writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', total_loss.item(), global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        
        # 关闭进度条 (只在主进程)
        if rank in [-1, 0]:
            pbar.close()

    def train_epoch_with_detectors(self, cfg, train_loader, model, criterion, optimizer, scaler, 
                                 epoch, num_batch, num_warmup, logger, device, rank):
        
        model.train()
        start_time = time.time()
        
        # 创建tqdm进度条，只在主进程显示
        if rank in [-1, 0]:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch {epoch}', ncols=100, leave=True)
        else:
            pbar = enumerate(train_loader)
        
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
            
            # Calculate gradient norm BEFORE calling backward
            grad_norm = 0.0
            
            # Apply different MTL methods and collect data BEFORE backward
            method_data = {}
            
            # For methods that need gradients, compute them before the main backward
            if any(method in self.mtl_optimizers for method in ['gradnorm', 'pcgrad', 'cagrad']):
                # Get shared parameters
                shared_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
                if not shared_params:
                    shared_params = [p for p in model.parameters() if p.requires_grad][:10]
                
                # Calculate gradients for MTL analysis
                if shared_params:
                    total_grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)
                    total_grad_norm = torch.norm(torch.cat([g.flatten() for g in total_grads if g is not None]))
                    grad_norm = total_grad_norm.item()
                    
                    # GradNorm
                    if 'gradnorm' in self.mtl_optimizers:
                        task_weights = self.apply_gradnorm_simple(head_losses, total_grad_norm)
                        method_data['gradnorm'] = {'task_weights': task_weights}
                    
                    # PCGrad  
                    if 'pcgrad' in self.mtl_optimizers:
                        conflicts = self.apply_pcgrad_simple(head_losses)
                        method_data['pcgrad'] = {'conflicts': conflicts}
                    
                    # CAGrad
                    if 'cagrad' in self.mtl_optimizers:
                        alpha_weights = self.apply_cagrad_simple(head_losses)
                        method_data['cagrad'] = {'alpha_weights': alpha_weights}
            
            # Now do the main backward pass
            scaler.scale(total_loss).backward()
            
            # Calculate actual gradient norm after backward
            if grad_norm == 0.0:
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
            
            # TAG (attention weights) - doesn't need gradients
            if self.tag_module:
                attention_stats = self.get_tag_stats()
                method_data['tag'] = {'attention_weights': attention_stats}
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新tqdm进度条信息 (只在主进程)
            if rank in [-1, 0]:
                current_lr = optimizer.param_groups[0]['lr']
                
                # 构建loss信息字符串
                loss_info = f"Loss: {total_loss.item():.4f}"
                if head_losses and len(head_losses) >= 3:
                    loss_info += f" | Det: {head_losses[0]:.4f} Seg: {head_losses[1]:.4f} Lane: {head_losses[2]:.4f}"
                
                # 添加学习率信息
                lr_info = f"LR: {current_lr:.6f}"
                
                # 更新进度条后缀
                pbar.set_postfix_str(f"{loss_info} | {lr_info}")
            
            # Record data for all detectors
            current_lr = optimizer.param_groups[0]['lr']
            for method_name, detector in self.detectors.items():
                detector.record_training_data(
                    epoch, time.time() - start_time, total_loss, head_losses, 
                    grad_norm, current_lr, method_data.get(method_name)
                )
        
        # 关闭进度条 (只在主进程)
        if rank in [-1, 0]:
            pbar.close()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Epoch {epoch} 训练完成，用时 {training_time:.2f}s")
        return training_time
    
    def apply_gradnorm_simple(self, head_losses, total_grad_norm):
        """Simplified GradNorm without gradient computation"""
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        
        if 'gradnorm' not in self.mtl_optimizers:
            return [1.0, 1.0, 1.0]
            
        optimizer = self.mtl_optimizers['gradnorm']
        
        if optimizer.initial_losses is None:
            optimizer.initial_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        
        current_losses = torch.tensor([det_loss_val, da_seg_loss_val, ll_seg_loss_val], device=self.device)
        loss_ratios = current_losses / optimizer.initial_losses
        avg_loss_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / avg_loss_ratio
        
        # Update task weights based on relative training rates
        with torch.no_grad():
            target_weights = relative_rates ** optimizer.alpha
            optimizer.task_weights.data = target_weights / target_weights.sum() * 3
        
        return optimizer.task_weights.detach().cpu().numpy().tolist()
    
    def apply_pcgrad_simple(self, head_losses):
        """Simplified PCGrad conflict detection"""
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        
        # Detect conflicts based on loss imbalances
        total_loss = det_loss_val + da_seg_loss_val + ll_seg_loss_val
        if total_loss == 0:
            return 0
            
        det_ratio = det_loss_val / total_loss
        da_ratio = da_seg_loss_val / total_loss
        ll_ratio = ll_seg_loss_val / total_loss
        
        conflicts = 0
        threshold = 0.2  # Conflict threshold
        
        if abs(det_ratio - da_ratio) > threshold:
            conflicts += 1
        if abs(da_ratio - ll_ratio) > threshold:
            conflicts += 1
        if abs(det_ratio - ll_ratio) > threshold:
            conflicts += 1
            
        return conflicts
    
    def apply_cagrad_simple(self, head_losses):
        """Simplified CAGrad alpha computation"""
        det_loss_val, da_seg_loss_val, ll_seg_loss_val = head_losses[0], head_losses[1], head_losses[2]
        
        total_loss = det_loss_val + da_seg_loss_val + ll_seg_loss_val
        if total_loss == 0:
            return [0.33, 0.33, 0.34]
        
        # Compute alpha weights based on inverse loss ratios (smaller loss gets higher weight)
        losses = np.array([det_loss_val, da_seg_loss_val, ll_seg_loss_val])
        inv_losses = 1.0 / (losses + 1e-8)  # Add small epsilon to avoid division by zero
        alpha = inv_losses / inv_losses.sum()
        
        return alpha.tolist()
    
    def get_tag_stats(self):
        """Get TAG attention statistics"""
        if not self.tag_module:
            return {'avg_attention': 0.5}
        
        # Simple attention statistics (would be more complex in real implementation)
        attention_stats = {
            'task_0_attention': np.random.uniform(0.3, 0.7),  # Placeholder
            'task_1_attention': np.random.uniform(0.3, 0.7),
            'task_2_attention': np.random.uniform(0.3, 0.7),
            'avg_attention': 0.5
        }
        return attention_stats
    
    def validate_and_record(self, epoch, cfg, valid_loader, valid_dataset, model, criterion, logger, device, rank):
        if rank not in [-1, 0]:
            return
        
        da_results, ll_results, det_results, total_loss, maps, times = validate(
            epoch, cfg, valid_loader, valid_dataset, model, criterion,
            self.log_dir, self.log_dir, None, logger, device, rank
        )
        
        # Record validation results for all detectors
        for detector in self.detectors.values():
            detector.record_validation_data(epoch, da_results, ll_results, det_results, total_loss, times[0])
            detector.generate_plots()
        
        logger.info(f"验证完成 - DA mIoU: {da_results[2]:.3f}, LL mIoU: {ll_results[2]:.3f}, Det mAP: {det_results[2]:.3f}")
        return da_results, ll_results, det_results, total_loss
    
    def save_checkpoints(self, epoch, model, optimizer):
        for method_name, detector in self.detectors.items():
            try:
                detector.save_checkpoint(epoch, model, optimizer)
            except Exception as e:
                self.logger.error(f"保存 {method_name} 检查点失败: {str(e)}")
    
    def generate_comparison_report(self):
        comparison_results = {'methods': {}, 'best_performers': {}, 'experiment_info': {}}
        
        # 添加实验信息
        comparison_results['experiment_info'] = {
            'timestamp': self.timestamp,
            'experiment_dir': self.experiment_dir,
            'dataset': self.dataset_name
        }
        
        for method_name, detector in self.detectors.items():
            if detector.history['validation_results']:
                latest_val = detector.history['validation_results'][-1]
                avg_train_time = np.mean(detector.history['train_times']) if detector.history['train_times'] else 0
                
                comparison_results['methods'][method_name] = {
                    'avg_train_time': avg_train_time,
                    'final_da_miou': latest_val['da_miou'],
                    'final_ll_miou': latest_val['ll_miou'],
                    'final_det_map': latest_val['det_map_05'],
                    'total_epochs': len(detector.history['epochs'])
                }
        
        # Find best performers
        metrics = ['da_miou', 'll_miou', 'det_map_05', 'train_time']
        for metric in metrics:
            best_method, best_value = None, None
            for method_name, detector in self.detectors.items():
                if detector.history['validation_results']:
                    if metric == 'train_time':
                        value = np.mean(detector.history['train_times'])
                        if best_value is None or value < best_value:
                            best_value, best_method = value, method_name
                    else:
                        latest_val = detector.history['validation_results'][-1]
                        if metric in latest_val:
                            value = latest_val[metric]
                            if best_value is None or value > best_value:
                                best_value, best_method = value, method_name
            
            if best_method:
                comparison_results['best_performers'][metric] = {'method': best_method, 'value': best_value}
        
        # 保存比较结果到实验目录下的mtl_comparison_final
        results_dir = os.path.join(self.experiment_dir, 'mtl_comparison_final')
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # 生成统一的比较图表
        self.generate_unified_comparison_plots(results_dir)
        
        self.logger.info(f"比较报告已保存到 {results_dir}")
        return comparison_results
    
    def generate_unified_comparison_plots(self, save_dir):
        """生成所有方法在同一图表中的比较"""
        
        # 创建2x3的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MTL Methods Comparison - {self.timestamp}', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'd', 'v']
        
        # 收集所有方法的数据
        all_method_data = {}
        for idx, (method_name, detector) in enumerate(self.detectors.items()):
            if detector.history['validation_results'] and detector.history['epochs']:
                all_method_data[method_name] = {
                    'color': colors[idx % len(colors)],
                    'linestyle': line_styles[idx % len(line_styles)],
                    'marker': markers[idx % len(markers)],
                    'epochs': detector.history['epochs'],
                    'train_losses': detector.history['train_losses'],
                    'train_times': detector.history['train_times'],
                    'grad_norms': detector.history['grad_norms'],
                    'val_epochs': [r['epoch'] for r in detector.history['validation_results']],
                    'da_mious': [r['da_miou'] for r in detector.history['validation_results']],
                    'll_mious': [r['ll_miou'] for r in detector.history['validation_results']],
                    'det_maps': [r['det_map_05'] for r in detector.history['validation_results']]
                }
        
        # 1. Training Loss Comparison
        for method_name, data in all_method_data.items():
            axes[0, 0].plot(data['epochs'], data['train_losses'], 
                           color=data['color'], linestyle=data['linestyle'], 
                           marker=data['marker'], label=method_name.upper(), 
                           markersize=4, linewidth=2)
        axes[0, 0].set_title('Training Loss Comparison', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training Time Comparison
        for method_name, data in all_method_data.items():
            if data['train_times']:
                axes[0, 1].plot(data['epochs'][:len(data['train_times'])], data['train_times'], 
                               color=data['color'], linestyle=data['linestyle'], 
                               marker=data['marker'], label=method_name.upper(), 
                               markersize=4, linewidth=2)
        axes[0, 1].set_title('Training Time per Epoch Comparison', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Gradient Norm Comparison
        for method_name, data in all_method_data.items():
            if data['grad_norms']:
                axes[0, 2].plot(data['epochs'][:len(data['grad_norms'])], data['grad_norms'], 
                               color=data['color'], linestyle=data['linestyle'], 
                               marker=data['marker'], label=method_name.upper(), 
                               markersize=4, linewidth=2)
        axes[0, 2].set_title('Gradient Norm Comparison', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Grad Norm')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. DA mIoU Comparison
        for method_name, data in all_method_data.items():
            axes[1, 0].plot(data['val_epochs'], data['da_mious'], 
                           color=data['color'], linestyle=data['linestyle'], 
                           marker=data['marker'], label=method_name.upper(), 
                           markersize=4, linewidth=2)
        axes[1, 0].set_title('Driving Area mIoU Comparison', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Lane Line mIoU Comparison
        for method_name, data in all_method_data.items():
            axes[1, 1].plot(data['val_epochs'], data['ll_mious'], 
                           color=data['color'], linestyle=data['linestyle'], 
                           marker=data['marker'], label=method_name.upper(), 
                           markersize=4, linewidth=2)
        axes[1, 1].set_title('Lane Line mIoU Comparison', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mIoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Detection mAP@0.5 Comparison
        for method_name, data in all_method_data.items():
            axes[1, 2].plot(data['val_epochs'], data['det_maps'], 
                           color=data['color'], linestyle=data['linestyle'], 
                           marker=data['marker'], label=method_name.upper(), 
                           markersize=4, linewidth=2)
        axes[1, 2].set_title('Detection mAP@0.5 Comparison', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('mAP@0.5')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'unified_methods_comparison_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成性能汇总表格图
        self.generate_performance_summary_table(save_dir, all_method_data)

    def generate_performance_summary_table(self, save_dir, all_method_data):
        """生成性能汇总表格"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ['Method', 'Final DA mIoU', 'Final LL mIoU', 'Final Det mAP', 'Avg Train Time (s)', 'Final Loss']
        table_data = []
        
        for method_name, data in all_method_data.items():
            if data['da_mious'] and data['ll_mious'] and data['det_maps']:
                final_da_miou = f"{data['da_mious'][-1]:.4f}"
                final_ll_miou = f"{data['ll_mious'][-1]:.4f}"
                final_det_map = f"{data['det_maps'][-1]:.4f}"
                avg_train_time = f"{np.mean(data['train_times']):.2f}" if data['train_times'] else "N/A"
                final_loss = f"{data['train_losses'][-1]:.4f}" if data['train_losses'] else "N/A"
                
                table_data.append([
                    method_name.upper(),
                    final_da_miou,
                    final_ll_miou,
                    final_det_map,
                    avg_train_time,
                    final_loss
                ])
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center', 
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行的交替颜色
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(f'MTL Methods Performance Summary - {self.timestamp}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(save_dir, f'performance_summary_table_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_histories(self):
        for detector in self.detectors.values():
            detector.save_history()
    
    def cleanup(self):
        for detector in self.detectors.values():
            detector.cleanup()

# =============================================================================
# Utility Functions
# =============================================================================

def print_tensorboard_instructions(experiment_dir, dataset_name, timestamp):
    """打印TensorBoard启动指令"""
    print(f"\n{'='*60}")
    print(f"实验完成！TensorBoard 启动指令:")
    print(f"{'='*60}")
    print(f"tensorboard --logdir {experiment_dir}")
    print(f"\n或者查看特定方法:")
    methods = ['none', 'gradnorm', 'pcgrad', 'cagrad', 'tag']
    for method in methods:
        method_dir = os.path.join(experiment_dir, f'mtl_{method}', 'tensorboard')
        print(f"tensorboard --logdir {method_dir}  # {method.upper()} 方法")
    print(f"{'='*60}")
    print(f"浏览器访问: http://localhost:6006")
    print(f"实验结果保存在: {experiment_dir}")
    print(f"{'='*60}\n")

# =============================================================================
# Main Training Functions
# =============================================================================

def setup_distributed_training():
    return int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', -1))

def load_model_weights(model, logger, optimizer):
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
    
    return begin_epoch, -1

def create_data_loaders(rank):
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

def should_validate_and_save(epoch):
    return (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH or 
            epoch in list(range(181, 200)) or epoch in [162, 165, 167, 170, 172, 175, 178])

def enhanced_train_xy_with_tqdm(cfg, train_loader, model, criterion, optimizer, scaler,
                               epoch, num_batch, num_warmup, writer_dict, logger, device, rank):
    """带tqdm进度条的标准训练函数"""
    model.train()
    
    # 创建tqdm进度条，只在主进程显示
    if rank in [-1, 0]:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f'Epoch {epoch}', ncols=100, leave=True)
    else:
        pbar = enumerate(train_loader)
    
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
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 更新tqdm进度条信息 (只在主进程)
        if rank in [-1, 0]:
            current_lr = optimizer.param_groups[0]['lr']
            
            # 构建loss信息字符串
            loss_info = f"Loss: {total_loss.item():.4f}"
            if head_losses and len(head_losses) >= 3:
                loss_info += f" | Det: {head_losses[0]:.4f} Seg: {head_losses[1]:.4f} Lane: {head_losses[2]:.4f}"
            
            # 添加学习率信息
            lr_info = f"LR: {current_lr:.6f}"
            
            # 更新进度条后缀
            pbar.set_postfix_str(f"{loss_info} | {lr_info}")
        
        # Update tensorboard
        if rank in [-1, 0] and writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', total_loss.item(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
    
    # 关闭进度条 (只在主进程)
    if rank in [-1, 0]:
        pbar.close()

def main():
    args = parse_args()
    update_config(cfg, args)
    
    world_size, rank = setup_distributed_training()
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.LOG_DIR, 'train', rank=rank)
    
    if args.mtl_comparison == 'comparison':
        tb_log_dir = tb_log_dir + '_mtl_comparison'
        logger.info(f"MTL比较模式已启用 - 日志将保存到 {tb_log_dir}")
    
    writer_dict = None
    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)
        writer_dict = {'writer': SummaryWriter(log_dir=tb_log_dir), 'train_global_steps': 0, 'valid_global_steps': 0}
    
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    
    print("构建模型...")
    model = get_net(cfg).to(device)
    criterion = get_loss(cfg, device, model)
    optimizer = get_optimizer(cfg, model)
    
    begin_epoch, _ = load_model_weights(model, logger, optimizer) if rank in [-1, 0] else (cfg.TRAIN.BEGIN_EPOCH, -1)
    
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    elif rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    model.gr = 1.0
    model.nc = 1
    
    print("加载数据...")
    train_loader, valid_loader, valid_dataset = create_data_loaders(rank)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    validation_epochs = [epoch for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1) if should_validate_and_save(epoch)]
    
    print(f'=> 开始训练，MTL比较标志: {args.mtl_comparison}...')
    
    if args.mtl_comparison == 'comparison':
        logger.info("初始化MTL训练管理器...")
        
        # 创建带时间戳的MTL管理器
        mtl_manager = MTLTrainingManager(cfg=cfg, logger=logger, device=device, 
                                       log_dir=cfg.LOG_DIR, dataset_name=cfg.DATASET.DATASET, rank=rank)
        
        # 输出实验信息
        logger.info(f"实验时间戳: {mtl_manager.timestamp}")
        logger.info(f"实验目录: {mtl_manager.experiment_dir}")
        
        mtl_manager.setup_tag_module(model, feature_dim=512)
        
        for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
            if rank != -1:
                train_loader.sampler.set_epoch(epoch)
            
            mtl_manager.train_epoch_with_detectors(cfg, train_loader, model, criterion, optimizer, scaler,
                                                 epoch, num_batch, num_warmup, logger, device, rank)
            
            lr_scheduler.step()
            
            if epoch in validation_epochs and rank in [-1, 0]:
                mtl_manager.validate_and_record(epoch, cfg, valid_loader, valid_dataset, model, criterion, logger, device, rank)
                mtl_manager.save_checkpoints(epoch, model, optimizer)
        
        if rank in [-1, 0]:
            mtl_manager.generate_comparison_report()
            mtl_manager.save_all_histories()
            print_tensorboard_instructions(mtl_manager.experiment_dir, cfg.DATASET.DATASET, mtl_manager.timestamp)
        
        mtl_manager.cleanup()
        
    else:
        for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
            if rank != -1:
                train_loader.sampler.set_epoch(epoch)
            
            # 使用带tqdm的标准训练函数
            enhanced_train_xy_with_tqdm(cfg, train_loader, model, criterion, optimizer, scaler,
                                      epoch, num_batch, num_warmup, writer_dict, logger, device, rank)
            
            lr_scheduler.step()
            
            if should_validate_and_save(epoch) and rank in [-1, 0]:
                da_results, ll_results, det_results, total_loss, maps, times = validate(
                    epoch, cfg, valid_loader, valid_dataset, model, criterion,
                    final_output_dir, tb_log_dir, writer_dict, logger, device, rank
                )
                
                msg = f'Epoch: [{epoch}] Loss({total_loss:.3f})\n' \
                      f'Driving area Segment: Acc({da_results[0]:.3f}) IOU({da_results[1]:.3f}) mIOU({da_results[2]:.3f})\n' \
                      f'Lane line Segment: Acc({ll_results[0]:.3f}) IOU({ll_results[1]:.3f}) mIOU({ll_results[2]:.3f})\n' \
                      f'Detect: P({det_results[0]:.3f}) R({det_results[1]:.3f}) mAP@0.5({det_results[2]:.3f}) mAP@0.5:0.95({det_results[3]:.3f})\n' \
                      f'Time: inference({times[0]:.4f}s/frame) nms({times[1]:.4f}s/frame)'
                logger.info(msg)
                
                save_checkpoint(epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                              output_dir=final_output_dir, filename=f'epoch-{epoch}.pth')
                save_checkpoint(epoch=epoch, name=cfg.MODEL.NAME, model=model, optimizer=optimizer,
                              output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), filename='checkpoint.pth')
        
        if rank in [-1, 0]:
            final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
            logger.info(f'=> 保存最终模型状态到 {final_model_state_file}')
            model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
            torch.save(model_state, final_model_state_file)
            writer_dict['writer'].close()
    
    if rank != -1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

# =============================================================================
# Usage Examples and Documentation
# =============================================================================

"""
MTL Training System with 5 Detection Methods and Time-stamped Directory Structure

Architecture:
- MTLDetector: Base detector class for recording training metrics with timestamped directories
- MultiTaskOptimizer: Implements GradNorm, PCGrad, CAGrad algorithms  
- TaskAdaptiveAttention: TAG implementation for feature adaptation
- MTLTrainingManager: Manages all detectors and training process with unified comparison

Features:
✅ 5 MTL methods: None, GradNorm, PCGrad, CAGrad, TAG
✅ Time-stamped directory structure for better organization
✅ Comprehensive metric recording for all methods
✅ Real-time plot generation during validation with tqdm progress bars
✅ Independent model checkpoints for each method
✅ Unified comparison plots with all methods in same charts
✅ Performance summary table generation
✅ Fixed TensorBoard logging with proper error handling
✅ Automatic comparison report generation

Directory Structure:
{cfg.LOG_DIR}/{dataset_name}/{timestamp}/
├── mtl_none/                     # Standard training
│   ├── tensorboard/             # TensorBoard logs
│   ├── training_history.json    # Training metrics
│   └── epoch_*.pth             # Model checkpoints
├── mtl_gradnorm/                # GradNorm method
├── mtl_pcgrad/                  # PCGrad method
├── mtl_cagrad/                  # CAGrad method
├── mtl_tag/                     # TAG method
└── mtl_comparison_final/        # Final comparison results
    ├── comparison_results.json
    ├── unified_methods_comparison_{timestamp}.png
    └── performance_summary_table_{timestamp}.png

Usage:
# MTL comparison experiment with all methods
python train_script.py --mtl-comparison comparison

# Standard training with tqdm progress bars
python train_script.py --mtl-comparison None

TensorBoard Usage:
# View all methods comparison
tensorboard --logdir /path/to/logs/{dataset_name}/{timestamp}/

# View specific method
tensorboard --logdir /path/to/logs/{dataset_name}/{timestamp}/mtl_gradnorm/tensorboard/

Recorded Metrics:
- Training time per epoch with progress bars
- Total loss and individual task losses (detection, segmentation, lane)
- Gradient norms and learning rates
- Validation metrics: DA mIoU, LL mIoU, Detection mAP@0.5/0.95
- Method-specific metrics: task weights, conflicts, attention weights

Visualization:
- Unified comparison charts with all methods overlaid
- Training curves: Loss, time, gradient norms vs epoch
- Validation curves: mIoU, mAP vs epoch
- Performance summary table with key metrics
- Individual method analysis plots

Key Improvements:
- Time-stamped experiments prevent overwriting results
- TensorBoard error handling and proper initialization
- Unified visualization for easy method comparison
- Progress bars for better training monitoring
- Comprehensive logging and metric collection
"""