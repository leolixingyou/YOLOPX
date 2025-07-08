import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GradientConflictDetector:
   def __init__(self, model):
       self.model = model
       self.similarity_history = []
       self.norm_history = []
       self.conflict_history = []
       
   def _is_shared_parameter(self, param_name):
       """判断是否为共享参数"""
       import re
       match = re.search(r'model\.(\d+)', param_name)
       if match:
           layer_num = int(match.group(1))
           return layer_num < 27
       return False
   
   def _compute_task_gradients(self, head_losses):
       """计算各任务在共享参数上的梯度 - 核心复用函数"""
       tensor_losses = [loss for loss in head_losses[:3] 
                       if isinstance(loss, torch.Tensor) and loss.requires_grad]
       
       if len(tensor_losses) < 2:
           return []
       
       # 收集共享参数名
       shared_param_names = [name for name, p in self.model.named_parameters() 
                            if self._is_shared_parameter(name)]
       
       grads = []
       for loss in tensor_losses:
           self.model.zero_grad()
           loss.backward(retain_graph=True)
           
           grad_list = []
           for name in shared_param_names:
               param = dict(self.model.named_parameters())[name]
               if param.grad is not None:
                   grad_list.append(param.grad.flatten())
               else:
                   grad_list.append(torch.zeros_like(param).flatten())
           
           if grad_list:
               grads.append(torch.cat(grad_list))
       
       return grads
   
   def detect_all_metrics(self, head_losses):
       """一次性计算所有梯度冲突指标"""
       grads = self._compute_task_gradients(head_losses)
       if len(grads) < 2:
           return {}
       
       n_tasks = len(grads)
       task_names = ['det', 'da_seg', 'll_seg']
       
       # 1. 计算相似度矩阵 (复用)
       similarity_matrix = torch.zeros(n_tasks, n_tasks)
       for i in range(n_tasks):
           for j in range(n_tasks):
               if i == j:
                   similarity_matrix[i, j] = 1.0
               else:
                   cos_sim = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0))
                   similarity_matrix[i, j] = cos_sim.item()
       
       # 2. 计算梯度范数
       norms = {f'{task_names[i]}_grad_norm': grads[i].norm().item() for i in range(n_tasks)}
       
       # 3. 梯度夹角 (Cosine Similarity)
       cosine_metrics = {}
       pairs = [('det_da', 0, 1), ('det_ll', 0, 2), ('da_ll', 1, 2)]
       for pair_name, i, j in pairs:
           if i < n_tasks and j < n_tasks:
               cosine_metrics[f'{pair_name}_cosine'] = similarity_matrix[i, j].item()
       
       # 4. 梯度冲突率 (GC)
       conflicts = [similarity_matrix[i, j].item() < 0 for i, j in [(0,1), (0,2), (1,2)] if i < n_tasks and j < n_tasks]
       gc_rate = sum(conflicts) / len(conflicts) if conflicts else 0
       
       # 5. 梯度冲突度 (GCD)
       gcd_metrics = {}
       for pair_name, i, j in pairs:
           if i < n_tasks and j < n_tasks:
               gcd_metrics[f'{pair_name}_gcd'] = 1 - similarity_matrix[i, j].item()
       
       # 6. 方向冲突 (Cdir)
       upper_triangle = [similarity_matrix[i, j].item() for i in range(n_tasks) for j in range(i+1, n_tasks)]
       cdir = np.mean(upper_triangle) if upper_triangle else 0
       
       # 7. 幅度冲突 (Cmg)
       norm_values = [norms[f'{task_names[i]}_grad_norm'] for i in range(n_tasks)]
       cmg = np.var(norm_values)
       
       # 8. 任务冲突强度 (TCI)
       tci = (1 - cdir) * cmg  # 综合方向与幅度
       
       # 合并所有指标
       all_metrics = {
           **cosine_metrics,
           **norms,
           **gcd_metrics,
           'gradient_conflict_rate': gc_rate,
           'directional_conflict': cdir,
           'magnitude_conflict': cmg,
           'task_conflict_intensity': tci,
           'max_min_norm_ratio': max(norm_values) / max(min(norm_values), 1e-8)
       }
       
       # 更新历史
       self.similarity_history.append(similarity_matrix.numpy())
       self.norm_history.append({k: v for k, v in norms.items()})
       self.conflict_history.append(all_metrics)
       
       return all_metrics
   
   def plot_similarity_heatmap(self):
       """绘制相似度热力图"""
       if not self.similarity_history:
           return None
       
       sim_matrix = self.similarity_history[-1]
       if sim_matrix.shape[0] < 2:
           return None
       
       plt.figure(figsize=(6, 5))
       sns.heatmap(sim_matrix, annot=True, cmap='RdBu_r', center=0,
                  xticklabels=['Detection', 'DA_Seg', 'Lane_Seg'][:sim_matrix.shape[0]],
                  yticklabels=['Detection', 'DA_Seg', 'Lane_Seg'][:sim_matrix.shape[1]],
                  vmin=-1, vmax=1, square=True)
       plt.title('Gradient Cosine Similarity')
       plt.tight_layout()
       return plt.gcf()
   
   def plot_gradient_norms(self):
       """绘制梯度范数对比图"""
       if not self.norm_history:
           return None
       
       latest_norms = self.norm_history[-1]
       plt.figure(figsize=(8, 6))
       
       tasks = list(latest_norms.keys())
       values = list(latest_norms.values())
       colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
       
       bars = plt.bar(tasks, values, color=colors[:len(tasks)])
       plt.title('Gradient Norms by Task')
       plt.ylabel('Gradient Norm')
       plt.xticks(rotation=45)
       
       for bar, value in zip(bars, values):
           plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
       
       plt.tight_layout()
       return plt.gcf()



