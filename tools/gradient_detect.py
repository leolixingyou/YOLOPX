import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

class GradientConflictDetector:
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.task_names = ['Detection', 'Driving_Area', 'Lane_Line']
        
    def _is_shared_parameter(self, param_name):
        """判断是否为共享参数"""
        import re
        match = re.search(r'model\.(\d+)', param_name)
        return match and int(match.group(1)) < 27
    
    def _compute_task_gradients(self, head_losses):
        """计算各任务在共享参数上的梯度"""
        tensor_losses = [loss for loss in head_losses[:3] 
                        if isinstance(loss, torch.Tensor) and loss.requires_grad]
        
        if len(tensor_losses) < 2:
            return []
        
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
        """计算所有梯度冲突指标并累积历史"""
        grads = self._compute_task_gradients(head_losses)
        if len(grads) < 2:
            return {}
        
        n_tasks = len(grads)
        
        # 1. 相似度矩阵
        similarity_matrix = torch.zeros(n_tasks, n_tasks)
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    cos_sim = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0))
                    similarity_matrix[i, j] = cos_sim.item()
        
        # 2. 梯度范数
        norms = [grads[i].norm().item() for i in range(n_tasks)]
        
        # 3. 任务对相似度
        pairs = [('det_da', 0, 1), ('det_ll', 0, 2), ('da_ll', 1, 2)]
        cosine_metrics = {}
        for pair_name, i, j in pairs:
            if i < n_tasks and j < n_tasks:
                cosine_metrics[f'{pair_name}_cosine'] = similarity_matrix[i, j].item()
        
        # 4. 综合指标
        upper_triangle = [similarity_matrix[i, j].item() for i in range(n_tasks) for j in range(i+1, n_tasks)]
        cdir = np.mean(upper_triangle) if upper_triangle else 0
        cmg = np.var(norms)
        tci = (1 - cdir) * cmg
        gc_rate = sum(1 for sim in upper_triangle if sim < 0) / len(upper_triangle) if upper_triangle else 0
        
        # 存储历史数据
        metrics = {
            'similarity_matrix': similarity_matrix.numpy(),
            'norms': norms,
            'directional_conflict': cdir,
            'magnitude_conflict': cmg,
            'task_conflict_intensity': tci,
            'gradient_conflict_rate': gc_rate,
            **cosine_metrics
        }
        
        for key, value in metrics.items():
            if key != 'similarity_matrix':
                self.metrics_history[key].append(value)
        
        self.metrics_history['similarity_matrix'].append(similarity_matrix.numpy())
        
        return metrics
    
    def generate_comprehensive_plots(self):
        """生成训练结束后的综合分析图表"""
        if not self.metrics_history:
            return {}
        
        plots = {}
        total_steps = len(self.metrics_history['similarity_matrix'])
        
        # 1. 梯度相似度混淆矩阵风格图
        avg_sim_matrix = np.mean(self.metrics_history['similarity_matrix'], axis=0)
        fig1 = plt.figure(figsize=(8, 6))
        
        sns.heatmap(avg_sim_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=self.task_names[:avg_sim_matrix.shape[0]],
                   yticklabels=self.task_names[:avg_sim_matrix.shape[1]],
                   vmin=-1, vmax=1, square=True, fmt='.3f', 
                   cbar_kws={'label': 'Cosine Similarity'},
                   linewidths=1, linecolor='black')
        
        plt.title(f'Multi-task Gradient Conflict Matrix (Avg over {total_steps} steps)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots['gradient_conflict_matrix'] = fig1
        
        # 2. 梯度范数变化趋势
        if 'norms' in self.metrics_history:
            fig2 = plt.figure(figsize=(12, 8))
            
            # 2.1 范数变化曲线
            plt.subplot(2, 2, 1)
            norms_array = np.array(self.metrics_history['norms'])
            for i, task in enumerate(self.task_names[:norms_array.shape[1]]):
                plt.plot(norms_array[:, i], label=task, linewidth=2)
            plt.title('Gradient Norm Evolution During Training', fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Gradient Norm')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2.2 范数分布箱线图
            plt.subplot(2, 2, 2)
            plt.boxplot([norms_array[:, i] for i in range(norms_array.shape[1])], 
                       labels=self.task_names[:norms_array.shape[1]])
            plt.title('Gradient Norm Distribution', fontweight='bold')
            plt.ylabel('Gradient Norm')
            plt.xticks(rotation=45)
            
            # 2.3 相对范数变化
            plt.subplot(2, 2, 3)
            relative_norms = norms_array / (norms_array.max(axis=1, keepdims=True) + 1e-8)
            for i, task in enumerate(self.task_names[:relative_norms.shape[1]]):
                plt.plot(relative_norms[:, i], label=task, linewidth=2)
            plt.title('Relative Gradient Norm (Normalized)', fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Relative Norm')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2.4 范数比率
            plt.subplot(2, 2, 4)
            max_norms = norms_array.max(axis=1)
            min_norms = norms_array.min(axis=1)
            ratio = max_norms / (min_norms + 1e-8)
            plt.plot(ratio, color='red', linewidth=2)
            plt.title('Max/Min Gradient Norm Ratio', fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['gradient_norms_analysis'] = fig2
        
        # 2. 分离的冲突分析图表
        # 2.1 方向冲突
        fig2 = plt.figure(figsize=(10, 6))
        if 'directional_conflict' in self.metrics_history:
            values = self.metrics_history['directional_conflict']
            plt.plot(values, color='blue', linewidth=2)
            plt.axhspan(0.3, 1.0, alpha=0.2, color='green', label='Healthy (>0.3)')
            plt.axhspan(0.0, 0.3, alpha=0.2, color='yellow', label='Moderate (0-0.3)')
            plt.axhspan(-1.0, 0.0, alpha=0.2, color='red', label='Conflicting (<0)')
            
            final_value = values[-1] if values else 0
            plt.title(f'Directional Conflict Evolution - Final: {final_value:.3f}', fontweight='bold')
            plt.ylabel('Average Cosine Similarity')
            plt.xlabel('Training Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['directional_conflict'] = fig2
        
        # 2.2 幅度冲突
        fig3 = plt.figure(figsize=(10, 6))
        if 'magnitude_conflict' in self.metrics_history:
            values = self.metrics_history['magnitude_conflict']
            plt.plot(values, color='green', linewidth=2)
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            stable_threshold = mean_val + std_val
            plt.axhline(y=stable_threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Instability Threshold: {stable_threshold:.3f}')
            
            final_value = values[-1] if values else 0
            status = "Stable" if final_value < stable_threshold else "Unstable"
            plt.title(f'Magnitude Conflict - Final: {final_value:.3f} ({status})', fontweight='bold')
            plt.ylabel('Gradient Norm Variance')
            plt.xlabel('Training Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['magnitude_conflict'] = fig3
        
        # 2.3 任务冲突强度
        fig4 = plt.figure(figsize=(10, 6))
        if 'task_conflict_intensity' in self.metrics_history:
            values = self.metrics_history['task_conflict_intensity']
            plt.plot(values, color='red', linewidth=2)
            
            final_value = values[-1] if values else 0
            if final_value < 0.1:
                status_color = 'green'
            elif final_value < 0.3:
                status_color = 'orange'
            else:
                status_color = 'red'
                
            plt.title(f'Task Conflict Intensity - Final: {final_value:.3f}', 
                     fontweight='bold', color=status_color)
            plt.ylabel('Conflict Intensity (Lower=Better)')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['task_conflict_intensity'] = fig4
        
        # 2.4 冲突率（修正解释）
        fig5 = plt.figure(figsize=(10, 6))
        if 'gradient_conflict_rate' in self.metrics_history:
            values = self.metrics_history['gradient_conflict_rate']
            plt.plot(values, color='purple', linewidth=2)
            
            final_value = values[-1] if values else 0
            plt.title(f'Gradient Conflict Rate - Final: {final_value:.1%}\n(Proportion of task pairs with negative similarity)', 
                     fontweight='bold')
            plt.ylabel('Conflict Rate (0=No Conflicts, 1=All Conflicting)')
            plt.xlabel('Training Step')
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Conflict (>50%)')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['gradient_conflict_rate'] = fig5
        
        # 2.5 训练健康度评分（添加图例）
        fig6 = plt.figure(figsize=(10, 6))
        health_scores = {}
        if 'directional_conflict' in self.metrics_history:
            dc_final = self.metrics_history['directional_conflict'][-1]
            health_scores['Direction'] = max(0, min(1, (dc_final + 1) / 2))
        
        if 'magnitude_conflict' in self.metrics_history:
            mc_values = self.metrics_history['magnitude_conflict']
            mc_final = mc_values[-1]
            mc_threshold = np.mean(mc_values) + np.std(mc_values)
            health_scores['Magnitude'] = max(0, min(1, 1 - mc_final / max(mc_threshold, 1e-8)))
        
        if 'gradient_conflict_rate' in self.metrics_history:
            gcr_final = self.metrics_history['gradient_conflict_rate'][-1]
            health_scores['Conflict Rate'] = 1 - gcr_final
        
        if health_scores:
            labels = list(health_scores.keys())
            values = list(health_scores.values())
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
            
            bars = plt.bar(labels, values, color=colors, alpha=0.7)
            plt.ylim(0, 1)
            plt.title('Multi-task Training Health Score', fontweight='bold')
            plt.ylabel('Health Score (0-1)')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', alpha=0.7, label='Excellent (>0.7)'),
                             Patch(facecolor='orange', alpha=0.7, label='Good (0.4-0.7)'),
                             Patch(facecolor='red', alpha=0.7, label='Poor (<0.4)')]
            plt.legend(handles=legend_elements, loc='upper right')
            
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plots['training_health_score'] = fig6
        

        
        # 4. 训练阶段分析
        if len(self.metrics_history.get('directional_conflict', [])) > 10:
            fig4 = plt.figure(figsize=(12, 8))
            
            total_steps = len(self.metrics_history['directional_conflict'])
            early_phase = total_steps // 3
            mid_phase = total_steps * 2 // 3
            
            phases = ['Early (0-33%)', 'Mid (33-66%)', 'Late (66-100%)']
            phase_data = {
                'directional_conflict': [
                    np.mean(self.metrics_history['directional_conflict'][:early_phase]),
                    np.mean(self.metrics_history['directional_conflict'][early_phase:mid_phase]),
                    np.mean(self.metrics_history['directional_conflict'][mid_phase:])
                ],
                'magnitude_conflict': [
                    np.mean(self.metrics_history['magnitude_conflict'][:early_phase]),
                    np.mean(self.metrics_history['magnitude_conflict'][early_phase:mid_phase]),
                    np.mean(self.metrics_history['magnitude_conflict'][mid_phase:])
                ] if 'magnitude_conflict' in self.metrics_history else [0, 0, 0]
            }
            
            plt.subplot(1, 2, 1)
            x = np.arange(len(phases))
            width = 0.35
            plt.bar(x - width/2, phase_data['directional_conflict'], width, label='Directional Conflict', alpha=0.8)
            plt.bar(x + width/2, phase_data['magnitude_conflict'], width, label='Magnitude Conflict', alpha=0.8)
            plt.xlabel('Training Phase')
            plt.ylabel('Conflict Value')
            plt.title('Multi-task Conflict Evolution by Training Phase', fontweight='bold')
            plt.xticks(x, phases)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 滑动窗口分析
            plt.subplot(1, 2, 2)
            window_size = max(10, total_steps // 20)
            if 'task_conflict_intensity' in self.metrics_history:
                tci_data = self.metrics_history['task_conflict_intensity']
                smoothed_tci = np.convolve(tci_data, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_tci, label='Smoothed TCI', linewidth=2)
            
            plt.title(f'Smoothed Task Conflict Intensity\n(Window Size: {window_size})', fontweight='bold')
            plt.xlabel('Training Step')
            plt.ylabel('Conflict Intensity')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['training_phase_analysis'] = fig4
        
        return plots
    
    def get_training_summary(self):
        """获取训练总结统计 - 改进版本，提供有意义的解释"""
        if not self.metrics_history:
            return {}
        
        summary = {}
        
        # 数值统计
        for metric, values in self.metrics_history.items():
            if metric != 'similarity_matrix' and values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_final'] = values[-1]
                summary[f'{metric}_trend'] = 'improving' if values[-1] > values[0] else 'declining'
        
        # 添加解释性总结
        interpretations = {}
        
        # 方向冲突解释
        if 'directional_conflict' in self.metrics_history:
            dc_final = self.metrics_history['directional_conflict'][-1]
            if dc_final > 0.3:
                interpretations['directional_status'] = 'Excellent - Tasks are cooperating well'
            elif dc_final > 0:
                interpretations['directional_status'] = 'Good - Moderate task cooperation'
            else:
                interpretations['directional_status'] = 'Poor - Tasks are conflicting'
        
        # 幅度冲突解释
        if 'magnitude_conflict' in self.metrics_history:
            mc_values = self.metrics_history['magnitude_conflict']
            mc_final = mc_values[-1]
            mc_threshold = np.mean(mc_values) + np.std(mc_values)
            if mc_final < mc_threshold:
                interpretations['magnitude_status'] = 'Stable - Balanced gradient magnitudes'
            else:
                interpretations['magnitude_status'] = 'Unstable - Imbalanced gradient magnitudes'
        
        # 冲突率解释
        if 'gradient_conflict_rate' in self.metrics_history:
            gcr_final = self.metrics_history['gradient_conflict_rate'][-1]
            if gcr_final < 0.2:
                interpretations['conflict_rate_status'] = 'Excellent - Low conflict rate'
            elif gcr_final < 0.5:
                interpretations['conflict_rate_status'] = 'Acceptable - Moderate conflict rate'
            else:
                interpretations['conflict_rate_status'] = 'High - Significant task conflicts'
        
        # 任务对相似度解释
        pair_interpretations = {}
        pairs = [('det_da_cosine', 'Detection-DrivingArea'), 
                ('det_ll_cosine', 'Detection-LaneLine'), 
                ('da_ll_cosine', 'DrivingArea-LaneLine')]
        
        for pair_key, pair_name in pairs:
            if pair_key in self.metrics_history:
                final_sim = self.metrics_history[pair_key][-1]
                if final_sim > 0.3:
                    pair_interpretations[pair_name] = f'Strong cooperation ({final_sim:.3f})'
                elif final_sim > 0:
                    pair_interpretations[pair_name] = f'Mild cooperation ({final_sim:.3f})'
                elif final_sim > -0.1:
                    pair_interpretations[pair_name] = f'Independent ({final_sim:.3f})'
                else:
                    pair_interpretations[pair_name] = f'Conflicting ({final_sim:.3f})'
        
        summary['interpretations'] = interpretations
        summary['task_pair_relationships'] = pair_interpretations
        
        return summary