import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch # Added for checking tensor types

class FixedGradientConflictDetector:
    """修正的梯度冲突检测器 - 完整版包含图表生成"""
    
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.task_names = ['Detection', 'Driving_Area', 'Lane_Line']
        
    def detect_conflicts_in_context(self, model, head_losses, optimizer, scaler, prefix=''):
        """在训练上下文中检测冲突，避免干扰正常训练流程"""
        # 确保 head_losses 至少有3个元素，以避免索引错误
        if len(head_losses) < 3:
            # 如果少于3个任务，无法计算多任务冲突指标
            return {}
        
        try:
            # 确保损失是标量，并转换为Python浮点数或numpy数组
            # 兼容 head_losses 可能是列表中的None (如果某个任务没有输出)
            valid_losses = [l.item() if torch.is_tensor(l) else l for l in head_losses if l is not None]
            
            # 如果有效损失少于3个，则不进行冲突计算
            if len(valid_losses) < 3:
                return {}

            losses_array = np.array(valid_losses)
            
            # 避免除以零和log(0)
            loss_sum = np.sum(losses_array)
            if loss_sum == 0: # 所有损失都为0，视为无冲突
                return {}

            loss_mean = np.mean(losses_array)
            loss_std = np.std(losses_array)
            
            # 避免除以零
            magnitude_conflict = loss_std / (loss_mean + 1e-8)
            
            max_loss = np.max(losses_array)
            min_loss = np.min(losses_array)
            # 避免除以零
            directional_conflict = (max_loss - min_loss) / (max_loss + min_loss + 1e-8)
            
            task_conflict_intensity = magnitude_conflict * directional_conflict
            
            normalized_losses = losses_array / (loss_sum + 1e-8)
            # 确保对数输入为正
            entropy = -np.sum(normalized_losses * np.log(normalized_losses + 1e-8))
            max_entropy = np.log(len(losses_array))
            gradient_conflict_rate = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
            metrics = {
                f'{prefix}task_conflict_intensity': task_conflict_intensity,
                f'{prefix}gradient_conflict_rate': gradient_conflict_rate,
                f'{prefix}directional_conflict': directional_conflict,
                f'{prefix}magnitude_conflict': magnitude_conflict,
            }
            
            # 避免除以零，添加损失比率，仅当分母非零时计算
            # 确保 head_losses_raw 至少有3个元素
            if len(head_losses) >= 3:
                det_loss_val = head_losses[0].item() if head_losses[0] is not None and torch.is_tensor(head_losses[0]) else 0.0
                da_loss_val = head_losses[1].item() if head_losses[1] is not None and torch.is_tensor(head_losses[1]) else 0.0
                ll_loss_val = head_losses[2].item() if head_losses[2] is not None and torch.is_tensor(head_losses[2]) else 0.0

                if da_loss_val + 1e-8 > 0:
                    metrics[f'{prefix}det_da_loss_ratio'] = det_loss_val / (da_loss_val + 1e-8)
                if ll_loss_val + 1e-8 > 0:
                    metrics[f'{prefix}det_ll_loss_ratio'] = det_loss_val / (ll_loss_val + 1e-8)
                if ll_loss_val + 1e-8 > 0: # This means DA/LL will be calculated if LL is valid
                    metrics[f'{prefix}da_ll_loss_ratio'] = da_loss_val / (ll_loss_val + 1e-8)
            
            for key, value in metrics.items():
                if not prefix: # 只记录未加前缀的原始冲突指标
                    self.metrics_history[key].append(value)
            
            return metrics
            
        except Exception as e:
            print(f"冲突检测错误: {e}")
            return {}
    
    def generate_comprehensive_plots(self):
        """生成训练结束后的综合分析图表"""
        if not self.metrics_history:
            print("No metrics history to generate plots.")
            return {}
        
        plots = {}
        
        # 1. 冲突强度变化图
        if 'task_conflict_intensity' in self.metrics_history and self.metrics_history['task_conflict_intensity']:
            fig1 = plt.figure(figsize=(10, 6))
            values = self.metrics_history['task_conflict_intensity']
            plt.plot(values, color='red', linewidth=2)
            plt.title('Task Conflict Intensity Over Training', fontweight='bold')
            plt.ylabel('Conflict Intensity')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            plots['task_conflict_intensity'] = fig1
        
        # 2. 损失比率变化图
        # 检查至少一个相关的历史记录存在且非空
        if any(key in self.metrics_history and self.metrics_history[key] for key in ['det_da_loss_ratio', 'det_ll_loss_ratio', 'da_ll_loss_ratio', 'gradient_conflict_rate']):
            fig2 = plt.figure(figsize=(12, 8))
            
            if 'det_da_loss_ratio' in self.metrics_history and self.metrics_history['det_da_loss_ratio']:
                plt.subplot(2, 2, 1)
                plt.plot(self.metrics_history['det_da_loss_ratio'], label='Det/DA', linewidth=2)
                plt.title('Detection vs Driving Area Loss Ratio')
                plt.ylabel('Loss Ratio')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if 'det_ll_loss_ratio' in self.metrics_history and self.metrics_history['det_ll_loss_ratio']:
                plt.subplot(2, 2, 2)
                plt.plot(self.metrics_history['det_ll_loss_ratio'], label='Det/LL', color='orange', linewidth=2)
                plt.title('Detection vs Lane Line Loss Ratio')
                plt.ylabel('Loss Ratio')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if 'da_ll_loss_ratio' in self.metrics_history and self.metrics_history['da_ll_loss_ratio']:
                plt.subplot(2, 2, 3)
                plt.plot(self.metrics_history['da_ll_loss_ratio'], label='DA/LL', color='green', linewidth=2)
                plt.title('Driving Area vs Lane Line Loss Ratio')
                plt.ylabel('Loss Ratio')
                plt.xlabel('Training Step')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if 'gradient_conflict_rate' in self.metrics_history and self.metrics_history['gradient_conflict_rate']:
                plt.subplot(2, 2, 4)
                plt.plot(self.metrics_history['gradient_conflict_rate'], color='purple', linewidth=2)
                plt.title('Gradient Conflict Rate')
                plt.ylabel('Conflict Rate')
                plt.xlabel('Training Step')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['loss_ratios_analysis'] = fig2
        
        # 3. 综合冲突指标图
        if ('magnitude_conflict' in self.metrics_history and self.metrics_history['magnitude_conflict'] and
            'directional_conflict' in self.metrics_history and self.metrics_history['directional_conflict']):
            fig3 = plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics_history['magnitude_conflict'], color='blue', linewidth=2)
            plt.title('Magnitude Conflict')
            plt.ylabel('Magnitude Conflict')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.metrics_history['directional_conflict'], color='green', linewidth=2)
            plt.title('Directional Conflict')
            plt.ylabel('Directional Conflict')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['conflict_metrics'] = fig3
        
        return plots
    
    def get_training_summary(self):
        """获取训练总结统计"""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for metric, values in self.metrics_history.items():
            if values: # 确保列表非空
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_final'] = values[-1]
                # 趋势判断：如果最终值小于初始值，认为在改善 (对于损失和冲突指标，越低越好)
                summary[f'{metric}_trend'] = 'improving' if values[-1] < values[0] else 'stable'
            else:
                summary[f'{metric}_mean'] = float('nan')
                summary[f'{metric}_final'] = float('nan')
                summary[f'{metric}_trend'] = 'no_data'
        
        return summary