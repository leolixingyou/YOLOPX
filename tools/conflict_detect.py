import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

class FixedGradientConflictDetector:
    """修正的梯度冲突检测器 - 完整版包含图表生成"""
    
    def __init__(self, model):
        self.model = model
        self.metrics_history = defaultdict(list)
        self.task_names = ['Detection', 'Driving_Area', 'Lane_Line']
        
    def detect_conflicts_in_context(self, model, head_losses, optimizer, scaler, prefix=''):
        """在训练上下文中检测冲突，避免干扰正常训练流程"""
        if len(head_losses) < 3:
            return {}
        
        try:
            det_loss = head_losses[0].item() if hasattr(head_losses[0], 'item') else head_losses[0]
            da_loss = head_losses[1].item() if hasattr(head_losses[1], 'item') else head_losses[1]
            ll_loss = head_losses[2].item() if hasattr(head_losses[2], 'item') else head_losses[2]
            
            losses_array = np.array([det_loss, da_loss, ll_loss])
            loss_mean = np.mean(losses_array)
            loss_std = np.std(losses_array)
            magnitude_conflict = loss_std / (loss_mean + 1e-8)
            
            max_loss = np.max(losses_array)
            min_loss = np.min(losses_array)
            directional_conflict = (max_loss - min_loss) / (max_loss + min_loss + 1e-8)
            
            task_conflict_intensity = magnitude_conflict * directional_conflict
            
            normalized_losses = losses_array / (np.sum(losses_array) + 1e-8)
            entropy = -np.sum(normalized_losses * np.log(normalized_losses + 1e-8))
            max_entropy = np.log(len(losses_array))
            gradient_conflict_rate = 1 - (entropy / max_entropy)
            
            metrics = {
                f'{prefix}task_conflict_intensity': task_conflict_intensity,
                f'{prefix}gradient_conflict_rate': gradient_conflict_rate,
                f'{prefix}directional_conflict': directional_conflict,
                f'{prefix}magnitude_conflict': magnitude_conflict,
                f'{prefix}det_da_loss_ratio': det_loss / (da_loss + 1e-8),
                f'{prefix}det_ll_loss_ratio': det_loss / (ll_loss + 1e-8),
                f'{prefix}da_ll_loss_ratio': da_loss / (ll_loss + 1e-8),
            }
            
            for key, value in metrics.items():
                if not prefix:
                    self.metrics_history[key].append(value)
            
            return metrics
            
        except Exception as e:
            print(f"冲突检测错误: {e}")
            return {}
    
    def generate_comprehensive_plots(self):
        """生成训练结束后的综合分析图表"""
        if not self.metrics_history:
            return {}
        
        plots = {}
        
        # 1. 冲突强度变化图
        if 'task_conflict_intensity' in self.metrics_history:
            fig1 = plt.figure(figsize=(10, 6))
            values = self.metrics_history['task_conflict_intensity']
            plt.plot(values, color='red', linewidth=2)
            plt.title('Task Conflict Intensity Over Training', fontweight='bold')
            plt.ylabel('Conflict Intensity')
            plt.xlabel('Training Step')
            plt.grid(True, alpha=0.3)
            plots['task_conflict_intensity'] = fig1
        
        # 2. 损失比率变化图
        if 'det_da_loss_ratio' in self.metrics_history:
            fig2 = plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics_history['det_da_loss_ratio'], label='Det/DA', linewidth=2)
            plt.title('Detection vs Driving Area Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics_history['det_ll_loss_ratio'], label='Det/LL', color='orange', linewidth=2)
            plt.title('Detection vs Lane Line Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.plot(self.metrics_history['da_ll_loss_ratio'], label='DA/LL', color='green', linewidth=2)
            plt.title('Driving Area vs Lane Line Loss Ratio')
            plt.ylabel('Loss Ratio')
            plt.xlabel('Training Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            if 'gradient_conflict_rate' in self.metrics_history:
                plt.plot(self.metrics_history['gradient_conflict_rate'], color='purple', linewidth=2)
                plt.title('Gradient Conflict Rate')
                plt.ylabel('Conflict Rate')
                plt.xlabel('Training Step')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['loss_ratios_analysis'] = fig2
        
        # 3. 综合冲突指标图
        if 'magnitude_conflict' in self.metrics_history and 'directional_conflict' in self.metrics_history:
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
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_final'] = values[-1]
                summary[f'{metric}_trend'] = 'improving' if values[-1] < values[0] else 'stable'
        
        return summary
    