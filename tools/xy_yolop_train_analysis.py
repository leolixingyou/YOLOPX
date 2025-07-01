#!/usr/bin/env python3
"""
Cross-Method MTL Comparator
跨方法多任务学习比较器

用于比较多个独立的MTL方法实验结果，生成详细的分析报告
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

class CrossMethodComparator:
    def __init__(self, log_dir, dataset_name):
        self.log_dir = Path(log_dir)
        self.dataset_name = dataset_name
        self.base_dir = self.log_dir / dataset_name
        self.methods = ['none', 'gradnorm', 'pcgrad', 'cagrad', 'tag']
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def find_related_experiments(self, pattern=None):
        """查找相关的实验组"""
        if not self.base_dir.exists():
            print(f"Warning: Base directory {self.base_dir} does not exist")
            return {}
        
        experiment_groups = defaultdict(dict)
        
        # 扫描所有实验目录
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 检查是否包含MTL方法目录
            mtl_dirs = [d for d in exp_dir.iterdir() if d.name.startswith('mtl_')]
            if not mtl_dirs:
                continue
            
            # 从实验名称中提取基础名称和方法
            exp_name = exp_dir.name
            base_name, method = self._parse_experiment_name(exp_name)
            
            if base_name and method:
                experiment_groups[base_name][method] = exp_dir
        
        return dict(experiment_groups)
    
    def _parse_experiment_name(self, exp_name):
        """解析实验名称，提取基础名称和方法"""
        # 模式1: mtl_exp_20241227_1122_gradnorm
        pattern1 = r'^(.+)_([a-z]+)$'
        match1 = re.match(pattern1, exp_name)
        if match1:
            base_name, method = match1.groups()
            if method in self.methods:
                return base_name, method
        
        # 模式2: gradnorm_20241227_1122
        pattern2 = r'^([a-z]+)_(.+)$'
        match2 = re.match(pattern2, exp_name)
        if match2:
            method, base_name = match2.groups()
            if method in self.methods:
                return base_name, method
        
        # 模式3: 包含方法名的任意位置
        for method in self.methods:
            if method in exp_name:
                base_name = exp_name.replace(f'_{method}', '').replace(f'{method}_', '')
                return base_name, method
        
        return None, None
    
    def load_experiment_results(self, experiment_dir, method):
        """加载单个实验的结果"""
        # 查找MTL目录
        mtl_dir = experiment_dir / f'mtl_{method}'
        if not mtl_dir.exists():
            print(f"Warning: MTL directory {mtl_dir} not found")
            return None
        
        history_file = mtl_dir / 'training_history.json'
        if not history_file.exists():
            print(f"Warning: Training history file {history_file} not found")
            return None
        
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
                
            # 验证数据完整性
            required_keys = ['epochs', 'train_losses', 'validation_results']
            for key in required_keys:
                if key not in data:
                    print(f"Warning: Missing key '{key}' in {history_file}")
                    return None
            
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {history_file}: {e}")
            return None
    
    def generate_comparison_report(self, experiment_group_name=None, output_dir=None):
        """生成跨方法比较报告"""
        experiment_groups = self.find_related_experiments()
        
        if not experiment_groups:
            print("未找到任何实验组")
            return
        
        if experiment_group_name:
            if experiment_group_name not in experiment_groups:
                print(f"未找到实验组: {experiment_group_name}")
                print(f"可用实验组: {list(experiment_groups.keys())}")
                return
            groups_to_process = {experiment_group_name: experiment_groups[experiment_group_name]}
        else:
            # 选择最新的实验组
            latest_group = max(experiment_groups.keys(), key=lambda x: max(
                exp_dir.stat().st_mtime for exp_dir in experiment_groups[x].values()
            ))
            groups_to_process = {latest_group: experiment_groups[latest_group]}
            print(f"使用最新实验组: {latest_group}")
        
        for group_name, experiments in groups_to_process.items():
            self._process_experiment_group(group_name, experiments, output_dir)
    
    def _process_experiment_group(self, group_name, experiments, output_dir):
        """处理单个实验组"""
        print(f"\n处理实验组: {group_name}")
        print(f"包含方法: {list(experiments.keys())}")
        
        # 加载所有方法的结果
        results = {}
        for method, exp_dir in experiments.items():
            data = self.load_experiment_results(exp_dir, method)
            if data:
                results[method] = data
                print(f"  ✓ 加载 {method} 数据")
            else:
                print(f"  ✗ 无法加载 {method} 数据")
        
        if not results:
            print("无可用数据，跳过此实验组")
            return
        
        if len(results) < 2:
            print("需要至少2个方法的结果才能进行比较")
            return
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.base_dir / f'comparison_{group_name}'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 生成报告
        try:
            self._generate_performance_summary(results, output_dir, group_name)
            self._generate_comparison_plots(results, output_dir, group_name)
            self._generate_detailed_report(results, output_dir, group_name)
            self._generate_method_specific_analysis(results, output_dir, group_name)
            
            print(f"✓ 比较报告已生成到: {output_dir}")
        except Exception as e:
            print(f"✗ 生成报告时出错: {e}")
    
    def _generate_performance_summary(self, results, output_dir, group_name):
        """生成性能汇总"""
        summary = {}
        
        for method, data in results.items():
            if data.get('validation_results'):
                latest_val = data['validation_results'][-1]
                avg_train_time = np.mean(data['train_times']) if data.get('train_times') else 0
                
                summary[method] = {
                    'final_da_miou': latest_val.get('da_miou', 0),
                    'final_ll_miou': latest_val.get('ll_miou', 0),
                    'final_det_map': latest_val.get('det_map_05', 0),
                    'avg_train_time': avg_train_time,
                    'total_epochs': len(data.get('epochs', [])),
                    'final_train_loss': data['train_losses'][-1] if data.get('train_losses') else 0
                }
        
        # 保存JSON
        with open(output_dir / 'performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 生成表格图
        self._plot_performance_table(summary, output_dir, group_name)
        
        # 生成性能排名
        self._generate_performance_ranking(summary, output_dir)
    
    def _plot_performance_table(self, summary, output_dir, group_name):
        """绘制性能汇总表格"""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        methods = list(summary.keys())
        headers = ['Method', 'DA mIoU', 'LL mIoU', 'Det mAP@0.5', 'Train Loss', 'Avg Time (s)', 'Epochs']
        
        table_data = []
        for method in methods:
            data = summary[method]
            table_data.append([
                method.upper(),
                f"{data['final_da_miou']:.4f}",
                f"{data['final_ll_miou']:.4f}",
                f"{data['final_det_map']:.4f}",
                f"{data['final_train_loss']:.4f}",
                f"{data['avg_train_time']:.2f}",
                f"{data['total_epochs']}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.0)
        
        # 设置样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 交替行颜色
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(f'MTL Methods Performance Summary - {group_name}', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.savefig(output_dir / 'performance_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_plots(self, results, output_dir, group_name):
        """生成对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f'MTL Methods Comparison - {group_name}', fontsize=20)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'd', 'v']
        
        method_styles = {}
        for idx, method in enumerate(results.keys()):
            method_styles[method] = {
                'color': colors[idx % len(colors)],
                'linestyle': line_styles[idx % len(line_styles)],
                'marker': markers[idx % len(markers)],
                'markersize': 4,
                'markevery': max(1, len(results[method].get('epochs', [])) // 10)
            }
        
        # 绘制各项指标
        for method, data in results.items():
            style = method_styles[method]
            
            # 训练损失
            if data.get('epochs') and data.get('train_losses'):
                axes[0, 0].plot(data['epochs'], data['train_losses'],
                               label=method.upper(), linewidth=2, **style)
            
            # 训练时间
            if data.get('train_times'):
                epochs_subset = data['epochs'][:len(data['train_times'])]
                axes[0, 1].plot(epochs_subset, data['train_times'],
                               label=method.upper(), linewidth=2, **style)
            
            # 梯度范数
            if data.get('grad_norms'):
                epochs_subset = data['epochs'][:len(data['grad_norms'])]
                axes[0, 2].plot(epochs_subset, data['grad_norms'],
                               label=method.upper(), linewidth=2, **style)
            
            # 验证指标
            if data.get('validation_results'):
                val_epochs = [r['epoch'] for r in data['validation_results']]
                da_mious = [r.get('da_miou', 0) for r in data['validation_results']]
                ll_mious = [r.get('ll_miou', 0) for r in data['validation_results']]
                det_maps = [r.get('det_map_05', 0) for r in data['validation_results']]
                
                axes[1, 0].plot(val_epochs, da_mious, label=method.upper(), linewidth=2, **style)
                axes[1, 1].plot(val_epochs, ll_mious, label=method.upper(), linewidth=2, **style)
                axes[1, 2].plot(val_epochs, det_maps, label=method.upper(), linewidth=2, **style)
        
        # 设置图表属性
        titles = ['Training Loss', 'Training Time per Epoch', 'Gradient Norm',
                 'Driving Area mIoU', 'Lane Line mIoU', 'Detection mAP@0.5']
        y_labels = ['Loss', 'Time (s)', 'Grad Norm', 'mIoU', 'mIoU', 'mAP@0.5']
        
        for i, ax in enumerate(axes.flat):
            ax.set_title(titles[i], fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(y_labels[i], fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_ranking(self, summary, output_dir):
        """生成性能排名"""
        metrics = ['final_da_miou', 'final_ll_miou', 'final_det_map', 'avg_train_time']
        metric_names = ['DA mIoU', 'LL mIoU', 'Detection mAP@0.5', 'Training Time']
        
        rankings = {}
        
        for metric, name in zip(metrics, metric_names):
            # 对于时间，越小越好；对于其他指标，越大越好
            reverse = metric != 'avg_train_time'
            
            sorted_methods = sorted(summary.items(), 
                                  key=lambda x: x[1][metric], 
                                  reverse=reverse)
            
            rankings[name] = [(method.upper(), data[metric]) for method, data in sorted_methods]
        
        # 保存排名
        with open(output_dir / 'performance_ranking.json', 'w') as f:
            json.dump(rankings, f, indent=2)
        
        # 生成排名可视化
        self._plot_rankings(rankings, output_dir)
    
    def _plot_rankings(self, rankings, output_dir):
        """绘制排名图"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Performance Rankings by Metric', fontsize=18)
        
        axes = axes.flatten()
        colors = ['#4CAF50', '#FF9800', '#F44336', '#2196F3', '#9C27B0']
        
        for idx, (metric_name, ranking) in enumerate(rankings.items()):
            ax = axes[idx]
            
            methods = [item[0] for item in ranking]
            values = [item[1] for item in ranking]
            
            bars = ax.bar(methods, values, color=colors[:len(methods)])
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(metric_name, fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.tick_params(labelsize=11)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_method_specific_analysis(self, results, output_dir, group_name):
        """生成方法特定分析"""
        method_analysis_dir = output_dir / 'method_analysis'
        method_analysis_dir.mkdir(exist_ok=True)
        
        for method, data in results.items():
            if method in ['gradnorm', 'pcgrad', 'cagrad', 'tag'] and data.get('method_specific'):
                self._plot_method_specific(method, data, method_analysis_dir)
    
    def _plot_method_specific(self, method, data, output_dir):
        """绘制方法特定图表"""
        method_data = data.get('method_specific', {})
        epochs = data.get('epochs', [])
        
        if method == 'gradnorm' and method_data.get('task_weights'):
            fig, ax = plt.subplots(figsize=(12, 8))
            task_weights = np.array(method_data['task_weights'])
            
            if len(task_weights.shape) == 1:
                # 单个权重序列
                ax.plot(epochs[:len(task_weights)], task_weights, 
                       label='Task Weight', linewidth=2, marker='o')
            else:
                # 多个任务权重
                for i in range(task_weights.shape[1]):
                    ax.plot(epochs[:len(task_weights)], task_weights[:, i], 
                           label=f'Task {i+1} Weight', linewidth=2, marker='o')
            
            ax.set_title('GradNorm Task Weights Evolution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Weight', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{method}_task_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        elif method == 'pcgrad' and method_data.get('conflict_counts'):
            fig, ax = plt.subplots(figsize=(12, 8))
            conflicts = method_data['conflict_counts']
            
            ax.plot(epochs[:len(conflicts)], conflicts, linewidth=2, color='red', marker='s')
            ax.set_title('PCGrad Gradient Conflicts', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Number of Conflicts', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{method}_conflicts.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        elif method == 'cagrad' and method_data.get('alpha_weights'):
            fig, ax = plt.subplots(figsize=(12, 8))
            alpha_weights = np.array(method_data['alpha_weights'])
            
            if len(alpha_weights.shape) == 1:
                ax.plot(epochs[:len(alpha_weights)], alpha_weights,
                       label='Alpha Weight', linewidth=2, marker='^')
            else:
                for i in range(alpha_weights.shape[1]):
                    ax.plot(epochs[:len(alpha_weights)], alpha_weights[:, i],
                           label=f'Task {i+1} Alpha', linewidth=2, marker='^')
            
            ax.set_title('CAGrad Alpha Weights Evolution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Alpha', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{method}_alpha_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        elif method == 'tag' and method_data.get('task_groups'):
            fig, ax = plt.subplots(figsize=(10, 6))
            task_groups = method_data['task_groups']
            
            # 可视化任务分组
            group_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            y_pos = 0
            for i, group in enumerate(task_groups):
                color = group_colors[i % len(group_colors)]
                for task_idx in group:
                    ax.barh(y_pos, 1, left=task_idx, height=0.8, 
                           color=color, alpha=0.7, edgecolor='black')
                y_pos += 1
            
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, len(task_groups) - 0.5)
            ax.set_xlabel('Task Index', fontsize=14)
            ax.set_ylabel('Group', fontsize=14)
            ax.set_title('TAG Task Grouping', fontsize=16, fontweight='bold')
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{method}_task_groups.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_detailed_report(self, results, output_dir, group_name):
        """生成详细Markdown报告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_content = f"""# MTL Methods Cross-Comparison Report

## Experiment Group: {group_name}
**Generated on:** {timestamp}
**Number of methods compared:** {len(results)}

## Executive Summary

This report compares the performance of different multi-task learning (MTL) conflict resolution methods on the {self.dataset_name} dataset.

## Methods Compared

"""
        
        method_descriptions = {
            'none': 'Standard multi-task learning without conflict resolution',
            'gradnorm': 'GradNorm - Dynamic task weight balancing based on gradient magnitudes',
            'pcgrad': 'PCGrad - Projects conflicting gradients to eliminate negative interference',
            'cagrad': 'CAGrad - Conflict-averse gradient descent for multi-task optimization',
            'tag': 'Task Affinity Grouping - Groups similar tasks based on gradient similarity'
        }
        
        for method in sorted(results.keys()):
            description = method_descriptions.get(method, 'No description available')
            report_content += f"- **{method.upper()}**: {description}\n"
        
        report_content += "\n## Performance Summary\n\n"
        
        # 性能汇总表格
        report_content += "| Method | DA mIoU | LL mIoU | Det mAP@0.5 | Train Loss | Avg Time (s) | Epochs |\n"
        report_content += "|--------|---------|---------|-------------|------------|--------------|--------|\n"
        
        for method in sorted(results.keys()):
            data = results[method]
            if data.get('validation_results'):
                latest_val = data['validation_results'][-1]
                avg_train_time = np.mean(data['train_times']) if data.get('train_times') else 0
                final_loss = data['train_losses'][-1] if data.get('train_losses') else 0
                
                report_content += f"| {method.upper()} | {latest_val.get('da_miou', 0):.4f} | {latest_val.get('ll_miou', 0):.4f} | {latest_val.get('det_map_05', 0):.4f} | {final_loss:.4f} | {avg_train_time:.2f} | {len(data.get('epochs', []))} |\n"
        
        # 最佳性能分析
        report_content += "\n## Best Performing Methods\n\n"
        
        metrics = [('da_miou', 'Driving Area mIoU'), ('ll_miou', 'Lane Line mIoU'), ('det_map_05', 'Detection mAP@0.5')]
        
        for metric_key, metric_name in metrics:
            best_method = None
            best_value = -1
            
            for method, data in results.items():
                if data.get('validation_results'):
                    latest_val = data['validation_results'][-1]
                    value = latest_val.get(metric_key, 0)
                    if value > best_value:
                        best_value = value
                        best_method = method
            
            if best_method:
                report_content += f"- **{metric_name}**: {best_method.upper()} ({best_value:.4f})\n"
        
        # 训练效率分析
        report_content += "\n## Training Efficiency Analysis\n\n"
        
        if len(results) > 1:
            # 找出训练时间最短的方法
            fastest_method = None
            fastest_time = float('inf')
            
            for method, data in results.items():
                if data.get('train_times'):
                    avg_time = np.mean(data['train_times'])
                    if avg_time < fastest_time:
                        fastest_time = avg_time
                        fastest_method = method
            
            if fastest_method:
                report_content += f"- **Fastest Training**: {fastest_method.upper()} ({fastest_time:.2f}s per epoch)\n"
            
            # 收敛性分析
            report_content += "\n### Convergence Analysis\n\n"
            for method, data in results.items():
                if data.get('train_losses') and len(data['train_losses']) > 10:
                    initial_loss = np.mean(data['train_losses'][:5])
                    final_loss = np.mean(data['train_losses'][-5:])
                    reduction = (initial_loss - final_loss) / initial_loss * 100
                    
                    report_content += f"- **{method.upper()}**: Loss reduction of {reduction:.1f}% (from {initial_loss:.4f} to {final_loss:.4f})\n"
        
        # 方法特定见解
        report_content += "\n## Method-Specific Insights\n\n"
        
        for method, data in results.items():
            method_data = data.get('method_specific', {})
            
            if method == 'gradnorm' and method_data.get('task_weights'):
                weights = method_data['task_weights']
                if weights:
                    final_weights = weights[-1] if isinstance(weights[-1], list) else [weights[-1]]
                    report_content += f"### GradNorm\n- Final task weights: {[f'{w:.3f}' for w in final_weights]}\n- Shows dynamic task balancing throughout training\n\n"
            
            elif method == 'pcgrad' and method_data.get('conflict_counts'):
                conflicts = method_data['conflict_counts']
                if conflicts:
                    avg_conflicts = np.mean(conflicts) if isinstance(conflicts, list) else conflicts
                    report_content += f"### PCGrad\n- Average conflicts per epoch: {avg_conflicts:.2f}\n- Effectively reduces gradient interference\n\n"
            
            elif method == 'cagrad' and method_data.get('alpha_weights'):
                alpha_weights = method_data['alpha_weights']
                if alpha_weights:
                    final_alphas = alpha_weights[-1] if isinstance(alpha_weights[-1], list) else [alpha_weights[-1]]
                    report_content += f"### CAGrad\n- Final alpha weights: {[f'{a:.3f}' for a in final_alphas]}\n- Adaptive conflict-averse optimization\n\n"
            
            elif method == 'tag' and method_data.get('task_groups'):
                groups = method_data['task_groups']
                if groups:
                    report_content += f"### TAG\n- Task groups: {groups}\n- Groups tasks based on gradient similarity\n\n"
        
        # 推荐建议
        report_content += "\n## Recommendations\n\n"
        
        # 找出总体最佳方法
        overall_scores = {}
        for method, data in results.items():
            if data.get('validation_results'):
                latest_val = data['validation_results'][-1]
                score = (latest_val.get('da_miou', 0) + 
                        latest_val.get('ll_miou', 0) + 
                        latest_val.get('det_map_05', 0)) / 3
                overall_scores[method] = score
        
        if overall_scores:
            best_overall = max(overall_scores.items(), key=lambda x: x[1])
            report_content += f"- **Overall Best Method**: {best_overall[0].upper()} (average score: {best_overall[1]:.4f})\n"
            
            # 给出使用建议
            if best_overall[0] == 'none':
                report_content += "- The standard MTL approach performs well, suggesting tasks are naturally compatible\n"
            elif best_overall[0] == 'gradnorm':
                report_content += "- GradNorm's dynamic weighting is effective for this task combination\n"
            elif best_overall[0] == 'pcgrad':
                report_content += "- PCGrad successfully handles gradient conflicts in this setting\n"
            elif best_overall[0] == 'cagrad':
                report_content += "- CAGrad's conflict-averse approach provides optimal results\n"
            elif best_overall[0] == 'tag':
                report_content += "- Task grouping strategy proves beneficial for this task configuration\n"
        
        report_content += f"\n## Generated Files\n\n"
        report_content += "- `performance_table.png` - Performance summary table\n"
        report_content += "- `methods_comparison.png` - Training curves comparison\n"
        report_content += "- `performance_rankings.png` - Performance rankings by metric\n"
        report_content += "- `performance_summary.json` - Raw performance data\n"
        report_content += "- `performance_ranking.json` - Ranking data\n"
        report_content += "- `method_analysis/` - Method-specific analysis plots\n"
        
        report_content += f"\n## Data Sources\n\n"
        for method in sorted(results.keys()):
            report_content += f"- **{method.upper()}**: `{group_name}_{method}/mtl_{method}/training_history.json`\n"
        
        report_content += f"\n---\n*Report generated by CrossMethodComparator v1.0*"
        
        # 保存报告
        with open(output_dir / 'comparison_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def list_experiment_groups(self):
        """列出所有实验组"""
        experiment_groups = self.find_related_experiments()
        
        if not experiment_groups:
            print("未找到任何实验组")
            return
        
        print("可用实验组:")
        print("=" * 60)
        
        for group_name, experiments in experiment_groups.items():
            methods = list(experiments.keys())
            latest_time = max(exp_dir.stat().st_mtime for exp_dir in experiments.values())
            latest_datetime = datetime.fromtimestamp(latest_time)
            
            print(f"\n📁 {group_name}")
            print(f"   方法: {', '.join(sorted(methods))} ({len(methods)}/{len(self.methods)})")
            print(f"   最后修改: {latest_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 检查数据完整性
            complete_methods = 0
            for method, exp_dir in experiments.items():
                mtl_dir = exp_dir / f'mtl_{method}'
                history_file = mtl_dir / 'training_history.json'
                if history_file.exists():
                    complete_methods += 1
            
            print(f"   数据完整: {complete_methods}/{len(methods)} 方法")
            
            if complete_methods >= 2:
                print(f"   状态: ✅ 可进行比较分析")
            else:
                print(f"   状态: ⚠️  数据不足，需要至少2个完整方法")
        
        print("\n" + "=" * 60)
        print(f"总计: {len(experiment_groups)} 个实验组")


def main():
    parser = argparse.ArgumentParser(description='跨方法MTL比较器')
    parser.add_argument('--log-dir', default='./runs', help='日志目录')
    parser.add_argument('--dataset', default='BDD100K', help='数据集名称')
    parser.add_argument('--action', choices=['list', 'compare', 'auto'], 
                       default='auto', help='操作类型')
    parser.add_argument('--group', help='实验组名称（可选，默认使用最新的）')
    parser.add_argument('--output', help='输出目录（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    comparator = CrossMethodComparator(args.log_dir, args.dataset)
    
    if args.action == 'list':
        comparator.list_experiment_groups()
    
    elif args.action == 'compare':
        if args.verbose:
            print(f"比较器配置:")
            print(f"  日志目录: {args.log_dir}")
            print(f"  数据集: {args.dataset}")
            print(f"  实验组: {args.group or '(最新)'}")
            print(f"  输出目录: {args.output or '(自动)'}")
            print()
        
        comparator.generate_comparison_report(args.group, args.output)
    
    elif args.action == 'auto':
        # 自动模式：先列出可用组，然后生成最新组的报告
        print("=" * 60)
        print("MTL跨方法比较器 - 自动模式")
        print("=" * 60)
        
        print("\n🔍 扫描可用实验组...")
        comparator.list_experiment_groups()
        
        print("\n📊 生成比较报告...")
        comparator.generate_comparison_report(args.group, args.output)
        
        print("\n✅ 自动分析完成!")


if __name__ == '__main__':
    main()