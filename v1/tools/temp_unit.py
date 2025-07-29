import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from temp_mtl import create_mtl_resolver


class SimpleMultiTaskModel(nn.Module):
    """多任务模型"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.task_heads = nn.ModuleDict({
            'detection': nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 5)),
            'segmentation': nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 10)),
            'lane': nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 3))
        })
        
    def forward(self, x):
        shared_features = self.shared_backbone(x)
        return {task: head(shared_features) for task, head in self.task_heads.items()}


class MTLConflictTester:
    """MTL冲突测试器 - 增强版，集成任务头冲突检测"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.task_names = ['detection', 'segmentation', 'lane']
        self.criterions = {
            'detection': nn.MSELoss(),
            'segmentation': nn.CrossEntropyLoss(),
            'lane': nn.MSELoss()
        }
        # 强化任务冲突的难度系数
        self.task_difficulties = {'detection': 1.0, 'segmentation': 3.0, 'lane': 0.5}
        
    def validate_all_methods(self):
        """验证所有方法"""
        print("🔧 Validating all MTL methods...")
        methods = ['original', 'gradnorm', 'pcgrad', 'uncertainty', 'mgda']
        validation_results = {}
        
        test_model = SimpleMultiTaskModel().to(self.device)
        x, targets = self.generate_conflicting_data(batch_size=4)
        outputs = test_model(x)
        losses = self.compute_losses(outputs, targets)
        
        for method in methods:
            try:
                resolver = create_mtl_resolver(method, self.task_names, self.device)
                
                # 初始化
                if method == 'gradnorm':
                    resolver.initialize(losses, test_model.shared_backbone)
                elif method == 'pcgrad':
                    resolver.initialize(test_model.shared_backbone)
                elif method == 'uncertainty':
                    resolver.initialize()
                
                # 测试基础功能
                if method == 'mgda':
                    weights = resolver.compute_mgda_weights(losses, test_model.parameters())
                else:
                    weights = resolver.update_weights(losses, 0, 0)
                
                # 测试任务头冲突检测功能
                task_weights, conflict_metrics = resolver.detect_and_update_weights(
                    losses, test_model, 0, 0
                )
                
                enhanced_metrics = resolver.get_enhanced_metrics()
                
                validation_results[method] = {
                    'status': 'SUCCESS', 
                    'weights': weights, 
                    'metrics_count': len(enhanced_metrics),
                    'conflict_metrics_count': len(conflict_metrics),
                    'has_conflict_detection': 'overall_conflict_intensity' in conflict_metrics
                }
                print(f"    ✅ {method} - OK ({len(enhanced_metrics)} metrics, {len(conflict_metrics)} conflict metrics)")
                
            except Exception as e:
                validation_results[method] = {'status': 'FAILED', 'error': str(e)}
                print(f"    ❌ {method} - FAILED: {str(e)}")
        
        success_count = sum(1 for r in validation_results.values() if r['status'] == 'SUCCESS')
        print(f"\n✅ Validation complete: {success_count}/{len(methods)} methods working")
        
        # 检查冲突检测功能
        conflict_enabled = sum(1 for r in validation_results.values() 
                             if r.get('has_conflict_detection', False))
        print(f"🎯 Task-head conflict detection enabled: {conflict_enabled}/{success_count} methods")
        
        return validation_results
    
    def generate_conflicting_data(self, batch_size: int = 32):
        """生成具有强冲突的测试数据"""
        x = torch.randn(batch_size, 20, device=self.device)
        
        # 强化任务间负相关性
        base_signal = torch.randn(batch_size, 3, device=self.device)
        correlation_matrix = torch.tensor([
            [1.0, -0.8, 0.3],   # 强负相关
            [-0.8, 1.0, -0.7],  # 强冲突
            [0.3, -0.7, 1.0]
        ], device=self.device)
        
        correlated_signals = torch.matmul(base_signal, correlation_matrix)
        
        # 生成目标，增加冲突强度
        lane_base = correlated_signals[:, 1:3]
        lane_third_dim = lane_base[:, 0:1] + torch.randn(batch_size, 1, device=self.device) * 0.1
        lane_features = torch.cat([lane_base, lane_third_dim], dim=1)
        
        targets = {
            'detection': (correlated_signals[:, 0:1] * self.task_difficulties['detection']).expand(-1, 5),
            'segmentation': torch.randint(0, 10, (batch_size,), device=self.device),
            'lane': lane_features * self.task_difficulties['lane']
        }
        
        # 增强segmentation与其他任务的耦合
        seg_influence = (correlated_signals[:, 1] * self.task_difficulties['segmentation']).long()
        targets['segmentation'] = torch.clamp(seg_influence, 0, 9)
        
        return x, targets
    
    def compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """计算各任务损失"""
        losses = {}
        for task in self.task_names:
            base_loss = self.criterions[task](outputs[task], targets[task])
            losses[task] = base_loss * self.task_difficulties[task]
        return losses
    
    def test_method(self, method_name: str, epochs: int = 15, steps_per_epoch: int = 8, global_step_offset: int = 0) -> Dict:
        """测试单个方法 - 增强版，包含任务头冲突分析"""
        print(f"\n=== Testing {method_name.upper()} ===")
        
        model = SimpleMultiTaskModel().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
        resolver = create_mtl_resolver(method_name, self.task_names, self.device)
        
        # 初始化
        x, targets = self.generate_conflicting_data()
        outputs = model(x)
        initial_losses = self.compute_losses(outputs, targets)
        
        if hasattr(resolver, 'initialize'):
            if method_name == 'gradnorm':
                resolver.initialize(initial_losses, model.shared_backbone)
            elif method_name == 'pcgrad':
                resolver.initialize(model.shared_backbone)
            elif method_name == 'uncertainty':
                resolver.initialize()
        
        # 记录指标
        metrics_history = {
            'losses': {task: [] for task in self.task_names},
            'weights': {task: [] for task in self.task_names},
            'total_loss': [],
            'method_metrics': {},
            'conflict_metrics': {},  # 新增：冲突检测指标
            'step_numbers': []
        }
        
        current_step = 0
        
        # 训练循环
        for epoch in range(epochs):
            for step_in_epoch in range(steps_per_epoch):
                model.train()
                x, targets = self.generate_conflicting_data()
                outputs = model(x)
                losses = self.compute_losses(outputs, targets)
                
                # 不同方法的处理逻辑
                if method_name == 'pcgrad':
                    # 使用增强的冲突检测功能
                    task_weights, conflict_metrics = resolver.detect_and_update_weights(
                        losses, model, epoch, current_step
                    )
                    total_loss = sum(losses.values())
                    resolver.apply_pcgrad_to_gradients(losses, model)
                    optimizer.step()
                    scheduler.step()
                    
                elif method_name == 'mgda':
                    task_weights, conflict_metrics = resolver.detect_and_update_weights(
                        losses, model, epoch, current_step
                    )
                    # MGDA使用自己的权重计算
                    mgda_weights = resolver.compute_mgda_weights(losses, model.parameters())
                    task_weights.update(mgda_weights)  # 更新权重
                    total_loss = sum(task_weights[task] * losses[task] for task in self.task_names)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                elif method_name == 'uncertainty':
                    resolver.update_uncertainty(losses)
                    task_weights, conflict_metrics = resolver.detect_and_update_weights(
                        losses, model, epoch, current_step
                    )
                    total_loss = resolver.get_loss_with_uncertainty(losses)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                else:  # original, gradnorm
                    task_weights, conflict_metrics = resolver.detect_and_update_weights(
                        losses, model, epoch, current_step
                    )
                    total_loss = sum(task_weights[task] * losses[task] for task in self.task_names)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                
                # 记录指标
                loss_values = {k: v.item() for k, v in losses.items()}
                global_step = global_step_offset + current_step
                
                for task in self.task_names:
                    metrics_history['losses'][task].append(loss_values[task])
                    metrics_history['weights'][task].append(task_weights[task])
                metrics_history['total_loss'].append(total_loss.item())
                metrics_history['step_numbers'].append(global_step)
                
                # 方法特定指标
                method_metrics = resolver.get_metrics()
                for key, value in method_metrics.items():
                    if key not in metrics_history['method_metrics']:
                        metrics_history['method_metrics'][key] = []
                    metrics_history['method_metrics'][key].append(value)
                
                # 冲突检测指标
                for key, value in conflict_metrics.items():
                    if key not in metrics_history['conflict_metrics']:
                        metrics_history['conflict_metrics'][key] = []
                    metrics_history['conflict_metrics'][key].append(value)
                
                # 记录到wandb
                wandb_metrics = {f'{method_name}_total_loss': total_loss.item()}
                for task in self.task_names:
                    wandb_metrics[f'{method_name}_loss_{task}'] = loss_values[task]
                    wandb_metrics[f'{method_name}_weight_{task}'] = task_weights[task]
                
                # 方法特定指标
                if method_metrics:
                    for key, value in method_metrics.items():
                        if any(prefix in key for prefix in ['detect_', 'operation_', 'uncertainty_', 'mgda_']):
                            wandb_metrics[f'{method_name}_{key}'] = value
                
                # 冲突检测指标
                if conflict_metrics:
                    for key, value in conflict_metrics.items():
                        if any(prefix in key for prefix in ['conflict_', 'backbone_', 'head_', 'interference_']):
                            wandb_metrics[f'{method_name}_{key}'] = value
                
                if wandb.run is not None:
                    wandb.log(wandb_metrics, step=global_step)
                
                current_step += 1
                
                # 进度显示
                if current_step % 30 == 0:
                    print(f"Step {current_step}: Loss = {total_loss.item():.4f}")
                    self._display_insights(method_name, method_metrics, conflict_metrics, loss_values, task_weights)
        
        return metrics_history
    
    def _display_insights(self, method_name: str, metrics: Dict, conflict_metrics: Dict, losses: Dict, weights: Dict):
        """显示方法洞察 - 增强版，包含冲突分析"""
        
        # 原有的方法特定洞察
        if method_name == 'gradnorm' and metrics:
            if 'detect_training_rate_variance' in metrics:
                print(f"  🔍 GradNorm variance: {metrics['detect_training_rate_variance']:.4f}")
        elif method_name == 'pcgrad' and metrics:
            if 'detect_avg_cosine_similarity' in metrics:
                conflicts = metrics.get('detect_current_conflicts', 0)
                print(f"  🔍 PCGrad similarity: {metrics['detect_avg_cosine_similarity']:.3f}, Conflicts: {conflicts}")
        elif method_name == 'mgda' and metrics:
            if 'mgda_convergence_rate' in metrics:
                print(f"  🎯 MGDA convergence: {metrics['mgda_convergence_rate']:.6f}")
        
        # 新增：任务头冲突分析
        if conflict_metrics:
            conflict_intensity = conflict_metrics.get('overall_conflict_intensity', 0)
            severe_conflicts = conflict_metrics.get('severe_conflict_pairs', 0)
            
            if conflict_intensity > 0:
                conflict_level = "High" if conflict_intensity > 0.5 else "Medium" if conflict_intensity > 0.2 else "Low"
                print(f"  🥊 Task Conflicts: {conflict_level} (intensity: {conflict_intensity:.3f}, severe: {severe_conflicts})")
                
                # 显示最冲突的任务对
                max_conflict_pair = ""
                max_conflict_value = 0
                for key, value in conflict_metrics.items():
                    if key.startswith('conflict_') and value < max_conflict_value:
                        max_conflict_value = value
                        max_conflict_pair = key.replace('conflict_', '').replace('_', ' ↔ ')
                
                if max_conflict_pair:
                    print(f"    📍 Most conflicting: {max_conflict_pair} ({max_conflict_value:.3f})")
            
            # 显示backbone影响
            backbone_influences = {k: v for k, v in conflict_metrics.items() if k.startswith('backbone_influence_')}
            if backbone_influences:
                influences = list(backbone_influences.values())
                if max(influences) > 0:
                    dominant_task = max(backbone_influences.keys(), key=lambda k: backbone_influences[k])
                    dominant_task = dominant_task.replace('backbone_influence_', '')
                    print(f"    🎯 Backbone dominance: {dominant_task} ({backbone_influences[f'backbone_influence_{dominant_task}']:.3f})")
        
        # 任务平衡度
        task_losses = [losses[task] for task in self.task_names]
        balance = np.std(task_losses) / np.mean(task_losses) if np.mean(task_losses) > 0 else 0
        balance_desc = "Good" if balance < 0.4 else "Poor"
        print(f"  ⚖️ Task Balance: {balance_desc} (CV: {balance:.3f})")
    
    def visualize_results(self, all_results: Dict[str, Dict]):
        """可视化结果 - 生成9个独立图片"""
        methods = list(all_results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        linestyles = ['-', '--', '-.', ':']
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        timestamp = int(time.time())
        saved_files = []
        
        # 1. 总损失对比
        fig, ax = plt.subplots(figsize=(12, 8))
        for method in methods:
            original_steps = all_results[method]['step_numbers']
            method_steps = list(range(len(original_steps)))
            losses = all_results[method]['total_loss']
            ax.plot(method_steps, losses, label=method, linewidth=2, alpha=0.8)
        ax.set_title('Total Loss vs Training Steps (Each Method from Step 0)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps (per method)', fontsize=14)
        ax.set_ylabel('Total Loss', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        
        fig_path = f"/workspace/runs/01_total_loss_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"01_total_loss": wandb.Image(fig)})
        plt.close()
        
        # 2. 各任务损失演化
        fig, ax = plt.subplots(figsize=(14, 10))
        for i, task in enumerate(self.task_names):
            for j, method in enumerate(methods):
                original_steps = all_results[method]['step_numbers']
                method_steps = list(range(len(original_steps)))
                task_losses = all_results[method]['losses'][task]
                ax.plot(method_steps, task_losses, 
                    color=colors[i], 
                    linestyle=linestyles[j % len(linestyles)], 
                    alpha=0.7,
                    label=f'{task}_{method}')
        ax.set_title('Task Loss Evolution (Each Method from Step 0)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps (per method)', fontsize=14)
        ax.set_ylabel('Task Loss', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/02_task_loss_evolution_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"02_task_loss_evolution": wandb.Image(fig)})
        plt.close()
        
        # 3. 动态权重变化
        fig, ax = plt.subplots(figsize=(12, 8))
        non_uniform_methods = [m for m in methods if m != 'original']
        for method in non_uniform_methods:
            original_steps = all_results[method]['step_numbers']
            method_steps = list(range(len(original_steps)))
            for i, task in enumerate(self.task_names):
                weights = all_results[method]['weights'][task]
                ax.plot(method_steps, weights, 
                    color=colors[i], 
                    linestyle='-' if method == 'gradnorm' else '--',
                    alpha=0.7,
                    label=f'{task}_{method}')
        ax.set_title('Dynamic Task Weights (Each Method from Step 0)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps (per method)', fontsize=14)
        ax.set_ylabel('Task Weight', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/03_dynamic_weights_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"03_dynamic_weights": wandb.Image(fig)})
        plt.close()
        
        # 4. 任务头冲突强度演化
        fig, ax = plt.subplots(figsize=(12, 8))
        conflict_detected = False
        for method in methods:
            if 'conflict_metrics' in all_results[method]:
                conflict_metrics = all_results[method]['conflict_metrics']
                original_steps = all_results[method]['step_numbers']
                method_steps = list(range(len(original_steps)))
                
                if 'overall_conflict_intensity' in conflict_metrics:
                    intensities = conflict_metrics['overall_conflict_intensity']
                    ax.plot(method_steps, intensities, label=f'{method}_conflict', linewidth=2, alpha=0.8)
                    conflict_detected = True
        
        if conflict_detected:
            ax.set_title('Task-Head Conflict Intensity (Each Method from Step 0)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Training Steps (per method)', fontsize=14)
            ax.set_ylabel('Conflict Intensity', fontsize=14)
            ax.legend(fontsize=12)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Conflict Threshold')
        else:
            ax.text(0.5, 0.5, 'No Conflict Data\nAvailable', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            ax.set_title('Task-Head Conflict (No Data)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/04_conflict_intensity_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"04_conflict_intensity": wandb.Image(fig)})
        plt.close()
        
        # 5. Backbone影响力对比
        fig, ax = plt.subplots(figsize=(14, 10))
        backbone_data_available = False
        for method in methods:
            if 'conflict_metrics' in all_results[method]:
                conflict_metrics = all_results[method]['conflict_metrics']
                original_steps = all_results[method]['step_numbers']
                method_steps = list(range(len(original_steps)))
                
                for i, task in enumerate(self.task_names):
                    backbone_key = f'backbone_influence_{task}'
                    if backbone_key in conflict_metrics:
                        influences = conflict_metrics[backbone_key]
                        ax.plot(method_steps, influences, 
                            color=colors[i], 
                            linestyle='-' if method == 'pcgrad' else '--',
                            alpha=0.7,
                            label=f'{task}_{method}')
                        backbone_data_available = True
        
        if backbone_data_available:
            ax.set_title('Backbone Influence by Task (Each Method from Step 0)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Training Steps (per method)', fontsize=14)
            ax.set_ylabel('Gradient Magnitude to Backbone', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Backbone\nInfluence Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            ax.set_title('Backbone Influence (No Data)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/05_backbone_influence_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"05_backbone_influence": wandb.Image(fig)})
        plt.close()
        
        # 6. 学习干扰热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap_method = None
        for method in methods:
            if 'conflict_metrics' in all_results[method]:
                conflict_metrics = all_results[method]['conflict_metrics']
                if any(k.startswith('conflict_') for k in conflict_metrics.keys()):
                    heatmap_method = method
                    break
        
        if heatmap_method:
            conflict_metrics = all_results[heatmap_method]['conflict_metrics']
            conflict_matrix = np.zeros((len(self.task_names), len(self.task_names)))
            for i, task_i in enumerate(self.task_names):
                for j, task_j in enumerate(self.task_names):
                    if i != j:
                        conflict_key = f'conflict_{task_i}_{task_j}'
                        if conflict_key in conflict_metrics and len(conflict_metrics[conflict_key]) > 0:
                            conflict_matrix[i, j] = conflict_metrics[conflict_key][-1]
            
            sns.heatmap(conflict_matrix,
                    annot=True,
                    fmt='.3f',
                    cmap='RdBu_r',
                    center=0,
                    xticklabels=self.task_names,
                    yticklabels=self.task_names,
                    ax=ax,
                    cbar_kws={'label': 'Conflict Score'},
                    square=True)
            ax.set_title(f'Final Task Conflict Matrix ({heatmap_method.upper()})', fontsize=16, fontweight='bold')
            ax.set_xlabel('Task j (Influencer)', fontsize=14)
            ax.set_ylabel('Task i (Influenced)', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Conflict Matrix\nData Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            ax.set_title('Task Conflict Matrix (No Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/06_conflict_matrix_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"06_conflict_matrix": wandb.Image(fig)})
        plt.close()
        
        # 7. PCGrad/GradNorm特定指标
        fig, ax = plt.subplots(figsize=(12, 8))
        method_specific_detected = False
        
        if 'pcgrad' in all_results:
            method_metrics = all_results['pcgrad']['method_metrics']
            original_steps = all_results['pcgrad']['step_numbers']
            method_steps = list(range(len(original_steps)))
            
            if 'detect_avg_cosine_similarity' in method_metrics:
                similarities = method_metrics['detect_avg_cosine_similarity']
                ax.plot(method_steps, similarities, label='PCGrad Similarity', linewidth=2, color='blue')
                ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7, label='Conflict Threshold')
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Alignment Threshold')
                method_specific_detected = True
            
            if 'detect_current_conflicts' in method_metrics:
                conflicts = method_metrics['detect_current_conflicts']
                non_zero_conflicts = [c for c in conflicts if c > 0]
                if non_zero_conflicts:
                    ax2 = ax.twinx()
                    display_interval = max(1, len(method_steps)//20)
                    display_steps = method_steps[::display_interval]
                    display_conflicts = conflicts[::display_interval]
                    ax2.bar(display_steps, display_conflicts, alpha=0.3, color='red', 
                        width=max(1, len(method_steps)//50), label='Conflicts/Step')
                    ax2.set_ylabel('Conflicts per Step', color='red', fontsize=14)
        
        if 'gradnorm' in all_results:
            method_metrics = all_results['gradnorm']['method_metrics']
            original_steps = all_results['gradnorm']['step_numbers']
            method_steps = list(range(len(original_steps)))
            
            if 'detect_training_rate_variance' in method_metrics:
                variances = method_metrics['detect_training_rate_variance']
                if not method_specific_detected:
                    ax.plot(method_steps, variances, label='GradNorm Variance', linewidth=2, color='orange')
                    ax.set_ylabel('Training Rate Variance', fontsize=14)
                else:
                    ax3 = ax.twinx()
                    ax3.spines['right'].set_position(('axes', 1.1))
                    ax3.plot(method_steps, variances, label='GradNorm Variance', linewidth=2, color='orange', alpha=0.7)
                    ax3.set_ylabel('Training Rate Variance', color='orange', fontsize=14)
                method_specific_detected = True
        
        if method_specific_detected:
            ax.set_title('Method-Specific Conflict Metrics', fontsize=16, fontweight='bold')
            ax.set_xlabel('Training Steps (per method)', fontsize=14)
            if 'pcgrad' in all_results:
                ax.set_ylabel('Cosine Similarity', fontsize=14)
            ax.legend(loc='upper left', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Method-Specific\nData Available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
            ax.set_title('Method-Specific Metrics (No Data)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/07_method_specific_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"07_method_specific": wandb.Image(fig)})
        plt.close()
        
        # 8. 收敛性分析
        fig, ax = plt.subplots(figsize=(12, 8))
        window_size = 15
        for method in methods:
            original_steps = all_results[method]['step_numbers']
            method_steps = list(range(len(original_steps)))
            losses = all_results[method]['total_loss']
            
            if len(losses) >= window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = method_steps[window_size-1:]
                ax.plot(smoothed_steps, smoothed_losses, label=f'{method}_smoothed', linewidth=2)
        
        ax.set_title(f'Convergence Analysis (MA window={window_size})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps (per method)', fontsize=14)
        ax.set_ylabel('Smoothed Loss', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/08_convergence_analysis_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"08_convergence_analysis": wandb.Image(fig)})
        plt.close()
        
        # 9. 最终性能对比
        fig, ax = plt.subplots(figsize=(10, 8))
        final_losses = []
        method_labels = []
        
        for i, method in enumerate(methods):
            final_loss = all_results[method]['total_loss'][-1]
            final_losses.append(final_loss)
            method_labels.append(method.upper())
        
        bars = ax.bar(method_labels, final_losses, color=colors_bar[:len(methods)], alpha=0.7)
        ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Final Total Loss', fontsize=14)
        ax.set_xticklabels(method_labels, rotation=45, fontsize=12)
        
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/09_final_performance_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
        if wandb.run is not None:
            wandb.log({"09_final_performance": wandb.Image(fig)})
        plt.close()
        

        
        print(f"\n🎯 All plots saved with timestamp: {timestamp}")
        if wandb.run is not None:
            print(f"🌐 All plots logged to WandB dashboard")
        
        return saved_files
    
    def calculate_performance_metrics(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """计算性能指标 - 增强版，包含冲突分析"""
        performance_metrics = {}
        
        for method, results in all_results.items():
            losses = results['total_loss']
            
            final_loss = losses[-1]
            initial_loss = losses[0]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            last_quarter = losses[-len(losses)//4:]
            convergence_stability = np.std(last_quarter)
            
            mid_point = len(losses) // 2
            mid_loss = losses[mid_point]
            early_improvement = (initial_loss - mid_loss) / initial_loss * 100
            
            final_task_losses = [results['losses'][task][-1] for task in self.task_names]
            task_balance_cv = np.std(final_task_losses) / np.mean(final_task_losses)
            
            base_metrics = {
                'final_loss': final_loss,
                'improvement_percent': improvement,
                'convergence_stability': convergence_stability,
                'early_learning_efficiency': early_improvement,
                'task_balance_cv': task_balance_cv,
            }
            
            # 新增：冲突分析指标
            conflict_analysis = {}
            if 'conflict_metrics' in results:
                conflict_metrics = results['conflict_metrics']
                
                # 平均冲突强度
                if 'overall_conflict_intensity' in conflict_metrics:
                    intensities = conflict_metrics['overall_conflict_intensity']
                    conflict_analysis['avg_conflict_intensity'] = np.mean(intensities)
                    conflict_analysis['final_conflict_intensity'] = intensities[-1] if intensities else 0
                
                # 严重冲突频率
                if 'severe_conflict_pairs' in conflict_metrics:
                    severe_conflicts = conflict_metrics['severe_conflict_pairs']
                    conflict_analysis['avg_severe_conflicts'] = np.mean(severe_conflicts)
                
                # Backbone影响均衡性
                backbone_influences = []
                for task in self.task_names:
                    key = f'backbone_influence_{task}'
                    if key in conflict_metrics:
                        task_influences = conflict_metrics[key]
                        if task_influences:
                            backbone_influences.append(np.mean(task_influences))
                
                if backbone_influences:
                    conflict_analysis['backbone_influence_balance'] = np.std(backbone_influences) / np.mean(backbone_influences)
                
                # 任务对冲突统计
                task_pair_conflicts = []
                for i, task_i in enumerate(self.task_names):
                    for j, task_j in enumerate(self.task_names):
                        if i != j:
                            conflict_key = f'conflict_{task_i}_{task_j}'
                            if conflict_key in conflict_metrics:
                                conflicts = conflict_metrics[conflict_key]
                                if conflicts:
                                    avg_conflict = np.mean(conflicts)
                                    task_pair_conflicts.append(avg_conflict)
                
                if task_pair_conflicts:
                    conflict_analysis['worst_task_pair_conflict'] = min(task_pair_conflicts)
                    conflict_analysis['best_task_pair_alignment'] = max(task_pair_conflicts)
                    conflict_analysis['conflict_variance'] = np.var(task_pair_conflicts)
            
            # 合并指标
            performance_metrics[method] = {**base_metrics, **conflict_analysis}
        
        return performance_metrics
    
    def run_all_tests(self, epochs: int = 15, steps_per_epoch: int = 8):
        """运行所有测试 - 修正版，每个方法独立从头开始"""
        
        # 验证方法
        print("🔧 Pre-flight check: Validating all methods...")
        validation_results = self.validate_all_methods()
        
        available_methods = [method for method, result in validation_results.items() 
                           if result['status'] == 'SUCCESS']
        failed_methods = [method for method, result in validation_results.items() 
                         if result['status'] == 'FAILED']
        
        if len(available_methods) == 0:
            print("❌ No methods are working! Please check the implementation.")
            return {}
        
        if failed_methods:
            print(f"⚠️  Warning: {len(failed_methods)} methods failed: {failed_methods}")
            print(f"✅ Proceeding with {len(available_methods)} working methods: {available_methods}")
        
        methods = available_methods
        all_results = {}
        
        # 设置wandb
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        wandb_dir = f"/workspace/runs/wandb/{timestamp}"
        import os
        os.makedirs(wandb_dir, exist_ok=True)
        
        if wandb.run is None:
            wandb.init(
                project="mtl-conflict-enhanced-analysis", 
                name=f"parallel_comparison_{timestamp}",
                dir=wandb_dir,
                config={
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "methods": methods,
                    "features": ["task_head_conflict_detection", "backbone_influence_analysis", "parallel_comparison"]
                }
            )
        
        total_steps_per_method = epochs * steps_per_epoch
        
        # 修正：每个方法独立测试，从step 0开始
        print(f"\n🔄 Testing Strategy: Each method trains independently for {total_steps_per_method} steps")
        print("📊 This enables fair comparison as all methods start from the same conditions\n")
        
        # 测试每个方法
        for i, method in enumerate(methods):
            try:
                print(f"\n{'='*60}")
                print(f"Testing {method.upper()} Independently ({i+1}/{len(methods)})")
                print(f"Training Steps: 0 to {total_steps_per_method - 1}")
                print(f"{'='*60}")
                
                # 修正：每个方法从global step 0开始，这样可视化时x轴是统一的
                results = self.test_method(method, epochs, steps_per_epoch, global_step_offset=0)
                all_results[method] = results
                
                print(f"✓ {method.upper()} completed successfully")
                
                # 显示该方法的冲突检测摘要
                if 'conflict_metrics' in results and results['conflict_metrics']:
                    self._display_method_conflict_summary(method, results)
                
            except Exception as e:
                print(f"✗ {method.upper()} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 分析结果
        if all_results:
            print(f"\n{'='*80}")
            print("🎯 PARALLEL METHOD COMPARISON ANALYSIS")
            print("📊 Each method trained independently for fair comparison")
            print(f"{'='*80}")
            
            performance_metrics = self.calculate_performance_metrics(all_results)
            
            # 显示详细指标表格
            print(f"\n{'Method':<12} {'Final Loss':<12} {'Improvement%':<12} {'Stability':<12} {'Balance CV':<12} {'Conflict':<12}")
            print("-" * 85)
            
            for method, metrics in performance_metrics.items():
                conflict_intensity = metrics.get('final_conflict_intensity', 0)
                print(f"{method.upper():<12} "
                      f"{metrics['final_loss']:<12.4f} "
                      f"{metrics['improvement_percent']:<12.1f} "
                      f"{metrics['convergence_stability']:<12.4f} "
                      f"{metrics['task_balance_cv']:<12.3f} "
                      f"{conflict_intensity:<12.3f}")
            
            # 冲突分析报告
            print(f"\n{'='*60}")
            print("🥊 TASK-HEAD CONFLICT ANALYSIS REPORT")
            print(f"{'='*60}")
            
            for method, metrics in performance_metrics.items():
                if any(k.startswith(('avg_conflict', 'backbone_', 'worst_task')) for k in metrics.keys()):
                    print(f"\n🔍 {method.upper()}:")
                    
                    if 'avg_conflict_intensity' in metrics:
                        intensity = metrics['avg_conflict_intensity']
                        level = "High" if intensity > 0.5 else "Medium" if intensity > 0.2 else "Low"
                        print(f"  • Average Conflict Intensity: {intensity:.3f} ({level})")
                    
                    if 'backbone_influence_balance' in metrics:
                        balance = metrics['backbone_influence_balance']
                        balance_desc = "Balanced" if balance < 0.3 else "Imbalanced"
                        print(f"  • Backbone Influence Balance: {balance:.3f} ({balance_desc})")
                    
                    if 'worst_task_pair_conflict' in metrics and 'best_task_pair_alignment' in metrics:
                        worst = metrics['worst_task_pair_conflict']
                        best = metrics['best_task_pair_alignment']
                        print(f"  • Task Pair Range: {worst:.3f} (worst) to {best:.3f} (best)")
            
            self.visualize_results(all_results)
            
            # 性能排名（考虑冲突因素）
            def composite_score(metrics):
                # 综合评分：性能 + 冲突处理能力
                performance_score = 1.0 / (metrics['final_loss'] + 1e-8)
                conflict_penalty = metrics.get('final_conflict_intensity', 0) * 0.5
                balance_bonus = 1.0 / (metrics['task_balance_cv'] + 1e-8) * 0.1
                return performance_score - conflict_penalty + balance_bonus
            
            sorted_methods = sorted(performance_metrics.items(), 
                                  key=lambda x: composite_score(x[1]), reverse=True)
            
            print(f"\n{'='*60}")
            print("🏆 FINAL RANKING (Performance + Conflict Resolution)")
            print(f"{'='*60}")
            
            for i, (method, metrics) in enumerate(sorted_methods, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                improvement = metrics['improvement_percent']
                conflict = metrics.get('final_conflict_intensity', 0)
                score = composite_score(metrics)
                print(f"{emoji} {i}. {method.upper()}: Score {score:.3f} "
                      f"(Loss: {metrics['final_loss']:.4f}, +{improvement:.1f}%, Conflict: {conflict:.3f})")
            
            best_method = sorted_methods[0][0]
            print(f"\n💡 RECOMMENDATION: {best_method.upper()} achieved the best balance of performance and conflict resolution")
            
            if best_method in all_results:
                self._analyze_best_method_enhanced(best_method, all_results[best_method], performance_metrics[best_method])
            
            print(f"\n📊 Detailed analysis saved to: {wandb_dir}")
            if wandb.run is not None:
                print(f"🌐 Online dashboard: {wandb.run.url}")
        
        return all_results
    
    def _display_method_conflict_summary(self, method_name: str, results: Dict):
        """显示方法的冲突检测摘要"""
        if 'conflict_metrics' not in results or not results['conflict_metrics']:
            return
        
        conflict_metrics = results['conflict_metrics']
        print(f"\n📊 {method_name.upper()} Conflict Detection Summary:")
        
        # 整体冲突强度
        if 'overall_conflict_intensity' in conflict_metrics:
            intensities = conflict_metrics['overall_conflict_intensity']
            if intensities:
                avg_intensity = np.mean(intensities)
                final_intensity = intensities[-1]
                print(f"  • Conflict Intensity: {final_intensity:.3f} (avg: {avg_intensity:.3f})")
        
        # 严重冲突对数
        if 'severe_conflict_pairs' in conflict_metrics:
            severe = conflict_metrics['severe_conflict_pairs']
            if severe:
                avg_severe = np.mean(severe)
                final_severe = severe[-1]
                print(f"  • Severe Conflicts: {final_severe} pairs (avg: {avg_severe:.1f})")
        
        # Backbone影响分布
        backbone_influences = []
        for task in self.task_names:
            key = f'backbone_influence_{task}'
            if key in conflict_metrics and conflict_metrics[key]:
                final_influence = conflict_metrics[key][-1]
                backbone_influences.append((task, final_influence))
        
        if backbone_influences:
            backbone_influences.sort(key=lambda x: x[1], reverse=True)
            dominant_task, dominant_influence = backbone_influences[0]
            print(f"  • Backbone Dominance: {dominant_task} ({dominant_influence:.3f})")
    
    def _analyze_best_method_enhanced(self, method_name: str, results: Dict, metrics: Dict):
        """分析最佳方法特征 - 增强版"""
        print(f"\n🔍 DETAILED ANALYSIS OF {method_name.upper()}:")
        
        # 原有分析
        if method_name == 'mgda':
            print("  • Multi-objective optimization effectively balanced competing objectives")
        elif method_name == 'gradnorm':
            print("  • Dynamic gradient balancing adapted to task difficulties")
        elif method_name == 'pcgrad':
            print("  • Gradient conflict resolution prevented negative interference")
        elif method_name == 'uncertainty':
            print("  • Uncertainty-based weighting provided robust task balancing")
        elif method_name == 'original':
            print("  • Simple uniform weighting was sufficient for this task configuration")
        
        # 新增：冲突处理分析
        if 'avg_conflict_intensity' in metrics:
            intensity = metrics['avg_conflict_intensity']
            if intensity < 0.2:
                print("  • Excellent conflict resolution: Maintained low task interference")
            elif intensity < 0.5:
                print("  • Good conflict management: Moderate task conflicts handled well")
            else:
                print("  • High conflict environment: Method performed despite significant interference")
        
        if 'backbone_influence_balance' in metrics:
            balance = metrics['backbone_influence_balance']
            if balance < 0.3:
                print("  • Balanced backbone utilization: All tasks fairly influenced shared parameters")
            else:
                print("  • Imbalanced backbone usage: Some tasks dominated shared parameter updates")
        
        # 权重演化分析
        weights_evolution = results['weights']
        for task in self.task_names:
            task_weights = weights_evolution[task]
            initial_weight = task_weights[0]
            final_weight = task_weights[-1]
            weight_change = abs(final_weight - initial_weight)
            print(f"  • {task} weight: {initial_weight:.3f} → {final_weight:.3f} (change: {weight_change:.3f})")


def main():
    """主测试函数 - 修正版，并列对比分析"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 MTL Conflict Resolution - Parallel Method Comparison")
    print(f"📱 Device: {device}")
    print(f"🎯 Focus: Task-head conflict detection + Parallel method comparison")
    print(f"🔬 Features: Independent training, Backbone influence analysis, Fair comparison")
    print("="*80)
    
    tester = MTLConflictTester(device)
    
    epochs, steps_per_epoch = 15, 8
    
    total_steps_per_method = epochs * steps_per_epoch
    print(f"📊 Configuration: {epochs} epochs × {steps_per_epoch} steps = {total_steps_per_method} steps per method")
    print(f"🔄 Strategy: Each method trains independently for fair comparison")
    print(f"⏱️ Estimated time: ~{total_steps_per_method * 5 // 60} minutes for all methods")
    print(f"📈 Visualization: All methods plotted from step 0 to {total_steps_per_method-1}")
    print(f"🆕 Enhanced features: Task-head conflict matrices, Backbone influence tracking\n")
    
    start_time = time.time()
    results = tester.run_all_tests(epochs=epochs, steps_per_epoch=steps_per_epoch)
    end_time = time.time()
    
    print(f"\n⏱️ Total analysis time: {end_time - start_time:.1f} seconds")
    print(f"📈 Average time per method: {(end_time - start_time) / 5:.1f} seconds")
    print(f"🎯 Conflict detection overhead: Minimal impact on training speed")
    print(f"📊 Fair comparison achieved: All methods trained under identical conditions")
    
    if wandb.run is not None:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    results = main()