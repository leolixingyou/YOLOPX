import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from xy_MTL_conflict import create_mtl_resolver


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
    """MTL冲突测试器"""
    
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
                
                # 测试
                if method == 'mgda':
                    weights = resolver.compute_mgda_weights(losses, test_model.parameters())
                else:
                    weights = resolver.update_weights(losses, 0, 0)
                
                metrics = resolver.get_metrics()
                validation_results[method] = {'status': 'SUCCESS', 'weights': weights, 'metrics_count': len(metrics)}
                print(f"    ✅ {method} - OK ({len(metrics)} metrics)")
                
            except Exception as e:
                validation_results[method] = {'status': 'FAILED', 'error': str(e)}
                print(f"    ❌ {method} - FAILED: {str(e)}")
        
        success_count = sum(1 for r in validation_results.values() if r['status'] == 'SUCCESS')
        print(f"\n✅ Validation complete: {success_count}/{len(methods)} methods working")
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
        """测试单个方法"""
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
                    task_weights = resolver.update_weights(losses, epoch, current_step)
                    total_loss = sum(losses.values())
                    resolver.apply_pcgrad_to_gradients(losses, model)
                    optimizer.step()
                    scheduler.step()
                    
                elif method_name == 'mgda':
                    task_weights = resolver.compute_mgda_weights(losses, model.parameters())
                    total_loss = sum(task_weights[task] * losses[task] for task in self.task_names)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                elif method_name == 'uncertainty':
                    resolver.update_uncertainty(losses)
                    task_weights = resolver.update_weights(losses, epoch, current_step)
                    total_loss = resolver.get_loss_with_uncertainty(losses)
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                else:  # original, gradnorm
                    task_weights = resolver.update_weights(losses, epoch, current_step)
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
                
                # 记录到wandb
                wandb_metrics = {f'{method_name}_total_loss': total_loss.item()}
                for task in self.task_names:
                    wandb_metrics[f'{method_name}_loss_{task}'] = loss_values[task]
                    wandb_metrics[f'{method_name}_weight_{task}'] = task_weights[task]
                
                if method_metrics:
                    for key, value in method_metrics.items():
                        if any(prefix in key for prefix in ['detect_', 'operation_', 'uncertainty_', 'mgda_']):
                            wandb_metrics[f'{method_name}_{key}'] = value
                
                if wandb.run is not None:
                    wandb.log(wandb_metrics, step=global_step)
                
                current_step += 1
                
                # 进度显示
                if current_step % 30 == 0:
                    print(f"Step {current_step}: Loss = {total_loss.item():.4f}")
                    self._display_insights(method_name, method_metrics, loss_values, task_weights)
        
        return metrics_history
    
    def _display_insights(self, method_name: str, metrics: Dict, losses: Dict, weights: Dict):
        """显示方法洞察"""
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
        
        # 任务平衡度
        task_losses = [losses[task] for task in self.task_names]
        balance = np.std(task_losses) / np.mean(task_losses) if np.mean(task_losses) > 0 else 0
        balance_desc = "Good" if balance < 0.4 else "Poor"
        print(f"  ⚖️ Task Balance: {balance_desc} (CV: {balance:.3f})")
    
    def visualize_results(self, all_results: Dict[str, Dict]):
        """可视化结果"""
        methods = list(all_results.keys())
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 总损失对比
        ax = axes[0, 0]
        for method in methods:
            steps = all_results[method]['step_numbers']
            losses = all_results[method]['total_loss']
            ax.plot(steps, losses, label=method, linewidth=2, alpha=0.8)
        ax.set_title('Total Loss vs Training Steps', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各任务损失演化
        ax = axes[0, 1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        linestyles = ['-', '--', '-.', ':']
        
        for i, task in enumerate(self.task_names):
            for j, method in enumerate(methods):
                steps = all_results[method]['step_numbers']
                task_losses = all_results[method]['losses'][task]
                ax.plot(steps, task_losses, 
                       color=colors[i], 
                       linestyle=linestyles[j % len(linestyles)], 
                       alpha=0.7,
                       label=f'{task}_{method}')
        ax.set_title('Task Loss Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Task Loss')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 3. 动态权重变化
        ax = axes[0, 2]
        non_uniform_methods = [m for m in methods if m != 'original']
        for method in non_uniform_methods:
            steps = all_results[method]['step_numbers']
            for i, task in enumerate(self.task_names):
                weights = all_results[method]['weights'][task]
                ax.plot(steps, weights, 
                       color=colors[i], 
                       linestyle='-' if method == 'gradnorm' else '--',
                       alpha=0.7,
                       label=f'{task}_{method}')
        ax.set_title('Dynamic Task Weights', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Task Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 冲突检测指标
        ax = axes[1, 0]
        conflict_detected = False
        
        if 'pcgrad' in all_results:
            method_metrics = all_results['pcgrad']['method_metrics']
            steps = all_results['pcgrad']['step_numbers']
            
            if 'detect_avg_cosine_similarity' in method_metrics:
                similarities = method_metrics['detect_avg_cosine_similarity']
                ax.plot(steps, similarities, label='PCGrad Similarity', linewidth=2, color='blue')
                ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7, label='Conflict Threshold')
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Alignment Threshold')
                conflict_detected = True
            
            if 'detect_current_conflicts' in method_metrics:
                conflicts = method_metrics['detect_current_conflicts']
                non_zero_conflicts = [c for c in conflicts if c > 0]
                if non_zero_conflicts:
                    ax2 = ax.twinx()
                    display_steps = steps[::max(1, len(steps)//20)]
                    display_conflicts = conflicts[::max(1, len(conflicts)//20)]
                    ax2.bar(display_steps, display_conflicts, alpha=0.3, color='red', 
                           width=max(1, len(steps)//50), label='Conflicts/Step')
                    ax2.set_ylabel('Conflicts per Step', color='red')
        
        if 'gradnorm' in all_results:
            method_metrics = all_results['gradnorm']['method_metrics']
            steps = all_results['gradnorm']['step_numbers']
            
            if 'detect_training_rate_variance' in method_metrics:
                variances = method_metrics['detect_training_rate_variance']
                if not conflict_detected:
                    ax.plot(steps, variances, label='GradNorm Variance', linewidth=2, color='orange')
                    ax.set_ylabel('Training Rate Variance')
                else:
                    ax3 = ax.twinx()
                    ax3.spines['right'].set_position(('axes', 1.1))
                    ax3.plot(steps, variances, label='GradNorm Variance', linewidth=2, color='orange', alpha=0.7)
                    ax3.set_ylabel('Training Rate Variance', color='orange')
                conflict_detected = True
        
        if conflict_detected:
            ax.set_title('Conflict Detection Metrics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Steps')
            if 'pcgrad' in all_results:
                ax.set_ylabel('Cosine Similarity')
            ax.legend(loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No Conflict Data\nAvailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Conflict Detection (No Data)')
        ax.grid(True, alpha=0.3)
        
        # 5. 收敛性分析
        ax = axes[1, 1]
        window_size = 15
        for method in methods:
            steps = all_results[method]['step_numbers']
            losses = all_results[method]['total_loss']
            
            if len(losses) >= window_size:
                smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[window_size-1:]
                ax.plot(smoothed_steps, smoothed_losses, label=f'{method}_smoothed', linewidth=2)
        
        ax.set_title(f'Convergence Analysis (MA window={window_size})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Smoothed Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 最终性能对比
        ax = axes[1, 2]
        final_losses = []
        method_labels = []
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, method in enumerate(methods):
            final_loss = all_results[method]['total_loss'][-1]
            final_losses.append(final_loss)
            method_labels.append(method.upper())
        
        bars = ax.bar(method_labels, final_losses, color=colors_bar[:len(methods)], alpha=0.7)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Final Total Loss')
        ax.set_xticklabels(method_labels, rotation=45)
        
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        fig_path = f"/workspace/runs/mtl_results_{int(time.time())}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"📊 Results plot saved to: {fig_path}")
        
        if wandb.run is not None:
            wandb.log({"mtl_step_analysis": wandb.Image(fig)})
        
        plt.close()  # 关闭图表释放内存
        return fig
    
    def calculate_performance_metrics(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """计算性能指标"""
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
            
            performance_metrics[method] = {
                'final_loss': final_loss,
                'improvement_percent': improvement,
                'convergence_stability': convergence_stability,
                'early_learning_efficiency': early_improvement,
                'task_balance_cv': task_balance_cv,
            }
        
        return performance_metrics
    
    def run_all_tests(self, epochs: int = 15, steps_per_epoch: int = 8):
        """运行所有测试"""
        
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
                project="mtl-conflict-step-analysis", 
                name=f"step_comparison_{timestamp}",
                dir=wandb_dir,
                config={
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "methods": methods
                }
            )
        
        total_steps_per_method = epochs * steps_per_epoch
        current_global_offset = 0
        
        # 测试每个方法
        for i, method in enumerate(methods):
            try:
                print(f"\n{'='*60}")
                print(f"Testing {method.upper()} ({i+1}/{len(methods)})")
                print(f"Steps: {current_global_offset} to {current_global_offset + total_steps_per_method - 1}")
                print(f"{'='*60}")
                
                results = self.test_method(method, epochs, steps_per_epoch, current_global_offset)
                all_results[method] = results
                current_global_offset += total_steps_per_method
                
                print(f"✓ {method.upper()} completed successfully")
                
            except Exception as e:
                print(f"✗ {method.upper()} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 分析结果
        if all_results:
            print(f"\n{'='*80}")
            print("🎯 STEP-LEVEL PERFORMANCE ANALYSIS")
            print(f"{'='*80}")
            
            performance_metrics = self.calculate_performance_metrics(all_results)
            
            print(f"\n{'Method':<12} {'Final Loss':<12} {'Improvement%':<12} {'Stability':<12} {'Balance CV':<12} {'Learning Eff%':<12}")
            print("-" * 85)
            
            for method, metrics in performance_metrics.items():
                print(f"{method.upper():<12} "
                      f"{metrics['final_loss']:<12.4f} "
                      f"{metrics['improvement_percent']:<12.1f} "
                      f"{metrics['convergence_stability']:<12.4f} "
                      f"{metrics['task_balance_cv']:<12.3f} "
                      f"{metrics['early_learning_efficiency']:<12.1f}")
            
            self.visualize_results(all_results)
            
            # 性能排名
            sorted_methods = sorted(performance_metrics.items(), key=lambda x: x[1]['final_loss'])
            
            print(f"\n{'='*60}")
            print("🏆 FINAL RANKING BY PERFORMANCE")
            print(f"{'='*60}")
            
            for i, (method, metrics) in enumerate(sorted_methods, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                improvement = metrics['improvement_percent']
                efficiency = metrics['early_learning_efficiency']
                print(f"{emoji} {i}. {method.upper()}: {metrics['final_loss']:.4f} "
                      f"(+{improvement:.1f}% improvement, {efficiency:.1f}% early efficiency)")
            
            best_method = sorted_methods[0][0]
            print(f"\n💡 RECOMMENDATION: {best_method.upper()} achieved the best overall performance")
            
            if best_method in all_results:
                self._analyze_best_method(best_method, all_results[best_method])
            
            print(f"\n📊 Detailed analysis saved to: {wandb_dir}")
            if wandb.run is not None:
                print(f"🌐 Online dashboard: {wandb.run.url}")
        
        return all_results
    
    def _analyze_best_method(self, method_name: str, results: Dict):
        """分析最佳方法特征"""
        print(f"\n🔍 DETAILED ANALYSIS OF {method_name.upper()}:")
        
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
        
        # 权重演化分析
        weights_evolution = results['weights']
        for task in self.task_names:
            task_weights = weights_evolution[task]
            initial_weight = task_weights[0]
            final_weight = task_weights[-1]
            weight_change = abs(final_weight - initial_weight)
            print(f"  • {task} weight: {initial_weight:.3f} → {final_weight:.3f} (change: {weight_change:.3f})")


def main():
    """主测试函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 MTL Conflict Resolution - Step-Level Analysis")
    print(f"📱 Device: {device}")
    print(f"🎯 Focus: Training step granularity for conflict dynamics")
    print("="*70)
    
    tester = MTLConflictTester(device)
    
    epochs, steps_per_epoch = 15, 8  # 简化配置
    
    total_steps = epochs * steps_per_epoch * 5  # 5 methods
    print(f"📊 Configuration: {epochs} epochs × {steps_per_epoch} steps × 5 methods = {total_steps} total steps")
    print(f"⏱️ Estimated time: ~{total_steps // 60} minutes\n")
    
    start_time = time.time()
    results = tester.run_all_tests(epochs=epochs, steps_per_epoch=steps_per_epoch)
    end_time = time.time()
    
    print(f"\n⏱️ Total analysis time: {end_time - start_time:.1f} seconds")
    print(f"📈 Steps per second: {total_steps / (end_time - start_time):.1f}")
    
    if wandb.run is not None:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    results = main()