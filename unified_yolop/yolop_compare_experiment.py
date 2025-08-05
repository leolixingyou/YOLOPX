#!/usr/bin/env python3
"""
YOLOP Models Comprehensive Comparison Experiment
This script trains all YOLOP models in parallel and generates comprehensive comparison reports.
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

class YOLOPComparisonExperiment:
    def __init__(self, epochs=20, batch_size=4, parallel=True, num_workers=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.parallel = parallel
        self.num_workers = min(num_workers, mp.cpu_count()) if num_workers > 0 else 0
        self.workers = num_workers  # For DataLoader workers, not parallel training workers
        self.models = ["yolop_v1", "yolop_v2", "yolop_v3", "yolopx"]
        
        # Create experiment directory
        self.exp_name = f"yolop_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = Path(self.exp_name)
        self.exp_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {}
        
    def train_single_model(self, model_name):
        """Train a single model and return results."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training for {model_name}")
        
        # Prepare paths
        log_file = self.exp_dir / f"{model_name}_training.log"
        
        # Prepare command
        cmd = [
            "python3", "train.py",
            "--model", model_name,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--workers", str(self.workers),
            "--name", f"{model_name}_{self.exp_name}"
        ]
        
        # Set environment
        env = os.environ.copy()
        env['WANDB_MODE'] = 'offline'
        env['CUDA_VISIBLE_DEVICES'] = str(self.models.index(model_name) % 4)  # Distribute across GPUs if available
        
        # Run training
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True
                )
            
            training_time = time.time() - start_time
            
            # Extract results
            result = self.extract_model_results(model_name, log_file, training_time)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed {model_name} in {training_time/60:.1f} minutes")
            return result
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed {model_name}: {e}")
            return {
                'model': model_name,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def extract_model_results(self, model_name, log_file, training_time):
        """Extract comprehensive results from model training."""
        result = {
            'model': model_name,
            'status': 'success',
            'training_time': training_time,
            'performance': {},
            'conflicts': {},
            'losses': {'train': [], 'val': []},
            'epochs_completed': 0
        }
        
        # Find latest run directory
        runs_dir = Path("runs")
        model_runs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and f"{model_name}_{self.exp_name}" in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if model_runs:
            latest_run = model_runs[0]
            
            # Copy conflict results
            conflict_file = latest_run / "conflict_results.json"
            if conflict_file.exists():
                import shutil
                dest_file = self.exp_dir / f"{model_name}_conflict_results.json"
                shutil.copy(conflict_file, dest_file)
                
                with open(conflict_file, 'r') as f:
                    conflict_data = json.load(f)
                    if 'conflict_summary' in conflict_data:
                        result['conflicts'] = conflict_data['conflict_summary']
            
            # Copy best checkpoint
            best_ckpt = latest_run / "best.pth"
            if best_ckpt.exists():
                shutil.copy(best_ckpt, self.exp_dir / f"{model_name}_best.pth")
        
        # Parse log file for metrics
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Extract performance metrics
            if "Metrics:" in line:
                for j in range(i+1, min(i+10, len(lines))):
                    metric_line = lines[j].strip()
                    if ':' in metric_line:
                        parts = metric_line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            try:
                                value = float(parts[1].strip())
                                result['performance'][key] = value
                            except:
                                pass
            
            # Extract epoch info
            if "Epoch [" in line and "/" in line:
                try:
                    epoch_info = line.split('[')[1].split(']')[0]
                    current_epoch = int(epoch_info.split('/')[0])
                    result['epochs_completed'] = max(result['epochs_completed'], current_epoch)
                except:
                    pass
            
            # Extract losses
            if "Training:" in line and "total_loss:" in line:
                try:
                    loss = float(line.split("total_loss:")[1].strip().split()[0])
                    result['losses']['train'].append(loss)
                except:
                    pass
            elif "Validation:" in line and "total_loss:" in line:
                try:
                    loss = float(line.split("total_loss:")[1].strip().split()[0])
                    result['losses']['val'].append(loss)
                except:
                    pass
        
        return result
    
    def run_parallel_training(self):
        """Run training for all models in parallel."""
        print(f"Starting parallel training with {self.num_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all training jobs
            future_to_model = {
                executor.submit(self.train_single_model, model): model 
                for model in self.models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    self.results[model] = result
                except Exception as e:
                    print(f"Error training {model}: {e}")
                    self.results[model] = {
                        'model': model,
                        'status': 'failed',
                        'error': str(e)
                    }
    
    def run_sequential_training(self):
        """Run training for all models sequentially."""
        print("Starting sequential training")
        
        for model in self.models:
            result = self.train_single_model(model)
            self.results[model] = result
    
    def generate_performance_chart(self):
        """Generate performance comparison chart."""
        models = []
        metrics_data = {
            'Overall Score': [],
            'Detection F1': [],
            'DA IoU': [],
            'Lane IoU': []
        }
        
        for model in self.models:
            if model in self.results and self.results[model]['status'] == 'success':
                models.append(model)
                perf = self.results[model]['performance']
                metrics_data['Overall Score'].append(perf.get('overall_score', 0))
                metrics_data['Detection F1'].append(perf.get('det_f1', 0))
                metrics_data['DA IoU'].append(perf.get('da_iou', 0))
                metrics_data['Lane IoU'].append(perf.get('ll_iou', 0))
        
        if not models:
            return
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance bar chart
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            offset = (i - 1.5) * width
            bars = ax1.bar(x + offset, values, width, label=metric)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Training time comparison
        times = []
        for model in models:
            times.append(self.results[model]['training_time'] / 60)  # Convert to minutes
        
        bars = ax2.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Training Time (minutes)')
        ax2.set_title('Training Time Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add time labels
        for bar, time in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'performance_comparison.png', dpi=300)
        plt.close()
    
    def generate_conflict_heatmap(self):
        """Generate task conflict heatmap."""
        conflict_types = [
            ('detection_da_seg_conflict_mean', 'Det→DA'),
            ('detection_ll_seg_conflict_mean', 'Det→LL'),
            ('da_seg_ll_seg_conflict_mean', 'DA→LL')
        ]
        
        models = []
        conflict_matrix = []
        
        for model in self.models:
            if model in self.results and self.results[model]['status'] == 'success':
                models.append(model)
                conflicts = self.results[model]['conflicts']
                row = []
                for conflict_key, _ in conflict_types:
                    row.append(conflicts.get(conflict_key, 0))
                conflict_matrix.append(row)
        
        if not models:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conflict_array = np.array(conflict_matrix).T
        im = ax.imshow(conflict_array, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(conflict_types)))
        ax.set_xticklabels(models)
        ax.set_yticklabels([name for _, name in conflict_types])
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Conflict Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(conflict_types)):
            for j in range(len(models)):
                text = ax.text(j, i, f'{conflict_array[i, j]:.4f}',
                             ha="center", va="center", color="black" if conflict_array[i, j] < 0.5 else "white")
        
        ax.set_title('Task Conflict Analysis Heatmap')
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'conflict_heatmap.png', dpi=300)
        plt.close()
    
    def generate_loss_curves(self):
        """Generate training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, model in enumerate(self.models):
            if model in self.results and self.results[model]['status'] == 'success':
                ax = axes[idx]
                result = self.results[model]
                
                # Plot losses
                epochs = range(1, len(result['losses']['train']) + 1)
                ax.plot(epochs, result['losses']['train'], 'b-', label='Train Loss')
                if result['losses']['val']:
                    ax.plot(epochs, result['losses']['val'], 'r-', label='Val Loss')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{model} Training Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'loss_curves.png', dpi=300)
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_lines = []
        
        # Header
        report_lines.append("# YOLOP Models Comprehensive Comparison Experiment")
        report_lines.append(f"\n**Experiment Name**: {self.exp_name}")
        report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Configuration**: {self.epochs} epochs, batch size {self.batch_size}")
        report_lines.append(f"**Training Mode**: {'Parallel' if self.parallel else 'Sequential'}")
        
        # Model Information
        report_lines.append("\n## Model Information")
        model_info = {
            "yolop_v1": {"params": "7.9M", "backbone": "CSPDarknet", "detection": "Anchor-based"},
            "yolop_v2": {"params": "57.5M", "backbone": "E-ELAN", "detection": "Anchor-based"},
            "yolop_v3": {"params": "~8M", "backbone": "ELANNet + SimAM", "detection": "Anchor-based"},
            "yolopx": {"params": "~8M", "backbone": "ELANNet", "detection": "Anchor-free (YOLOX)"}
        }
        
        report_lines.append("\n| Model | Parameters | Backbone | Detection Type |")
        report_lines.append("|-------|------------|----------|----------------|")
        for model, info in model_info.items():
            report_lines.append(f"| {model} | {info['params']} | {info['backbone']} | {info['detection']} |")
        
        # Training Status
        report_lines.append("\n## Training Status")
        report_lines.append("\n| Model | Status | Epochs Completed | Training Time |")
        report_lines.append("|-------|--------|------------------|---------------|")
        
        total_time = 0
        for model in self.models:
            if model in self.results:
                result = self.results[model]
                status = "✓ Success" if result['status'] == 'success' else "✗ Failed"
                epochs = result.get('epochs_completed', 0)
                time_min = result['training_time'] / 60
                total_time += result['training_time']
                report_lines.append(f"| {model} | {status} | {epochs}/{self.epochs} | {time_min:.1f} min |")
        
        report_lines.append(f"\n**Total Training Time**: {total_time/60:.1f} minutes")
        if self.parallel:
            report_lines.append(f"**Time Saved by Parallel Training**: ~{(total_time * 0.75)/60:.1f} minutes")
        
        # Performance Comparison
        report_lines.append("\n## Performance Comparison")
        report_lines.append("\n| Model | Overall Score | Det Precision | Det Recall | Det F1 | DA IoU | Lane IoU |")
        report_lines.append("|-------|---------------|---------------|------------|--------|--------|----------|")
        
        best_overall = {'model': None, 'score': 0}
        
        for model in self.models:
            if model in self.results and self.results[model]['status'] == 'success':
                perf = self.results[model]['performance']
                overall = perf.get('overall_score', 0)
                
                if overall > best_overall['score']:
                    best_overall = {'model': model, 'score': overall}
                
                report_lines.append(
                    f"| {model} | "
                    f"**{overall:.4f}** | "
                    f"{perf.get('det_precision', 0):.4f} | "
                    f"{perf.get('det_recall', 0):.4f} | "
                    f"{perf.get('det_f1', 0):.4f} | "
                    f"{perf.get('da_iou', 0):.4f} | "
                    f"{perf.get('ll_iou', 0):.4f} |"
                )
        
        # Task Conflict Analysis
        report_lines.append("\n## Task Conflict Analysis")
        report_lines.append("\n| Model | Det→DA | Det→LL | DA→LL | Average | Max |")
        report_lines.append("|-------|--------|--------|-------|---------|-----|")
        
        least_conflict = {'model': None, 'avg': float('inf')}
        
        for model in self.models:
            if model in self.results and self.results[model]['status'] == 'success':
                conflicts = self.results[model]['conflicts']
                
                det_da = conflicts.get('detection_da_seg_conflict_mean', 0)
                det_ll = conflicts.get('detection_ll_seg_conflict_mean', 0)
                da_ll = conflicts.get('da_seg_ll_seg_conflict_mean', 0)
                
                avg_conflict = (det_da + det_ll + da_ll) / 3
                max_conflict = max(det_da, det_ll, da_ll)
                
                if avg_conflict < least_conflict['avg']:
                    least_conflict = {'model': model, 'avg': avg_conflict}
                
                report_lines.append(
                    f"| {model} | "
                    f"{det_da:.4f} | "
                    f"{det_ll:.4f} | "
                    f"{da_ll:.4f} | "
                    f"{avg_conflict:.4f} | "
                    f"{max_conflict:.4f} |"
                )
        
        # Key Findings
        report_lines.append("\n## Key Findings")
        
        if best_overall['model']:
            report_lines.append(f"\n### 🏆 Best Overall Performance")
            report_lines.append(f"- **Model**: {best_overall['model']}")
            report_lines.append(f"- **Score**: {best_overall['score']:.4f}")
        
        if least_conflict['model']:
            report_lines.append(f"\n### 🤝 Least Task Conflicts")
            report_lines.append(f"- **Model**: {least_conflict['model']}")
            report_lines.append(f"- **Average Conflict**: {least_conflict['avg']:.4f}")
        
        # Recommendations
        report_lines.append("\n## Recommendations")
        
        if best_overall['model'] == least_conflict['model']:
            report_lines.append(f"\n✅ **{best_overall['model']}** achieves both best performance and least conflicts!")
        else:
            report_lines.append(f"\n- For **best accuracy**: Use {best_overall['model']}")
            report_lines.append(f"- For **least conflicts**: Use {least_conflict['model']}")
        
        # File list
        report_lines.append("\n## Generated Files")
        report_lines.append(f"\n- 📁 Experiment Directory: `{self.exp_dir}/`")
        report_lines.append("- 📊 `performance_comparison.png` - Performance metrics comparison")
        report_lines.append("- 🔥 `conflict_heatmap.png` - Task conflict heatmap")
        report_lines.append("- 📈 `loss_curves.png` - Training loss curves")
        report_lines.append("- 📝 `{model}_training.log` - Individual training logs")
        report_lines.append("- 🎯 `{model}_best.pth` - Best model checkpoints")
        report_lines.append("- 📊 `experiment_results.json` - Complete results data")
        
        # Save report
        report_path = self.exp_dir / 'experiment_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save as text
        text_path = self.exp_dir / 'experiment_report.txt'
        with open(text_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return '\n'.join(report_lines)
    
    def save_results_json(self):
        """Save complete results as JSON."""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results_data = {
            'experiment_name': self.exp_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'parallel': self.parallel,
                'num_workers': self.num_workers
            },
            'results': convert_types(self.results)
        }
        
        with open(self.exp_dir / 'experiment_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def run_experiment(self):
        """Run the complete experiment."""
        print(f"\n{'='*60}")
        print(f"YOLOP Models Comparison Experiment")
        print(f"{'='*60}")
        print(f"Experiment: {self.exp_name}")
        print(f"Configuration: {self.epochs} epochs, batch size {self.batch_size}")
        print(f"Models: {', '.join(self.models)}")
        print(f"{'='*60}\n")
        
        # Start training
        start_time = time.time()
        
        if self.parallel:
            self.run_parallel_training()
        else:
            self.run_sequential_training()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")
        
        # Generate visualizations and reports
        print("Generating visualizations...")
        self.generate_performance_chart()
        print("✓ Performance comparison chart")
        
        self.generate_conflict_heatmap()
        print("✓ Conflict heatmap")
        
        self.generate_loss_curves()
        print("✓ Loss curves")
        
        # Save results
        self.save_results_json()
        print("✓ Results JSON")
        
        # Generate final report
        print("\nGenerating final report...")
        report = self.generate_summary_report()
        print("✓ Experiment report")
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(report)
        
        print(f"\n✅ Experiment complete! Results saved to: {self.exp_dir}/")
        
        return self.results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLOP Models Comparison Experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequential', action='store_true', help='Train models sequentially instead of parallel')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers')
    parser.add_argument('--quick', action='store_true', help='Quick test with 2 epochs')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 2
        print("Running quick test with 2 epochs...")
    
    # Run experiment
    experiment = YOLOPComparisonExperiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        parallel=not args.sequential,
        num_workers=args.workers
    )
    
    results = experiment.run_experiment()
    
    # Return success code based on results
    failed_models = [m for m, r in results.items() if r['status'] != 'success']
    if failed_models:
        print(f"\n⚠️  Warning: {len(failed_models)} models failed: {', '.join(failed_models)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())