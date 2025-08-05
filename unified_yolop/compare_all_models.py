#!/usr/bin/env python3
"""
Comprehensive comparison script for all YOLOP models.
This script trains all models and generates a detailed comparison report.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

class ModelComparison:
    def __init__(self, epochs=20, batch_size=4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = ["yolop_v1", "yolop_v2", "yolop_v3", "yolopx"]
        self.results_dir = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(self.results_dir).mkdir(exist_ok=True)
        
    def train_model(self, model_name):
        """Train a single model and collect results."""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Prepare command
        cmd = [
            "python3", "train.py",
            "--model", model_name,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--workers", "0",
            "--name", f"{model_name}_comparison"
        ]
        
        # Run training
        log_file = Path(self.results_dir) / f"{model_name}_training.log"
        with open(log_file, 'w') as f:
            # Set environment to disable wandb if needed
            env = os.environ.copy()
            env['WANDB_MODE'] = 'offline'
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            
            # Stream output and save to file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                
            process.wait()
            
        # Find the latest run directory
        runs_dir = Path("runs")
        model_runs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and model_name in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if model_runs:
            latest_run = model_runs[0]
            
            # Copy important files
            conflict_file = latest_run / "conflict_results.json"
            if conflict_file.exists():
                import shutil
                shutil.copy(
                    conflict_file, 
                    Path(self.results_dir) / f"{model_name}_conflict_results.json"
                )
                
            # Extract metrics from log
            return self.extract_metrics(log_file, model_name)
        
        return None
    
    def extract_metrics(self, log_file, model_name):
        """Extract metrics from training log."""
        metrics = {
            'model': model_name,
            'performance': {},
            'conflicts': {},
            'training_time': 0
        }
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Find best performance metrics
        for i, line in enumerate(lines):
            if "Saved checkpoint" in line and "best.pth" in line:
                # Look for metrics in previous lines
                for j in range(max(0, i-15), i):
                    if "Metrics:" in lines[j]:
                        # Parse subsequent lines for metrics
                        for k in range(j+1, min(j+10, len(lines))):
                            metric_line = lines[k].strip()
                            if ':' in metric_line:
                                parts = metric_line.split(':')
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    try:
                                        value = float(parts[1].strip())
                                        metrics['performance'][key] = value
                                    except:
                                        pass
        
        # Load conflict results
        conflict_file = Path(self.results_dir) / f"{model_name}_conflict_results.json"
        if conflict_file.exists():
            with open(conflict_file, 'r') as f:
                conflict_data = json.load(f)
                if 'conflict_summary' in conflict_data:
                    metrics['conflicts'] = conflict_data['conflict_summary']
        
        return metrics
    
    def generate_report(self, results):
        """Generate comprehensive comparison report."""
        report_lines = []
        
        # Header
        report_lines.append("# YOLOP Models Comprehensive Comparison Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Configuration: {self.epochs} epochs, batch size {self.batch_size}")
        report_lines.append("\n## Model Overview")
        
        # Model descriptions
        model_info = {
            "yolop_v1": "Original YOLOP with CSPDarknet backbone (7.9M params)",
            "yolop_v2": "E-ELAN backbone with separate segmentation heads (57.5M params)",
            "yolop_v3": "Enhanced YOLOP with attention mechanisms (~8M params)",
            "yolopx": "Anchor-free detection with YOLOX-style head (~8M params)"
        }
        
        for model, desc in model_info.items():
            report_lines.append(f"- **{model}**: {desc}")
        
        # Performance comparison table
        report_lines.append("\n## Performance Comparison")
        report_lines.append("\n| Model | Det Precision | Det Recall | Det F1 | DA IoU | LL IoU | Overall Score |")
        report_lines.append("|-------|---------------|------------|--------|--------|--------|---------------|")
        
        for result in results:
            if result:
                perf = result['performance']
                model = result['model']
                report_lines.append(
                    f"| {model} | "
                    f"{perf.get('det_precision', 0):.4f} | "
                    f"{perf.get('det_recall', 0):.4f} | "
                    f"{perf.get('det_f1', 0):.4f} | "
                    f"{perf.get('da_iou', 0):.4f} | "
                    f"{perf.get('ll_iou', 0):.4f} | "
                    f"{perf.get('overall_score', 0):.4f} |"
                )
        
        # Task conflict comparison
        report_lines.append("\n## Task Conflict Analysis")
        report_lines.append("\n| Model | Det-DA Conflict | Det-LL Conflict | DA-LL Conflict | Average Conflict |")
        report_lines.append("|-------|-----------------|-----------------|----------------|------------------|")
        
        for result in results:
            if result and result['conflicts']:
                conf = result['conflicts']
                model = result['model']
                
                det_da = conf.get('detection_da_seg_conflict_mean', 0)
                det_ll = conf.get('detection_ll_seg_conflict_mean', 0)
                da_ll = conf.get('da_seg_ll_seg_conflict_mean', 0)
                avg_conflict = (det_da + det_ll + da_ll) / 3
                
                report_lines.append(
                    f"| {model} | "
                    f"{det_da:.4f} | "
                    f"{det_ll:.4f} | "
                    f"{da_ll:.4f} | "
                    f"{avg_conflict:.4f} |"
                )
        
        # Key findings
        report_lines.append("\n## Key Findings")
        
        # Find best performing model
        best_overall = max(
            results, 
            key=lambda x: x['performance'].get('overall_score', 0) if x else 0
        )
        if best_overall:
            report_lines.append(
                f"\n- **Best Overall Performance**: {best_overall['model']} "
                f"(score: {best_overall['performance'].get('overall_score', 0):.4f})"
            )
        
        # Find model with least conflicts
        valid_results = [r for r in results if r and r['conflicts']]
        if valid_results:
            least_conflict = min(
                valid_results,
                key=lambda x: (
                    x['conflicts'].get('detection_da_seg_conflict_mean', 0) +
                    x['conflicts'].get('detection_ll_seg_conflict_mean', 0) +
                    x['conflicts'].get('da_seg_ll_seg_conflict_mean', 0)
                ) / 3
            )
            avg_conf = (
                least_conflict['conflicts'].get('detection_da_seg_conflict_mean', 0) +
                least_conflict['conflicts'].get('detection_ll_seg_conflict_mean', 0) +
                least_conflict['conflicts'].get('da_seg_ll_seg_conflict_mean', 0)
            ) / 3
            report_lines.append(
                f"- **Least Task Conflicts**: {least_conflict['model']} "
                f"(average conflict: {avg_conf:.4f})"
            )
        
        # Save report
        report_path = Path(self.results_dir) / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        # Also save as text
        text_path = Path(self.results_dir) / "comparison_report.txt"
        with open(text_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        return '\n'.join(report_lines)
    
    def run_comparison(self):
        """Run the complete comparison."""
        print(f"Starting model comparison with {self.epochs} epochs")
        print(f"Results will be saved to: {self.results_dir}")
        
        results = []
        
        # Train each model
        for model in self.models:
            try:
                result = self.train_model(model)
                results.append(result)
                print(f"\n✓ Completed {model}")
            except Exception as e:
                print(f"\n✗ Failed {model}: {e}")
                results.append(None)
        
        # Generate report
        print(f"\n{'='*60}")
        print("Generating Comparison Report")
        print(f"{'='*60}")
        
        report = self.generate_report(results)
        print(report)
        
        print(f"\n\nResults saved to: {self.results_dir}/")
        print("- comparison_report.md: Detailed markdown report")
        print("- comparison_report.txt: Plain text report")
        print("- {model}_training.log: Individual training logs")
        print("- {model}_conflict_results.json: Conflict analysis data")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare all YOLOP models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--quick', action='store_true', help='Quick test with 2 epochs')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 2
        
    comparison = ModelComparison(epochs=args.epochs, batch_size=args.batch_size)
    comparison.run_comparison()


if __name__ == "__main__":
    main()