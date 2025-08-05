#!/usr/bin/env python3
"""
Visualize comparison results from trained models.
Creates bar charts and tables for easy comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_comparison_data(results_dir):
    """Load comparison data from results directory."""
    models = ["yolop_v1", "yolop_v2", "yolop_v3", "yolopx"]
    data = {}
    
    for model in models:
        data[model] = {
            'performance': {},
            'conflicts': {}
        }
        
        # Try to load conflict results
        conflict_file = Path(results_dir) / f"{model}_conflict_results.json"
        if conflict_file.exists():
            with open(conflict_file, 'r') as f:
                conflict_data = json.load(f)
                if 'conflict_summary' in conflict_data:
                    data[model]['conflicts'] = conflict_data['conflict_summary']
        
        # Parse metrics from log
        log_file = Path(results_dir) / f"{model}_training.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if "overall_score:" in line:
                    try:
                        score = float(line.split(":")[-1].strip())
                        data[model]['performance']['overall_score'] = score
                    except:
                        pass
                elif "det_f1:" in line:
                    try:
                        f1 = float(line.split(":")[-1].strip())
                        data[model]['performance']['det_f1'] = f1
                    except:
                        pass
                elif "da_iou:" in line:
                    try:
                        iou = float(line.split(":")[-1].strip())
                        data[model]['performance']['da_iou'] = iou
                    except:
                        pass
                elif "ll_iou:" in line:
                    try:
                        iou = float(line.split(":")[-1].strip())
                        data[model]['performance']['ll_iou'] = iou
                    except:
                        pass
    
    return data

def create_performance_chart(data, save_path):
    """Create performance comparison bar chart."""
    models = list(data.keys())
    metrics = ['overall_score', 'det_f1', 'da_iou', 'll_iou']
    metric_names = ['Overall Score', 'Detection F1', 'DA IoU', 'Lane IoU']
    
    # Prepare data for plotting
    values = {metric: [] for metric in metrics}
    for model in models:
        for metric in metrics:
            val = data[model]['performance'].get(metric, 0)
            values[metric].append(val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    # Plot bars for each metric
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values[metric], width, label=name)
    
    # Customize plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('YOLOP Models Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        offset = (i - 1.5) * width
        for j, v in enumerate(values[metric]):
            if v > 0:
                ax.text(j + offset, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path / 'performance_comparison.png', dpi=300)
    plt.close()

def create_conflict_chart(data, save_path):
    """Create task conflict comparison chart."""
    models = list(data.keys())
    conflict_types = [
        'detection_da_seg_conflict_mean',
        'detection_ll_seg_conflict_mean', 
        'da_seg_ll_seg_conflict_mean'
    ]
    conflict_names = ['Det-DA', 'Det-LL', 'DA-LL']
    
    # Prepare data
    values = {conflict: [] for conflict in conflict_types}
    for model in models:
        for conflict in conflict_types:
            val = data[model]['conflicts'].get(conflict, 0)
            values[conflict].append(val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars
    for i, (conflict, name) in enumerate(zip(conflict_types, conflict_names)):
        offset = (i - 1) * width
        ax.bar(x + offset, values[conflict], width, label=name)
    
    # Customize plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Conflict Score')
    ax.set_title('Task Conflict Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (conflict, name) in enumerate(zip(conflict_types, conflict_names)):
        offset = (i - 1) * width
        for j, v in enumerate(values[conflict]):
            if v > 0:
                ax.text(j + offset, v + 0.0001, f'{v:.4f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path / 'conflict_comparison.png', dpi=300)
    plt.close()

def create_summary_table(data, save_path):
    """Create a summary table visualization."""
    models = list(data.keys())
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Params', 'Overall', 'Det F1', 'DA IoU', 'LL IoU', 'Avg Conflict']
    
    param_counts = {
        'yolop_v1': '7.9M',
        'yolop_v2': '57.5M',
        'yolop_v3': '~8M',
        'yolopx': '~8M'
    }
    
    for model in models:
        perf = data[model]['performance']
        conf = data[model]['conflicts']
        
        # Calculate average conflict
        conflicts = [
            conf.get('detection_da_seg_conflict_mean', 0),
            conf.get('detection_ll_seg_conflict_mean', 0),
            conf.get('da_seg_ll_seg_conflict_mean', 0)
        ]
        avg_conflict = np.mean(conflicts) if conflicts else 0
        
        row = [
            model,
            param_counts.get(model, 'N/A'),
            f"{perf.get('overall_score', 0):.4f}",
            f"{perf.get('det_f1', 0):.4f}",
            f"{perf.get('da_iou', 0):.4f}",
            f"{perf.get('ll_iou', 0):.4f}",
            f"{avg_conflict:.4f}"
        ]
        table_data.append(row)
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for col_idx in range(2, 6):  # Performance columns
        col_values = [float(row[col_idx]) for row in table_data]
        if col_values:
            max_idx = col_values.index(max(col_values))
            table[(max_idx + 1, col_idx)].set_facecolor('#E8F5E9')
    
    # Highlight lowest conflict
    conflict_values = [float(row[6]) for row in table_data]
    if conflict_values:
        min_idx = conflict_values.index(min(conflict_values))
        table[(min_idx + 1, 6)].set_facecolor('#E8F5E9')
    
    plt.title('YOLOP Models Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model comparison results')
    parser.add_argument('results_dir', help='Directory containing comparison results')
    parser.add_argument('--no-display', action='store_true', help='Only save, do not display')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory {results_path} not found")
        return
    
    print(f"Loading data from {results_path}")
    data = load_comparison_data(results_path)
    
    print("Creating visualizations...")
    create_performance_chart(data, results_path)
    print("- Saved performance_comparison.png")
    
    create_conflict_chart(data, results_path)
    print("- Saved conflict_comparison.png")
    
    create_summary_table(data, results_path)
    print("- Saved summary_table.png")
    
    print(f"\nVisualizations saved to {results_path}/")
    
    if not args.no_display:
        # Display images if running in interactive environment
        try:
            from IPython.display import Image, display
            display(Image(results_path / 'summary_table.png'))
            display(Image(results_path / 'performance_comparison.png'))
            display(Image(results_path / 'conflict_comparison.png'))
        except:
            print("Run in Jupyter notebook to display images directly")

if __name__ == "__main__":
    main()