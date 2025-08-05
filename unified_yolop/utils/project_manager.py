"""Project manager for unified result saving and reporting."""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class ProjectManager:
    """Manages project directories and results for YOLOP experiments."""
    
    def __init__(self, models: List[str], base_dir: str = "runs", 
                 project_name: str = None, main_script: str = None,
                 use_shared_project: bool = True, create_directories: bool = True):
        """Initialize project manager.
        
        Args:
            models: List of model names to train
            base_dir: Base directory for all runs
            project_name: Custom project name (if None, auto-generate)
            main_script: Name of the main script being run
            use_shared_project: If True, reuse existing project for sub-experiments
            create_directories: If False, don't create any directories (for child processes)
        """
        self.models = models
        self.base_dir = Path(base_dir)
        self.use_shared_project = use_shared_project
        self.create_directories = create_directories
        
        # Only create base directory if needed
        if create_directories:
            self.base_dir.mkdir(exist_ok=True)
        
        # Create project name based on configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if project_name:
            self.project_name = f"{project_name}_{timestamp}"
        elif main_script:
            # Use main script name as project name
            script_base = Path(main_script).stem
            self.project_name = f"{script_base}_{timestamp}"
        else:
            # Fallback to old naming
            num_models = len(models)
            self.project_name = f"yolop_model_{num_models}_{timestamp}"
        
        # Create project directory only if flag is True
        self.project_dir = self.base_dir / self.project_name
        if create_directories:
            self.project_dir.mkdir(exist_ok=True)
        
        # Create model subdirectories
        self.model_dirs = {}
        if create_directories:
            for model in models:
                model_dir = self.project_dir / model
                model_dir.mkdir(exist_ok=True)
                self.model_dirs[model] = model_dir
            
        # Project metadata
        self.metadata = {
            "project_name": self.project_name,
            "models": models,
            "num_models": len(models),
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "main_script": main_script if main_script else "unknown"
        }
        
        # Save metadata only if creating directories
        if create_directories:
            self.save_metadata()
        
    def get_model_dir(self, model_name: str) -> Path:
        """Get directory for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model directory
        """
        return self.model_dirs.get(model_name, self.project_dir / model_name)
        
    def get_checkpoint_dir(self, model_name: str) -> Path:
        """Get checkpoint directory for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to checkpoint directory
        """
        checkpoint_dir = self.get_model_dir(model_name) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir
        
    def get_log_dir(self, model_name: str) -> Path:
        """Get log directory for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to log directory
        """
        log_dir = self.get_model_dir(model_name) / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir
        
    def get_visualization_dir(self, model_name: str, epoch: Optional[int] = None) -> Path:
        """Get visualization directory for a model.
        
        Args:
            model_name: Name of the model
            epoch: Epoch number (optional)
            
        Returns:
            Path to visualization directory
        """
        # Use train_output to be consistent with run_experiment.py
        vis_dir = self.get_model_dir(model_name) / "train_output"
        if epoch is not None:
            vis_dir = vis_dir / f"val_epoch_{epoch}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        return vis_dir
        
    def save_model_results(self, model_name: str, results: Dict):
        """Save results for a specific model.
        
        Args:
            model_name: Name of the model
            results: Results dictionary
        """
        results_path = self.get_model_dir(model_name) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def save_model_config(self, model_name: str, config: Dict):
        """Save configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        config_path = self.get_model_dir(model_name) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def save_training_log(self, model_name: str, log_content: str):
        """Save training log for a model.
        
        Args:
            model_name: Name of the model
            log_content: Log content as string
        """
        log_path = self.get_model_dir(model_name) / "training.log"
        with open(log_path, 'w') as f:
            f.write(log_content)
            
    def update_metadata(self, updates: Dict):
        """Update project metadata.
        
        Args:
            updates: Dictionary of updates
        """
        self.metadata.update(updates)
        self.save_metadata()
        
    def save_metadata(self):
        """Save project metadata."""
        metadata_path = self.project_dir / "project_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
    def generate_comparison_report(self):
        """Generate comparison report for all models."""
        report = {
            "project": self.project_name,
            "created_at": datetime.now().isoformat(),
            "models": {}
        }
        
        # Collect results from all models
        for model in self.models:
            results_path = self.get_model_dir(model) / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    report["models"][model] = json.load(f)
                    
        # Generate comparison tables
        if len(report["models"]) > 1:
            report["comparison"] = self._generate_comparison_tables(report["models"])
            
        # Save report
        report_path = self.project_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        # Generate markdown report
        self._generate_markdown_report(report)
        
        return report
        
    def _generate_comparison_tables(self, model_results: Dict) -> Dict:
        """Generate comparison tables from model results."""
        comparison = {
            "performance": {},
            "training_time": {},
            "conflicts": {}
        }
        
        # Extract metrics for comparison
        for model, results in model_results.items():
            if "final_metrics" in results:
                metrics = results["final_metrics"]
                comparison["performance"][model] = {
                    "det_f1": metrics.get("det_f1", 0),
                    "da_iou": metrics.get("da_iou", 0),
                    "ll_iou": metrics.get("ll_iou", 0),
                    "overall_score": metrics.get("overall_score", 0)
                }
                
            if "training_time" in results:
                comparison["training_time"][model] = results["training_time"]
                
            if "conflict_summary" in results:
                comparison["conflicts"][model] = results["conflict_summary"]
                
        return comparison
        
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown report from JSON report."""
        md_lines = []
        
        # Header
        md_lines.append(f"# YOLOP Model Comparison Report")
        md_lines.append(f"\n**Project**: {report['project']}")
        md_lines.append(f"**Generated**: {report['created_at']}")
        md_lines.append(f"**Models**: {', '.join(self.models)}")
        md_lines.append("\n---\n")
        
        # Individual model results
        for model, results in report["models"].items():
            md_lines.append(f"\n## {model.upper()}")
            
            if "final_metrics" in results:
                md_lines.append("\n### Performance Metrics")
                metrics = results["final_metrics"]
                md_lines.append(f"- Detection F1: {metrics.get('det_f1', 0):.4f}")
                md_lines.append(f"- Drivable Area IoU: {metrics.get('da_iou', 0):.4f}")
                md_lines.append(f"- Lane Line IoU: {metrics.get('ll_iou', 0):.4f}")
                md_lines.append(f"- **Overall Score**: {metrics.get('overall_score', 0):.4f}")
                
            if "training_time" in results:
                md_lines.append(f"\n### Training Time: {results['training_time']:.2f}s")
                
        # Comparison tables if multiple models
        if "comparison" in report and len(self.models) > 1:
            md_lines.append("\n---\n\n## Model Comparison")
            
            # Performance table
            if "performance" in report["comparison"]:
                md_lines.append("\n### Performance Comparison")
                md_lines.append("\n| Model | Det F1 | DA IoU | Lane IoU | Overall |")
                md_lines.append("|-------|--------|--------|----------|---------|")
                
                for model, metrics in report["comparison"]["performance"].items():
                    md_lines.append(
                        f"| {model} | "
                        f"{metrics['det_f1']:.4f} | "
                        f"{metrics['da_iou']:.4f} | "
                        f"{metrics['ll_iou']:.4f} | "
                        f"**{metrics['overall_score']:.4f}** |"
                    )
                    
            # Conflict analysis
            if "conflicts" in report["comparison"]:
                md_lines.append("\n### Task Conflict Analysis")
                md_lines.append("\n| Model | Overall TCI | Det-DA | Det-LL | DA-LL |")
                md_lines.append("|-------|-------------|--------|--------|--------|")
                
                for model, conflicts in report["comparison"]["conflicts"].items():
                    if isinstance(conflicts, dict):
                        md_lines.append(
                            f"| {model} | "
                            f"{conflicts.get('overall_tci', 0):.4f} | "
                            f"{conflicts.get('detection_da_seg_tci_mean', 0):.4f} | "
                            f"{conflicts.get('detection_ll_seg_tci_mean', 0):.4f} | "
                            f"{conflicts.get('da_seg_ll_seg_tci_mean', 0):.4f} |"
                        )
                        
        # Save markdown report
        md_path = self.project_dir / "comparison_report.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
            
    def copy_legacy_results(self, legacy_dir: Path, model_name: str):
        """Copy results from legacy directory structure.
        
        Args:
            legacy_dir: Path to legacy results directory
            model_name: Name of the model
        """
        if legacy_dir.exists():
            # Copy checkpoints
            for file in legacy_dir.glob("*.pth"):
                shutil.copy2(file, self.get_checkpoint_dir(model_name))
                
            # Copy logs
            for file in legacy_dir.glob("*.log"):
                shutil.copy2(file, self.get_model_dir(model_name))
                
            # Copy visualizations
            for vis_dir in legacy_dir.glob("val_epoch_*"):
                epoch = int(vis_dir.name.split("_")[-1])
                target_dir = self.get_visualization_dir(model_name, epoch)
                if vis_dir.is_dir():
                    shutil.copytree(vis_dir, target_dir, dirs_exist_ok=True)