#!/usr/bin/env python3
"""Generate comparison report for existing models."""

import json
import argparse
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.project_manager import ProjectManager

def main():
    parser = argparse.ArgumentParser(description='Generate comparison report')
    parser.add_argument('--project-dir', type=str, required=True,
                       help='Project directory path')
    args = parser.parse_args()
    
    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        print(f"Project directory not found: {project_dir}")
        return 1
        
    # Load metadata
    metadata_path = project_dir / "project_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print("Project metadata not found")
        return 1
        
    # Create project manager instance
    models = metadata.get('models', [])
    project_manager = ProjectManager([], str(project_dir.parent))
    project_manager.project_dir = project_dir
    project_manager.project_name = project_dir.name
    project_manager.metadata = metadata
    project_manager.models = models
    
    # Create model directories mapping
    project_manager.model_dirs = {}
    for model in models:
        model_dir = project_manager.project_dir / model
        if model_dir.exists():
            project_manager.model_dirs[model] = model_dir
            
    # Generate comparison report
    print(f"Generating comparison report for models: {models}")
    report = project_manager.generate_comparison_report()
    
    print(f"Report saved to: {project_dir / 'comparison_report.md'}")
    return 0

if __name__ == '__main__':
    sys.exit(main())