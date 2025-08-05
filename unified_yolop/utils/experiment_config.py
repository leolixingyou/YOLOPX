"""Experiment configuration manager for YAML-based configs."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

class ExperimentConfig:
    """Manages experiment configurations from YAML files."""
    
    def __init__(self, config_path: str = None):
        """Initialize experiment configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = {}
        
        # Load base configuration first
        base_config_path = Path(__file__).parent.parent / "experiments" / "base_config.yaml"
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Override with specific config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                specific_config = yaml.safe_load(f)
                self._merge_configs(self.config, specific_config)
                
    def _merge_configs(self, base: Dict, override: Dict):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
                
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "training.epochs")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set the value
        config[keys[-1]] = value
        
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Map common arguments to config paths
        arg_mapping = {
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'lr': 'training.learning_rate',
            'workers': 'training.workers',
            'device': 'training.device',
            'seed': 'training.seed',
            'train_samples': 'dataset.train_samples',
            'val_samples': 'dataset.val_samples',
            'quick_test': 'dataset.quick_test',
            'wandb': 'logging.wandb',
            'models': 'models',
            'model': 'models'  # For single model
        }
        
        for arg_name, config_path in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    # Handle single model as list
                    if arg_name == 'model' and value is not None:
                        value = [value]
                    self.set(config_path, value)
                    
    def get_project_name(self, main_script: str) -> str:
        """Get project name based on configuration and main script.
        
        Args:
            main_script: Path to main script being run
            
        Returns:
            Project name
        """
        # Check if we should use main script name
        pattern = self.get('experiment.project_name_pattern', '{main_script}_{timestamp}')
        script_base = Path(main_script).stem
        
        # Check script priority
        priority_list = self.get('system.main_script_priority', [])
        if priority_list and script_base not in priority_list:
            # If not in priority list, use the experiment name
            return self.get('experiment.name', 'experiment')
        
        # Return script base name
        return script_base
        
    def save(self, path: str):
        """Save current configuration to YAML file.
        
        Args:
            path: Path to save configuration
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False)