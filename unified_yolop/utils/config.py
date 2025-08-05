"""Configuration management for Unified YOLOP framework."""

import os
import yaml
import argparse
from typing import Dict, Any, Optional

class Config:
    """Configuration manager that handles YAML files and command-line overrides."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Load default configuration
        self.config = self._load_default_config()
        
        # Override with custom config file if provided
        if config_file and os.path.exists(config_file):
            custom_config = self._load_yaml(config_file)
            self._merge_config(custom_config)
            
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'configs', 'default.yaml'
        )
        return self._load_yaml(default_path)
        
    def _load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _merge_config(self, new_config: Dict[str, Any]):
        """Recursively merge new configuration into existing."""
        def recursive_merge(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    recursive_merge(base[key], value)
                else:
                    base[key] = value
                    
        recursive_merge(self.config, new_config)
        
    def load_model_config(self, model_name: str):
        """Load model-specific configuration.
        
        Args:
            model_name: Name of the model (e.g., 'yolop_v1', 'yolopx')
        """
        model_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'configs', 'models', f'{model_name}.yaml'
        )
        
        if os.path.exists(model_config_path):
            model_config = self._load_yaml(model_config_path)
            self._merge_config(model_config)
        else:
            print(f"Warning: Model config not found: {model_config_path}")
            
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Update model name
        if hasattr(args, 'model') and args.model:
            self.config['MODEL']['NAME'] = args.model
            
        # Update device
        if hasattr(args, 'device') and args.device:
            self.config['SYSTEM']['DEVICE'] = args.device
            
        # Update batch size
        if hasattr(args, 'batch_size') and args.batch_size:
            self.config['TRAIN']['BATCH_SIZE'] = args.batch_size
            
        # Update epochs
        if hasattr(args, 'epochs') and args.epochs:
            self.config['TRAIN']['EPOCHS'] = args.epochs
            
        # Update other common arguments
        if hasattr(args, 'workers') and args.workers:
            self.config['TRAIN']['WORKERS'] = args.workers
            
        if hasattr(args, 'resume') and args.resume:
            self.config['TRAIN']['RESUME'] = args.resume
            
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary-style access."""
        return self.config[key]
        
    def __getattr__(self, key: str) -> Any:
        """Get configuration value using attribute-style access."""
        return self.config.get(key)
        
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config, default_flow_style=False)
        
    def save(self, save_path: str):
        """Save configuration to YAML file."""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
def get_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Get configuration object.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Config object
    """
    # Create config object
    config_file = args.config if args and hasattr(args, 'config') else None
    cfg = Config(config_file)
    
    # Load model-specific config
    if args and hasattr(args, 'model') and args.model:
        cfg.load_model_config(args.model)
    elif 'MODEL' in cfg.config and 'NAME' in cfg.config['MODEL']:
        cfg.load_model_config(cfg.config['MODEL']['NAME'])
        
    # Update from command-line arguments
    if args:
        cfg.update_from_args(args)
        
    return cfg