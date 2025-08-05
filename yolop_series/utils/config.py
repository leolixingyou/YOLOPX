"""Configuration management system for YOLOP series models."""

import os
import yaml
from easydict import EasyDict as edict

class Config:
    """Configuration class for managing YOLOP series configs."""
    
    def __init__(self, cfg_path=None):
        """Initialize configuration.
        
        Args:
            cfg_path: Path to configuration file
        """
        self.cfg = edict()
        
        # Load default configuration
        default_cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'configs', 'default.yaml')
        self._load_yaml(default_cfg_path)
        
        # Load specific configuration if provided
        if cfg_path:
            self._load_yaml(cfg_path)
            
    def _load_yaml(self, yaml_path):
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        """
        with open(yaml_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
            
        self._merge_cfg(yaml_cfg)
        
    def _merge_cfg(self, new_cfg):
        """Recursively merge new configuration into existing.
        
        Args:
            new_cfg: New configuration dictionary
        """
        def recursive_merge(base, new):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    recursive_merge(base[key], value)
                else:
                    base[key] = value
                    
        recursive_merge(self.cfg, new_cfg)
        
    def merge_from_list(self, cfg_list):
        """Merge configuration from list of key-value pairs.
        
        Args:
            cfg_list: List of configuration options ['key1', 'value1', 'key2', 'value2', ...]
        """
        assert len(cfg_list) % 2 == 0, "Config list must have even number of elements"
        
        for i in range(0, len(cfg_list), 2):
            key_list = cfg_list[i].split('.')
            value = cfg_list[i + 1]
            
            # Navigate to the correct position in config
            d = self.cfg
            for k in key_list[:-1]:
                if k not in d:
                    d[k] = edict()
                d = d[k]
            d[key_list[-1]] = value
            
    def __getattr__(self, name):
        """Allow attribute-style access to configuration."""
        return self.cfg[name]
        
    def __repr__(self):
        """String representation of configuration."""
        return yaml.dump(dict(self.cfg), default_flow_style=False)