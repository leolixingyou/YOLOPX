"""Model factory for creating different YOLOP variants."""

import torch
import torch.nn as nn
import sys
import os

# Add paths for different model implementations
sys.path.append('/workspace/YOLOP_v1_official')
sys.path.append('/workspace/YOLOP_v3_official') 
sys.path.append('/workspace/YOLOPX')

class ModelFactory:
    """Factory class for creating YOLOP series models."""
    
    @staticmethod
    def create_model(cfg):
        """Create model based on configuration.
        
        Args:
            cfg: Configuration object
            
        Returns:
            model: PyTorch model instance
        """
        model_type = cfg.MODEL.TYPE.lower()
        
        if model_type == 'yolop_v1':
            return ModelFactory._create_yolop_v1(cfg)
        elif model_type == 'yolop_v3':
            return ModelFactory._create_yolop_v3(cfg)
        elif model_type == 'yolopx':
            return ModelFactory._create_yolopx(cfg)
        elif model_type == 'yolop_v2':
            # YOLOPv2 only has inference model
            raise NotImplementedError("YOLOPv2 training not supported (inference only)")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    @staticmethod
    def _create_yolop_v1(cfg):
        """Create YOLOP v1 model."""
        # Import v1 specific modules
        from lib.models.YOLOP import MCnet_SPP, AutoDriveModel
        from lib.config import default as v1_default
        from lib.config import cfg as v1_cfg
        
        # Override v1 config with our unified config
        v1_cfg.merge_from_other_cfg(v1_default._C)
        
        # Create model
        model = AutoDriveModel(cfg=v1_cfg, ch=3, nc=cfg.DATASET.NUM_CLASSES, 
                              anchors=cfg.MODEL.ANCHORS)
        return model
        
    @staticmethod
    def _create_yolop_v3(cfg):
        """Create YOLOP v3 model."""
        # Import v3 specific modules
        from lib.models.YOLOP import YOLOP, AutoDriveModel
        from lib.config import default as v3_default
        from lib.config import cfg as v3_cfg
        
        # Override v3 config
        v3_cfg.merge_from_other_cfg(v3_default._C)
        
        # Create model
        model = AutoDriveModel(cfg=v3_cfg, ch=3, nc=cfg.DATASET.NUM_CLASSES,
                              anchors=cfg.MODEL.ANCHORS)
        return model
        
    @staticmethod
    def _create_yolopx(cfg):
        """Create YOLOPX model."""
        # Import YOLOPX specific modules
        from lib.models.YOLOP import YOLOPV1Ours, AutoDriveModel
        from lib.config import default as px_default
        from lib.config import cfg as px_cfg
        
        # Override YOLOPX config
        px_cfg.merge_from_other_cfg(px_default._C)
        
        # Create model  
        model = AutoDriveModel(cfg=px_cfg, ch=3, nc=cfg.DATASET.NUM_CLASSES)
        return model