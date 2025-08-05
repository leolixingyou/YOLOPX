"""Model factory for creating different YOLOP variants."""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for different model implementations
workspace_path = Path('/workspace')
sys.path.append(str(workspace_path / 'YOLOP_v1_official'))
sys.path.append(str(workspace_path / 'YOLOP_v3_official'))
sys.path.append(str(workspace_path / 'YOLOPX'))

class ModelWrapper(nn.Module):
    """Wrapper to provide unified interface for different models.
    
    This wrapper is now minimal since model-specific wrappers handle
    the interface conversion.
    """
    
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.model_type = model_type
        
    def forward(self, x):
        """Forward pass with unified output format."""
        # Model-specific wrappers already return dict format
        outputs = self.model(x)
        
        # All wrapped models should return dict format
        if isinstance(outputs, dict):
            return outputs
        else:
            # This should not happen with proper wrappers
            raise ValueError(f"Model {self.model_type} did not return expected dict format")

class ModelFactory:
    """Factory for creating YOLOP model variants."""
    
    @staticmethod
    def create_model(cfg) -> nn.Module:
        """Create model based on configuration.
        
        Args:
            cfg: Configuration object
            
        Returns:
            Model instance wrapped for unified interface
        """
        model_name = cfg.get('MODEL.NAME', 'yolopx').lower()
        
        print(f"Creating model: {model_name}")
        
        try:
            if model_name == 'yolop_v1':
                model = ModelFactory._create_yolop_v1(cfg)
            elif model_name == 'yolop_v3':
                model = ModelFactory._create_yolop_v3(cfg)
            elif model_name == 'yolopx':
                model = ModelFactory._create_yolopx(cfg)
            elif model_name == 'yolop_v2':
                model = ModelFactory._create_yolop_v2(cfg)
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Wrap model for unified interface
            return ModelWrapper(model, model_name)
            
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
            print("Creating placeholder model for demonstration")
            return ModelFactory._create_placeholder(cfg)
            
    @staticmethod
    def _create_yolop_v1(cfg) -> nn.Module:
        """Create YOLOP v1 model."""
        from .yolop_v1_wrapper import create_yolop_v1_model
        return create_yolop_v1_model(cfg)
            
    @staticmethod
    def _create_yolop_v3(cfg) -> nn.Module:
        """Create YOLOP v3 model."""
        from .yolop_v3_wrapper import create_yolop_v3_model
        return create_yolop_v3_model(cfg)
            
    @staticmethod
    def _create_yolopx(cfg) -> nn.Module:
        """Create YOLOPX model."""
        try:
            # Use the dedicated YOLOPx wrapper
            from .yolopx_wrapper import create_yolopx_model
            return create_yolopx_model(cfg)
            
        except Exception as e:
            print(f"Error creating YOLOPx: {e}")
            raise NotImplementedError("YOLOPX model creation needs full implementation")
            
    @staticmethod
    def _create_yolop_v2(cfg) -> nn.Module:
        """Create YOLOPv2 model."""
        try:
            # Import our YOLOPv2 implementation
            from .yolop_v2 import yolop_v2
            
            # Create model
            model = yolop_v2(cfg)
            
            # Load pretrained weights if available
            model_path = cfg.get('MODEL.PRETRAINED', '')
            if model_path and os.path.exists(model_path):
                print(f"Loading YOLOPv2 weights from: {model_path}")
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            
            return model
            
        except Exception as e:
            print(f"Error creating YOLOPv2: {e}")
            raise NotImplementedError("YOLOPv2 model creation failed")
            
    @staticmethod
    def _create_placeholder(cfg) -> nn.Module:
        """Create a simple placeholder model for testing."""
        
        class PlaceholderModel(nn.Module):
            def __init__(self, num_classes=1):
                super().__init__()
                # Simple backbone
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1),
                    
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(128, 256, 3, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                
                # Detection head (simplified)
                self.det_head = nn.Conv2d(256, 5 + num_classes, 1)  # bbox + obj + classes
                
                # Segmentation heads
                self.da_seg_head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 2, 1)  # 2 classes for driving area
                )
                
                self.ll_seg_head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 2, 1)  # 2 classes for lane line
                )
                
            def forward(self, x):
                # Extract features
                feat = self.backbone(x)
                
                # Detection output (return as tensor, not list)
                det = self.det_head(feat)
                
                # Segmentation outputs
                da_seg = self.da_seg_head(feat)
                ll_seg = self.ll_seg_head(feat)
                
                # Return outputs in consistent format
                return {
                    'detection': det,  # Single tensor for placeholder
                    'da_seg': da_seg,
                    'll_seg': ll_seg
                }
                
        return PlaceholderModel(num_classes=cfg.get('DATASET.NUM_CLASSES', 1))