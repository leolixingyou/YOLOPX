"""YOLOPv3-specific model wrapper to handle its unique output format and configuration."""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Union

class YOLOPv3Wrapper(nn.Module):
    """Wrapper for YOLOPv3 model to ensure unified output format.
    
    YOLOPv3 specific characteristics:
    - Improved anchor-based detection with 4 scales
    - No sigmoid activation in forward pass (applied in post-processing)
    - Returns list format: [detection_list, da_seg, ll_seg]
    - Designed for remote sensing with larger input sizes
    """
    
    def __init__(self, cfg):
        super().__init__()
        # Save current sys.path
        original_path = sys.path.copy()
        
        try:
            # Temporarily add YOLOPv3 path
            workspace_path = Path('/workspace')
            v3_path = str(workspace_path / 'YOLOP_v3_official')
            sys.path.insert(0, v3_path)
            
            # Import YOLOPv3 specific modules
            from lib.config import cfg as v3_cfg
            from lib.models.YOLOP import get_net
            
            # Update v3 config with our settings
            v3_cfg.defrost()
            # YOLOPv3 is designed for remote sensing, might use larger sizes
            v3_cfg.MODEL.IMAGE_SIZE = cfg.get('MODEL.IMAGE_SIZE', [640, 640])
            v3_cfg.MODEL.NC = cfg.get('DATASET.NUM_CLASSES', 1)
            # YOLOPv3 uses 4-scale detection
            if hasattr(v3_cfg.MODEL, 'ANCHORS'):
                v3_cfg.MODEL.ANCHORS = cfg.get('MODEL.ANCHORS', v3_cfg.MODEL.ANCHORS)
            v3_cfg.freeze()
            
            # Create model
            self.model = get_net(v3_cfg)
            
        finally:
            # Restore original sys.path to avoid conflicts
            sys.path = original_path
    
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass with unified output format.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary with keys 'detection', 'da_seg', 'll_seg'
        """
        # Get raw outputs from YOLOPv3
        outputs = self.model(x)
        
        # YOLOPv3 returns list: [detection_list, da_seg, ll_seg]
        if isinstance(outputs, list) and len(outputs) == 3:
            det_output = outputs[0]
            da_seg = outputs[1]
            ll_seg = outputs[2]
            
            # YOLOPv3 does NOT apply sigmoid in forward pass
            # Activation is handled in loss computation
            
            return {
                'detection': det_output,  # Keep as list for multi-scale (4 scales in v3)
                'da_seg': da_seg,  # Raw logits
                'll_seg': ll_seg   # Raw logits
            }
        else:
            raise ValueError(f"Unexpected YOLOPv3 output format: {type(outputs)}")

def create_yolop_v3_model(cfg):
    """Create YOLOPv3 model with proper wrapper.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Wrapped YOLOPv3 model with unified interface
    """
    return YOLOPv3Wrapper(cfg)