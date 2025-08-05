"""YOLOPv1-specific model wrapper to handle its unique output format and configuration."""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Union

class YOLOPv1Wrapper(nn.Module):
    """Wrapper for YOLOPv1 model to ensure unified output format.
    
    YOLOPv1 specific characteristics:
    - Anchor-based detection with 3 scales
    - Applies sigmoid activation to segmentation outputs in forward pass
    - Returns list format: [detection_list, da_seg, ll_seg]
    """
    
    def __init__(self, cfg):
        super().__init__()
        # Save current sys.path
        original_path = sys.path.copy()
        
        try:
            # Temporarily add YOLOPv1 path
            workspace_path = Path('/workspace')
            v1_path = str(workspace_path / 'YOLOP_v1_official')
            sys.path.insert(0, v1_path)
            
            # Import YOLOPv1 specific modules
            from lib.config import cfg as v1_cfg
            from lib.models.YOLOP import get_net
            
            # Update v1 config with our settings
            v1_cfg.defrost()
            v1_cfg.MODEL.IMAGE_SIZE = cfg.get('MODEL.IMAGE_SIZE', [640, 640])
            v1_cfg.MODEL.ANCHORS = cfg.get('MODEL.ANCHORS', 
                [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]])
            v1_cfg.MODEL.NC = cfg.get('DATASET.NUM_CLASSES', 1)
            v1_cfg.freeze()
            
            # Create model
            self.model = get_net(v1_cfg)
            self.sigmoid = nn.Sigmoid()
            
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
        # Get raw outputs from YOLOPv1
        outputs = self.model(x)
        
        # YOLOPv1 returns list: [detection_list, da_seg, ll_seg]
        if isinstance(outputs, list) and len(outputs) == 3:
            det_output = outputs[0]
            da_seg = outputs[1]
            ll_seg = outputs[2]
            
            # YOLOPv1 applies sigmoid in model.py, but double-check
            # In v1, segmentation outputs already have sigmoid applied
            
            return {
                'detection': det_output,  # Keep as list for multi-scale
                'da_seg': da_seg,
                'll_seg': ll_seg
            }
        else:
            raise ValueError(f"Unexpected YOLOPv1 output format: {type(outputs)}")

def create_yolop_v1_model(cfg):
    """Create YOLOPv1 model with proper wrapper.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Wrapped YOLOPv1 model with unified interface
    """
    return YOLOPv1Wrapper(cfg)