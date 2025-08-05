"""YOLOPx-specific model wrapper to handle its unique output format."""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Tuple, List

class YOLOPXWrapper(nn.Module):
    """Wrapper for YOLOPx model to ensure unified output format.
    
    YOLOPx output structure (from debug):
    - outputs[0]: tuple (detection outputs)
    - outputs[1]: tensor shape [B, 2, 640, 640] (da_seg)
    - outputs[2]: tensor shape [B, 2, 640, 640] (ll_seg)
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with unified output format.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary with keys 'detection', 'da_seg', 'll_seg'
        """
        # Get raw outputs from YOLOPx
        outputs = self.model(x)
        
        # Convert to unified format based on observed structure
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            # Handle detection output (first element is a tuple)
            det_output = outputs[0]
            if isinstance(det_output, tuple):
                # YOLOPx may return multiple detection scales
                # For now, use the first scale or combine them
                if len(det_output) > 0 and isinstance(det_output[0], torch.Tensor):
                    detection = det_output[0]
                else:
                    detection = None
            else:
                detection = det_output
                
            # Segmentation outputs are straightforward
            da_seg = outputs[1] if len(outputs) > 1 else None
            ll_seg = outputs[2] if len(outputs) > 2 else None
            
            return {
                'detection': detection,
                'da_seg': da_seg,
                'll_seg': ll_seg
            }
        else:
            # Fallback for unexpected format
            print(f"Warning: Unexpected YOLOPx output format: {type(outputs)}")
            return {
                'detection': outputs if isinstance(outputs, torch.Tensor) else None,
                'da_seg': None,
                'll_seg': None
            }

def create_yolopx_model(cfg):
    """Create YOLOPx model with proper wrapper.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Wrapped YOLOPx model with unified interface
    """
    import sys
    from pathlib import Path
    
    # Add YOLOPX path
    workspace_path = Path('/workspace')
    sys.path.append(str(workspace_path / 'YOLOPX'))
    
    try:
        from lib.config import cfg as px_cfg
        from lib.models.YOLOP import get_net
        
        # Update config
        px_cfg.defrost()
        px_cfg.MODEL.IMAGE_SIZE = cfg.get('MODEL.IMAGE_SIZE', [640, 640])
        px_cfg.freeze()
        
        # Create base model
        base_model = get_net(px_cfg)
        
        # Wrap with our unified interface
        return YOLOPXWrapper(base_model)
        
    except Exception as e:
        print(f"Error creating YOLOPx model: {e}")
        raise