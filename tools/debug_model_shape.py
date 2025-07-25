#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lib.models import get_net_from_yaml
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config

def debug_model_shapes():
    # Update config 
    cfg.MODEL.CONFIG = '/workspace/YOLOPX/lib/config/yolopx.yaml'
    
    # Load model
    model = get_net_from_yaml(cfg.MODEL.CONFIG)
    model.eval()
    
    # Create dummy input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        # Get model output
        outputs = model(input_tensor)
        
        print(f"Number of outputs: {len(outputs)}")
        
        if len(outputs) >= 3:
            det_out, da_seg_out, ll_seg_out = outputs[:3]
            
            if isinstance(det_out, torch.Tensor):
                print(f"Detection output shape: {det_out.shape}")
            else:
                print(f"Detection output shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in det_out]}")
            print(f"Drivable area output shape: {da_seg_out.shape}")  
            print(f"Lane line output shape: {ll_seg_out.shape}")
            
            # Check detection output structure
            if isinstance(det_out, (list, tuple)):
                print("Detection output is a list/tuple:")
                for i, x in enumerate(det_out):
                    if hasattr(x, 'shape'):
                        print(f"  Level {i}: {x.shape} - Total elements: {x.numel()}")
                        
                        # This is the correctly formatted output [batch, anchors, channels]
                        if len(x.shape) == 3:
                            batch, anchors, channels = x.shape
                            print(f"    Batch: {batch}, Anchors: {anchors}, Channels: {channels}")
                            
                            # Check if this matches expected YOLOX format
                            # YOLOX expects multiple levels, each with shape [batch, channels, h, w]
                            # But we have [batch, anchors, channels] which is already processed
                            print(f"    This appears to be already processed detection output")
                    else:
                        print(f"  Level {i}: {type(x)} - {x}")
                        
                # Let's also examine what YOLOX_Loss expects
                print(f"\nYOLOX_Loss expects multiple levels with shapes like [batch, channels, h, w]")
                print(f"But we have processed output [batch, {det_out[0].shape[1]}, {det_out[0].shape[2]}]")
                
            else:
                print(f"Detection output is a tensor: {det_out.shape}")

if __name__ == "__main__":
    debug_model_shapes()