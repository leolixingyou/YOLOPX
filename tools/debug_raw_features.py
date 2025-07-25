#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lib.models import get_net_from_yaml
from lib.config import cfg_xy as cfg

def debug_raw_features():
    # Load model
    model = get_net_from_yaml('/workspace/YOLOPX/lib/config/yolopx.yaml')
    model.eval()
    
    # Create smaller input for debugging
    batch_size = 6  # Same as the error
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        outputs = model(input_tensor)
        det_out, da_seg_out, ll_seg_out = outputs[:3]
        
        print(f"\nDetection output structure:")
        if isinstance(det_out, (list, tuple)):
            print(f"Type: {type(det_out)}, Length: {len(det_out)}")
            
            # Check processed detections
            if len(det_out) > 0 and hasattr(det_out[0], 'shape'):
                print(f"Processed detections: {det_out[0].shape}")
            
            # Check raw features
            if len(det_out) > 1:
                raw_features = det_out[1]
                print(f"Raw features type: {type(raw_features)}")
                
                if isinstance(raw_features, (list, tuple)):
                    print(f"Raw features list length: {len(raw_features)}")
                    for i, feat in enumerate(raw_features):
                        if hasattr(feat, 'shape'):
                            print(f"  Level {i}: {feat.shape} - Total elements: {feat.numel()}")
                            
                            # Check if this matches YOLOX expected format
                            if len(feat.shape) == 4:  # [batch, channels, h, w]
                                b, c, h, w = feat.shape
                                print(f"    Batch: {b}, Channels: {c}, H: {h}, W: {w}")
                                
                                # Expected elements for view(b, n_anchors=1, n_ch=6, h, w)
                                expected_n_ch = 6
                                expected_total = b * 1 * expected_n_ch * h * w
                                actual_total = feat.numel()
                                print(f"    Expected for reshape: {expected_total}, Actual: {actual_total}")
                                print(f"    Channel ratio: {c / expected_n_ch}")
                else:
                    print(f"Raw features shape: {raw_features.shape}")

if __name__ == "__main__":
    debug_raw_features()