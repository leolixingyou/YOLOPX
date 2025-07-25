#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg_xy as cfg
from lib.config import update_config_xy as update_config
from lib.models import get_net_from_yaml

def debug_training_shapes():
    # Set up model and data loader (same as xy_train_gemini.py)
    model = get_net_from_yaml('/workspace/YOLOPX/lib/config/yolopx.yaml')
    model.eval()
    
    # Create data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg=cfg, is_train=True, inputsize=cfg.MODEL.IMAGE_SIZE, transform=transform)
    train_loader = DataLoaderX(train_dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=False, collate_fn=dataset.AutoDriveDataset.collate_fn)
    
    # Get one batch
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"Input shape: {input.shape}")
        print(f"Target shapes: {[t.shape for t in target]}")
        
        with torch.no_grad():
            outputs = model(input)
            det_out, da_seg_out, ll_seg_out = outputs[:3]
            
            print(f"Detection output type: {type(det_out)}")
            if isinstance(det_out, (list, tuple)):
                print(f"Detection output length: {len(det_out)}")
                
                # Check processed output
                if len(det_out) > 0:
                    print(f"Processed output: {det_out[0].shape}")
                
                # Check raw features  
                if len(det_out) > 1:
                    raw_features = det_out[1]
                    print(f"Raw features type: {type(raw_features)}")
                    
                    if isinstance(raw_features, (list, tuple)):
                        print(f"Raw features:")
                        for j, feat in enumerate(raw_features):
                            if hasattr(feat, 'shape'):
                                print(f"  Level {j}: {feat.shape} - Elements: {feat.numel()}")
                                
                                # Check if it can be reshaped for YOLOX_Loss
                                b, c, h, w = feat.shape
                                expected_elements = b * 1 * 6 * h * w
                                actual_elements = feat.numel()
                                print(f"    Expected for view(b,1,6,h,w): {expected_elements}, Actual: {actual_elements}")
                                if expected_elements == actual_elements:
                                    print(f"    ✓ Can reshape to [{b}, 1, 6, {h}, {w}]")
                                else:
                                    print(f"    ✗ Cannot reshape: ratio = {expected_elements/actual_elements}")
        
        break  # Only check first batch

if __name__ == "__main__":
    debug_training_shapes()