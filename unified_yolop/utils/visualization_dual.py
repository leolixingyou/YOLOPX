"""Dual visualization showing both input and output side by side."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .nms import non_max_suppression, xywh2xyxy

def scale_coords(img1_shape, coords, img0_shape):
    """Scale coordinates from img1_shape to img0_shape."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def process_detection_output(detection: torch.Tensor, img_shape: Tuple[int, int], 
                           conf_thresh: float = 0.25) -> List[Tuple[int, int, int, int, float]]:
    """Process raw detection output to get bounding boxes.
    
    Args:
        detection: Raw detection tensor from model
        img_shape: Image shape (H, W)
        conf_thresh: Confidence threshold
        
    Returns:
        List of (x1, y1, x2, y2, conf) tuples
    """
    boxes = []
    
    if detection is None:
        return boxes
        
    # Handle different detection formats
    if isinstance(detection, (list, tuple)):
        detection = detection[0] if len(detection) > 0 else None
        
    if detection is None:
        return boxes
        
    # Convert to numpy
    if isinstance(detection, torch.Tensor):
        detection = detection.cpu().numpy()
        
    # YOLO format: [batch, anchors, grid_y, grid_x, (x, y, w, h, obj_conf, class_probs...)]
    if len(detection.shape) == 5:
        batch_size, num_anchors, grid_h, grid_w, box_attrs = detection.shape
        
        # Create grid coordinates
        grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        
        # Process each anchor
        for anchor_idx in range(num_anchors):
            anchor_preds = detection[0, anchor_idx]  # Take first batch
            
            # Extract predictions
            x_offset = anchor_preds[:, :, 0]
            y_offset = anchor_preds[:, :, 1]
            w_pred = anchor_preds[:, :, 2]
            h_pred = anchor_preds[:, :, 3]
            obj_conf = anchor_preds[:, :, 4]
            
            # Apply sigmoid to offsets and confidence
            x_offset = 1 / (1 + np.exp(-x_offset))
            y_offset = 1 / (1 + np.exp(-y_offset))
            obj_conf = 1 / (1 + np.exp(-obj_conf))
            
            # Filter by confidence
            conf_mask = obj_conf > conf_thresh
            
            if np.any(conf_mask):
                # Get valid predictions
                valid_indices = np.where(conf_mask)
                
                for idx in range(len(valid_indices[0])):
                    gy = valid_indices[0][idx]
                    gx = valid_indices[1][idx]
                    
                    # Calculate actual coordinates
                    x_center = (gx + x_offset[gy, gx]) / grid_w
                    y_center = (gy + y_offset[gy, gx]) / grid_h
                    
                    # Clip exp values to prevent overflow
                    w_exp = np.clip(w_pred[gy, gx], -10, 10)  # Prevent exp overflow
                    h_exp = np.clip(h_pred[gy, gx], -10, 10)
                    w = np.exp(w_exp) * 0.1  # Arbitrary anchor scale
                    h = np.exp(h_exp) * 0.1
                    
                    # Convert to pixel coordinates
                    x_center *= img_shape[1]
                    y_center *= img_shape[0]
                    w *= img_shape[1]
                    h *= img_shape[0]
                    
                    # Convert to corners
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, img_shape[1]-1))
                    y1 = max(0, min(y1, img_shape[0]-1))
                    x2 = max(0, min(x2, img_shape[1]-1))
                    y2 = max(0, min(y2, img_shape[0]-1))
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2, obj_conf[gy, gx]))
                        
    return boxes

def save_dual_visualization(images: torch.Tensor, 
                           outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor], 
                           save_dir: Path, 
                           batch_idx: int,
                           max_images: int = 2):
    """Save dual visualization showing input and output side by side.
    
    Args:
        images: Input images tensor [B, 3, H, W]
        outputs: Model outputs dict
        targets: Ground truth targets
        save_dir: Directory to save images
        batch_idx: Batch index
        max_images: Maximum images to save per batch
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    imgs_np = (imgs_np * 255).astype(np.uint8)
    
    batch_size = min(images.shape[0], max_images)
    
    for i in range(batch_size):
        # Original image
        img_orig = imgs_np[i].copy()
        h, w = img_orig.shape[:2]
        
        # Create output overlay
        img_pred = img_orig.copy()
        
        # Draw drivable area
        if 'da_seg' in outputs and outputs['da_seg'] is not None:
            da_pred = torch.argmax(outputs['da_seg'][i], dim=0).cpu().numpy()
            da_overlay = np.zeros_like(img_pred)
            da_overlay[da_pred == 1] = [0, 255, 0]  # Green
            mask_indices = da_pred == 1
            if np.any(mask_indices):
                img_pred[mask_indices] = cv2.addWeighted(
                    img_pred[mask_indices], 0.6, da_overlay[mask_indices], 0.4, 0
                )
            
        # Draw lane lines
        if 'll_seg' in outputs and outputs['ll_seg'] is not None:
            ll_pred = torch.argmax(outputs['ll_seg'][i], dim=0).cpu().numpy()
            ll_overlay = np.zeros_like(img_pred)
            ll_overlay[ll_pred == 1] = [255, 0, 0]  # Red
            mask_indices = ll_pred == 1
            if np.any(mask_indices):
                img_pred[mask_indices] = cv2.addWeighted(
                    img_pred[mask_indices], 0.4, ll_overlay[mask_indices], 0.6, 0
                )
            
        # Draw detection boxes
        if 'detection' in outputs and outputs['detection'] is not None:
            det_output = outputs['detection']
            # Handle tuple format (from YOLOPv1)
            if isinstance(det_output, tuple):
                det_output = det_output[0] if len(det_output) > 0 else None
            
            if det_output is not None:
                # Apply NMS to get final detections
                # Handle list format (multi-scale outputs)
                if isinstance(det_output, list):
                    # For multi-scale outputs, take the first scale
                    det_output = det_output[0] if len(det_output) > 0 else None
                
                if det_output is not None:
                    # Make sure we have the right batch
                    if hasattr(det_output, 'shape') and det_output.shape[0] > i:
                        det_batch = det_output[i:i+1]
                    else:
                        det_batch = det_output
                else:
                    det_batch = None
                
                # Apply non-maximum suppression
                det_results = non_max_suppression(det_batch, conf_thres=0.25, iou_thres=0.45)
                
                # Process first image results
                if len(det_results) > 0 and det_results[0] is not None and len(det_results[0]) > 0:
                    dets = det_results[0]  # [n_boxes, 6] format: x1,y1,x2,y2,conf,cls
                    
                    # Scale boxes to image size (from 640x640 to actual size)
                    dets[:, :4] = scale_coords((640, 640), dets[:, :4], (h, w))
                    
                    # Draw each detection
                    for det in dets:
                        x1, y1, x2, y2, conf = det[:5]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Use yellow color for bounding boxes (BGR format)
                        cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        label = f'Vehicle {conf:.2f}'
                        cv2.putText(img_pred, label, (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Create side-by-side image
        gap = 20
        combined = np.ones((h, w*2 + gap, 3), dtype=np.uint8) * 255
        
        # Place original and prediction
        combined[:, :w] = img_orig
        combined[:, w+gap:] = img_pred
        
        # Add labels
        cv2.putText(combined, "Input", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(combined, "Model Output", (w+gap+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add legend
        legend_y = h - 60
        cv2.putText(combined, "Green: Drivable Area", (w+gap+10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "Red: Lane Lines", (w+gap+10, legend_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(combined, "Yellow Box: Vehicle", (w+gap+10, legend_y+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Save combined image
        save_path = save_dir / f'batch{batch_idx}_img{i}_dual.jpg'
        cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        # Also save individual outputs for debugging
        if i == 0:  # Only for first image
            # Save input
            cv2.imwrite(
                str(save_dir / f'batch{batch_idx}_img{i}_input.jpg'),
                cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
            )
            
            # Save output
            cv2.imwrite(
                str(save_dir / f'batch{batch_idx}_img{i}_output.jpg'),
                cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            )
            
            # Save individual task outputs
            if 'da_seg' in outputs and outputs['da_seg'] is not None:
                da_vis = (da_pred * 255).astype(np.uint8)
                cv2.imwrite(
                    str(save_dir / f'batch{batch_idx}_img{i}_da_seg.jpg'),
                    da_vis
                )
                
            if 'll_seg' in outputs and outputs['ll_seg'] is not None:
                ll_vis = (ll_pred * 255).astype(np.uint8)
                cv2.imwrite(
                    str(save_dir / f'batch{batch_idx}_img{i}_ll_seg.jpg'),
                    ll_vis
                )