"""Improved visualization utilities for YOLOP validation results."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

def draw_detection_boxes(img: np.ndarray, detections: torch.Tensor, 
                        conf_thresh: float = 0.5) -> np.ndarray:
    """Draw detection bounding boxes on image.
    
    Args:
        img: Input image (H, W, 3)
        detections: Detection output tensor
        conf_thresh: Confidence threshold
        
    Returns:
        Image with drawn boxes
    """
    if detections is None:
        return img
        
    # Handle different detection formats
    if isinstance(detections, (list, tuple)):
        # Multi-scale detection outputs
        detections = detections[0] if len(detections) > 0 else None
        
    if detections is None:
        return img
        
    # Convert to numpy if needed
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
        
    # Process detections based on format
    # YOLO format: [batch, anchors, grid_y, grid_x, (x, y, w, h, conf, classes...)]
    if len(detections.shape) == 5:
        # Reshape to [N, pred_per_anchor]
        batch_size, anchors, gy, gx, pred_size = detections.shape
        detections = detections.reshape(batch_size, -1, pred_size)
        
    # Now detections should be [batch, N, pred_size] or [N, pred_size]
    if len(detections.shape) == 3:
        # Take first image in batch
        detections = detections[0]
        
    # Filter by confidence
    if detections.shape[-1] >= 5:  # Has confidence score
        conf_mask = detections[:, 4] > conf_thresh
        detections = detections[conf_mask]
        
    # Draw boxes
    img_h, img_w = img.shape[:2]
    for det in detections:
        if len(det) < 4:
            continue
            
        # Get box coordinates (assuming normalized coordinates)
        x_center, y_center, w, h = det[:4]
        
        # Convert to pixel coordinates
        x1 = int((x_center - w/2) * img_w)
        y1 = int((y_center - h/2) * img_h)
        x2 = int((x_center + w/2) * img_w)
        y2 = int((y_center + h/2) * img_h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_w-1))
        y1 = max(0, min(y1, img_h-1))
        x2 = max(0, min(x2, img_w-1))
        y2 = max(0, min(y2, img_h-1))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence if available
        if len(det) >= 5:
            conf = det[4]
            label = f'{conf:.2f}'
            cv2.putText(img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
    return img

def save_validation_images_improved(images: torch.Tensor, 
                                  outputs: Dict[str, torch.Tensor],
                                  targets: Dict[str, torch.Tensor], 
                                  img_info: List[Dict],
                                  save_dir: Path, 
                                  batch_idx: int):
    """Save validation images with all task predictions overlaid.
    
    Args:
        images: Input images tensor [B, 3, H, W]
        outputs: Model outputs dict with 'detection', 'da_seg', 'll_seg'
        targets: Ground truth targets (optional)
        img_info: Image metadata
        save_dir: Directory to save images
        batch_idx: Batch index for naming
    """
    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert images to numpy
    batch_size = images.shape[0]
    
    # Denormalize images if needed (assuming standard ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    
    # Ensure values are in [0, 1]
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy [B, H, W, 3]
    imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    imgs_np = (imgs_np * 255).astype(np.uint8)
    
    # Process each image
    for i in range(batch_size):
        # Get original image
        img = imgs_np[i].copy()
        
        # Create overlay for segmentation masks
        overlay = img.copy()
        
        # Draw drivable area segmentation
        if 'da_seg' in outputs and outputs['da_seg'] is not None:
            da_pred = torch.argmax(outputs['da_seg'][i], dim=0).cpu().numpy()
            # Create green mask for drivable area
            da_mask = np.zeros_like(overlay)
            da_mask[da_pred == 1] = [0, 255, 0]  # Green
            # Apply with transparency
            mask_indices = da_pred == 1
            overlay[mask_indices] = cv2.addWeighted(
                img[mask_indices], 0.5, da_mask[mask_indices], 0.5, 0
            )
            
        # Draw lane line segmentation
        if 'll_seg' in outputs and outputs['ll_seg'] is not None:
            ll_pred = torch.argmax(outputs['ll_seg'][i], dim=0).cpu().numpy()
            # Create red mask for lane lines
            ll_mask = np.zeros_like(overlay)
            ll_mask[ll_pred == 1] = [255, 0, 0]  # Red
            # Apply with higher opacity for lane lines
            mask_indices = ll_pred == 1
            overlay[mask_indices] = cv2.addWeighted(
                overlay[mask_indices], 0.3, ll_mask[mask_indices], 0.7, 0
            )
            
        # Draw detection boxes
        if 'detection' in outputs and outputs['detection'] is not None:
            # Extract detection for current image
            det_output = outputs['detection']
            if isinstance(det_output, torch.Tensor) and len(det_output.shape) >= 2:
                if det_output.shape[0] == batch_size:
                    # Has batch dimension
                    overlay = draw_detection_boxes(overlay, det_output[i:i+1])
                else:
                    # Single image detection
                    overlay = draw_detection_boxes(overlay, det_output)
                    
        # Save the final image
        save_path = save_dir / f'batch{batch_idx}_img{i}.jpg'
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Optionally save ground truth comparison
        if targets and i == 0:  # Save GT for first image only
            gt_img = img.copy()
            
            # Draw GT drivable area
            if 'da_seg_masks' in targets:
                da_gt = targets['da_seg_masks'][i].cpu().numpy()
                gt_mask = np.zeros_like(gt_img)
                gt_mask[da_gt == 1] = [0, 128, 0]  # Darker green
                gt_img = cv2.addWeighted(gt_img, 0.7, gt_mask, 0.3, 0)
                
            # Draw GT lane lines
            if 'll_seg_masks' in targets:
                ll_gt = targets['ll_seg_masks'][i].cpu().numpy()
                ll_mask = np.zeros_like(gt_img)
                ll_mask[ll_gt == 1] = [128, 0, 0]  # Darker red
                gt_img = cv2.addWeighted(gt_img, 0.7, ll_mask, 0.3, 0)
                
            gt_save_path = save_dir / f'batch{batch_idx}_img{i}_gt.jpg'
            cv2.imwrite(str(gt_save_path), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))