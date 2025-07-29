"""
Visualization utility functions for the YOLOPX project.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _denormalize_img_tensor(img_tensor):
    """De-normalize a tensor image for visualization."""
    img = img_tensor.cpu().float().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def _overlay_mask(image_np, mask_np, color, alpha=0.4):
    """Overlay a segmentation mask on an image."""
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np.astype(np.uint8), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    colored_mask[mask_np == 1] = color
    return cv2.addWeighted(image_np, 1, colored_mask, alpha, 0)

def _get_class_colors():
    """Returns a dict of predefined colors for different object classes."""
    return {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
        4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 128), 7: (255, 165, 0),
    }

def _draw_boxes(img, boxes, class_indices, names, is_gt=False):
    """Draw bounding boxes on an image."""
    class_colors = _get_class_colors()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        class_idx = int(class_indices[i])
        color = class_colors.get(class_idx, (128, 128, 128))
        
        if is_gt:
            # Draw GT boxes as semi-transparent filled rectangles
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        else:
            # Draw predicted boxes as outlines
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def _add_detection_legend(ax, pred_classes, gt_classes, names):
    """Adds a legend for detection classes to a matplotlib axis."""
    # Implementation details...
    pass

def _add_segmentation_legend(ax):
    """Adds a GT/Pred legend for segmentation to a matplotlib axis."""
    # Implementation details...
    pass
