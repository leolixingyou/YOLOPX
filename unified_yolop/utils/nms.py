"""Non-Maximum Suppression utilities for YOLOP models."""

import torch
import numpy as np

def box_iou(box1, box2):
    """Calculate IoU between two sets of boxes."""
    # box1: [N, 4] (x1, y1, x2, y2)
    # box2: [M, 4] (x1, y1, x2, y2)
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Intersection
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / union
    return iou

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Perform non-maximum suppression on predictions.
    
    Args:
        prediction: Model output tensor [batch, anchors, grid_y, grid_x, attrs]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        
    Returns:
        List of detections for each image
    """
    # Handle different input formats
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
        
    if prediction is None:
        return [torch.zeros((0, 6))]
        
    # Convert to expected shape if needed
    if len(prediction.shape) == 5:
        # [batch, anchors, gy, gx, attrs] -> [batch, n_boxes, attrs]
        bs, na, gy, gx, no = prediction.shape
        prediction = prediction.view(bs, na * gy * gx, no)
        
    batch_size = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * batch_size
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Get scores and classes
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        # Apply NMS
        c = x[:, 5:6] * 4096  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision_nms(boxes, scores, iou_thres)
        
        output[xi] = x[i]
        
    return output

def xywh2xyxy(x):
    """Convert bounding boxes from (center x, center y, width, height) to (x1, y1, x2, y2)."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def torchvision_nms(boxes, scores, iou_threshold):
    """Wrapper for torchvision.ops.nms."""
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except:
        # Fallback implementation
        keep = []
        idxs = scores.argsort(descending=True)
        
        while idxs.numel() > 0:
            # Pick the box with highest score
            idx = idxs[0]
            keep.append(idx)
            
            if idxs.numel() == 1:
                break
                
            # Compute IoU with remaining boxes
            ious = box_iou(boxes[idx].unsqueeze(0), boxes[idxs[1:]])[0]
            
            # Remove boxes with IoU > threshold
            idxs = idxs[1:][ious <= iou_threshold]
            
        return torch.tensor(keep, device=boxes.device)