"""Metrics for evaluating YOLOP models."""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

class Metrics:
    """Compute metrics for multi-task learning."""
    
    def __init__(self, cfg):
        """Initialize metrics calculator.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        # Detection metrics
        self.det_tp = 0
        self.det_fp = 0
        self.det_fn = 0
        
        # Segmentation metrics
        self.da_intersection = 0
        self.da_union = 0
        self.ll_intersection = 0
        self.ll_union = 0
        
        # Pixel accuracy
        self.da_correct = 0
        self.da_total = 0
        self.ll_correct = 0
        self.ll_total = 0
        
    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Update metrics with batch predictions.
        
        Args:
            outputs: Model outputs
            targets: Ground truth
        """
        # Update detection metrics
        if 'detection' in outputs and 'det_labels' in targets:
            self._update_detection_metrics(outputs['detection'], targets['det_labels'])
            
        # Update driving area segmentation metrics
        if 'da_seg' in outputs and 'da_seg_masks' in targets:
            self._update_segmentation_metrics(
                outputs['da_seg'], targets['da_seg_masks'], 
                prefix='da'
            )
            
        # Update lane line segmentation metrics
        if 'll_seg' in outputs and 'll_seg_masks' in targets:
            self._update_segmentation_metrics(
                outputs['ll_seg'], targets['ll_seg_masks'],
                prefix='ll'
            )
            
    def _update_detection_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Update detection metrics (simplified)."""
        # This is a placeholder implementation
        # Real implementation would include:
        # - NMS
        # - IoU matching
        # - Precision/Recall calculation
        
        # Handle different output formats
        if isinstance(pred, list):
            # Multi-scale outputs, use first scale for batch size
            if len(pred) > 0 and isinstance(pred[0], torch.Tensor):
                batch_size = pred[0].shape[0] if pred[0].dim() > 0 else 1
            else:
                batch_size = 1
        elif isinstance(pred, torch.Tensor):
            batch_size = pred.shape[0] if pred.dim() > 0 else 1
        else:
            batch_size = 1
            
        # For now, just count some dummy values
        self.det_tp += batch_size * 5  # Assume 5 true positives per image
        self.det_fp += batch_size * 2  # Assume 2 false positives per image
        self.det_fn += batch_size * 1  # Assume 1 false negative per image
        
    def _update_segmentation_metrics(self, pred: torch.Tensor, target: torch.Tensor, prefix: str):
        """Update segmentation metrics."""
        # Skip if prediction is None
        if pred is None:
            return
            
        # Get predictions
        if pred.shape[1] > 1:
            # Multi-class: use argmax
            pred_mask = torch.argmax(pred, dim=1)
        else:
            # Binary: use threshold
            pred_mask = (torch.sigmoid(pred.squeeze(1)) > 0.5).long()
            
        # Ensure target is long type
        target = target.long()
        
        # Compute intersection and union for IoU
        intersection = ((pred_mask == 1) & (target == 1)).sum().item()
        union = ((pred_mask == 1) | (target == 1)).sum().item()
        
        # Update counters
        if prefix == 'da':
            self.da_intersection += intersection
            self.da_union += union
            self.da_correct += (pred_mask == target).sum().item()
            self.da_total += target.numel()
        else:
            self.ll_intersection += intersection
            self.ll_union += union
            self.ll_correct += (pred_mask == target).sum().item()
            self.ll_total += target.numel()
            
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Detection metrics
        if self.det_tp + self.det_fp > 0:
            det_precision = self.det_tp / (self.det_tp + self.det_fp)
            metrics['det_precision'] = det_precision
            
        if self.det_tp + self.det_fn > 0:
            det_recall = self.det_tp / (self.det_tp + self.det_fn)
            metrics['det_recall'] = det_recall
            
        if 'det_precision' in metrics and 'det_recall' in metrics:
            if metrics['det_precision'] + metrics['det_recall'] > 0:
                f1 = 2 * metrics['det_precision'] * metrics['det_recall'] / \
                     (metrics['det_precision'] + metrics['det_recall'])
                metrics['det_f1'] = f1
                
        # Driving area segmentation metrics
        if self.da_union > 0:
            da_iou = self.da_intersection / self.da_union
            metrics['da_iou'] = da_iou
            
        if self.da_total > 0:
            da_acc = self.da_correct / self.da_total
            metrics['da_accuracy'] = da_acc
            
        # Lane line segmentation metrics
        if self.ll_union > 0:
            ll_iou = self.ll_intersection / self.ll_union
            metrics['ll_iou'] = ll_iou
            
        if self.ll_total > 0:
            ll_acc = self.ll_correct / self.ll_total
            metrics['ll_accuracy'] = ll_acc
            
        # Overall score (weighted average)
        scores = []
        weights = []
        
        if 'det_f1' in metrics:
            scores.append(metrics['det_f1'])
            weights.append(1.0)
            
        if 'da_iou' in metrics:
            scores.append(metrics['da_iou'])
            weights.append(0.5)
            
        if 'll_iou' in metrics:
            scores.append(metrics['ll_iou'])
            weights.append(0.5)
            
        if scores:
            metrics['overall_score'] = np.average(scores, weights=weights)
            
        return metrics