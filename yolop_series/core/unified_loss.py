"""
统一的损失函数模块
整合所有损失函数，提供统一接口
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation tasks"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        true_pos = (pred * target).sum()
        false_neg = ((1 - pred) * target).sum()
        false_pos = (pred * (1 - target)).sum()
        
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky_index


class IoULoss(nn.Module):
    """IoU Loss for bounding box regression"""
    def __init__(self, reduction='mean', loss_type='ciou'):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        pred/target: [N, 4] (x1, y1, x2, y2)
        """
        # Calculate IoU
        b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + 1e-16
        
        iou = inter / union
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        elif self.loss_type == 'ciou':
            # Complete IoU
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw ** 2 + ch ** 2 + 1e-16
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + 1e-16))
            loss = 1 - iou + (rho2 / c2 + v * alpha)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """统一的多任务损失函数"""
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # 任务权重
        self.det_weight = cfg.LOSS.get('DET_WEIGHT', 1.0)
        self.da_weight = cfg.LOSS.get('DA_SEG_GAIN', 0.2)
        self.ll_weight = cfg.LOSS.get('LL_SEG_GAIN', 0.2)
        
        # 检测损失
        self.use_focal = cfg.LOSS.get('USE_FOCAL', True)
        if self.use_focal:
            self.cls_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.iou_loss = IoULoss(loss_type='ciou')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 分割损失
        self.seg_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.ll_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, outputs, targets, model=None):
        """
        outputs: 模型输出 [det_output, da_seg_output, ll_seg_output]
        targets: 标签 [det_targets, da_seg_targets, ll_seg_targets]
        """
        det_out, da_seg_out, ll_seg_out = outputs
        det_target, da_seg_target, ll_seg_target = targets
        
        # 检测损失
        det_loss = self._compute_detection_loss(det_out, det_target)
        
        # 驾驶区域分割损失
        da_seg_loss = self.seg_loss(da_seg_out, da_seg_target)
        
        # 车道线分割损失
        ll_seg_loss = self.ll_loss(ll_seg_out, ll_seg_target)
        
        # 总损失
        total_loss = (self.det_weight * det_loss + 
                     self.da_weight * da_seg_loss + 
                     self.ll_weight * ll_seg_loss)
        
        # 返回总损失和各任务损失
        task_losses = {
            'det_loss': det_loss,
            'da_seg_loss': da_seg_loss,
            'll_seg_loss': ll_seg_loss
        }
        
        return total_loss, task_losses

    def _compute_detection_loss(self, pred, target):
        """计算检测损失"""
        # 这里简化处理，实际需要根据具体的检测头输出格式来计算
        # 包括分类损失、边界框回归损失和目标性损失
        device = pred[0].device if isinstance(pred, list) else pred.device
        
        # 占位符损失，实际使用时需要替换为真实的检测损失计算
        det_loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return det_loss


def get_loss(cfg, device='cuda'):
    """损失函数工厂函数"""
    return MultiTaskLoss(cfg, device)