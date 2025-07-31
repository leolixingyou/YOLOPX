"""
改进的损失函数实现
优化了多任务损失权重平衡
"""

import torch
import torch.nn as nn


class ImprovedMultiHeadLoss(nn.Module):
    """改进的多任务损失函数"""
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        
        self.loss_list = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg
        
        # 改进的损失权重 - 更平衡的设计
        self.det_weight = 0.05      # 降低检测权重
        self.da_seg_weight = 0.3    # 适中的驾驶区域权重
        self.ll_seg_weight = 0.4    # 提高车道线权重
        self.ll_iou_weight = 0.3    # 提高IoU权重
        
    def forward(self, head_fields, head_targets, shapes, model, imgs):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model, imgs)
        return total_loss, head_losses
        
    def _forward_impl(self, predictions, targets, shapes, model, imgs):
        cfg = self.cfg
        device = targets[0].device
        Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss = self.loss_list
        
        # 计算各项损失
        det_all_loss = Det_loss(predictions[0], targets[0], imgs)
        
        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        da_seg_loss = Da_Seg_Loss(drive_area_seg_predicts, drive_area_seg_targets)
        
        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        ll_seg_loss = Ll_Seg_Loss(lane_line_seg_predicts, lane_line_seg_targets)
        
        tversky_predicts = predictions[2]
        tversky_targets = targets[2]
        ll_tversky_loss = Tversky_Loss(tversky_predicts, tversky_targets)
        
        # 应用改进的权重
        det_all_loss *= self.det_weight * self.lambdas[1]
        da_seg_loss *= self.da_seg_weight * self.lambdas[2]
        ll_seg_loss *= self.ll_seg_weight * self.lambdas[3]
        ll_tversky_loss *= self.ll_iou_weight * self.lambdas[4]
        
        loss = det_all_loss + da_seg_loss + ll_seg_loss + ll_tversky_loss
        return loss, (det_all_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, loss)

def get_improved_loss(cfg, device, model):
    """获取改进的损失函数"""
    from lib.core.loss import YOLOX_Loss, TverskyLoss, FocalLossSeg
    
    Det_loss = YOLOX_Loss(device, 1)
    Da_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Ll_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Tversky_Loss = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3).to(device)
    
    gamma = cfg.LOSS.FL_GAMMA
    if gamma > 0.0:
        Ll_Seg_Loss = FocalLossSeg(Ll_Seg_Loss, gamma)
    
    losses = [Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss]
    return ImprovedMultiHeadLoss(losses, cfg)

