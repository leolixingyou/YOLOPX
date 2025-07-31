import torch.nn as nn
import torch
from models.heads.yolox_loss import YOLOX_Loss

class MultiHeadLoss(nn.Module):
    def __init__(self, losses, cfg, lambdas=None):
        super().__init__()
        if not lambdas:
            lambdas = [1.0] * 5 # det, da, ll, ll_iou
        self.loss_list = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes=None, model=None):
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, model)
        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, model):
        Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss = self.loss_list
        
        det_predictions = predictions[0]
        det_all_loss = torch.tensor(0.0, device=targets[0].device)
        
        if det_predictions is not None:
            if isinstance(det_predictions, list):
                for det_pred in det_predictions:
                    det_all_loss += Det_loss(det_pred, targets[0])
            else:
                det_all_loss = Det_loss(det_predictions, targets[0])

        da_seg_loss = Da_Seg_Loss(predictions[1], targets[1])
        ll_seg_loss = Ll_Seg_Loss(predictions[2], targets[2])
        ll_tversky_loss = Tversky_Loss(predictions[2], targets[2])

        det_all_loss *= self.lambdas[0]
        da_seg_loss *= self.lambdas[1]
        ll_seg_loss *= self.lambdas[2]
        ll_tversky_loss *= self.lambdas[3]
        
        loss = det_all_loss + da_seg_loss + ll_seg_loss + ll_tversky_loss
        return loss, (det_all_loss, da_seg_loss, ll_seg_loss, ll_tversky_loss, loss)

def get_loss(cfg, device, model):
    Det_loss = YOLOX_Loss(device=device, num_classes=cfg.MODEL.NC)
    Da_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Ll_Seg_Loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    Tversky_Loss = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0 / 3).to(device)

    if cfg.LOSS.FL_GAMMA > 0:
        Ll_Seg_Loss = FocalLossSeg(Ll_Seg_Loss, gamma=cfg.LOSS.FL_GAMMA)

    loss_list = [Det_loss, Da_Seg_Loss, Ll_Seg_Loss, Tversky_Loss]
    return MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)

class FocalLossSeg(nn.Module):
    def __init__(self, loss_fcn, gamma=2.0, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        p_t = torch.exp(-loss)
        alpha_factor = self.alpha * true + (1 - self.alpha) * (1 - true)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum() if self.reduction == 'sum' else loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = y_pred.sigmoid()
        y_true = y_true.float()
        
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()
        
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - score).pow(self.gamma)
