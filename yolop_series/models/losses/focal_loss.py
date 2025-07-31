import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred = pred.sigmoid()
        pt = torch.where(target == 1, pred, 1 - pred)
        log_pt = torch.log(pt + 1e-9)
        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
        return loss.mean()
