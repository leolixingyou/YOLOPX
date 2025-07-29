import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.sigmoid()
        
        # flatten label and prediction tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()    
        FP = ((1-target) * pred).sum()
        FN = (target * (1-pred)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)  
        
        return 1 - Tversky
