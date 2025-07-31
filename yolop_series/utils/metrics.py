"""
Evaluation metrics for object detection and segmentation.
"""
import numpy as np
import torch

def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue
        
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall_curve = tpc / (n_l + 1e-16)
        r.append(np.interp(-0.5, -conf[i], recall_curve))

        precision_curve = tpc / (tpc + fpc)
        p.append(np.interp(-0.5, -conf[i], precision_curve))

        ap.append(compute_ap(recall_curve, precision_curve))

    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """Compute the average precision (AP) from the recall and precision curves."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class ConfusionMatrix:
    """
    A confusion matrix for object detection.
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        # Implementation details...
        pass

class SegmentationMetric:
    """
    Computes metrics for semantic segmentation.
    """
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        return np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / (union + 1e-16)
        return np.nanmean(IoU)
    
    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / (union + 1e-16)
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        return count.reshape(self.numClass, self.numClass)

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict.flatten(), imgLabel.flatten())

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
