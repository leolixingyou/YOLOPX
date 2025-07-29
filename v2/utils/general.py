"""
General utility functions for the YOLOPX project.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class DataLoaderX(DataLoader):
    """Prefetch Dataloader for faster training."""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_optimizer(cfg, model):
    """Initializes and returns an optimizer."""
    optimizer = None
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(g0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(g0, lr=cfg.TRAIN.LR0, betas=(cfg.TRAIN.MOMENTUM, 0.999))
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(g0, lr=cfg.TRAIN.LR0, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    
    optimizer.add_param_group({'params': g1, 'weight_decay': cfg.TRAIN.WD})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
        
    return optimizer

def is_parallel(model):
    """Checks if a model is wrapped in DataParallel or DistributedDataParallel."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def time_synchronized():
    """Synchronizes CUDA timers."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]      # width
    y[:, 3] = x[:, 3] - x[:, 1]      # height
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)."""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, multi_label=False, max_det=300):
    """Performs Non-Maximum Suppression (NMS) on inference results."""
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box = xywh2xyxy(x[:, :4])
        
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue
        
        # Batched NMS
        c = x[:, 5:6] * 4096 # Max image size
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i]
        
    return output
