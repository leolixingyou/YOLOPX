"""
PyTorch specific utility functions.
"""
import torch.nn as nn

def initialize_weights(model):
    """Initializes model weights."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # Kaiming normal initialization can be done here
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
