"""YOLOPv2 implementation based on the paper 'YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception'."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ELAN(nn.Module):
    """E-ELAN block from YOLOv7 for YOLOPv2 backbone."""
    def __init__(self, in_channels, out_channels, n=4, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = nn.ModuleList([Conv(c_, c_, 3, 1) for _ in range(n)])
        self.cv4 = Conv(c_ * (n + 2), out_channels, 1, 1)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for m in self.cv3:
            y.append(m(y[-1]))
        return self.cv4(torch.cat(y, dim=1))

class SPP(nn.Module):
    """Spatial Pyramid Pooling layer."""
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class YOLOPv2Backbone(nn.Module):
    """YOLOPv2 backbone using E-ELAN blocks."""
    def __init__(self, base_channels=64):
        super().__init__()
        # Stem
        self.stem = Conv(3, base_channels, 6, 2, 2)  # 640 -> 320
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),  # 320 -> 160
            ELAN(base_channels * 2, base_channels * 2)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),  # 160 -> 80
            ELAN(base_channels * 4, base_channels * 4)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),  # 80 -> 40
            ELAN(base_channels * 8, base_channels * 8)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),  # 40 -> 20
            ELAN(base_channels * 16, base_channels * 16)
        )
        
        # SPP
        self.spp = SPP(base_channels * 16, base_channels * 16)

    def forward(self, x):
        x = self.stem(x)
        stage1_out = self.stage1(x)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        spp_out = self.spp(stage4_out)
        
        return stage2_out, stage3_out, spp_out

class YOLOPv2Neck(nn.Module):
    """YOLOPv2 neck with FPN + PAN."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # FPN (top-down)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv1 = Conv(in_channels[2] + in_channels[1], out_channels[1], 1, 1)
        self.cv2 = ELAN(out_channels[1], out_channels[1])
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv3 = Conv(out_channels[1] + in_channels[0], out_channels[0], 1, 1)
        self.cv4 = ELAN(out_channels[0], out_channels[0])
        
        # PAN (bottom-up)
        self.cv5 = Conv(out_channels[0], out_channels[0], 3, 2)
        self.cv6 = Conv(out_channels[0] + out_channels[1], out_channels[1], 1, 1)
        self.cv7 = ELAN(out_channels[1], out_channels[1])
        
        self.cv8 = Conv(out_channels[1], out_channels[1], 3, 2)
        self.cv9 = Conv(out_channels[1] + in_channels[2], out_channels[2], 1, 1)
        self.cv10 = ELAN(out_channels[2], out_channels[2])

    def forward(self, x):
        p3, p4, p5 = x
        
        # FPN
        p5_up = self.up1(p5)
        p4_cat = torch.cat([p5_up, p4], dim=1)
        p4_out = self.cv2(self.cv1(p4_cat))
        
        p4_up = self.up2(p4_out)
        p3_cat = torch.cat([p4_up, p3], dim=1)
        p3_out = self.cv4(self.cv3(p3_cat))
        
        # PAN
        p3_down = self.cv5(p3_out)
        p4_cat2 = torch.cat([p3_down, p4_out], dim=1)
        p4_out2 = self.cv7(self.cv6(p4_cat2))
        
        p4_down = self.cv8(p4_out2)
        p5_cat = torch.cat([p4_down, p5], dim=1)
        p5_out = self.cv10(self.cv9(p5_cat))
        
        return p3_out, p4_out2, p5_out

class DetectionHead(nn.Module):
    """YOLOPv2 detection head (anchor-based)."""
    def __init__(self, nc, anchors, ch):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        
        self.m = nn.ModuleList([Conv(x, self.no * self.na, 1) for x in ch])

    def forward(self, x):
        outputs = []
        for i, (xi, m) in enumerate(zip(x, self.m)):
            out = m(xi)
            bs, _, h, w = out.shape
            out = out.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(out)
        return outputs

class DrivingAreaHead(nn.Module):
    """Driving area segmentation head (connected before FPN)."""
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        # Progressive upsampling with skip connections
        self.cv1 = Conv(in_channels, 256, 3, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.cv2 = Conv(256, 128, 3, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.cv3 = Conv(128, 64, 3, 1)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.cv4 = Conv(64, 32, 3, 1)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # x is from stage3_out which is 1/16 scale
        x = self.cv1(x)
        x = self.up1(x)  # 1/8
        x = self.cv2(x)
        x = self.up2(x)  # 1/4
        x = self.cv3(x)
        x = self.up3(x)  # 1/2
        x = self.cv4(x)
        x = self.up4(x)  # 1/1
        return self.out(x)

class LaneDetectionHead(nn.Module):
    """Lane detection head with deconvolution (connected after FPN)."""
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        # Use deconvolution for better lane detection
        # p3 is at 1/8 scale, we need to upsample to full resolution
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # x is from p3 which is 1/8 scale
        x = self.relu1(self.bn1(self.deconv1(x)))  # 1/4
        x = self.relu2(self.bn2(self.deconv2(x)))  # 1/2
        x = self.relu3(self.bn3(self.deconv3(x)))  # 1/1
        return self.out(x)

class YOLOPv2(nn.Module):
    """YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception."""
    def __init__(self, nc=1, num_seg_classes=2, anchors=None):
        super().__init__()
        
        # Default anchors if not provided
        if anchors is None:
            anchors = [[10,13, 16,30, 33,23],  # P3/8
                      [30,61, 62,45, 59,119],  # P4/16
                      [116,90, 156,198, 373,326]]  # P5/32
        
        base_channels = 64
        
        # Backbone
        self.backbone = YOLOPv2Backbone(base_channels)
        
        # Neck
        neck_in_channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        neck_out_channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.neck = YOLOPv2Neck(neck_in_channels, neck_out_channels)
        
        # Detection head
        self.detection_head = DetectionHead(nc, anchors, neck_out_channels)
        
        # Driving area head (connected to backbone stage3 output)
        self.da_head = DrivingAreaHead(base_channels * 8, num_seg_classes)
        
        # Lane detection head (connected to neck output)
        self.ll_head = LaneDetectionHead(base_channels * 4, num_seg_classes)

    def forward(self, x):
        # Backbone
        stage2_out, stage3_out, spp_out = self.backbone(x)
        
        # Driving area segmentation (from stage3)
        da_seg = self.da_head(stage3_out)
        
        # Neck
        p3, p4, p5 = self.neck([stage2_out, stage3_out, spp_out])
        
        # Detection
        detection = self.detection_head([p3, p4, p5])
        
        # Lane detection (from neck p3)
        ll_seg = self.ll_head(p3)
        
        return {
            'detection': detection,
            'da_seg': da_seg,
            'll_seg': ll_seg
        }

# Model creation function
def yolop_v2(cfg):
    """Create YOLOPv2 model from config."""
    model = YOLOPv2(
        nc=cfg.get('DATASET.NUM_CLASSES', 1),
        num_seg_classes=cfg.get('DATASET.NUM_SEG_CLASSES', 2)
    )
    return model