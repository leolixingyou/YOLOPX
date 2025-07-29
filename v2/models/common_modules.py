import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch.nn import Upsample
from torch.nn import SiLU

class seg_head(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["sigmoid", "softmax"]:
            raise ValueError(
                "`merge_policy` must be one of: ['sigmoid', 'softmax'], got {}".format(
                    policy
                )
            )
        self.policy = policy
        if self.policy == "sigmoid":
            self.m = nn.Sigmoid()
        else:
            self.m = nn.Softmax(dim=1)

    def forward(self, x):
        return self.m(x)

class FPN_C2(nn.Module):
    def __init__(self):
        super(FPN_C2, self).__init__()
    def forward(self, x):
        return x[0]

class FPN_C3(nn.Module):
    def __init__(self):
        super(FPN_C3, self).__init__()
    def forward(self, x):
        return x[4]  # c13 - 128通道 (恢复原设置)

class FPN_C4(nn.Module):
    def __init__(self):
        super(FPN_C4, self).__init__()
    def forward(self, x):
        return x[2]  # c8 - 512通道

# TAG专用的FPN选择器
class FPN_C3_TAG(nn.Module):
    def __init__(self):
        super(FPN_C3_TAG, self).__init__()
    def forward(self, x):
        return x[5]  # c16 - 256通道 (TAG专用)

class FPN_C4_TAG(nn.Module):
    def __init__(self):
        super(FPN_C4_TAG, self).__init__()
    def forward(self, x):
        return x[2]  # c8 - 512通道

class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act=True):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv3 = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, act=act),
            Conv(inter_dim, inter_dim, k=3, act=act)
        )
        self.cv4 = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, act=act),
            Conv(inter_dim, inter_dim, k=3, act=act)
        )
        self.out = Conv(inter_dim*4, out_dim, k=1)
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))
        return out

class DownSample(nn.Module):
    def __init__(self, in_dim, act=True):
            super().__init__()
            inter_dim = in_dim // 2
            self.mp = nn.MaxPool2d(2, 2)
            self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
            self.cv2 = nn.Sequential(
                Conv(in_dim, inter_dim, k=1, act=act),
                Conv(inter_dim, inter_dim, k=3, s=2, act=act)
            )
    
    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)
        return out

class ELANBlock_Head(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act=True):
        super(ELANBlock_Head, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv3 = Conv(inter_dim, inter_dim2, k=3, act=act)
        self.cv4 = Conv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv5 = Conv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv6 = Conv(inter_dim2, inter_dim2, k=3, act=act)
        self.out = Conv(inter_dim*2+inter_dim2*4, out_dim, k=1, act=act)
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))
        return out
 
class DownSample_Head(nn.Module):
    def __init__(self, in_dim, act=True):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d(2, 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act=act),
            Conv(inter_dim, inter_dim, k=3, s=2, act=act)
        )
 
    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)
        return out

class ELANNet(nn.Module):
    def __init__(self, use_C2 = False):
        super(ELANNet, self).__init__()
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3),      
            Conv(32, 64, k=3, s=2),
            Conv(64, 64, k=3)
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, s=2),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5)
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5)
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512),             
            ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5)
        )
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024),             
            ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25)
        )
        self.use_C2 = use_C2
 
    def forward(self, x):
        x = self.layer_1(x)
        c2 = self.layer_2(x)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        if self.use_C2:
            return c2, c3, c4, c5
        else:
            return c3, c4, c5

class PaFPNELAN(nn.Module):
    def __init__(self, in_dims=[512, 1024, 1024], out_dim=[128, 256, 512], act=True):
        super(PaFPNELAN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
        self.cv1 = Conv(c5//2, 256, k=1, act=act)
        self.cv2 = Conv(c4, 256, k=1, act=act)
        self.head_elan_1 = ELANBlock_Head(in_dim=512, out_dim=256, act=act)
        self.cv3 = Conv(256, 128, k=1, act=act)
        self.cv4 = Conv(c3, 128, k=1, act=act)
        self.head_elan_2 = ELANBlock_Head(in_dim=256, out_dim=128, act=act)
        self.mp1 = DownSample_Head(128, act=act)
        self.head_elan_3 = ELANBlock_Head(in_dim=512, out_dim=256, act=act)
        self.mp2 = DownSample_Head(256, act=act)
        self.head_elan_4 = ELANBlock_Head(in_dim=1024, out_dim=512, act=act)
        self.SPPF = GhostSPPCSPC(c5, 512)

    def forward(self, features):
        C2, c3, c4, c5 = features
        c5 = self.SPPF(c5)
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)
        return C2, c5, c8, c12, c13, c16, c19

class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2
        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.softmax_left = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.act = SiLU()

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        out = self.spatial_pool(x) + self.channel_pool(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)