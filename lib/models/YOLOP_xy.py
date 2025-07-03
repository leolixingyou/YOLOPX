import yaml
import torch
import torch.nn as nn
from torch.nn import Upsample
import sys
import os

# 添加路径（根据你的项目结构调整）
sys.path.append(os.getcwd())
from lib.utils import initialize_weights, check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common import Conv, seg_head, PSA_p, MergeBlock
from lib.models.common import Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, IDetect, RepConv
from lib.models.YOLOX_Head_scales_noshare import YOLOXHead


class YAMLModelBuilder:
    """从YAML配置文件构建模型的类"""
    
    def __init__(self, config_path):
        """
        初始化配置加载器
        Args:
            config_path: YAML配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 构建模块映射表
        self.module_map = {
            'ELANNet': ELANNet,
            'PaFPNELAN': PaFPNELAN,
            'YOLOXHead': YOLOXHead,
            'FPN_C2': FPN_C2,
            'FPN_C3': FPN_C3,
            'FPN_C4': FPN_C4,
            'Conv': Conv,
            'Upsample': Upsample,
            'ELANBlock_Head': ELANBlock_Head,
            'seg_head': seg_head,
            'MergeBlock': MergeBlock,
            'PSA_p': PSA_p,
            # 可以根据需要添加更多模块
        }
    
    def parse_config(self):
        """解析YAML配置，返回原始格式的配置列表"""
        # 获取预测头索引
        prediction_heads = self.config['prediction_heads']
        det_out_idx = prediction_heads['det_out_idx']
        da_seg_out_idx = prediction_heads['da_seg_out_idx']
        ll_seg_out_idx = prediction_heads['ll_seg_out_idx']
        
        # 构建配置列表，第一项是预测头索引
        block_cfg = [[det_out_idx, da_seg_out_idx, ll_seg_out_idx]]
        
        # 添加网络层配置
        for layer in self.config['layers']:
            from_idx, module_name, args = layer
            # 处理参数中的null值
            processed_args = self._process_args(args)
            block_cfg.append([from_idx, module_name, processed_args])
        
        return block_cfg
    
    def _process_args(self, args):
        """处理参数列表，将null转换为None，将字符串转换为对应类型"""
        if not isinstance(args, list):
            return args
        
        processed = []
        for arg in args:
            if arg is None or arg == "null":
                processed.append(None)
            elif isinstance(arg, bool):
                processed.append(arg)
            elif isinstance(arg, str):
                processed.append(arg)
            elif isinstance(arg, (int, float)):
                processed.append(arg)
            elif isinstance(arg, list):
                processed.append(self._process_args(arg))
            else:
                processed.append(arg)
        
        return processed


class MCnetFromYAML(nn.Module):
    """从YAML配置构建的MCnet模型"""
    
    def __init__(self, config_path, **kwargs):
        super(MCnetFromYAML, self).__init__()
        
        # 加载配置
        builder = YAMLModelBuilder(config_path)
        block_cfg = builder.parse_config()
        
        layers, save = [], []
        self.nc = builder.config.get('model_config', {}).get('nc', 1)
        self.detector_index = -1
        
        # 获取输出索引
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        
        # 构建模型
        for i, (from_, block_name, args) in enumerate(block_cfg[1:]):
            # 获取模块类
            if isinstance(block_name, str):
                if block_name in builder.module_map:
                    block = builder.module_map[block_name]
                else:
                    raise ValueError(f"Unknown module: {block_name}")
            else:
                block = block_name
            
            if block is YOLOXHead:
                self.detector_index = i
            
            # 实例化模块
            block_ = block(*args) if args else block()
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            
            # 保存索引
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        
        assert self.detector_index == block_cfg[0][0]
        
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        
        # 设置检测器的stride和anchor
        Detector = self.model[self.detector_index]
        if isinstance(Detector, YOLOXHead):
            s = 512
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
            self.stride = Detector.strides
            Detector.initialize_biases(1e-2)
        
        initialize_weights(self)
    
    def forward(self, x):
        """前向传播"""
        cache = []
        out = []
        det_out = None
        
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]
            
            x = block(x)
            
            if i in self.seg_out_idx:
                out.append(x)
            if i == self.detector_index:
                det_out = x
            
            cache.append(x if block.index in self.save else None)
        
        out.insert(0, det_out)
        return out
    
    def fuse(self):
        """融合Conv2d和BatchNorm2d层"""
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        return self


def get_net_from_yaml(config_path, **kwargs):
    """
    从YAML配置文件创建网络
    Args:
        config_path: YAML配置文件路径
        **kwargs: 其他参数
    Returns:
        模型实例
    """
    model = MCnetFromYAML(config_path, **kwargs)
    return model


def fuse_conv_and_bn(conv, bn):
    """融合卷积层和批归一化层"""
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


if __name__ == "__main__":
    # 测试代码
    config_path = "/workspace/YOLOPX/lib/config/yolopx.yaml"  # YAML配置文件路径
    
    try:
        model = get_net_from_yaml(config_path)
        print("模型创建成功！")
        
        # 测试前向传播
        input_tensor = torch.randn((1, 3, 256, 256))
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print(f"检测输出形状: {outputs[0][0].shape}")
        print(f"驾驶区域分割输出形状: {outputs[1].shape}")
        print(f"车道线分割输出形状: {outputs[2].shape}")
        
    except Exception as e:
        print(f"模型创建失败: {e}")