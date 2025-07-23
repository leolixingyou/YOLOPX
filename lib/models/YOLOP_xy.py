import yaml
import torch
import torch.nn as nn
from torch.nn import Upsample
import sys
import os

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, seg_head, PSA_p, MergeBlock, Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, IDetect, RepConv
from lib.models.YOLOX_Head_scales_noshare import YOLOXHead
from lib.models.tag_module import TaskAttention

class Select(nn.Module):
    def __init__(self, index):
        super(Select, self).__init__()
        self.index = int(index)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        return x

class YAMLModelBuilder:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.module_map = {
            'ELANNet': ELANNet, 'PaFPNELAN': PaFPNELAN, 'YOLOXHead': YOLOXHead,
            'FPN_C2': FPN_C2, 'FPN_C3': FPN_C3, 'FPN_C4': FPN_C4,
            'Conv': Conv, 'Upsample': Upsample, 'ELANBlock_Head': ELANBlock_Head,
            'seg_head': seg_head, 'MergeBlock': MergeBlock, 'PSA_p': PSA_p,
            'TaskAttention': TaskAttention, 'Select': Select,
        }

    def parse_config(self):
        prediction_heads = self.config['prediction_heads']
        block_cfg = [[prediction_heads['det_out_idx'], prediction_heads['da_seg_out_idx'], prediction_heads['ll_seg_out_idx']]]
        
        for layer in self.config['layers']:
            from_idx, module_name, args = layer
            processed_args = self._process_args(args)
            block_cfg.append([from_idx, module_name, processed_args])
        
        return block_cfg

    def _process_args(self, args):
        if not isinstance(args, list):
            return args
        processed = []
        for arg in args:
            if arg is None or arg == 'null':
                processed.append(None)
            else:
                processed.append(arg)
        return processed

class MCnetFromYAML(nn.Module):
    def __init__(self, config_path, **kwargs):
        super(MCnetFromYAML, self).__init__()
        builder = YAMLModelBuilder(config_path)
        block_cfg = builder.parse_config()
        
        layers, save = [], []
        self.nc = builder.config.get('model_config', {}).get('nc', 1)
        self.det_out_idx, self.da_seg_out_idx, self.ll_seg_out_idx = block_cfg[0]
        
        for i, (from_, block_name, args) in enumerate(block_cfg[1:]):
            block = builder.module_map[block_name]
            block_ = block(*args) if args else block()
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        
        initialize_weights(self)

    def forward(self, x):
        cache = {}
        outputs = {}
        
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                if isinstance(block.from_, int):
                    x = cache[block.from_]
                else:
                    # Handle the special case of -1, which refers to the previous layer's output
                    x = [x if j == -1 else cache[j] for j in block.from_]
            
            x = block(x)
            cache[i] = x

            if i == self.det_out_idx:
                outputs['det'] = x
            elif i == self.da_seg_out_idx:
                outputs['da'] = x
            elif i == self.ll_seg_out_idx:
                outputs['ll'] = x

        # Ensure all task outputs are present
        det_out = outputs.get('det', None)
        da_out = outputs.get('da', None)
        ll_out = outputs.get('ll', None)

        return det_out, da_out, ll_out

def get_net_from_yaml(config_path, **kwargs):
    model = MCnetFromYAML(config_path, **kwargs)
    return model
