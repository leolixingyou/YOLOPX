import yaml
import torch
import torch.nn as nn
from torch.nn import Upsample
import sys
import os

from utils.torch_utils import initialize_weights
from models.common_modules import Conv, seg_head, PSA_p, MergeBlock, Concat, FPN_C2, FPN_C3, FPN_C4, FPN_C3_TAG, FPN_C4_TAG, ELANNet, ELANBlock_Head, PaFPNELAN, RepConv
from models.heads.yolox_head import YOLOXHead
from mtl.feature_solvers import TaskAttention, ImprovedTaskAttention, AdaptiveFeatureFusion, ImprovedSelect
from mtl.tag_module import TaskAdaptiveAttentionGenerator, TAGSelect, TAGFusionLayer

class Select(nn.Module):
    def __init__(self, index):
        super(Select, self).__init__()
        self.index = int(index)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        return x

class MCnetFromYAML(nn.Module):
    def __init__(self, config_path, **kwargs):
        super(MCnetFromYAML, self).__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.module_map = {
            'ELANNet': ELANNet, 'PaFPNELAN': PaFPNELAN, 'YOLOXHead': YOLOXHead,
            'FPN_C2': FPN_C2, 'FPN_C3': FPN_C3, 'FPN_C4': FPN_C4,
            'FPN_C3_TAG': FPN_C3_TAG, 'FPN_C4_TAG': FPN_C4_TAG,
            'Conv': Conv, 'Upsample': Upsample, 'ELANBlock_Head': ELANBlock_Head,
            'seg_head': seg_head, 'MergeBlock': MergeBlock, 'PSA_p': PSA_p,
            'TaskAttention': TaskAttention, 'Select': Select,
            'ImprovedTaskAttention': ImprovedTaskAttention, 'ImprovedSelect': ImprovedSelect,
            'AdaptiveFeatureFusion': AdaptiveFeatureFusion,
            'TaskAdaptiveAttentionGenerator': TaskAdaptiveAttentionGenerator,
            'TAGSelect': TAGSelect, 'TAGFusionLayer': TAGFusionLayer,
            'Concat': Concat,
        }

        prediction_heads = self.config['prediction_heads']
        self.det_out_idx = prediction_heads['det_out_idx']
        self.da_seg_out_idx = prediction_heads['da_seg_out_idx']
        self.ll_seg_out_idx = prediction_heads['ll_seg_out_idx']
        
        self.layers = nn.ModuleList()
        for from_idx, module_name, args in self.config['layers']:
            module_class = self.module_map[module_name]
            module = module_class(*args) if args else module_class()
            module.from_ = from_idx
            self.layers.append(module)
            
        initialize_weights(self)

    def forward(self, x):
        cache = {}
        outputs = {}
        
        # The input to PaFPNELAN (layer 1) is the output of ELANNet (layer 0)
        # The input to YOLOXHead (layer 2) is the output of PaFPNELAN (layer 1)
        
        for i, block in enumerate(self.layers):
            from_indices = block.from_ if isinstance(block.from_, list) else [block.from_]
            
            inputs = []
            for idx in from_indices:
                inputs.append(x if idx == -1 else cache[idx])
            
            if len(inputs) == 1:
                model_input = inputs[0]
            else:
                model_input = inputs
            
            # print(f"Layer {i}: {block.__class__.__name__}, Input shape: {[inp.shape if isinstance(inp, torch.Tensor) else [i.shape for i in inp] for inp in inputs]}")
            x = block(model_input)
            # print(f"Layer {i}: {block.__class__.__name__}, Output shape: {x.shape if isinstance(x, torch.Tensor) else [o.shape for o in x]}")
            cache[i] = x

            if i == self.det_out_idx: outputs['det'] = x
            elif i == self.da_seg_out_idx: outputs['da'] = x
            elif i == self.ll_seg_out_idx: outputs['ll'] = x

        return outputs.get('det'), outputs.get('da'), outputs.get('ll')

def get_net_from_yaml(config_path, **kwargs):
    model = MCnetFromYAML(config_path, **kwargs)
    return model
