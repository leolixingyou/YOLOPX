import yaml
import torch
import torch.nn as nn
from torch.nn import Upsample
import sys
import os
import importlib

from utils.torch_utils import initialize_weights

class Select(nn.Module):
    def __init__(self, index):
        super(Select, self).__init__()
        self.index = int(index)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        return x

def _load_modules(module_path):
    """Dynamically loads all nn.Module classes from a given module file."""
    module = importlib.import_module(module_path)
    module_map = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module):
            module_map[attr_name] = attr
    # Add utility modules that might not be in the common files
    module_map['Select'] = Select
    return module_map

class MCnetFromYAML(nn.Module):
    def __init__(self, config_path, **kwargs):
        super(MCnetFromYAML, self).__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Determine which module set to use based on the config file
        model_family = self.config.get('model_family', 'yolopx_v2') # Default to our v2 modules

        if model_family == 'yolop_v1_official':
            self.module_map = _load_modules('models.yolop_v1_official_modules.modules')
        elif model_family == 'yolop_v3_official':
            self.module_map = _load_modules('models.yolop_v3_official_modules.modules')
        else: # Default case for yolopx_v2
            # Manually import for the default case to maintain original structure
            from models.common_modules import Conv, seg_head, PSA_p, MergeBlock, Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, RepConv
            from models.heads.yolox_head import YOLOXHead
            from models.heads.yolop_head import Detect as YOLOPHead
            self.module_map = {
                'ELANNet': ELANNet, 'PaFPNELAN': PaFPNELAN, 'YOLOXHead': YOLOXHead, 'YOLOPHead': YOLOPHead,
                'FPN_C2': FPN_C2, 'FPN_C3': FPN_C3, 'FPN_C4': FPN_C4, 'Conv': Conv, 'Upsample': Upsample, 
                'ELANBlock_Head': ELANBlock_Head, 'seg_head': seg_head, 'MergeBlock': MergeBlock, 
                'PSA_p': PSA_p, 'Concat': Concat, 'Select': Select, 'RepConv': RepConv
            }

        prediction_heads = self.config['prediction_heads']
        self.det_head_indices = []
        if prediction_heads.get('det_out_idx'): # For single head models like YOLOPv1
            self.det_head_indices.append(prediction_heads['det_out_idx'])
        else: # For multi-head models
            i = 1
            while f'det_head_{i}_idx' in prediction_heads:
                self.det_head_indices.append(prediction_heads[f'det_head_{i}_idx'])
                i += 1

        self.da_seg_out_idx = prediction_heads['da_seg_out_idx']
        self.ll_seg_out_idx = prediction_heads['ll_seg_out_idx']
        
        self.layers = nn.ModuleList()
        for from_idx, module_name, args in self.config['layers']:
            if module_name not in self.module_map:
                raise KeyError(f"Module '{module_name}' not found in the module map for model family '{model_family}'.")
            module_class = self.module_map[module_name]
            # Ensure args is a list for unpacking
            args = args if isinstance(args, list) else [args]
            module = module_class(*args)
            module.from_ = from_idx
            self.layers.append(module)
            
        initialize_weights(self)

    def forward(self, x):
        cache = {}
        outputs = {}
        det_outputs = []
        
        for i, block in enumerate(self.layers):
            from_indices = block.from_ if isinstance(block.from_, list) else [block.from_]
            
            inputs = [x if idx == -1 else cache[idx] for idx in from_indices]
            
            model_input = inputs[0] if len(inputs) == 1 else inputs
            
            output = block(model_input)
            cache[i] = output

            if i in self.det_head_indices:
                det_outputs.append(output)
            
            if i == self.da_seg_out_idx: outputs['da'] = output
            elif i == self.ll_seg_out_idx: outputs['ll'] = output

        if len(det_outputs) > 1:
            outputs['det'] = det_outputs
        elif len(det_outputs) == 1:
            outputs['det'] = det_outputs[0]
        else:
            outputs['det'] = None

        return outputs.get('det'), outputs.get('da'), outputs.get('ll')

def get_net_from_yaml(config_path, **kwargs):
    model = MCnetFromYAML(config_path, **kwargs)
    return model
