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
from lib.models.tag_module import TaskAttention, MultiScaleTaskAttention
from lib.models.improved_tag_module import ImprovedTaskAttention, AdaptiveFeatureFusion, ImprovedSelect

class Select(nn.Module):
    def __init__(self, index):
        super(Select, self).__init__()
        self.index = int(index)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            if self.index >= len(x):
                print(f"Select error: trying to access index {self.index} from list of length {len(x)}")
                print(f"List contents types: {[type(item).__name__ for item in x]}")
                if hasattr(x[0], 'shape'):
                    print(f"List contents shapes: {[item.shape if hasattr(item, 'shape') else 'no shape' for item in x]}")
                raise IndexError(f"list index out of range: {self.index} >= {len(x)}")
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
            'TaskAttention': TaskAttention, 'MultiScaleTaskAttention': MultiScaleTaskAttention, 'Select': Select,
            # Improved TAG modules
            'ImprovedTaskAttention': ImprovedTaskAttention, 
            'AdaptiveFeatureFusion': AdaptiveFeatureFusion,
            'ImprovedSelect': ImprovedSelect,
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
            block_.from_ = from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        
        initialize_weights(self)

    def forward(self, x):
        cache = {}
        outputs = {}
        
        for i, block in enumerate(self.model):
            input_for_block = None
            
            if block.from_ == -1:
                input_for_block = x
            elif isinstance(block.from_, int):
                input_for_block = cache[block.from_]
            elif isinstance(block.from_, list):
                # Collect inputs from multiple sources
                collected_inputs = []
                for j in block.from_:
                    if j == -1:
                        collected_inputs.append(x) # Previous block's output
                    else:
                        collected_inputs.append(cache[j])
                
                # Determine how to pass collected_inputs to the block
                if isinstance(block, AdaptiveFeatureFusion):
                    # AdaptiveFeatureFusion expects unpacked arguments (feature1, feature2)
                    x = block(*collected_inputs)
                    cache[i] = x
                    continue # Skip the general block(input_for_block) call below
                elif isinstance(block, ImprovedSelect):
                    # ImprovedSelect expects a single list as input (task_features)
                    input_for_block = collected_inputs
                else:
                    # Default for other multi-input modules: pass as a list
                    input_for_block = collected_inputs
            
            # Execute the block with the prepared input_for_block
            x = block(input_for_block)
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
