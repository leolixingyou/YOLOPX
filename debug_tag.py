import sys, os
sys.path.append('.')
from lib.models import get_net_from_yaml
import torch

model = get_net_from_yaml('lib/config/yolopx-tag-debug.yaml')
print(f'Model has {len(model.model)} blocks')

for i, block in enumerate(model.model):
    print(f'Block {i}: {type(block).__name__} from {block.from_}')
    if hasattr(block, 'index'):
        print(f'  Select index: {block.index}')

dummy_input = torch.randn(1, 3, 384, 640)
cache = {}
x = dummy_input

try:
    for i, block in enumerate(model.model):
        print(f'Processing block {i}: {type(block).__name__}')
        
        if block.from_ != -1:
            if isinstance(block.from_, int):
                x = cache[block.from_]
                print(f'  Input from cache[{block.from_}]')
            else:
                x = [x if j == -1 else cache[j] for j in block.from_]
                print(f'  Input from multiple sources')
        
        print(f'  Input type: {type(x).__name__}')
        if isinstance(x, list):
            print(f'  Input length: {len(x)}')
            print(f'  Input item types: {[type(item).__name__ for item in x]}')
            
        with torch.no_grad():
            x = block(x)
        cache[i] = x
        
        print(f'  Output type: {type(x).__name__}')
        if isinstance(x, list):
            print(f'  Output length: {len(x)}')
        print()
        
        if i >= 3:
            break
            
except Exception as e:
    print(f'Error at block {i}: {e}')
    import traceback
    traceback.print_exc()