"""
BDD100k dataset implementation.
"""
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

# --- Refactored Imports ---
from data.autodrive_dataset import AutoDriveDataset

# --- Inlined utility functions to remove dependencies ---
def convert_bbox(size, box):
    """Converts BDD100k bbox format to YOLO format."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

ID_DICT = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4, 'bike': 5, 'motor': 6, 'traffic light': 7, 'traffic sign': 8, 'train': 9}
ID_DICT_SINGLE = {'car': 0, 'bus': 1, 'truck': 2}

class BddDataset(AutoDriveDataset):
    """
    BDD100k dataset loader.
    """
    def __init__(self, cfg, is_train, transform=None):
        super().__init__(cfg, is_train, transform)
        self.db = self._get_db()

    def _get_db(self):
        """
        Loads the BDD100k database from annotation files.
        """
        print('Building BDD100k database...')
        gt_db = []
        
        img_root = Path(self.cfg.DATASET.DATAROOT)
        label_root = Path(self.cfg.DATASET.LABELROOT)
        mask_root = Path(self.cfg.DATASET.MASKROOT)
        lane_root = Path(self.cfg.DATASET.LANEROOT)
        
        indicator = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET
        
        # List all image files
        img_paths = list((img_root / indicator).glob('*.jpg'))

        # Limit number of images if specified
        num_images = self.cfg.DATASET.get('NUMBER_IMAGE', -1)
        if num_images > 0:
            if self.is_train:
                if len(img_paths) > num_images:
                    img_paths = img_paths[:num_images]
            else:
                val_num_images = num_images // 5
                if len(img_paths) > val_num_images:
                    img_paths = img_paths[:val_num_images]

        for img_path in tqdm(img_paths):
            label_path = label_root / indicator / f"{img_path.stem}.json"
            mask_path = mask_root / indicator / f"{img_path.stem}.png"
            lane_path = lane_root / indicator / f"{img_path.stem}.png"

            if not all([label_path.exists(), mask_path.exists(), lane_path.exists()]):
                continue

            with open(label_path, 'r') as f:
                label_data = json.load(f)
            
            objects = label_data['frames'][0]['objects']
            
            gt = []
            for obj in objects:
                if 'box2d' not in obj:
                    continue
                
                category = obj['category']
                if self.cfg.DATASET.get('SINGLE_CLS', False):
                    if category not in ID_DICT_SINGLE:
                        continue
                    cls_id = ID_DICT_SINGLE[category]
                else:
                    if category not in ID_DICT:
                        continue
                    cls_id = ID_DICT[category]
                
                x1, y1, x2, y2 = obj['box2d'].values()
                bbox = convert_bbox(self.cfg.DATASET.ORG_IMG_SIZE, (x1, x2, y1, y2))
                gt.append([cls_id, *bbox])

            gt_db.append({
                'image': str(img_path),
                'label': gt,
                'mask': str(mask_path),
                'lane': str(lane_path)
            })
            
        print(f'Database build finish. Found {len(gt_db)} images.')
        return gt_db