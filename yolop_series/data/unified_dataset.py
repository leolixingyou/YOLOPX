"""
统一的数据集模块
整合所有数据集相关功能，避免重复
"""
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

# 统一的转换函数
def convert_bbox(size, box):
    """将bbox转换为YOLO格式 (x_center, y_center, width, height)"""
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

def convert_to_bdd_bbox(x1, y1, x2, y2):
    """将坐标转换为BDD格式"""
    return [float(x1), float(y1), float(x2), float(y2)]

# ID映射字典
ID_DICT_SINGLE = {'car': 0}  # 单类别
ID_DICT_MULTI = {
    'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4,
    'bike': 5, 'motor': 6, 'traffic light': 7, 'traffic sign': 8, 'train': 9
}

class AutoDriveDataset(Dataset):
    """
    统一的自动驾驶数据集基类
    支持检测、驾驶区域分割和车道线分割三个任务
    """
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.transform = transform
        self.input_size = cfg.DATASET.IMAGE_SIZE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
        
        # 初始化ID映射
        self.id_dict = ID_DICT_SINGLE if cfg.DATASET.NC == 1 else ID_DICT_MULTI
        
        # 数据集限制
        self.num_images_limit = cfg.DATASET.get('NUMBER_IMAGE', -1)
        self.num_val_images = cfg.DATASET.get('NUMBER_VAL', -1)
        
        # 加载数据
        self.db = self._get_db()

    def _get_db(self):
        """子类需要实现的数据加载方法"""
        raise NotImplementedError

    def _apply_image_limit(self, data_list):
        """应用图片数量限制"""
        if self.is_train and self.num_images_limit > 0:
            return data_list[:self.num_images_limit]
        elif not self.is_train:
            if self.num_val_images > 0:
                return data_list[:self.num_val_images]
            elif self.num_images_limit > 0:
                # 如果没有指定验证集大小，使用训练集的20%
                val_size = max(1, self.num_images_limit // 5)
                return data_list[:val_size]
        return data_list

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx]
        img = cv2.imread(data['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整图片大小
        resized_shape = self.input_size
        if isinstance(resized_shape, list):
            resized_shape = (resized_shape[0], resized_shape[1])
        h0, w0 = img.shape[:2]
        r = min(resized_shape[0] / h0, resized_shape[1] / w0)
        if r != 1:
            img = cv2.resize(
                img,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_LINEAR
            )
        
        h, w = img.shape[:2]
        img, ratio, pad = self.letterbox(img, resized_shape, auto=False, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        
        # 加载分割标签
        if data.get('mask'):
            da_seg_mask = cv2.imread(data['mask'], cv2.IMREAD_UNCHANGED)
            da_seg_mask = cv2.resize(da_seg_mask, (w, h))
            da_seg_mask = self.letterbox_seg(da_seg_mask, resized_shape, pad[0], pad[1])
        else:
            da_seg_mask = np.zeros((resized_shape[0], resized_shape[1]), dtype=np.uint8)
            
        if data.get('lane'):
            ll_seg_mask = cv2.imread(data['lane'], cv2.IMREAD_UNCHANGED)
            ll_seg_mask = cv2.resize(ll_seg_mask, (w, h))
            ll_seg_mask = self.letterbox_seg(ll_seg_mask, resized_shape, pad[0], pad[1])
        else:
            ll_seg_mask = np.zeros((resized_shape[0], resized_shape[1]), dtype=np.uint8)
        
        # 应用数据增强
        if self.transform and self.is_train:
            img, da_seg_mask, ll_seg_mask = self.transform(img, da_seg_mask, ll_seg_mask)
        
        # 转换为tensor
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        da_seg_mask = torch.from_numpy(da_seg_mask).long()
        ll_seg_mask = torch.from_numpy(ll_seg_mask).long()
        
        # 准备检测标签
        det_label = self.process_detection_labels(data, ratio, pad, resized_shape)
        
        target = [det_label, da_seg_mask, ll_seg_mask]
        return img, target, data['image'], shapes

    def process_detection_labels(self, data, ratio, pad, resized_shape):
        """处理检测标签"""
        labels = data.get('label', [])
        det_label = []
        
        for label in labels:
            category = label['category']
            if category in self.id_dict:
                x1 = label['x1'] * ratio[0] + pad[0]
                y1 = label['y1'] * ratio[1] + pad[1]
                x2 = label['x2'] * ratio[0] + pad[0]
                y2 = label['y2'] * ratio[1] + pad[1]
                
                cls_id = self.id_dict[category]
                det_label.append([cls_id, x1, y1, x2, y2])
        
        return np.array(det_label) if det_label else np.zeros((0, 5))

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """调整图片大小并填充"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def letterbox_seg(self, mask, new_shape, dw, dh):
        """调整分割mask大小"""
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return mask

    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        
        for i, l in enumerate(label):
            if isinstance(l[0], np.ndarray) and len(l[0]) > 0:
                l_det = l[0].copy()
                l_det[:, 0] = i  # add target image index
                label_det.append(l_det)
            label_seg.append(l[1])
            label_lane.append(l[2])
        
        return (
            torch.stack(img, 0),
            [torch.from_numpy(np.concatenate(label_det, 0)) if label_det else torch.zeros((0, 6)),
             torch.stack(label_seg, 0),
             torch.stack(label_lane, 0)],
            paths,
            shapes
        )


class BddDataset(AutoDriveDataset):
    """BDD100k数据集实现"""
    
    def _get_db(self):
        """加载BDD100k数据"""
        print('Loading BDD100k dataset...')
        gt_db = []
        
        # 设置路径
        img_root = Path(self.cfg.DATASET.DATAROOT)
        label_root = Path(self.cfg.DATASET.LABELROOT)
        mask_root = Path(self.cfg.DATASET.MASKROOT)
        lane_root = Path(self.cfg.DATASET.LANEROOT)
        
        indicator = self.cfg.DATASET.TRAIN_SET if self.is_train else self.cfg.DATASET.TEST_SET
        
        # 获取图片列表
        img_dir = img_root / indicator
        img_files = sorted(img_dir.glob(f'*.{self.cfg.DATASET.DATA_FORMAT}'))
        
        # 应用数量限制
        img_files = self._apply_image_limit(img_files)
        
        print(f'Found {len(img_files)} images for {"training" if self.is_train else "validation"}')
        
        # 加载标注
        for img_file in tqdm(img_files, desc='Loading annotations'):
            img_name = img_file.stem
            
            # 构建路径
            rec = {
                'image': str(img_file),
                'mask': str(mask_root / indicator / f'{img_name}.png'),
                'lane': str(lane_root / indicator / f'{img_name}.png'),
                'label': []
            }
            
            # 加载检测标签
            label_file = label_root / indicator / f'{img_name}.json'
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    for obj in label_data.get('frames', [{}])[0].get('objects', []):
                        if 'box2d' in obj:
                            box = obj['box2d']
                            rec['label'].append({
                                'category': obj['category'],
                                'x1': box['x1'],
                                'y1': box['y1'],
                                'x2': box['x2'],
                                'y2': box['y2']
                            })
            
            gt_db.append(rec)
        
        print(f'Database loaded. Total samples: {len(gt_db)}')
        return gt_db