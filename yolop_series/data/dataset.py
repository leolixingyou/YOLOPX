"""Unified dataset module for YOLOP series."""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random

class YOLOPDataset(Dataset):
    """Unified dataset for all YOLOP variants."""
    
    def __init__(self, cfg, split='train', transform=None):
        """Initialize dataset.
        
        Args:
            cfg: Configuration object
            split: 'train', 'val', or 'test'
            transform: Optional data augmentation transforms
        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.img_size = cfg.DATASET.IMAGE_SIZE
        
        # Set paths based on BDD100K structure
        self.data_root = cfg.DATASET.DATAROOT
        self.img_dir = os.path.join(self.data_root, 'images', '100k', split)
        self.det_label_dir = os.path.join(self.data_root, 'labels', 'det', split)
        self.da_seg_dir = os.path.join(self.data_root, 'labels', 'da_seg', split)
        self.ll_seg_dir = os.path.join(self.data_root, 'labels', 'll_seg', split)
        
        # Get image list
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        print(f"Found {len(self.img_list)} images for {split} set")
        
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        """Get item by index.
        
        Returns:
            data_dict: Dictionary containing:
                - image: Tensor [3, H, W]
                - det_label: Detection labels [N, 5] (class, x, y, w, h)
                - da_seg_mask: Driving area segmentation mask [H, W]
                - ll_seg_mask: Lane line segmentation mask [H, W]
                - img_path: Path to image file
        """
        img_path = self.img_list[idx]
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt')
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Resize image
        img, ratio, pad = self._resize_img(img, self.img_size)
        h, w = img.shape[:2]
        
        # Load detection labels
        det_label_path = os.path.join(self.det_label_dir, label_name)
        det_labels = self._load_det_labels(det_label_path, ratio, pad, w, h)
        
        # Load segmentation masks
        da_seg_path = os.path.join(self.da_seg_dir, img_name.replace('.jpg', '.png'))
        da_seg_mask = self._load_seg_mask(da_seg_path, self.img_size)
        
        ll_seg_path = os.path.join(self.ll_seg_dir, img_name.replace('.jpg', '.png'))
        ll_seg_mask = self._load_seg_mask(ll_seg_path, self.img_size)
        
        # Apply transforms if any
        if self.transform and self.split == 'train':
            img, det_labels, da_seg_mask, ll_seg_mask = self.transform(
                img, det_labels, da_seg_mask, ll_seg_mask)
        
        # Convert to tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        det_labels = torch.from_numpy(det_labels).float()
        da_seg_mask = torch.from_numpy(da_seg_mask).long()
        ll_seg_mask = torch.from_numpy(ll_seg_mask).long()
        
        return {
            'image': img,
            'det_label': det_labels,
            'da_seg_mask': da_seg_mask,
            'll_seg_mask': ll_seg_mask,
            'img_path': img_path,
            'shapes': (h0, w0),
            'ratio_pad': (ratio, pad)
        }
        
    def _resize_img(self, img, new_shape):
        """Resize image with padding to maintain aspect ratio."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw // 2, dh // 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = dh, dh
        left, right = dw, dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, r, (dw, dh)
        
    def _load_det_labels(self, label_path, ratio, pad, w, h):
        """Load and process detection labels."""
        if not os.path.exists(label_path):
            return np.zeros((0, 5), dtype=np.float32)
            
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) == 5:
                    cls, x, y, w_box, h_box = map(float, data)
                    # Convert from normalized to pixel coordinates
                    x = (x * w - w_box * w / 2) * ratio + pad[0]
                    y = (y * h - h_box * h / 2) * ratio + pad[1]
                    w_box = w_box * w * ratio
                    h_box = h_box * h * ratio
                    labels.append([cls, x, y, w_box, h_box])
                    
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        
    def _load_seg_mask(self, mask_path, new_shape):
        """Load and resize segmentation mask."""
        if not os.path.exists(mask_path):
            return np.zeros(new_shape, dtype=np.uint8)
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (new_shape[1], new_shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        return mask
        
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader."""
        images = torch.stack([item['image'] for item in batch])
        
        # For detection labels, we need to add batch index
        det_labels = []
        for i, item in enumerate(batch):
            labels = item['det_label']
            if labels.shape[0] > 0:
                batch_labels = torch.zeros((labels.shape[0], 6))
                batch_labels[:, 0] = i  # batch index
                batch_labels[:, 1:] = labels
                det_labels.append(batch_labels)
        det_labels = torch.cat(det_labels, 0) if det_labels else torch.zeros((0, 6))
        
        da_seg_masks = torch.stack([item['da_seg_mask'] for item in batch])
        ll_seg_masks = torch.stack([item['ll_seg_mask'] for item in batch])
        
        return {
            'images': images,
            'det_labels': det_labels,
            'da_seg_masks': da_seg_masks,
            'll_seg_masks': ll_seg_masks,
            'img_paths': [item['img_path'] for item in batch],
            'shapes': [item['shapes'] for item in batch],
            'ratio_pad': [item['ratio_pad'] for item in batch]
        }