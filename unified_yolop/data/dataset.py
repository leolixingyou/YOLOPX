"""Unified dataset for YOLOP series models with correct BDD100K paths."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class UnifiedYOLOPDataset(Dataset):
    """Dataset for all YOLOP model variants."""
    
    def __init__(self, cfg, split='train', transform=None):
        """Initialize dataset.
        
        Args:
            cfg: Configuration object
            split: 'train' or 'val'
            transform: Optional augmentation transforms
        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        
        # Fixed dataset paths for BDD100K YOLOP
        self.root = Path('/workspace/bdd100k/yolop_train/')
        self.img_size = cfg.get('MODEL.IMAGE_SIZE', [640, 640])
        
        # Correct paths - no try/except, direct assignment
        self.img_dir = self.root / 'images' / split
        self.det_label_dir = self.root / 'bdd_det' / split
        self.da_seg_dir = self.root / 'bdd_seg_gt' / split
        self.ll_seg_dir = self.root / 'bdd_lane_gt' / split
        
        # Verify directories exist
        assert self.img_dir.exists(), f"Image directory not found: {self.img_dir}"
        assert self.det_label_dir.exists(), f"Detection label directory not found: {self.det_label_dir}"
        assert self.da_seg_dir.exists(), f"DA segmentation directory not found: {self.da_seg_dir}"
        assert self.ll_seg_dir.exists(), f"LL segmentation directory not found: {self.ll_seg_dir}"
        
        # Get image list
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')))
        assert len(self.img_files) > 0, f"No images found in {self.img_dir}"
        
        # For quick testing or custom limits, use only a subset of data
        if split == 'train':
            # First check for specific train samples limit
            train_samples = cfg.get('DATASET.TRAIN_SAMPLES', None)
            if train_samples and train_samples > 0:
                self.img_files = self.img_files[:train_samples]
            else:
                # Fall back to general max_samples
                max_samples = cfg.get('DATASET.MAX_SAMPLES', None)
                if max_samples and max_samples > 0:
                    self.img_files = self.img_files[:max_samples]
        elif split == 'val':
            # First check for specific val samples limit
            val_samples = cfg.get('DATASET.VAL_SAMPLES', None)
            if val_samples and val_samples > 0:
                self.img_files = self.img_files[:val_samples]
            else:
                # Fall back to general max_samples
                max_samples = cfg.get('DATASET.MAX_SAMPLES', None)
                if max_samples and max_samples > 0:
                    self.img_files = self.img_files[:max_samples]
        
        print(f"Dataset initialized: {split} split with {len(self.img_files)} images")
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.img_files)
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get item by index.
        
        Returns:
            Dictionary containing:
                - image: Tensor [3, H, W]
                - det_labels: Detection labels [N, 5] (class, x, y, w, h)
                - da_seg_mask: Driving area mask [H, W]
                - ll_seg_mask: Lane line mask [H, W]
                - img_info: Image metadata
        """
        # Get image path
        img_path = self.img_files[index]
        img_name = img_path.stem
        
        # Load image - no try/except
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img, ratio, pad = self._resize_img(img, self.img_size)
        
        # Load labels
        det_labels = self._load_det_labels(img_name, ratio, pad)
        da_seg_mask = self._load_seg_mask(self.da_seg_dir / f"{img_name}.png", self.img_size)
        ll_seg_mask = self._load_seg_mask(self.ll_seg_dir / f"{img_name}.png", self.img_size)
        
        # Apply augmentations if specified
        if self.transform:
            # TODO: Add augmentation pipeline
            pass
            
        # Convert to tensors
        # Normalize image
        img = img.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Convert to CHW format
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        # Convert masks to tensors
        da_seg_mask = torch.from_numpy(da_seg_mask).long()
        ll_seg_mask = torch.from_numpy(ll_seg_mask).long()
        
        # Convert labels
        if len(det_labels) > 0:
            det_labels = torch.from_numpy(det_labels)
        else:
            det_labels = torch.zeros((0, 5), dtype=torch.float32)
            
        return {
            'image': img,
            'det_labels': det_labels,
            'da_seg_mask': da_seg_mask,
            'll_seg_mask': ll_seg_mask,
            'img_info': {
                'file_name': str(img_path),
                'height': self.img_size[1],
                'width': self.img_size[0],
                'id': index
            }
        }
        
    def _resize_img(self, img: np.ndarray, new_shape: List[int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with padding to maintain aspect ratio."""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, r, (dw, dh)
        
    def _load_det_labels(self, img_name: str, ratio: float, pad: Tuple[int, int]) -> np.ndarray:
        """Load detection labels."""
        label_path = self.det_label_dir / f"{img_name}.txt"
        
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
            
        # Read labels
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    cls, x, y, w, h = map(float, parts)
                    
                    # Keep normalized format for YOLO
                    labels.append([cls, x, y, w, h])
                    
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        
    def _load_seg_mask(self, mask_path: Path, new_shape: List[int]) -> np.ndarray:
        """Load segmentation mask."""
        if not mask_path.exists():
            return np.zeros((new_shape[1], new_shape[0]), dtype=np.uint8)
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((new_shape[1], new_shape[0]), dtype=np.uint8)
            
        # Resize to match image size
        mask = cv2.resize(mask, (new_shape[0], new_shape[1]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary mask
        mask = (mask > 0).astype(np.uint8)
        
        return mask
        
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Stack masks
        da_seg_masks = torch.stack([item['da_seg_mask'] for item in batch])
        ll_seg_masks = torch.stack([item['ll_seg_mask'] for item in batch])
        
        # Handle variable length detection labels
        max_det = max(len(item['det_labels']) for item in batch)
        if max_det > 0:
            det_labels = torch.zeros((len(batch), max_det, 5))
            for i, item in enumerate(batch):
                if len(item['det_labels']) > 0:
                    det_labels[i, :len(item['det_labels'])] = item['det_labels']
        else:
            det_labels = torch.zeros((len(batch), 1, 5))
            
        # Collect image info
        img_info = [item['img_info'] for item in batch]
        
        return {
            'image': images,
            'det_labels': det_labels,
            'da_seg_mask': da_seg_masks,
            'll_seg_mask': ll_seg_masks,
            'img_info': img_info
        }