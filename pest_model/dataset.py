import random
import os
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A


class PestDataset(Dataset):
    def mixup(self, img1, label1, img2, label2, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        img = lam * img1 + (1 - lam) * img2
        label = lam * label1 + (1 - lam) * label2
        return img, label
    """Image folder dataset. Expects structure: root/{train,test}/{class_name}/*.jpg

    Returns: (image_tensor, label_int)
    """

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root) / split
        assert self.root.exists(), f"Dataset path {self.root} does not exist"
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_path = self.root / cls
            for img in cls_path.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((img, self.class_to_idx[cls]))

        self.transform = transform or get_default_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose):
            augmented = self.transform(image=img)
            img = augmented["image"]
        elif callable(self.transform):
            img = self.transform(img)

        # For train split, always return one-hot labels
        if hasattr(self, 'mixup_enabled') and self.mixup_enabled:
            label_onehot = np.zeros(len(self.classes), dtype=np.float32)
            label_onehot[label] = 1.0
            if random.random() < 0.5:
                idx2 = random.randint(0, len(self.samples) - 1)
                path2, label2 = self.samples[idx2]
                img2 = cv2.imread(str(path2))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                if isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose):
                    img2 = self.transform(image=img2)["image"]
                elif callable(self.transform):
                    img2 = self.transform(img2)
                label2_onehot = np.zeros(len(self.classes), dtype=np.float32)
                label2_onehot[label2] = 1.0
                img, label_onehot = self.mixup(img, label_onehot, img2, label2_onehot)
            img = img.astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
            label = torch.from_numpy(label_onehot)
            return img, label
        # ensure float tensor CHW
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
        return img, label


def get_default_transforms(split: str = "train", size: int = 224):
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(),
        ])
    else:
        return A.Compose([
            A.Resize(width=size, height=size),
            A.Normalize(),
        ])


def make_loader(root: str, split: str, batch_size: int = 16, num_workers: int = 4, shuffle: bool = True, size: int = 224):
    ds = PestDataset(root, split=split, transform=get_default_transforms(split, size))
    ds.mixup_enabled = False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader
