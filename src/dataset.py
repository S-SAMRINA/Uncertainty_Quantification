"""
dataset.py

PyTorch Dataset for sweet pepper semantic segmentation.
Loads RGB images + corresponding PNG masks, applies augmentation.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMAGE_SIZE, NUM_CLASSES, MASK_DIR, DATA_DIR


# ── ImageNet normalisation ────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class PepperSegDataset(Dataset):
    """
    Dataset for sweet pepper semantic segmentation.

    Args:
        json_path : path to COCO-format JSON (annotations_train.json etc.)
        split     : 'train', 'val', or 'eval' — used to locate mask files
        transform : albumentations transform pipeline
    """

    def __init__(self, json_path, split, transform=None):
        self.split     = split
        self.transform = transform
        self.mask_dir  = MASK_DIR / split

        with open(json_path) as f:
            data = json.load(f)

        # Build list of (image_path, mask_path) pairs
        self.samples = []
        for img_info in data["images"]:
            stem       = Path(img_info["file_name"]).stem
            img_path   = str(DATA_DIR / "images" / img_info["file_name"])
            mask_path  = str(self.mask_dir / f"{stem}.png")

            # Only include if both files exist
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path, img_info["id"]))
            else:
                if not os.path.exists(img_path):
                    print(f"  WARNING: image not found: {img_path}")
                if not os.path.exists(mask_path):
                    print(f"  WARNING: mask not found: {mask_path}")

        print(f"[{split}] {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, img_id = self.samples[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (single-channel PNG, pixel = class index)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Clip mask values to valid range
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]          # float32 tensor (3, H, W)
            mask  = transformed["mask"].long()    # int64 tensor (H, W)
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return {
            "image":    image,
            "mask":     mask,
            "image_id": img_id,
            "img_path": img_path,
        }


def get_dataloaders(batch_size=4, num_workers=4):
    """Returns train, val, eval DataLoaders."""
    from torch.utils.data import DataLoader
    from config import TRAIN_JSON, VAL_JSON, EVAL_JSON, BATCH_SIZE, NUM_WORKERS

    train_ds = PepperSegDataset(TRAIN_JSON, "train", get_train_transforms())
    val_ds   = PepperSegDataset(VAL_JSON,   "val",   get_val_transforms())
    eval_ds  = PepperSegDataset(EVAL_JSON,  "eval",  get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=1,
                               shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)

    return train_loader, val_loader, eval_loader