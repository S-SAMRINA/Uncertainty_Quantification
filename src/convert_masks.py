"""
convert_masks.py

Converts COCO polygon annotations to per-image semantic segmentation masks.
Each output PNG is the same size as the source image.
Pixel value = class ID (0=background, 1=red, 2=yellow, 3=green,
                        4=mixed_red, 5=mixed_yellow).

Run once before training:
    python convert_masks.py
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as coco_mask
from tqdm import tqdm

from config import (
    TRAIN_JSON, VAL_JSON, EVAL_JSON,
    DATA_DIR, MASK_DIR, CLASS_NAMES
)

# ── Category ID → class index mapping ────────────────────────────────────────
# Will be built dynamically from the JSON so order matches CLASS_NAMES exactly.

def build_cat_map(categories):
    """
    Maps COCO category_id → our class index (1-based, 0=background).
    Matches by name to CLASS_NAMES.
    """
    name_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    cat_map = {}
    for cat in categories:
        name = cat["name"].lower().replace(" ", "_")
        if name in name_to_idx:
            cat_map[cat["id"]] = name_to_idx[name]
        else:
            print(f"  WARNING: unknown category '{cat['name']}' — skipped")
    return cat_map


def polygon_to_mask(segmentation, height, width):
    """Convert a COCO polygon segmentation to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def convert_split(json_path, split_name):
    """Convert all annotations in one JSON split to PNG masks."""
    print(f"\n[{split_name}] Loading {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    cat_map = build_cat_map(data["categories"])
    print(f"  Category map: {cat_map}")

    # image_id → image info
    img_info = {img["id"]: img for img in data["images"]}

    # image_id → list of annotations
    ann_by_image = {}
    for ann in data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    out_dir = MASK_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for img_id, info in tqdm(img_info.items(), desc=f"  {split_name}"):
        h, w = info["height"], info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background

        for ann in ann_by_image.get(img_id, []):
            class_idx = cat_map.get(ann["category_id"])
            if class_idx is None:
                skipped += 1
                continue

            # Handle both polygon and RLE annotations
            if isinstance(ann["segmentation"], dict):
                # RLE format
                rle   = coco_mask.frPyObjects(ann["segmentation"], h, w)
                bmask = coco_mask.decode(rle).astype(np.uint8)
            else:
                # Polygon format
                bmask = polygon_to_mask(ann["segmentation"], h, w)

            # Write class index into mask (later annotations overwrite earlier ones)
            mask[bmask > 0] = class_idx

        # Save as PNG — lossless, pixel values preserved exactly
        stem     = Path(info["file_name"]).stem
        out_path = out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"  Saved {len(img_info)} masks to {out_dir}")
    if skipped:
        print(f"  Skipped {skipped} annotations (unknown category)")


if __name__ == "__main__":
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    for split, path in [
        ("train", TRAIN_JSON),
        ("val",   VAL_JSON),
        ("eval",  EVAL_JSON),
    ]:
        if not Path(path).exists():
            print(f"  WARNING: {path} not found — skipping {split}")
            continue
        convert_split(path, split)

    print("\nDone. Masks saved to:", MASK_DIR)
    print("Next step: python unet/train.py")