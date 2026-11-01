"""
evaluate.py
Semantic segmentation evaluation script for the sweet pepper dataset.

What it does:
- Loads the best trained checkpoint
- Runs inference on the eval split
- Saves predicted masks as PNGs
- Computes per-class IoU and Dice
- Saves side-by-side visualizations:
    image | ground truth | prediction | error map
- Writes a JSON metrics summary

Run from the Scripts directory:
    python evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Make local imports work when this file is run directly from Scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    PRED_DIR,
    PLOT_DIR,
    EVAL_JSON,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_WORKERS,
    PRETRAINED,
)
from dataset import PepperSegDataset, get_val_transforms
from model import build_model


def ensure_dirs() -> Tuple[Path, Path, Path]:
    pred_dir = PRED_DIR / "eval"
    viz_dir = PLOT_DIR / "eval_examples"
    metrics_dir = OUTPUT_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return pred_dir, viz_dir, metrics_dir


def resolve_eval_json() -> Path:
    candidates = [Path(EVAL_JSON), DATA_DIR / "annotations_eval.json"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find eval JSON. Tried: {candidates}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)

    return ckpt


@torch.no_grad()
def confusion_matrix_from_preds(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]
    idx = targets * num_classes + preds
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes)


def metrics_from_confusion(conf: torch.Tensor) -> Dict[str, object]:
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn

    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    pixel_acc = tp.sum() / conf.sum().clamp(min=1.0)

    dice_denom = (2 * tp + fp + fn)
    dice = torch.where(dice_denom > 0, (2 * tp) / dice_denom, torch.zeros_like(tp))

    valid = denom > 0
    miou_all = iou[valid].mean().item() if valid.any() else 0.0
    mdice_all = dice[valid].mean().item() if valid.any() else 0.0

    fg_valid = valid[1:]
    miou_fg = iou[1:][fg_valid].mean().item() if fg_valid.any() else 0.0
    mdice_fg = dice[1:][fg_valid].mean().item() if fg_valid.any() else 0.0

    return {
        "pixel_accuracy": pixel_acc.item(),
        "iou_per_class": iou.cpu().tolist(),
        "dice_per_class": dice.cpu().tolist(),
        "miou_all": miou_all,
        "mdice_all": mdice_all,
        "miou_foreground": miou_fg,
        "mdice_foreground": mdice_fg,
    }


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        (0, 0, 0),
        (220, 50, 50),
        (240, 200, 50),
        (50, 180, 50),
        (180, 100, 50),
        (180, 180, 50),
    ], dtype=np.uint8)
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def save_visualization(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, out_path: Path, title: str = ""):
    gt_vis = colorize_mask(gt_mask)
    pred_vis = colorize_mask(pred_mask)
    err = (gt_mask != pred_mask).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(gt_vis)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_vis)
    axes[2].set_title("Prediction")
    axes[3].imshow(err, cmap="gray")
    axes[3].set_title("Error Map")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir, viz_dir, metrics_dir = ensure_dirs()

    eval_json = resolve_eval_json()
    print(f"Using device: {device}")
    print(f"Eval JSON: {eval_json}")

    eval_ds = PepperSegDataset(eval_json, "eval", get_val_transforms())
    if len(eval_ds) == 0:
        raise RuntimeError("Eval dataset is empty. Check masks/eval and image paths.")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(pretrained=PRETRAINED, dropout_p=0.0, device=device.type)
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = load_checkpoint(model, ckpt_path, device)
    model.eval()

    if "epoch" in ckpt:
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Checkpoint: {ckpt_path}")

    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    print("\nRunning inference on eval split...")
    for batch in tqdm(eval_loader, desc="Eval"):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        img_path = batch["img_path"][0]

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1)

        total_conf += confusion_matrix_from_preds(pred, mask, NUM_CLASSES)

        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)
        gt_np = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)

        stem = Path(img_path).stem
        cv2.imwrite(str(pred_dir / f"{stem}.png"), pred_np)

        # save a handful of visualizations
        if len(list(viz_dir.glob("*.png"))) < 12:
            img_np = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = np.clip((img_np * std) + mean, 0.0, 1.0)
            save_visualization(img_np, gt_np, pred_np, viz_dir / f"{stem}.png", title=stem)

    summary = metrics_from_confusion(total_conf)

    metrics = {
        "split": "eval",
        "num_samples": len(eval_ds),
        "class_names": CLASS_NAMES,
        "pixel_accuracy": summary["pixel_accuracy"],
        "miou_all": summary["miou_all"],
        "mdice_all": summary["mdice_all"],
        "miou_foreground": summary["miou_foreground"],
        "mdice_foreground": summary["mdice_foreground"],
        "iou_per_class": {CLASS_NAMES[i]: float(summary["iou_per_class"][i]) for i in range(NUM_CLASSES)},
        "dice_per_class": {CLASS_NAMES[i]: float(summary["dice_per_class"][i]) for i in range(NUM_CLASSES)},
        "checkpoint": str(ckpt_path),
    }

    metrics_path = metrics_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    report_path = metrics_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write("Eval summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Num samples: {metrics['num_samples']}\n")
        f.write(f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"mIoU (all): {metrics['miou_all']:.4f}\n")
        f.write(f"mDice (all): {metrics['mdice_all']:.4f}\n")
        f.write(f"mIoU (foreground): {metrics['miou_foreground']:.4f}\n")
        f.write(f"mDice (foreground): {metrics['mdice_foreground']:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for k, v in metrics["iou_per_class"].items():
            f.write(f"  {k:>15}: {v:.4f}\n")
        f.write("\nPer-class Dice:\n")
        for k, v in metrics["dice_per_class"].items():
            f.write(f"  {k:>15}: {v:.4f}\n")

    print("\nDone.")
    print(f"Predicted masks saved to: {pred_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Metrics JSON saved to: {metrics_path}")
    print(f"Text report saved to: {report_path}")
    print("\nSummary:")
    print(f"  Pixel accuracy      : {metrics['pixel_accuracy']:.4f}")
    print(f"  mIoU (all classes)  : {metrics['miou_all']:.4f}")
    print(f"  mDice (all classes) : {metrics['mdice_all']:.4f}")
    print(f"  mIoU (foreground)   : {metrics['miou_foreground']:.4f}")
    print(f"  mDice (foreground)  : {metrics['mdice_foreground']:.4f}")


if __name__ == "__main__":
    main()
