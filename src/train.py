"""
train.py
Semantic segmentation training script for the sweet pepper dataset.

Features:
- Train/validation loop
- CrossEntropy + Dice loss
- mIoU and pixel accuracy
- ReduceLROnPlateau scheduler
- Early stopping
- Best/last checkpoint saving
- AMP mixed precision on CUDA
- Works with config.py, dataset.py, and model.py

Run from the Scripts directory:
    python train.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make local imports work when this file is run directly from Scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    PLOT_DIR,
    PRED_DIR,
    UQ_DIR,
    TRAIN_JSON,
    VAL_JSON,
    EVAL_JSON,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    LR_PATIENCE,
    EARLY_STOP,
    NUM_WORKERS,
    SEED,
    PRETRAINED,
    DICE_WEIGHT,
)
from dataset import PepperSegDataset, get_train_transforms, get_val_transforms
from model import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dirs() -> None:
    for p in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, PRED_DIR, UQ_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def resolve_json_path(split: str) -> Path:
    """
    Robust JSON resolver.
    Your config.py currently has VAL_JSON pointing to annotations_valid.json,
    but the file in DATA_DIR is annotations_val.json. This helper handles both.
    """
    split = split.lower()

    if split == "train":
        candidates = [Path(TRAIN_JSON), DATA_DIR / "annotations_train.json"]
    elif split == "val":
        candidates = [
            Path(VAL_JSON),
            DATA_DIR / "annotations_val.json",
            DATA_DIR / "annotations_valid.json",
        ]
    elif split == "eval":
        candidates = [Path(EVAL_JSON), DATA_DIR / "annotations_eval.json"]
    else:
        raise ValueError(f"Unknown split: {split}")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find annotation JSON for split '{split}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def make_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    train_json = resolve_json_path("train")
    val_json = resolve_json_path("val")
    eval_json = resolve_json_path("eval")

    train_ds = PepperSegDataset(train_json, "train", get_train_transforms())
    val_ds = PepperSegDataset(val_json, "val", get_val_transforms())
    eval_ds = PepperSegDataset(eval_json, "eval", get_val_transforms())

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check that masks/train exists and that "
            "DATA_DIR/images contains the training images."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset is empty. Fix VAL_JSON, rerun convert_masks.py, "
            "and make sure masks/val exists."
        )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, eval_loader


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets.long(), num_classes=self.num_classes)  # [B,H,W,C]
        targets_1h = targets_1h.permute(0, 3, 1, 2).float()                   # [B,C,H,W]

        if self.ignore_background and self.num_classes > 1:
            probs = probs[:, 1:, :, :]
            targets_1h = targets_1h[:, 1:, :, :]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_1h, dims)
        denominator = torch.sum(probs, dims) + torch.sum(targets_1h, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes, ignore_background=True)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def fast_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()

    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    idx = targets * num_classes + preds
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.reshape(num_classes, num_classes)


def confusion_to_metrics(conf: torch.Tensor) -> Dict[str, object]:
    conf = conf.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn

    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    acc = tp.sum() / conf.sum().clamp(min=1.0)

    valid = denom > 0
    miou_all = iou[valid].mean().item() if valid.any() else 0.0

    fg_valid = valid[1:]
    fg_iou = iou[1:]
    miou_fg = fg_iou[fg_valid].mean().item() if fg_valid.any() else 0.0

    return {
        "pixel_accuracy": acc.item(),
        "iou_per_class": iou.cpu().tolist(),
        "miou_all": miou_all,
        "miou_foreground": miou_fg,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_conf += fast_confusion_matrix(preds, masks, NUM_CLASSES)

        metrics = confusion_to_metrics(total_conf)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{metrics['pixel_accuracy']:.4f}", miou=f"{metrics['miou_foreground']:.4f}")

    metrics = confusion_to_metrics(total_conf)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_conf += fast_confusion_matrix(preds, masks, NUM_CLASSES)

        metrics = confusion_to_metrics(total_conf)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{metrics['pixel_accuracy']:.4f}", miou=f"{metrics['miou_foreground']:.4f}")

    metrics = confusion_to_metrics(total_conf)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    history: Dict[str, List[float]],
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "history": history,
        "config": {
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dice_weight": DICE_WEIGHT,
            "seed": SEED,
        },
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(pretrained=PRETRAINED, dropout_p=0.0, device=device.type)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, dice_weight=DICE_WEIGHT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
        verbose=True,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_miou_foreground": [],
        "val_miou_foreground": [],
        "lr": [],
    }

    print("\nStarting training...")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Classes: {CLASS_NAMES}")

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["pixel_accuracy"])
        history["val_acc"].append(val_metrics["pixel_accuracy"])
        history["train_miou_foreground"].append(train_metrics["miou_foreground"])
        history["val_miou_foreground"].append(val_metrics["miou_foreground"])
        history["lr"].append(current_lr)

        elapsed = time.time() - start

        print(f"  train loss: {train_metrics['loss']:.4f} | val loss: {val_metrics['loss']:.4f}")
        print(f"  train acc : {train_metrics['pixel_accuracy']:.4f} | val acc : {val_metrics['pixel_accuracy']:.4f}")
        print(f"  train mIoU(fg): {train_metrics['miou_foreground']:.4f} | val mIoU(fg): {val_metrics['miou_foreground']:.4f}")
        print(f"  lr: {current_lr:.2e} | time: {elapsed/60:.1f} min")

        # Always save last checkpoint
        save_checkpoint(
            CHECKPOINT_DIR / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            history,
            extra={"epoch_metrics": {"train": train_metrics, "val": val_metrics}},
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0

            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                history,
                extra={"best_epoch": best_epoch, "best_val_metrics": val_metrics},
            )
            print(f"  ✓ Saved new best checkpoint (epoch {epoch})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement for {epochs_no_improve} epoch(s)")

        with open(OUTPUT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= EARLY_STOP:
            print(f"\nEarly stopping triggered after {EARLY_STOP} epochs without improvement.")
            break

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Last checkpoint: {CHECKPOINT_DIR / 'last_model.pth'}")
    print(f"History saved to: {OUTPUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()
