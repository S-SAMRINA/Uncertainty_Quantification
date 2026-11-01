"""
temperature_scaling.py

Post-hoc calibration for the U-Net sweet pepper segmentation baseline via
temperature scaling (Guo et al., 2017 -- "On Calibration of Modern Neural
Networks").

Fits a single scalar temperature T on the VAL split by minimising NLL with
LBFGS over the full per-pixel logits (pixels subsampled per image during
fitting to bound memory), then reports segmentation + calibration metrics
on the EVAL split -- matching the output convention used by tta.py /
tta_uq_eval.py in this project (per-image figures + a nested metrics JSON
saved under OUTPUT_DIR / "temperature_scaling", same as OUTPUT_DIR / "tta").

IMPORTANT: T only rescales confidence -- it does NOT change the argmax
prediction, so mIoU / pixel accuracy / Dice are identical before and after.
This script only touches calibration metrics (ECE, Brier).

Requires: the real RGB images referenced by annotations_val.json /
annotations_eval.json to be reachable via DATA_DIR (see config.py). On the
cluster this is the BUP20 dataset directory (e.g.
<BUP20_ROOT>/CKA_sweet_pepper_2020_summer/<date>/<row>/).

Run from the Scripts directory:
    python temperature_scaling.py
or submit as a batch job (see scripts/slurm/temp_scaling_job.sh):
    sbatch scripts/slurm/temp_scaling_job.sh
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, DATA_DIR,
    CLASS_NAMES, NUM_CLASSES, NUM_WORKERS,
)
from dataset import PepperSegDataset, get_val_transforms
from model import build_model
from evaluate import (
    load_checkpoint,
    confusion_matrix_from_preds,
    metrics_from_confusion,
    colorize_mask,
)
from mc_dropout import calculate_ece, calculate_brier_score  # reuse existing metric code

PIXELS_PER_IMAGE_FOR_FIT = 20_000   # subsample cap while fitting T (memory)
N_FIGURES = 10                       # how many eval images to save panel figures for


# ── Temperature scaling module ────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Wraps a single learnable scalar T. logits_scaled = logits / T."""

    def __init__(self, init_T: float = 1.5):
        super().__init__()
        # Parameterise in log-space so T stays positive without constraining LBFGS.
        self.log_T = nn.Parameter(torch.log(torch.tensor(float(init_T))))

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


@torch.no_grad()
def collect_val_logits(model, dataset, device,
                        pixels_per_image: int = PIXELS_PER_IMAGE_FOR_FIT,
                        seed: int = 42):
    """
    Single pass over the val split with the frozen model. Keeps up to
    `pixels_per_image` randomly-sampled pixels per image so the LBFGS fit
    stays memory-light regardless of how large the val split is.

    Returns:
        logits : (N, C) float tensor
        labels : (N,)   long tensor
    """
    rng = np.random.default_rng(seed)
    model.eval()

    all_logits, all_labels = [], []
    for i in tqdm(range(len(dataset)), desc="Collecting val logits"):
        sample = dataset[i]
        image  = sample["image"].unsqueeze(0).to(device)
        mask   = sample["mask"].numpy()

        logits = model(image).squeeze(0).cpu()                 # (C, H, W)
        C, H, W = logits.shape
        logits_flat = logits.permute(1, 2, 0).reshape(-1, C)   # (H*W, C)
        labels_flat = mask.reshape(-1)                         # (H*W,)

        n_pixels = labels_flat.shape[0]
        n_keep   = min(pixels_per_image, n_pixels)
        keep_idx = rng.choice(n_pixels, n_keep, replace=False)

        all_logits.append(logits_flat[keep_idx])
        all_labels.append(torch.from_numpy(labels_flat[keep_idx]).long())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device,
                     max_iter: int = 100, lr: float = 0.01):
    """
    Fits a single scalar T minimising NLL (cross-entropy) over held-out
    (val-split) pixel logits, using LBFGS -- the standard recipe from
    Guo et al. 2017.
    """
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler(init_T=1.5).to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter)

    nll_before = nll_criterion(logits, labels).item()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        nll_after = nll_criterion(scaler(logits), labels).item()
        T_final = scaler.T.item()

    return T_final, nll_before, nll_after


# ── Visualization (matches tta.py's 6-panel style) ────────────────────────────

def save_visualization(image_rgb, gt_mask, pred_mask,
                        confidence_before, confidence_after,
                        output_path: Path, title: str):
    """
    6 panels: RGB image | Ground truth | Prediction | Error |
    Confidence before scaling | Confidence after scaling (T).
    """
    gt_color   = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)
    error      = (gt_mask != pred_mask).astype(np.uint8)

    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("RGB image")

    axes[1].imshow(gt_color)
    axes[1].set_title("Ground truth")

    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")

    axes[3].imshow(error, cmap="gray")
    axes[3].set_title("Error")

    im4 = axes[4].imshow(confidence_before, cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title("Confidence (before)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    im5 = axes[5].imshow(confidence_after, cmap="viridis", vmin=0, vmax=1)
    axes[5].set_title("Confidence (after T)")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_reliability_diagram(bins: dict, T: float, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = [
        (axes[0, 0], bins["before_all"], "Before scaling — All pixels"),
        (axes[0, 1], bins["after_all"],  f"After scaling (T={T:.2f}) — All pixels"),
        (axes[1, 0], bins["before_fg"],  "Before scaling — Foreground"),
        (axes[1, 1], bins["after_fg"],   f"After scaling (T={T:.2f}) — Foreground"),
    ]
    for ax, bin_data, title in panels:
        if bin_data:
            conf   = [b["confidence"] for b in bin_data]
            acc    = [b["accuracy"]   for b in bin_data]
            frac   = np.array([b["fraction"] for b in bin_data])
            widths = [b["bin_upper"] - b["bin_lower"] for b in bin_data]
            ax.bar(conf, acc, width=widths, alpha=0.75,
                   color=plt.cm.viridis(frac / frac.max()),
                   edgecolor="black", linewidth=0.5, align="center")
        ax.plot([0, 1], [0, 1], "--", color="red", linewidth=1.5, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
    fig.suptitle("Reliability Diagram — Temperature Scaling", fontsize=15)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reliability diagram -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(pretrained=False, dropout_p=0.0, device=device.type)
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = load_checkpoint(model, ckpt_path, device)
    model.eval()
    print("Checkpoint:", ckpt_path)
    print("Checkpoint epoch:", ckpt.get("epoch"))

    val_json  = DATA_DIR / "annotations_val.json"
    eval_json = DATA_DIR / "annotations_eval.json"
    val_ds  = PepperSegDataset(val_json,  "val",  get_val_transforms())
    eval_ds = PepperSegDataset(eval_json, "eval", get_val_transforms())

    if len(val_ds) == 0:
        raise RuntimeError(
            "Val dataset is empty -- cannot fit temperature. "
            "Check masks/val and that the RGB images referenced by "
            "annotations_val.json are actually reachable."
        )
    if len(eval_ds) == 0:
        raise RuntimeError("Eval dataset is empty. Check masks/eval and image paths.")

    # ── Fit T on val split ────────────────────────────────────────────────────
    print(f"\nCollecting val-split logits (up to {PIXELS_PER_IMAGE_FOR_FIT} px/image, "
          f"{len(val_ds)} images)...")
    val_logits, val_labels = collect_val_logits(model, val_ds, device)
    print(f"Fitting temperature on {val_logits.shape[0]} pixels...")
    T, nll_before, nll_after = fit_temperature(val_logits, val_labels, device)
    print(f"  T = {T:.4f}")
    print(f"  NLL before scaling: {nll_before:.4f}")
    print(f"  NLL after  scaling: {nll_after:.4f}")

    # ── Output dirs (mirrors OUTPUT_DIR / "tta" convention) ──────────────────
    ts_dir      = OUTPUT_DIR / "temperature_scaling"
    figure_dir  = ts_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate on eval split: segmentation + calibration before/after ──────
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    ece_all_b, ece_all_a   = [], []
    ece_fg_b,  ece_fg_a    = [], []
    brier_all_b, brier_all_a = [], []
    brier_fg_b,  brier_fg_a  = [], []

    bins_before_all, bins_before_fg = [], []
    bins_after_all,  bins_after_fg  = [], []

    processed = 0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print(f"\nEvaluating calibration on eval split ({len(eval_ds)} images)...")
    for batch in tqdm(eval_loader, desc="Temperature scaling eval"):
        image    = batch["image"].to(device)
        mask     = batch["mask"].to(device)
        img_path = batch["img_path"][0]
        stem     = Path(img_path).stem

        with torch.no_grad():
            logits = model(image)
            pred   = logits.argmax(dim=1)

        total_conf += confusion_matrix_from_preds(pred, mask, NUM_CLASSES)

        logits_np    = logits.squeeze(0).cpu().numpy()
        gt_np        = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        pred_np      = pred.squeeze(0).cpu().numpy().astype(np.uint8)

        proba_before = torch.softmax(torch.from_numpy(logits_np), dim=0).numpy()
        proba_after  = torch.softmax(torch.from_numpy(logits_np) / T, dim=0).numpy()

        ece_b,  bd_b  = calculate_ece(proba_before, gt_np, n_bins=15, foreground_only=False)
        ece_bf, bd_bf = calculate_ece(proba_before, gt_np, n_bins=15, foreground_only=True)
        ece_a,  bd_a  = calculate_ece(proba_after,  gt_np, n_bins=15, foreground_only=False)
        ece_af, bd_af = calculate_ece(proba_after,  gt_np, n_bins=15, foreground_only=True)

        brier_b  = calculate_brier_score(proba_before, gt_np, foreground_only=False)
        brier_bf = calculate_brier_score(proba_before, gt_np, foreground_only=True)
        brier_a  = calculate_brier_score(proba_after,  gt_np, foreground_only=False)
        brier_af = calculate_brier_score(proba_after,  gt_np, foreground_only=True)

        ece_all_b.append(ece_b);   ece_all_a.append(ece_a)
        ece_fg_b.append(ece_bf);   ece_fg_a.append(ece_af)
        brier_all_b.append(brier_b); brier_all_a.append(brier_a)
        brier_fg_b.append(brier_bf); brier_fg_a.append(brier_af)

        bins_before_all.extend(bd_b); bins_before_fg.extend(bd_bf)
        bins_after_all.extend(bd_a);  bins_after_fg.extend(bd_af)

        if processed < N_FIGURES:
            img_np = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip((img_np * std) + mean, 0.0, 1.0)
            confidence_before = proba_before.max(axis=0)
            confidence_after  = proba_after.max(axis=0)
            save_visualization(img_np, gt_np, pred_np,
                                confidence_before, confidence_after,
                                figure_dir / f"{stem}.png", stem)

        processed += 1

    segmentation = metrics_from_confusion(total_conf)

    bins = {
        "before_all": bins_before_all, "before_fg": bins_before_fg,
        "after_all":  bins_after_all,  "after_fg":  bins_after_fg,
    }
    save_reliability_diagram(bins, T, ts_dir / "reliability_diagram.png")

    results = {
        "method": "TemperatureScaling",
        "num_images": processed,
        "fit_split": "val",
        "n_val_images": len(val_ds),
        "n_val_pixels_used": int(val_logits.shape[0]),
        "temperature": T,
        "nll_val_before": nll_before,
        "nll_val_after": nll_after,

        "segmentation": {
            "pixel_accuracy":   segmentation["pixel_accuracy"],
            "miou_all":         segmentation["miou_all"],
            "mdice_all":        segmentation["mdice_all"],
            "miou_foreground":  segmentation["miou_foreground"],
            "mdice_foreground": segmentation["mdice_foreground"],
            "iou_per_class": {
                CLASS_NAMES[i]: float(segmentation["iou_per_class"][i])
                for i in range(NUM_CLASSES)
            },
            "dice_per_class": {
                CLASS_NAMES[i]: float(segmentation["dice_per_class"][i])
                for i in range(NUM_CLASSES)
            },
        },

        "calibration": {
            "ece_all_before":        float(np.mean(ece_all_b)),
            "ece_all_after":         float(np.mean(ece_all_a)),
            "ece_foreground_before": float(np.mean(ece_fg_b)),
            "ece_foreground_after":  float(np.mean(ece_fg_a)),
            "brier_all_before":        float(np.mean(brier_all_b)),
            "brier_all_after":         float(np.mean(brier_all_a)),
            "brier_foreground_before": float(np.mean(brier_fg_b)),
            "brier_foreground_after":  float(np.mean(brier_fg_a)),
        },

        "note": ("Temperature scaling only rescales confidence; argmax predictions "
                 "(and therefore mIoU / pixel accuracy / Dice under 'segmentation') "
                 "are IDENTICAL to the plain evaluate.py run -- only the 'calibration' "
                 "numbers change between _before and _after."),
    }

    metrics_path = ts_dir / "temperature_scaling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    report_path = ts_dir / "temperature_scaling_report.txt"
    with open(report_path, "w") as f:
        f.write("Temperature Scaling summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Fitted T: {T:.4f}\n")
        f.write(f"Num eval images: {processed}\n\n")
        f.write(f"{'Metric':<28}{'Before':>12}{'After':>12}\n")
        f.write(f"{'ECE (all pixels)':<28}{results['calibration']['ece_all_before']:>12.4f}"
                f"{results['calibration']['ece_all_after']:>12.4f}\n")
        f.write(f"{'ECE (foreground)':<28}{results['calibration']['ece_foreground_before']:>12.4f}"
                f"{results['calibration']['ece_foreground_after']:>12.4f}\n")
        f.write(f"{'Brier (all pixels)':<28}{results['calibration']['brier_all_before']:>12.4f}"
                f"{results['calibration']['brier_all_after']:>12.4f}\n")
        f.write(f"{'Brier (foreground)':<28}{results['calibration']['brier_foreground_before']:>12.4f}"
                f"{results['calibration']['brier_foreground_after']:>12.4f}\n\n")
        f.write("Segmentation (unaffected by T):\n")
        f.write(f"  Pixel accuracy:     {results['segmentation']['pixel_accuracy']:.4f}\n")
        f.write(f"  mIoU (foreground):  {results['segmentation']['miou_foreground']:.4f}\n")
        f.write(f"  mDice (foreground): {results['segmentation']['mdice_foreground']:.4f}\n")

    print("\n" + "=" * 65)
    print("TEMPERATURE SCALING — FINAL SUMMARY")
    print("=" * 65)
    print(f"Fitted T:               {T:.4f}")
    print(f"{'Metric':<28}{'Before':>14}{'After':>14}")
    print(f"{'ECE (all pixels)':<28}{results['calibration']['ece_all_before']:>14.4f}"
          f"{results['calibration']['ece_all_after']:>14.4f}")
    print(f"{'ECE (foreground)':<28}{results['calibration']['ece_foreground_before']:>14.4f}"
          f"{results['calibration']['ece_foreground_after']:>14.4f}")
    print(f"{'Brier (all pixels)':<28}{results['calibration']['brier_all_before']:>14.4f}"
          f"{results['calibration']['brier_all_after']:>14.4f}")
    print(f"{'Brier (foreground)':<28}{results['calibration']['brier_foreground_before']:>14.4f}"
          f"{results['calibration']['brier_foreground_after']:>14.4f}")
    print("=" * 65)
    print("\nSaved:", metrics_path)
    print("Report:", report_path)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()
