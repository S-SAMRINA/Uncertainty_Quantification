"""
msp_uq.py

Maximum Softmax Probability (MSP) uncertainty quantification for the U-Net
sweet pepper segmentation baseline (Hendrycks & Gimpel, 2017 -- "A Baseline
for Detecting Misclassified and Out-of-Distribution Examples").

The simplest possible UQ signal: uncertainty = 1 - max(softmax probability)
per pixel, from a single deterministic forward pass -- no extra training, no
multiple forward passes (unlike MC Dropout / TTA), no fitting step (unlike
temperature scaling). This is the naive baseline every other UQ method in
this project should be able to beat.

Reports:
  - segmentation metrics (identical to evaluate.py -- MSP doesn't change
    predictions, it only scores how confident the model is in them)
  - calibration (ECE / Brier), reusing mc_dropout.py's helper functions
  - uncertainty quality: AUROC / AUPR of (1 - max softmax) as a pixel-level
    error detector, aggregated across the whole eval split (same recipe as
    mc_dropout.py's global AUROC/AUPR)

Output convention matches temperature_scaling.py / tta_uq_eval.py in this
project: OUTPUT_DIR / "msp_uq" / {figures/, msp_uq_metrics.json, msp_uq_report.txt}

Run from the Scripts directory:
    python msp_uq.py
or submit as a batch job (see scripts/slurm/msp_uq_job.sh):
    sbatch scripts/slurm/msp_uq_job.sh
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch
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
from mc_dropout import calculate_ece, calculate_brier_score, calculate_uncertainty_auroc_aupr

N_FIGURES = 10                  # how many eval images to save panel figures for
MAX_PIX_FOR_AUROC = 5_000_000   # subsample cap for global AUROC/AUPR (memory)


# ── Visualization ──────────────────────────────────────────────────────────────

def save_visualization(image_rgb, gt_mask, pred_mask, msp_uncertainty,
                        output_path: Path, title: str):
    """5 panels: RGB | Ground truth | Prediction | Error | MSP uncertainty."""
    gt_color   = colorize_mask(gt_mask)
    pred_color = colorize_mask(pred_mask)
    error      = (gt_mask != pred_mask).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(21, 5))

    axes[0].imshow(image_rgb);          axes[0].set_title("RGB image")
    axes[1].imshow(gt_color);           axes[1].set_title("Ground truth")
    axes[2].imshow(pred_color);         axes[2].set_title("Prediction")
    axes[3].imshow(error, cmap="gray"); axes[3].set_title("Error")

    im4 = axes[4].imshow(msp_uncertainty, cmap="inferno", vmin=0, vmax=1)
    axes[4].set_title("MSP uncertainty\n(1 - max softmax)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


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

    eval_json = DATA_DIR / "annotations_eval.json"
    eval_ds   = PepperSegDataset(eval_json, "eval", get_val_transforms())
    if len(eval_ds) == 0:
        raise RuntimeError("Eval dataset is empty. Check masks/eval and image paths.")

    out_dir    = OUTPUT_DIR / "msp_uq"
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    total_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    per_img_ece, per_img_ece_fg     = [], []
    per_img_brier, per_img_brier_fg = [], []

    all_unc, all_pred, all_gt = [], [], []   # for global AUROC/AUPR

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed = 0
    print(f"\nRunning MSP uncertainty on eval split ({len(eval_ds)} images)...")
    for batch in tqdm(eval_loader, desc="MSP eval"):
        image    = batch["image"].to(device)
        mask     = batch["mask"].to(device)
        img_path = batch["img_path"][0]
        stem     = Path(img_path).stem

        with torch.no_grad():
            logits = model(image)
            proba  = torch.softmax(logits, dim=1)
            pred   = proba.argmax(dim=1)

        total_conf += confusion_matrix_from_preds(pred, mask, NUM_CLASSES)

        proba_np = proba.squeeze(0).cpu().numpy()      # (C, H, W)
        gt_np    = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        pred_np  = pred.squeeze(0).cpu().numpy().astype(np.uint8)

        confidence = proba_np.max(axis=0)              # (H, W) max softmax prob
        msp_unc    = 1.0 - confidence                  # (H, W) MSP uncertainty

        ece,    bd    = calculate_ece(proba_np, gt_np, n_bins=15, foreground_only=False)
        ece_fg, bd_fg = calculate_ece(proba_np, gt_np, n_bins=15, foreground_only=True)
        brier    = calculate_brier_score(proba_np, gt_np, foreground_only=False)
        brier_fg = calculate_brier_score(proba_np, gt_np, foreground_only=True)

        per_img_ece.append(ece);     per_img_ece_fg.append(ece_fg)
        per_img_brier.append(brier); per_img_brier_fg.append(brier_fg)

        all_unc.append(msp_unc.reshape(-1))
        all_pred.append(pred_np.reshape(-1))
        all_gt.append(gt_np.reshape(-1))

        if processed < N_FIGURES:
            img_np = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
            img_np = np.clip((img_np * std) + mean, 0.0, 1.0)
            save_visualization(img_np, gt_np, pred_np, msp_unc,
                                figure_dir / f"{stem}.png", stem)

        processed += 1

    segmentation = metrics_from_confusion(total_conf)

    # ── Global AUROC / AUPR (pool pixels across all images, subsample if huge) ──
    all_unc_np  = np.concatenate(all_unc)
    all_pred_np = np.concatenate(all_pred)
    all_gt_np   = np.concatenate(all_gt)

    if len(all_unc_np) > MAX_PIX_FOR_AUROC:
        rng  = np.random.default_rng(42)
        keep = rng.choice(len(all_unc_np), MAX_PIX_FOR_AUROC, replace=False)
        all_unc_np, all_pred_np, all_gt_np = all_unc_np[keep], all_pred_np[keep], all_gt_np[keep]

    auroc_all, aupr_all = calculate_uncertainty_auroc_aupr(
        all_unc_np, all_pred_np, all_gt_np, foreground_only=False)
    auroc_fg, aupr_fg = calculate_uncertainty_auroc_aupr(
        all_unc_np, all_pred_np, all_gt_np, foreground_only=True)

    results = {
        "method": "MSP",  # Maximum Softmax Probability (Hendrycks & Gimpel, 2017)
        "num_images": processed,

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
            "ece_all":          float(np.mean(per_img_ece)),
            "ece_foreground":   float(np.mean(per_img_ece_fg)),
            "brier_all":        float(np.mean(per_img_brier)),
            "brier_foreground": float(np.mean(per_img_brier_fg)),
        },

        "uncertainty_quality": {
            "auroc_all_pixels": auroc_all,
            "auroc_foreground": auroc_fg,
            "aupr_all_pixels":  aupr_all,
            "aupr_foreground":  aupr_fg,
        },

        "note": ("MSP (1 - max softmax probability) is the naive single-pass "
                 "uncertainty baseline -- no extra training or forward passes "
                 "beyond a normal prediction. Segmentation metrics are identical "
                 "to evaluate.py since MSP does not alter predictions, it only "
                 "scores confidence in them."),
    }

    metrics_path = out_dir / "msp_uq_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    report_path = out_dir / "msp_uq_report.txt"
    with open(report_path, "w") as f:
        f.write("MSP (Maximum Softmax Probability) UQ summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Num eval images: {processed}\n\n")
        f.write("Segmentation:\n")
        f.write(f"  Pixel accuracy:     {results['segmentation']['pixel_accuracy']:.4f}\n")
        f.write(f"  mIoU (foreground):  {results['segmentation']['miou_foreground']:.4f}\n")
        f.write(f"  mDice (foreground): {results['segmentation']['mdice_foreground']:.4f}\n\n")
        f.write("Calibration:\n")
        f.write(f"  ECE (all pixels):    {results['calibration']['ece_all']:.4f}\n")
        f.write(f"  ECE (foreground):    {results['calibration']['ece_foreground']:.4f}\n")
        f.write(f"  Brier (all pixels):  {results['calibration']['brier_all']:.4f}\n")
        f.write(f"  Brier (foreground):  {results['calibration']['brier_foreground']:.4f}\n\n")
        f.write("Uncertainty quality (error detection):\n")
        f.write(f"  AUROC (all pixels):  {results['uncertainty_quality']['auroc_all_pixels']:.4f}\n")
        f.write(f"  AUROC (foreground):  {results['uncertainty_quality']['auroc_foreground']:.4f}\n")
        f.write(f"  AUPR  (all pixels):  {results['uncertainty_quality']['aupr_all_pixels']:.4f}\n")
        f.write(f"  AUPR  (foreground):  {results['uncertainty_quality']['aupr_foreground']:.4f}\n")

    print("\n" + "=" * 65)
    print("MSP UNCERTAINTY — FINAL SUMMARY")
    print("=" * 65)
    print(f"Pixel accuracy:      {results['segmentation']['pixel_accuracy']:.4f}")
    print(f"mIoU (foreground):   {results['segmentation']['miou_foreground']:.4f}")
    print(f"ECE (foreground):    {results['calibration']['ece_foreground']:.4f}")
    print(f"Brier (foreground):  {results['calibration']['brier_foreground']:.4f}")
    print(f"AUROC (foreground):  {results['uncertainty_quality']['auroc_foreground']:.4f}")
    print(f"AUPR  (foreground):  {results['uncertainty_quality']['aupr_foreground']:.4f}")
    print("=" * 65)
    print("\nSaved:", metrics_path)
    print("Report:", report_path)
    print("Figures:", figure_dir)


if __name__ == "__main__":
    main()
