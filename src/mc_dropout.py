from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    OUTPUT_DIR, CHECKPOINT_DIR, DATA_DIR,
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, PRETRAINED,
    MC_DROPOUT_T, MC_DROPOUT_P, NUM_WORKERS,
)
from dataset import PepperSegDataset, get_val_transforms
from model import build_model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PALETTE = np.array([
    [0,   0,   0  ],
    [220, 50,  50 ],
    [240, 200, 50 ],
    [50,  180, 50 ],
    [180, 100, 50 ],
    [180, 180, 50 ],
], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * STD + MEAN, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    return PALETTE[mask]


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray,
                              alpha: float = 0.55) -> np.ndarray:
    cmap = plt.get_cmap("inferno")
    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (image_rgb.astype(np.float32) * (1 - alpha) +
               heat_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── MC Dropout inference ──────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_forward(model: nn.Module, image_tensor: torch.Tensor,
                       device: torch.device, T: int = 30):
    """
    Returns:
        mean_proba    : (C, H, W) mean softmax probability
        pred_mask     : (H, W) argmax prediction
        entropy_map   : (H, W) normalised predictive entropy [0,1] — total uncertainty
        variance_map  : (H, W) normalised mean per-class variance [0,1] — epistemic proxy
        confidence_map: (H, W) max softmax probability [0,1]
        mi_map        : (H, W) normalised mutual information [0,1] — cleaner epistemic estimate
        proba_stack   : (T, C, H, W) per-pass probabilities
    """
    model.eval()
    model.enable_dropout()

    image_tensor = image_tensor.unsqueeze(0).to(device)

    proba_stack = []
    for _ in range(T):
        logits = model(image_tensor)
        proba  = torch.softmax(logits, dim=1)
        proba_stack.append(proba.cpu())

    proba_stack = torch.cat(proba_stack, dim=0)   # (T, C, H, W)
    mean_proba  = proba_stack.mean(dim=0)         # (C, H, W)

    # Total uncertainty — predictive entropy of the mean
    p = mean_proba.clamp(min=1e-7)
    entropy = -(p * p.log()).sum(dim=0)
    entropy_norm = (entropy / np.log(NUM_CLASSES)).numpy()

    # Epistemic proxy — mean per-class variance across passes
    variance = proba_stack.var(dim=0).mean(dim=0)
    variance_norm = np.clip(variance.numpy() / 0.05, 0.0, 1.0)

    # Confidence
    confidence_map = mean_proba.max(dim=0).values.numpy()

    # ── Mutual Information (epistemic uncertainty) ────────────────────────────
    # MI = H[E_p] − E_p[H]
    # Total entropy of the mean minus the mean entropy of each stochastic pass.
    # Captures how much the individual passes *disagree* — pure epistemic signal.
    per_pass_entropy = -(proba_stack.clamp(min=1e-7) *
                         proba_stack.clamp(min=1e-7).log()).sum(dim=1)  # (T, H, W)
    mean_per_pass_entropy = per_pass_entropy.mean(dim=0)                # (H, W)
    mi = entropy - mean_per_pass_entropy                                 # (H, W)
    mi = mi.clamp(min=0.0)   # numerical safety; MI >= 0 by definition
    mi_norm = (mi / np.log(NUM_CLASSES)).numpy()

    pred_mask = mean_proba.argmax(dim=0).numpy().astype(np.uint8)

    return (mean_proba.numpy(), pred_mask, entropy_norm, variance_norm,
            confidence_map, mi_norm, proba_stack)


# ── Deterministic pass ────────────────────────────────────────────────────────

@torch.no_grad()
def deterministic_forward(model: nn.Module, image_tensor: torch.Tensor,
                           device: torch.device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    logits = model(image_tensor)
    proba  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred   = proba.argmax(axis=0).astype(np.uint8)
    return proba, pred


# ── Confusion matrix / segmentation metrics ───────────────────────────────────

def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (gt >= 0) & (gt < num_classes)
    idx  = num_classes * gt[mask].astype(int) + pred[mask].astype(int)
    conf = np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def metrics_from_confusion(conf: np.ndarray) -> dict:
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom_iou  = tp + fp + fn
    denom_dice = 2 * tp + fp + fn

    iou  = np.divide(tp, denom_iou,  out=np.zeros_like(tp), where=denom_iou  > 0)
    dice = np.divide(2 * tp, denom_dice, out=np.zeros_like(tp), where=denom_dice > 0)

    pixel_acc = tp.sum() / max(conf.sum(), 1)
    valid = denom_iou > 0
    miou_all  = iou[valid].mean()  if valid.any()      else 0.0
    mdice_all = dice[valid].mean() if valid.any()      else 0.0

    fg_valid   = valid[1:]
    miou_fg    = iou[1:][fg_valid].mean()  if fg_valid.any() else 0.0
    mdice_fg   = dice[1:][fg_valid].mean() if fg_valid.any() else 0.0

    return {
        "pixel_accuracy":    float(pixel_acc),
        "iou_per_class":     iou.tolist(),
        "dice_per_class":    dice.tolist(),
        "miou_all":          float(miou_all),
        "mdice_all":         float(mdice_all),
        "miou_foreground":   float(miou_fg),
        "mdice_foreground":  float(mdice_fg),
    }


# ── ECE ───────────────────────────────────────────────────────────────────────

def calculate_ece(probs: np.ndarray, labels: np.ndarray,
                  n_bins: int = 15, foreground_only: bool = False):
    pred       = np.argmax(probs, axis=0).reshape(-1)
    confidence = np.max(probs, axis=0).reshape(-1)
    labels     = labels.reshape(-1)

    if foreground_only:
        m = labels != 0
        pred, confidence, labels = pred[m], confidence[m], labels[m]

    if len(labels) == 0:
        return 0.0, []

    correct     = (pred == labels).astype(np.float32)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bin_data = 0.0, []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
        count  = in_bin.sum()
        if count == 0:
            continue
        bin_conf = confidence[in_bin].mean()
        bin_acc  = correct[in_bin].mean()
        frac     = count / len(labels)
        ece     += frac * abs(bin_acc - bin_conf)
        bin_data.append({"bin_lower": float(lo), "bin_upper": float(hi),
                         "confidence": float(bin_conf), "accuracy": float(bin_acc),
                         "fraction": float(frac), "count": int(count)})

    return float(ece), bin_data


# ── Brier score ───────────────────────────────────────────────────────────────

def calculate_brier_score(probs: np.ndarray, labels: np.ndarray,
                           foreground_only: bool = False) -> float:
    """
    Multiclass Brier score: mean squared error between predicted probabilities
    and one-hot ground truth, averaged over pixels.
    Range [0, 2]. Lower is better; a perfectly calibrated model scores 0.
    Complements ECE — ECE measures reliability in confidence bins,
    Brier score is a proper scoring rule measuring sharpness + calibration jointly.
    """
    C, H, W  = probs.shape
    flat_gt  = labels.reshape(-1)          # (H*W,)
    flat_prob = probs.reshape(C, -1).T     # (H*W, C)

    if foreground_only:
        m = flat_gt != 0
        flat_gt   = flat_gt[m]
        flat_prob = flat_prob[m]

    if len(flat_gt) == 0:
        return 0.0

    # One-hot encode ground truth
    one_hot = np.zeros_like(flat_prob)
    one_hot[np.arange(len(flat_gt)), flat_gt.astype(int)] = 1.0

    return float(np.mean(np.sum((flat_prob - one_hot) ** 2, axis=1)))


# ── AUROC / AUPR for uncertainty as an error detector ─────────────────────────

def calculate_uncertainty_auroc_aupr(uncertainty_map: np.ndarray,
                                     pred_mask: np.ndarray,
                                     gt_mask: np.ndarray,
                                     foreground_only: bool = False
                                     ) -> tuple[float, float]:
    """
    Treats uncertainty as a binary classifier for pixel-level errors.
    A useful UQ method should assign high uncertainty to misclassified pixels.

    AUROC: probability that a random wrong pixel has higher uncertainty than a
           random correct pixel.  0.5 = random, 1.0 = perfect detector.
    AUPR : area under precision-recall curve (positive = error pixel).
           More informative when errors are rare (which they are here at ~2%).

    Uses a threshold-free trapezoidal approximation — no sklearn dependency.
    """
    unc   = uncertainty_map.reshape(-1)
    error = (pred_mask != gt_mask).astype(np.uint8).reshape(-1)

    if foreground_only:
        fg = gt_mask.reshape(-1) != 0
        unc, error = unc[fg], error[fg]

    n_pos = error.sum()
    n_neg = len(error) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Sort by descending uncertainty
    order  = np.argsort(-unc)
    error_sorted = error[order]

    # AUROC via Mann–Whitney U statistic
    tp_cum  = np.cumsum(error_sorted)
    fp_cum  = np.cumsum(1 - error_sorted)
    tpr     = tp_cum / n_pos
    fpr     = fp_cum / n_neg
    auroc   = float(np.trapz(tpr, fpr))

    # AUPR (positive class = error)
    precision = tp_cum / (tp_cum + fp_cum).clip(min=1)
    recall    = tpr
    aupr      = float(np.trapz(precision, recall))

    return auroc, aupr


# ── Reliability diagram ───────────────────────────────────────────────────────

def save_reliability_diagram(bin_data_all: list, bin_data_fg: list,
                              save_path: Path):
    """
    Plots confidence (x) vs accuracy (y) for all pixels and foreground pixels.
    The diagonal is perfect calibration. Points below = overconfident.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#12121f")
    titles    = ["All Pixels", "Foreground Only"]

    for ax, bdata, title in zip(axes, [bin_data_all, bin_data_fg], titles):
        ax.set_facecolor("#12121f")
        if bdata:
            conf = [b["confidence"] for b in bdata]
            acc  = [b["accuracy"]   for b in bdata]
            frac = np.array([b["fraction"] for b in bdata])
            widths = [b["bin_upper"] - b["bin_lower"] for b in bdata]

            # Bar height = accuracy, width = bin width, alpha encodes population
            bars = ax.bar(conf, acc, width=widths, alpha=0.75,
                          color=plt.cm.viridis(frac / frac.max()),
                          edgecolor="white", linewidth=0.5, align="center")

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "--", color="cyan", linewidth=1.5,
                label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", color="white")
        ax.set_ylabel("Accuracy",   color="white")
        ax.set_title(title, color="white", fontsize=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.legend(facecolor="#12121f", labelcolor="white")

    fig.suptitle("Reliability Diagram — MC Dropout", color="white", fontsize=15)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved reliability diagram -> {save_path}")


# ── Visualisation helpers (unchanged) ─────────────────────────────────────────

def save_full_figure(image_rgb, gt_mask, pred_mask,
                     confidence_map, entropy_map, variance_map,
                     save_path: Path, stem: str, T: int, dropout_p: float):
    gt_vis   = colorize(gt_mask)
    pred_vis = colorize(pred_mask)
    fig = plt.figure(figsize=(21, 11), facecolor="#12121f")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15)
    panels = [
        (image_rgb,      "Input Image", None),
        (gt_vis,         "Ground Truth", None),
        (pred_vis,       f"Prediction (mean of T={T} passes)", None),
        (confidence_map, "Confidence Map\n(max softmax probability)", "viridis"),
        (entropy_map,    "Entropy Uncertainty\n(total uncertainty)", "inferno"),
        (variance_map,   "Variance Uncertainty\n(epistemic, across passes)", "inferno"),
    ]
    for i, (img, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_facecolor("#12121f")
        if cmap:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")
    fig.suptitle(f"MC Dropout Uncertainty — {stem}   (T={T}, dropout p={dropout_p})",
                 color="white", fontsize=15, y=0.99)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_overlay_figure(image_rgb, entropy_map, variance_map, save_path: Path, stem: str):
    entropy_overlay  = overlay_heatmap_on_image(image_rgb, entropy_map)
    variance_overlay = overlay_heatmap_on_image(image_rgb, variance_map)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#12121f")
    for ax, img, title in zip(axes,
                               [image_rgb, entropy_overlay, variance_overlay],
                               ["Input Image", "Entropy Overlay\n(bright = uncertain)",
                                "Variance Overlay\n(bright = uncertain)"]):
        ax.set_facecolor("#12121f"); ax.imshow(img)
        ax.set_title(title, color="white", fontsize=12); ax.axis("off")
    fig.suptitle(f"Uncertainty Overlay — {stem}", color="white", fontsize=13)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_comparison_grid(records: list, save_path: Path, n_show: int = 6):
    n_show = min(n_show, len(records))
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show), facecolor="#12121f")
    if n_show == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Input", "Ground Truth", "Prediction", "Entropy (Uncertainty)"]
    for row, rec in enumerate(records[:n_show]):
        imgs = [rec["image_rgb"], colorize(rec["gt_mask"]),
                colorize(rec["pred_mask"]), rec["entropy_map"]]
        for col, img in enumerate(imgs):
            ax = axes[row, col]; ax.set_facecolor("#12121f")
            if col == 3:
                im = ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
                if row == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(img)
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=13)
            ax.axis("off")
            if col == 0:
                ax.text(-0.08, 0.5, rec["stem"], transform=ax.transAxes,
                        color="white", fontsize=9, rotation=90, va="center", ha="center")
    fig.suptitle("MC Dropout — Uncertainty Across Multiple Images",
                 color="white", fontsize=16, y=1.0)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved comparison grid -> {save_path}")


def save_class_uncertainty_ranking(class_uncertainty: dict, save_path: Path):
    ranked  = sorted(class_uncertainty.items(), key=lambda x: -x[1])
    names   = [r[0] for r in ranked]
    vals    = [r[1] for r in ranked]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#12121f")
    ax.set_facecolor("#12121f")
    colours = plt.cm.inferno(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names, vals, color=colours, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Mean Predictive Entropy (Uncertainty)", color="white")
    ax.set_title("Class Uncertainty Ranking — MC Dropout", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.invert_yaxis()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved class uncertainty ranking -> {save_path}")


def save_uncertainty_vs_error_scatter(all_entropy: np.ndarray, all_error: np.ndarray,
                                      save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#12121f")
    ax.set_facecolor("#12121f")
    ax.scatter(all_entropy, all_error, c=all_entropy, cmap="inferno",
               s=60, edgecolor="white", linewidth=0.5, alpha=0.85)
    if len(all_entropy) > 2:
        z = np.polyfit(all_entropy, all_error, 1)
        x_line = np.linspace(all_entropy.min(), all_entropy.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="cyan", linewidth=2,
                label=f"trend (slope={z[0]:.3f})")
        corr = np.corrcoef(all_entropy, all_error)[0, 1]
        ax.text(0.05, 0.95, f"Correlation r = {corr:.3f}", transform=ax.transAxes,
                color="white", fontsize=11, va="top",
                bbox=dict(facecolor="#12121f", edgecolor="white"))
    ax.set_xlabel("Mean Uncertainty (Entropy)", color="white")
    ax.set_ylabel("Pixel Error Rate", color="white")
    ax.set_title("Uncertainty vs Error — Is Uncertainty Meaningful?",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.legend(facecolor="#12121f", labelcolor="white")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved uncertainty-vs-error scatter -> {save_path}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run(n_images: int = 20, T: int = MC_DROPOUT_T, dropout_p: float = MC_DROPOUT_P):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MC Dropout: T={T} passes, dropout p={dropout_p}")

    model = build_model(pretrained=PRETRAINED, dropout_p=dropout_p, device=device.type)
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    eval_json = DATA_DIR / "annotations_eval.json"
    dataset   = PepperSegDataset(eval_json, "eval", get_val_transforms())
    print(f"Eval set: {len(dataset)} images, processing {min(n_images, len(dataset))}")

    out_dir     = OUTPUT_DIR / "uncertainty_maps" / "mc_dropout"
    overlay_dir = out_dir / "overlays"
    figure_dir  = out_dir / "full_figures"
    for d in [out_dir, overlay_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    indices = list(range(min(n_images, len(dataset))))

    # ── Accumulators ──────────────────────────────────────────────────────────
    conf_mc_total  = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_det_total = np.zeros((NUM_CLASSES, NUM_CLASSES))

    per_image_summary   = []
    class_entropy_sum   = {c: 0.0 for c in CLASS_NAMES}
    class_entropy_count = {c: 0   for c in CLASS_NAMES}

    grid_records     = []
    all_mean_entropy = []
    all_mean_mi      = []
    all_error_rate   = []

    all_mc_probs = []
    all_gt_masks = []

    # For AUROC/AUPR aggregation across images
    all_unc_flat   = []
    all_error_flat = []
    all_unc_fg     = []
    all_error_fg   = []

    # For global Brier and ECE accumulation
    per_img_brier    = []
    per_img_brier_fg = []
    per_img_ece      = []
    per_img_ece_fg   = []

    # For reliability diagram — accumulate bin data across images
    global_bin_data_all = []
    global_bin_data_fg  = []

    for idx in tqdm(indices, desc="MC Dropout inference"):
        sample   = dataset[idx]
        image_t  = sample["image"]
        gt_mask  = sample["mask"].numpy().astype(np.uint8)
        img_path = sample["img_path"]
        stem     = Path(img_path).stem

        image_rgb = denormalize(image_t)

        # MC Dropout forward — now also returns mi_map
        (mean_proba, pred_mc, entropy_map, variance_map,
         confidence_map, mi_map, _) = mc_dropout_forward(model, image_t, device, T=T)

        all_mc_probs.append(mean_proba)
        all_gt_masks.append(gt_mask)

        _, pred_det = deterministic_forward(model, image_t, device)

        conf_mc_total  += confusion_matrix(pred_mc,  gt_mask, NUM_CLASSES)
        conf_det_total += confusion_matrix(pred_det, gt_mask, NUM_CLASSES)

        error_rate = (pred_mc != gt_mask).mean()
        all_mean_entropy.append(float(entropy_map.mean()))
        all_mean_mi.append(float(mi_map.mean()))
        all_error_rate.append(float(error_rate))

        # Flatten for global AUROC/AUPR accumulation
        all_unc_flat.append(entropy_map.reshape(-1))
        all_error_flat.append((pred_mc != gt_mask).reshape(-1).astype(np.uint8))
        fg_px = gt_mask.reshape(-1) != 0
        all_unc_fg.append(entropy_map.reshape(-1)[fg_px])
        all_error_fg.append((pred_mc != gt_mask).reshape(-1).astype(np.uint8)[fg_px])

        # ECE (per image, then averaged)
        ece, bd_all = calculate_ece(mean_proba, gt_mask, n_bins=15, foreground_only=False)
        ece_fg, bd_fg = calculate_ece(mean_proba, gt_mask, n_bins=15, foreground_only=True)
        per_img_ece.append(ece)
        per_img_ece_fg.append(ece_fg)
        global_bin_data_all.extend(bd_all)
        global_bin_data_fg.extend(bd_fg)

        # Brier score
        brier    = calculate_brier_score(mean_proba, gt_mask, foreground_only=False)
        brier_fg = calculate_brier_score(mean_proba, gt_mask, foreground_only=True)
        per_img_brier.append(brier)
        per_img_brier_fg.append(brier_fg)

        # Per-class entropy accumulation
        for c_idx, c_name in enumerate(CLASS_NAMES):
            cls_pixels = (pred_mc == c_idx)
            if cls_pixels.sum() > 0:
                class_entropy_sum[c_name]   += float(entropy_map[cls_pixels].sum())
                class_entropy_count[c_name] += int(cls_pixels.sum())

        if idx < 12:
            save_full_figure(image_rgb, gt_mask, pred_mc,
                             confidence_map, entropy_map, variance_map,
                             figure_dir / f"{stem}_full.png", stem, T, dropout_p)
            save_overlay_figure(image_rgb, entropy_map, variance_map,
                                overlay_dir / f"{stem}_overlay.png", stem)

        if idx < 6:
            grid_records.append({"stem": stem, "image_rgb": image_rgb,
                                  "gt_mask": gt_mask, "pred_mask": pred_mc,
                                  "entropy_map": entropy_map})

        per_image_summary.append({
            "stem":             stem,
            "mean_entropy":     float(entropy_map.mean()),
            "max_entropy":      float(entropy_map.max()),
            "mean_mi":          float(mi_map.mean()),
            "mean_variance":    float(variance_map.mean()),
            "mean_confidence":  float(confidence_map.mean()),
            "error_rate":       float(error_rate),
            "brier":            float(brier),
            "brier_foreground": float(brier_fg),
            "ece":              float(ece),
            "ece_foreground":   float(ece_fg),
        })

    # ── Aggregate segmentation metrics ────────────────────────────────────────
    metrics_mc  = metrics_from_confusion(conf_mc_total)
    metrics_det = metrics_from_confusion(conf_det_total)

    # ── ECE / Brier averages ──────────────────────────────────────────────────
    mean_ece        = float(np.mean(per_img_ece))
    mean_ece_fg     = float(np.mean(per_img_ece_fg))
    mean_brier      = float(np.mean(per_img_brier))
    mean_brier_fg   = float(np.mean(per_img_brier_fg))

    # ── Global AUROC / AUPR (aggregate all pixels across all images) ──────────
    all_unc_flat_np   = np.concatenate(all_unc_flat)
    all_error_flat_np = np.concatenate(all_error_flat)
    all_unc_fg_np     = np.concatenate(all_unc_fg)
    all_error_fg_np   = np.concatenate(all_error_fg)

    # Subsample for speed if very large (>5M pixels)
    MAX_PIX = 5_000_000
    if len(all_unc_flat_np) > MAX_PIX:
        rng  = np.random.default_rng(42)
        idx_ = rng.choice(len(all_unc_flat_np), MAX_PIX, replace=False)
        all_unc_flat_np   = all_unc_flat_np[idx_]
        all_error_flat_np = all_error_flat_np[idx_]
    if len(all_unc_fg_np) > MAX_PIX:
        rng  = np.random.default_rng(42)
        idx_ = rng.choice(len(all_unc_fg_np), MAX_PIX, replace=False)
        all_unc_fg_np   = all_unc_fg_np[idx_]
        all_error_fg_np = all_error_fg_np[idx_]

    auroc_all, aupr_all = calculate_uncertainty_auroc_aupr(
        all_unc_flat_np, all_error_flat_np.reshape(all_unc_flat_np.shape),
        all_error_flat_np.reshape(all_unc_flat_np.shape))
    # cleaner call: pass uncertainty, pred, gt all as flat arrays
    auroc_all = aupr_all = float("nan")  # recompute properly below

    def _auroc_aupr(unc: np.ndarray, error: np.ndarray):
        n_pos = error.sum(); n_neg = len(error) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan"), float("nan")
        order        = np.argsort(-unc)
        err_sorted   = error[order]
        tp_cum       = np.cumsum(err_sorted)
        fp_cum       = np.cumsum(1 - err_sorted)
        tpr          = tp_cum / n_pos
        fpr          = fp_cum / n_neg
        auroc        = float(np.trapz(tpr, fpr))
        precision    = tp_cum / (tp_cum + fp_cum).clip(min=1)
        aupr         = float(np.trapz(precision, tpr))
        return auroc, aupr

    auroc_all, aupr_all = _auroc_aupr(all_unc_flat_np, all_error_flat_np)
    auroc_fg,  aupr_fg  = _auroc_aupr(all_unc_fg_np,  all_error_fg_np)

    # ── Class uncertainty ranking ─────────────────────────────────────────────
    class_mean_entropy = {
        c: (class_entropy_sum[c] / class_entropy_count[c]
            if class_entropy_count[c] > 0 else 0.0)
        for c in CLASS_NAMES
    }

    # ── Plots ──────────────────────────────────────────────────────────────────
    save_comparison_grid(grid_records, out_dir / "comparison_grid.png", n_show=6)
    save_class_uncertainty_ranking(
        {c: v for c, v in class_mean_entropy.items() if c != "background"},
        out_dir / "class_uncertainty_ranking.png")
    save_uncertainty_vs_error_scatter(
        np.array(all_mean_entropy), np.array(all_error_rate),
        out_dir / "uncertainty_vs_error.png")
    save_reliability_diagram(
        global_bin_data_all, global_bin_data_fg,
        out_dir / "reliability_diagram.png")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MC DROPOUT — SEMANTIC SEGMENTATION UNCERTAINTY SUMMARY")
    print("=" * 65)
    print(f"  Images processed:          {len(indices)}")
    print(f"  Stochastic passes (T):     {T}")
    print(f"  Dropout rate:              {dropout_p}")
    print("-" * 65)
    print(f"  {'Metric':<32}{'Deterministic':>14}{'MC Dropout':>14}")
    print(f"  {'Pixel accuracy':<32}{metrics_det['pixel_accuracy']:>14.4f}{metrics_mc['pixel_accuracy']:>14.4f}")
    print(f"  {'mIoU (all classes)':<32}{metrics_det['miou_all']:>14.4f}{metrics_mc['miou_all']:>14.4f}")
    print(f"  {'mIoU (foreground)':<32}{metrics_det['miou_foreground']:>14.4f}{metrics_mc['miou_foreground']:>14.4f}")
    print(f"  {'mDice (all classes)':<32}{metrics_det['mdice_all']:>14.4f}{metrics_mc['mdice_all']:>14.4f}")
    print(f"  {'mDice (foreground)':<32}{metrics_det['mdice_foreground']:>14.4f}{metrics_mc['mdice_foreground']:>14.4f}")
    print("-" * 65)
    print("  Calibration:")
    print(f"    ECE  (all pixels):       {mean_ece:.4f}")
    print(f"    ECE  (foreground):       {mean_ece_fg:.4f}")
    print(f"    Brier (all pixels):      {mean_brier:.4f}")
    print(f"    Brier (foreground):      {mean_brier_fg:.4f}")
    print("-" * 65)
    print("  Uncertainty quality (error detection):")
    print(f"    AUROC (all pixels):      {auroc_all:.4f}")
    print(f"    AUROC (foreground):      {auroc_fg:.4f}")
    print(f"    AUPR  (all pixels):      {aupr_all:.4f}")
    print(f"    AUPR  (foreground):      {aupr_fg:.4f}")
    print("-" * 65)
    print("  Uncertainty magnitude:")
    print(f"    Mean entropy:            {np.mean(all_mean_entropy):.4f}")
    print(f"    Max  entropy:            {np.max(all_mean_entropy):.4f}")
    print(f"    Min  entropy:            {np.min(all_mean_entropy):.4f}")
    print(f"    Mean MI (epistemic):     {np.mean(all_mean_mi):.4f}")
    print("-" * 65)
    print("  Class uncertainty (most → least uncertain):")
    for c, v in sorted(class_mean_entropy.items(), key=lambda x: -x[1]):
        if c == "background":
            continue
        print(f"    {c:<18} {v:.4f}")
    print("=" * 65)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "method": "mc_dropout",
        "T": T,
        "dropout_p": dropout_p,
        "n_images": len(indices),
        "metrics_deterministic": metrics_det,
        "metrics_mc_dropout":    metrics_mc,
        "accuracy_cost": {
            "pixel_accuracy":  metrics_det["pixel_accuracy"]  - metrics_mc["pixel_accuracy"],
            "miou_foreground": metrics_det["miou_foreground"] - metrics_mc["miou_foreground"],
        },
        # Calibration
        "ece":              mean_ece,
        "ece_foreground":   mean_ece_fg,
        "brier":            mean_brier,
        "brier_foreground": mean_brier_fg,
        # Uncertainty quality
        "auroc_all_pixels": auroc_all,
        "auroc_foreground": auroc_fg,
        "aupr_all_pixels":  aupr_all,
        "aupr_foreground":  aupr_fg,
        # Uncertainty magnitude
        "mean_entropy": float(np.mean(all_mean_entropy)),
        "max_entropy":  float(np.max(all_mean_entropy)),
        "min_entropy":  float(np.min(all_mean_entropy)),
        "mean_mi":      float(np.mean(all_mean_mi)),
        "class_uncertainty_ranking": {
            c: v for c, v in sorted(class_mean_entropy.items(), key=lambda x: -x[1])
        },
        "per_image": per_image_summary,
    }

    out_json = out_dir / "mc_dropout_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:          {out_json}")
    print(f"Full figures:     {figure_dir}")
    print(f"Overlays:         {overlay_dir}")
    print(f"Reliability diag: {out_dir / 'reliability_diagram.png'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--T",        type=int, default=MC_DROPOUT_T)
    parser.add_argument("--dropout",  type=float, default=MC_DROPOUT_P)
    args = parser.parse_args()
    run(n_images=args.n_images, T=args.T, dropout_p=args.dropout)