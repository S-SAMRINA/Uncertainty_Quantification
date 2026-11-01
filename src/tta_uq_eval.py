import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    EVAL_JSON,
    NUM_CLASSES,
)
from dataset import PepperSegDataset, get_val_transforms
from model import build_model
from evaluate import (
    load_checkpoint,
    confusion_matrix_from_preds,
    metrics_from_confusion,
)
from tta import tta_predict, TTA_TRANSFORMS
from mc_dropout import calculate_ece, calculate_brier_score


def auroc_aupr(scores, errors):
    scores = np.asarray(scores, dtype=np.float32)
    errors = np.asarray(errors, dtype=np.uint8)

    positives = errors.sum()
    negatives = len(errors) - positives

    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    order = np.argsort(-scores)
    scores = scores[order]
    errors = errors[order]

    # Handle equal uncertainty values together
    change = np.r_[True, scores[1:] != scores[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(scores)]

    tp = []
    fp = []

    tp_count = 0
    fp_count = 0

    for start, end in zip(starts, ends):
        block = errors[start:end]
        tp_count += int(block.sum())
        fp_count += int(len(block) - block.sum())

        tp.append(tp_count)
        fp.append(fp_count)

    tpr = np.r_[0.0, np.asarray(tp) / positives]
    fpr = np.r_[0.0, np.asarray(fp) / negatives]

    auroc = float(np.trapz(tpr, fpr))

    precision = np.asarray(tp) / np.maximum(
        np.asarray(tp) + np.asarray(fp), 1
    )
    recall = np.asarray(tp) / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    aupr = float(np.trapz(precision, recall))

    return auroc, aupr


def subsample(scores, errors, max_pixels=5_000_000):
    scores = np.concatenate(scores)
    errors = np.concatenate(errors)

    if len(scores) > max_pixels:
        rng = np.random.default_rng(42)
        idx = rng.choice(
            len(scores),
            max_pixels,
            replace=False
        )
        scores = scores[idx]
        errors = errors[idx]

    return scores, errors


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("TTA views:", len(TTA_TRANSFORMS))

    dataset = PepperSegDataset(
        EVAL_JSON,
        "eval",
        get_val_transforms()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        pretrained=False,
        dropout_p=0.0,
        device=device.type
    )

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    ckpt = load_checkpoint(
        model,
        checkpoint_path,
        device
    )

    model.eval()

    total_conf = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        device=device
    )

    entropy_scores = []
    disagreement_scores = []
    errors_all = []

    entropy_fg = []
    disagreement_fg = []
    errors_fg = []

    ece_values = []
    ece_fg_values = []

    brier_values = []
    brier_fg_values = []

    for batch in tqdm(loader, desc="TTA UQ evaluation"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        pred, mean_probs, entropy, variance, disagreement = \
            tta_predict(model, image)

        total_conf += confusion_matrix_from_preds(
            pred,
            mask,
            NUM_CLASSES
        )

        pred_np = pred.squeeze(0).cpu().numpy()
        gt_np = mask.squeeze(0).cpu().numpy()
        probs_np = mean_probs.squeeze(0).cpu().numpy()

        entropy_np = entropy.squeeze(0).cpu().numpy()
        disagreement_np = disagreement.squeeze(0).cpu().numpy()

        error_np = (pred_np != gt_np).astype(np.uint8)

        entropy_scores.append(entropy_np.reshape(-1))
        disagreement_scores.append(disagreement_np.reshape(-1))
        errors_all.append(error_np.reshape(-1))

        fg = gt_np.reshape(-1) != 0

        entropy_fg.append(entropy_np.reshape(-1)[fg])
        disagreement_fg.append(
            disagreement_np.reshape(-1)[fg]
        )
        errors_fg.append(error_np.reshape(-1)[fg])

        ece, _ = calculate_ece(
            probs_np,
            gt_np,
            n_bins=15,
            foreground_only=False
        )

        ece_fg_value, _ = calculate_ece(
            probs_np,
            gt_np,
            n_bins=15,
            foreground_only=True
        )

        brier = calculate_brier_score(
            probs_np,
            gt_np,
            foreground_only=False
        )

        brier_fg = calculate_brier_score(
            probs_np,
            gt_np,
            foreground_only=True
        )

        ece_values.append(ece)
        ece_fg_values.append(ece_fg_value)
        brier_values.append(brier)
        brier_fg_values.append(brier_fg)

    metrics = metrics_from_confusion(total_conf)

    entropy_all, errors_all_flat = subsample(
        entropy_scores,
        errors_all
    )

    disagreement_all, _ = subsample(
        disagreement_scores,
        errors_all
    )

    entropy_fg_flat, errors_fg_flat = subsample(
        entropy_fg,
        errors_fg
    )

    disagreement_fg_flat, _ = subsample(
        disagreement_fg,
        errors_fg
    )

    entropy_auroc, entropy_aupr = auroc_aupr(
        entropy_all,
        errors_all_flat
    )

    disagreement_auroc, disagreement_aupr = auroc_aupr(
        disagreement_all,
        errors_all_flat
    )

    entropy_auroc_fg, entropy_aupr_fg = auroc_aupr(
        entropy_fg_flat,
        errors_fg_flat
    )

    disagreement_auroc_fg, disagreement_aupr_fg = auroc_aupr(
        disagreement_fg_flat,
        errors_fg_flat
    )

    results = {
        "method": "TTA",
        "num_images": len(dataset),
        "num_tta_views": len(TTA_TRANSFORMS),

        "segmentation": metrics,

        "calibration": {
            "ece_all": float(np.mean(ece_values)),
            "ece_foreground": float(np.mean(ece_fg_values)),
            "brier_all": float(np.mean(brier_values)),
            "brier_foreground": float(np.mean(brier_fg_values)),
        },

        "entropy_error_detection": {
            "auroc_all": entropy_auroc,
            "aupr_all": entropy_aupr,
            "auroc_foreground": entropy_auroc_fg,
            "aupr_foreground": entropy_aupr_fg,
        },

        "disagreement_error_detection": {
            "auroc_all": disagreement_auroc,
            "aupr_all": disagreement_aupr,
            "auroc_foreground": disagreement_auroc_fg,
            "aupr_foreground": disagreement_aupr_fg,
        },
    }

    output_path = OUTPUT_DIR / "tta" / "tta_uq_metrics.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========================================")
    print("TTA — FINAL UQ EVALUATION")
    print("========================================")

    print("\nSegmentation:")
    print(f"Pixel accuracy:     {metrics['pixel_accuracy']:.4f}")
    print(f"mIoU foreground:   {metrics['miou_foreground']:.4f}")
    print(f"mDice foreground:  {metrics['mdice_foreground']:.4f}")

    print("\nCalibration:")
    print(f"ECE all:           {np.mean(ece_values):.4f}")
    print(f"ECE foreground:    {np.mean(ece_fg_values):.4f}")
    print(f"Brier all:         {np.mean(brier_values):.4f}")
    print(f"Brier foreground:  {np.mean(brier_fg_values):.4f}")

    print("\nPredictive entropy — error detection:")
    print(f"AUROC all:         {entropy_auroc:.4f}")
    print(f"AUPR all:          {entropy_aupr:.4f}")
    print(f"AUROC foreground:  {entropy_auroc_fg:.4f}")
    print(f"AUPR foreground:   {entropy_aupr_fg:.4f}")

    print("\nTTA disagreement — error detection:")
    print(f"AUROC all:         {disagreement_auroc:.4f}")
    print(f"AUPR all:          {disagreement_aupr:.4f}")
    print(f"AUROC foreground:  {disagreement_auroc_fg:.4f}")
    print(f"AUPR foreground:   {disagreement_aupr_fg:.4f}")

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
