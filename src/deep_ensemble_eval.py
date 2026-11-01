import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, EVAL_JSON, NUM_CLASSES, CLASS_NAMES
from dataset import PepperSegDataset, get_val_transforms
from model import build_model
from evaluate import (
    load_checkpoint,
    confusion_matrix_from_preds,
    metrics_from_confusion,
    colorize_mask,
)
from mc_dropout import calculate_ece, calculate_brier_score


ENSEMBLE_DIR = OUTPUT_DIR / "deep_ensemble"
NUM_MEMBERS = 5


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


def flatten_and_subsample(score_list, error_list, max_pixels=5_000_000):
    scores = np.concatenate(score_list)
    errors = np.concatenate(error_list)

    if len(scores) > max_pixels:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(scores), max_pixels, replace=False)
        scores = scores[idx]
        errors = errors[idx]

    return scores, errors


@torch.no_grad()
def ensemble_predict(models, image):
    probs = []

    for model in models:
        logits = model(image)
        probs.append(torch.softmax(logits, dim=1))

    # [members, batch, classes, height, width]
    all_probs = torch.stack(probs, dim=0)

    mean_probs = all_probs.mean(dim=0)
    pred = mean_probs.argmax(dim=1)

    # Total predictive uncertainty
    p = mean_probs.clamp(min=1e-7)
    entropy = -(p * p.log()).sum(dim=1)
    entropy = entropy / math.log(NUM_CLASSES)

    # Epistemic uncertainty:
    # H(mean prediction) - average H(individual models)
    member_p = all_probs.clamp(min=1e-7)
    member_entropy = -(member_p * member_p.log()).sum(dim=2)
    mean_member_entropy = member_entropy.mean(dim=0)

    raw_entropy = -(p * p.log()).sum(dim=1)
    mutual_information = raw_entropy - mean_member_entropy
    mutual_information = mutual_information.clamp(min=0.0)
    mutual_information = mutual_information / math.log(NUM_CLASSES)

    # Probability variance between ensemble members
    variance = all_probs.var(dim=0, unbiased=False).mean(dim=1)

    # Hard-class disagreement
    labels = all_probs.argmax(dim=2)

    votes = F.one_hot(
        labels,
        num_classes=NUM_CLASSES
    ).float().sum(dim=0)

    max_votes = votes.max(dim=-1).values
    disagreement = 1.0 - (max_votes / NUM_MEMBERS)

    return (
        pred,
        mean_probs,
        entropy,
        mutual_information,
        variance,
        disagreement,
    )


def save_figure(
    image_tensor,
    gt,
    pred,
    entropy,
    mutual_information,
    disagreement,
    path,
    title,
):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * std + mean, 0, 1)

    error = (gt != pred).astype(np.uint8)

    fig, axes = plt.subplots(1, 7, figsize=(28, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB")

    axes[1].imshow(colorize_mask(gt))
    axes[1].set_title("Ground Truth")

    axes[2].imshow(colorize_mask(pred))
    axes[2].set_title("Ensemble Prediction")

    axes[3].imshow(error, cmap="gray")
    axes[3].set_title("Error")

    axes[4].imshow(entropy, cmap="inferno", vmin=0, vmax=1)
    axes[4].set_title("Predictive Entropy")

    axes[5].imshow(mutual_information, cmap="inferno", vmin=0)
    axes[5].set_title("Mutual Information")

    axes[6].imshow(disagreement, cmap="inferno", vmin=0, vmax=0.8)
    axes[6].set_title("Model Disagreement")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Ensemble members:", NUM_MEMBERS)

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

    # --------------------------------------------------------
    # Load all five independently trained models
    # --------------------------------------------------------
    models = []

    for member in range(1, NUM_MEMBERS + 1):
        checkpoint = (
            ENSEMBLE_DIR
            / f"member_{member}"
            / "best_model.pth"
        )

        print(f"Loading member {member}: {checkpoint}")

        model = build_model(
            pretrained=False,
            dropout_p=0.0,
            device=device.type
        )

        ckpt = load_checkpoint(model, checkpoint, device)
        model.eval()

        print(
            f"  seed={ckpt.get('config', {}).get('seed')} "
            f"epoch={ckpt.get('epoch')}"
        )

        models.append(model)

    output_dir = ENSEMBLE_DIR / "evaluation"
    figure_dir = output_dir / "figures"
    entropy_dir = output_dir / "entropy"
    mi_dir = output_dir / "mutual_information"
    disagreement_dir = output_dir / "disagreement"
    pred_dir = output_dir / "predictions"

    for d in [
        figure_dir,
        entropy_dir,
        mi_dir,
        disagreement_dir,
        pred_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    total_conf = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        device=device
    )

    entropy_scores = []
    mi_scores = []
    disagreement_scores = []
    errors_all = []

    entropy_fg = []
    mi_fg = []
    disagreement_fg = []
    errors_fg = []

    ece_values = []
    ece_fg_values = []
    brier_values = []
    brier_fg_values = []

    entropy_correct_sum = 0.0
    entropy_wrong_sum = 0.0
    mi_correct_sum = 0.0
    mi_wrong_sum = 0.0

    correct_count = 0
    wrong_count = 0

    processed = 0

    print("\nRunning Deep Ensemble inference...")

    for batch in tqdm(loader, desc="Deep Ensemble"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        img_path = batch["img_path"][0]

        (
            pred,
            mean_probs,
            entropy,
            mutual_information,
            variance,
            disagreement,
        ) = ensemble_predict(models, image)

        total_conf += confusion_matrix_from_preds(
            pred,
            mask,
            NUM_CLASSES
        )

        error = pred != mask
        correct = ~error

        entropy_wrong_sum += entropy[error].sum().item()
        entropy_correct_sum += entropy[correct].sum().item()

        mi_wrong_sum += mutual_information[error].sum().item()
        mi_correct_sum += mutual_information[correct].sum().item()

        wrong_count += error.sum().item()
        correct_count += correct.sum().item()

        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        gt_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        probs_np = mean_probs.squeeze(0).cpu().numpy()

        entropy_np = entropy.squeeze(0).cpu().numpy()
        mi_np = mutual_information.squeeze(0).cpu().numpy()
        disagreement_np = disagreement.squeeze(0).cpu().numpy()

        error_np = (pred_np != gt_np).astype(np.uint8)

        entropy_scores.append(entropy_np.reshape(-1))
        mi_scores.append(mi_np.reshape(-1))
        disagreement_scores.append(disagreement_np.reshape(-1))
        errors_all.append(error_np.reshape(-1))

        fg = gt_np.reshape(-1) != 0

        entropy_fg.append(entropy_np.reshape(-1)[fg])
        mi_fg.append(mi_np.reshape(-1)[fg])
        disagreement_fg.append(disagreement_np.reshape(-1)[fg])
        errors_fg.append(error_np.reshape(-1)[fg])

        ece, _ = calculate_ece(
            probs_np,
            gt_np,
            n_bins=15,
            foreground_only=False,
        )

        ece_fg, _ = calculate_ece(
            probs_np,
            gt_np,
            n_bins=15,
            foreground_only=True,
        )

        brier = calculate_brier_score(
            probs_np,
            gt_np,
            foreground_only=False,
        )

        brier_fg = calculate_brier_score(
            probs_np,
            gt_np,
            foreground_only=True,
        )

        ece_values.append(ece)
        ece_fg_values.append(ece_fg)
        brier_values.append(brier)
        brier_fg_values.append(brier_fg)

        stem = Path(img_path).stem

        cv2.imwrite(
            str(pred_dir / f"{stem}.png"),
            pred_np
        )

        np.save(
            entropy_dir / f"{stem}.npy",
            entropy_np.astype(np.float32)
        )

        np.save(
            mi_dir / f"{stem}.npy",
            mi_np.astype(np.float32)
        )

        np.save(
            disagreement_dir / f"{stem}.npy",
            disagreement_np.astype(np.float32)
        )

        if processed < 10:
            save_figure(
                batch["image"].squeeze(0),
                gt_np,
                pred_np,
                entropy_np,
                mi_np,
                disagreement_np,
                figure_dir / f"{stem}.png",
                stem,
            )

        processed += 1

    segmentation = metrics_from_confusion(total_conf)

    entropy_all, errors_all_flat = flatten_and_subsample(
        entropy_scores,
        errors_all
    )

    mi_all, _ = flatten_and_subsample(
        mi_scores,
        errors_all
    )

    disagreement_all, _ = flatten_and_subsample(
        disagreement_scores,
        errors_all
    )

    entropy_fg_flat, errors_fg_flat = flatten_and_subsample(
        entropy_fg,
        errors_fg
    )

    mi_fg_flat, _ = flatten_and_subsample(
        mi_fg,
        errors_fg
    )

    disagreement_fg_flat, _ = flatten_and_subsample(
        disagreement_fg,
        errors_fg
    )

    entropy_auroc, entropy_aupr = auroc_aupr(
        entropy_all,
        errors_all_flat
    )

    mi_auroc, mi_aupr = auroc_aupr(
        mi_all,
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

    mi_auroc_fg, mi_aupr_fg = auroc_aupr(
        mi_fg_flat,
        errors_fg_flat
    )

    disagreement_auroc_fg, disagreement_aupr_fg = auroc_aupr(
        disagreement_fg_flat,
        errors_fg_flat
    )

    mean_entropy_correct = (
        entropy_correct_sum / max(correct_count, 1)
    )

    mean_entropy_wrong = (
        entropy_wrong_sum / max(wrong_count, 1)
    )

    mean_mi_correct = (
        mi_correct_sum / max(correct_count, 1)
    )

    mean_mi_wrong = (
        mi_wrong_sum / max(wrong_count, 1)
    )

    results = {
        "method": "Deep Ensemble",
        "num_members": NUM_MEMBERS,
        "num_images": processed,
        "member_seeds": [42, 43, 44, 45, 46],

        "segmentation": segmentation,

        "calibration": {
            "ece_all": float(np.mean(ece_values)),
            "ece_foreground": float(np.mean(ece_fg_values)),
            "brier_all": float(np.mean(brier_values)),
            "brier_foreground": float(np.mean(brier_fg_values)),
        },

        "uncertainty_means": {
            "entropy_correct": mean_entropy_correct,
            "entropy_wrong": mean_entropy_wrong,
            "mutual_information_correct": mean_mi_correct,
            "mutual_information_wrong": mean_mi_wrong,
        },

        "predictive_entropy_error_detection": {
            "auroc_all": entropy_auroc,
            "aupr_all": entropy_aupr,
            "auroc_foreground": entropy_auroc_fg,
            "aupr_foreground": entropy_aupr_fg,
        },

        "mutual_information_error_detection": {
            "auroc_all": mi_auroc,
            "aupr_all": mi_aupr,
            "auroc_foreground": mi_auroc_fg,
            "aupr_foreground": mi_aupr_fg,
        },

        "disagreement_error_detection": {
            "auroc_all": disagreement_auroc,
            "aupr_all": disagreement_aupr,
            "auroc_foreground": disagreement_auroc_fg,
            "aupr_foreground": disagreement_aupr_fg,
        },
    }

    result_path = output_dir / "deep_ensemble_metrics.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========================================")
    print("DEEP ENSEMBLE — FINAL UQ EVALUATION")
    print("========================================")

    print("\nSegmentation:")
    print(f"Pixel accuracy:      {segmentation['pixel_accuracy']:.4f}")
    print(f"mIoU all:           {segmentation['miou_all']:.4f}")
    print(f"mDice all:          {segmentation['mdice_all']:.4f}")
    print(f"mIoU foreground:    {segmentation['miou_foreground']:.4f}")
    print(f"mDice foreground:   {segmentation['mdice_foreground']:.4f}")

    print("\nCalibration:")
    print(f"ECE all:            {np.mean(ece_values):.4f}")
    print(f"ECE foreground:     {np.mean(ece_fg_values):.4f}")
    print(f"Brier all:          {np.mean(brier_values):.4f}")
    print(f"Brier foreground:   {np.mean(brier_fg_values):.4f}")

    print("\nPredictive entropy:")
    print(f"AUROC all:          {entropy_auroc:.4f}")
    print(f"AUPR all:           {entropy_aupr:.4f}")
    print(f"AUROC foreground:   {entropy_auroc_fg:.4f}")
    print(f"AUPR foreground:    {entropy_aupr_fg:.4f}")

    print("\nMutual information:")
    print(f"AUROC all:          {mi_auroc:.4f}")
    print(f"AUPR all:           {mi_aupr:.4f}")
    print(f"AUROC foreground:   {mi_auroc_fg:.4f}")
    print(f"AUPR foreground:    {mi_aupr_fg:.4f}")

    print("\nModel disagreement:")
    print(f"AUROC all:          {disagreement_auroc:.4f}")
    print(f"AUPR all:           {disagreement_aupr:.4f}")
    print(f"AUROC foreground:   {disagreement_auroc_fg:.4f}")
    print(f"AUPR foreground:    {disagreement_aupr_fg:.4f}")

    print("\nMean uncertainty:")
    print(f"Entropy correct:    {mean_entropy_correct:.6f}")
    print(f"Entropy wrong:      {mean_entropy_wrong:.6f}")
    print(f"MI correct:         {mean_mi_correct:.6f}")
    print(f"MI wrong:           {mean_mi_wrong:.6f}")

    print("\nSaved:", result_path)


if __name__ == "__main__":
    main()
