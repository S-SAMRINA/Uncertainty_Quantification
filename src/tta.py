import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    EVAL_JSON,
    NUM_CLASSES,
    CLASS_NAMES,
)
from dataset import PepperSegDataset, get_val_transforms
from model import build_model
from evaluate import (
    load_checkpoint,
    confusion_matrix_from_preds,
    metrics_from_confusion,
    colorize_mask,
)


# ------------------------------------------------------------
# 8 TTA views = original + 7 geometric augmentations
# ------------------------------------------------------------

def identity(x):
    return x

def rot90(x):
    return torch.rot90(x, 1, dims=(-2, -1))

def inv_rot90(x):
    return torch.rot90(x, 3, dims=(-2, -1))

def rot180(x):
    return torch.rot90(x, 2, dims=(-2, -1))

def rot270(x):
    return torch.rot90(x, 3, dims=(-2, -1))

def inv_rot270(x):
    return torch.rot90(x, 1, dims=(-2, -1))

def hflip(x):
    return torch.flip(x, dims=(-1,))

def vflip(x):
    return torch.flip(x, dims=(-2,))

def diagonal_flip(x):
    return x.transpose(-2, -1)

def anti_diagonal_flip(x):
    return torch.flip(x.transpose(-2, -1), dims=(-2, -1))


TTA_TRANSFORMS = [
    ("original", identity, identity),
    ("rot90", rot90, inv_rot90),
    ("rot180", rot180, rot180),
    ("rot270", rot270, inv_rot270),
    ("horizontal_flip", hflip, hflip),
    ("vertical_flip", vflip, vflip),
    ("diagonal_flip", diagonal_flip, diagonal_flip),
    ("anti_diagonal_flip", anti_diagonal_flip, anti_diagonal_flip),
]


@torch.no_grad()
def tta_predict(model, image):
    predictions = []

    for name, augment, inverse in TTA_TRANSFORMS:
        augmented_image = augment(image)

        logits = model(augmented_image)
        probabilities = torch.softmax(logits, dim=1)

        # Bring prediction back to the original image orientation
        probabilities = inverse(probabilities)

        predictions.append(probabilities)

    # Shape:
    # [number_of_TTA_views, batch, classes, height, width]
    all_probs = torch.stack(predictions, dim=0)

    # Average prediction across all TTA views
    mean_probs = all_probs.mean(dim=0)

    # Final semantic segmentation prediction
    pred = mean_probs.argmax(dim=1)

    # --------------------------------------------------------
    # Uncertainty measure 1: Predictive entropy
    # Normalized to approximately [0,1]
    # --------------------------------------------------------
    entropy = -(
        mean_probs * torch.clamp(mean_probs, min=1e-7).log()
    ).sum(dim=1)

    entropy = entropy / math.log(NUM_CLASSES)

    # --------------------------------------------------------
    # Uncertainty measure 2: Variance across TTA probabilities
    # --------------------------------------------------------
    variance = all_probs.var(dim=0, unbiased=False).mean(dim=1)

    # --------------------------------------------------------
    # Uncertainty measure 3: TTA disagreement
    #
    # If all 8 TTA predictions give the same class:
    # disagreement = 0
    #
    # More disagreement = higher uncertainty
    # --------------------------------------------------------
    labels = all_probs.argmax(dim=2)

    votes = F.one_hot(
        labels,
        num_classes=NUM_CLASSES
    ).float().sum(dim=0)

    max_votes = votes.max(dim=-1).values

    disagreement = 1.0 - (
        max_votes / len(TTA_TRANSFORMS)
    )

    return pred, mean_probs, entropy, variance, disagreement


def save_visualization(
    image,
    gt,
    pred,
    entropy,
    disagreement,
    output_path,
    title,
):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * std + mean, 0, 1)

    gt_color = colorize_mask(gt)
    pred_color = colorize_mask(pred)

    error = (gt != pred).astype(np.uint8)

    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB image")

    axes[1].imshow(gt_color)
    axes[1].set_title("Ground truth")

    axes[2].imshow(pred_color)
    axes[2].set_title("TTA prediction")

    axes[3].imshow(error, cmap="gray")
    axes[3].set_title("Error")

    axes[4].imshow(entropy, cmap="inferno", vmin=0, vmax=1)
    axes[4].set_title("Predictive entropy")

    axes[5].imshow(disagreement, cmap="inferno", vmin=0, vmax=0.875)
    axes[5].set_title("TTA disagreement")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only process first N images (useful for testing)"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("TTA views:", len(TTA_TRANSFORMS))
    print(
        "Transforms:",
        [name for name, _, _ in TTA_TRANSFORMS]
    )

    # Dataset
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

    # Model
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

    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", ckpt.get("epoch"))

    # Output directories
    tta_dir = OUTPUT_DIR / "tta"
    figure_dir = tta_dir / "figures"
    entropy_dir = tta_dir / "entropy"
    disagreement_dir = tta_dir / "disagreement"

    figure_dir.mkdir(parents=True, exist_ok=True)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    disagreement_dir.mkdir(parents=True, exist_ok=True)

    total_conf = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        device=device
    )

    entropy_correct_sum = 0.0
    entropy_wrong_sum = 0.0
    correct_count = 0
    wrong_count = 0

    disagreement_correct_sum = 0.0
    disagreement_wrong_sum = 0.0

    processed = 0

    print("\nRunning TTA inference...")

    for batch in tqdm(loader, desc="TTA"):
        if (
            args.max_images is not None
            and processed >= args.max_images
        ):
            break

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        img_path = batch["img_path"][0]

        pred, mean_probs, entropy, variance, disagreement = \
            tta_predict(model, image)

        total_conf += confusion_matrix_from_preds(
            pred,
            mask,
            NUM_CLASSES
        )

        # Correct vs incorrect pixels
        error = pred != mask
        correct = ~error

        entropy_wrong_sum += entropy[error].sum().item()
        entropy_correct_sum += entropy[correct].sum().item()

        disagreement_wrong_sum += disagreement[error].sum().item()
        disagreement_correct_sum += disagreement[correct].sum().item()

        wrong_count += error.sum().item()
        correct_count += correct.sum().item()

        # Convert to numpy
        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        gt_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        entropy_np = entropy.squeeze(0).cpu().numpy()
        disagreement_np = disagreement.squeeze(0).cpu().numpy()

        stem = Path(img_path).stem

        # Save exact uncertainty arrays
        np.save(
            entropy_dir / f"{stem}.npy",
            entropy_np.astype(np.float32)
        )

        np.save(
            disagreement_dir / f"{stem}.npy",
            disagreement_np.astype(np.float32)
        )

        # Save visualization for first 10 images
        if processed < 10:
            save_visualization(
                batch["image"].squeeze(0),
                gt_np,
                pred_np,
                entropy_np,
                disagreement_np,
                figure_dir / f"{stem}.png",
                stem,
            )

        processed += 1

    # Segmentation metrics
    summary = metrics_from_confusion(total_conf)

    mean_entropy_correct = (
        entropy_correct_sum / max(correct_count, 1)
    )

    mean_entropy_wrong = (
        entropy_wrong_sum / max(wrong_count, 1)
    )

    mean_disagreement_correct = (
        disagreement_correct_sum / max(correct_count, 1)
    )

    mean_disagreement_wrong = (
        disagreement_wrong_sum / max(wrong_count, 1)
    )

    results = {
        "method": "TTA",
        "num_tta_views": len(TTA_TRANSFORMS),
        "num_images": processed,
        "pixel_accuracy": summary["pixel_accuracy"],
        "miou_all": summary["miou_all"],
        "mdice_all": summary["mdice_all"],
        "miou_foreground": summary["miou_foreground"],
        "mdice_foreground": summary["mdice_foreground"],
        "iou_per_class": {
            CLASS_NAMES[i]: float(summary["iou_per_class"][i])
            for i in range(NUM_CLASSES)
        },
        "dice_per_class": {
            CLASS_NAMES[i]: float(summary["dice_per_class"][i])
            for i in range(NUM_CLASSES)
        },
        "mean_entropy_correct_pixels": mean_entropy_correct,
        "mean_entropy_wrong_pixels": mean_entropy_wrong,
        "mean_disagreement_correct_pixels": mean_disagreement_correct,
        "mean_disagreement_wrong_pixels": mean_disagreement_wrong,
    }

    results_path = tta_dir / "tta_metrics.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== TTA RESULTS =====")
    print("Images:", processed)
    print(f"Pixel accuracy:       {summary['pixel_accuracy']:.4f}")
    print(f"mIoU all:            {summary['miou_all']:.4f}")
    print(f"mDice all:           {summary['mdice_all']:.4f}")
    print(f"mIoU foreground:     {summary['miou_foreground']:.4f}")
    print(f"mDice foreground:    {summary['mdice_foreground']:.4f}")

    print("\n===== UNCERTAINTY CHECK =====")
    print(
        f"Entropy correct pixels:     "
        f"{mean_entropy_correct:.6f}"
    )
    print(
        f"Entropy wrong pixels:       "
        f"{mean_entropy_wrong:.6f}"
    )
    print(
        f"Disagreement correct pixels:"
        f" {mean_disagreement_correct:.6f}"
    )
    print(
        f"Disagreement wrong pixels:  "
        f"{mean_disagreement_wrong:.6f}"
    )

    print("\nSaved to:", tta_dir)


if __name__ == "__main__":
    main()
