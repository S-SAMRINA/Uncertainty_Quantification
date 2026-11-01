from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    DATA_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    NUM_WORKERS,
)

from dataset import PepperSegDataset, get_val_transforms
from model import build_model

from evaluate import (
    resolve_eval_json,
    confusion_matrix_from_preds,
    metrics_from_confusion,
)

# IMPORTANT:
# reuse exactly the same calibration functions as the other UQ methods
from mc_dropout import calculate_ece, calculate_brier_score


CHECKPOINT = (
    BASE_OUTPUT_DIR
    / "evidential"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = (
    BASE_OUTPUT_DIR
    / "evidential"
    / "evaluation"
)

PRED_DIR = OUTPUT_DIR / "predictions"
UNC_DIR = OUTPUT_DIR / "uncertainty"
ENTROPY_DIR = OUTPUT_DIR / "entropy"
FIG_DIR = OUTPUT_DIR / "figures"

for d in [
    OUTPUT_DIR,
    PRED_DIR,
    UNC_DIR,
    ENTROPY_DIR,
    FIG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def safe_auc(scores, errors):
    scores = np.asarray(scores).reshape(-1)
    errors = np.asarray(errors).astype(np.uint8).reshape(-1)

    if len(np.unique(errors)) < 2:
        return float("nan"), float("nan")

    return (
        float(roc_auc_score(errors, scores)),
        float(average_precision_score(errors, scores)),
    )


@torch.no_grad()
def evidential_forward(model, image):
    """
    Single-pass Evidential Deep Learning inference.

    logits
      -> Softplus evidence
      -> alpha = evidence + 1
      -> expected Dirichlet probabilities

    uncertainty = K / sum(alpha)
    """

    logits = model(image)

    # Non-negative evidence
    evidence = F.softplus(logits)

    # Dirichlet parameters
    alpha = evidence + 1.0

    # Total evidence / Dirichlet strength
    strength = alpha.sum(dim=1, keepdim=True)

    # Expected class probabilities
    probs = alpha / strength

    # Prediction
    pred = probs.argmax(dim=1)

    # EDL uncertainty:
    # high when total evidence is low
    uncertainty = NUM_CLASSES / strength

    # Predictive entropy
    entropy = -torch.sum(
        probs * torch.log(probs.clamp_min(1e-8)),
        dim=1,
    ) / np.log(NUM_CLASSES)

    return probs, pred, uncertainty.squeeze(1), entropy


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("EVIDENTIAL DEEP LEARNING EVALUATION")
    print("=" * 65)
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    eval_json = resolve_eval_json()

    eval_ds = PepperSegDataset(
        eval_json,
        "eval",
        get_val_transforms(),
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("Eval samples:", len(eval_ds))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build_model(
        pretrained=False,
        dropout_p=0.0,
        device=device.type,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )

    # --------------------------------------------------
    # Containers
    # --------------------------------------------------

    total_conf = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        device=device,
    )

    ece_all_list = []
    ece_fg_list = []

    brier_all_list = []
    brier_fg_list = []

    uncertainty_all = []
    error_all = []

    uncertainty_fg = []
    error_fg = []

    entropy_all = []
    entropy_fg = []

    mean_unc_correct = []
    mean_unc_wrong = []

    mean_entropy_correct = []
    mean_entropy_wrong = []

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\nRunning EDL inference...")

    for idx, batch in enumerate(
        tqdm(eval_loader, desc="EDL Eval")
    ):

        image = batch["image"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        img_path = batch["img_path"][0]

        probs, pred, uncertainty, entropy = (
            evidential_forward(
                model,
                image,
            )
        )

        total_conf += confusion_matrix_from_preds(
            pred,
            mask,
            NUM_CLASSES,
        )

        # ----------------------------------------------
        # Convert one sample to NumPy
        # ----------------------------------------------

        probs_np = (
            probs[0]
            .detach()
            .cpu()
            .numpy()
        )

        pred_np = (
            pred[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        gt_np = (
            mask[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        uncertainty_np = (
            uncertainty[0]
            .detach()
            .cpu()
            .numpy()
        )

        entropy_np = (
            entropy[0]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------
        # Calibration
        # exactly same helper functions used previously
        # ----------------------------------------------

        ece_all_list.append(
            calculate_ece(
                probs_np,
                gt_np,
            )[0]
        )

        brier_all_list.append(
            calculate_brier_score(
                probs_np,
                gt_np,
            )
        )

        fg_mask = gt_np != 0

        if fg_mask.any():

            # Keep 3D shape expected by the shared ECE/Brier helpers:
            # probabilities: [C, 1, N], labels: [1, N]
            fg_probs = probs_np[:, fg_mask][:, None, :]
            fg_labels = gt_np[fg_mask][None, :]

            ece_fg_list.append(
                calculate_ece(
                    fg_probs,
                    fg_labels,
                )[0]
            )

            brier_fg_list.append(
                calculate_brier_score(
                    fg_probs,
                    fg_labels,
                )
            )

        # ----------------------------------------------
        # Error detection
        # ----------------------------------------------

        error_map = (
            pred_np != gt_np
        ).astype(np.uint8)

        uncertainty_all.append(
            uncertainty_np.reshape(-1)
        )

        entropy_all.append(
            entropy_np.reshape(-1)
        )

        error_all.append(
            error_map.reshape(-1)
        )

        if fg_mask.any():

            uncertainty_fg.append(
                uncertainty_np[fg_mask]
            )

            entropy_fg.append(
                entropy_np[fg_mask]
            )

            error_fg.append(
                error_map[fg_mask]
            )

        # ----------------------------------------------
        # Correct vs incorrect uncertainty
        # ----------------------------------------------

        correct = error_map == 0
        wrong = error_map == 1

        if correct.any():

            mean_unc_correct.append(
                uncertainty_np[correct].mean()
            )

            mean_entropy_correct.append(
                entropy_np[correct].mean()
            )

        if wrong.any():

            mean_unc_wrong.append(
                uncertainty_np[wrong].mean()
            )

            mean_entropy_wrong.append(
                entropy_np[wrong].mean()
            )

        # ----------------------------------------------
        # Save maps
        # ----------------------------------------------

        stem = Path(img_path).stem

        cv2.imwrite(
            str(PRED_DIR / f"{stem}.png"),
            pred_np,
        )

        np.save(
            UNC_DIR / f"{stem}.npy",
            uncertainty_np,
        )

        np.save(
            ENTROPY_DIR / f"{stem}.npy",
            entropy_np,
        )

        # ----------------------------------------------
        # First 10 presentation figures
        # ----------------------------------------------

        if idx < 10:

            image_np = (
                image[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            # ImageNet de-normalisation
            mean = np.array(
                [0.485, 0.456, 0.406]
            )

            std = np.array(
                [0.229, 0.224, 0.225]
            )

            image_np = (
                image_np * std + mean
            )

            image_np = np.clip(
                image_np,
                0,
                1,
            )

            fig, axes = plt.subplots(
                1,
                6,
                figsize=(20, 4),
            )

            axes[0].imshow(image_np)
            axes[0].set_title("RGB")

            axes[1].imshow(
                gt_np,
                vmin=0,
                vmax=NUM_CLASSES - 1,
            )
            axes[1].set_title("Ground Truth")

            axes[2].imshow(
                pred_np,
                vmin=0,
                vmax=NUM_CLASSES - 1,
            )
            axes[2].set_title("EDL Prediction")

            axes[3].imshow(
                error_map,
                cmap="gray",
            )
            axes[3].set_title("Error")

            im4 = axes[4].imshow(
                uncertainty_np,
                cmap="inferno",
                vmin=0,
                vmax=1,
            )
            axes[4].set_title(
                "Evidential Uncertainty"
            )

            fig.colorbar(
                im4,
                ax=axes[4],
                fraction=0.046,
            )

            im5 = axes[5].imshow(
                entropy_np,
                cmap="inferno",
                vmin=0,
                vmax=1,
            )

            axes[5].set_title(
                "Predictive Entropy"
            )

            fig.colorbar(
                im5,
                ax=axes[5],
                fraction=0.046,
            )

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()

            plt.savefig(
                FIG_DIR / f"{stem}.png",
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(fig)

    # --------------------------------------------------
    # Segmentation metrics
    # --------------------------------------------------

    segmentation = metrics_from_confusion(
        total_conf
    )

    # --------------------------------------------------
    # Combine uncertainty arrays
    # --------------------------------------------------

    uncertainty_all = np.concatenate(
        uncertainty_all
    )

    entropy_all = np.concatenate(
        entropy_all
    )

    error_all = np.concatenate(
        error_all
    )

    uncertainty_fg = np.concatenate(
        uncertainty_fg
    )

    entropy_fg = np.concatenate(
        entropy_fg
    )

    error_fg = np.concatenate(
        error_fg
    )

    # --------------------------------------------------
    # AUROC / AUPR
    # --------------------------------------------------

    uq_auc_all, uq_aupr_all = safe_auc(
        uncertainty_all,
        error_all,
    )

    uq_auc_fg, uq_aupr_fg = safe_auc(
        uncertainty_fg,
        error_fg,
    )

    ent_auc_all, ent_aupr_all = safe_auc(
        entropy_all,
        error_all,
    )

    ent_auc_fg, ent_aupr_fg = safe_auc(
        entropy_fg,
        error_fg,
    )

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    metrics = {
        "method": "Evidential Deep Learning",

        "checkpoint_epoch":
            checkpoint.get("epoch"),

        "segmentation":
            segmentation,

        "calibration": {
            "ece_all":
                float(np.mean(ece_all_list)),

            "ece_foreground":
                float(np.mean(ece_fg_list)),

            "brier_all":
                float(np.mean(brier_all_list)),

            "brier_foreground":
                float(np.mean(brier_fg_list)),
        },

        "evidential_uncertainty": {
            "auroc_all":
                uq_auc_all,

            "aupr_all":
                uq_aupr_all,

            "auroc_foreground":
                uq_auc_fg,

            "aupr_foreground":
                uq_aupr_fg,
        },

        "predictive_entropy": {
            "auroc_all":
                ent_auc_all,

            "aupr_all":
                ent_aupr_all,

            "auroc_foreground":
                ent_auc_fg,

            "aupr_foreground":
                ent_aupr_fg,
        },

        "mean_uncertainty": {
            "correct":
                float(np.mean(mean_unc_correct)),

            "wrong":
                float(np.mean(mean_unc_wrong)),

            "entropy_correct":
                float(np.mean(mean_entropy_correct)),

            "entropy_wrong":
                float(np.mean(mean_entropy_wrong)),
        },
    }

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    metrics_file = (
        OUTPUT_DIR
        / "evidential_metrics.json"
    )

    with open(
        metrics_file,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # Print important results
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EDL RESULTS")
    print("=" * 65)

    print("\nSEGMENTATION")
    print(json.dumps(
        segmentation,
        indent=2,
    ))

    print("\nCALIBRATION")

    print(
        f"ECE all: "
        f"{np.mean(ece_all_list):.4f}"
    )

    print(
        f"ECE foreground: "
        f"{np.mean(ece_fg_list):.4f}"
    )

    print(
        f"Brier all: "
        f"{np.mean(brier_all_list):.4f}"
    )

    print(
        f"Brier foreground: "
        f"{np.mean(brier_fg_list):.4f}"
    )

    print("\nEVIDENTIAL UNCERTAINTY")

    print(
        f"AUROC all: {uq_auc_all:.4f}"
    )

    print(
        f"AUPR all: {uq_aupr_all:.4f}"
    )

    print(
        f"AUROC foreground: "
        f"{uq_auc_fg:.4f}"
    )

    print(
        f"AUPR foreground: "
        f"{uq_aupr_fg:.4f}"
    )

    print("\nPREDICTIVE ENTROPY")

    print(
        f"AUROC all: {ent_auc_all:.4f}"
    )

    print(
        f"AUPR all: {ent_aupr_all:.4f}"
    )

    print(
        f"AUROC foreground: "
        f"{ent_auc_fg:.4f}"
    )

    print(
        f"AUPR foreground: "
        f"{ent_aupr_fg:.4f}"
    )

    print("\nMEAN UNCERTAINTY")

    print(
        "correct:",
        float(np.mean(mean_unc_correct)),
    )

    print(
        "wrong:",
        float(np.mean(mean_unc_wrong)),
    )

    print(
        "entropy correct:",
        float(np.mean(mean_entropy_correct)),
    )

    print(
        "entropy wrong:",
        float(np.mean(mean_entropy_wrong)),
    )

    print("\nSaved:")
    print(metrics_file)
    print("Figures:", FIG_DIR)

    print("=" * 65)


if __name__ == "__main__":
    main()
