import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import train as t
from config import OUTPUT_DIR as BASE_OUTPUT_DIR


def dirichlet_kl(alpha):
    """
    KL divergence between Dirichlet(alpha) and uniform Dirichlet(1).
    alpha shape: [B, C, H, W]
    """
    num_classes = alpha.shape[1]

    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)

    lnB_alpha = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    )

    lnB_beta = (
        torch.lgamma(beta).sum(dim=1, keepdim=True)
        - torch.lgamma(sum_beta)
    )

    digamma_term = (
        (alpha - beta)
        * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)

    kl = digamma_term - lnB_alpha - lnB_beta

    return kl.mean()


class EvidentialSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        kl_weight: float = 1e-3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.kl_weight = kl_weight

    def forward(self, logits, targets):

        # --------------------------------------------------
        # 1. Convert normal logits into non-negative evidence
        # --------------------------------------------------
        evidence = F.softplus(logits)

        # Dirichlet parameters
        alpha = evidence + 1.0

        # Total Dirichlet strength
        strength = alpha.sum(dim=1, keepdim=True)

        # Expected class probabilities
        probs = alpha / strength

        # --------------------------------------------------
        # 2. Evidential classification loss
        # --------------------------------------------------
        one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        evidential_ce = (
            one_hot
            * (
                torch.digamma(strength)
                - torch.digamma(alpha)
            )
        ).sum(dim=1).mean()

        # --------------------------------------------------
        # 3. Dice loss (foreground only)
        # --------------------------------------------------
        probs_fg = probs[:, 1:, :, :]
        target_fg = one_hot[:, 1:, :, :]

        intersection = (
            probs_fg * target_fg
        ).sum(dim=(0, 2, 3))

        denominator = (
            probs_fg + target_fg
        ).sum(dim=(0, 2, 3))

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        dice_loss = 1.0 - dice.mean()

        # --------------------------------------------------
        # 4. Penalize unsupported evidence
        # --------------------------------------------------
        adjusted_alpha = (
            one_hot
            + (1.0 - one_hot) * alpha
        )

        kl_loss = dirichlet_kl(adjusted_alpha)

        # --------------------------------------------------
        # Final loss
        # --------------------------------------------------
        segmentation_loss = (
            (1.0 - self.dice_weight) * evidential_ce
            + self.dice_weight * dice_loss
        )

        return segmentation_loss + self.kl_weight * kl_loss


# Replace ONLY the loss used inside train.py.
# Existing baseline/TTA/ensemble files remain unchanged.
t.CombinedLoss = EvidentialSegmentationLoss


# Separate EDL output directory
t.OUTPUT_DIR = BASE_OUTPUT_DIR / "evidential"
t.CHECKPOINT_DIR = t.OUTPUT_DIR / "checkpoints"
t.PLOT_DIR = t.OUTPUT_DIR / "plots"
t.PRED_DIR = t.OUTPUT_DIR / "predictions"
t.UQ_DIR = t.OUTPUT_DIR / "uncertainty"

for directory in [
    t.OUTPUT_DIR,
    t.CHECKPOINT_DIR,
    t.PLOT_DIR,
    t.PRED_DIR,
    t.UQ_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("EVIDENTIAL DEEP LEARNING TRAINING")
    print("=" * 60)
    print("Evidence function : Softplus")
    print("Dirichlet alpha   : evidence + 1")
    print("KL weight         : 1e-3")
    print(f"Output directory  : {t.OUTPUT_DIR}")
    print("=" * 60)

    t.main()
