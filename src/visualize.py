"""
visualize.py

Plots training curves (loss, mIoU, accuracy, learning rate) and the final
evaluation metrics produced by train.py / evaluate.py.

Run from the repo root:
    python src/visualize.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR

METRICS_JSON = OUTPUT_DIR / "metrics" / "eval_metrics.json"
TRAINING_JSON = OUTPUT_DIR / "training_history.json"

SAVE_DIR = str(OUTPUT_DIR / "final_figures")
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

with open(METRICS_JSON) as f:
    metrics = json.load(f)

with open(TRAINING_JSON) as f:
    history = json.load(f)

# ------------------------------------------------------------
# Figure style
# ------------------------------------------------------------

plt.style.use("ggplot")
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (8,5),
    "axes.titlesize": 15,
    "axes.labelsize": 13
})

# ============================================================
# 1. Overall Metrics
# ============================================================

metric_names = [
    "Pixel Accuracy",
    "mIoU",
    "mDice",
    "FG mIoU",
    "FG Dice"
]

metric_values = [
    metrics["pixel_accuracy"],
    metrics["miou_all"],
    metrics["mdice_all"],
    metrics["miou_foreground"],
    metrics["mdice_foreground"]
]

plt.figure()

bars = plt.bar(metric_names, metric_values)

plt.ylim(0,1.05)
plt.ylabel("Score")
plt.title("Overall Evaluation Metrics")

for b,v in zip(bars,metric_values):
    plt.text(
        b.get_x()+b.get_width()/2,
        v+0.015,
        f"{v:.3f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"overall_metrics.png"),dpi=300)
plt.close()

# ============================================================
# 2. Per-class IoU and Dice
# ============================================================

classes = metrics["class_names"]

iou = [metrics["iou_per_class"][c] for c in classes]
dice = [metrics["dice_per_class"][c] for c in classes]

x = np.arange(len(classes))
w = 0.35

plt.figure(figsize=(10,5))

plt.bar(x-w/2,iou,width=w,label="IoU")
plt.bar(x+w/2,dice,width=w,label="Dice")

plt.xticks(x,classes,rotation=20)
plt.ylim(0,1)

plt.ylabel("Score")
plt.title("Per-Class Performance")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,"per_class_metrics.png"),dpi=300)
plt.close()

# ============================================================
# Training history helper
# ============================================================

epochs = np.arange(1,len(history["train_loss"])+1)

def plot_curve(train,val,title,ylabel,filename):

    plt.figure()

    plt.plot(
        epochs,
        train,
        label="Train",
        linewidth=2
    )

    plt.plot(
        epochs,
        val,
        label="Validation",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR,filename),
        dpi=300
    )

    plt.close()


# ============================================================
# 3. Loss
# ============================================================

plot_curve(
    history["train_loss"],
    history["val_loss"],
    "Training Loss",
    "Loss",
    "loss_curve.png"
)

# ============================================================
# 4. Accuracy
# ============================================================

plot_curve(
    history["train_acc"],
    history["val_acc"],
    "Pixel Accuracy",
    "Accuracy",
    "accuracy_curve.png"
)

# ============================================================
# 5. Foreground mIoU
# ============================================================

plot_curve(
    history["train_miou_foreground"],
    history["val_miou_foreground"],
    "Foreground mIoU",
    "mIoU",
    "miou_curve.png"
)

# ============================================================
# 6. Learning Rate
# ============================================================

plt.figure()

plt.plot(
    epochs,
    history["lr"],
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

plt.title("Learning Rate Schedule")

plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR,"learning_rate.png"),
    dpi=300
)

plt.close()

# ============================================================
# 7. Summary Table
# ============================================================

fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')

table_data = [
    ["Pixel Accuracy",f"{metrics['pixel_accuracy']:.4f}"],
    ["mIoU",f"{metrics['miou_all']:.4f}"],
    ["mDice",f"{metrics['mdice_all']:.4f}"],
    ["Foreground mIoU",f"{metrics['miou_foreground']:.4f}"],
    ["Foreground Dice",f"{metrics['mdice_foreground']:.4f}"],
]

table = ax.table(
    cellText=table_data,
    colLabels=["Metric","Value"],
    loc="center"
)

table.scale(1.3,2.0)

plt.savefig(
    os.path.join(SAVE_DIR,"summary_table.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 8. Text Summary
# ============================================================

best_epoch = history.get("best_epoch", "Unknown")
best_loss = history.get("best_val_loss", "Unknown")

summary = f"""
=========================
Evaluation Summary
=========================

Samples:
{metrics["num_samples"]}

Pixel Accuracy:
{metrics["pixel_accuracy"]:.4f}

Mean IoU:
{metrics["miou_all"]:.4f}

Foreground mIoU:
{metrics["miou_foreground"]:.4f}

Mean Dice:
{metrics["mdice_all"]:.4f}

Foreground Dice:
{metrics["mdice_foreground"]:.4f}

Best Epoch:
{best_epoch}

Best Validation Loss:
{best_loss}

Checkpoint:
{metrics["checkpoint"]}

Per-Class IoU

"""

for c in classes:
    summary += f"{c:15s}: {metrics['iou_per_class'][c]:.4f}\n"

summary += "\nPer-Class Dice\n\n"

for c in classes:
    summary += f"{c:15s}: {metrics['dice_per_class'][c]:.4f}\n"

with open(os.path.join(SAVE_DIR,"summary.txt"),"w") as f:
    f.write(summary)

print("\nAll figures saved to:")
print(SAVE_DIR)