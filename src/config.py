"""
config.py

Central configuration for the project: paths, hyperparameters, class
definitions and UQ settings. Every other script imports from here.

Paths default to folders inside this repository and can be overridden
with environment variables, e.g. on the cluster:

    export BUP20_DATA_DIR=/path/to/bup20_uq/data
    export UQ_OUTPUT_DIR=/scratch/$USER/uq_output
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parent.parent           # repo root
DATA_DIR       = Path(os.environ.get("BUP20_DATA_DIR", PROJECT_DIR / "data"))
MASK_DIR       = Path(os.environ.get("UQ_MASK_DIR",    PROJECT_DIR / "masks"))
OUTPUT_DIR     = Path(os.environ.get("UQ_OUTPUT_DIR",  PROJECT_DIR / "output"))
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR       = OUTPUT_DIR / "predictions"
UQ_DIR         = OUTPUT_DIR / "uncertainty_maps"
PLOT_DIR       = OUTPUT_DIR / "plots"

# ── Dataset splits (COCO-format annotation files) ─────────────────────────────
TRAIN_JSON = DATA_DIR / "annotations_train.json"
VAL_JSON   = DATA_DIR / "annotations_val.json"
EVAL_JSON  = DATA_DIR / "annotations_eval.json"

# ── Classes ───────────────────────────────────────────────────────────────────
# 0 = background (unlabelled pixels), 1-5 = pepper ripeness classes
CLASS_NAMES = ["background", "red", "yellow", "green", "mixed_red", "mixed_yellow"]
NUM_CLASSES = len(CLASS_NAMES)   # 6

# Colour palette for visualisation (RGB)
CLASS_COLOURS = {
    0: (0,   0,   0),    # background — black
    1: (220, 50,  50),   # red
    2: (240, 200, 50),   # yellow
    3: (50,  180, 50),   # green
    4: (180, 100, 50),   # mixed_red
    5: (180, 180, 50),   # mixed_yellow
}

# ── Training hyperparameters ──────────────────────────────────────────────────
IMAGE_SIZE    = 512          # both H and W
BATCH_SIZE    = 4
NUM_EPOCHS    = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
LR_PATIENCE   = 5            # ReduceLROnPlateau patience
EARLY_STOP    = 15           # early stopping patience
NUM_WORKERS   = 4
SEED          = 42

# ── Model ─────────────────────────────────────────────────────────────────────
ENCODER       = "resnet50"   # ImageNet pretrained
PRETRAINED    = True

# ── Loss ──────────────────────────────────────────────────────────────────────
DICE_WEIGHT   = 0.5          # combined loss = CE * (1-w) + Dice * w

# ── Evaluation ────────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5

# ── Uncertainty quantification ────────────────────────────────────────────────
TTA_N_AUGMENTS = 7
MC_DROPOUT_T   = 30          # stochastic forward passes
MC_DROPOUT_P   = 0.3         # dropout probability at test time
