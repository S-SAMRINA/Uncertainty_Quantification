import argparse
from pathlib import Path

import train as t
from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--member", type=int, required=True)
    args = parser.parse_args()

    member_dir = (
        OUTPUT_DIR / "deep_ensemble"
        / f"member_{args.member}"
    )

    # Override train.py settings for this ensemble member
    t.SEED = args.seed
    t.OUTPUT_DIR = member_dir
    t.CHECKPOINT_DIR = member_dir
    t.PLOT_DIR = member_dir / "plots"
    t.PRED_DIR = member_dir / "predictions"
    t.UQ_DIR = member_dir / "uncertainty_maps"

    print("====================================")
    print(f"Deep Ensemble Member {args.member}")
    print(f"Seed: {args.seed}")
    print(f"Output: {member_dir}")
    print("====================================")

    t.main()


if __name__ == "__main__":
    main()
