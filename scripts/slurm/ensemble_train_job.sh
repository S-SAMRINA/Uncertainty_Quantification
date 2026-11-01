#!/bin/bash
#SBATCH --job-name=pepper_ensemble
#SBATCH --partition=A40short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --array=1-5
#SBATCH --output=ensemble_%A_%a.out
#SBATCH --error=ensemble_%A_%a.err

# Trains the 5 ensemble members in parallel (seeds 42-46).
# Submit from the repo root:  sbatch scripts/slurm/ensemble_train_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

MEMBER=$SLURM_ARRAY_TASK_ID
SEED=$((MEMBER + 41))
python src/ensemble_train.py --seed $SEED --member $MEMBER
