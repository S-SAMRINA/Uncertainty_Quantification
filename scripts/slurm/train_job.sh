#!/bin/bash
#SBATCH --job-name=pepper_train
#SBATCH --partition=A40devel
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Submit from the repo root:  sbatch scripts/slurm/train_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/train.py
