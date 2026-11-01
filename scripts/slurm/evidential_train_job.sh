#!/bin/bash
#SBATCH --job-name=pepper_evidential_train
#SBATCH --partition=A40short
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=evidential_train_%j.out
#SBATCH --error=evidential_train_%j.err

# Submit from the repo root:  sbatch scripts/slurm/evidential_train_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/evidential_train.py
