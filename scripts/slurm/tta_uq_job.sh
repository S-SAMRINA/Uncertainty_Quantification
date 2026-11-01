#!/bin/bash
#SBATCH --job-name=pepper_tta_uq
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=tta_uq_%j.out
#SBATCH --error=tta_uq_%j.err

# Submit from the repo root:  sbatch scripts/slurm/tta_uq_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/tta_uq_eval.py
