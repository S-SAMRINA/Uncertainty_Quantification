#!/bin/bash
#SBATCH --job-name=pepper_ensemble_eval
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=ensemble_eval_%j.out
#SBATCH --error=ensemble_eval_%j.err

# Submit from the repo root:  sbatch scripts/slurm/ensemble_eval_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/deep_ensemble_eval.py
