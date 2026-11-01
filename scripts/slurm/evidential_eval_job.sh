#!/bin/bash
#SBATCH --job-name=pepper_evidential_eval
#SBATCH --partition=A40devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=evidential_eval_%j.out
#SBATCH --error=evidential_eval_%j.err

# Submit from the repo root:  sbatch scripts/slurm/evidential_eval_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/evidential_eval.py
