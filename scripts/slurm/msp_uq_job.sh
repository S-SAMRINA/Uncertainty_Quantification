#!/bin/bash
#SBATCH --job-name=pepper_msp_uq
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=msp_uq_%j.out
#SBATCH --error=msp_uq_%j.err

# Submit from the repo root:  sbatch scripts/slurm/msp_uq_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/msp_uq.py
