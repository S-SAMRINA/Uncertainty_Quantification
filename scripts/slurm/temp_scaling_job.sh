#!/bin/bash
#SBATCH --job-name=pepper_temp_scaling
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=temp_scaling_%j.out
#SBATCH --error=temp_scaling_%j.err

# Submit from the repo root:  sbatch scripts/slurm/temp_scaling_job.sh
module purge
source venv/bin/activate            # adjust to your environment
export MPLBACKEND=Agg
export BUP20_DATA_DIR=${BUP20_DATA_DIR:-$PWD/data}

python src/temperature_scaling.py
