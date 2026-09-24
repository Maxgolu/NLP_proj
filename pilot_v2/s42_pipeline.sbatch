#!/usr/bin/env bash
#SBATCH --job-name=s42all
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --exclude=s-002
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --signal=TERM@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from pilot_v2}"
package="${S42_PACKAGE_DIR:?Use submit_s42_pipeline.sh}"
source runtime.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
srun --gpu-bind=none python3 -u "$package/s42_pipeline.py" "$@"
date
