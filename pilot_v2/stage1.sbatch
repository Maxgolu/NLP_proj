#!/usr/bin/env bash
#SBATCH --job-name=stage1-scan
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --exclude=s-002,s-005
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from pilot_v2}"
source runtime.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
hostname
srun python3 -u stage1_scan.py "$@"
