#!/usr/bin/env bash
#SBATCH --job-name=ri-test
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --exclude=s-002,s-005
#SBATCH --nodelist=s-004
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from pilot_v2}"
source runtime.sh
# The wrapper resolves the actual storage root. Do not modify runtime.sh.
export PILOT_STORAGE="${1:?storage root}"
export PILOT_RUNS="$PILOT_STORAGE/runs"
export HF_HOME="$PILOT_STORAGE/hf"
export TMPDIR="$PILOT_STORAGE/tmp"
export TRITON_CACHE_DIR="$PILOT_STORAGE/triton-cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
srun python3 -u ri_test_audit_v2/ri_test_audit.py run --source "${2:?source run}" --out "${3:?output run}"
