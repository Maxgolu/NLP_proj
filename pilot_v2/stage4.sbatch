#!/usr/bin/env bash
#SBATCH --job-name=stage4paths
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --exclude=s-002
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from pilot_v2}"
source runtime.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
# The final Slurm signal is relayed by srun; incomplete runs have no done.json.
srun python3 -u "${STAGE4_PACKAGE_DIR:-$PWD/stage4_s41_v1}/stage4_run.py" run "$@"
date
