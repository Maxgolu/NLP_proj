#!/usr/bin/env bash
# Usage: bash submit_stage2.sh 6 --name stage2_v1 [--resume]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gpus="${1:-6}"
if [[ ! "$gpus" =~ ^[2-6]$ ]]; then
  echo 'Choose 2 through 6 GPUs.' >&2
  exit 2
fi
if (( $# )); then shift; fi
replicas=$((gpus / 2))
mkdir -p logs
sbatch --gres="gpu:$gpus" --cpus-per-task="$((4 * replicas))" --mem="$((32 * replicas))G" stage2.sbatch --gpus "$gpus" "$@"
