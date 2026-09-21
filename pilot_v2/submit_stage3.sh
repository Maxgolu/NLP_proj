#!/usr/bin/env bash
# Submit from the isolated stage3_v1 package; keep the established parent runtime.
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(dirname -- "$package")"
gpus="${1:-6}"
if [[ ! "$gpus" =~ ^(2|4|6)$ ]]; then echo 'Choose 2, 4 or 6 GPUs.' >&2; exit 2; fi
if (( $# )); then shift; fi
extra=(); args=()
while (( $# )); do
  case "$1" in
    --nodelist|--exclude|--time)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      extra+=("$1=$2"); shift 2;;
    *) args+=("$1"); shift;;
  esac
done
cd -- "$parent"
test -f runtime.sh || { echo 'Extract the stage3_v1 directory inside the existing pilot_v2 folder.' >&2; exit 1; }
source runtime.sh
python3 "$package/stage3_run.py" --check-inputs --inputs "$package/inputs"
test -d "$PILOT_STORAGE/model/7df9a82518afdecae4e8c026b27adccc8c1f0032" || { echo 'Pinned model directory missing.' >&2; exit 1; }
replicas=$((gpus / 2))
mkdir -p logs
sbatch --gres="gpu:$gpus" --cpus-per-task="$((4*replicas))" --mem="$((32*replicas))G" "${extra[@]}" \
  "$package/stage3.sbatch" --gpus "$gpus" --inputs "$package/inputs" "${args[@]}"
