#!/usr/bin/env bash
# Submit the Section-1.5 readout from the isolated stage3_readout_v1 package (2 GPUs, one replica).
# Usage: bash stage3_readout_v1/submit_stage3_readout.sh --name NAME [--gate-only] [--resume] [--nodelist s-004] [--exclude s-002,s-005] [--time 02:00:00]
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(dirname -- "$package")"
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
test -f runtime.sh || { echo 'Extract stage3_readout_v1 inside the existing pilot_v2 folder.' >&2; exit 1; }
source runtime.sh
python3 - <<PY
import json,sys
p=json.load(open('$package/inputs/readout_plan.json'))
print('plan items:',len(p['items']),'forwards:',p['forwards'])
PY
test -d "$PILOT_STORAGE/model/7df9a82518afdecae4e8c026b27adccc8c1f0032" || { echo 'Pinned model directory missing.' >&2; exit 1; }
mkdir -p logs
sbatch "${extra[@]}" "$package/stage3_readout.sbatch" --plan "$package/inputs/readout_plan.json" "${args[@]}"
