#!/usr/bin/env bash
# Usage: bash submit_stage2_extension_v2.sh 6 [--name stage2_v2_extension] [--resume] [--gate-only] [--nodelist s-004]
# Positional 1: GPU count (2-6). Two GPUs per replica: 2/3 -> 1 replica, 4/5 -> 2, 6 -> 3.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
gpus="${1:-6}"
if [[ ! "$gpus" =~ ^[2-6]$ ]]; then echo 'Choose 2 through 6 GPUs.' >&2; exit 2; fi
if (( $# )); then shift; fi
extra=()
args=()
while (( $# )); do
  case "$1" in
    --nodelist) extra+=("--nodelist=$2"); shift 2;;
    *) args+=("$1"); shift;;
  esac
done
test -f runtime.sh || { echo 'Run from the cluster pilot_v2 directory.' >&2; exit 1; }
test -f extension_v2_heads.json || { echo 'extension_v2_heads.json missing.' >&2; exit 1; }
for f in stage2_extension_v2.py stage2_engine.py stage2_common.py stage1_scan.py stage1_audit.py model_lock_olmo2.json; do
  test -f "$f" || { echo "Missing $f" >&2; exit 1; }
done
source runtime.sh
test -f "$PILOT_RUNS/stage2_v1/summary.json" || { echo "Source run not found: $PILOT_RUNS/stage2_v1" >&2; exit 1; }
python3 - <<'PY'
import json,os
s=json.load(open(os.path.join(os.environ['PILOT_RUNS'],'stage2_v1','summary.json')))
assert s.get('complete'),'stage2_v1 is not complete'
h=json.load(open('extension_v2_heads.json'))
print('Scope P heads:',len(h['scopeP_heads_138_pairs']),' Scope F heads:',len(h['scopeF_heads_178_pairs']))
PY
replicas=$((gpus / 2))
mkdir -p logs
sbatch --gres="gpu:$gpus" --cpus-per-task="$((4 * replicas))" --mem="$((32 * replicas))G" "${extra[@]}" stage2_extension_v2.sbatch --gpus "$gpus" "${args[@]}"
