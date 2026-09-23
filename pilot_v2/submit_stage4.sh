#!/usr/bin/env bash
# Run from the isolated stage4_s41_v1 package; one two-GPU model replica per job.
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(dirname -- "$package")"
extra=(); args=(); name=''; schedule=''; shard=0; shards=1
while (( $# )); do
  case "$1" in
    --name) name="$2"; shift 2;;
    --schedule) schedule="$2"; shift 2;;
    --shard) shard="$2"; shift 2;;
    --shards) shards="$2"; shift 2;;
    --nodelist|--exclude|--time|--mem) extra+=("$1=$2"); shift 2;;
    --gate-only|--resume) args+=("$1"); shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Pass a simple --name (no path)' >&2; exit 2; }
[[ -n "$schedule" ]] || { echo '--schedule is required' >&2; exit 2; }
cd -- "$parent"
source runtime.sh
test -d "$PILOT_STORAGE/model/7df9a82518afdecae4e8c026b27adccc8c1f0032"
python3 "$package/stage4_run.py" check --inputs "$package/inputs" --schedule "$schedule"
mkdir -p logs
export STAGE4_PACKAGE_DIR="$package"
sbatch "${extra[@]}" "$package/stage4.sbatch" --inputs "$package/inputs" --schedule "$schedule" \
  --out "$PILOT_RUNS/$name" --shard "$shard" --shards "$shards" "${args[@]}"
