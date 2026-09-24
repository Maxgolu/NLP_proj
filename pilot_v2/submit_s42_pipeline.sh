#!/usr/bin/env bash
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent="$(dirname -- "$package")"
gpus=6; name=''; wall=''; extra=(); args=()
while (( $# )); do
  case "$1" in
    --name|--gpus|--time|--nodelist|--exclude|--mem)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --name) name="$2";;
        --gpus) gpus="$2";;
        --time) wall="$2";;
        *) extra+=("$1=$2");;
      esac
      shift 2;;
    --resume) args+=("$1"); shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Pass a simple --name' >&2; exit 2; }
(( ${#name} <= 80 )) || { echo 'Run name too long' >&2; exit 2; }
[[ "$gpus" =~ ^(2|4|6)$ ]] || { echo '--gpus must be 2, 4 or 6' >&2; exit 2; }
cd -- "$parent"
source runtime.sh
test -d "$PILOT_STORAGE/model/7df9a82518afdecae4e8c026b27adccc8c1f0032"
mkdir -p "$PILOT_RUNS" logs
exec 9>"$PILOT_RUNS/.submit_s42_$name.lock"
flock -n 9 || { echo 'Another submission check is in progress' >&2; exit 2; }
python3 "$package/s42_submit.py" --out "$PILOT_RUNS/$name" --job-name "s42_$name" "${args[@]}"
python3 "$package/s42_review.py" bundle --package "$package"
python3 "$package/s42_run.py" check --inputs "$package/inputs"
if [[ -z "$wall" ]]; then
  partition_info="$(scontrol show partition studentkillable -o)"
  wall="$(python3 - "$partition_info" <<'PY'
import re,sys
m=re.search(r'\bMaxTime=(\S+)',sys.argv[1])
if not m:raise SystemExit('Cannot read partition limit; pass --time')
value=m[1]
if value in ('UNLIMITED','INFINITE'):seconds=48*3600
else:
    if '-' in value:days,clock=value.split('-',1);days=int(days)
    else:days=0;clock=value
    parts=list(map(int,clock.split(':')))
    if len(parts)==3:h,m,s=parts
    elif len(parts)==2:h=0;m,s=parts
    else:h=0;m=parts[0];s=0
    seconds=min(48*3600,days*86400+h*3600+m*60+s)
if seconds<=0:raise SystemExit('Unusable partition limit')
days,seconds=divmod(seconds,86400);h,seconds=divmod(seconds,3600);m,s=divmod(seconds,60)
print(f'{days}-{h:02}:{m:02}:{s:02}')
PY
)"
fi
replicas=$((gpus / 2))
export S42_PACKAGE_DIR="$package"
echo "Whole S4.2 pipeline: $gpus GPUs, $replicas replicas, wall limit $wall"
sbatch --job-name="s42_$name" --gres="gpu:$gpus" --cpus-per-task="$((4*replicas))" --mem="$((48*replicas))G" --time="$wall" \
  "${extra[@]}" "$package/s42_pipeline.sbatch" --inputs "$package/inputs" \
  --out "$PILOT_RUNS/$name" --gpus "$gpus" "${args[@]}"
