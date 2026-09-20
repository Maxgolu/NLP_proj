#!/usr/bin/env bash
# Run from pilot_v2. Optional: storage root, source run name, new output name.
set -euo pipefail
test -f runtime.sh || { echo 'Run from the cluster pilot_v2 directory.' >&2; exit 1; }
source_name="${2:-stage1_v4_calibrated}"
output_name="${3:-ri_test_v2}"
[[ "$source_name" =~ ^[A-Za-z0-9_-]+$ && "$output_name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$source_name" != "$output_name" ]] || { echo 'Source/output must differ.' >&2; exit 2; }
if [[ -n "${1:-}" ]]; then
    storage_root="$(realpath -e -- "$1")"
else
    choices=()
    for candidate in ../storage storage; do
        if [[ -f "$candidate/runs/$source_name/config.json" ]]; then
            resolved="$(realpath -e -- "$candidate")"
            if [[ " ${choices[*]:-} " != *" $resolved "* ]]; then choices+=("$resolved"); fi
        fi
    done
    [[ ${#choices[@]} -eq 1 ]] || {
        echo 'Cannot uniquely locate the saved Stage-1 run. Supply the storage root as argument 1.' >&2
        exit 2
    }
    storage_root="${choices[0]}"
fi
source_run="$storage_root/runs/$source_name"
out_run="$storage_root/runs/$output_name"
for input in config.json provenance.json head_stats.csv audit_prompts.jsonl.gz ri_events.jsonl.gz; do
    test -f "$source_run/$input" || { echo "Missing $source_run/$input" >&2; exit 2; }
done
test -d "$storage_root/model/7df9a82518afdecae4e8c026b27adccc8c1f0032" || {
    echo "Pinned local model not found under $storage_root/model" >&2; exit 2;
}
test ! -f "$out_run/.running.lock" || { echo "Run locked: $out_run" >&2; exit 2; }
mkdir -p logs
printf 'Source: %s\nOutput: %s\nResources: s-004, 2 GPUs, 32 GB host RAM, limit 4 hours\n' "$source_run" "$out_run"
sbatch ri_test_audit_v2/ri_test_audit.sbatch "$storage_root" "$source_run" "$out_run"
