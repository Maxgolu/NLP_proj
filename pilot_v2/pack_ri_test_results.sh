#!/usr/bin/env bash
# Run from pilot_v2 after Slurm COMPLETED / ExitCode 0:0.
set -euo pipefail
name="${1:-ri_test_v2}"
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
choices=()
for candidate in ../storage storage; do
    if [[ -f "$candidate/runs/$name/summary.json" ]]; then
        resolved="$(realpath -e -- "$candidate/runs/$name")"
        if [[ " ${choices[*]:-} " != *" $resolved "* ]]; then choices+=("$resolved"); fi
    fi
done
[[ ${#choices[@]} -eq 1 ]] || { echo 'Cannot uniquely locate completed results.' >&2; exit 2; }
run="${choices[0]}"
test ! -f "$run/.running.lock" || { echo 'Results are still locked.' >&2; exit 2; }
python3 ri_test_audit_v2/verify_ri_test_results.py "$run"
archive="${name}_results.tar.gz"
test ! -e "$archive" || { echo "Archive exists: $archive. Move it aside before repacking." >&2; exit 2; }
tar -czf "$archive" -C "$(dirname -- "$run")" "$name"
printf 'Ready: %s/%s\n' "$PWD" "$archive"
