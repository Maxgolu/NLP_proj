# Stage 1 anatomy and gated Stage 2

The anatomy analysis is independent. It never gates Stage 2 or loads a model.
Stage 2 uses the existing pinned OLMo model, data, runtime, and Stage-1 discovery
split. No model download is needed.

## 1. Saved-event anatomy (CPU only)

Already generated locally under `results/stage1_v4_review/anatomy_all_heads`.
To repeat from the project directory, choose a new output directory:

```powershell
python pilot_v2/stage1_anatomy.py results/stage1_v4_review/stage1_v4_calibrated --out results/stage1_v4_review/anatomy_repeat
```

Outputs: `head_anatomy.csv`, `summary.md`, `provenance.json`. The analysis
partitions all saved QK-passing events by fact block, current-position block,
and source distance (self, previous token, or at least two tokens). It reports
opportunities, pass counts, RI strength and family coverage, and reconciles
every head's counts with Stage 1. The answer position is the last bare-prompt
token, before any shared answer prefix. These are descriptive measurements,
not independent causal evidence. Repeated events within families remain related.

## 2. Upload and submit

From PowerShell in the project directory, using the previously verified c-002 key:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\stage2_v1_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

Then in Bash on Slurm:

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
tar -xzf stage2_v1_update.tar.gz && bash submit_stage2.sh 6 --name stage2_v1
```

Choose 2, 3, 4, 5 or 6 GPUs. Two GPUs per model replica is the default minimum:
2/3 GPUs give one replica; 4/5 give two; 6 give three. Whole families are assigned
by estimated work, including the much heavier all-head exact subset. The helper
requests one node, 4 CPUs and 32 GB host RAM per replica, and eight hours.
An extra third GPU in a replica does not guarantee proportional speedup.
Only submit one job for a given run name at a time.

All replicas load once and run the mandatory gate. The full experiment starts
only after every replica passes. A worker failure stops the job. Initial model
loading is staggered to reduce simultaneous reads from shared storage. Cold
weight loading can still be slow.

The gate checks model/data identity, token alignment for all pairs, behavioral
replication, self-patching, self-attribution and full embedding replacement. It
also measures six heads on four families (both orders), including shared answer
prefix cases. Head usefulness or approximation agreement is not a gate condition.

## 3. Full experiment scope

- First-order attribution: all 1,024 heads on all 178 discovery pairs.
- Exact head patching: all 1,024 heads on 40 pairs (20 randomly chosen families,
  both orders). This is **not** exact all-head patching on all 178 pairs.
- If Spearman correlation of mean signed head effects on those 40 pairs is
  below 0.7 or undefined, compute a five-step input-embedding IG estimate on all
  178 pairs. This remains an approximation; it is not an exact head-path integral.
- Exact extension on all 178 pairs: union of the top 25 supported first-target
  RI heads, top 25 last-target RI heads, top 25 absolute mean estimated effects,
  L3H11/L9H22/L16H4, and 25 random controls. Existing exact results are reused.
- If IG agreement also fails, exact measurements are retained and the summary
  explicitly marks the estimate as unvalidated.

The metric is the logit difference at the first token where the two answers
differ. Its sign always follows the clean answer. Corrupted head outputs replace
clean outputs at all original prompt positions. Shared answer-prefix positions
are recomputed, not copied. A negative delta means movement toward the corrupted
answer. These effects identify intervention-sensitive heads; they do not alone
identify circuit edges or establish semantic specificity.

## Monitor and resume

Replace JOBID with the number returned by submission:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
tail -n 40 logs/stage2-JOBID.out
source runtime.sh
tail -n 30 "$PILOT_RUNS/stage2_v1/worker_0.log"
```

The main log reports worker states every 30 seconds. Worker logs contain model
loading and pair/head progress. Results are saved after each pair. After the
previous job has stopped, resume with identical code, inputs, GPU count and flags:

```bash
bash submit_stage2.sh 6 --name stage2_v1 --resume
```

The gate runs again; completed pair files are reused. For a gate-only diagnostic
use a distinct name and `--gate-only`; the normal command already includes it.

## Results and download

Completion requires successful Slurm exit and `summary.json` with `complete: true`.
Inspect `gate_*.json`, `head_effects.csv`, `first_order_quality.json`,
`final_estimate_quality.json`, `shortlist.json`, and `manifest.json`. Per-pair
arrays retain individual effects for subsequent family-level statistical analysis.

```bash
source runtime.sh
tar -czf stage2_v1_results.tar.gz -C "$PILOT_RUNS" stage2_v1
```

From local PowerShell:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage2_v1_results.tar.gz" .\results\
```

## Git update (project directory)

Stage only this change; keep unrelated documents, results and local dependencies out:

```powershell
git add -- pilot_v2/stage1_anatomy.py pilot_v2/stage2_common.py pilot_v2/stage2_engine.py pilot_v2/stage2_run.py pilot_v2/test_stage2.py pilot_v2/stage2.sbatch pilot_v2/submit_stage2.sh pilot_v2/README_STAGE2.md
git diff --cached --stat
git commit -m "Add independent stage1 anatomy and gated multi-GPU stage2"
git push origin main
```

Local validation: 16 tests passed, including linear exact/Taylor/IG agreement,
nonlinear finite differences, shared-prefix patch scope, self controls and work
partitioning. The GPU preflight has not yet been executed on OLMo. The upload
archive contains code only; no model weights, data or local test dependencies.
