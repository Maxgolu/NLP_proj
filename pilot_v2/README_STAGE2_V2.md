# Stage 2 extension v2 — targeted exact patching for the updated Stage-1 candidates

Adds exact measurements to the completed, immutable `stage2_v1` run. Same model,
data, metric and patch mechanics (`stage2_engine.py` is imported unchanged).

Head lists come from the CPU coverage audit (`stage2_coverage_audit_v2.py`,
results under `results/stage2_v2_coverage/`) and are frozen in
`extension_v2_heads.json`:

- **Scope P** (all original prompt positions): 56 heads = (A ∪ B ∪ C) \ D, on the
  138 pairs outside the v1 all-head exact subset. v1 already holds these heads on the
  40 common pairs; those values are copied, not recomputed.
- **Scope F** (last prompt position only, the colon after `Answer`): 69 heads = A ∪ B,
  on all 178 pairs.
- Groups: A = 59 test-only RI candidates; B = 19 final-position test-QK heads;
  C = top-25 |exact| on the 40 common pairs (all already in D); D = the 92 v1 extension heads.

Every patched forward stores `(M, logit_clean_answer, logit_corrupted_answer, logit_source_name)`
at the first divergent answer token. Expected work: 7,728 Scope-P + 12,282 Scope-F
forwards ≈ 50 min on one replica at 0.15 s/forward, plus model loading (~40 min on NFS).

## Gate (every replica, before any measurement)

The full v1 gate (identity, token alignment, behavioral replication, self-patch,
self-attribution, embedding endpoint) plus, on the four gate families:

- Scope-F self patch (donor = clean) reproduces the clean metric, `< 1e-3`.
- Patching corrupted activations at all positions **before the first differing token**
  changes nothing, `< 1e-3` (causal attention on an identical prefix); the raw
  activation difference on that prefix is also recorded.
- On the gate pair that is in the v1 exact subset (family 0), Scope-P patching of six probe
  heads reproduces v1's saved `exact_delta` within `--replication-tolerance` (default 0.05 logits).
- `M == logit_clean − logit_corrupted` to `1e-4`.

## Upload and submit

Local PowerShell, from the project directory (verified c-002 key):

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\stage2_v2_extension_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

Cluster Bash:

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
tar -xzf stage2_v2_extension_update.tar.gz
python3 -m unittest test_stage2_extension_v2 -v      # planning test runs without torch; engine tests need CPU torch
bash submit_stage2_extension_v2.sh 6 --name stage2_v2_extension
```

`--nodelist s-004` pins the node if needed; the default excludes s-002 and s-005 only.
Use `--gate-only` with a distinct `--name` for a gate-only diagnostic. Only one job per
run name at a time.

## Monitor

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -n 30 logs/stage2ext2-JOBID.out
source runtime.sh
tail -n 20 "$PILOT_RUNS/stage2_v2_extension/worker_0.log"
```

Resume after a stopped job with identical code, inputs, GPU count and flags:

```bash
bash submit_stage2_extension_v2.sh 6 --name stage2_v2_extension --resume
```

## Verify and pack (cluster, after COMPLETED / 0:0)

```bash
source runtime.sh
python3 verify_stage2_extension_v2.py "$PILOT_RUNS/stage2_v2_extension" "$PILOT_RUNS/stage2_v1"
tar -czf stage2_v2_extension_results.tar.gz -C "$PILOT_RUNS" stage2_v2_extension
```

Local PowerShell download and extraction:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage2_v2_extension_results.tar.gz" .\results\
tar -xzf .\results\stage2_v2_extension_results.tar.gz -C .\results\
python pilot_v2\verify_stage2_extension_v2.py results\stage2_v2_extension results\stage2_v1
```

Outputs: `manifest.json`, `gate_*.json`, `pairs/*.npz` (178), `head_effects_v2.csv`, `summary.json`.
`head_effects_v2.csv` reports family-level importance (orders averaged first) for Scope P
(178 pairs) and Scope F (178 pairs), and the Scope-F mean change of each stored logit.
