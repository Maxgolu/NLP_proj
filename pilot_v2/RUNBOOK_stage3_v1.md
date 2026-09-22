# Stage 3 — characterize individual heads

Prepared 21 September 2026. **The 7B GPU experiment has not run.** Local tests
use randomly initialized tiny OLMo2 models; they are implementation tests, not
research results. First submit the GPU gate, inspect its cost estimate, then
submit the complete experiment. Stage 4 remains separate.

## What is ready

- Frozen inventory: 105 heads; all applicable group labels retained.
- Reuse exact Scope P on all 178 pairs and existing Scope F for 59 inventory heads.
- New Scope F: 46 heads × 178 pairs = 8,188 interventions.
- Position profiles: all 105 heads, original common 40 pairs, first changed token
  through final colon. 176,820 extra interventions; final-colon results reused.
- Reverse P and F: fixed ten heads × 178 pairs × 2 = 3,560 interventions.
- Attention/value factorial: all 105 heads × 178 pairs × 4 = 74,760 interventions.
- Unconditional attention/output anatomy: 20 common families × two orders ×
  three variants = 120 prompts, plus 40 changed-query controls.
- Contextual RI: the same saved scored events, controls, anchors and visible
  token populations. Zero-event heads are explicitly insufficient support.
- Copying weights: all inventory heads, unique first/last test-name token IDs,
  plus a seeded 128-token ordinary-word reference population.
- Both synthetic benchmarks: all inventory heads, 32 trials each. Repeated
  sequence length 32, distinct tokens, two repeats; 31 second-repeat next-token
  probes per trial. Source is the first-repeat occurrence of the next token.
  Retrieval uses 16 distinct key/value pairs, explicit key/value framing and one
  requested value. Queries cycle through all 16 slots twice. Only the value's
  unique record occurrence is the retrieval source. Save unconditional scores
  and scores conditional on a correct greedy next-token copy, with denominators.

Total: **263,328 new patched forwards**, plus gates, captures and diagnostics.
There is no automatic head dropping or optional selection among the four agreed
diagnostics. Patchscopes, broad spectral analysis and function-vector mediation
are optional, not implemented as mandatory runs. Paths, group interventions and
residual divergence belong to Stage 4.

## Reused data and CPU preparation

Already prepared locally: `results/stage3_inputs_v1/`. No additional local
preparation is needed before uploading the supplied archive.

`stage3_prepare.py` reads the original Stage-1/2 files and the saved behavioral
baseline inside `results/singlehop_results.tar.gz`. It verifies the baseline's
original SHA256. It writes portable input records, references and hashes;
it does not change the original runs. New diagnostic files include:

- `matched_approximation_error.csv`: exact and estimated effects on identical
  40 pairs, avoiding the old cross-sample comparison.
- `corrected_median_comparison.json`: 20,000 paired resamples of all 89 families
  for the RI-selected versus random median comparison. Selection is held fixed;
  this is descriptive, not a selection-adjusted confidence interval.
- `current_name_diagnostics.json`: saved RI events split by current-token
  overlap with target/control/neither; first/last anchors kept separate.

To reproduce preparation later (requires the original raw results), use a new
output directory; the tool intentionally refuses to overwrite an existing one:

```powershell
python pilot_v2/stage3_prepare.py --out results/stage3_inputs_rebuilt
python pilot_v2/stage3_run.py --check-inputs --inputs results/stage3_inputs_rebuilt
python pilot_v2/build_stage3_bundle.py --inputs results/stage3_inputs_rebuilt
```

Only NumPy is required for preparation, packaging and final analysis. The cluster
uses its **existing** PyTorch/Transformers environment and cached model; do not
upgrade packages or download a different model for this run.

## 1. Record the code in Git (local PowerShell)

Run from the project directory. Inspect the changes before committing. These
commands select the implementation and method files, not unrelated PDFs or raw
research archives. The current branch at preparation time was `main`.

```powershell
git status --short
git diff -- README.md RESEARCH_HANDOFF.md .gitignore
git add README.md RESEARCH_HANDOFF.md .gitignore
git add pilot_v2/build_stage3_inventory.py pilot_v2/build_stage3_bundle.py
git add pilot_v2/stage3_common.py pilot_v2/stage3_prepare.py pilot_v2/stage3_engine.py
git add pilot_v2/stage3_measure.py pilot_v2/stage3_run.py pilot_v2/stage3_analyze.py
git add pilot_v2/test_stage3.py pilot_v2/test_stage3_analysis.py pilot_v2/submit_stage3.sh pilot_v2/stage3.sbatch
git add pilot_v2/RUNBOOK_stage3_v1.md pilot_v2/stage3_v1_update.tar.gz.sha256.json
git add results/stage3_plan results/stage3_inputs_v1
git add "חומר כתוב/Stage3_Head_Level_Characterization_revised.tex"
git diff --cached --stat
git commit -m "Implement Stage 3 head characterization and reproducible Slurm workflow"
git push origin HEAD
```

Large prepared arrays/events and the upload archive are ignored by Git. Keep
them on OneDrive and transfer the archive separately. No commit/push was made
by the assistant. Git stores the code; the following archive updates the actual
cluster execution directory, which does not require a Git checkout or pull.

## 2. Upload the prepared archive (local PowerShell)

```powershell
Set-Location 'C:\Users\User\OneDrive\Documents\computer science\4B\NLP\final proj\project'
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\stage3_v1_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

This uses the already verified c-002 host key. It does not change host trust.
The archive extracts into **`pilot_v2/stage3_v1/`** and leaves Stage-1/2 code and
results intact. After measurements start, do not overwrite that package; resume
requires the same code and input hashes. Use a new package/run version for code
changes.

## 3. Extract, verify and run the GPU gate (cluster Bash)

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
test ! -e stage3_v1 || { echo 'stage3_v1 already exists; do not overwrite an active experiment'; exit 1; }
tar -xzf stage3_v1_update.tar.gz
source runtime.sh
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('stage3_v1')
for name,expected in json.loads((p/'bundle_hashes.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print('All uploaded file hashes match.')
PY
PYTHONPATH="$PWD/stage3_v1:$PYTHONPATH" python3 -m unittest test_stage3 test_stage3_analysis -v
python3 stage3_v1/stage3_run.py --check-inputs --inputs stage3_v1/inputs
bash stage3_v1/submit_stage3.sh 6 --name stage3_v1_gate --gate-only
```

Local-only tokenizer/input tests can be skipped in the portable cluster package;
the real tiny-OLMo2 hook tests should pass there. The GPU gate validates:
behavioral replication, self patches, unchanged prefix patches, saved exact P,
first-divergent-token alignment, unchanged capture forwards, QK/RoPE-aware
attention reconstruction, raw RI replication, and joint attention/value versus
ordinary output replacement. Every replica must pass before measurements start.

The installed runtime must expose SDPA for OLMo2. Unsupported capture or numeric
drift raises an error; it is never converted into missing/zero findings.

Two GPUs hold one model replica. Supported allocations: 2, 4, 6 GPUs (one, two,
three replicas). The default submission time limit is four hours. Node selection
and time can be supplied with `--nodelist s-004`, `--exclude ...`, `--time ...`;
the default excludes s-002 and s-005, as in the successful Stage-2 workflow.

### Retry after job 916685

Job 916685 failed on s-002 during CUDA initialization, at `cudaMemGetInfo`,
before model weights were loaded. This is not an observed out-of-memory error.
The message about changing CUDA_VISIBLE_DEVICES is a generic possible cause;
workers receive their GPU masks before a fresh Python process starts, as in
Stage 2. The underlying node/driver/device cause is not yet established.
For the already-uploaded package, retry on the previously successful s-004
without changing code, deleting results, or uploading again:

```bash
bash stage3_v1/submit_stage3.sh 6 --name stage3_v1_gate_s004 --gate-only --nodelist s-004
```

This submits only the gate. It may wait for s-004 resources. Inspect the new job
and `storage/runs/stage3_v1_gate_s004/` rather than the failed run directory.
If it succeeds, use `--nodelist s-004` for the full run too. If the same CUDA
initialization error recurs there, inspect allocation/device mapping and the
driver instead of repeatedly resubmitting. The local sbatch and rebuilt upload
archive now restore the historical exclusions; the retry command also works
with the original uploaded archive.

## 4. Inspect the gate, then run the experiment

Replace `JOBID` with the ID printed by `sbatch`:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -n 30 logs/stage3heads-JOBID.out
source runtime.sh
cat "$PILOT_RUNS/stage3_v1_gate/cost_estimate.json"
cat "$PILOT_RUNS/stage3_v1_gate/summary.json"
```

For the gate, `gate_passed: true` and `gate_only: true` are expected;
`complete: false` correctly means the research experiment has not run yet.
The estimate covers patched forwards; allow additional loading/diagnostic/IO
time. If it does not fit the partition limit, use the checkpoint/resume procedure
below. Do not reduce the head or family population to fit a job.

```bash
bash stage3_v1/submit_stage3.sh 6 --name stage3_v1
```

The full run repeats the gates. Read its progress with:

```bash
source runtime.sh
cat "$PILOT_RUNS/stage3_v1/state_0.json"
tail -n 30 "$PILOT_RUNS/stage3_v1/worker_0.log"
```

Each pair/batch of five heads is an atomic, checksummed checkpoint. Attention,
output and contextual RI are checkpointed by prompt. Resume retains completed
work; an interrupted unfinished batch is recomputed. The initial scheduling uses
fixed head shards; resume must keep the same GPU count.

## 5. Resume a stopped run

First confirm the old job is no longer running. Never run two jobs with the same
name. Normal termination releases the lock automatically:

```bash
bash stage3_v1/submit_stage3.sh 6 --name stage3_v1 --resume
```

If a hard kill left `.running.lock`, inspect it and confirm its recorded job has
ended before removing this **single lock file**:

```bash
source runtime.sh
cat "$PILOT_RUNS/stage3_v1/.running.lock"
sacct -j OLD_JOBID --format=JobID,State,ExitCode
# Only after confirming that the recorded job has stopped:
rm -- "$PILOT_RUNS/stage3_v1/.running.lock"
bash stage3_v1/submit_stage3.sh 6 --name stage3_v1 --resume
```

A failure of a numerical gate, checksum, model identity or schema is not a
preemption. Keep its logs and diagnose it before rerunning; do not relax the
tolerances to obtain a pass.

## 6. Verify, archive and download

The CPU audit/analysis runs automatically after all workers finish. It can be
repeated without a GPU. It requires all expected files, heads, pairs, event IDs,
synthetic probes and checksums before writing `complete: true`.

```bash
source runtime.sh
python3 stage3_v1/stage3_analyze.py "$PILOT_RUNS/stage3_v1" --inputs stage3_v1/inputs
cat "$PILOT_RUNS/stage3_v1/summary.json"
tar -czf stage3_v1_results.tar.gz -C "$PILOT_RUNS" stage3_v1
```

Local PowerShell:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage3_v1_results.tar.gz" .\results\
tar -xzf .\results\stage3_v1_results.tar.gz -C .\results\
python pilot_v2/stage3_analyze.py results/stage3_v1 --inputs results/stage3_inputs_v1
```

Use the same prepared inputs that were uploaded. Analysis accepts their local
path and verifies the plan hash rather than relying on the cluster path stored
in the manifest. Send back the results archive and the gate cost estimate/logs
for review. No extra model run is needed to read the saved attention matrices,
logits, matched RI scores or per-position interventions.

## Interpretation and outputs

`analysis/head_profiles.csv` summarizes every head. Additional tables contain
per-position effects, family effects, paired query/reorder controls, unconditional
attention, contextual output, copying weights, synthetic fingerprints, matched
contextual RI and support-filtered correlations. `head_cards.md` is an evidence
index, not an automatic mechanism classifier.

All causal signs use the original clean answer: positive importance is a fall
in its contrast under a corrupted donor. Reverse effects are recovery relative
to the corrupted baseline. Attention/value interaction is
`-(M11 - M10 - M01 + M00)`, on the same importance scale.

Projection ranks are **within visible first/last test-name anchors plus the
current token**, not the full vocabulary. Weight ranks use their stated name or
ordinary-word reference pool. Colliding token IDs retain ties. OLMo2's shared
post-attention normalization is not assigned additively to individual heads.
Output projections and RI are diagnostics, not final-logit effects.

The contextual RI summary uses identical valid event support for raw/contextual
comparisons, at least 20 comparisons across ten families for correlations.
Correlations report both the fixed 178-pair causal mean and a sensitivity using
only each head's RI-supported family/order pairs, with family weighting.
All heads still receive mandatory anatomy, copying and synthetic measurements;
zero RI or correct-copy event support is explicitly reported as insufficient.
Orders/events are averaged within family before averaging families. Synthetic
summaries average within trial and then across trials. Paired attention controls
also retain both orders and families for subsequent paired analysis.

The five questions at the end of Stage 2 map as follows: **1–3 are addressed by
Stage 3** (effect locations; relation-tracking attention; output/copying).
These tests may support or reject a hypothesis rather than guarantee a positive
answer. **4–5 require Stage 4** (paths between heads; joint/group interactions).
