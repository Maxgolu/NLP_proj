# Stage 3, Section 1.5 — controlled representation readout (optional extension)

Prepared 22 September 2026. **The 7B GPU readout has not run.** Local tests use a
tiny random OLMo2; they are implementation tests, not research results.

## What it measures

For the two writer heads whose effect enters at earlier positions, the residual
stream at their causal sites is captured after the writer's own block and injected
into a fixed readout prompt at the same block (Patchscopes-style). The next-token
log-probabilities of the candidate names are recorded.

| Writer | Main site | Second site | Layer of capture/injection |
|---|---|---|---|
| L17H1  | query-fact `is` (after the mother) | query-fact period | output of block 17 |
| L15H25 | last token of the queried child | query-fact `is` | output of block 15 |

Source conditions per site: `intact_clean`, `head_site` (writer replaced by the
corrupted donor at the site only), `head_span` (replaced from the first differing
token to the site), `intact_corrupted` (entity swap), `irrelevant` (same token role in
the other answer-side fact), `none` (readout prompt alone). Two readout prompts:
`identity` ("cat -> cat ; ... ; x ->") and `mother_of` ("Fact about a person: x
Question: Who is the mother of this person? Answer:"). Candidates: first token of
the query mother, the other candidate mother, the remaining distinct mothers and the
queried child. Population: the 40 common pairs (20 discovery families, both orders).

Primary statistic: `contrast = logprob(query mother) - logprob(other candidate)`;
`shift(condition) = contrast(condition) - contrast(intact clean)`. The entity-swap
shift is the reference scale; the head-replaced shifts measure how much of the site's
decodable mother identity depends on the writer head. Family means (orders averaged),
plus the fraction of families with a negative shift.

What it can and cannot show: a residual mixes all components up to that layer, so
only the intact-vs-head-replaced contrast says anything about the head; decoded text
is exploratory; a null result does not show the information is absent. Nothing here
is a channel or circuit claim.

Work: 240 source captures + 1,602 readout forwards (minutes on GPU; loading dominates).

## 1. CPU: prepare the frozen plan (local PowerShell, project root)

```powershell
Set-Location 'C:\Users\User\OneDrive\Documents\computer science\4B\NLP\final proj\project'
python pilot_v2\stage3_readout.py prepare --inputs results\stage3_inputs_v1 --run results\stage3_v1 --out results\stage3_readout_inputs_v1
python -m unittest discover -s pilot_v2 -p "test_stage3_readout.py" -v
python pilot_v2\build_stage3_readout_bundle.py
```

`prepare` reads only saved tables (`pairs.jsonl.gz`, `analysis/position_profiles.csv`);
no tokenizer or model. It refuses to overwrite an existing output directory. The tiny
model tests need CPU torch + transformers (the `tmp/stage3_test_deps` environment used
for Stage 3 works: prefix the commands with
`$env:PYTHONPATH='tmp\stage3_test_deps'`); without them the model tests are skipped
and the plan/analysis tests still run. The bundle is `pilot_v2/stage3_readout_v1_update.tar.gz`.

## 2. Upload (local PowerShell)

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\stage3_readout_v1_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

## 3. Cluster: extract, verify, gate, run (Bash)

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
test ! -e stage3_readout_v1 || { echo 'stage3_readout_v1 already exists'; exit 1; }
tar -xzf stage3_readout_v1_update.tar.gz
source runtime.sh
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('stage3_readout_v1')
for name,expected in json.loads((p/'bundle_hashes.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print('All uploaded file hashes match.')
PY
PYTHONPATH="$PWD/stage3_readout_v1:$PYTHONPATH" python3 -m unittest test_stage3_readout -v
bash stage3_readout_v1/submit_stage3_readout.sh --name stage3_readout_v1_gate --gate-only
```

The gate checks, on the real model: identity injection (injecting a prompt's own
residual back into itself leaves the readout unchanged), a self patch of the writer
head leaves the residual unchanged, and the saved Stage-3 single-position importance
at three probe sites is replicated within 0.05 logits. Then:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,NodeList
cat "$PILOT_RUNS/stage3_readout_v1_gate/gate.json"
bash stage3_readout_v1/submit_stage3_readout.sh --name stage3_readout_v1
```

The full run repeats the gate and then measures; progress in
`$PILOT_RUNS/stage3_readout_v1/state.json`. Records are appended per pair
(`readout_records.jsonl.gz`); an interrupted run resumes with
`--name stage3_readout_v1 --resume` (same plan and code). Default exclusion: s-002
only. Two GPUs, one replica.

## 4. Analyze (cluster or local, CPU) and download

```bash
source runtime.sh
python3 stage3_readout_v1/stage3_readout.py analyze "$PILOT_RUNS/stage3_readout_v1"
tar -czf stage3_readout_v1_results.tar.gz -C "$PILOT_RUNS" stage3_readout_v1
```

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/stage3_readout_v1_results.tar.gz" .\results\
tar -xzf .\results\stage3_readout_v1_results.tar.gz -C .\results\
python pilot_v2\stage3_readout.py analyze results\stage3_readout_v1
```

Outputs in `analysis/`: `readout_events.csv` (one row per record),
`readout_summary.csv` (family means per writer/site/readout/condition),
`readout_paired.csv` (per-pair shifts), `readout_family_summary.csv` (the table to
report). `analyze` refuses incomplete runs and record counts that differ from the plan.

## Reading the result

- Entity-swap shift strongly negative and the head-span shift close to it: the site's
  decodable mother identity depends on the writer's own writes (supports "content").
- Entity-swap shift strongly negative but head-span shift near zero (and near the
  irrelevant shift): the mother is decodable at the site but not through this head
  (supports "address/other components"); the head's 5.7-logit effect then acts through
  something the readout does not decode.
- Entity-swap shift near zero: the site does not expose the mother to this readout;
  the readout prompts are the first suspect, not the model.
- `mother_of` vs `identity` at the child's last token (L15H25): whether the child's
  position exposes its mother (a binding) or only the child's own identity.
