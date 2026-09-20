# Stage 1 extension: test-only Relation Index audit

This is a discovery analysis of saved Stage-1 QK events, with missing raw-embedding
OV scores completed from the pinned OLMo-2-1124-7B weights. It does not read Stage-2
effects, run new model forward passes, train a model, or use validation families.
The original experiment is preserved. Nothing is submitted automatically.

## Run from the existing Slurm environment

The release archive extracts into a **new** `ri_test_audit_v1/` directory under
`pilot_v2`. It includes isolated copies of the existing model loader/helper/lock;
it does not replace your existing runtime or Stage-1/Stage-2 scripts.

From local PowerShell, in the project directory:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\ri_test_audit_v1_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

On Slurm, in Bash:

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
tar -xzf ri_test_audit_v1_update.tar.gz
bash ri_test_audit_v1/submit_ri_test_audit.sh
```

The wrapper finds `stage1_v4_calibrated` under `../storage/runs` or `storage/runs`,
resolves symlinks, and refuses an ambiguous/missing source. The output is
`runs/ri_test_v1` under that same storage root. It prints both paths before submission.
It uses **2 GPUs, 4 CPUs, 32 GB host RAM, one node, and a 4-hour limit**, with the
existing partition/account/exclusions. Four hours is an allocation limit, not a
measured runtime estimate. Checkpoint loading may be slow on shared storage.

If a different storage root or run name is needed, give it explicitly:

```bash
bash ri_test_audit_v1/submit_ri_test_audit.sh /home/yandex/DLWorkShop2025b/maximg/storage stage1_v4_calibrated ri_test_v1
```

No manual `source runtime.sh` is necessary for these commands: the job does that
internally and then uses the actual resolved storage root. No package installation
or model download is requested. The existing local pinned model must be present.

## Monitor, completion, download

Replace `JOBID` with the number returned by `sbatch`:

```bash
sacct -j JOBID --format=JobID,JobName%20,State,Elapsed,ExitCode
tail -n 30 logs/ri-test-JOBID.out
tail -n 30 logs/ri-test-JOBID.err
```

Success requires `COMPLETED`, exit code `0:0`, and the final `COMPLETE` log line.
The pack helper additionally validates the completion marker and saved-file
integrity internally; no manual checksum commands are needed.

On Slurm:

```bash
bash ri_test_audit_v1/pack_ri_test_results.sh
```

Locally, from the project directory:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/ri_test_v1_results.tar.gz" .\results\
tar -xzf .\results\ri_test_v1_results.tar.gz -C .\results
```

This produces `results/ri_test_v1/`. Do not extract over an existing result folder
you want to preserve. The pack helper refuses to overwrite an existing archive.
The SSH command retains the previously verified c-002 host key; if the key changes,
verify the new key before proceeding rather than disabling host checking.

## Phases and resumption

The submitted job runs three phases in sequence, so you only submit once:

1. **CPU prepare:** verify 534 prompts / 89 even-ID discovery families; reconcile
   source event counts against `head_stats.csv`; identify query-relevant facts;
   freeze source/code hashes, policy and prepared inputs.
2. **GPU OV completion:** tokenize all saved prompts to verify token/offset identity;
   load the pinned model once; evaluate raw embedding -> head V -> head O -> vocabulary.
   No calls to `model.forward`. For every event, compare recomputed first/last target
   scores, saved entity controls and available fixed-token controls to historical
   scores. Absolute tolerance `2e-4`, relative tolerance `2e-3`; failure stops the run.
   Save one atomic, verified shard per head. The random baseline cannot be replayed
   because its sampled token ID was not saved in the original run.
3. **CPU analyze:** check every expected event/shard, aggregate by family, build
   paired-variant tables and the descriptive shortlist, and write `summary.json`
   with `complete: true` only at the end.

Completed head shards are reused after interruption. Submit the **same command**
after confirming the old job has stopped; do not submit concurrent jobs for one
output directory. Input or Python code changes are rejected on resume. To change
the policy/code, use a fresh output run name and keep the previous results.

A hard Slurm kill may leave `runs/ri_test_v1/.running.lock`. If that happens, first
confirm with `sacct` that the owning job has stopped. Only then remove that one
lock file at the printed output path and resubmit. Do not delete the result folder.
If `gate_failure.json` exists, inspect it before retrying; do not relax the numerical
tolerance merely to get a completion marker.

For independent local CPU work, Python + NumPy are enough:

```powershell
python pilot_v2/ri_test_audit.py prepare --source results/stage1_v4_review/stage1_v4_calibrated --out results/ri_test_v1_prepared
python pilot_v2/ri_test_audit.py analyze --out results/ri_test_v1
```

The second command requires complete GPU shards from this exact code/policy version.
It reuses the same analysis as the job. You do **not** need to run it after the
standard job, which already produces all analysis outputs.

## Frozen measurement specification

- **Population:** saved events with fact AND current position in the test block;
  original source-last-token anchor and QK dominance `>2.2`. All 1024 heads are
  represented, including zero-pass heads. QK failures are not treated as scored RI=0.
- **Two scopes:** all four test facts, or the one fact whose source is the entity
  named in the question and whose target equals that prompt's gold answer.
- **Positions:** all_test, earlier, final bare-prompt token. Each is also divided
  into all/self/previous/long-distance. These groups overlap; do not sum them.
  The final bare-prompt position precedes any shared answer prefix used in Stage 2.
- **Two anchors:** first and last token of the target, with matching first/last
  anchors for each control. These are separate analyses, not a score for the full name.
- **Name controls:** every other fully visible fact-tail name in the same test.
  No equal-length exclusion. Names are equally weighted; anchor-token collisions
  stay in the comparison as ties. A source-only name is not a name control here.
- **General controls:** up to three unique, case-sensitive, alphabetic non-name
  word types sampled from the visible test using seed 20260916 and prompt ID/position.
  The first fully visible occurrence supplies the token anchors. The sample is
  shared across heads and facts at that prompt/position, and its words, token IDs,
  positions and scores are saved. Punctuation and all entity names are excluded.
  A fully visible word means its last overlapping tokenizer token is at or before j.
  This is a new general-background control, not an exact replication of the paper's
  fixed tenth-token control. Because the template is small, many words recur.
- **Normalization:** exactly the original full visible context's unique token IDs,
  including demonstrations. Restricting events/controls to test does not remove ICL
  conditioning or change the normalization denominator.
- **Event statistics:** target score, control mean, signed gap, rank interval
  (ties within `1e-7`), fraction of controls beaten/tied, top including ties and
  strict top, token-collision count. First and last are both saved.
- **Aggregation:** average each event metric within a family, then average active
  family means equally. No event means an undefined conditional score, not zero.
  Each metric reports its own event count and family support. `names_target_mean`
  and `words_target_mean` use the same populations as their control means; the
  unrestricted `target_mean` can have a different population. For QK frequency,
  average family pass/opportunity fractions including zero-pass families; also
  report the event-weighted fraction separately.
- **Variant comparisons:** match family, order and source-name identity (not fact
  index, which changes under reorder). Compare base/corrupted and base/reorder
  separately. Final-position comparisons align the measurement position; all_test
  conditional means may contain different positions. Missing passes/scores are
  explicit. Tables retain pairs with at least one pass; neither-pass pairs can be
  recovered from the frozen prompt universe and are not evidence for RI strength.

Raw-embedding OV scores are fixed for fixed head/current token/visible token set.
Changing a fact's assigned target can therefore change which coordinate is read
without changing the OV vector. The paired comparison does not by itself prove
contextual semantic tracking. No independence of variants/orders is assumed.

## Discovery shortlist (chosen before new OV results)

Union of top **10** positive family means for each of:

- target RI, name-control gap, general-word-control gap;
- first and last anchors;
- all_facts and query_fact;
- all_test and final positions.

Require at least **20** scored events and **10** active families for the particular
metric. Ties in ranking use layer/head order deterministically. These thresholds
are explicit discovery heuristics, not calibrated significance. The lower event
cutoff than the old 50-event rule allows sparse final-position patterns with broad
family coverage to enter. Up to 240 union slots exist; overlap reduces the number
of distinct heads. All supported and unsupported heads remain in the full tables.
Do not increase top-k or relax support simply because the shortlist is small.

Historical L3H11/L9H22 are recorded as references, not forced into the selected set.
Token concentration, leave-family-out diagnostics, and detailed case studies are
deferred until after the shortlist, as agreed. Validation families remain unused.
There are no new p-values or claims of causal/semantic validation.

## Files to inspect after completion

| File | Contents |
|---|---|
| `manifest.json` | Frozen policy, original run/model provenance, code/input hashes |
| `model_gate.json`, `heads/LxHy.json` | Tokenizer/model identity and per-head replication checks |
| `heads/LxHy.jsonl.gz` | Every test QK event with first/last target and control scores/ranks |
| `head_summary.csv` | All-head scope/position/distance/variant/anchor summaries, support and frequencies |
| `family_metrics.csv` | Active-family means and support, including per-variant results |
| `paired_variants.csv` | Source-matched base/corrupted and base/reorder comparisons |
| `candidates.json` | Candidate union and each head's explicit selection reasons |
| `summary.json` | Final completion, counts and output integrity information |

## Local checks

```powershell
python -m unittest discover -s pilot_v2 -p test_ri_test_audit.py -v
```

Tests cover scope, relevant-fact matching, visibility, reproducible word controls,
ties, zero-pass denominators, family weighting, paired comparisons and locks/hash
checks. An optional CPU PyTorch test checks the head OV slice against full linear
projections. These are not OLMo GPU validation: the actual replication gate runs
in your Slurm job.
