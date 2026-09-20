# Stage 1 follow-up audit (2026-09-13)

Version 4 adds CPU-only matched-target calibration after the scan.
This update keeps the dataset, model, raw-embedding OV formula, source anchor,
first-target primary score, and last-target sensitivity score unchanged.
It does not implement stage 2. The new randomization test is a separate
conditional statistic, not a replacement of the original pooled RI definition.

## Changes

- Intervention contrast: first *divergent* answer token, conditioned on the
  common candidate token prefix. A terminal period handles prefix names.
  Invalid tokenization boundaries fail explicitly. Every selected prompt's
  specification is saved in `metric_specs.json`. Only original prompt positions
  are patched; any common answer prefix is recomputed, never donor-patched.
- Dominance: all qualifying ratios from all heads, layers and selected prompts.
  Qualification is `argmax == source`, before applying tau. Raw float32 values
  are saved in global and per-layer files; exact empirical percentiles are
  calculated in place on disk-backed arrays. No sampling cap. Population is
  event-weighted and includes repeated demo-prefix events; this is not a
  family-balanced or population-independent calibration.
- `ri_events.jsonl.gz`: every QK pass, scored or unscored, with prompt/family,
  fact, destination position, demo/test categories and block indices, token
  identities, first/last scores, random-token control, and other visible tail
  names within the same individual block as descriptive entity controls.
  Shared control/target tokens are flagged, not counted as distinct controls.
- `ri_opportunities.jsonl.gz`: exact denominators per prompt, fact, layer and
  destination category, shared by heads within that layer. Zero-pass groups
  remain reconstructible. `audit_prompts.jsonl.gz` preserves text, annotations,
  token strings/IDs, offsets and demo IDs for inspecting specific events.
- `analyze_stage1_audit.py` summarizes L9H22 and L3H11 by family, prompt and
  source/destination category. It reports unique prefix/fact conditions and raw
  event counts separately, and provides concrete top-scoring examples and
  descriptive target-versus-other-tail comparisons. Family means are
  conditional on scored events, not behavioral accuracy or confidence bounds.
- Source/data/baseline hashes and selected IDs are recorded. Use a NEW run name;
  an existing run directory is rejected rather than overwritten.

## Files to upload

Upload `stage1_v4_calibrated_update.tar.gz` and extract it in `pilot_v2`.
It contains the scan, helpers, analysis/calibration tools, job wrapper, tests
and this document. Existing `runtime.sh`, pinned model and baseline files on
the cluster are used as before. No model download is included.

From the existing `pilot_v2` directory on Slurm:

```bash
mkdir -p logs
sbatch stage1_calibrated.sbatch --name stage1_v4_calibrated --shots 4 --tau 2.2
```

This job runs the scan, two-head descriptive audit, and all-head calibration
in sequence. A subprocess failure stops the job. The model process exits before
CPU calibration starts, although the Slurm GPU allocation is retained until
the short postprocessing finishes. To rerun ONLY calibration after a successful
scan, without loading the model, use a new output directory name:

```bash
source runtime.sh
python3 calibrate_stage1_null.py "$PILOT_RUNS/stage1_v4_calibrated" \
  --output-name null_calibration_repeat
```

If `PILOT_RUNS` overrides the runs location, use that location instead.
For the separately labelled sensitivity run, after the reference run finishes:

```bash
source runtime.sh
sbatch stage1_calibrated.sbatch --name stage1_v4_p95 --shots 4 \
  --tau-from "$PILOT_RUNS/stage1_v4_calibrated/dominance_percentiles.json"
```

The complete population is measured at each run; tau does not alter that
population. It only alters which events receive RI scores. Old `stage1_v2`
aggregate files cannot reconstruct the new per-event diagnostics.

## Matched-target calibration specification

- Primary scope: test facts whose target is one of the two answer candidates,
  observed at positions in the test block. Demo events remain descriptive.
  Both candidate tail mentions must be fully visible by the scored position,
  equal in token length, and have distinct first-token IDs at those occurrences.
  This deliberately narrower population is reported separately from pooled RI.
- For each head/family, average the saved true-target scores and the saved
  scores of the other answer name. Hold QK selection, the input, OV scores and
  normalization fixed. Weight active families equally, not by event count.
- In each of 100,000 draws, use one fair swap decision per family. The same
  decision applies to every event, variant, fact order and head in that family.
  Generate a null distribution of the family-averaged target score. This
  requires no additional model forwards or gradients.
- A one-sided Monte Carlo p-value is `(1 + number of null scores >= observed)
  / (B + 1)`, with tolerance for floating-point ties. Apply Holm across all
  1,024 heads, including unsupported heads as p=1 internally. Alpha is 0.05.
- Require at least 50 matched scored events across at least 10 families.
  These are newly documented support rules, not the original all-event RI
  eligibility rule. Unsupported heads are reported as such, not negative
  discoveries. Thresholds are configurable and any change is recorded.
- Interpretation requires conditional exchangeability of the true/control
  assignments under the null. Matching and family blocking help but do not
  prove this assumption. Token/name preferences, QK selection, and selecting
  the behavioral working set can limit interpretation. This is not evidence
  of causality and does not establish an unconditional false-positive guarantee.
- The test covers the FIRST-target primary score only. Last-target RI and demo
  scores remain sensitivity/descriptive analyses; they are not extra unreported
  significance tests. No bootstrap confidence intervals are implemented.

Outputs in `null_calibration/`: `head_null_calibration.csv` (including effect,
null quantiles, raw and Holm-adjusted p-values, support and selection),
`family_scores.json`, `coverage.json` (exclusion counts), and `method.json`
(seed, parameters, assumptions, input/code hashes and elapsed time). Family
score tables and seed permit regenerating null draws; full draws are not saved.

Implementation references: [SciPy paired permutation tests and Monte Carlo
p-values](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
and [statsmodels multiple-testing methods](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
Neither package is required by the implementation; it uses existing NumPy.

## Validation and pending work

CPU unit tests cover shared-prefix scoring specification, no-prefix behavior,
exact all-layer quantiles and duplicate-condition recognition. Full model
execution and GPU self-patch checks must still run on the cluster. The existing
preflight is intentionally small, not a new multi-family causal study.

Additional tests cover Holm correction, ties, unsupported heads, agreement with
a small enumerated null, family duplication and control eligibility exclusions.
A synthetic CPU benchmark (244 tested heads, 89 families, 100,000 draws,
1,024-head correction) took 1.28 seconds locally; this excludes event-file
reading and is not a cluster runtime promise. Matrices are processed in blocks
of 16 heads, and no full draws-by-heads array is saved.

The old random-token scores and null reservoir remain descriptive. No
layer-specific tau, contextual-residual OV variant or new causal preview was
introduced. Full GPU execution remains pending.
