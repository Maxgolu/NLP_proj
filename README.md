# NLP Final Project — From Semantic Induction Heads to Circuits

This repository studies how language models implement in-context semantic
relations. The original proposal targeted multi-hop reasoning and asked whether
it is carried by isolated Semantic Induction Heads (SIHs) or by distributed
Semantic Induction Circuits (SICs). After behavioral pilots exposed several task
confounds, the current priority became a controlled **single-hop causal audit**:

> Do heads selected by the Relation Index (RI) definition of Ren et al. causally
> carry OLMo-2-7B's in-context relational ability, and does RI miss important
> components of a larger circuit?

Two-hop composition and checkpoint-development analyses remain later extensions.
No circuit has yet been established.

## Current status (20 September 2026)

Model: base `allenai/OLMo-2-1124-7B`, revision
`7df9a82518afdecae4e8c026b27adccc8c1f0032`, with 32 layers and 32 attention
heads. The model is used without fine-tuning.

- Stage 0: hook, metric, numerical, and costing checks completed.
- Stage 1: observational RI scan completed for all 1,024 heads on 534 prompts;
  matched-target calibration and saved-event anatomy completed.
- Stage 1 extension: test-only RI with name/word controls (`ri_test_v2`, job
  905839) completed; 59 descriptive candidates. Report:
  `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`.
- Stage 2: `stage2_v1` (job 888691) completed and analyzed: first-order screen
  for all heads on 178 pairs, exact patching for all heads on 40 pairs, exact
  extension for 92 heads on 178 pairs. Screen calibration Spearman 0.945.
- Stage 2 extension v2: `stage2_v2_extension` (job 912879) completed: exact
  all-position patching for the 56 uncovered new candidates on the remaining
  138 pairs, and final-position-only patching for 69 heads on all 178 pairs,
  with separate answer/source logits. Report (updated in place):
  `חומר כתוב/Stage2_Results_and_Analysis.tex`; analysis
  `results/stage2_v2_analysis/`.
- Main Stage-2 findings: the historical RI heads have no effect; the test-only
  RI priority heads are causally negligible; the name-gap statistic is
  negatively rank-correlated with causal importance (copying hypothesis, untested);
  causal heads split into layer-6–11 heads acting at test-fact positions and
  layer-16–26 heads acting entirely at the answer position.
- Stages 3–4: not started. Next: final-position scope for L17H1/L27H6/L18H18,
  weight copying score, path patching writer→reader.

The authoritative transition notes are in `RESEARCH_HANDOFF.md` (its top
"UPDATE" section is the latest state). For the current method and evidence,
read the files in this order:

1. `חומר כתוב/Single_Hop_Methodology_stage1_updated.tex` (+ Section 4 and 5
   replacement fragments in the same folder, not yet merged)
2. `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`
3. `חומר כתוב/Stage2_Results_and_Analysis.tex`
4. `pilot_v2/RUNBOOK_stage2_v1.md`, `pilot_v2/RUNBOOK_stage2_v2_extension.md` (run books)

## Documentation map

There is one entry point: this file. Every other `README*`/`*_README*` is a
**run book** for a specific experiment (commands to upload, submit, monitor,
verify and download) written at the time of that run, or a historical record.

| File | Role | Status |
|---|---|---|
| `README.md` (this file) | project entry point, status, reading order | current |
| `RESEARCH_HANDOFF.md` | assistant/collaborator handoff, evidence hierarchy | current |
| `pilot_v2/RUNBOOK_stage2_v1.md` | run book: `stage2_v1` (job 888691) | current |
| `pilot_v2/RUNBOOK_stage2_v2_extension.md` | run book: `stage2_v2_extension` (job 912879) | current |
| `pilot_v2/RUNBOOK_stage1_audit.md` | run book: Stage-1 audit and calibration | current |

Historical run books of the behavioral pilots (Pythia-1B, completion v1–v3.1,
OLMo-2 comparison) were removed on 20 September 2026; they remain in the Git
history up to that commit, and their scientific content is in
`research_history_overleaf_en.tex` and `experimental_setup.tex`.

## Why the project changed direction

The project began with a multi-hop semantic-reasoning proposal. Behavioral work
then proceeded through several controlled revisions:

- Pythia-1B and Pythia-6.9B yes/no prompts showed strong label bias and poor
  compliance.
- Name-completion v1 produced a real direct-retrieval signal, but two-hop scores
  were confounded by candidate frequency, option position, and valid bridge-name
  completions.
- v2 balanced those factors but used unnatural graph-navigation instructions;
  performance fell to chance even for one-hop retrieval. It is retained as a
  negative result.
- v3/v3.1 returned to natural language and named composed relations. Crossing
  hop count with fact order exposed a strong layout dependence in Pythia.
- A dual-tokenizer audit and model comparison selected OLMo-2-7B as the primary
  mechanistic platform. OLMo improved order symmetry and shuffled-fact behavior,
  but still showed a composition bottleneck on rewired two-hop examples.
- To avoid building a multi-hop explanation on an unverified primitive, Phase A
  now audits the single-hop mechanism first. Phase B will compose verified
  single-hop components only if the evidence supports it.

Historical context is preserved in `research_history_overleaf_en.tex`,
`experimental_setup.tex` and `חומר כתוב/NLP___current_position.pdf`. These are
historical records, not the final Stage-2 specification.

## Current experimental substrate

The single-hop dataset is generated by `pilot_v3/build_data_singlehop.py` and
stored under `pilot_v3/data_singlehop/`. It contains 200 fictitious kinship
families using the relation *mother of*. Every family has:

- a base prompt;
- a token-aligned corrupted twin in which the two answer-side names swap;
- a reordered-facts control;
- both fact orders;
- construction-time fact, entity, and character-span annotations;
- four single-hop-only demonstrations shared across its variants and orders.

Behavioral eligibility retained 176 families. The discovery set contains 89
even-ID families; 87 odd-ID families are reserved for later validation. At four
shots, both-orders accuracy was 0.93 for base, 0.94 for corrupted, and 0.89 for
reordered prompts. The full report is
`results/singlehop_baseline/singlehop_baseline_report.md`.

## Stage 1: what was measured

The final run is under
`results/stage1_v4_review/stage1_v4_calibrated/`.

For each annotated fact and eligible current position, RI first applies a QK
condition: the head's maximal attention must point to the last token of the
annotated source name and exceed the strongest competitor by `tau = 2.2`. In
the kinship task the child is the source and the mother is the target.

The OV statistic then projects the **raw current-token embedding** through that
head's value and output matrices and the unembedding. It is normalized over
visible context tokens. The first target token is primary and the last target
token is a sensitivity measure. This is not the head's contextual output, a
final-logit contribution, or a causal effect.

The historical pooled rule selected L3H11 and L9H22, but event inspection showed
that their scores were concentrated in repeated demonstrations and local
attention patterns:

- L3H11: 149 scored passes, all one token backward from a period/newline token;
- L9H22: 86 scored passes, all self-attention;
- more than 99% of each head's summed first-target score came from demonstrations.

A separate matched-target randomization test retained test-block events with
both candidate mentions visible, equal token lengths, and distinct first tokens.
It weighted active families equally, used 100,000 family-consistent swaps, and
applied Holm correction across all 1,024 heads. Only 80 heads had sufficient
support, and none survived correction. L16H4 had the smallest raw p-value
(`p = 0.00581`, 87 events across 40 families). Insufficient support is not a
negative causal result.

An offline anatomy of all saved QK-passing events found 20 heads with at least
one dominant-attention event from the final bare-prompt token. There were 748
such events, including 744 to test-fact sources. L16H21, L16H1, and L16H4 had
230, 223, and 63 test-source events respectively. These are attention events,
not correct answers or proof of target promotion.

## Stage 2: implemented causal map

The implemented Stage-2 code is in `pilot_v2/stage2_run.py`,
`pilot_v2/stage2_engine.py`, and `pilot_v2/stage2_common.py`.

The intervention metric is the clean-answer logit minus the corrupted-answer
logit at the first answer token where the candidates diverge, conditioned on
their shared answer prefix. Its sign remains fixed to the clean answer even on
the corrupted input. A negative patching delta therefore means movement toward
the corrupted answer.

For one head, exact patching replaces its `o_proj` input slice with corrupted-run
activations at all original prompt positions. Shared answer-prefix positions are
recomputed. The completed `stage2_v1` run contains:

1. first-order gradient attribution for all 1,024 heads on all 178 discovery
   clean/corrupted pairs;
2. exact patching for all 1,024 heads on 40 pairs (20 whole families, both
   orders);
3. a five-step input-embedding IG fallback if signed-mean Spearman agreement
   between first-order and exact effects is below 0.7 or undefined;
4. exact patching on all 178 pairs for the union of supported top-RI heads,
   top estimated-effect heads, L3H11/L9H22/L16H4, and random controls.

The IG fallback is still an approximation: gradients are sampled along the
input-embedding path and contracted with fixed clean-to-corrupted head-output
deltas. Exact measurements remain the record for individually discussed heads.

All three persistent model replicas must pass the gate before the experiment
continues. The gate checks model/data identity, all-pair token alignment,
behavioral replication, self-patching, self-attribution, complete corrupted
embedding replacement, and six-head exact/gradient previews. It does not gate
on effect sign, head usefulness, or approximation agreement. CPU unit tests
passed locally, but those tests do not establish that the OLMo GPU gate passed.

Monitoring, resumption, and safe download commands are documented in
`pilot_v2/RUNBOOK_stage2_v1.md`. Completion requires both a successful Slurm exit and
`summary.json` with `complete: true`. Both `stage2_v1` and `stage2_v2_extension`
met this; see the Stage-2 report for results.

The extension (`pilot_v2/stage2_extension_v2.py`, run book
`pilot_v2/RUNBOOK_stage2_v2_extension.md`) adds a second patch scope — the final prompt
position only — and stores the clean-answer, corrupted-answer and source-name
logits separately. Head lists come from `pilot_v2/stage2_coverage_audit_v2.py`.
Analysis: `analyze_stage2_results.py` (v1) and `analyze_stage2_v2_results.py` (v2).

## Interpreting Stage-2 results

Before scientific interpretation, verify the gate files, manifest and hashes,
successful job exit, pair counts, completion marker, and absence of missing or
non-finite vectors. Keep these distinctions explicit:

- exact all-head results cover 40 pairs; exact all-pair results cover only the
  selected extension heads;
- gradient and IG values are estimates until calibrated against exact effects;
- repeated variants and orders within a family are not independent samples;
- a high-RI/low-effect head is not automatically a false positive because patch
  scope, redundancy, nonlinear interactions, and the chosen output metric matter;
- a list of intervention-sensitive heads is not an edge-validated circuit.

If Stage 2 passes these checks, the next scientific stage is QK/OV mechanism
analysis, path interventions, redundancy tests, and a minimal-circuit
faithfulness evaluation on the held-out validation families. The pre-registered
faithfulness criterion is at least 80% recovery of the full model's
clean–corrupted logit-difference gap.

## Repository map

- `pilot_v2/` — OLMo/Pythia runners, Stage 0–2 code, Slurm wrappers, tests, model
  locks, and operational documentation.
- `pilot_v3/` — later behavioral generators, tokenizer audits, and the current
  single-hop generator/data.
- `results/` — downloaded run outputs, audits, and reports.
- `חומר כתוב/` — current methodology and research reports for Overleaf.
- `research_history_overleaf_en.tex` — detailed pre-Phase-A research history.
- `RESEARCH_HANDOFF.md` — current handoff, evidence hierarchy, and collaboration
  guidance.
- `analyze_stage2_results.py`, `analyze_stage2_v2_results.py` — Stage-2 analysis
  scripts (CPU only). Older `analyze_*.py` files at the root belong to the
  behavioral pilots.

The original proposal and course requirements are one directory above this
repository. The final submission is an ACL-format paper limited to eight pages,
excluding references and appendix; it must report reproducible settings,
baselines, limitations, negative results, and an AI disclosure.
