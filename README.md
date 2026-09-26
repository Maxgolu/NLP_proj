# NLP Final Project — From Semantic Induction Heads to Circuits

This repository studies how language models implement in-context semantic
relations. The original proposal targeted multi-hop reasoning and asked whether
it is carried by isolated Semantic Induction Heads (SIHs) or by distributed
Semantic Induction Circuits (SICs). After behavioral pilots exposed several task
confounds, the current priority became a controlled **single-hop causal audit**:

> Do heads selected by the Relation Index (RI) definition of Ren et al. causally
> carry OLMo-2-7B's in-context relational ability, and does RI miss important
> components of a larger circuit?

Two-hop composition remains a later extension. A separate checkpoint-development
experiment is now completed and documented below. No sufficient causal circuit has
yet been established.

## Current status (26 September 2026)

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
- Main Stage-2 findings: several RI priority heads have small signed mean effects;
  some large effects occur outside RI selection. Associations depend on the RI
  definition and population. Scope-P/Scope-F differences motivate position
  profiles; copying and writer/reader roles remain hypotheses to test.
- Stage 3: `stage3_v1` completed (3 replicas, 6 GPUs; gates passed; 263,328 patched
  forwards; `summary.json` complete) and analyzed for the fixed 105-head inventory:
  Scope-F for all heads, single-position causal profiles on the 40 common pairs,
  reverse patching, attention/output profiles with changed-query, reorder and
  corruption controls, and all four planned diagnostics (weight copying, attended-name
  mover classification, matched contextual RI, attention/value factorial, synthetic
  fingerprints). Report: `חומר כתוב/Stage3_Results_and_Analysis.tex` (figures in
  `חומר כתוב/figs/`, table in `חומר כתוב/tables/`), corrected after an external
  review; analysis tables in `results/stage3_v1/analysis/`.
- Main Stage-3 findings: the strongest head L17H1 (5.7 logits) has no effect at the
  answer position; its effect enters inside the query-fact sentence (mother tokens,
  "is", period). Twenty strong heads act only at the final colon, mainly through their
  values; their attention follows the queried fact under a changed query. Six heads
  promote the first token of the name they attend to (L27H6 writes the answer directly,
  +3 logits), one suppresses it (L20H1); several strong effects are not explained by the
  head's own vocabulary projection (hypothesis for Stage-4 path tests). Neither raw nor
  contextual RI tracks the causal effects. Labels are descriptive; no circuit is claimed.
- Stage 3, optional Section 1.5: controlled representation readout of the L17H1 and
  L15H25 sites completed (`stage3_readout_v1`, Slurm job 918689; gate passed with zero
  identity-injection and self-patch drift; 1,920 records; analysis regenerated locally
  byte-for-byte). Code `pilot_v2/stage3_readout.py`, run book
  `pilot_v2/RUNBOOK_stage3_readout.md`, plan `results/stage3_readout_inputs_v1/`,
  results `results/stage3_readout_v1/` (analysis CSVs committed; raw records on
  OneDrive/cluster). Reported as Section 7.7 of the Stage-3 report. Finding: replacing
  L17H1 alone removes ~88% of the readable mother-token preference at the query-fact
  "is" (negative in 20/20 families); L15H25's child-last site is unresolved because the
  source-swap control itself exposes no mother preference. Independent check:
  `results/stage3_readout_review_20260922/Stage3_section1_5_analysis.md`.
- Stage 4.1: `stage4_all_v2` (Slurm 923239) completed and independently verified:
  70,408 records, including controls, local refinements and both discovery extensions.
  The map supports L17H1 fact-site V influence into L18H18/L18H19 and an upstream
  L15H25–L16H1/L16H21 motif, alongside opposing routes and unresolved sources.
  Separately measured edges do not establish a composed chain.
- Stage 4.2: `stage4_s42_all_v1` (Slurm 924204) completed and analyzed:
  51,904 records (44,754 new endpoints and 7,150 reused), all 49 numerical exports
  reproduced locally. Group tests repeatedly implicate L18H18/L18H19; clamping O3
  removes about 88–90% of L17H1's tested intervention effect. Residual RI participation
  is heterogeneous and baseline-sensitive; the tested L26H31 backup is unsupported.
  Fact order and donor-versus-mean replacement materially affect interpretation.
- Current combined report: [Stage 4 PDF](output/pdf/Stage4_Experiment_Report.pdf);
  [Overleaf sources](<חומר כתוב/Stage4_Overleaf/main.tex>). Sections 1–6 cover S4.1,
  7–10 S4.2 results, 11 hypotheses/limits, and 12 the exact prospective S4.3–S4.5 plan.
  Analysis/audits: `results/stage4_review_20260924/`,
  `results/stage4_s42_readiness_20260924/`, `results/stage4_s42_analysis_20260924/`.
- Stage 4.3: **completed** (Daniella Simonovsky, Google Colab, 25–26 September; integrated
  26 September). Bounded local expansion exactly as planned in
  `results/stage4_s42_analysis_20260924/next_stage_plan.json`: an initial noising screen of
  305 configurations on the 40 common pairs (`results/stage4_s43_all_v1/initial_v2/`, 12,200
  records), a frozen selection of 16 route contrasts plus 11 direct prerequisites, and a
  bidirectional extension to all 178 discovery pairs
  (`results/stage4_s43_all_v1/extension_v4_family034_final/`, 9,612 records). All 16 selected
  routes retain in both directions; 14 are coherent. Main results: the composed dependency
  L17H1 → {L18H18, L18H19} → L27H6 Q (+1.64 noising / +1.51 restoration, 98.9% family sign
  match), the L15H25 → {L16H1, L16H21} → layer-18 Q chains (+0.15 to +0.22), attenuation of
  the direct L8H15 → layer-18 V routes when MLP9 is released, a block-mediated L26H23 effect,
  and an unstable L20H7 downstream sensitivity. L13H18 and L16H31 produced no retained route.
  Frozen inputs: `results/stage4_s43_inputs_v1/`, `results/stage4_s43_extension_inputs_v1/`;
  code `pilot_v2/s43_engine.py`, `s43_run_v2.py`, `s43_extension_run_v3.py`, `test_s43.py`;
  run book `pilot_v2/RUNBOOK_stage4_s43.md`; reports
  `חומר כתוב/Stage4_3_Methodology_and_Results.pdf`, `Stage4_3_Summary.pdf`; independent
  re-analysis `results/stage4_s43_review_20260926/`. Held-out families were not accessed.
- Stage 4.4–4.5 remain **planned, not implemented or submitted**. S4.4 tests
  groups/mechanisms only; S4.4/S4.5 share cached measurements. Exact named sets, caps and
  conditional decisions are in `results/stage4_s42_analysis_20260924/next_stage_plan.json`.
  The 87 held-out families remain sealed.
- Developmental side experiment: completed ICL and RI checkpoint sweeps in
  [Daniella's test ICL vs SIH](<Daniella's test ICL vs SIH/README.md>). This asks whether
  behavioral ICL acquisition coincides with emerging SIH-like structure. The supplied
  analysis finds an early source-routing increase near the ICL transition, but no
  clear population-wide jump in conditional RI. This is temporal association,
  not a causal demonstration or an exact replication of the original SIH result.

The authoritative transition notes are in `RESEARCH_HANDOFF.md` (the top status section is the latest state). For the current method and evidence,
read the files in this order:

1. `חומר כתוב/Single_Hop_Methodology_stage1_updated.tex` (+ Section 4 and 5
   replacement fragments in the same folder, not yet merged)
2. `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`
3. `חומר כתוב/Stage2_Results_and_Analysis.tex`
4. `pilot_v2/RUNBOOK_stage2_v1.md`, `pilot_v2/RUNBOOK_stage2_v2_extension.md` (run books)
5. `חומר כתוב/Stage3_Head_Level_Characterization_revised.tex` (Stage-3 plan; the
   attended-name paragraph is in `Stage3_plan_attended_name_paragraph.tex`)
6. `חומר כתוב/Stage3_Results_and_Analysis.tex` (Stage-3 results and analysis,
   including the Section-7.7 readout; Overleaf copy `חומר כתוב/Stage3_Overleaf/main.tex`)
7. `pilot_v2/RUNBOOK_stage3_v1.md`, `pilot_v2/RUNBOOK_stage3_readout.md` (run books)
8. `חומר כתוב/Stage4_Overleaf/main.tex` and `s42_results.tex` (current S4.1/S4.2
   results and revised follow-up plan); `חומר כתוב/Stage4_Protocol.tex` is the
   historical protocol, preserved rather than retroactively rewritten.
9. `Daniella's test ICL vs SIH/README.md` (separate developmental experiment,
   data provenance, execution order, results and reproduction limits).

## Paper

The ACL-format paper lives in `חומר כתוב/paper/` (`main.tex`, `sections/`, `appendix/`,
`figs/`, `refs.bib`); Overleaf holds the compiled master copy, the folder is the Git copy and
transfer medium. `main.pdf` there is a local compile only.

## Documentation map

This file is the project entry point. Experiment-specific READMEs explain their
local data and evidence; `RUNBOOK_*` files contain operational instructions written
at the time of a run. Status in old run books does not supersede the current handoff.

| File | Role | Status |
|---|---|---|
| `README.md` (this file) | project entry point, status, reading order | current |
| `RESEARCH_HANDOFF.md` | tracked collaborator handoff, evidence hierarchy | current |
| `חומר כתוב/Stage4_Overleaf/` | combined S4.1/S4.2 report and exact follow-up design | current |
| `Daniella's test ICL vs SIH/README.md` | developmental side-experiment guide and repository policy | current |
| `חומר כתוב/Stage4_3_Methodology_and_Results.pdf` | S4.3 methodology and results (executed 25–26 Sept.) | current |
| `pilot_v2/RUNBOOK_stage4_s43.md` | run book: S4.3 (executed on Colab; how to re-run on the cluster, what is missing) | current |
| `pilot_v2/RUNBOOK_stage3_v1.md` | run book: `stage3_v1` (completed) — preparation, gate/run, analysis, transfer | current |
| `pilot_v2/RUNBOOK_stage3_readout.md` | run book: Section-1.5 readout (`stage3_readout_v1`, job 918689, completed) — prepare, gate/run, analysis, transfer | current |
| `pilot_v2/RUNBOOK_stage2_v1.md` | run book: `stage2_v1` (job 888691) | current |
| `pilot_v2/RUNBOOK_stage2_v2_extension.md` | run book: `stage2_v2_extension` (job 912879) | current |
| `pilot_v2/RUNBOOK_stage1_audit.md` | run book: Stage-1 audit and calibration | current |
| `pilot_v2/RUNBOOK_ri_test_v2.md` | run book: Stage-1 test-only RI extension (`ri_test_v2`, job 905839) | current |
| `pilot_v2/RUNBOOK_ri_test_v2_retry.md` | notes on the failed attempts (jobs 899730, 905799, 905807) before the completed run | current |

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
Analysis: `analyze_stage2_results.py` (v1), `analyze_stage2_v2_results.py` (v2),
`pilot_v2/stage3_analyze.py` (Stage-3 integrity audit, summaries and the attended-name
classification; `--attended-only` re-runs only the latter), `pilot_v2/stage3_report_figures.py`
(report figures and numbers), `pilot_v2/stage3_readout.py analyze` (Section-1.5).

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

Stage 3 characterized the heads individually; S4.1/S4.2 have now measured routes,
joint interventions and conditional contributions. Retained-mechanism faithfulness,
completeness, fact-by-query behavior and frozen held-out validation are still future
work. The current prospective criteria and stopping rules are in Section 12 of the
combined Stage-4 report; passing a single recovery score would not establish a
complete or uniquely minimal circuit.

## Repository map

- `pilot_v2/` — OLMo/Pythia runners, Stage 0–4.3 code (incl. the Section-1.5 readout),
  Slurm wrappers, tests, model locks, and operational documentation.
- `pilot_v3/` — later behavioral generators, tokenizer audits, and the current
  single-hop generator/data.
- `results/` — versioned plans, run metadata, numerical summaries and audits;
  local copies additionally contain raw outputs excluded from Git.
- `Daniella's test ICL vs SIH/` — separate developmental checkpoint experiment.
- `output/pdf/Stage4_Experiment_Report.pdf` — current compiled Stage-4 report.
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

## What is versioned and how to reproduce

Git contains the code, tests, frozen plan/selection metadata, current report sources
and figures, compact measured summaries, and the collaborator handoff. Stage-4
raw worker chunks, mean-bank tensors, event dumps, copied runtime packages, archive
bundles, temporary logs and duplicate analysis reproductions stay on OneDrive/cluster.
Archive identities and package hashes are retained so restored data can be checked.
The Overleaf ZIP is regenerable from its tracked directory and is not committed.

Reading the reports and saved summaries needs no GPU. Re-running a raw-data audit
requires restoring the named run archive and matching executed package to the paths
in `RESEARCH_HANDOFF.md`; Git alone does not contain those raw measurements or model
weights. The S4.2 review and audit code checks identities, coverage and reused records
before interpretation. S4.3 follows the same rule: frozen inputs, run manifests, gates,
per-chunk `.ok` hashes, compact analysis tables and the two reports are versioned; the raw
chunk files and the large per-pair tables stay on OneDrive (see
`pilot_v2/RUNBOOK_stage4_s43.md`). S4.4–S4.5 runs are not part of this repository.

The developmental folder follows the same separation: generated input streams,
clean notebook sources, small manifests, current summary tables/figures and two
reports are versioned. Raw ICL predictions, RI chunks and large all-head tables remain
external. Its README gives exact exclusions, checksums and preprocessing caveats.
