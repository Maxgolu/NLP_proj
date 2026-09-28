# NLP Final Project — A Causal Audit of Semantic Induction Heads

This project asks whether heads selected by the relation index (RI) participate
in OLMo-2's contextual relation retrieval, how they participate, and whether RI
identifies the causally important components. The mature-model audit uses synthetic
single-hop **mother-of** prompts. A separate training-checkpoint study compares RI,
attention routing and in-context learning.

## Status — 28 September 2026

**All planned experiments are complete**, including S4.3, the revised final S4.4
experiment (named **S45** in the implementation), and its frozen evaluation on
87 held-out families. No further GPU experiment is pending. Final manuscript
review is in progress; no sufficient retained head circuit was established.

- Base model: `allenai/OLMo-2-1124-7B`, revision
  `7df9a82518afdecae4e8c026b27adccc8c1f0032`; 32 layers × 32 heads, no fine-tuning.
- Data: 200 generated families; 176 behaviorally eligible; 89 discovery families
  (178 clean/corrupted pairs), including the 20-family common panel; 87 disjoint
  validation families (174 pairs). Each family contributes two fact orders.
- Final validation: completed run `945595`; the received mean-baseline family
  export passes all 226 independent consistency checks. Held-out families are
  **no longer sealed**. Selection was frozen before their evaluation.
- Two-hop composition was not tested mechanistically and is outside the completed
  project; historical Phase-B plans are not outstanding work.

## Findings

1. **RI is a poor selector of causal importance.** The pooled rule selects two
   nearly inert heads. Of 61 pooled/test-only candidates, 48 have absolute
   standalone effects below 0.1 logits; only eight of the 25 strong heads in the
   full-coverage set are RI-selected. No eligible head passes matched-target Holm
   calibration. Descriptive selection and statistical significance are distinct.
2. **Some RI heads do participate in measured paths.** Fact-site writers influence
   answer-position readers; selective intermediate releases establish composed
   paths containing RI heads. The weaker RI head L8H15 also has direct value-channel
   routes. Group effects and receiver blocking reveal conditional contributions.
3. **Weaker RI heads have collective and background-dependent effects.** S4.2's RI23
   excludes the eight strong RI heads and has larger absolute group effects than
   layer-matched reference cohorts. This does not imply that every member is useful
   or that the group is a sufficient circuit.
4. **Two additional candidates generalize behaviorally, with baseline dependence.**
   In the partial C50 role-mean background, held-out four-cell gains for L9H16 and
   L1H27 are 0.585 and 0.261 logits, positive in all 87 family means. Their
   original fact-axis effects nearly vanish under paired-donor replacement.
   No candidate–route modulation passed the frozen rule; new attachment tests were
   consequently not triggered. C50 retains only 12.8% of the full fact-swap gap
   under means (17.6% under donors), so it is not a sufficient retained mechanism.
5. **Training-time association does not establish a persistent head mechanism.**
   Early source-routing changes accompany the behavioral transition, but fixed-support
   conditional RI has no clear corresponding jump and early high-RI head identities
   scarcely overlap the final set. This is observational evidence.

## Paper and current source of truth

Overleaf remains the compilation master. The latest supplied author snapshot is
[`חומר כתוב/paper/paper3/`](<חומר כתוב/paper/paper3/>): `main.md` contains the body and
appendices, supplemented by the newly supplied `.tex` files and corrected `refs.bib`.
The older modular files directly under `paper/` are **not fully synchronized** with
that snapshot. Do not overwrite Overleaf with those older files.

The latest snapshot is a review export, not a self-contained Overleaf bundle:
some files have download suffixes such as `main (1).tex`, figures and separate
appendix sources were not re-exported, and its `\input` paths assume the original
Overleaf directory layout. Export the complete final Overleaf project to preserve
the exact submission version.

The final review is [here](results/paper_final_review_20260928/REVIEW_HE.md).
It reports proposed corrections without changing manuscript content. The supplied
abstract has been reviewed. The user confirms developmental results exist in
Overleaf, but neither supplied results file includes that subsection; its actual
text still needs reconciliation before a complete content sign-off.

## Evidence map and reading order

| Component | Narrative source | Data / code |
|---|---|---|
| Current status, interpretation, workflow | [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) | Final evidence hierarchy and provenance |
| Task and split | `results/singlehop_baseline/singlehop_baseline_report.md` | `pilot_v3/build_data_singlehop.py`, `pilot_v3/data_singlehop/` |
| Stage 1: RI audit | `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`; `Stage1_RI_Test_Extension_Results_and_Analysis.md` | `results/stage1_v4_review/`, `results/ri_test_v2/`, `results/ri_test_v2_analysis/` |
| Stage 2: causal map | `חומר כתוב/Stage2_Results_and_Analysis.tex` | `results/stage2_v1_analysis/`, `results/stage2_v2_analysis/`; `pilot_v2/stage2_*`, `pilot_v2/analyze_stage2*_results.py` |
| Stage 3: head profiles/readout | `חומר כתוב/Stage3_Results_and_Analysis.tex` | `results/stage3_v1/analysis/`, `results/stage3_readout_v1/`, associated review folders |
| S4.1 and S4.2 | `חומר כתוב/Stage4_Overleaf/main.tex`, `s42_results.tex`; `חומר כתוב/Stage4_Experiment_Report.pdf` | `results/stage4_all_v2/`, `results/stage4_s42_all_v1/`, `results/stage4_s42_analysis_20260924/` |
| S4.3: composed paths/local expansion | `חומר כתוב/Stage4_3_Methodology_and_Results.pdf` | `results/stage4_s43_all_v1/`, `results/stage4_s43_review_20260926/`; `pilot_v2/s43_*` |
| Final S4.4/S45: implemented design | `חומר כתוב/RI_Structure_Final_Protocol_20260926.md` | `pilot_v2/s45/`, `pilot_v2/s45/RUNBOOK_stage4_s45.md` |
| Final discovery | `results/stage4_s45_discovery_v2/S45_discovery_analysis.md` **together with** `results/stage4_s45_review_20260927/REVIEW_HE.md` | Frozen inputs, summaries and selection under `results/stage4_s45_*` |
| Final held-out | `results/stage4_s45_heldout_received_20260928/REVIEW_HE.md` | `results/heldout_plan.json`; validation CSVs, gate, completion manifest and `audit_heldout.py` in that folder |
| Developmental study | [Experiment README](<Daniella's test ICL vs SIH/README.md>) and its full report | Five notebooks, manifests, frozen inputs and compact tables in the same folder |

The superseded `Final_S44_S45_Proposal_20260926` and the old greedy-reduction plan
are not the executed final experiment. Old runbooks describe their historical
execution steps; their future-tense statements do not supersede this status.
See also [the results directory map](results/README.md).

## Reproduction and version-control policy

The repository versions code, frozen plans and selections, manifests, compact
measurements, CPU analyses, manuscript sources and figures. Model weights, large
raw worker chunks, event arrays, mean banks, temporary renders, compiled duplicate
PDFs and transfer archives stay on research storage / OneDrive, as specified in
`.gitignore` and the runbooks. ZIP bundles are transfer conveniences, not evidence
that a particular Overleaf version is current.

Reports and compact tables require no model rerun. Reproducing raw-data analyses
requires restoring the documented excluded files. In particular, the local
89-family final-discovery export supplies extension summaries but not all family
vectors; the held-out export supplies 522 complete mean-baseline family records,
but donor results are summary-only. Do not describe donor confidence intervals or
the full 89-family extension as independently reconstructed from absent vectors.

Run the received held-out audit from the project root with a Python environment
containing NumPy:

```powershell
python results/stage4_s45_heldout_received_20260928/audit_heldout.py
```

This recomputes the CPU audit JSONs; it does not run OLMo. GPU reproduction uses the
stage-specific runbooks, pinned model, input identities and gates.

Selective final upload commands and the checked remote state are documented in
[GIT_UPLOAD.md](results/paper_final_review_20260928/GIT_UPLOAD.md). No commit or push
was performed as part of the final content review.
