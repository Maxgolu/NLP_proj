# Research handoff — current state, 28 September 2026

**All experiments are complete. The project is in final manuscript review.**
This file replaces obsolete top-level statements that S4.3–S4.5 were unexecuted
or that the 87 validation families were still sealed. The complete prior handoff
is preserved at `results/paper_final_review_20260928/handoff_before_update.md` for
historical details; its pending-work instructions are not current.

## 1. Active request and manuscript authority

The user requested a final content audit, current README/handoff, and instructions
for a selective Git upload. **Do not insert proposed corrections into the paper**
until asked. No further experiment, rerun, submission, commit or push is authorized
by this review request.

Overleaf is the compilation master. Current review inputs:
- `חומר כתוב/paper/paper3/main.md`: partner's body-plus-appendices export.
- The added `paper3/*.tex` files, including `00_abstract.tex`,
  `01_introduction.tex`, `02b_related.tex`, `03_setting.tex`, `04_method.tex`,
  `05_results (1).tex`, `06_conclusions.tex`, `07_limitations.tex`,
  `08_ai_disclosure.tex`, `main (1).tex`, plus updated `refs.bib` and style files.
- Separate updated appendix sources and figures were not supplied again; use the
  appendix text in `main.md` for this review, not an assumption of a newer PDF.

The local modular files directly under `paper/` are older than this snapshot and
must not overwrite Overleaf. The new bibliography resolves old TODOs and includes
the AGENDA citation; do not report the old bibliography's defects as current ones.
`paper3` is not a ready-to-upload standalone project: it is a flattened partial
export with suffixed download filenames. A complete final Overleaf export remains
the way to synchronize exact submission sources.

The user confirms Abstract and developmental results exist in Overleaf. The added
abstract was read. However, `05_results (1).tex` still ends at 5.4, `main.md` has
no developmental Results subsection, and `main (1).tex` has no separate input for
one. **Reconcile the actual developmental Results text; do not assert that it is
absent from Overleaf or that it has been reviewed.**

Final review: `results/paper_final_review_20260928/REVIEW_HE.md`; source snapshot
hashes and numerical checks in `verification.json`. Manuscript content was not edited.

## 2. Research question and scope

Distinguish RI's ability to identify causally important heads from participation of
particular RI-selected heads in contextual retrieval. Ask both whether such heads
participate and what they contribute to measured groups/routes. Task-specific
effects are not a sufficient isolated semantic circuit or proof of relation
encoding in every individual head.

Mature model: base OLMo-2-1124-7B, revision
`7df9a82518afdecae4e8c026b27adccc8c1f0032`, 32 × 32 heads, fp16, no fine-tuning.
Synthetic single-hop mother-of task: 200 families, 176 behaviorally eligible,
89 discovery (178 pairs), including 20 core families (40 pairs), and 87 disjoint
held-out (174 pairs). Two fact orders per family. The family, not each prompt,
is the statistical unit. Two-hop composition was not part of the final audit.

## 3. Completed evidence and reading order

### Stages 1–3

- Stage 1: `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`,
  `Stage1_RI_Test_Extension_Results_and_Analysis.md`; `results/ri_test_v2/`.
  Pooled selection: L3H11/L9H22, local attention and >99% demonstration score.
  Test-only selection: 59 heads; pooled union: 61. No eligible head passes Holm.
  Selection is descriptive, not significance. Raw colon projection is invariant
  under mother reassignment when the visible token set is unchanged.
- Stage 2: `חומר כתוב/Stage2_Results_and_Analysis.tex`,
  `results/stage2_v2_analysis/head_table_v2.csv` and `summary.json`.
  All-head exact patching covers 40 pairs; full 178-pair exact coverage is 148
  heads. There are 25 strong heads in that set, 17 positive / 8 negative; 8 are
  RI-selected. 48 of 61 RI heads have |I|<0.1. Exact-screen signed Spearman 0.945;
  pooled-RI correlation +0.135; supported name-gap correlation -0.252, -0.302
  within layer. These are signed-importance correlations. L17H1 +5.701, L18H19
  +3.072, L27H6 +3.071, L18H18 +2.704. Mean full discovery gap 11.170.
- Stage 3: `חומר כתוב/Stage3_Results_and_Analysis.tex`,
  `results/stage3_v1/analysis/`, `results/stage3_readout_v1/`, corresponding reviews.
  105-head inventory; 20/25 strong heads are colon-localized. L17H1 instead writes
  at query-fact mother/syntax positions; identity readout at “is” changes -0.458
  under head replacement versus -0.521 under source swap (20/20 negative).
  L15H25's child-last content is unresolved. Query-following attention and
  value-dominant means motivate routes, while family A/V interactions are large.
  Matched contextual RI still has negative name-gap correlation (-0.259 to -0.170,
  68 supported heads). Copying does not determine causal role.

### S4.1–S4.3

- Combined S4.1/S4.2 report: `חומר כתוב/Stage4_Overleaf/main.tex` and
  `s42_results.tex`; PDF `חומר כתוב/Stage4_Experiment_Report.pdf`.
  Source folders: `results/stage4_all_v2/`, `stage4_review_20260924/`,
  `stage4_s42_all_v1/`, `stage4_s42_analysis_20260924/`.
- S4.1 job 923239, 70,408 records. Of 96 extended configurations, 95/93 retain
  under noising/restoration. L17H1 writer-union → L18H18/L18H19 V:
  +3.400/+2.074. L15H25 child-last → L16H1/L16H21 V: +0.182/+0.210.
- S4.2 job 924204, 51,904 records, 49 numerical exports reproduced.
  Layer-18 pair joint effect 5.630; nonadditivity and group-minus-member effects
  matter. O3 blocking removes 89.4% of L17H1 writer-union restoration, not an
  exclusive additive mediation share. RI31 contains eight strong and 23 residual
  RI heads. **RI23 means RI31 minus the strong eight, not the 23 rank-selected heads.**
  RI23 donor effects: signed +0.116 / absolute 0.594; reference absolute 0.215.
  Mean replacement reverses the signed effect (-0.633). Individual weakened-reader
  tests do not uncover a broad population of hidden backups. L26H31 fails its
  backup rule in all five tested backgrounds.
- S4.3 by Daniella, completed 25–26 September. Report:
  `חומר כתוב/Stage4_3_Methodology_and_Results.pdf`; raw/analysis:
  `results/stage4_s43_all_v1/`; audit `results/stage4_s43_review_20260926/`.
  Initial 305 configurations; 16 selected contrasts + 11 prerequisites extended
  to all 178 pairs in both directions. All 16 retain; 14 coherent.
  L17H1 → {L18H18,L18H19} → L27H6 Q: +1.635/+1.511.
  L15H25 child-last → {L16H1,L16H21} → L18H18 Q: +0.172/+0.179.
  The no-intermediate fact-to-colon-Q comparator is structurally zero.
  Direct L8H15 sentence → L18H18/L18H19 V: +0.465/+0.460; MLP9 release
  attenuates the positive routes to +0.321/+0.313. Negative increments are not
  negative raw routes. L26H23's block27 test includes L27H6 but does not isolate
  it. L20H7 is heterogeneous, not a stable opposing route.

### Final revised S4.4, named S45 in code

Executed protocol: `חומר כתוב/RI_Structure_Final_Protocol_20260926.md`;
code/runbook: `pilot_v2/s45/`. It replaces both the earlier
`Final_S44_S45_Proposal_20260926` and the old greedy-reduction S4.4/S4.5 design.

Structures: T1 L17H1 → layer-18 pair → L27H6 Q; T2 L15H25 child-last →
layer-16 pair → L18H18 Q (node set also contains L18H19); T3 nested T1 with only
L18H18 released; T4 direct L8H15 → L18H18 V; T5 opposing L20H1 colon → L27H6 Q.
C33: 25 strong + two coverage additions + six moderate/conditional RI heads.
C50 adds the other 17 RI31 members: 31 RI + 19 non-RI. Six additional candidates:
L1H27, L11H4, L17H5, L23H10, L25H18, L9H16; the last is outside C50 and yields a
51-head toggle. There are 32 distinct RI heads in this experiment, not six total.

Four cells cross fact swap and changed query, with answers (a,b,b,a).
B = mean[(M00-M10-M01+M11)/4], orders averaged within family.
Delta B is behavioral contribution; Gamma is change in an isolated route effect.
Mean-replaced excluded heads are replaced at all test-block positions; embeddings,
MLPs, norms, demonstrations and answer-prefix computation remain live.

Discovery job 937878: `results/stage4_s45_discovery_v2/`. Read its analysis with
`results/stage4_s45_review_20260927/REVIEW_HE.md`, which corrects overinterpretation.
Independent audit: 1,035 checks pass. Core C33/C50 fidelity 5.2%/11.5%, C50
candidate accuracy 50%, first-chain preference 85%; fallback C50 is partial.
Delta B L9H16 +0.601, L1H27 +0.260, all 20 families positive. T1/T3/T4 remain
retained; no candidate–route pair passes modulation. **Stage D was not triggered**.
Post hoc T1−T3: added L18H19 release +0.225 in C50 / +0.805 in full; discovery-only.
89-family original-axis candidate effects almost vanish under donor replacement.
The extension is summary-only locally, so its paired intervals were not reconstructed.

Held-out job 945595: completed, received and audited 28 September. Inputs:
`results/heldout_plan.json`, archive `results/s45_heldout_v1_analysis.tar.gz`.
Extracted evidence and CPU audit: `results/stage4_s45_heldout_received_20260928/`.
226 checks pass, rerun during final review. 87 families disjoint from discovery;
freeze/plan/code identities match. Six mean configurations × four cells and four
donor configurations × original fact axis; no scientific route/attachment validation.

| Quantity | L9H16 | L1H27 | RI31 jointly |
|---|---:|---:|---:|
| Held-out Delta B, means | 0.584510 | 0.261250 | 0.727472 |
| Positive family means | 87/87 | 87/87 | 87/87 |
| Fact-gap Delta g, means | 1.170212 | 0.532168 | 1.403844 |
| Fact-gap Delta g, donors | -0.002318 | 0.029242 | 4.427494 |

C50 fidelity is 12.8% means / 17.6% donors, versus 98.7% full candidate accuracy
and 51.0% C50 accuracy under means. L9H16 raises accuracy by 2.3 percentage points.
Only mean family vectors are included (522 records); donor figures come from the
supplied summary. Automatic “functional participant” labels ignore donor sensitivity.
Report replicated functional effects **and** replacement dependence. Do not infer
that the candidates are outside the routes, parallel to them, redundant, or validated
new circuit members. RI31 includes strong heads; its validation is not RI23 validation.

### Developmental study

Start at `Daniella's test ICL vs SIH/README.md`, then
`Documentataion/OLMo_ICL_RI_side_experiment_complete.pdf` and `results/` there.
17 behavioral checkpoints, 10 RI checkpoints, four synthetic classification tasks,
1,600 assessments/checkpoint, 700 AGENDA triplets across seven relations.
Transition begins between steps 700–850; all four 20-shot confidence intervals
are above chance by 900. Routing peaks near 850; fixed-support conditional RI
shows no clear corresponding jump. Early/final top-head Jaccard is 0–0.00493.
Dense behavioral follow-up at 700/850/900 followed the observed RI routing peak.
The conditional RI population curve is **not monotonic** (600→850 falls); the
current appendix caption needs correction. No developmental GPU rerun was done.

## 4. Writing and implementation rules

- Concise Hebrew discussion, English paper; prioritize central results, roles and
  implications. Exact rosters, thresholds and extended evidence belong in appendices.
- Do not repeatedly insert generic caveats. Correct false/generalized claims,
  distinguish genuinely different interventions, and keep scope readable.
- Main 5.4 budget: about 400 words and one figure; conclusions ≤200 words, no
  future-work paragraph (user decision). Preserve equations/figures when they save words.
- Paper sections stay modular. Name precisely which Overleaf paths would change.
  Latest user/partner exports override older local text. No paper edits in this review.
- Course requirements: ACL-style paper, up to eight body pages excluding references
  and appendices; AI Disclosure and Reflection required. Course PDF is one directory
  above this repository. Content review is not a compiled-layout compliance check.
- Earlier user-approved local format edits: Times 10.5 pt, baseline 13.2 pt, side
  margins 1.9 cm; pages after the first extend one line upward and downward. Preserve
  the newly supplied style rather than reapplying an old patch blindly.
- Local Windows TeX compilation was blocked by Application Control. No successful
  local compilation of the current submission has been claimed; Overleaf renders it.
- Use CPU evidence checks when needed; do not launch GPU jobs or alter frozen plans.

## 5. Git and reproducibility status

At this review, local HEAD and live GitHub main both equal
`ff627501b7265c66aeb16a4dbc122d588d4d5c1a` (27 September). No tracked experiment-engine
changes exist under `pilot_v2/` or `pilot_v3/` relative to that commit. There are
uncommitted manuscript/style/document changes, new held-out evidence and CPU audit
code, figures and the partner snapshot. These do require an upload.

Selective commands: `results/paper_final_review_20260928/GIT_UPLOAD.md`.
No staging, commit or push was performed. Exclude old scratch protocol copies,
QA render PNGs, `Claude outputs/main_preview.pdf`, duplicate report PDFs and archives.
Raw data/model/mean-bank exclusions remain as documented in `.gitignore`.
`results/README.md` and the root README now include final validation.
