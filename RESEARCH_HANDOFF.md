# Research handoff — authoritative current state

Updated 25 September 2026. **The project is now in the paper-writing phase.** Read the
section "PAPER WRITING" below first; it supersedes every "Where we are NOW / Immediate
next action" statement further down (those describe the experimental phase and are kept
as provenance). The old chronology is preserved in
`research_history/RESEARCH_HANDOFF_before_stage4_consolidation_20260923.md`.

## PAPER WRITING — status on 25 September 2026 (READ THIS FIRST)

### 0. One-paragraph status

All experiments that will appear in the paper are finished except Stage 4.3–4.5, which are
planned in detail but not executed (see Section "Latest S4.2 status" below; the plan is
`חומר כתוב/Stage4_Overleaf/s42_results.tex`, last section, and
`results/stage4_s42_analysis_20260924/next_stage_plan.json`). Daniella's developmental side
experiment is complete (Section 1 below). Paper writing started on 24 September. Sections 2
(Background and Related Work), 3 (Experimental Setting) and 4 (Method, five subsections) and
appendices A–I are written, reviewed line by line by the user and approved. **The next task
is Section 5, Results, starting with 5.1** (the observational relation-index findings and
their relation to causal importance, with the paper's first results figure). Deadline:
30 September 2026. Do not run, resubmit or re-analyze experiments unless the user asks;
the job now is writing, and every number in the paper must be taken from the existing
reports/CSVs listed in Section 5 below.

### 0b. Final experiment S4.5 — implemented 26 September, NOT executed

The last experiment is specified in `חומר כתוב/RI_Structure_Final_Protocol_20260926.md`
("RI participation in measured structures", supersedes `Final_S44_S45_Proposal_20260926.md`;
the old S4.4/S4.5 greedy-reduction design is dropped). Implementation lives in `pilot_v2/s45/`
(self-contained package: `s45_plan.py`, `s45_means.py`, `s45_engine.py`, `s45_run.py`,
`s45_analyze.py`, `s45_freeze.py`, `s45_pipeline.py`, Slurm wrappers, `test_s45.py`,
`test_s45_pipeline.py`, byte-identical copies of the frozen S4.1–S4.3 dependencies) and the
Hebrew runbook `pilot_v2/s45/RUNBOOK_stage4_s45.md` (prepare → discovery submission →
freeze → prepare-heldout → held-out submission). Design decisions accepted by the user:
one Slurm pipeline with frozen decision files, split into a discovery submission (Stages
A–D + 89-family behaviour + freeze manifest) and a separate held-out submission; the "empty
head mask" is all 1,024 heads mean-replaced at test-block positions; x01/x11 question-name
roles fall back by bucket to the original-query bank; 6 GPUs / 3 replicas. The user's four
precision notes are implemented and tested: (a) changed-query cells rebuild every derived
field (query_source/question_entity, token_ids, offsets, masks, prefix metadata, cell
identity); (b) the mean bank has `lofo` (discovery) and `full` (held-out) modes and refuses
the wrong population; (c) recipient/donor/hybrid/endpoint are captured inside the same
background object, freezing never cancels source replacement, and a control receiver must
be live in both compared contexts (loader refuses otherwise; a clamped receiver is a
documented silent null); (d) the held-out opens only with a signed freeze manifest listing
completed stages, identities, code/data versions, validation spec and primary signs.
Tiny-model tests pass on CPU (15 engine/plan/means/freeze gates in `test_s45.py`, 8 analysis-formula
tests with known answers in `test_s45_analysis.py`, an end-to-end Stage D/freeze/validation exercise in
`test_s45_pipeline.py`, and a full discovery-pipeline dry run on a random 32×32-head model). `prepare`
was also run once locally on the real discovery data with the pinned tokenizer: every historical
reproduction gate passed and the 87 held-out families are identified from the behavioural baseline. Nothing has run
on OLMo-2-7B yet; the runbook says how. Known deviation from the spec: intermediate captures
are not stored to disk (recomputed deterministically, shared per pair/state in process).
Realistic prior: low chance of a positive result; a null is a legitimate conclusion.

### 1. Daniella's developmental folder (received 24 September)

Path: `project/Daniella's test ICL vs SIH/`. Author: Daniella Simonovsky (project partner).
Entry point: its `README.md`. Contents:
- `Documentataion/` (sic): `OLMo_ICL_RI_side_experiment_complete.pdf` (full report),
  `OLMo_ICL_RI_side_experiment_methodology.pdf`, `OLMo_ICL_RI_summary.pdf` (two pages).
  Read the summary first; the complete report is the primary source for numbers.
- `code/`: five notebooks — `data_generation.ipynb`, `olmo_icl_checkpoint_sweep.ipynb`,
  `olmo_icl_results_analysis.ipynb`, `olmo_ri_developmental_sweep.ipynb`,
  `olmo_ri_results_analysis.ipynb` (outputs cleared for Git; originals local).
- `data/`: `icl_stream.jsonl`, `ri_agenda_stream.jsonl`, `ri_agenda_olmo_safe.jsonl`
  (the frozen inputs); `results/icl/`, `results/ri/` (tables, figures, per-checkpoint
  summaries); `repository_manifest.json` (hashes of inputs and measurements).

What it is: a replication, on the public OLMo-2-1124-7B intermediate checkpoints, of the
developmental analysis of Ren et al. (2024): does the relation index (RI) rise together with
in-context learning ability during pretraining? 17 behavioural checkpoints (4 synthetic ICL
tasks, constrained accuracy, 1,600 fixed assessments per checkpoint) and 10 RI checkpoints
(steps 150, 600, 700, 850, 900, 1000, 2000, 3000, 9000 and the final 928646; 700 balanced
AGENDA-style forward-relation triplets; all 1,024 heads). Three RI indices are kept separate:
QK pass rate (routing), conditional RI given a pass, and per-opportunity RI. A fixed-final-head
backward trace with Jaccard overlap asks whether the heads that score high at the end are the
heads that scored high early.

Findings (as reported by Daniella; the assistant curated the folder for Git but did not
re-run the numbers): the ICL transition happens around steps 850–900 (≈4B tokens); the QK
routing index peaks early (≈step 850, +0.013); conditional RI shows no clear jump on the
fixed support; Jaccard overlap between final high-RI heads and early high-RI heads is
0–0.005, i.e. the final "semantic induction heads" are not the early ones. The dense
behavioural checkpoints 700/850/900 were added after the RI peak was observed, so the timing
comparison is partly post hoc. Preprocessing is not an exact replication of the paper's
pipeline (no spaCy filtering). This is temporal association only; it is deliberately kept
apart from the causal Stage-4 evidence.

How the paper uses it: §4.5 "Developmental check" (`sections/04_method.tex`, label
`sec:method-dev`) and Appendix I (`appendix/I_dev.tex`, label `app:dev`) are written and
approved; §2.1 ends with one sentence pointing to `sec:method-dev` and `sec:results-dev`.
The results subsection (planned 5.5 or 5.7, label `sec:results-dev`, working title
"Emergence is not identity") is still to be written from the complete PDF.

### 2. Stage 4 status relevant to the paper

S4.1 (route mapping; Slurm 923239, `results/stage4_all_v2`) and S4.2 (groups, receiver
blocking, RI31 audit, backup test; Slurm 924204, `results/stage4_s42_all_v1`) are executed,
verified and reported in `חומר כתוב/Stage4_Overleaf/main.tex` (S4.1) and `s42_results.tex`
(S4.2 + revised S4.3–S4.5 plan). Compiled: `output/pdf/Stage4_Experiment_Report.pdf`.
S4.3 (bounded local expansion, 249 configurations), S4.4 (fact×query panel) and S4.5
(retained mechanism C, blocks W/D/P/A/N/R6/T/U, ≤2 passes) are planned only. The paper's
§4.4 and Appendix H were written by the rules of the plan, with `TODO` markers where the
executed rosters/values of S4.3–S4.5 would go. If those stages are never run before the
deadline, the Results and Limitations must say so plainly and the TODOs in Appendix H must
be replaced by "not executed" statements — do not leave TODOs in the submitted PDF.

### 3. The paper: where it lives and how the Overleaf workflow operates

Location on the user's computer: `project/חומר כתוב/paper/` (the assistant keeps a working
copy at `/home/claude/paper/` in its sandbox and compiles there with `pdflatex` + `bibtex`;
pdflatex IS available in the cloud sandbox, contrary to the older note below). The same
files are uploaded by the user, by hand, into an Overleaf project with the identical folder
layout. **Overleaf is the master copy for compilation; the local folder is the transfer
medium and the Git copy.** The user sometimes edits text directly in Overleaf (he edited the
last sentence of `sections/02_background.tex` himself); before editing any file, re-stage it
from the user's folder or ask whether Overleaf has newer text.

File layout (all paths relative to `paper/`):
- `main.tex` — ACL 2023+ template (`\usepackage[final]{acl}`, times, T1, utf8, `\IfFileExists`
  guards for microtype/inconsolata, graphicx, booktabs, amsmath+amssymb, tikz with libraries
  `positioning,arrows.meta,decorations.pathreplacing,fit`, `\graphicspath{{figs/}}`). Title
  "SIH Behind the Stage: A Causal Audit of Semantic Induction Heads"; authors Maxim Golubkov
  and Daniella Simonovsky, Tel Aviv University (emails still TODO). Abstract and Introduction
  are placeholders (written last). Sections are pulled in with `\input{sections/...}`; the
  Results/Limitations/AI-Disclosure placeholders live inline in `main.tex`, then
  `\input{papers_used}`, `\bibliography{refs}`, `\appendix`, `\input{appendix/...}`.
- `sections/02_background.tex` (§2 + 2.1 SIH: Eq. RI, TikZ Fig. 1 of the relation index, the
  three "what the original study does not establish" points), `sections/02b_related.tex`
  (2.2 Causal interventions on heads, 2.3 Reading what a head writes, 2.4 From heads to
  circuits, 2.5 The gap — three questions), `sections/03_setting.tex` (model, task, data,
  metric and scopes, developmental setting, reproducibility), `sections/04_method.tex`
  (4.1 Reproducing the SIH criterion; 4.2 Causal importance of every head with Eq. importance,
  Eq. screen, Table 1 coverage, Fig. 2 patching schematic; 4.3 Characterizing important heads,
  three questions, Table 2; 4.4 From heads to a mechanism, three questions, Fig. 3 route,
  Fig. 4 retained-mechanism grid, Eq. bd; 4.5 Developmental check).
- `appendix/A_ri.tex` (exact RI definition) … `I_dev.tex`; labels `app:ri, app:ih,
  app:compute, app:data, app:stage1, app:stage2, app:stage3, app:stage4, app:dev`.
- `refs.bib` (14 entries; several carry `note = {TODO-VERIFY ...}` — verify venues/pages
  against the ACL Anthology before submission and delete the notes), `papers_used.tex`
  (working list of every cited paper with bib key and link, placed before References;
  decide before submission whether to keep it), `acl.sty`, `acl_natbib.bst` (from the
  acl-org/acl-style-files repository), `figs/` (empty so far; all figures are TikZ or tables;
  any raster/plot figure must be added as a PDF/vector file under `figs/`).
- `main.pdf` — local compile output only. **Never upload `main.pdf` to Overleaf.**

Section labels in use: `sec:background, sec:sih, sec:rw-patching, sec:rw-readout,
sec:rw-circuits, sec:rw-gap, sec:setting, sec:method, sec:method-ri, sec:method-causal,
sec:method-char, sec:method-circuits, sec:method-dev`. Forward references already made from
§2/§4 to unwritten parts: `sec:results`, `sec:results-why`, `sec:results-dev`. These three
are currently defined by placeholder lines in `main.tex` (lines "Placeholder labels
referenced from Section 2"); when `sections/05_results.tex` is written it must define them
and the placeholder lines must be deleted, otherwise LaTeX reports duplicate labels.

Per-round workflow (this is what the user expects each time a section is written or edited):
1. Read the relevant primary sources (stage reports in `חומר כתוב/`, CSV/JSON under
   `results/`), draft the section in English, compile locally twice (`pdflatex; bibtex;
   pdflatex; pdflatex`) and check page count and overfull boxes.
2. Commit the changed files to the user's folder `חומר כתוב/paper/...` with
   `device_commit_files` (there is no device shell; use stage/commit). Update `main.tex` only
   when a new `\input` line or placeholder change is needed.
3. Tell the user, in Hebrew, exactly which files are **NEW** (he creates them in the matching
   Overleaf folder: `sections/`, `appendix/`, or the root) and which are **REPLACED**
   (he re-uploads the same path, overwriting). Always list `main.tex` if it changed. Remind
   him to recompile twice in Overleaf if new labels/citations were added (otherwise `??`).
4. Give a short Hebrew explanation of what the section says and why it is organized so.
   The user then reviews and returns numbered corrections; apply all of them, re-commit,
   and repeat the NEW/REPLACED list.
5. Keep `\input` modularity: one file per section/appendix; never paste a section's text
   into `main.tex`.

Current size: 18 pages compiled in total, of which the body (§2–§4) is ≈5.3 pages.
The ACL limit is 8 pages of body (references and appendices excluded). Budget agreed:
Results ≈2.2 pages, Introduction ≈0.6, Discussion+Limitations+Conclusion ≈0.6–0.8,
Abstract ≈0.15. If Section 5 overruns, shorten §4 (move detail to appendices) rather than
cutting results.

### 4. Rules the writing must follow

Sources of the rules:
- Course guidelines PDF: `C:/Users/User/OneDrive/Documents/computer science/4B/NLP/final proj/
  NLP_course_2025b___project_guidelines.pdf` (the folder `final proj` above `project/` is
  granted to the session; re-stage the PDF when a rule must be checked). Requirements used so
  far: ACL format, ≤8 pages body (references/appendix excluded), `\citet`/`\citep`, a mandatory
  section titled "AI Disclosure and Reflection", a Limitations section recommended, figures as
  vector/PDF. Grading: literature review 20, method 20, results 20, presentation 20,
  research question 10, ambition 10.
- Writing-style guide the user wants followed: Vered Shwartz, "Tips for writing NLP papers",
  https://medium.com/@vered1986/tips-for-writing-nlp-papers-9c729a2f9e1f (concise; every
  paragraph has one point; claims proportional to evidence; figures and tables carry the
  numbers; no chronology of the work; related work compared, not listed).
- Format model the user pointed to: the ACL 2025 paper "Position-aware Automatic Circuit
  Discovery" (Haklay et al.), in `final proj/papers/`.

Content and style rules established with the user (violating these produced most of his
corrections so far — reread before every draft):
- Method text explains **what was done and why**, in general rules; exact thresholds,
  rosters, head names, counts of configurations and job details go to the appendices. The
  body names no individual head (L17H1 etc.) except where a figure/table needs it; head
  names are allowed in appendices and in Results where a finding is about a specific head.
- No chronology ("first we did… then we…"), no work-log, no "stage" jargon as narrative;
  stages appear only as the four questions the method answers.
- Use equations, figures and tables instead of prose wherever they save words; every
  figure is TikZ or a vector PDF; every table uses `booktabs`.
- Claims must be no stronger than the evidence: "consistent with", "does not establish",
  "selected discovery result"; never "proves", never "first ever"; the layer-matched
  comparison is "sound", the Stage-4 findings are discovery-only (89 families), the 87
  held-out families are still sealed and must be described as such.
- Terminology fixed in the text: relation index (RI); QK gate with τ=2.2 (4.48 only as a
  sensitivity); Scope P / Scope F; importance I_h (noising) — J_h (denoising) appears only in
  Stage 4; "route", "receiver blocking", "retained mechanism C"; "constrained accuracy" for
  the developmental tasks; three developmental indices (pass rate, conditional RI,
  per-opportunity RI), never "the RI" alone in that context.
- Cite only papers that are in `final proj/papers/` or already in `papers_used.tex`; a new
  paper must be added to `refs.bib` and to `papers_used.tex` marked "(not in folder)" if the
  PDF is missing, and the user is told.
- Attribution-patching screen: Î_h = −⟨∇_{a_h} M(x_c), a_h^r − a_h^c⟩, cite Syed et al.
  (2023); Spearman 0.945 with exact patching.
- Communication with the user: concise Hebrew. **Never mix English terms, numbers or file
  names inside a Hebrew sentence** — put every English term, number, label or path in its own
  code span or code block, and keep the Hebrew sentence free of them. Paper text itself is
  English. No praise, no recap of what he already knows, no repeated questions.

### 5. Plan for the remaining sections and where their numbers come from

Section 5, Results (`sections/05_results.tex`, NEW; add `\input{sections/05_results}` to
`main.tex` and delete the placeholder block). Mirror the order of §4:
- 5.1 The criterion selects heads that are not causally important (label `sec:results-why`,
  referenced from §2.1). Sources: `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`,
  `Stage1_RI_Test_Extension_Results_and_Analysis.md`, `results/ri_test_v2/candidates.json`,
  `results/ri_test_v2_analysis/`; for the RI-vs-importance contrast
  `חומר כתוב/Stage2_Results_and_Analysis.tex` and `results/stage2_v2_analysis/`. Planned first
  results figure: RI rank/score of the 59 (61) selected heads against their Scope-P and
  Scope-F importance, showing that the strong heads are mostly outside the selection and the
  selected heads are mostly weak (numbers: 25 strong heads with |P|≥0.3; eight of them RI-
  selected: positives L16H1, L16H21, L17H24, L18H19, L22H5; negatives L19H16, L23H15,
  L26H23). Also the two pooled-index heads (L3H11, L9H22) and the calibration null (no head
  passes Holm) — those numbers are already in Appendix E tables, so the body cites them.
- 5.2 Causal map: coverage and screen calibration, the 25/21 strong heads, positives vs
  negatives, P vs F split (writer-like early heads vs colon heads). Source: Stage 2 report and
  `results/stage2_v2_analysis/`.
- 5.3 What the important heads do: positions (single-position scan), attention vs value,
  contextual projection, attended-name outcomes (name movers, negative movers), readout
  (L17H1 mother-token preference at `is`, L15H25 unresolved). Source:
  `חומר כתוב/Stage3_Results_and_Analysis.tex` §5–§7.7, `results/stage3_v1/analysis/*.csv`,
  `results/stage3_readout_v1/`, and the two review files under `results/stage3_*review_*/`.
- 5.4 Routes and groups (S4.1, S4.2): L17H1→L18H18/L18H19 V routes, L15H25→L16H1/L16H21,
  receiver blocking 89%, RI31 vs cohorts (larger absolute, sign-unstable), K={L21H18} leaves
  66% of the gap, L26H31 not a backup. Source: `Stage4_Overleaf/main.tex`, `s42_results.tex`,
  `results/stage4_review_20260924/report_facts.json`, `results/stage4_s42_analysis_20260924/`.
  If S4.3–S4.5 are still unexecuted, one paragraph says what the plan would test and that it
  is future work.
- 5.5 Emergence is not identity (label `sec:results-dev`): the developmental findings of
  Section 1 above, one figure from Daniella's results if a vector version exists (otherwise a
  small table).
Then: 6 Discussion (answer the three questions of §2.5; what the RI criterion does and does
not capture; what a causal audit adds), Limitations (single model, single relation, 89
discovery families, discovery-only Stage 4, sealed held-out not opened, developmental
association only, readout scope), Conclusion, then Introduction and Abstract (last), and the
"AI Disclosure and Reflection" section (the user will describe how the assistant was used:
experiment design partner, code, analysis, LaTeX drafting; every number checked by the
authors — write it with him, not for him).

Outstanding TODOs inside the paper (grep `TODO` in `paper/`): author emails; Appendix C GPU
type and job-id/wall-time table (known so far: RI test-only 905839; Stage 2 causal map 888691 1h15m, Stage 2
extension 912879 33m; Stage 3 characterization 916699 ≈3.7 h on 6 GPUs; Stage 3 readout
918689 33m; S4.1 923239; S4.2 924204; the GPU type and the Stage-1 scan job id must be
asked from the user);
Appendix D reorder line order to verify against `results/stage3_inputs_v1/`; Appendix H
executed rosters for S4.3–S4.5; Appendix I null-target controls wording; `refs.bib`
TODO-VERIFY entries.

### 6. Housekeeping still open

- `README.md` (project root) was updated on 24 September for S4.1/S4.2 but says nothing
  about the paper folder; add a short "Paper" paragraph when convenient.
- Git: `חומר כתוב/paper/` (sources only, not `main.pdf`, `.aux`, `.log`) has not been added
  to the repository yet; give the user explicit `git add` lines for it. Continue to avoid
  `git add .`.
- The project-space copy of this handoff (Claude project doc `RESEARCH_HANDOFF.md`) is synced
  from this file whenever it changes.

## Repository update and developmental side experiment

The user authorized updating README/handoff, publishing the current code and evidence,
and selecting appropriate content from `Daniella's test ICL vs SIH/` for the existing
GitHub repository. The handoff is tracked (it was already tracked before this update).
The root README now reflects completed S4.1/S4.2 rather than the stale planning status.
Current future-design authority remains report Section 12, not a new GPU implementation.

Side-experiment entry point: `Daniella's test ICL vs SIH/README.md`; evidence in
`Documentataion/OLMo_ICL_RI_side_experiment_complete.pdf` and its two-page summary.
According to the supplied completed analysis, 17 behavioral checkpoints (four tasks,
eight shot counts, 1,600 fixed assessments/checkpoint) and 10 RI checkpoints (700
balanced forward-relation triplets, 1,024 heads) show an ICL transition around steps
850–900 and an early QK/source-routing peak at 850, but no clear fixed-support jump
in conditional RI. This is temporal association, not causality; the dense behavioral
700/850/900 follow-up was motivated by observed RI timing. Preprocessing is not an
exact paper replication; the frozen stream lacks the paper's exact spaCy removal.
Do not mix this developmental evidence with Stage-4 causal interventions.

Publication choices: notebook source cells, generated input streams, compact current
tables/figures, manifests and the two distinct reports are included. Embedded notebook
outputs/counts were cleared without changing source cells; originals are preserved
locally under `tmp/git_publish_notebook_originals/`. Raw ICL predictions, RI chunks,
large all-head tables, outdated analysis and duplicate PDF/download copies stay local.
`repository_manifest.json` records input and external-measurement hashes. The RI input
hash matches the executed run config. This curation was not a fresh numerical
replication of Daniella's analysis. See its README for restoration requirements.

Stage-4 Git content similarly includes source, frozen JSON plans, completion metadata,
numerical summaries, report sources/figures/PDF and verification results. Raw chunks,
mean-bank tensors, copied runtime code, duplicate reproductions and archive bundles
remain external. To rerun audits, restore the matching run/package archives at the
documented paths. No new experimental run is implied by this repository update.

## Latest S4.2 status — full analysis and report completed

The user subsequently authorized full/deep S4.2 interpretation and the updated experiment
report, with exact prospective S4.3/S4.4/S4.5 head groups and dependencies. That work is
complete. No follow-up experimental implementation or GPU submission was requested or done.
The readiness-only status below is historical and superseded by this section.

Current combined report: `חומר כתוב/Stage4_Overleaf/main.tex` plus `s42_results.tex`;
portable sources/data: `חומר כתוב/Stage4_Experiment_Overleaf.zip`;
compiled PDF: `output/pdf/Stage4_Experiment_Report.pdf` (27 pages, 13 figures total,
seven newly added). Sections 7–11 contain S4.2 findings/interpretation; Section 12 contains
the revised follow-up plan. All pages were rendered and visually reviewed; package files,
references, assets and hashes were checked. Verification and reproducible CPU analysis:
`results/stage4_s42_analysis_20260924/` (`analyze_s42.py`, `build_assets.py`,
`update_report.py`, `package_report.py`, `report_verification.json`). The old report was
archived under `prior_report/`. GPU data and executed packages remain unchanged.

Important findings and limits:
- The L18H18/L18H19 pair is central across overlapping G1 groups, not several independent
  circuit discoveries. Additional donor-noising interactions for six groups on all 89
  families were computed on CPU from compatible saved whole-group and F-single terms.
- L17H1 union/sentence restoration blocking by O3 removes 89.4%/89.6% of the source effect
  on 89 families. L15H25 blocking by L16H1/L16H21 is similarly strong on the core panel.
  These ratios are not exclusive mediation shares or measured composed chains.
- Full-model gap itself depends on fact order (14.575 versus 7.765). Group differences
  persist after proper same-order normalization. Donor and mean interventions can have
  very different accuracies: O3 leaves 36.0% versus 97.8% clean candidate accuracy.
- K={L21H18} leaves 66.4% of the symmetric gap. Fourteen RI candidates were extended;
  six added candidates are L6H24,L8H15,L14H23,L15H3,L16H31,L20H7, with nonuniform evidence.
  Conditional participation usually is not a newly unmasked backup.
- Residual RI23 has larger absolute effects than four fixed reference cohorts, but signed
  effects flip across donor/mean replacement. References are non-null and not norm matched.
- L26H31 is below the backup rule in all five measured backgrounds; retire its dedicated
  backup follow-up. It remains eligible only as part of a predeclared collective RI addback.
- Discovery only. Core-only/unextended and mean-reverse limitations remain explicit.

Latest prospective design is report Section 12 and
`results/stage4_s42_analysis_20260924/next_stage_plan.json` (also in the Overleaf data).
This explicitly revises future choices in `Stage4_Protocol.tex`; do not modify the historical
executed protocol to disguise these changes.
- Adopt the user's request to remove S4.3's broad second expansion. One local round only:
  L13H18, L8H15, L16H31, L20H7, L26H23, plus the two named composed chains and slot controls.
  Exact sites/receivers/mediators are enumerated. Main panel 249 configurations, up to 56
  separate direct comparators before reuse (305 scientific configurations total); source,
  self and temporal gates extra. At most 16 retained route contrasts extend. No block split,
  attribution screen, inventory fallback or automatic second round.
- S4.4 is group/mechanism-only: full, full-minus-O3-at-colon, full-minus-RI23-at-test,
  retained C*, C*-minus-RI, C*-minus-nonRI. Cross fact assignment and queried child,
  four cells with both orders; 20 families initially. Explicit conditional extension rule.
- S4.5 starts C0=33 named heads: strong25 + L13H18/L24H19 + the six conditional candidates.
  All test positions, outside heads replaced by LOFO means; MLPs/demos/prefix remain live.
  Include full and all-head-mean baselines. One RI17 addback to C50 if C0 fails; then stop if
  still failing. Group-first reduction, eight named disjoint blocks (plus RI17 if added),
  two passes/96-trial cap, F/L/accuracy guards, full89 recheck, matched group completeness.
- S4.4/S4.5 share endpoint registry/cache and expanded mean statistics, not separate forward
  campaigns. S4.3, fixed C0 pilot, and three full-model S4.4 configurations can start in
  parallel. Final C-based S4.4 waits for S4.5 reduction/completeness; exact final masks are
  adaptive. Freeze everything before opening held-out87. Implementation remains future work.

## S4.2 readiness verification (retained provenance)

S4.2 run `stage4_s42_all_v1` (Slurm 924204, runtime host s-004) completed and was
downloaded as `results/stage4_s42_all_v1_results.tar.gz`, then safely extracted locally.
This supersedes the implementation-only/not-submitted S4.2 status below.
At the initial download, the user authorized readiness checks only; full analysis was
authorized later and is completed above.
All 51,904 expected records passed portable checksum/identity/coverage verification.
K selection, conditional/refinement/extension schedules, sensitivity closure, and the mean
bank reproduce. All 16 measurement/root gate reports passed their six-pair checks.
All 49 exported analysis/final-summary files reproduce from raw records within 1e-10
numerical tolerance. Historical and cross-phase reuse were independently checked.
The eight G1 groups, six blocking patterns, 31 RI candidates, seven RI/reference cohorts,
G3 diagnostics, downstream attention null and pre-own-replacement equality are present/valid.
Only the 89 discovery families were used. No additional GPU measurements or mandatory
CPU completion stage are needed for the declared executed S4.2 scope.
Audit: `results/stage4_s42_readiness_20260924/readiness.json`; reproducible audit script
and numerical reproduction outputs are alongside it. Original limits remain: historical P effects may
lack individual logits, unextended core G1 claims are provisional, mean reverse is a
sensitivity test, and this is discovery rather than held-out circuit validation.

## Latest verified update — 24 September 2026

**S4.1 is complete, downloaded, locally verified and analyzed, including the implemented
controls/refinement and both full-discovery extensions.** This supersedes the waiting/no-output
and not-yet-downloaded statements in the historical sections below. Do not resubmit S4.1.

Run: `results/stage4_all_v2`, Slurm 923239; executed package `results/stage4_s41_v2`.
All 21 measurement workers and the root gate passed saved full/tiny gates. Archive SHA,
source/input identities, chunk checksums and exact IDs verified locally. All seven phases'
analysis CSVs, interaction exports and six adaptive scheduling decisions reproduce.
70,408 records =70,328 new endpoints +80 reused Stage-2 P effects. Counts: coverage 712;
role localization 240; slot localization 320; seed 30,760 (769 configs); refinement 4,200;
extensions 25,632 +8,544 (96 routes x178 pairs x2 directions).

Audit: `results/stage4_review_20260924/{REVIEW.md,verification.json,report_facts.json}`.
Report: `חומר כתוב/Stage4_Overleaf/main.tex`; portable `Stage4_Experiment_Overleaf.zip`;
The original S4.1-only report had 13 pages and is now archived. The current 27-page
combined report and revised prospective design are described above; the executed protocol
remains unchanged as historical provenance.

Main findings: L17H1 fact-site V routes into L18H18/L18H19 are strong (union I +3.400/+2.074),
with opposing routes also present. L15H25 child-last V reaches L16H1/L16H21 (+0.182/+0.210),
whose colon Q routes reach the layer-18 pair. Consecutive separately tested edges do NOT
establish a composed chain. Layer-18 sources have Q/KV, MLP and bypass exits.
95/96 noising and 93/96 restoration routes retain on 89 families; 94/92 on the additional 69.
All 96 mean signs agree across directions. These remain selected discovery results.
19 retained core configurations are unextended, not null. All 55 controls are smaller than
their target in mean absolute family effect, but 7/17 other-site controls retain; 0/38 random
receivers retain. Controls are not norm-matched. Strong route magnitudes are order-sensitive.
L13H18 P +0.314/F +0.022: no retained route among 138 localized direct configs; local mediation
is warranted. L24H19 P +0.344/F +0.349 has an extended bypass +0.165.

Historical implementation checkpoint, superseded by the completed run and analysis above:
S4.2 was **implemented and locally tested before submission**.
The user explicitly authorized implementation from report Section 7 and
`results/stage4_review_20260924/s42_proposal.json`. Eight G1 groups deduplicate
to 49 colon masks (13 singles +36 joint). Seven singles are in bundled saved F references;
six need Stage-3 raw measurements. With compatible reuse: 1,440 new base joint endpoints on 40
pairs, excluding blocking/baselines/gates/sensitivity/extensions. Six source-mask blocking
patterns and unchanged G2 RI31/K/reference rules plus G3 are specified. L17H24->L21H23 KV is
core-only and provisional; extended V is below rule. Do not silently treat it as confirmed.

New isolated package: `pilot_v2/stage4_s42_v1_update.tar.gz`; extracted verification copy
`results/stage4_s42_v1`; frozen inputs `results/stage4_s42_inputs_v1`.
Sources: `pilot_v2/s42_{plan,engine,means,run,analyze,pipeline,review,submit}.py`,
`test_s42.py`, `test_s42_pipeline.py`, `build_s42_bundle.py`, and S4.2 Slurm wrappers.
22 tests passed from the extracted package (including six inherited tiny OLMo2 gates),
with no skips; Bash syntax, frozen inputs, bundle hashes and pipeline CLI verified.
No new GPU jobs were submitted. The user will execute the supplied chat commands.
Run name proposed: `stage4_s42_all_v1`. Do not invent a job ID or claim full-model gates passed.
No subagents were used.

One allocation supports up to six GPUs in three independent two-GPU workers, sharded by
family. Automatic sequence: gates, core G1/blocking/K/G3, selected-K RI audit, triggered
G1 splits, bounded discovery extensions, discovery mean collection, mean sensitivity,
CPU analyses. Core initial schedule: 74 unique configurations, 3,800 direction/pair
records, of which 480 are reusable historical F records (L27H6 is recomputed where G3
requires pre-intervention diagnostics). New joint base G1 remains 36 configurations.
All 13 group F references were recovered from checksummed Stage-3 chunks. Scope-P
historical references retain their original limitation: some store effects only,
not individual patched logits; new endpoints save both logits and candidate readouts.

Execution details frozen in `s42_plan.POLICY`: K uses the symmetric post-intervention
clean-minus-corrupt gap, (G-I_K-J_K)/G; deterministic sign splits and top conditional
pair; at most 16 ranked G1/blocking contrasts extended with all matched terms;
informative RI/G3 comparisons extended; selected extended comparisons receive mean
sensitivity in both directions, including those failing the expanded donor rule.
Other core-only G1 claims remain provisional. Mean baselines balance events within
family then families, exclude the evaluated family, use role/slot/bucket fallbacks
with at least ten supporting families, and preserve demonstrations/answer prefixes.
Mean reverse is a corrupted-recipient sensitivity test, not clean restoration.
Reuse requires complete configuration, donor mode, direction, exact inputs and mean-bank
identity. Incomplete/corrupt coverage fails closed. Submission checks existing Slurm jobs;
resume recovers locks only after Slurm confirms the old job has terminated.

Held-out 87 remain sealed. Plan S4.4/S4.5 together later to avoid duplicated measurements;
new nonlinear interventions still require GPU work. Current report includes this distinction.

## 1. Where we are NOW

**Stage 3, including optional section 1.5, is complete and analyzed. Stage 4.1 is implemented.
The user now describes us as waiting for results of this first part of Stage 4.**
Do not restart Stage 3, redesign the agreed experiment from scratch, or submit a duplicate job.

Evidence/status distinctions:
- Locally: prepared immutable inputs, 10 core CPU tests plus 4 orchestration tests passed;
  shell syntax and extracted upload bundle hashes/CLI checked. No full-model GPU execution by the assistant.
- User pasted successful local v1 preparation/tests and v1 upload, extraction/hash verification,
  and remote tests: 7 passed, 3 skipped because original source datasets are absent from the isolated bundle.
  Those 3 passed locally. The six tiny-model intervention tests passed remotely.
- We then supplied exact instructions to upload v2 and start the one-command pipeline.
- **No v2 Slurm job ID, submission log, full-model gate result or scientific Stage-4 output has yet been
  shown in this conversation.** Waiting for results is the user's latest stated operational status;
  confirm the actual job/status from their next output rather than asserting it passed/is still running.
- Authoritative next run name proposed: `stage4_all_v2`; isolated package: `stage4_s41_v2`.
  v1 and any prior runs must be preserved. Do not resume a v1 run with changed v2 code.

Immediate next action: obtain/check current job ID and `pipeline_state.json`, `gate.json` per worker,
`pipeline_done.json`, and logs/results when available. Diagnose failures before scientific interpretation.
No scientific conclusion about any Stage-4 route or group exists yet.

## 2. Research question and why Stage 4 matters

Current priority is **single-hop in-context mother-of reasoning** in pinned OLMo-2-1124-7B,
not two-hop composition and not broad semantic understanding across relation domains.
We ask: **which cooperating heads/communication routes implement contextual relational retrieval,
and do heads selected by the RI/SIH criterion participate in those mechanisms, and how?**
Audit both omissions (important non-RI components) and selective/conditional/opposing participation
of RI candidates. A head's RI score, attention pattern, decoded name, or standalone causal importance
is not by itself evidence of circuit membership. Do not equate RI-selected candidates with proven SIHs.
Do not assume a low standalone effect excludes a backup/conditional circuit role.

Trajectory:
1. Stage 1: observational RI selection/diagnostics; test-only union of 59 heads (61 with two historical extras).
   RI uses raw current-token OV and contextual attention qualification; it has lexical/tokenization limitations.
   At identical current token and visible token-ID set (notably the final colon), its raw-OV candidate scores
   cannot adapt to swapping the fact assignments. This is a measurement limitation, not lack of model capability.
2. Stage 2: causal head interventions. All 1,024 heads on 40 common pairs; extended shortlist on 178 pairs.
   Large importance mismatches motivated distinguishing mechanisms from observational selection.
3. Stage 3: deep characterization of 105 fixed heads: positions, attention changes, contextual outputs,
   attention/value interventions, reverse patching, diagnostics and controlled readouts.
   Established writer-like versus answer-position/mixed candidates, but not their communication graph.
4. Stage 4: map answer-relevant causal routes, form evidence-based groups, test cooperation/backups,
   evaluate a faithful head mechanism, and assess RI participation. The current S4.1 run maps routes;
   later group and faithfulness tests are essential before claiming a circuit.

Model revision: `7df9a82518afdecae4e8c026b27adccc8c1f0032`.
Discovery: 89 families / 178 paired orders; initial mapping: 20 common families / 40 pairs.
Held-out: 87 families of the SAME task, still sealed; one frozen final suite, no adaptive selection there.
Main metric: clean-answer minus corrupted-answer logits at the FIRST DIVERGENT answer token,
with shared answer prefix teacher-forced. I = intact-clean margin minus patched-clean margin;
J = restored-corrupt margin minus intact-corrupt margin. Preserve both logits and negative effects.
Average orders within family, then families; distinguish signed mean from mean absolute family effect.
Scope P = all original prompt positions; Scope F = original final colon. Do not confuse colon with a later
metric position on shared-prefix pairs. Demonstration donor differences may be zero, but scope is still explicit.

## 3. Source hierarchy: read selectively, verify claims against primary results

Project root: `C:/Users/User/OneDrive/Documents/computer science/4B/NLP/final proj/project`.
Root README and old single-hop methodology Stage 3/4 sections are NOT the current plan.

Current research documents:
- `חומר כתוב/Stage4_Protocol.tex`: authoritative revised Stage-4 plan (v2, route-first).
  `Stage4_Protocol_Overleaf.zip` is a generated standalone Overleaf package.
- `חומר כתוב/Stage4_Experiment.tex` and `Stage4_Circuits_and_Communication_preliminary.tex`:
  earlier user proposals/inspiration, superseded where inconsistent with Protocol and this handoff.
- `חומר כתוב/Stage3_Results_and_Analysis.tex`: integrated final Stage-3 report, including section 1.5.
  Overleaf mirror: `חומר כתוב/Stage3_Overleaf/main.tex`; figures and tables are packaged there.
  Do not select an outdated duplicate under `Claude outputs/` by a broad filename glob.
- Stage-3 plan: `חומר כתוב/Stage3_Head_Level_Characterization_revised.tex`
  (user referred to new_stage_3_experiment). Read actual source before judging implementation.
- Stage-1 report: `חומר כתוב/Stage1_Results_and_Analysis_updated.tex`;
  supporting `Stage1_RI_Test_Extension_Results_and_Analysis.md`.
- Stage-2 report: `חומר כתוב/Stage2_Results_and_Analysis.tex`.
  Single-hop methodology remains useful for stages 1/2; its original stages 3/4 were redesigned.

Primary/local evidence:
- `results/ri_test_v2/candidates.json`, supporting RI outputs and `results/ri_test_v2_analysis/`.
- `results/stage2_v1/`, `stage2_v2_extension/`, `stage2_v2_analysis/`.
- `results/stage3_v1/analysis/` and raw replicas; `results/stage3_inputs_v1/` contains aligned pairs/references.
- Independent checks: `results/stage3_review_20260922/Stage3_review.md`.
- Readout: `results/stage3_readout_v1/`, frozen `results/stage3_readout_inputs_v1/readout_plan.json`;
  review: `results/stage3_readout_review_20260922/Stage3_section1_5_analysis.md`.
- Stage-4 design: `results/stage4_design_v2/manifest_proposal.json`, `ri_reduced_selection.csv`.
  v1 design's 506-route registry is superseded; do not execute it.
- Stage-4 implementation inputs: `results/stage4_inputs_v1/plan.json`, `pairs.jsonl.gz`,
  `coverage.json`, `seed.json`. The seed is preview-only until Coverage selection.

## 4. Findings to carry forward, and limits

- 25 strong Stage-3 candidates (|P| >= 0.3); 21 have |F| >= 0.3: 14 positive, seven negative.
  Four further strong P heads are L13H10, L14H26, L15H25, L17H1.
- L17H1: P +5.701, F -0.056; strong fact-position effects including `is` +1.769 and period +1.377.
  Readout: head replacement changes identity contrast at `is` by -0.458 vs full source swap -0.521,
  same negative sign in all 20 families. Supports head-dependent mother-token preference there;
  not free name decoding, semantic binding proof, or an 88% causal mediation fraction.
- L15H25: P +0.679, F +0.169; child-last site about +0.429.
  Readout is inconclusive because the scope is weakly sensitive to source corruption, not a null proof.
- L13H10: P -0.681, F -0.029; L14H26: P +0.399, F +0.007: additional writer candidates.
- L17H3: P -0.546, F -0.402; mixed role. Early non-query mother importance is weak on average.
  Do not call it an established strong early writer, or partition all heads rigidly into reader/writer types.
- Eight strong original RI candidates: positives L16H1,L16H21,L17H24,L18H19,L22H5;
  negatives L19H16,L23H15,L26H23. Negatives matter; opposition can coexist with participation.
- L26H31: weak standalone causal effect, relevant attended-name behavior, possible backup; unproven.
- Outside-inventory reserves L13H18 and L24H19 have strong common-40 P (+0.387,+0.390),
  not yet established full-discovery effects/sites; Coverage addresses this.
- Large per-family interactions can cancel in signed averages. Projection gaps are NOT proof of indirect action;
  shared OLMo normalization prevents interpreting raw head-unembedding projection as exact causal attribution.
- Query-change control already measured on 40 prompts (reported success 38/40). The double-change condition
  was not measured, but user explicitly declined a new standalone behavior/capability survey.
- Stage-3/readout source computations were independently reproduced; see reviews for nuanced evidence.
  The older handoff contains overstated mover/indirect/additivity claims: do not revive them.
- Prior 3.7-hour Stage-3 estimate was multi-replica wall time, not a valid universal GPU-hour estimate.

## 5. Agreed Stage-4 redesign (do not revert to the broad first draft)

S4.1 = exact P route mapping with controls. Initial pool 25 unique strong heads; five initial sources:
L13H10, L14H26, L15H25, L17H1 and mixed L17H3.
- P1: L17H1 source roles -> later heads' K/V and local MLPs.
- P2: L15H25 child-last -> later candidates, prioritizing L16H1/L16H21; `is` control.
- P3: colon-source outputs -> later Q/KV, MLP or exact residual bypass. Helps distinguish chains from parallel branches.
- P4: secondary sources/mixed masks. Weak early sources are conditional follow-up, not a forced broad sweep.
A route must affect the answer via the isolated channel, not just change an activation.
K/V injection is at source/fact positions, output row at colon; KV is intentional, not a typo for QK.
Same-layer head-to-head/future-to-past are impossible. Same-layer MLP after attention is eligible.
Freeze post-normalization residual increments; source shared normalization remains live.
A query-only change cannot alter earlier fact activations. A direct cross-position Q residual route is a null.

S4.2 = groups after sufficient mapping (does not require exhausting every P route first):
- G1: outgoing shared-source and incoming shared-receiver groups supported by routes. Singles, whole group,
  group-minus-one and focused splits when interaction warrants it; NO exhaustive 2^25 or arbitrary fixed seven-head sweep.
  Includes former G4 and receiver blocking: restore source in corrupt recipient, clamp recipient outputs to original
  corrupt values, measure loss of restoration, with self-clamp and reverse controls.
  Do not assume L15H25/L17H1 form a special pair; test jointly only if evidence motivates shared mechanism.
- G2: narrowed original 59 to **31**, retaining best original eligible RI rank <=3 in ANY original ranking
  OR max(|P|,|F|) >=0.1. 23 qualify by rank +8 further causal candidates; 28 deprioritized, not proven absent.
  All eight strong RI and L26H31 retained. G2 residual group is 31 minus eight =23, distinct from rank-selected 23.
  Reuse original paired-donor intact-model singles; new conditional singles on weakened-reader background K,
  initially 40 pairs. Use same scope/baseline for comparison. No redundant intact single sweep.
  Three RI groups (5 strong positive,3 strong negative,23 residual); four matched 23-head non-RI reference groups.
  Matched references are not selected strong heads. Extend informative findings to 178; mean replacement is sensitivity.
- G3: targeted L26H31 backup test, may run alongside mapping after gates/baselines; incoming upstream removals can
  alter its attention, downstream L27H6 removal cannot. Conditional answer effect may still change.
G2/G3 can nominate further route candidates without waiting for the complete graph.

S4.3: triggered bounded expansion/mediators; can be invoked during mapping if strong effects remain unexplained.
S4.4: order robustness; Four-condition panel is OPTIONAL for an unresolved mechanism question, NOT an entry gate.
User explicitly accepts necessary intact baselines for any genuinely new intervention, but rejects redundant
capability demonstrations or automatic broader behavioral reruns. Reuse saved compatible measurements.
S4.5: assemble/reduce faithful head mechanism, evaluate completeness/backups, then freeze validation suite.
Primary scope is conditional on live MLPs, demos, embeddings, normalization and answer-prefix computation;
not a fully isolated edge circuit. Circuit masks must retain corresponding roles in ALL facts, not oracle answer-fact masks.
Faithfulness: inherited >=80% gap plus overshoot/error/accuracy guards; multiple metrics and baseline sensitivity.
Membership requires a tested answer-relevant route AND conditional causal contribution, possibly negative;
group effect alone does not establish each member. Absence from one reduced circuit is not universal nonparticipation.
No new observational RI, parent-coverage, location-world transfer, default SVD/readout sweep.

## 6. What code EXISTS, and exactly how to continue

Entry guide: `pilot_v2/RUNBOOK_stage4_s41.md` (Hebrew, local PowerShell vs remote Bash).
Implementation: `stage4_plan.py`, `stage4_engine.py`, `stage4_run.py`, `stage4_analyze.py`;
controller `stage4_pipeline.py`; tests `test_stage4.py`, `test_stage4_pipeline.py`.
Prepared v2 isolated archive via `build_stage4_bundle.py`; archive SHA sidecar beside it.
`stage4.sbatch`/`submit_stage4.sh` support manual phases; prefer
`stage4_pipeline.sbatch`/`submit_stage4_pipeline.sh` for the automatic workflow.

GPU command already supplied to user (remote pilot_v2 directory):
```bash
bash stage4_s41_v2/submit_stage4_pipeline.sh --name stage4_all_v2 --gpus 6
```
One node, one Slurm allocation, three independent two-GPU workers; also supports 2/4 GPUs.
Default 48h capped by partition MaxTime; studentkillable may preempt regardless. No automatic requeue promised.
`--resume` resumes same name/code/inputs/GPU count, skipping verified complete work. Never submit duplicate live run.
Model is loaded per worker/phase, not per intervention; GPU kernels currently batch=1.

Automatic sequence: tiny-model gates + six full-model gate pairs -> Coverage -> optional role/slot localization ->
584-seed-config preview expanded by Coverage if warranted -> core refinement -> selected full-discovery extensions ->
CPU analyses. Seed endpoints: 584*40=23,360, excluding captures/gates. Coverage: 712 records, 80 reused P measurements.
Total full-discovery route cap 96: up to72 seed, remainder refinement; overflow remains explicitly untested.
No-follow-up is a valid boundary, not a failed run. Computational failure stops, never relax tolerances to obtain results.
All stages repeat necessary gates; root pipeline gate precedes any scientific phase.

Outputs below `$PILOT_RUNS/stage4_all_v2/`:
- `pipeline_manifest.json`, `pipeline_state.json`, final `pipeline_done.json`;
- `runs/<phase>_rN/`: gate/tiny_gate, manifest, runtime, checksummed chunks, state/done, baseline residual profiles;
- `schedules/`: actual adaptive frozen schedules (DOWNLOAD THESE, not just original seed preview);
- `analysis/<phase>/`: events, family effects, route summary, context summary, verification;
- `analysis/interactions_core/`: paired KV and union-mask contrasts, unavailable terms explicitly reported;
- `logs/<phase>_rN.log`: worker traceback/progress.

Next assistant must:
1. Inspect job/state/logs; if failure, separate runtime/gate issue from a scientific null. Preserve failing artifacts.
2. Once completed, download the complete pipeline tree (or schedules + all relevant run chunks/manifests),
   verify every shard/ID/checksum, and distinguish recycled Stage-2 rows from new Stage-4 records.
3. Review family-first effects, opposite signs, mean absolute effects, self controls, wrong-site/random controls,
   reverse restoration, prefix sensitivity, intervention norms and sample extension. Do not infer from signed mean alone.
4. Translate supported routes into candidate shared-source/receiver groups and chains; identify unexplained strong sources
   and conditional candidates. Compare with Stage 3 and RI31 selection, explicitly noting bounded untested possibilities.
5. Discuss concrete group selection with user, then implement agreed S4.2 G1/receiver blocking and G2/G3.
   Trigger S4.3 mediation only where gaps justify it. Do not assume those later stages already exist in code.
6. Keep later report/Overleaf updates grounded in actual outputs, and preserve the sealed validation population.

IMPORTANT: only S4.1 and its prerequisites/refinements are implemented. No G1/G2/G3, mean-mask/circuit machinery,
S4.3 mediated expansion or S4.4/S4.5 GPU code yet. The mean-all-retained gate belongs to that later implementation.
The full-model gates are not validated until the actual server results show it; CPU tests do not prove GPU/runtime parity.
Verification records: `results/stage4_implementation_review/{cpu_verification,pipeline_verification}.json`.

## 7. Environment, tools and reproducibility

Local: Windows PowerShell. Bundled Python:
`C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
CPU torch/transformers available via `PYTHONPATH=tmp/stage3_test_deps` (tested 2.6.0+cpu /4.51.3).
The local `pilot_v2/tokenizer_source` is OLD GPT-NeoX, not OLMo2. Never use it for OLMo verification.
Runtime loads pinned tokenizer/weights from the server model directory and asserts exact saved token IDs/prefixes.
Use rg for search, inspect real source and actual raw/processed results. Avoid dumping giant files; no broad re-analysis
when provenance/reproduction is already established. Use a PDF skill only when actually reading/rendering PDFs;
prefer provided .tex for scientific content. Use relevant tools/skills when required, not by keyword habit.
No subagents unless user explicitly requests delegation. Do not invent tool availability; discover it.

Remote Bash directory: `/home/yandex/DLWorkShop2025b/maximg/pilot_v2`.
Always `source runtime.sh` there; it defines PILOT_STORAGE/PILOT_RUNS/HF_HOME and reuses working package directories.
Current runtime uses storage under pilot_v2; don't copy an older handoff's hardcoded storage path.
Pinned model expected at `$PILOT_STORAGE/model/7df9a82518afdecae4e8c026b27adccc8c1f0032`.
Do not upgrade packages or overwrite runtime.sh casually. Partition studentkillable, account gpu-students,
current wrappers exclude s-002, not pinned to s-004. Node availability/time/QOS limits need fresh evidence.

SSH: user handles interactive auth; no established assistant SSH session. Direct verified endpoint
`maximg@132.67.130.126`, known-hosts `tmp/slurm_c002_known_hosts`, strict ED25519 checking.
Preserve existing SSH trust; do not accept new keys to bypass failures. Copyable upload/download examples in runbook.
Do not resubmit, change running code, or delete locks until verifying the previous job is no longer active.
The controller only cleans locks of its own confirmed-dead child processes; SIGKILL can leave stale locks.

No local TeX compiler found; do not claim PDF compilation. Overleaf source has static syntax/reference checks.
Preserve text/images when user requests format-only edits. Standalone Stage3 Overleaf main.tex has tables inline;
figures require assets. Do not edit an older PDF and assume its tex was updated.

## 8. Working with the user as research mentor

- Speak concise Hebrew; artifacts/research LaTeX in clear professional English. Runbook in Hebrew is intentional.
- Be rigorous in transformers, causal inference, experimental design and statistics; teach judgment, not jargon.
  Explain what a test measures, why needed, intervention, metric/denominator and limits. Concrete examples help.
- Separate observations, hypotheses, interpretations, code behavior, proposed commands and actually executed runs.
  A report/assistant assertion is secondary evidence; inspect source data for substantive conclusions.
- User welcomes disagreement/correction, not reflexive agreement or defending our earlier design. Focus on material
  logical/scientific issues, not forced nitpicks. They understand basics; keep answers proportional to the question.
- Prefer practical, informative tests. Avoid method proliferation, arbitrary coalition sweeps, redundant reruns,
  and extra capability surveys. Explain any meaningful scope/cost tradeoff and preserve prior comparisons.
- Work autonomously on authorized tasks, short progress updates, no repeated permission questions for routine work.
  Do not submit new resource-consuming jobs, change credentials/trust, publish, or modify unrelated work without scope.
- Commands must be copyable: separate LOCAL PowerShell from REMOTE Bash; no escaped underscores/@ in code blocks.
  Favor one-command adaptive workflow, multiple GPUs where useful, generous allowed time and reliable resume.
- Preserve raw data, original proposals, unrelated edits and pre-registration. Do not retrospectively tune thresholds
  to claim confirmation. Threshold .1 is practical selection, not statistical significance; RI31 causal preselection
  cannot be used as an independent proof that RI predicts causal importance.
- When tools/tests fail, diagnose and state what is blocked. Do not claim an archive, test or GPU gate passed without evidence.

## 9. Git and transfer

Fresh local inspection: branch `main`, remote `origin = https://github.com/Maxgolu/NLP_proj.git`.
Many current files still untracked: Stage4 sources, current research tex/Overleaf package, design/input metadata.
Do not `git add .`: unrelated `Claude outputs/`, stage3_analyze-1.py, runtime state and extracted smoke-test trees exist.
No staging/commit/push was performed by the assistant for this handoff. User will receive explicit commands in chat;
no new Git-command file requested or created now. Existing update_stage4_git.ps1 predates this request.
Sandbox git ownership warning can be handled per-command with `-c safe.directory=<exact project>`; avoid global changes.
PowerShell may disallow ps1 scripts: use ordinary commands, do not change execution policy for this task.
Archives/ZIPs, raw large results, tmp environments are deliberately not in Git. Keep OneDrive/cluster copies.
Track source/protocol/design JSON, hashes and small verification records; do not stage extracted bundle_smoke trees.
Older history is available in the archive only when needed; avoid spending tokens reading it all by default.
