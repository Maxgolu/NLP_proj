# Research handoff — read this first

## 22 September 2026 — Stage-3 run COMPLETE; results analyzed; report drafted (LATEST)

Observed (from files downloaded by the user to `results/stage3_v1/`): `summary.json`
`complete: true`, `gate_passed.json` passed on all 3 replicas, 105 heads x 178 pairs,
40 common position-scan pairs, 160 core prompts, 34,526 matched RI comparisons; gate
cost 0.151-0.152 s/patched forward, 263,328 forwards (~3.7 GPU-h). The user ran
`stage3_analyze.py` on the cluster and downloaded the archive; `analysis/*.csv` exist.

Deliverables produced in this session (written to the user's folder on 22 Sept):
- `Stage3_Results_and_Analysis.tex` (+ figs/fig1..fig9, fig2b, tables/tab_profiles.tex),
  compiled locally to a 12-page PDF; intended location `חומר כתוב/`.
- `stage3_analyze.py` UPDATED: attended-name mover classification implemented
  (`attended_name_event`, `summarize_attended`, `attended_only`, CLI `--attended-only`),
  also integrated into the full `analyze()`; writes `analysis/attended_name_movers.csv`,
  `attended_name_events.csv`, `attended_name_rule.json`, appends to `head_cards.md` and
  `summary.json`. Decisions taken (documented in the rule file): primary contrast on the
  FIRST token of the attended name (last token = sensitivity); controls = all other
  visible distinct names, both roles (same-role = sensitivity); self events included
  (non-self = sensitivity); all four core variants incl. query_change; extra sensitivity
  labels for final-site-only and base-only. Cross-check uses the SIGN of the 178-pair
  Scope-F importance (name mover <-> I_F>0; negative <-> I_F<0); |I_F|<0.05 reported as
  "no causal contrast to check".
- Attended-name classification FINAL (all 480 core prompt records, 94,015 events):
  name movers L27H6 (96%), L17H24 (94%), L26H31 (91%, mean +2.09 but I_F=+0.027 —
  moves names it attends to but attends to the answer in only 6% of events: moving and
  selecting are separable), L18H19 (84%), L22H5 (82%), L21H6 (82%); negative name mover
  L20H1 (17% positive). L21H18 79%, L19H22 78% (colon-site-only: name movers), L18H18
  74%, L23H15 64% (NOT a negative mover; effect indirect), L30H18 67% positive on the
  distractor it attends. 19 heads insufficient. Files: `results/stage3_v1/analysis/
  attended_name_movers.csv`, `attended_name_events.csv`, `attended_name_rule.json`;
  `head_cards.md` and `summary.json` updated. The two *_PROVISIONAL files can be deleted.
- Plan paragraph with corrected cross-check sentence: `חומר כתוב/Stage3_plan_attended_name_paragraph.tex`
  (the local plan .tex never contained the paragraph; paste this version into Overleaf).

Main Stage-3 findings (family-level descriptives, 89 discovery families; details and
exact numbers in the report and `results/stage3_v1/analysis/`):
- L17H1 (I_P=+5.70) has NO colon effect (I_F=-0.056). Its effect enters inside the
  query-fact sentence "<mother> is the mother of <child>.": mother middle +1.15, mother
  last +1.67, "is" +1.77, "of" +0.36, period +1.38; nothing at question/colon; small
  negative at distractor mothers. Reverse patching recovers +5.21 (P), -0.055 (F).
  The Stage-2 label "layer-16-26 heads are readers at the colon" does not hold for it.
- Other early-position heads: L15H25 (+0.68; acts at the query child's LAST token,
  +0.43), L14H26 (+0.40; diffuse), L13H10 (-0.68; "is"/"of"/mother-last, negative).
  All other 21 heads with |I_P|>=0.3 have I_F/I_P in [0.96,1.07].
- Single-position patches are near-additive within a head: sum over positions vs
  Scope-P on the same 40 pairs, Spearman 0.913, median |resid| 0.013, max 0.135.
- Logit decomposition: promotion (corrupted-answer logit) carries 88-98% of I_F for all
  strong heads except L27H6 (P=+1.96, S=+1.10).
- Attention at the colon (base, 20 families): mother-attenders L27H6 .58, L21H18 .60,
  L23H15 .51, L22H5 .34, L19H22 .31, L21H6 .30, L19H16/L26H23 .21, L20H1 .17, L18H18
  .14, L18H19 .13; child-attenders L16H21 .40, L16H1 .36; self-attenders L25H17 .57,
  L21H23/L30H13/L17H17 ~.19, L13H10 .42; L30H18 attends distractor mothers (.11) more
  than the query mother (.05). Changed-query control moves attention to the new fact in
  100% of families (e.g. L21H18 +1.07, L16H21 +0.83 to the new child); reorder <=0.11,
  corruption <=0.085; random controls <=0.18. Model answers the changed query correctly
  in 38/40 prompts.
- Attention/value factorial: values carry the effect for every head; routing term small
  (only negative heads -0.1..-0.22 and L27H6 +0.27); interactions within +-0.27.
- Contextual output o_j W_U at the colon (query mother - distractor mothers, first
  token): L27H6 +2.99 (direct writer), L30H13 +.76, L22H5 +.41, L21H6 +.33, L21H18 +.25;
  L30H18 -0.88; INDIRECT heads: L18H19 +.16 (I_F 3.08), L18H18 +.11, L20H1 -.09,
  L23H15 +.04, L19H16 -.04, L25H17 +.16; L16H21/L16H1 ~0. Raw-embedding projection
  uninformative for all heads (|x|<0.008; rho -0.07); contextual rho +0.32.
- Weight copying (314 name tokens): rho with signed I_P +0.37, with |I_P| +0.02.
  Copiers: L30H18 +.60 (!), L21H6 +.35, L17H24 +.19, L19H22 +.17, L18H19 +.13, L21H18
  +.12; anti-copiers L26H23 -.20, L20H1 -.17, L19H16 -.09, L18H18 -.08. L27H6 +.04,
  L17H1 +.01. Random controls within +-0.044.
- Attended-name movers (80% rule, final): name movers L27H6 (96%), L17H24 (94%), L26H31
  (91%, no causal effect), L18H19 (84%), L22H5 (82%), L21H6 (82%); negative name mover
  L20H1 (17% positive). L21H18 79% (borderline). L19H16 77% negative, L26H23 79%
  negative (below rule). L23H15 is NOT a negative mover: positive contrast in 64% of
  2,428 events while I_F=-0.95 -> its negative effect is indirect. L30H18: positive
  contrast (67%, +0.60) on the DISTRACTOR it attends -> negative effect by copying the
  wrong name. All labels vanish at the last-token anchor.
- Matched contextual RI does not repair RI: names_gap rho -0.26 (raw) -> -0.17
  (contextual); target ~0. 35 inventory heads insufficient.
- Synthetic fingerprints (model correct: KV 100%, repeated 96.1%): retrieval-type
  attention L19H16 .97, L21H18 .88, L23H15 .84, L18H18 .56, L16H1 .47; induction-type
  L23H15 .66, L21H6 .64, L16H1 .45, L21H18 .39, L19H16 .36. Output sign: L21H6/L21H18/
  L18H19/L22H5/L16H1 promote the copied token; L19H16 -.24, L26H23 -.19, L23H15 -.05 do
  not. L27H6, L17H1, L20H1, L30H18 show no fingerprint (task-specific).
- Hypothesis assessments: copying partly supported (5 movers; not necessary: L18H18,
  L27H6 weights ~0; not sufficient: L30H18); contextual-information supported for
  location (attended positions, values), content open; early/late supported and
  sharpened (writer L17H1 in the same layer band as readers).
- Stage-4 hand-off: path patching L17H1 (mother tokens/"is"/period) -> mother-attenders'
  values vs keys; L15H25 -> L16H21/L16H1; downstream readers of the indirect heads
  L18H19/L18H18/L20H1/L23H15/L19H16 at the colon; joint positive-vs-negative group
  interventions; only then faithfulness on the 87 held-out families.
- Bridge note: the desktop link works only briefly after each user message; stage/commit
  in the first calls of a turn.

## LATEST OBSERVED STATUS — 22 September 2026, job 916699

Supersedes the unconfirmed full-run status below. User supplied scheduler and
run outputs: full Stage-3 job 916699 is RUNNING on s-004, elapsed 02:27:47 at
the first snapshot. Global gate passed. Cost estimate: 263,328 patched forwards,
0.1524–0.1532 seconds per patch across three replicas, 3.7364 estimated patch
wall hours EXCLUDING loading, diagnostics, captures and IO.

Two snapshots approximately 6–7 minutes apart show causal-profile progress:
replicas moved from pair/chunk 101/3, 94/1, 103/4 to 105/6, 97/6, 106/1
(178 pairs and seven chunks each). Completed checkpoint markers rose from
2,071 to 2,158. This is evidence of ongoing computation, not a final scientific
correctness verdict or completion of Stage 3. Pair progress is not proportional
to total runtime: common-subset position scans cost more, and diagnostics follow.

User attempted `scontrol update JobId=916699 TimeLimit=08:00:00` and received
Access/permission denied. No extension was confirmed; submitted default remains
four hours unless an administrator changes it. Let the job continue; request an
administrator extension if desired, otherwise resume only after it has stopped
with identical code/inputs/six-GPU allocation and `--name stage3_v1 --resume`.
For future submissions the user permits s-005 and excludes only s-002. If a hard
kill leaves a lock, verify its recorded job has ended before removing only that
lock. Full completion and output integrity still need verification.

## 22 September — תכנון ראשוני מוסכם לשלב 4

נכתב מסמך עצמאי לאוברליף, בזמן ההמתנה לתוצאות שלב 3:
`חומר כתוב/Stage4_Circuits_and_Communication_preliminary.tex`.
זהו תכנון בלבד; לא בוצעו מדידות שלב 4 ולא שונה קוד הריצה הנוכחית.

- שלב 4 עוסק במסלולים, קבוצות, מעגלים ובהשתתפות מועמדי RI. 105 הראשים הם נקודת פתיחה, לא גבול החיפוש; יש לאפשר רכיבים נוספים ותיווך MLP.
- חברות דורשת מסלול סיבתי ותרומה מותנית למנגנון, גם בנוכחות גיבויים. תוצאה קטנה בבדיקה בודדת אינה שוללת חברות; אפקט קבוצתי אינו מוכיח תרומה של כל חבר.
- הסדר: מפת הבדלים בין שכבות; סינון קשתות מודע־פוזיציה וכיול נפרד; אימות מסלולים בשילוב בדיקות קבוצתיות; שני צירי השחתה (עובדות ושאילתה); בניית מעגל ובדיקות נאמנות, שלמות ומינימליות; אימות קפוא.
- קריאת ייצוגים ופירוק ערוצי תקשורת הם העמקות מותנות. עקומת המרחק בין ייצוגים היא תיאורית ואינה שוללת תפקיד בשכבות מאוחרות.
- המשתמש הסכים להשאיר את 87 משפחות האימות של אותה משימת אם–בת, ולהוציא משלב 4 את עולם location, מדד תצפיתי חדש ו-parent-coverage.
- בחירת ראשים/אתרים, ספים, תקציבי חיפוש וכללי עצירה ייקבעו לאחר שלב 3 ולפני המדידות הרלוונטיות. סף הנאמנות המקורי של 80% נשמר. אין להציג את המסמך עדיין כפרוטוקול מוכן להגשה.
- נבדקו מבנה LaTeX והפניות פנימיות; לא בוצעה קומפילציה מקומית.

## 21 September (evening) — Stage-3 plan addition: attended-name mover classification

Decided in the plan-review conversation while the Stage-3 full run was still in
progress (no code change; no new measurement). The plan paragraph
"Copying and Token-Preference Diagnostic" in
`Stage3_Head_Level_Characterization_revised.tex` /
`חומר כתוב/new_stage_3_experiment.pdf` gained a second paragraph
"Attended-name mover classification" (user pasted it into Overleaf; recompile).

- What: for each inventory head, take the saved core-family test-block records
  (`anatomy` records in `prompts/*.jsonl.gz`: attention `argmax`, per-fact
  source/target mass, and `projections['output']` = `o_j^h W_U` over visible name
  tokens). Select events whose argmax falls on a name token; compute the contextual
  logit of that attended name minus the mean of the other visible names.
- Labels: name mover = contrast positive in >= 80% of events; negative name mover =
  negative in >= 80%; support rule >= 20 events and >= 10 families; below support
  report "insufficient", not "neither". The 80% threshold was fixed before results.
- Cross-check against Scope-F logit decomposition sign; a mismatch is reported as
  a finding, not relabeled. Candidates checked first: L23H15, L19H16, L26H23;
  classification applies to the whole inventory incl. random controls.
- Limitation: anatomy records exist only for the 20 `common_families` prompts
  (~60 prompts, all test positions), not all 89 families.
- Implementation: CPU-only addition to `pilot_v2/stage3_analyze.py` after the run;
  report it in the evidence cards under "contextual output" with the four status
  values of §1.6. Not yet implemented.
- Rejected for now (user decision): relation-specificity test (swap `mother` for
  another relation word in the 178 prompts, compare attention/output per head).
  Kept here as a future idea only.
- Plan review status: user has gone through §1.2–§1.6 of the Stage-3 plan with the
  mentor. Stage-3 results analysis will start in a NEW conversation once the run
  completes and outputs are verified/downloaded.

### 22 September (late) — review corrections applied; Section-1.5 readout code written (NOT run)
- External review of the Stage-3 report accepted almost entirely; report .tex corrected in place:
  signed means vs absolute per-family magnitudes (position additivity, AV interactions,
  reorder/corruption attention changes), projection != attribution (no "indirect/mediated"
  claims; hypotheses for Stage 4), self-attenders do read fact spans (L17H17 0.22),
  L26H31 colon-site answer fraction 12.9%, L18H19 colon-site label insufficient (17 ev/9 fam),
  count 20 (not 21) with L17H3 as mixed-position, new subsection on the current-token RI
  partition (current_name_diagnostics.json), Section-1.5 recommendation stated.
- Section-1.5 readout implemented: `pilot_v2/stage3_readout.py` (prepare/run/analyze),
  `test_stage3_readout.py` (5 tests incl. tiny-model end-to-end dry run, all pass),
  `stage3_readout.sbatch`, `submit_stage3_readout.sh`, `build_stage3_readout_bundle.py`,
  `RUNBOOK_stage3_readout.md`; frozen plan `results/stage3_readout_inputs_v1/` (160 items:
  L17H1 at query-fact "is"/period, L15H25 at child last token/"is"; 6 conditions x 2 readouts;
  240 captures + 1,602 readout forwards; 2 GPUs). Policy in READOUT_POLICY. GPU run not done.
- `git_update_2026-09-22.ps1` at project root stages/commits/pushes the Stage-3 state
  (adds .gitignore rules for attended_name_events.csv and readout raw records). Not executed.

## (superseded) Stage-3 submission clarifications and conversation transfer

This section supersedes older scheduling instructions below. This file is a
curated research handoff and source index, not a complete transcript or a copy
of all data/code/papers. A new assistant must read the current methodology,
implementation and primary result files relevant to its task before making
claims. Older interpretations (copying, writer/reader and negative-mover labels)
remain hypotheses; do not inherit them as established mechanisms.

- Job 916685 failed on s-002 at CUDA initialization, before model loading.
- User then submitted job 916696: `stage3_v1_gate_s004 --gate-only --nodelist s-004`.
  Its last observed state in the conversation was PENDING (Resources).
  Gate-only jobs NEVER continue into the full experiment.
- User prefers any eligible node rather than pinning s-004. They were instructed
  to cancel 916696 and submit `stage3_v1` without `--gate-only`, excluding
  s-002,s-005 at that time. A full run gates every replica and automatically
  proceeds to measurements in the SAME job only after all gates pass. No separate
  gate job is required. Cancellation and the new full-run job ID have NOT been
  confirmed by a pasted scheduler result; do not invent their status.
- Subsequent user instruction: for FUTURE submissions exclude only s-002;
  permit s-005. Other users running there do not prove our runtime works, so rely
  on our gate. The locally rebuilt sbatch/archive still exclude s-002,s-005;
  use explicit `--exclude s-002` to override, or update defaults before a future
  release. Do not silently change a submitted job or overwrite its frozen code.
- The full run includes BOTH sections 1.3 and 1.4 of
  `חומר כתוב/new_stage_3_experiment.pdf` (source:
  `Stage3_Head_Level_Characterization_revised.tex`): core profiles plus ALL four
  mandatory diagnostics (copying/token preference, matched contextual RI,
  attention/value interventions, repeated-sequence AND retrieval fingerprints).
  There is no results-review pause between mandatory analyses. Section 1.5
  optional extensions remain conditional on analysis of the mandatory results.
- Successful full execution plus automatic CPU integrity/coverage analysis
  should supply mandatory Stage-3 data when combined with saved Stage-1/2 data.
  Require `summary.json` with `complete: true`; a scheduler finish alone is not
  sufficient. Insufficient RI/copy-event support is explicitly reported, not
  zero-filled. Four hours is the default job limit; interrupted runs can resume
  with identical code/inputs/GPU count. No successful 7B Stage-3 gate or completed
  Stage-3 research results have yet been supplied in this conversation.
- Next action: inspect the actual current job status/logs supplied by the user;
  resolve any failure, or verify/download completed outputs and analyze them.
- Handoff maintenance: update at substantive decisions/results, not implicitly
  after every message. Keep the latest actionable state at the top, distinguish
  observed outcomes from commands merely suggested, and keep explanations short
  in Hebrew with English commands/identifiers on separate lines.

## Stage-3 gate failure: job 916685

User submitted the gate; it failed on s-002 during CUDA initialization in
`torch.cuda.mem_get_info`, before loading weights. No Stage-3 measurements ran.
Do not label this OOM or assert that the generic CUDA_VISIBLE_DEVICES hint proves
a mask bug. The worker mask is passed before fresh subprocess startup, matching
the Stage-2 launcher. Restore default exclusions s-002,s-005 (done locally and
in rebuilt archive). Retry the existing cluster package with a new gate name
`stage3_v1_gate_s004 --gate-only --nodelist s-004`; no reupload needed for this
explicit node selection. Underlying cause remains unconfirmed until the retry.

## UPDATE 21 September 2026 — Stage-3 implementation and pending GPU gate

- User authorized code, local CPU preparation/tests, README and run instructions;
  user will upload and submit. No SSH connection, GPU job, Git commit or push was
  performed. Stage-3 research results do not yet exist.
- Current plan: `חומר כתוב/Stage3_Head_Level_Characterization_revised.tex`.
  Stage 3 characterizes 105 individual heads. All four agreed diagnostics run
  for applicable populations; none is selected only after promising results.
  Stage 4 owns paths, groups, circuits and residual divergence.
- Code: `pilot_v2/stage3_{common,prepare,engine,measure,run,analyze}.py`,
  `test_stage3.py`, `submit_stage3.sh`, `stage3.sbatch`, `build_stage3_bundle.py`.
  Run book: `pilot_v2/RUNBOOK_stage3_v1.md`. Root README updated.
- Portable inputs: `results/stage3_inputs_v1/`; 178 pairs / 89 discovery families,
  534 saved prompts, 17,263 saved scored RI events (34,526 anchor comparisons), 105 heads. Original
  Stage-1/2 results unchanged. Behavioral gate reference copied from the archived
  baseline and verified against the Stage-2 manifest hash. Old preparatory
  `results/stage3_inputs/` is superseded, ignored by Git and not uploaded.
- CPU correction outputs are explicit: same-40-pair approximation errors and
  20,000 paired resamples using all 89 families, alongside target/control overlap
  diagnostics. Do not confuse these sensitivity summaries with new model results.
- New interventions: 8,188 missing F + 176,820 position-profile forwards + 74,760
  attention/value factorial forwards + 3,560 reverse forwards = 263,328. Reuse
  valid saved F elsewhere. Original prompt positions only; shared answer-prefix
  positions remain live. Capture actual SDPA Q/K after QK norm/RoPE and actual V;
  preserve installed forward implementation. OLMo2 shared post-attention norm
  does not become an additive per-head logit attribution.
- Cluster upload extracts into isolated `pilot_v2/stage3_v1/`, sourcing parent
  `runtime.sh`. First run `stage3_v1_gate` with `--gate-only`, inspect its measured
  costing, then `stage3_v1`. Two GPUs per replica; support 2/4/6. Resume requires
  identical inputs/code/allocation, retains checksummed pair/head-batch and prompt
  checkpoints, and reruns gates. Never remove a lock while its recorded job runs.
- Local implementation tests cover real tiny random OLMo2 SDPA hooks, QK/RoPE
  reconstruction, F/shared-prefix handling, projection weights, the real saved
  tokenizer/prompts, contextual RI and causal diagnostics, plus resampling and
  checkpoint integrity. All 12 tests passed, including complete CPU reporting
  from synthetic fixtures and rejection of missing required artifacts. They do
  not replace the pinned-7B GPU gate.
- The five final Stage-2 questions: 1–3 are addressed by Stage 3; 4–5 require
  Stage 4. Mechanism labels in older reports/handoff remain hypotheses.

## UPDATE 20 September 2026 (supersedes the "Immediate next task" below)

Completed since the 18 September snapshot:
- Coverage audit of stage2_v1 against the new Stage-1 candidates: `pilot_v2/stage2_coverage_audit_v2.py`, outputs `results/stage2_v2_coverage/`. Groups: A = 59 test-only RI candidates; B = 19 final-position test-QK heads (L31H21 excluded: demo sources only); C = top-25 |exact| on the 40 common pairs (all already in D); D = 92 v1 extension heads. Missing Scope-P coverage: 56 heads x 138 pairs.
- Methodology Section 5 rewritten: `חומר כתוב/Single_Hop_Methodology_section5_Stage2_updated.tex` (fragment to paste; not merged locally). Scope T (test block) was rejected as equivalent to Scope P under causal attention (prefix identical: verified, activation diff exactly 0). Scope F = final position only.
- GPU run `stage2_v2_extension` (job 912879, 32m51s, exit 0, s-004): Scope P for 56 heads on 138 pairs; Scope F for 69 heads (A∪B) on 178 pairs; separate logits stored. Gates passed on 3 replicas; v1 replication max deviation 0.0. Code: `pilot_v2/stage2_extension_v2.py`, tests, verify script, sbatch/submit, `RUNBOOK_stage2_v2_extension.md`. Results local: `results/stage2_v2_extension/` (+ tar.gz).
- Analysis: `analyze_stage2_v2_results.py` -> `results/stage2_v2_analysis/` (head_table_v2.csv, ri_vs_causal_v2.csv, summary.json, 5 figures). Report `חומר כתוב/Stage2_Results_and_Analysis.tex` updated IN PLACE (new sections 6-7, verdicts under the new definition, updated interpretation/conclusion); compiles with pdfLaTeX (11 pages; lmodern/microtype not available locally, otherwise clean). Figures `stage2_v2_fig1..5_*.png` copied into `חומר כתוב/`. PDF not regenerated by the user yet.

Main findings (exact, 178 pairs, family-level):
- Stage-1 priority heads causally negligible: L11H4 +0.004 [-0.000,0.008]; L17H5 -0.013; L9H16 +0.095 [0.039,0.155] with Scope-F ~0 (acts at earlier positions). Largest of the 46 newly measured A heads: L8H15 +0.279. 8 A heads with |I|>0.3 were all already in D via attribution.
- New name-gap statistic NEGATIVELY rank-correlated with exact importance: -0.252 (CI [-0.289,-0.169]) on 40 common pairs, -0.302 within-layer, -0.537 on self events; control-name mean +0.143. Hypothesis (untested): causal heads copy names; the name-gap control penalizes copying. Test in Stage 3: weight copying score W_E W_OV W_U on name tokens; current-token score at self events.
- Scope F/P ratio bimodal by layer: layers 6-11 (L6H24, L8H15, L9H16, L10H16, L11H15) ratio ~0 (writers at test positions); layers 16-26 (L18H19, L22H5, L16H21, L16H1, L17H24; negative: L23H15 -0.95, L19H16 -0.82, L26H23 -0.37) ratio 0.91-1.09 (readers at the colon). Logit decomposition: effect = promotion of the corrupted answer (e.g. L18H19 +2.88 vs clean -0.20); negative heads suppress the retrieved answer (negative name movers).
- Verdicts under new definition: incomplete supported (88/102 top-decile heads outside A; 45 QK-stage misses); misleading not supported by the letter (median A 0.0033 vs random 0.0002, CI excludes 0) but negligible in magnitude; sound weak/uneven (+0.035 layer-matched, 12/28 layers positive).

Open items / next: (1) Scope F for L17H1, L27H6, L18H18 (not in A∪B, so not measured); per-position profiles for writer candidates; (2) copying-score test of the negative-correlation hypothesis; (3) path patching writer->reader; (4) Git not yet synced (remote last commit 13 Sept; sync instructions given; user has not confirmed); (5) category-1 cleanup files (pycache, tmp/stage2_test_deps, parallel-1.py, pilot-1.py, pilot_v3/upload, pilot_validation.json) to be deleted by the user; do not delete phase_a_methodology.tex (pre-registered plan record).
- User preference recorded: separate Hebrew prose from English terms (code spans / separate lines); keep answers short.


Snapshot: 18 September 2026, after the test-only Stage-1 RI extension and the in-place Overleaf results-report update. This supersedes the 13 September handoff's pending-job status. Stage 2 job 888691 and RI-extension job 905839 completed; their results are local. Read the artifacts before making scientific claims.

**Immediate next task:** decide whether the existing Stage-2 measurements already cover the new Stage-1 candidates, or whether a targeted additional calculation is needed. Do NOT assume that a changed RI shortlist requires rerunning Stage 2. Conversely, do NOT assume all new candidates have exact measurements on all families. Audit coverage first, then discuss the necessary update to the Stage-2 results report with the user. This coverage comparison has NOT yet been performed.

### 21 September — corrections and Stage-3 plan
- Two analysis errors, flagged in the Stage-3 plan, fixed in `analyze_stage2_v2_results.py`: (1) the "misleading" median-difference bootstrap resampled 20 family indices (common subset) instead of the 89 families of the 178-pair data; corrected CI is [0.0009, 0.0067] (report updated in place; conclusion unchanged). (2) `residual_40` in `stage2_v2_coverage/coverage_v2.csv` mixed the 178-pair attribution mean with the 40-pair exact mean; renamed `residual_cross_sample_178attr_vs_40exact`, and a same-pair residual `residual_40_same_pairs` (attribution and exact on the identical 40 pairs) was added to `stage2_v2_analysis/head_table_v2.csv`. Same-pair residuals: L17H1 2.24, L27H6 0.51, L18H19 0.28, L16H21 0.22, L16H1 0.04 — the first-order estimate under-shoots the strongest heads, as the methodology predicted.
- Name-control definition verified in `ri_test_audit.py`: controls = tails (MOTHER names) of the other test facts, fully visible at j. Child names are controls only when they are also mothers in another fact (chain structure: "X is the mother of Y. Z is the mother of X."). The copy hypothesis therefore predicts negative gaps specifically at self events on chain-middle names (child in one fact, mother/control in the previous), and positive target scores at events that process a target name's second mention. Not yet tested event-by-event (planned as Stage-3 "Copying and Token-Preference Diagnostic", partition by current-token role).
- Stage 3/4 redesigned by the user: Stage 3 = head-level characterization (fixed 105-head inventory in `results/stage3_plan/`, five groups + moderate supplement; core profiles: evidence review, causal position profile + reverse patching, relation-tracking attention profile, contextual output profile; four planned diagnostics: copying/token preference, matched contextual RI, attention–value intervention, repeated-sequence/retrieval fingerprints; optional Patchscopes-style readout). Stage 4 = circuits: paths, group interventions, faithfulness, validation families. Plan: `חומר כתוב/new_stage_3_experiment.pdf` (source `Stage3_Head_Level_Characterization_revised.tex`); earlier draft `Stage3 new idea BETA.pdf`. No Stage-3 measurements yet.

## 1. Working with the researcher

Act as a rigorous NLP research collaborator and mentor, with strong mathematical grounding in transformers, probability, statistics, causal inference, experimental design and mechanistic interpretability. Demonstrate expertise through correct analysis, not claims of exceptional authority. The user wants to become an outstanding independent NLP researcher: help them develop judgment, technical skill and the ability to challenge interpretations.

- Speak Hebrew in conversation. Write research documents and Overleaf LaTeX in simple, professional, concise English. Avoid long narratives and unnecessary jargon.
- Keep explanations direct; the user explicitly finds long responses tiring. Separate Hebrew prose from English formulas/tables where possible. In experiment reports briefly remind the reader what was tested, why, and why that comparison was chosen. Expand only where it changes the interpretation.
- Edit existing Overleaf sources in place when requested, rather than replacing the report with a newly structured document. Preserve authors, historical results and meaningful section structure. Do not confuse a PDF's user-facing name with the local source filename.
- Explain a metric by stating its inputs, operation, denominator, meaning and limitations. Use a concrete prompt/token example when helpful. Define terms rather than inventing labels. Distinguish characters, tokens, entities and positions.
- Be direct about what was actually measured. Separate observations, interpretations, hypotheses and causal evidence. Do not infer a circuit from several important heads or from correlations between their outputs.
- Welcome the user's objections. Correct yourself explicitly when needed; do not defend an earlier implementation merely because you wrote it, or agree without checking.
- The user knows the course material in the parent NLP directory and has studied the core Anthropic papers. Do not repeatedly explain basics, but develop mathematical arguments when they answer the question.
- Work autonomously on authorized tasks. Avoid repeated permission requests and unnecessary reruns. For consequential methodological changes, make the change explicit and preserve comparison with the previous specification.
- Give copyable commands, clearly separating local PowerShell from remote Bash. Never put markdown escape backslashes before underscores or @ inside commands.
- Inspect actual code, prompts, results and provenance before reporting conclusions. A saved assistant report is a secondary source. Validate against raw artifacts where necessary.
- Show progress during longer work. Report what changed, what was tested, what remains unverified, and the next concrete action.
- Keep artifacts under this project. Preserve existing results and unrelated edits. Do not download models, submit extra jobs, rewrite SSH trust, or publish messages on the user's behalf without authorization for that action.

## 2. Source hierarchy and reading order

All paths below are relative to:
`C:/Users/User/OneDrive/Documents/computer science/4B/NLP/final proj/project`

Start with these, not the root README:

1. `חומר כתוב/Stage1_Results_and_Analysis_updated.tex` — CURRENT concise Overleaf report, updated in place on 18 September. Same six main sections and authors; Section 1 adds extension validation, Section 5 now has four subsections covering the new experiment/results/diagnostics, and Section 6 updates the conclusion. This is the source corresponding to the user's `pahse_1___results_and_analysis` report. The local PDF with that name and older `.tex` are not the updated authority. Structural LaTeX checks passed; no local compiler was available and no new PDF was compiled.
2. `חומר כתוב/Stage1_RI_Test_Extension_Results_and_Analysis.md` — detailed supporting analysis, including candidate qualifications and raw-OV invariance. `results/ri_test_v2_analysis/post_selection_audit.json` contains the calculations. Neither replaces the concise Overleaf report.
3. `חומר כתוב/Single_Hop_Methodology_section4_RI_extension.tex` — replacement methodology Section 4 for the new Stage-1 design. Read alongside `חומר כתוב/Single_Hop_Methodology_stage1_updated.tex` for the overall design. Do not assume the fragment was merged into the full local document. The full document still contains a stale Stage-3 SVD statement (see Section 10 below).
4. `חומר כתוב/Stage2_Results_and_Analysis.tex`, then `pilot_v2/RUNBOOK_stage2_v1.md`, `stage2_run.py`, `stage2_common.py`, `stage2_engine.py`, and `analyze_stage2_results.py`. This is the existing source corresponding to `phase_2__Results_and_Analysis.pdf`. It predates the new RI candidate analysis and has NOT been updated for it.
5. Actual Stage-2 outputs under `results/stage2_v1/` and derived analysis under `results/stage2_v1_analysis/`; actual extension outputs under `results/ri_test_v2/`, extension implementation `pilot_v2/ri_test_audit.py`, and independent analysis `pilot_v2/analyze_ri_test_results.py`.
6. `pilot_v2/STAGE1_AUDIT_README.md` and original Stage-1 code/results below for the historical definitions/calibration. For earlier history: `research_history_overleaf_en.tex`, `חומר כתוב/NLP___current_position.pdf`, `pilot_v3/README_HE.md`, and `pilot_v2/README_OLMO2.md`.

The root `README.md` is stale: it still describes Pythia and an earlier multi-hop behavioral pilot. Do not treat it as the current model/task specification. No root/project `AGENTS.md` was found during this handoff; inspect any instructions supplied by the new session normally. This file supplements rather than replaces them.

The original proposal and course instructions are in the parent directory: `../NLP Projects_Proposal.pdf`, `../NLP_course_2025b___project_guidelines.pdf`. The parent also has course PDFs and `../papers/`. Refer to these instead of reconstructing their contents from this handoff.

## 3. Research objective and literature

Original direction: identify semantic induction circuits involved in double-hop reasoning. Following discussion and instructor approval, the current priority is **single-hop first**: audit whether heads selected by the semantic induction head (SIH) definition actually carry the model's in-context relational ability, whether the definition misses important heads, and whether the mechanism requires a larger circuit. Two-hop composition is a later extension, not the experiment currently running.

Core sources (available locally where noted):
- Ren et al., *Identifying Semantic Induction Heads to Understand In-Context Learning*: `../papers/Identifying Semantic Induction Heads to Understand In-Context Learning.pdf`; https://aclanthology.org/2024.findings-acl.412.pdf
- Olsson et al., *In-context Learning and Induction Heads*: local paper; https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Elhage et al., *A Mathematical Framework for Transformer Circuits*: https://transformer-circuits.pub/2021/framework/index.html
- Local supporting papers include *Towards Best Practices of Activation Patching in Language Models*, *Towards Automated Circuit Discovery*, *Position-aware Automatic Circuit Discovery*, *Have Faith in Faithfulness*, *Function Vectors*, *Patchscopes*, and the IOI circuit paper. Read the relevant paper before attributing a specific method or claim to it.

Do not equate Ren-style SIH selection with verification of the classical induction circuit. An intervention can support a causal effect within a specified computational experiment; it does not automatically establish semantic specificity, a complete circuit, or a universal causal account.

## 4. History in brief

The project began with the proposal, literature study and methodological planning. Behavioral pilots used Pythia-1B and then Pythia-6.9B to test whether a tractable model could perform the task before attempting internal interpretation.

The dataset and scoring were revised repeatedly: yes/no prompts had strong answer biases; names and 0/4/12-shot conditions were explored; name completion exposed ambiguity of intermediate versus final entities, mention-frequency shortcuts and option-order effects. An explicit graph-navigation wording failed even on simple checks. Natural-language named relations and balanced mentions were developed, with counterfactual/broken-link checks and tokenizer audits. Poor results were not attributed automatically to model size. Exact earlier numbers belong in the saved behavioral reports, not in recollection.

OLMo-2-1124-7B was then adopted. After the single-hop scope decision, Stage 0 established hooks, metric checks and cost estimates; Stage 1 scanned all heads and was revised to collect full dominance distributions, event-level metadata and matched-target calibration. Stage 2 subsequently completed and was analyzed. The user then revisited Stage 1 in depth: repeated demonstrations, fact-position denominators, restrictive matched-name controls, first/last token choice and semantic specificity motivated a broader test-only RI extension. That extension is now complete and analyzed, and its results have been integrated into the existing Stage-1 Overleaf report. The next question is how this changes the Stage-2 analysis, not starting the project again.

## 5. Current substrate

Model: base `allenai/OLMo-2-1124-7B`, revision `7df9a82518afdecae4e8c026b27adccc8c1f0032`, pinned in `pilot_v2/model_lock_olmo2.json`. 32 layers x 32 heads = 1,024 heads. HF Transformers/PyTorch, fp16, model sharding across GPUs; CPU offload rejected by the loader. No fine-tuning in these stages.

Single-hop dataset: generator `pilot_v3/build_data_singlehop.py`, local data `pilot_v3/data_singlehop/`; cluster-facing input `pilot_v2/data/singlehop_v1_4shot.jsonl`. Inspect the file for exact templates and annotations.

There were 200 families; behavioral eligibility retained 176. Discovery uses 89 even-ID families; 87 odd-ID validation families remain reserved for later validation. Stage 1: 89 x 3 variants (base/corrupted/reorder) x 2 orders = 534 prompts, four ICL demonstrations each. Stage 2: clean/corrupted pairs, both orders = 178 discovery pairs. Shared demonstration prefixes are intentional controls within families, not independent observations. Do not select heads or tune thresholds on held-out validation families.

Baseline report: `results/singlehop_baseline/singlehop_baseline_report.md`. Stage-1 provenance pins the data and baseline hashes.

## 6. Stage 1: precise definition and results

Authoritative run: `results/stage1_v4_review/stage1_v4_calibrated/`, archive `results/stage1_v4_calibrated_results.tar.gz`. Read its config/provenance, `head_stats.csv`, `metric_specs.json`, `audit_prompts.jsonl.gz`, `ri_events.jsonl.gz` and calibration outputs as needed.

Relevant code: `pilot_v2/stage1_scan.py`, `stage1_audit.py`, `run_stage1_audit.py`, `analyze_stage1_audit.py`, and calibration helpers referenced by those files.

QK condition: at current position j, maximal attention must point to annotated SOURCE position s and exceed the strongest competitor by tau=2.2. Source anchor = last token of the source name. In the kinship task, the child is the source and mother is the target. Do not reverse them. Eligible positions include j=s when both names are visible.

OV statistic: raw current-token embedding e_j projected through head W_V W_O and unembedding; normalize using visible-context tokens as implemented. First target token is primary, last token is sensitivity. This is NOT the actual contextual head output and NOT its final-logit causal effect. Strength is conditional on scored passes; frequency uses fact-position opportunities, not prompt count.

Historical pooled selection chose L3H11 and L9H22. That old cross-head control-mean percentile is descriptive, not calibrated significance:
- L3H11: 149 passes, pooled RI .2431; 144 demo passes (.2502), 5 test passes (.0384). All attend one token backward from the same period/newline token. Events span 17 families; test events only one family.
- L9H22: 86 passes, pooled RI .3739; 72 demo passes (.4453), 14 test passes (.00674). All are self-attention. Events span 15 families; test events span four.
- Demonstrations account for >99% of summed first-target scores for each. These are static raw-embedding preferences conditioned on QK events; earlier layers might still write relational information into those positions. Neither a semantic mechanism nor its absence has been established.

Matched-target calibration is a DIFFERENT, restricted statistic: matched test-block events, both candidate mentions visible, same token length, distinct first token. Minimum 50 events across 10 families. Equal active-family means, 100,000 family-consistent target/control swaps; Holm across 1,024. 80 heads eligible, 944 insufficient support; neither pooled candidate had any eligible matched events. No head survived correction. L16H4 had smallest raw p=.00581 (87 events, 40 families, target .01471 vs control .00930). Insufficient data is not a negative causal result; exchangeability assumptions still matter.

Full dominance collection: 1,455,940 ratios, p95=4.482, versus old restricted-sample 3.658. Tau is a selection threshold, NOT temperature. Both event weighting and dataset/tokenization affect it; it is not evidence of a purely model-specific difference. Filtering saved events at 4.482 retained 72,797 passes; no new model run or null recalibration for that sensitivity.

Stage-1 loading took 38m44s, scan 12.5min, calibration ~7s. Earlier long run was waiting on I/O (`D`, `folio_wait_bit_common`, growing read_bytes), not established to be an infinite loop.

## 7. Independent saved-event analysis — completed

Code: `pilot_v2/stage1_anatomy.py`. Output: `results/stage1_v4_review/anatomy_all_heads/` (`head_anatomy.csv`, `summary.md`, provenance). No GPU forward passes. All heads' event/denominator counts reconcile with the scan.

Partitioned QK-passing events by distance (self / previous / >=2), fact block (demo/test), current-position block (demo/test/final bare-prompt token). Twenty heads emerged with at least one qualifying event from the final prompt token: they were NOT a preselected set of 20. 748 such events total, 744 to test-fact sources and 4 to demo sources. Neither L3H11 nor L9H22 appears in that set.

Examples, final-prompt attention to test-fact source satisfying BOTH argmax and dominance>2.2:
- L16H21: 230 events, 79 families.
- L16H1: 223 events, 76 families.
- L16H4: 63 events, 32 families.

These counts are not correct-answer counts, independent trials, proof of query-relevant fact selection, or evidence of target promotion. No common semantic rule for all 20 was established. The final bare-prompt position precedes any common answer prefix used by Stage 2. This analysis did not change the already submitted Stage-2 shortlist rule.

## 8. Stage 2 — completed, existing coverage must be reconciled with new candidates

User supplied `sacct`: job **888691**, `stage2`, COMPLETED in **01:14:42**, exit **0:0**, on s-004 with six GPUs. Local artifacts: `results/stage2_v1/`, archive `results/stage2_v1_results.tar.gz`. Its saved `summary.json` confirms complete=true, 178 pairs, 40 all-head exact pairs, **92 full-coverage extension heads**, first-order estimate and estimate_validated=true. The existing results report records all three replica gates passed and raw-array/provenance checks. This handoff update inspected the summary/quality metadata, not a fresh full-array rerun of those checks.

Code: `pilot_v2/stage2_run.py` (controller/workers), `stage2_engine.py` (metrics/hooks), `stage2_common.py` (pairing/work plan), `stage2.sbatch`, `submit_stage2.sh`, `test_stage2.py`. Read these before analyzing output. The README provides all commands; do not substitute the older methodology's Stage-2 description for actual code.

Six GPUs -> three persistent replicas, two GPUs each. Whole-family scheduling weighted by expected exact work. Initial loading staggered to reduce shared-storage contention. Every replica must pass its gate before any begins the full experiment; failures stop workers. Models remain loaded between phases.

Gate: identity/hash and token-alignment checks, behavioral replication, self-patch, self-attribution, full corrupted embedding replacement, and six-head exact/gradient previews on four families/both orders including common answer prefixes. Does NOT gate on scientific usefulness, effect sign or estimate agreement.

Metric: clean-answer logit minus corrupted-answer logit at FIRST DIVERGENT answer token, conditioned on shared answer prefix. Fix clean-gold sign even on corrupted input. Head patch replaces the o_proj input slice at ALL ORIGINAL PROMPT positions using corrupted donor activations. Common answer-prefix positions remain live/recomputed. Negative delta means movement toward corrupted answer. Not loss difference, sequence likelihood, or single-position patching.

Scope:
1. First-order gradient attribution for all 1,024 heads on all 178 pairs.
2. Exact single-head patching for all 1,024 heads on 40 pairs (20 randomly sampled whole families, both orders). NOT all-head exact patching on all 178.
3. If signed-mean head Spearman against exact results is <.7 or undefined, five-step midpoint INPUT-EMBEDDING IG estimate on all 178. It is an approximation contracting gradients with fixed head deltas, not an exact head-path integral. If agreement still fails, mark estimate unvalidated; exact results remain useful.
4. Exact extension to all pairs for union of top25 supported first-target RI, top25 last-target RI, top25 absolute mean estimated effects, L3H11/L9H22/L16H4, plus25 random controls. Reuse the 40-pair exact results.

Local validation before submission: all 16 CPU tests passed, including exact/Taylor/IG agreement on a linear toy model, nonlinear finite differences, prefix patch scope, self controls and scheduling. GPU gates subsequently passed as recorded in the results. Test-only CPU PyTorch under `tmp/stage2_test_deps` is not the Slurm runtime and must not enter Git/upload bundles.

Saved outputs include manifest, gate files, worker logs, per-pair NPZ, quality reports, `shortlist.json`, `head_effects.csv` and final `summary.json`. Signed mean-head Spearman was **0.945364891**, absolute mean-head correlation **0.920853814**, pair-head correlation **0.902767406**. First-order passed the prespecified agreement check, so the IG fallback was NOT run. Global agreement does not guarantee each new candidate's approximation accuracy.

Existing report/analysis: `חומר כתוב/Stage2_Results_and_Analysis.tex`, `analyze_stage2_results.py`, `results/stage2_v1_analysis/summary.json`, figure `results/stage2_v1_analysis/ri_vs_causal_effect.png` (the Overleaf source expects an image named `stage2_ri_vs_causal_effect.png`; check its local/upload copy). The existing comparison uses historical pooled RI; its reported RI/exact Spearman ~0.135 is NOT a result for the new test-only contrasts. Its earlier Stage-3 prioritization must be reassessed, not automatically overwritten or assumed validated for the new shortlist.

### 8A. Test-only Stage-1 extension: implementation and execution history

The user wanted Stage 1 to produce a strong, broad **RI-based discovery list**, keeping contextual/intervention methods for later stages. They explicitly approved test-only events, first/last anchors, general-word and alternative-name controls, fact/query distinction, position/distance separation, paired variants, and post-selection token/family concentration checks.

Code: `pilot_v2/ri_test_audit.py` (prepare/OV/analyze/run), `ri_test_audit.sbatch`, `submit_ri_test_audit.sh`, `ri_test_diagnostics.py`, `build_ri_test_bundle.py`, `pack_ri_test_results.sh`, `verify_ri_test_results.py`, `test_ri_test_audit.py`, `test_ri_test_diagnostics.py`. Code-only archives `ri_test_audit_v1_update.tar.gz` and `ri_test_audit_v2_update.tar.gz` deploy into versioned remote directories; do not confuse local flat code paths with the remote `ri_test_audit_v2/` import directory.

Preparation reuses original saved attention events and metadata. GPU work adds the missing alternative-name last-token scores and sampled word-control scores, while replicating saved scores; it loads weights but performs no full-model forwards. CPU analysis then summarizes and selects candidates. Do not describe this as a new attention scan or as entirely GPU-free.

Execution history:
- Job **899730**, v1, s-003, failed after 2h01m21s. Weight loading alone took ~1h57m. Error: `Previously scored event now has zero denominator`. Do not claim a proven root cause or silently relax RI math/tolerances.
- v2 added loading/environment/I/O diagnostics and concurrency protection. Job **905799**, s-004, failed immediately on the existing `.running.lock` while another job used the same output. This was a collision, not evidence that s-004 was broken. Never remove a lock without checking job liveness.
- Job **905807**, s-002, failed during CUDA initialization (`CUDA unknown error`) after 5m06s, before model load completed. The message suggests possible environment problems but does not prove the exact cause. No evidence here establishes that s-005 was repaired either.
- Job **905839**, s-004, completed in **47m06s**, exit **0:0**. Authoritative output `results/ri_test_v2/`; archive `results/ri_test_v2_results.tar.gz`. Model loading still took ~38m38s. Weights are on NFS (`netapp1:/Netapp5_yandex`); resource/node/cache/I/O differences can affect loading. Do not attribute success or slowness to a single unproven cause.
- The user wants short practical transfer commands, not unnecessary environment setup or checksum ceremonies. An unset `$PILOT_RUNS` and the nonexistent relative `storage/runs` previously caused tar failures. Use verified absolute storage paths when needed.

Frozen extension policy: tau=2.2; seed=20260916; up to three unique visible non-name word types; all other fully visible test fact-tail names; no equal-length exclusion; first/last target and control anchors; collisions retained as ties. Test-only excludes demo events, not demo context or normalization tokens. QK failures are skipped, not RI=0; valid scored zeros remain zeros. RI subtracts the mean, clamps max(0, ...), and normalizes over unique visible token IDs.

Conditional means: average within family, then equally over active families for that metric. Frequency includes zero-pass families. All-test includes final; all-facts includes query-fact. Do not present overlapping views as independent replications. Counts for target-only and name-comparison subsets can differ because controls must be fully visible.

Selection: union of top10 positive family means for target/name-gap/word-gap over two anchors, two fact scopes and all-test/final position scopes; minimum **20 events and 10 families for each corresponding metric**. Output **59 descriptive candidates**, with **36 entering at least one name-gap ranking**. This is NOT the earlier matched calibration (50 events, same token lengths, distinct first tokens, family swaps/Holm) and NOT a new multiplicity-corrected discovery claim.

### 8B. Completed local analysis and main findings

`pilot_v2/analyze_ri_test_results.py` was added for independent read-only verification/post-selection diagnostics; output `results/ri_test_v2_analysis/post_selection_audit.json`. It does not alter frozen source results or selection. It checks every event and all active-head aggregates and analyzes all 59 selected heads plus both historical heads, not just the reported examples.

Verified: **46,634/46,634 scored events**, 340 active heads, summaries for 1,024 heads; 237,685 replication checks, maximum absolute error **0.000195890665**, within unchanged atol=0.0002/rtol=0.002. Independent checks reproduced **8,160** target/name-gap/word-gap summary values and checked denominators, identities, ranks, collision ties and empty-head summaries. These are saved-calculation checks, not an independent new GPU replication.

Key all-facts/all-test **name-comparison-subset** results (equal-family means; events are comparisons, not wins):

| Head | Anchor | Events / families | Name gap | Interpretation |
|---|---|---:|---:|---|
| L11H4 | first | 94 / 39 | +0.006523 | Priority candidate; survives single-family and dominant-token removal; 78 nonlocal events, no current-target-name overlap. Query subset 53 / 29, +0.007080. Last-token gap slightly negative. |
| L17H5 | first | 65 / 34 | +0.006059 | Priority candidate; all nonlocal, no current-target-name overlap. Query subset 32 / 20, +0.008183. Current `mother` token contributes 66.3% of target RI; last-token gap negative; descriptive bootstrap includes zero. |
| L9H16 | last | 343 / 87 | +0.007398 | Priority candidate; survives deletions, all nonlocal. But 299 events are reorder, 21 base, 23 corrupted; base gap slightly negative. Query subset 244 / 76, +0.006938. |
| L12H2 | first | 52 / 32 | +0.007156 | Secondary; query support only 7 events / 6 families. |
| L25H18 | last | 46 / 17 | +0.013071 | Secondary; query support spans only 9 families. |
| L23H10 | first | 60 / 24 | +0.021505 | Highest eligible first-token gap; 31 events process the target name itself. Removing these leaves +0.007177 on 29 / 17. Useful comparator, not automatically best semantic candidate. |

Additional comparators: L15H18 positive at both anchors but **104/110** comparisons process the target name itself and only one concerns the query fact; L6H2 last-gap **+0.006683**, 927 comparisons, all 89 families, but all self-attention. Do not reject self-attention as logically incapable of contributing to semantics; distinguish what raw RI measures from contextual computation.

The historical heads remain low-support: L3H11 five test events in one family (three with name controls); L9H22 fourteen in four (nine with controls, first gap ~+0.000018). The new interpretive priorities are NOT a replacement calibrated rule and are not proof of semantics.

Paired diagnostics: L11H4 has 21 jointly passing base/corrupted source-order pairs across 18 families, all targets changed, but only five families positive in both versions; only five jointly passing reorder pairs. L17H5 has none jointly passing under reorder. Positive pooled variant averages do not prove paired robustness. All-test paired means may involve different positions. Leave-family-out and dominant-token removal diagnose concentration, not significance. 5,000 family-bootstrap resamples are descriptive and not selection-adjusted intervals.

### 8C. Answer-boundary interpretation and the last clarification to the user

Among 744 final-position test QK events, **718 are query-relevant**. These are head-prompt events, not distinct prompts. The query-final frequency denominator is exactly 534 (one opportunity/prompt), unlike general fact-position frequency.
- L16H21: 222 query events / 77 families (41.6%); 222 of its 230 final test events are query-relevant. First name gap -0.000417; last +0.003350. Removing target suffix `la` reverses last gap to -0.006410; it supplies 55.9% of target RI.
- L16H1: 223 / 76 (41.8%); both anchor name gaps negative. Useful QK-versus-OV contrast.
- L16H24: 68 / 35; last gap +0.003586, reverses after removing `na` (71.9% target-RI share).
- L16H4's small first-token final advantage and L16H31's last-token final advantage fail single-family deletion.

**Crucial limitation:** raw-current-token OV, not contextual output. At final position the input token is always `:`. Same head + same current-token ID + same visible token-ID set implies the same RI score for each candidate, even if mothers are swapped. Directly verified for all 49 jointly passing L16H21 base/corrupted query pairs: old-vs-new correct-name contrasts negate exactly at both anchors, with no strict correct win in both versions. This is an algebraic limitation of the measurement, not a GPU bug or evidence the model cannot follow a swap. Contextual QK can still change.

The user asked whether this issue disappears earlier in the test. Clarified: the constant colon is final-specific, but the invariance applies at **any** position with identical current token and visible token set. Earlier tokens can vary and produce different scores; that alone does not show adaptation to the relation. We do measure earlier events, but the reported all-test aggregate includes final; do not claim an earlier-only aggregate was the main table. The saved position metadata permits a separate earlier-only analysis if useful.

## 9. Remote environment, Git and operations

Cluster directory: `/home/yandex/DLWorkShop2025b/maximg/pilot_v2`.
Storage: `/home/yandex/DLWorkShop2025b/maximg/storage`; `source runtime.sh` sets PILOT_RUNS and reuses the working CUDA/Python package directories from the old pythia_pilot. Do not casually upgrade packages or overwrite runtime.sh.

Resources: partition studentkillable, account gpu-students. The current local RI sbatch file excludes s-002,s-005 AND pins s-004; the user previously also requested an unpinned option for the first available suitable node. Inspect wrapper/CLI overrides before giving commands; do not silently require s-004 for every future task. The failed 905807 run shows s-002 was not usable for that CUDA job on 17 September, not its perpetual status. Maximum previously requested for Stage 2: six GPUs. Cluster access/storage exist; actual quotas and present deadline are not known. Earlier three-week deadline estimate is historical, not a current countdown.

SSH hostname `slurm-client.cs.tau.ac.il` repeatedly presented differing keys. We used a direct endpoint verified via the user's already connected session:
- c-002 at `132.67.130.126`.
- ED25519 fingerprint `SHA256:L711K3bD0LdigTycyRlvdk8tutdEuD5AejKRKXxn7mY`.
- Local known-hosts file `tmp/slurm_c002_known_hosts`.
- Full safe upload/download commands in pilot_v2/RUNBOOK_stage2_v1.md. Keep strict host checking. Do not accept a newly changed key merely because the user needs results quickly. User handles interactive authentication; do not assume this assistant has SSH access.

Update bundle: `pilot_v2/stage2_v1_update.tar.gz`, code-only, verified archive members/LF. User already submitted Stage 2 from it; no need to re-upload unless actually changing code for a subsequent run.

Git: main branch, origin `https://github.com/Maxgolu/NLP_proj.git`. We prepared explicit add/commit/push commands but did NOT execute commit/push. User may since have done so: inspect fresh status. Pre-existing unrelated edits included deletion of `research_history_overleaf.tex`, modification of the methodology PDF, and untracked reports/results. Do not stage everything or revert those. Root README remains outdated.

## 10. Next actions — audit Stage-2 coverage before deciding on reruns

The user explicitly wants to resolve whether the updated Stage-1 candidate list requires new Stage-2 computation, or whether existing results suffice and only the Stage-2 analysis/report needs updating. **Do not ask them to download the completed runs again. Do not submit a new job just because the candidate list changed.** This handoff task did not perform the candidate-overlap analysis or modify Stage-2 code/results/report.

1. Read the updated Stage-1 Overleaf report and the actual extension policy/candidates, then the existing Stage-2 report, implementation, manifest, shortlist and per-pair outputs. Verify identity/coverage as needed. Use the full 59-head discovery union and label the smaller interpretive priority/comparator groups separately.
2. Build a head-by-head coverage table: new RI selection reason/anchor/scope; first-order coverage on 178 pairs; exact coverage on the common 40 pairs; membership in the old 92-head full-exact extension; actual available exact pair/family counts. Decode flattened head IDs consistently (32 heads/layer, zero-based) and confirm against implementation. Do not infer measured coverage solely from a candidate being mentioned in an old report.
3. Distinguish conclusions already answerable locally from genuinely missing measurements. Every newly selected head should already have first-order estimates on all pairs and exact estimates on the common 40-pair subset under the implemented design; verify actual files. Some may also have exact results on all 178 pairs. If broader exact coverage is scientifically needed for heads outside the old extension, discuss a **targeted extension of missing head-pair measurements**, not an automatic full rerun. Keep the original run immutable; changed-shortlist computation is not ordinary identical-config resume.
4. Reanalyze correlations/comparisons with the **new** RI definitions, respecting conditional support, anchors, query/all-facts and positions. Keep historical pooled-RI comparisons explicitly labeled rather than silently replacing their meaning. Use common exact subsets for fair all-head comparisons; do not mix 20-family and 89-family exact means as if identically supported. Aggregate at family level. Selection and post-selection inference remain exploratory.
5. Report a concise evidence-backed recommendation to the user: report-only/local reanalysis sufficient, or precise missing measurements and why they matter. Then update `חומר כתוב/Stage2_Results_and_Analysis.tex` in place when proceeding with the agreed scope, along with any affected figure. The user calls this report `phase_2__Results_and_Analysis`; its existing PDF may be stale. Do not rewrite from scratch or silently turn partial-coverage estimates into full-coverage exact claims.

Stage-2 patches replace a head at **all original prompt positions**, including demonstrations; the new Stage-1 selection uses **test events**. Candidate selection alone does not change this intervention scope. The outcome is the first divergent answer token, which may follow a shared prefix, whereas Stage-1 final events occur at the bare prompt boundary. These distinctions can limit interpretation without invalidating the existing run. Explain if a new scientific question would require a genuinely different intervention.

Do not call high-RI/low-effect heads false positives immediately: position scope, metric, redundancy and nonlinear interactions matter. Low RI does not establish lack of semantic processing. A head list is not an edge-validated circuit. Keep the 87 validation families held out from tuning. Stage 3 and later faithfulness/generalization checks follow only after this Stage-2 reconciliation; two-hop/checkpoint extensions come later if justified.

### Stage-3 methodology correction discussed, not yet a completed experiment

The user raised the incorrect planned phrase ``SVD of W_OV^A W_QK^B`` and RoPE/QK-normalization complications. A replacement for methodology Section 6.5 was provided in chat for the user to paste. The full local `Single_Hop_Methodology_stage1_updated.tex` still contains the old writer-reader-product SVD wording, so do NOT assume the local document was patched or that an experiment was implemented.

Design direction to verify against the original sources before implementation: decompose the **writer's OV matrix itself** into rank-one components, then evaluate components against reader channels, rather than presenting an SVD of the writer-reader product as the referenced method. A global static QK-composition interpretation is problematic with position-dependent RoPE and input-dependent QK normalization in OLMo-2. Contextual/position-aware measurements and causal path/component interventions should carry the interpretation; any static composition score is a limited descriptive diagnostic. The exact chat replacement may need to be recovered from the user if they want that wording preserved. No Stage-3 run or new validated decomposition is claimed here.

Similarly, discussion of Ren et al.'s threshold usage in different paper sections is not permission to assert that the paper is internally inconsistent without checking the source. The implemented RI skip/clamp/unique-token rules are recorded above; keep implementation facts separate from literature interpretation.

### Local workflow notes

Bundled Python used successfully: `C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`. Source RI results were not modified by the post-selection analysis. No new GPU submission, commit or push occurred during the latest local analysis/report/handoff updates. Git status can report dubious ownership under the sandbox account; do not change global trust configuration merely to inspect this report task. The Stage-1 `.tex` passed brace/environment checks, but no local `pdflatex`/`tectonic` was available. Do not claim PDF compilation or visual verification.

The new assistant should acknowledge what it has actually read and any gaps, summarize the current research state briefly, and continue mentoring/research with the user. Do not claim complete project mastery after reading only this handoff.
