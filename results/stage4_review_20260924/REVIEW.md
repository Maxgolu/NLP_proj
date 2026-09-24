# S4.1 audit and interpretation — 24 September 2026

The completed stage4_all_v2 run is intact and reproducible. No additional GPU job
was submitted. All new computations in this review use saved records on CPU.

## Verified evidence

- Archive SHA-256 matches the downloaded sidecar.
- Executed code/bundle/input identities agree; fallback shared dependencies match
  their exact expected bytes in pilot_v2.
- All checksummed chunks and exact pair/configuration/direction IDs verified.
- 21 measurement workers and the initial gate worker pass their stored six-pair
  gates and tiny-model gates. All 2,856 recorded configuration self endpoints have
  zero drift. These stored checks do not constitute an independent GPU rerun.
- 70,408 records: 70,328 newly measured and 80 explicitly reused Scope-P rows.
- Seven phases' four CSV exports match CPU regeneration within 1e-12. Both
  interaction exports match; all six adaptive selection decisions reproduce.
- Endpoint effects independently recomputed from their answer logits.
- New family-level analyses separate the initial 20 and additional 69 families,
  noising/restoration, order and shared-prefix conditions. Held-out data unopened.

## Main interpretation

L17H1 has strong V routes into L18H18/L18H19, and opposing routes into several
other receivers. L15H25 reaches L16H1/L16H21 via V; those heads have colon Q routes
into L18H18/L18H19. Separately measured consecutive edges are not a tested chain.
The central layer-18 pair has downstream Q/KV, MLP and residual exits.

95/96 expanded noising routes and 93/96 restoration routes retain on 89 families;
94/96 and 92/96 retain on the additional 69. These are selected discovery
consistency checks, not independent confirmation. All 96 mean signs agree between
directions. Nineteen retained core routes remain unextended under the fixed cap.

All 55 controls have smaller mean absolute family effects than linked targets.
None of 38 random-receiver controls retains, but seven of 17 other-site controls
do. Controls are not norm-matched and do not establish exclusively query-fact use.
Strong route magnitudes change substantially with fact order. Eight of 40 available
KV/union interactions meet the original rule; signed averages can hide variation.

L13H18's Scope-P effect is supported, but none of its 138 tested localized direct
configurations retains. It is an explicit local-mediation candidate, not an absent
member. L24H19's colon effect and residual bypass add a distinct coverage finding.

## Prospective S4.2

See s42_proposal.json and report Section 7. Eight proposed groups require only 49
distinct colon node configurations (13 singles, 36 joint masks). Seven singles are
available in the bundled prior references; six require completed Stage-3 records.
With compatible singles reused, the base common-panel joint panel has 1,440 new
endpoints, excluding blocking/gates/baselines/sensitivity/extensions.

Six source-mask blocking patterns, unchanged RI31/reference cohorts and predefined
adaptive K selection are specified. G3 remains a targeted L26H31 backup test.
No S4.2 intervention code has been implemented, and no S4.2 results are claimed.

## Deliverables

- חומר כתוב/Stage4_Overleaf/main.tex (canonical report, portable assets beside it)
- חומר כתוב/Stage4_Experiment_Overleaf.zip
- output/pdf/Stage4_Experiment_Report.pdf
- verification.json, report_facts.json, reproduced/, and supplementary CSV files

The PDF compiled with Tectonic and was visually inspected after rendering all 13
pages. No overfull boxes, unresolved references or missing assets remain. Two
underfull-box warnings are harmless spacing warnings. Source and PDF hashes are
recorded in delivery_verification.json. Original GPU artifacts were not edited.
