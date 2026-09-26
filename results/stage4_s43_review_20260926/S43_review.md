# S4.3 independent review (26 September 2026)

Scope: the S4.3 package received from Daniella Simonovsky on 26 September and integrated
under `results/stage4_s43_all_v1/`, `results/stage4_s43_inputs_v1/`,
`results/stage4_s43_extension_inputs_v1/`, `pilot_v2/s43_*.py` and
`חומר כתוב/Stage4_3_Methodology_and_Results.pdf` / `Stage4_3_Summary.pdf`.
The check is a CPU re-analysis of the released per-pair tables and a spot check of raw
chunks; no GPU work was repeated. Script: `verify_s43.py` (this folder).

## What was verified

- Frozen inputs: `stage4_s43_inputs_v1/pairs.jsonl.gz` and the recovered Stage-3 pairs /
  Stage-1 prompts are byte-identical (SHA-256) to `results/stage3_inputs_v1/pairs.jsonl.gz`
  and `results/ri_test_v2/prompts.jsonl.gz`. The four engine files the S4.3 code imports
  (`s42_engine.py`, `stage4_engine.py`, `stage1_scan.py`, `stage3_common.py`) are
  byte-identical to ours.
- Extension table (`extension_analysis_v1/pair_order_matched.csv`): 5,696 rows = 16 selected
  configurations x 178 pairs x 2 directions; `raw - direct == increment` holds to 1e-16.
- Report Table 2 (16 routes): noising and restoration family means, coherent /
  heterogeneous / retained labels, family sign-match fractions and the L20H7 correlation
  (-0.527) all reproduce exactly from the per-pair table under the frozen rule.
- Common-20 reproduction: rerun means equal the initial-screen means to 1e-16.
- 69 selection-excluded families: 14/16 coherent in both directions; the same two routes
  (L20H7 full release, L26H23 block27 only) are heterogeneous-only; L20H7 -0.059 / -0.069,
  sign match 43.5%, r = -0.575 — as reported.
- Raw chunks: 800 records from 25 of the 178 production chunk files match the table
  exactly (the remaining chunks were not staged for this check; their `.ok` files carry
  SHA-256 and record counts).
- Record counts: 9,612 = 178 x 27 x 2; self gate 0.0 error; family-034 exception as
  described in the report (two pair ids, reference error 0.0; both pairs kept).

No numerical error was found in the report.

## Points the report does not state or under-reports

1. **All seven chain comparators are exactly zero in every pair.** For the chain routes
   (L17H1 or L15H25 as source, fact-site writes, Q of a later head as receiver) the
   "direct / no-release" comparator has no live intermediate head, so nothing carries the
   fact-position change to the colon and the receiver's query is unchanged: a structural
   null, consistent with the S4.1 rule that a direct cross-position Q route is a null. The
   matched increment therefore equals the raw chain effect. This is correct, but the
   report presents the subtraction as if it were informative for every route; the paper
   should say that for chains the reported number is the raw effect through the released
   heads.
2. **The direct prerequisites are results in their own right and are never tabulated.**
   On all 89 families, in both directions:
   - `L8H15 / query_sentence -> L18H18 / V`: +0.465 (noise) / +0.442 (restore), sign
     agreement 100%, coherent; `-> L18H19 / V`: +0.460 / +0.446, coherent.
   - `L26H23 / colon -> logits (bypass)`: -0.172 / -0.173, coherent (93% agreement).
   - `L20H7 / colon -> logits (bypass)`: +0.039 / +0.038, not retained.
   L8H15 was not an S4.1 source, so the two L8H15 -> layer-18 V routes are new retained
   routes, larger than L8H15's whole single-head importance (+0.28 on 178 pairs).
3. **The L8H15 "negative routes" are attenuations of a positive direct route.** The raw
   effects with MLP9 (or MLP8+MLP9) released are +0.29 to +0.33; the direct route is
   +0.44 to +0.47; the increment (-0.13 to -0.16) is the difference. The report gives only
   the increment and calls it an "opposing / attenuating dependency"; without the raw and
   direct values a reader may take the L8H15 routes to be negative. The paper should
   report raw, direct and increment together.
4. **Block releases contain the strongest reader.** The L26H23 "full release" opens
   MLP26, block27 and block28; block27 contains L27H6, the strongest answer-position head,
   and block27 alone carries most of the increment (-0.089 of -0.183). The report
   correctly refuses head attribution from a block release, but the natural hypothesis
   (L26H23's corrupted output changes L27H6's read at the colon) should be named as the
   candidate for any follow-up.
5. **Order sensitivity is general, not specific to L17H1.** Noising increments by prompt
   order differ by a factor of two for the largest chain (2.220 vs 1.050) and by 30–60%
   for several others (L26H23 full release -0.138 vs -0.228; L15H25 chain 0.178 vs 0.257;
   L26H23 block27 -0.058 vs -0.120). The report mentions L17H1 and "similar, smaller"
   cases; the paper's Limitations should say that S4.3 effects are order-averaged and
   order-sensitive throughout.
6. **No uncertainty intervals.** S4.1/S4.2 reported 20,000-draw family bootstrap intervals;
   S4.3 reports means, standard deviations and sign fractions only. For the paper, the
   same descriptive bootstrap should be added from `family_bidirectional.csv` so the
   stages are comparable.

## Reproducibility gaps (code not in the package)

The runners consume frozen input bundles but do not build them. Missing: the script or
notebook that derived `main_registry.jsonl` / `direct_comparators.jsonl` from
`next_stage_plan.json`; the smoke-run script that produced `smoke_v1/smoke_report.json`;
the analysis code that produced `analysis_v1/`, `selection.json` and
`extension_analysis_v1/`; the compatibility-diagnostic script behind
`diagnostics/a100_*_compatibility.json`. `verify_s43.py` here re-derives the extension
analysis; the initial-screen selection and the input construction are not yet
re-derivable from code.
