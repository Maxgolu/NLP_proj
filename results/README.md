# results/ — map from result folders to the paper

Every number in the paper comes from a folder listed here. Naming: `<stage>_inputs_vN` = frozen,
content-addressed inputs (never edit); `<stage>_vN` / `<stage>_all_vN` = executed run (checksummed
chunks, gates, manifests); `<stage>_analysis_*` / `<stage>_review_*` = CPU re-analysis and audits.
Large raw archives (`*_results.tar.gz`) are kept beside the folders or in the archive; they are not
in git. Raw chunk folders of the largest runs are git-ignored.

| Paper section | Result folders | What they hold |
|---|---|---|
| §3 Setting; App. D (data) | `singlehop_baseline/`, `pilot_v3/data_singlehop/` (outside results), `completion_*`, `prompt_v3_*`, `full_4shot_*`, `full_sdpa_*`, `olmo2_vs_pythia/` | Behavioural baseline (working set 89/87 families), prompt-design and model-choice pilots |
| App. C (compute) | `*/runtime.json`, `*/manifest.json` in each run; `stage4_s43_all_v1/colab_setup_manifest.json` | Hardware, library and model-revision records per run |
| §5.1, App. A, App. E (Stage 1: RI) | `stage1_analysis/`, `stage1_v4_review/`, `ri_test_v2/` (candidates.json = the original 59-head selection), `ri_test_v2_analysis/`, `stage2_ri_extension_coverage/` | RI scan, calibration, matched-target test, 59-head test-only selection, post-selection audit |
| §5.2, App. F (Stage 2: causal map) | `stage2_v1/`, `stage2_v1_analysis/`, `stage2_v2_extension/`, `stage2_v2_analysis/` (head_table_v2.csv), `stage2_v2_coverage/` | Gradient screen, exact Scope-P/Scope-F patching, 25 strong heads, audit criteria |
| §5.3, App. G (Stage 3: head profiles) | `stage3_plan/`, `stage3_inputs_v1/`, `stage3_v1/` (analysis/), `stage3_review_20260922/`, `stage3_readout_inputs_v1/`, `stage3_readout_v1/`, `stage3_readout_review_20260922/` | Position profiles, attention/value separation, projections, contextual RI, controlled readout |
| §5.4, App. H (Stage 4: routes, groups, mechanisms) | `stage4_design_v1/`, `stage4_design_v2/`, `stage4_inputs_v1/`, `stage4_all_v2/` (S4.1 routes), `stage4_review_20260924/`, `stage4_s42_inputs_v1/`, `stage4_s42_all_v1/`, `stage4_s42_analysis_20260924/` (next_stage_plan.json), `stage4_s43_inputs_v1/`, `stage4_s43_extension_inputs_v1/`, `stage4_s43_all_v1/`, `stage4_s43_review_20260926/`, `stage4_s45_inputs_v1/`, `stage4_s45_discovery_v2/`, `stage4_s45_review_20260927/`, `stage4_s45_heldout_received_20260928/`, `heldout_plan.json` | S4.1 route mapping; S4.2 groups/blocking/RI31/backup; S4.3 local expansion and chains; revised S4.4 / code S45 RI participation in retained backgrounds (discovery and completed frozen 87-family validation) |
| §5.5, App. I (developmental) | `../Daniella's test ICL vs SIH/results/` | Training-time RI/ICL measurements (Daniella) |
| Housekeeping | `stage4_implementation_review/`, `stage4_s42_readiness_20260924/`, `stage4_s42_build_verification.json`, `stage4_s41_v2/`, `stage4_s42_v1/` | Implementation reviews, bundle verification, superseded/early run stubs |

Analysis documents to read first per stage: `stage1_v4_review/`, `stage2_v2_analysis/summary.json`,
`stage3_review_20260922/`, `stage4_review_20260924/`, `stage4_s42_analysis_20260924/`,
`stage4_s43_review_20260926/S43_review.md`, `stage4_s45_discovery_v2/S45_discovery_analysis.md` and
`stage4_s45_review_20260927/REVIEW_HE.md`, and `stage4_s45_heldout_received_20260928/REVIEW_HE.md`. The narrative reports live in `חומר כתוב/`.

Final manuscript review (no manuscript edits): `paper_final_review_20260928/REVIEW_HE.md`.
The held-out export includes 522 mean-baseline family records; donor effects are summary-only.
The final discovery extension likewise lacks all 89-family vectors locally. Preserve these
reconstruction limits when interpreting audits; all GPU experiments are now complete.
