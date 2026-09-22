# Git update after Stage 3 (analysis, report, attended-name classification, Section-1.5 readout code).
# Run from PowerShell. Stages only the listed files; unrelated local edits are left untouched.
# Review `git status --short` and `git diff --cached --stat` before the commit line runs.
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\User\OneDrive\Documents\computer science\4B\NLP\final proj\project'

# --- keep large or regenerable Stage-3 artifacts out of Git -----------------------------
$ignore = @(
  '# Stage 3 attended-name event table (21 MB; regenerate with stage3_analyze.py --attended-only)',
  'results/stage3_v1/analysis/attended_name_events.csv',
  'results/stage3_v1/analysis/*_PROVISIONAL.*',
  '# Section-1.5 readout: raw run directory lives on OneDrive + cluster; analysis CSVs are committed',
  'results/stage3_readout_v1/readout_records.jsonl.gz',
  'results/stage3_readout_v1/state.json',
  'results/stage3_readout_v1/.running.lock'
)
$gi = Get-Content .gitignore -Raw
foreach ($line in $ignore) { if ($gi -notmatch [regex]::Escape($line)) { Add-Content .gitignore $line } }

git status --short

# --- Stage-3 report (Overleaf source, figures, table, compiled local PDF) ----------------
git add "חומר כתוב/Stage3_Results_and_Analysis.tex"
git add "חומר כתוב/Stage3_Results_and_Analysis_local.pdf"
git add "חומר כתוב/tables/tab_profiles.tex"
git add "חומר כתוב/figs/"
git add "חומר כתוב/Stage3_plan_attended_name_paragraph.tex"
git add "חומר כתוב/Stage3_Head_Level_Characterization_revised.tex"

# --- Stage-3 code and analysis outputs ----------------------------------------------------
git add pilot_v2/stage3_analyze.py pilot_v2/stage3_report_figures.py
git add results/stage3_v1/summary.json results/stage3_v1/manifest.json results/stage3_v1/gate_passed.json
git add results/stage3_v1/cost_estimate.json results/stage3_v1/gate_0.json results/stage3_v1/gate_1.json results/stage3_v1/gate_2.json
git add results/stage3_v1/analysis/head_profiles.csv results/stage3_v1/analysis/head_cards.md
git add results/stage3_v1/analysis/attention_profiles.csv results/stage3_v1/analysis/output_profiles.csv
git add results/stage3_v1/analysis/paired_attention_controls.csv results/stage3_v1/analysis/contextual_ri.csv
git add results/stage3_v1/analysis/contextual_ri_correlations.csv results/stage3_v1/analysis/copying_weights.csv
git add results/stage3_v1/analysis/synthetic_profiles.csv results/stage3_v1/analysis/synthetic_support.csv
git add results/stage3_v1/analysis/attended_name_movers.csv results/stage3_v1/analysis/attended_name_rule.json

# --- Section-1.5 readout (new) ------------------------------------------------------------
git add pilot_v2/stage3_readout.py pilot_v2/test_stage3_readout.py pilot_v2/build_stage3_readout_bundle.py
git add pilot_v2/stage3_readout.sbatch pilot_v2/submit_stage3_readout.sh pilot_v2/RUNBOOK_stage3_readout.md
if (Test-Path results/stage3_readout_inputs_v1) { git add results/stage3_readout_inputs_v1/readout_plan.json results/stage3_readout_inputs_v1/pairs.jsonl.gz }
if (Test-Path pilot_v2/stage3_readout_v1_update.tar.gz.sha256.json) { git add pilot_v2/stage3_readout_v1_update.tar.gz.sha256.json }

# --- handoff and ignore rules -------------------------------------------------------------
git add RESEARCH_HANDOFF.md .gitignore

git diff --cached --stat
git commit -m "Stage 3: results analysis and report, attended-name mover classification, Section-1.5 readout code" -m "Stage-3 run stage3_v1 analyzed (105 heads, 178 pairs); report Stage3_Results_and_Analysis.tex with 10 figures and head-profile table, corrected after review (signed vs absolute aggregates, projection != attribution, mover labels as descriptive, current-token RI partition added, L17H3 counted). stage3_analyze.py gains the attended-name mover classification (--attended-only). New stage3_readout.py (prepare/run/analyze), tests, sbatch/submit, runbook and bundle builder for the optional Section-1.5 controlled representation readout on L17H1 and L15H25 sites; not yet run on the 7B model."
git push origin HEAD
