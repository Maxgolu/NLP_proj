# Git update: Stage 3 complete including the Section-1.5 readout run; Stage-4 protocol proposal.
# Run from PowerShell at the project root. Stages only the listed files; unrelated local edits are left alone.
# Read the output of `git status --short` and `git diff --cached --stat` before the commit executes.
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\User\OneDrive\Documents\computer science\4B\NLP\final proj\project'

git fetch origin
git status --short

# --- keep raw/heavy artifacts out of Git (idempotent) -----------------------------------
$ignore = @(
  '# Stage 3 result archives (1.2 GB / 0.2 MB): regenerable from the run directories',
  'results/stage3_v1_results.tar.gz',
  'results/stage3_readout_v1_results.tar.gz',
  'results/stage3_readout_v1/runtime.json',
  '# Overleaf ZIP bundles: regenerable from the .tex sources',
  '*.zip'
)
$gi = Get-Content .gitignore -Raw
foreach ($line in $ignore) { if ($gi -notmatch [regex]::Escape($line)) { Add-Content .gitignore $line } }

# --- Stage-3 report: integrated source, Overleaf copy, figures, table, compiled PDF -------
git add "חומר כתוב/Stage3_Results_and_Analysis.tex"
git add "חומר כתוב/Stage3_Results_and_Analysis.pdf"
git add "חומר כתוב/Stage3_Overleaf/"
git add "חומר כתוב/tables/tab_profiles.tex"
git add "חומר כתוב/figs/"

# --- Section-1.5 readout: code fix (locate_plan), run book, run metadata and analysis -----
git add pilot_v2/stage3_readout.py pilot_v2/RUNBOOK_stage3_readout.md
git add results/stage3_readout_v1/manifest.json results/stage3_readout_v1/gate.json
git add results/stage3_readout_v1/done.json results/stage3_readout_v1/summary.json
git add results/stage3_readout_v1/analysis/readout_events.csv
git add results/stage3_readout_v1/analysis/readout_paired.csv
git add results/stage3_readout_v1/analysis/readout_summary.csv
git add results/stage3_readout_v1/analysis/readout_family_summary.csv
if (Test-Path results/stage3_readout_inputs_v1) { git add results/stage3_readout_inputs_v1/readout_plan.json results/stage3_readout_inputs_v1/pairs.jsonl.gz }

# --- independent reviews of Stage 3 and of the readout ------------------------------------
if (Test-Path results/stage3_review_20260922)         { git add results/stage3_review_20260922/ }
if (Test-Path results/stage3_readout_review_20260922) { git add results/stage3_readout_review_20260922/ }

# --- Stage-4 planning documents (proposal only; no Stage-4 code or runs exist) -----------
git add "חומר כתוב/Stage4_Protocol.tex"
git add "חומר כתוב/Stage4_Circuits_and_Communication_preliminary.tex"
git add "חומר כתוב/Stage4_Circuits_and_Communication_preliminary_.pdf"
git add "חומר כתוב/Stage4_Experiment.tex" "חומר כתוב/Stage4_experiment.pdf"
if (Test-Path results/stage4_design_v1) { git add results/stage4_design_v1/ }

# --- entry point, handoff, ignore rules, this script ------------------------------------
git add README.md RESEARCH_HANDOFF.md .gitignore git_update_2026-09-23.ps1

git diff --cached --stat
git commit -m "Stage 3 complete: Section-1.5 controlled readout run, report Section 7.7, Stage-4 protocol proposal" -m "Readout run stage3_readout_v1 (job 918689) on the L17H1 and L15H25 sites: gate passed (zero identity-injection/self-patch drift, saved importances reproduced), 1,920 records, analysis regenerated locally byte-for-byte. Report gains Section 7.7 and updated Sections 8-11; three unbalanced LaTeX quotes fixed. stage3_readout.py: analyze locates the frozen plan by SHA-256 when the manifest path is not local. README/RUNBOOK/RESEARCH_HANDOFF updated to the completed state. Stage-4 protocol proposal and configuration registry added (planning documents; no Stage-4 code or GPU runs)."
git push origin HEAD
git log --oneline -3
