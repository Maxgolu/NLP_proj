param([switch]$Stage, [switch]$Commit)
$ErrorActionPreference = 'Stop'
$stage4Root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $stage4Root
$stage4Files = @(
  'pilot_v2/stage4_engine.py', 'pilot_v2/stage4_plan.py', 'pilot_v2/stage4_run.py',
  'pilot_v2/stage4_analyze.py', 'pilot_v2/test_stage4.py', 'pilot_v2/stage4.sbatch',
  'pilot_v2/stage4_pipeline.py', 'pilot_v2/test_stage4_pipeline.py',
  'pilot_v2/stage4_pipeline.sbatch', 'pilot_v2/submit_stage4_pipeline.sh',
  'pilot_v2/submit_stage4.sh', 'pilot_v2/build_stage4_bundle.py',
  'pilot_v2/RUNBOOK_stage4_s41.md', 'pilot_v2/update_stage4_git.ps1',
  'pilot_v2/stage4_s41_v2_update.tar.gz.sha256.json',
  'results/stage4_design_v2/manifest_proposal.json',
  'results/stage4_design_v2/ri_reduced_selection.csv'
)
foreach ($stage4File in $stage4Files) {
  if (-not (Test-Path -LiteralPath $stage4File)) { throw "Missing $stage4File" }
}
if ($Commit) {
  $stage4AlreadyStaged = @(& git -c "safe.directory=$stage4Root" diff --cached --name-only)
  if ($LASTEXITCODE -ne 0) { throw 'Git index check failed' }
  if ($stage4AlreadyStaged.Count -gt 0) { throw 'Commit mode requires an empty index; preserve and handle existing staged work first.' }
}
if ($Stage -or $Commit) {
  & git -c "safe.directory=$stage4Root" add -- $stage4Files
  if ($LASTEXITCODE -ne 0) { throw 'Git add failed' }
  & git -c "safe.directory=$stage4Root" diff --cached --stat
} else {
  Write-Output 'Preview only. Use -Stage to stage these files, or -Commit to stage and commit with an initially empty index.'
  & git -c "safe.directory=$stage4Root" status --short -- $stage4Files
}
if ($Commit) {
  & git -c "safe.directory=$stage4Root" commit -m 'Implement gated Stage 4.1 route mapping and reproducible cluster workflow'
  if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
}
# No push, branch switch, reset, or unrelated-file staging.
