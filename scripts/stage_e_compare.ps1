$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.compare_visualizations `
  --baseline reports\visualizations\baseline_toy `
  --se reports\visualizations\se_toy `
  --cbam reports\visualizations\cbam_toy `
  --se-cbam reports\visualizations\se_cbam_toy `
  --output reports\visualizations\ablation_compare
