$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.visualize --config configs/baseline_toy.yaml --checkpoint runs\stage_b\20260323_102823_baseline_toy_unet\best.pt --split test --limit 4 --output reports\visualizations\baseline_toy
