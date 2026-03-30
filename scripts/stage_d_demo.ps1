$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.summarize_ablation --config configs/ablation_toy.yaml
