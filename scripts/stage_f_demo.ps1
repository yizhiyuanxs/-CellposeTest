$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.benchmark --config configs/baseline_toy.yaml --checkpoint runs\stage_b\20260323_102823_baseline_toy_unet\best.pt --input data\toy_cells\test\images --resize 1200 --output reports\benchmarks
python -m research.engine.summarize_benchmarks --config configs\benchmark_toy.yaml
