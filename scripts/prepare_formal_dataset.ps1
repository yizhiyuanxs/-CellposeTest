$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.datasets.prepare_dataset `
  --images data/formal_dataset/raw/images `
  --masks data/formal_dataset/raw/masks `
  --output data/formal_dataset/processed `
  --train-ratio 0.7 `
  --val-ratio 0.2 `
  --test-ratio 0.1 `
  --seed 42 `
  --clean-output
