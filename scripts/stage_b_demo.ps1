$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.datasets.generate_toy_dataset --output data/toy_cells
python -m research.engine.train --config configs/baseline_toy.yaml
