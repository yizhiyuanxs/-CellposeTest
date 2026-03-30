$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.train --config configs/se_toy.yaml
python -m research.engine.train --config configs/cbam_toy.yaml
python -m research.engine.train --config configs/se_cbam_toy.yaml
