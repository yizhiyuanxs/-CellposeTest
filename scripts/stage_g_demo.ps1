$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

python -m research.engine.generate_report --ablation reports\ablation_toy\ablation_summary.json --benchmark reports\benchmarks\benchmark_1200px.json --visuals reports\visualizations\baseline_toy --output reports\研究报告草稿.md
