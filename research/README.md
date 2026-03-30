# Research Workspace

This directory is the isolated workspace for the research-oriented pipeline.

Stage A provides:

- YAML-based config loading
- Standard dataset layout validation
- A minimal bootstrap run that prints config and writes a summary report

Stage B adds:

- Toy dataset generation for local smoke tests
- A small UNet baseline
- SE and CBAM attention variants
- Train, evaluate, and infer entrypoints

Recommended commands:

```powershell
python -m research.engine.minimal_run --config configs/stage_a_minimal.yaml
python -m research.datasets.generate_toy_dataset --output data/toy_cells
python -m research.engine.train --config configs/baseline_toy.yaml
python -m research.engine.infer --config configs/baseline_toy.yaml --checkpoint <checkpoint_path> --input data/toy_cells/test/images
```
