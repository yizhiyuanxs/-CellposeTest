# Dataset Layout

The research pipeline expects the dataset root to follow this structure:

```text
dataset_root/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Rules:

- Image files and mask files must use the same stem, for example `cell_001.png` and `cell_001.png`.
- Images and masks can use different allowed extensions, but the stem must still match.
- Empty directories are allowed in Stage A so the bootstrap command can run before the real dataset is prepared.

For a real labeled dataset, you can also start from a flat layout:

```text
data/formal_dataset/raw/
  images/
  masks/
```

Then run:

```powershell
python -m research.datasets.prepare_dataset --images data/formal_dataset/raw/images --masks data/formal_dataset/raw/masks --output data/formal_dataset/processed --clean-output
```
