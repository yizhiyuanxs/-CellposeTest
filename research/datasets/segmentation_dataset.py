from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from research.utils.image_io import load_grayscale_image, load_mask_image


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
        mask_extensions: tuple[str, ...] = (".png", ".tif", ".tiff", ".bmp"),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / image_dir_name
        self.mask_dir = self.root / split / mask_dir_name
        self.samples: list[tuple[Path, Path]] = []

        image_candidates = {
            path.stem: path
            for path in sorted(self.image_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in image_extensions
        }
        mask_candidates = {
            path.stem: path
            for path in sorted(self.mask_dir.glob("*"))
            if path.is_file() and path.suffix.lower() in mask_extensions
        }
        for stem in sorted(set(image_candidates) & set(mask_candidates)):
            self.samples.append((image_candidates[stem], mask_candidates[stem]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image = load_grayscale_image(image_path)
        mask = load_mask_image(mask_path)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
