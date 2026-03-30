from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def load_mask_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    return (arr > 127).astype(np.float32)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    arr = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_overlay(path: str | Path, image: np.ndarray, mask: np.ndarray) -> None:
    base = np.clip(image, 0.0, 1.0)
    base_rgb = np.stack([base, base, base], axis=-1)
    mask_bool = mask > 0.5
    overlay = base_rgb.copy()
    overlay[mask_bool] = 0.55 * overlay[mask_bool] + 0.45 * np.array([1.0, 0.1, 0.1])
    arr = (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    arr = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
