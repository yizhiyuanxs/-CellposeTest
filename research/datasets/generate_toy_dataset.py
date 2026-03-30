from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _draw_sample(image_size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    image = Image.new("L", (image_size, image_size), color=10)
    mask = Image.new("L", (image_size, image_size), color=0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    object_count = rng.randint(1, 3)
    for _ in range(object_count):
        radius_x = rng.randint(image_size // 10, image_size // 5)
        radius_y = rng.randint(image_size // 10, image_size // 5)
        center_x = rng.randint(radius_x + 2, image_size - radius_x - 2)
        center_y = rng.randint(radius_y + 2, image_size - radius_y - 2)
        bbox = [
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        ]
        intensity = rng.randint(150, 240)
        image_draw.ellipse(bbox, fill=intensity)
        mask_draw.ellipse(bbox, fill=255)

    image_arr = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0.0, 12.0, size=image_arr.shape)
    image_arr = np.clip(image_arr + noise, 0, 255).astype(np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    return image_arr, mask_arr


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 toy 分割数据集。")
    parser.add_argument("--output", default="data/toy_cells")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.output).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    rng = random.Random(args.seed)
    split_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }

    for split, count in split_counts.items():
        image_dir = root / split / "images"
        mask_dir = root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image_arr, mask_arr = _draw_sample(args.image_size, rng)
            stem = f"{split}_{idx:03d}"
            Image.fromarray(image_arr, mode="L").save(image_dir / f"{stem}.png")
            Image.fromarray(mask_arr, mode="L").save(mask_dir / f"{stem}.png")

    print(f"toy 数据集已写入：{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
