from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from research.utils.config import project_root


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _collect(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")
    items: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            items[path.stem] = path
    return items


def _copy_pairs(items: list[tuple[Path, Path]], output_root: Path, split: str) -> dict[str, int]:
    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in items:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(mask_path, mask_dir / mask_path.name)
    return {"pairs": len(items)}


def main() -> int:
    parser = argparse.ArgumentParser(description="将正式数据集整理为 train/val/test 结构。")
    parser.add_argument("--images", default="data/formal_dataset/raw/images")
    parser.add_argument("--masks", default="data/formal_dataset/raw/masks")
    parser.add_argument("--output", default="data/formal_dataset/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio 必须等于 1.0")

    images_dir = _resolve_path(args.images)
    masks_dir = _resolve_path(args.masks)
    output_root = _resolve_path(args.output)

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_map = _collect(images_dir, IMAGE_EXTENSIONS)
    mask_map = _collect(masks_dir, MASK_EXTENSIONS)
    stems = sorted(set(image_map) & set(mask_map))
    if not stems:
        raise ValueError("在原始数据集目录中未找到匹配的图像/掩膜对。")

    rng = random.Random(args.seed)
    rng.shuffle(stems)

    total = len(stems)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": stems[:train_end],
        "val": stems[train_end:val_end],
        "test": stems[val_end:],
    }

    summary = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "output_root": str(output_root),
        "total_pairs": total,
        "seed": args.seed,
        "splits": {},
        "unmatched_images": sorted(set(image_map) - set(mask_map)),
        "unmatched_masks": sorted(set(mask_map) - set(image_map)),
    }

    for split, split_stems in splits.items():
        pairs = [(image_map[stem], mask_map[stem]) for stem in split_stems]
        split_summary = _copy_pairs(pairs, output_root, split)
        split_summary["stems"] = split_stems
        summary["splits"][split] = split_summary

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"整理后的数据集已写入：{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
