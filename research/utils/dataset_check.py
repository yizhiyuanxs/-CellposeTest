from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.utils.config import config_to_pretty_json, load_config


def _collect_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.exists():
        return {}
    files: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        files[path.stem] = path
    return files


def validate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"])
    splits = dataset_cfg.get("splits", ["train", "val", "test"])
    image_dir_name = dataset_cfg.get("image_dir_name", "images")
    mask_dir_name = dataset_cfg.get("mask_dir_name", "masks")
    image_exts = {ext.lower() for ext in dataset_cfg.get("image_extensions", [])}
    mask_exts = {ext.lower() for ext in dataset_cfg.get("mask_extensions", [])}

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.exists(),
        "splits": {},
        "errors": [],
        "warnings": [],
    }

    if not dataset_root.exists():
        summary["errors"].append(f"数据集根目录不存在：{dataset_root}")
        return summary

    for split in splits:
        image_dir = dataset_root / split / image_dir_name
        mask_dir = dataset_root / split / mask_dir_name
        image_files = _collect_files(image_dir, image_exts)
        mask_files = _collect_files(mask_dir, mask_exts)

        matched = sorted(set(image_files) & set(mask_files))
        images_only = sorted(set(image_files) - set(mask_files))
        masks_only = sorted(set(mask_files) - set(image_files))

        split_summary = {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "image_count": len(image_files),
            "mask_count": len(mask_files),
            "matched_pairs": len(matched),
            "images_without_masks": images_only,
            "masks_without_images": masks_only,
            "image_dir_exists": image_dir.exists(),
            "mask_dir_exists": mask_dir.exists(),
        }
        summary["splits"][split] = split_summary

        if not image_dir.exists():
            summary["errors"].append(f"[{split}] 图像目录缺失：{image_dir}")
        if not mask_dir.exists():
            summary["errors"].append(f"[{split}] 掩膜目录缺失：{mask_dir}")
        if image_dir.exists() and mask_dir.exists() and len(matched) == 0:
            summary["warnings"].append(f"[{split}] 未找到匹配的图像/掩膜对")
        if images_only:
            summary["warnings"].append(
                f"[{split}] 有 {len(images_only)} 个图像文件没有对应掩膜"
            )
        if masks_only:
            summary["warnings"].append(
                f"[{split}] 有 {len(masks_only)} 个掩膜文件没有对应图像"
            )

    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description="校验研究数据集目录结构。")
    parser.add_argument("--config", default="configs/stage_a_minimal.yaml")
    parser.add_argument("--json", action="store_true", help="输出 JSON，而不是普通摘要文本。")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = validate_dataset(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(config_to_pretty_json(config))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
