from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from research.utils.config import config_to_pretty_json, load_config
from research.utils.dataset_check import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 A 启动入口。")
    parser.add_argument("--config", default="configs/stage_a_minimal.yaml")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="当数据集为空或存在未匹配文件时直接失败。",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = validate_dataset(config)

    print("=== 已加载配置 ===")
    print(config_to_pretty_json(config))
    print("=== 数据集摘要 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output_root = Path(config["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "status": "ready",
        "strict_mode": args.strict,
        "config": config,
        "dataset_summary": summary,
        "next_step": "阶段 A 已完成。下一步可在该骨架基础上进入阶段 B 并构建基线模型。",
    }

    summary_file = run_dir / "stage_a_summary.json"
    with summary_file.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    allow_empty = bool(config["runtime"].get("allow_empty_dataset", True))
    has_errors = bool(summary["errors"])
    has_pairs = any(
        split_info["matched_pairs"] > 0 for split_info in summary["splits"].values()
    )
    has_unmatched = any(
        split_info["images_without_masks"] or split_info["masks_without_images"]
        for split_info in summary["splits"].values()
    )

    if has_errors:
        print(f"阶段 A 启动失败。摘要已写入：{summary_file}")
        return 1
    if args.strict and ((not has_pairs and not allow_empty) or has_unmatched):
        print(f"阶段 A 在严格模式下失败。摘要已写入：{summary_file}")
        return 1

    if not has_pairs:
        print("阶段 A 已完成，当前仅存在占位用数据集目录。")
    else:
        print("阶段 A 已完成，且至少存在一组匹配的数据样本。")
    print(f"摘要已写入：{summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
