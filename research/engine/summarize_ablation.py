from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from research.engine.common import build_dataloader, build_model, save_json
from research.engine.evaluate import evaluate_model
from research.utils.config import load_config, project_root
from research.utils.runtime import resolve_device, set_seed


def _resolve_latest_glob(glob_pattern: str) -> Path:
    pattern = Path(glob_pattern)
    if not pattern.is_absolute():
        pattern = project_root() / pattern
    matches = [Path(item) for item in glob.glob(str(pattern), recursive=True)]
    if not matches:
        raise FileNotFoundError(f"没有 checkpoint 匹配该模式：{glob_pattern}")
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def _write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = [
        "label",
        "split",
        "dice",
        "iou",
        "precision",
        "recall",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "loss",
    ]
    lines = [
        "# 消融实验汇总",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="评估并汇总消融实验矩阵。")
    parser.add_argument("--config", default="configs/ablation_toy.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        summary_config = yaml.safe_load(fh) or {}

    output_dir = Path(summary_config.get("summary", {}).get("output_dir", "reports/ablation_toy"))
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split = str(summary_config.get("summary", {}).get("split", "val"))

    rows: list[dict[str, Any]] = []
    for item in summary_config.get("items", []):
        item_config = load_config(item["config"])
        set_seed(int(item_config["project"].get("seed", 42)))
        device = resolve_device(item_config["training"].get("device", "cpu"))
        threshold = float(item_config.get("inference", {}).get("threshold", 0.5))
        checkpoint_path = _resolve_latest_glob(item["checkpoint_glob"])

        model = build_model(item_config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        loader = build_dataloader(item_config, split, shuffle=False)
        metrics = evaluate_model(model, loader, device, threshold)

        row = {
            "label": item["label"],
            "split": split,
            **metrics,
            "checkpoint": str(checkpoint_path),
            "experiment_name": item_config["experiment"]["name"],
            "attention": item_config["model"].get("attention", "baseline"),
            "config": item["config"],
        }
        rows.append(row)

    rows.sort(key=lambda row: row["label"])
    save_json(output_dir / "ablation_summary.json", {"rows": rows, "split": split})

    csv_headers = [
        "label",
        "split",
        "dice",
        "iou",
        "precision",
        "recall",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "loss",
        "attention",
        "experiment_name",
        "checkpoint",
        "config",
    ]
    with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_markdown(rows, output_dir / "ablation_summary.md")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"汇总结果已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
