from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from research.engine.common import save_json
from research.utils.config import project_root


def _resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = [
        "label",
        "device",
        "target_resolution",
        "image_count",
        "mean_seconds",
        "median_seconds",
        "p95_seconds",
        "threshold_seconds",
        "meets_requirement",
    ]
    lines = [
        "# 测速结果汇总",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        line_values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                line_values.append(f"{value:.4f}")
            else:
                line_values.append(str(value))
        lines.append("| " + " | ".join(line_values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总多个测速结果文件。")
    parser.add_argument("--config", default="configs/benchmark_toy.yaml")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    output_dir = _resolve(config.get("summary", {}).get("output_dir", "reports/benchmarks/summary"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in config.get("items", []):
        benchmark_path = _resolve(item["benchmark"])
        payload = _load_json(benchmark_path)
        rows.append(
            {
                "label": item["label"],
                "device": payload["device"],
                "target_resolution": payload["target_resolution"],
                "image_count": payload["image_count"],
                "mean_seconds": payload["mean_seconds"],
                "median_seconds": payload["median_seconds"],
                "p95_seconds": payload["p95_seconds"],
                "threshold_seconds": payload["threshold_seconds"],
                "meets_requirement": payload["meets_requirement"],
                "benchmark": str(benchmark_path),
            }
        )

    rows.sort(key=lambda row: row["label"])
    save_json(output_dir / "benchmark_summary.json", {"rows": rows})

    with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "device",
                "target_resolution",
                "image_count",
                "mean_seconds",
                "median_seconds",
                "p95_seconds",
                "threshold_seconds",
                "meets_requirement",
                "benchmark",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_markdown(rows, output_dir / "benchmark_summary.md")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"测速汇总已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
