from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return str(path.resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = project_root() / config_file
    config_file = config_file.resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config["config_path"] = str(config_file)
    dataset = config.setdefault("dataset", {})
    runtime = config.setdefault("runtime", {})
    dataset["root"] = _resolve_path(dataset.get("root", "data/sample_dataset"))
    runtime["output_root"] = _resolve_path(runtime.get("output_root", "runs/stage_a"))
    return config


def config_to_pretty_json(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False)
