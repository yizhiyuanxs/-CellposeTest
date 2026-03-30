from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from research.datasets.segmentation_dataset import SegmentationDataset
from research.models.unet import UNet


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    if model_cfg.get("name", "unet").lower() != "unet":
        raise ValueError(f"不支持的模型：{model_cfg.get('name')}")
    return UNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        attention=str(model_cfg.get("attention", "baseline")),
        attention_reduction=int(model_cfg.get("attention_reduction", 16)),
    )


def build_dataloader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    dataset_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    dataset = SegmentationDataset(
        root=dataset_cfg["root"],
        split=split,
        image_dir_name=dataset_cfg.get("image_dir_name", "images"),
        mask_dir_name=dataset_cfg.get("mask_dir_name", "masks"),
        image_extensions=tuple(dataset_cfg.get("image_extensions", [".png"])),
        mask_extensions=tuple(dataset_cfg.get("mask_extensions", [".png"])),
    )
    if len(dataset) == 0:
        raise ValueError(f"在 {dataset.root} 中未找到 split='{split}' 的匹配样本")

    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(config["runtime"].get("num_workers", 0)),
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
