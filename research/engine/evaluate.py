from __future__ import annotations

import argparse
import json

import torch

from research.engine.common import build_dataloader, build_model, save_json
from research.utils.config import load_config
from research.utils.metrics import compute_binary_metrics, compute_boundary_f1, dice_bce_loss
from research.utils.runtime import resolve_device, set_seed


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "boundary_precision": 0.0,
        "boundary_recall": 0.0,
        "boundary_f1": 0.0,
    }
    batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        metrics = compute_binary_metrics(logits, masks, threshold=threshold)
        metrics.update(compute_boundary_f1(logits, masks, threshold=threshold))

        total_loss += float(loss.item())
        for key, value in metrics.items():
            total_metrics[key] += value
        batches += 1

    if batches == 0:
        raise ValueError("评估数据加载器为空。")

    result = {"loss": total_loss / batches}
    for key, value in total_metrics.items():
        result[key] = value / batches
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="评估分割模型。")
    parser.add_argument("--config", default="configs/baseline_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = resolve_device(config["training"].get("device", "cpu"))
    threshold = float(config.get("inference", {}).get("threshold", 0.5))

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    loader = build_dataloader(config, args.split, shuffle=False)
    metrics = evaluate_model(model, loader, device, threshold)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = checkpoint.get("run_dir")
    if output_path:
        save_json(f"{output_path}/eval_{args.split}.json", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
