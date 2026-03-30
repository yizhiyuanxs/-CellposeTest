from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from research.engine.common import build_model
from research.utils.config import load_config
from research.utils.image_io import load_grayscale_image, save_mask, save_overlay
from research.utils.runtime import resolve_device, set_seed


def _collect_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(
        path for path in input_path.glob("*") if path.is_file() and path.suffix.lower() in supported
    )


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="使用分割模型执行推理。")
    parser.add_argument("--config", default="configs/baseline_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="runs/infer")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = resolve_device(config["training"].get("device", "cpu"))
    threshold = float(config.get("inference", {}).get("threshold", 0.5))

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    input_path = input_path.resolve()
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_input_images(input_path)
    if not image_paths:
        raise ValueError(f"未找到输入图像：{input_path}")

    for image_path in image_paths:
        image = load_grayscale_image(image_path)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (prob >= threshold).astype(np.float32)

        stem = image_path.stem
        save_mask(output_dir / f"{stem}_pred.png", pred_mask)
        save_overlay(output_dir / f"{stem}_overlay.png", image, pred_mask)

    print(f"推理完成，结果已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
