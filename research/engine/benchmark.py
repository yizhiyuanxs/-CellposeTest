from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.engine.common import build_model, save_json
from research.utils.config import load_config
from research.utils.image_io import load_grayscale_image
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
    parser = argparse.ArgumentParser(description="对分割模型推理速度进行测速。")
    parser.add_argument("--config", default="configs/baseline_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/benchmarks")
    parser.add_argument("--resize", type=int, default=1200, help="推理前将输入缩放为指定正方形尺寸。")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = resolve_device(config["training"].get("device", "cpu"))
    threshold_seconds = 5.0

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    input_path = input_path.resolve()
    image_paths = _collect_input_images(input_path)
    if not image_paths:
        raise ValueError(f"用于测速的输入图像不存在：{input_path}")

    for _ in range(args.warmup):
        image = load_grayscale_image(image_paths[0])
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        if args.resize > 0:
            tensor = F.interpolate(tensor, size=(args.resize, args.resize), mode="bilinear", align_corners=False)
        _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    rows = []
    for image_path in image_paths:
        image = load_grayscale_image(image_path)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        if args.resize > 0:
            tensor = F.interpolate(tensor, size=(args.resize, args.resize), mode="bilinear", align_corners=False)

        start = time.perf_counter()
        logits = model(tensor)
        _ = torch.sigmoid(logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        rows.append(
            {
                "image": str(image_path),
                "seconds": elapsed,
                "height": int(tensor.shape[-2]),
                "width": int(tensor.shape[-1]),
            }
        )

    mean_seconds = statistics.mean(timings)
    median_seconds = statistics.median(timings)
    p95_seconds = float(np.percentile(np.array(timings, dtype=np.float64), 95))
    passed = p95_seconds <= threshold_seconds and mean_seconds <= threshold_seconds

    payload = {
        "device": str(device),
        "image_count": len(rows),
        "target_resolution": args.resize,
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "p95_seconds": p95_seconds,
        "threshold_seconds": threshold_seconds,
        "meets_requirement": passed,
        "rows": rows,
    }

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"benchmark_{args.resize}px.json"
    save_json(result_path, payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"测速结果已写入：{result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
