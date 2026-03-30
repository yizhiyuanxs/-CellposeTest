from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from research.engine.common import save_json
from research.utils.config import project_root


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _label_strip(width: int, text: str, height: int = 28) -> np.ndarray:
    image = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.text((8, 7), text, fill=(20, 20, 20))
    return np.asarray(image, dtype=np.uint8)


def _resolve_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成并排可视化对比图。")
    parser.add_argument("--baseline", default="reports/visualizations/baseline_toy")
    parser.add_argument("--se", default="reports/visualizations/se_toy")
    parser.add_argument("--cbam", default="reports/visualizations/cbam_toy")
    parser.add_argument("--se-cbam", dest="se_cbam", default="reports/visualizations/se_cbam_toy")
    parser.add_argument("--output", default="reports/visualizations/ablation_compare")
    args = parser.parse_args()

    sources = {
        "baseline": _resolve_dir(args.baseline),
        "se": _resolve_dir(args.se),
        "cbam": _resolve_dir(args.cbam),
        "se_cbam": _resolve_dir(args.se_cbam),
    }
    output_dir = _resolve_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_sets = []
    for path in sources.values():
        sample_sets.append({item.name for item in path.iterdir() if item.is_dir()})
    common_samples = sorted(set.intersection(*sample_sets)) if sample_sets else []
    if not common_samples:
        raise ValueError("在各可视化结果目录中未找到共同样本目录。")

    manifest = {"sources": {key: str(value) for key, value in sources.items()}, "samples": []}
    for sample in common_samples:
        panels = []
        for label, folder in sources.items():
            panel_path = folder / sample / "panel.png"
            if not panel_path.exists():
                raise FileNotFoundError(f"缺少对比面板图像：{panel_path}")
            panel = _load_rgb(panel_path)
            labeled_panel = np.concatenate([_label_strip(panel.shape[1], label), panel], axis=0)
            panels.append(labeled_panel)
        comparison = np.concatenate(panels, axis=1)
        out_path = output_dir / f"{sample}_compare.png"
        Image.fromarray(comparison).save(out_path)
        manifest["samples"].append({"sample": sample, "output": str(out_path)})

    save_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"对比图已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
