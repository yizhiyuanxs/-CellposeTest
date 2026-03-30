"""Call Cellpose from Python for local image segmentation.

Examples:
  python main.py --input ./images --output ./outputs
  python main.py --input ./images/sample.tif --gpu --diameter 30
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_CELLPOSE_PYTHON = r"D:\Tools\Anaconda\envs\cellpose\python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在本地运行 Cellpose 分割。")
    parser.add_argument(
        "--input",
        default="images",
        help="输入图像文件或文件夹路径（默认：images）。",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="保存分割结果的目录（默认：outputs）。",
    )
    parser.add_argument(
        "--model",
        default="cpsam",
        help="Cellpose 模型名称或路径（默认：cpsam）。",
    )
    parser.add_argument("--gpu", action="store_true", help="若可用则使用 GPU。")
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="细胞直径，单位为像素。留空则由模型自动估计。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="推理批大小（默认：8）。",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="用于 mask 质量控制的 flow 阈值（默认：0.4）。",
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="细胞概率阈值（默认：0.0）。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当输入为文件夹时递归搜索图像。",
    )
    return parser.parse_args()


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图像文件：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"输入路径不存在：{input_path}")

    pattern = "**/*" if recursive else "*"
    images = [
        p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"在以下路径中未找到图像：{input_path}")
    return sorted(images)


def save_results(
    image_path: Path,
    output_dir: Path,
    image: np.ndarray,
    masks: np.ndarray,
    flows: Iterable[np.ndarray],
    io_module,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = image_path.stem
    prefix = f"{timestamp}_{stem}"
    mask_tif = output_dir / f"{prefix}_cp_masks.tif"
    mask_npy = output_dir / f"{prefix}_cp_masks.npy"
    flow_npy = output_dir / f"{prefix}_cp_flows.npy"
    overlay_png = output_dir / f"{prefix}_cp_overlay.png"

    io_module.imsave(str(mask_tif), masks.astype(np.uint16))
    np.save(mask_npy, masks)
    np.save(flow_npy, np.array(flows, dtype=object), allow_pickle=True)
    io_module.imsave(str(overlay_png), make_overlay(image, masks))
    return overlay_png


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"该图像形状不支持可视化：{arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v <= min_v:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - min_v) / (max_v - min_v)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def make_overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    base = to_rgb_uint8(image).astype(np.float32)
    labels = masks.astype(np.int64, copy=False)
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return base.astype(np.uint8)

    rng = np.random.default_rng(42)
    colors = rng.integers(64, 255, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = 0
    color_mask = colors[np.clip(labels, 0, max_label)].astype(np.float32)

    overlay = base.copy()
    fg = labels > 0
    overlay[fg] = 0.65 * base[fg] + 0.35 * color_mask[fg]

    up = np.roll(labels, -1, axis=0)
    down = np.roll(labels, 1, axis=0)
    left = np.roll(labels, -1, axis=1)
    right = np.roll(labels, 1, axis=1)
    boundary = fg & ((labels != up) | (labels != down) | (labels != left) | (labels != right))
    overlay[boundary] = np.array([255, 60, 60], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_cellpose_modules():
    try:
        from cellpose import io, models
    except ImportError:
        return None, None
    return io, models


def resolve_pretrained_model(model_arg: str) -> str:
    candidate = Path(model_arg).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if model_arg == "cpsam":
        project_model = Path(__file__).resolve().parent / "models" / "cpsam"
        user_cache_model = Path.home() / ".cellpose" / "models" / "cpsam"
        for local_model in (project_model, user_cache_model):
            if local_model.exists():
                return str(local_model.resolve())

    return model_arg


def is_download_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, urllib.error.URLError):
            return True
        message = str(current)
        if "urlopen error" in message or "WinError 10060" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def offline_model_help() -> str:
    project_hint = Path(__file__).resolve().parent / "models" / "cpsam"
    user_hint = Path.home() / ".cellpose" / "models" / "cpsam"
    return (
        "下载 Cellpose 模型 'cpsam' 失败。\n"
        "当前机器可能无法访问 HuggingFace。\n"
        "离线修复方法：\n"
        "1）在可联网的机器上运行：\n"
        "   python -c \"from cellpose import models; models.CellposeModel(gpu=False, pretrained_model='cpsam')\"\n"
        "2）将下载得到的 'cpsam' 文件复制到本机以下任一路径：\n"
        f"   - {user_hint}\n"
        f"   - {project_hint}\n"
        "3）重新运行本脚本，或通过 --model 传入本地模型完整路径。"
    )


def eval_with_console_progress(model, image, **eval_kwargs):
    start = time.time()
    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.wait(15):
            elapsed = time.time() - start
            print(f"    正在处理... 已耗时 {elapsed:.0f}s", flush=True)

    reporter = threading.Thread(target=_heartbeat, daemon=True)
    reporter.start()
    try:
        return model.eval(image, **eval_kwargs)
    finally:
        stop.set()
        reporter.join(timeout=0.2)
        total = time.time() - start
        print(f"    处理完成，总耗时 {total:.1f}s", flush=True)


def maybe_rerun_with_cellpose_python() -> bool:
    candidates = [
        os.environ.get("CELLPOSE_PYTHON_EXE", "").strip(),
        DEFAULT_CELLPOSE_PYTHON,
    ]
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue
        if candidate_path.resolve() == current:
            continue
        print("当前解释器未安装 cellpose，正在切换解释器重新启动：")
        print(candidate_path)
        result = subprocess.run([str(candidate_path), __file__, *sys.argv[1:]], check=False)
        raise SystemExit(result.returncode)
    return False


def main() -> None:
    args = parse_args()

    io_module, models_module = load_cellpose_modules()
    if io_module is None or models_module is None:
        maybe_rerun_with_cellpose_python()
        raise SystemExit(
            "当前 Python 解释器中不可用 cellpose。\n"
            f"当前解释器：{sys.executable}\n"
            "已尝试的备用解释器：D:\\Tools\\Anaconda\\envs\\cellpose\\python.exe\n"
            "你也可以通过环境变量 CELLPOSE_PYTHON_EXE 指定其他解释器路径。"
        )

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = collect_images(input_path, recursive=args.recursive)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("正在加载模型...", flush=True)
    model_load_start = time.time()
    model_name_or_path = resolve_pretrained_model(args.model)
    if model_name_or_path != args.model:
        print(f"使用本地模型文件：{model_name_or_path}", flush=True)
    try:
        model = models_module.CellposeModel(gpu=args.gpu, pretrained_model=model_name_or_path)
    except Exception as exc:
        if is_download_error(exc):
            raise SystemExit(offline_model_help()) from exc
        raise
    print(f"模型加载完成，耗时 {time.time() - model_load_start:.1f}s", flush=True)

    print(f"模型：{args.model}")
    print(f"输入图像数量：{len(images)}")
    print(f"输出目录：{output_dir}")

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] 正在处理：{image_path.name}")
        image = io_module.imread(str(image_path))
        masks, flows, _ = eval_with_console_progress(
            model,
            image,
            diameter=args.diameter,
            batch_size=args.batch_size,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        overlay_png = save_results(image_path, output_dir, image, masks, flows, io_module)
        print(
            f"[{idx}/{len(images)}] 完成：{image_path.name}，"
            f"masks={int(np.max(masks))}, "
            f"overlay={overlay_png.name}"
        )


if __name__ == "__main__":
    main()
