from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.datasets.segmentation_dataset import SegmentationDataset
from research.engine.common import build_model
from research.utils.config import load_config
from research.utils.image_io import save_mask, save_rgb
from research.utils.runtime import resolve_device, set_seed


def _to_rgb(image: np.ndarray) -> np.ndarray:
    base = np.clip(image, 0.0, 1.0)
    return (np.stack([base, base, base], axis=-1) * 255.0).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask > 0.5] = np.array(color, dtype=np.uint8)
    return rgb


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0.5).astype(np.uint8)
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)
    boundary = (mask > 0) & ((mask != up) | (mask != down) | (mask != left) | (mask != right))
    boundary[0, :] = 0
    boundary[-1, :] = 0
    boundary[:, 0] = 0
    boundary[:, -1] = 0
    return boundary.astype(np.uint8)


def _error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    tp = (pred > 0.5) & (gt > 0.5)
    fp = (pred > 0.5) & (gt <= 0.5)
    fn = (pred <= 0.5) & (gt > 0.5)
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[tp] = np.array([80, 200, 120], dtype=np.uint8)
    rgb[fp] = np.array([240, 80, 80], dtype=np.uint8)
    rgb[fn] = np.array([80, 120, 240], dtype=np.uint8)
    return rgb


def _boundary_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    rgb = _to_rgb(image).astype(np.float32)
    gt_boundary = _boundary(gt)
    pred_boundary = _boundary(pred)
    overlap = (gt_boundary > 0) & (pred_boundary > 0)
    rgb[gt_boundary > 0] = np.array([50, 220, 50], dtype=np.float32)
    rgb[pred_boundary > 0] = np.array([240, 60, 60], dtype=np.float32)
    rgb[overlap] = np.array([250, 220, 40], dtype=np.float32)
    return rgb.astype(np.uint8)


def _heatmap_to_rgb(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    red = cam
    green = np.clip(1.0 - np.abs(cam - 0.5) * 2.0, 0.0, 1.0)
    blue = 1.0 - cam
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _overlay_heatmap(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    base = _to_rgb(image).astype(np.float32)
    heatmap = _heatmap_to_rgb(cam).astype(np.float32)
    blended = 0.55 * base + 0.45 * heatmap
    return np.clip(blended, 0, 255).astype(np.uint8)


def _save_panel(path: Path, images: list[np.ndarray]) -> None:
    panel = np.concatenate(images, axis=1)
    save_rgb(path, panel)


class GradCAM:
    def __init__(self, target_module: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, logits: torch.Tensor) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM 钩子未捕获到激活值或梯度。")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-6)
        return cam.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


@torch.no_grad()
def _predict(model: torch.nn.Module, image_tensor: torch.Tensor, threshold: float) -> tuple[np.ndarray, torch.Tensor]:
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()[0, 0].cpu().numpy()
    return pred, logits


def main() -> int:
    parser = argparse.ArgumentParser(description="生成分割可视化结果。")
    parser.add_argument("--config", default="configs/baseline_toy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", default="reports/visualizations")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = resolve_device(config["training"].get("device", "cpu"))
    threshold = float(config.get("inference", {}).get("threshold", 0.5))

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = SegmentationDataset(
        root=config["dataset"]["root"],
        split=args.split,
        image_dir_name=config["dataset"].get("image_dir_name", "images"),
        mask_dir_name=config["dataset"].get("mask_dir_name", "masks"),
        image_extensions=tuple(config["dataset"].get("image_extensions", [".png"])),
        mask_extensions=tuple(config["dataset"].get("mask_extensions", [".png"])),
    )
    if len(dataset) == 0:
        raise ValueError(f"在 split='{args.split}' 中未找到可视化样本。")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model.down3.conv)
    count = min(args.limit, len(dataset))
    for idx in range(count):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        mask_np = sample["mask"].squeeze(0).numpy()
        image_np = sample["image"].squeeze(0).numpy()

        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        focus_mask = (probs >= threshold).float()
        if focus_mask.sum().item() == 0:
            focus_mask = torch.ones_like(probs)
        score = (logits * focus_mask).mean()
        score.backward()
        heatmap = cam.generate(logits)

        pred_np = (probs[0, 0].detach().cpu().numpy() >= threshold).astype(np.float32)
        base_name = Path(str(sample["image_path"])).stem
        sample_dir = output_dir / base_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        original_rgb = _to_rgb(image_np)
        gt_rgb = _mask_to_rgb(mask_np, (60, 220, 60))
        pred_rgb = _mask_to_rgb(pred_np, (240, 70, 70))
        error_rgb = _error_map(pred_np, mask_np)
        heatmap_rgb = _overlay_heatmap(image_np, heatmap)
        boundary_rgb = _boundary_overlay(image_np, pred_np, mask_np)

        save_rgb(sample_dir / "original.png", original_rgb)
        save_rgb(sample_dir / "gt.png", gt_rgb)
        save_rgb(sample_dir / "pred.png", pred_rgb)
        save_rgb(sample_dir / "error.png", error_rgb)
        save_rgb(sample_dir / "heatmap.png", heatmap_rgb)
        save_rgb(sample_dir / "boundary.png", boundary_rgb)
        save_mask(sample_dir / "pred_mask.png", pred_np)
        _save_panel(
            sample_dir / "panel.png",
            [original_rgb, gt_rgb, pred_rgb, error_rgb, heatmap_rgb, boundary_rgb],
        )

    cam.close()
    print(f"可视化结果已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
