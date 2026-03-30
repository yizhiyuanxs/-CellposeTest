from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / (den + eps)


def _boundary_map(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask >= 0.5).float()
    dilation = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    erosion = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilation - erosion > 0).float()


def compute_boundary_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    pred_boundary = _boundary_map(preds)
    target_boundary = _boundary_map(targets)

    pred_match = F.max_pool2d(pred_boundary, kernel_size=3, stride=1, padding=1)
    target_match = F.max_pool2d(target_boundary, kernel_size=3, stride=1, padding=1)

    dims = (1, 2, 3)
    precision_tp = (pred_boundary * target_match).sum(dim=dims)
    recall_tp = (target_boundary * pred_match).sum(dim=dims)
    pred_boundary_sum = pred_boundary.sum(dim=dims)
    target_boundary_sum = target_boundary.sum(dim=dims)

    boundary_precision = _safe_div(precision_tp, pred_boundary_sum).mean()
    boundary_recall = _safe_div(recall_tp, target_boundary_sum).mean()
    boundary_f1 = _safe_div(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )

    return {
        "boundary_precision": float(boundary_precision.item()),
        "boundary_recall": float(boundary_recall.item()),
        "boundary_f1": float(boundary_f1.item()),
    }


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)

    dice = _safe_div(2.0 * tp, 2.0 * tp + fp + fn).mean()
    iou = _safe_div(tp, tp + fp + fn).mean()
    precision = _safe_div(tp, tp + fp).mean()
    recall = _safe_div(tp, tp + fn).mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
    return bce + dice_loss
