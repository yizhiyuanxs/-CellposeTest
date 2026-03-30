from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from research.engine.common import build_dataloader, build_model, save_json
from research.engine.evaluate import evaluate_model
from research.utils.config import load_config
from research.utils.metrics import dice_bce_loss
from research.utils.runtime import make_run_dir, resolve_device, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="训练分割模型。")
    parser.add_argument("--config", default="configs/baseline_toy.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"].get("seed", 42)))
    device = resolve_device(config["training"].get("device", "cpu"))
    threshold = float(config.get("inference", {}).get("threshold", 0.5))

    run_dir = make_run_dir(config["runtime"]["output_root"], config["experiment"]["name"])
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    epochs = int(config["training"].get("epochs", 1))
    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_checkpoint_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"第 {epoch}/{epochs} 轮", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device, threshold)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "run_dir": str(run_dir),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, best_checkpoint_path)

    save_json(run_dir / "history.json", {"history": history, "best_dice": best_dice})
    print(f"训练完成。最佳 checkpoint：{best_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
