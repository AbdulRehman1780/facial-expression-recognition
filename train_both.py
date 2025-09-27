import os
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from facial_expression_dataset import build_dataloaders
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet
from multitask_loss import MultiTaskLoss, LossWeightScheduler, LossWeights


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (pred - target).abs().mean().item()


def train_or_validate_epoch(model: nn.Module,
                             criterion: MultiTaskLoss,
                             dataloader,
                             device: torch.device,
                             epoch: int,
                             total_epochs: int,
                             train: bool,
                             optimizer: Adam = None,
                             scaler: torch.cuda.amp.GradScaler = None,
                             global_progress_offset: float = 0.0,
                             global_progress_span: float = 1.0) -> Dict[str, float]:
    if train:
        model.train()
        desc = f"Train [{epoch+1}/{total_epochs}]"
    else:
        model.eval()
        desc = f"Val   [{epoch+1}/{total_epochs}]"

    running = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "acc": 0.0, "mae": 0.0}

    pbar = tqdm(dataloader, desc=desc, ncols=110)
    for i, (images, emotions, valences, arousals) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        emotions = emotions.to(device, non_blocking=True)
        va_targets = torch.stack([valences, arousals], dim=1).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, va_pred = model(images)
            # Scheduling progress only during training
            progress = None
            if train:
                frac = (i + 1) / max(1, len(dataloader))
                progress = global_progress_offset + global_progress_span * frac
                progress = float(max(0.0, min(1.0, progress)))
            loss_dict = criterion(logits, va_pred, emotions, va_targets, progress=progress)
            loss = loss_dict["total"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_acc = accuracy(logits, emotions)
        batch_mae = mae(va_pred, va_targets)

        running["loss"] += loss.item()
        running["cls"] += loss_dict["cls"].item()
        running["reg"] += loss_dict["reg"].item()
        running["acc"] += batch_acc
        running["mae"] += batch_mae

        pbar.set_postfix({
            "loss": f"{running['loss']/(i+1):.4f}",
            "cls": f"{running['cls']/(i+1):.4f}",
            "reg": f"{running['reg']/(i+1):.4f}",
            "acc": f"{running['acc']/(i+1):.3f}",
            "mae": f"{running['mae']/(i+1):.3f}",
        })

        # optional memory hygiene
        if (i + 1) % 50 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    n = max(1, len(dataloader))
    return {k: v / n for k, v in running.items()}


def train_model(data_root: str,
                out_root: Path,
                model_name: str,
                model: nn.Module,
                device: torch.device,
                epochs: int = 50,
                batch_size: int = 32,
                val_ratio: float = 0.2,
                seed: int = 42,
                lr_backbone: float = 1e-4,
                lr_heads: float = 1e-3,
                num_workers: int = 0,
                alpha_cls: float = 1.0,
                beta_reg_start: float = 0.5,
                beta_reg_end: float = 1.0) -> None:
    # Data
    train_loader, val_loader = build_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=True,
    )

    # Criterion with scheduling for regression weight
    scheduler = LossWeightScheduler(
        start=LossWeights(alpha_cls=alpha_cls, beta_reg=beta_reg_start),
        end=LossWeights(alpha_cls=alpha_cls, beta_reg=beta_reg_end),
        mode="linear",
    )
    criterion = MultiTaskLoss(num_classes=8, class_weights=None, alpha_cls=alpha_cls,
                              beta_reg=beta_reg_end, scheduler=scheduler, device=device).to(device)

    # Optimizer: differential LR
    head_params = list(model.classifier.parameters()) + list(model.regressor.parameters())
    backbone_params = [p for n, p in model.named_parameters() if ("classifier" not in n and "regressor" not in n and p.requires_grad)]
    optimizer = Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_heads},
    ], weight_decay=1e-4)

    # LR scheduler on val loss
    lr_sched = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run_dir = out_root / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": []}
    best_val = float('inf')

    for epoch in range(epochs):
        train_stats = train_or_validate_epoch(model, criterion, train_loader, device, epoch, epochs,
                                              train=True, optimizer=optimizer, scaler=scaler,
                                              global_progress_offset=epoch/epochs, global_progress_span=1.0/epochs)
        val_stats = train_or_validate_epoch(model, criterion, val_loader, device, epoch, epochs,
                                            train=False)

        lr_sched.step(val_stats["loss"])

        history["train"].append(train_stats)
        history["val"].append(val_stats)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save last and best
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }
        torch.save(state, run_dir / "last.pth")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(state, run_dir / "best.pth")

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} - Train: {train_stats} - Val: {val_stats}")

    print(f"[{model_name}] Training done. Best val loss: {best_val:.4f}. Artifacts at {run_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ResNet and EfficientNet multi-task models")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base LR for heads; backbone uses 1e-4")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path("checkpoints_both") / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialize models
    resnet = MultiTaskResNet(backbone="resnet50", pretrained=True, freeze_up_to="layer2").to(device)
    effnet = MultiTaskEfficientNet(variant="efficientnet-b0", pretrained=True, freeze_until_block=8).to(device)

    # Train ResNet
    train_model(data_root=args.data_root,
                out_root=out_root,
                model_name="resnet50",
                model=resnet,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                lr_backbone=1e-4,
                lr_heads=args.lr,
                num_workers=args.num_workers)

    # Free up memory between models
    if device.type == "cuda":
        del resnet
        torch.cuda.empty_cache()

    # Train EfficientNet
    train_model(data_root=args.data_root,
                out_root=out_root,
                model_name="efficientnet_b0",
                model=effnet,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                lr_backbone=1e-4,
                lr_heads=args.lr,
                num_workers=args.num_workers)

    print(f"All training complete. Outputs in {out_root}")


if __name__ == "__main__":
    main()
