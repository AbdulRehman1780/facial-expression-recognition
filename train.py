import os
import json
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Local modules
from facial_expression_dataset import build_dataloaders
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet
from multitask_loss import MultiTaskLoss, LossWeightScheduler, LossWeights


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.numel()
        return correct / max(1, total)


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (pred - target).abs().mean().item()


def train_one_epoch(model: nn.Module,
                    criterion: MultiTaskLoss,
                    dataloader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    scheduler: ReduceLROnPlateau = None,
                    global_step: int = 0) -> Dict[str, float]:
    model.train()

    running_total = 0.0
    running_cls = 0.0
    running_reg = 0.0
    running_acc = 0.0
    running_reg_mae = 0.0

    pbar = tqdm(dataloader, desc=f"Train [{epoch+1}/{total_epochs}]", ncols=100)
    for i, batch in enumerate(pbar):
        images, emotions, valences, arousals = batch
        images = images.to(device, non_blocking=True)
        emotions = emotions.to(device, non_blocking=True)
        va_targets = torch.stack([valences, arousals], dim=1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, va_pred = model(images)

        # progress for loss scheduling (0..1 over entire training)
        progress = (epoch + i / max(1, len(dataloader))) / max(1, total_epochs)
        loss_dict = criterion(logits, va_pred, emotions, va_targets, progress=progress)

        loss = loss_dict["total"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # metrics
        batch_acc = accuracy(logits, emotions)
        batch_mae = mae(va_pred, va_targets)

        running_total += loss.item()
        running_cls += loss_dict["cls"].item()
        running_reg += loss_dict["reg"].item()
        running_acc += batch_acc
        running_reg_mae += batch_mae

        pbar.set_postfix({
            "loss": f"{running_total/(i+1):.4f}",
            "cls": f"{running_cls/(i+1):.4f}",
            "reg": f"{running_reg/(i+1):.4f}",
            "acc": f"{running_acc/(i+1):.3f}",
            "mae": f"{running_reg_mae/(i+1):.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    epoch_stats = {
        "loss": running_total / max(1, len(dataloader)),
        "cls": running_cls / max(1, len(dataloader)),
        "reg": running_reg / max(1, len(dataloader)),
        "acc": running_acc / max(1, len(dataloader)),
        "mae": running_reg_mae / max(1, len(dataloader)),
    }

    return epoch_stats


def validate_one_epoch(model: nn.Module,
                       criterion: MultiTaskLoss,
                       dataloader,
                       device: torch.device,
                       epoch: int,
                       total_epochs: int) -> Dict[str, float]:
    model.eval()

    running_total = 0.0
    running_cls = 0.0
    running_reg = 0.0
    running_acc = 0.0
    running_reg_mae = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val   [{epoch+1}/{total_epochs}]", ncols=100)
        for i, batch in enumerate(pbar):
            images, emotions, valences, arousals = batch
            images = images.to(device, non_blocking=True)
            emotions = emotions.to(device, non_blocking=True)
            va_targets = torch.stack([valences, arousals], dim=1).to(device, non_blocking=True)

            logits, va_pred = model(images)
            # Use static weights in validation (no scheduling)
            loss_dict = criterion(logits, va_pred, emotions, va_targets, progress=None)

            loss = loss_dict["total"]
            batch_acc = accuracy(logits, emotions)
            batch_mae = mae(va_pred, va_targets)

            running_total += loss.item()
            running_cls += loss_dict["cls"].item()
            running_reg += loss_dict["reg"].item()
            running_acc += batch_acc
            running_reg_mae += batch_mae

            pbar.set_postfix({
                "loss": f"{running_total/(i+1):.4f}",
                "cls": f"{running_cls/(i+1):.4f}",
                "reg": f"{running_reg/(i+1):.4f}",
                "acc": f"{running_acc/(i+1):.3f}",
                "mae": f"{running_reg_mae/(i+1):.3f}",
            })

    epoch_stats = {
        "loss": running_total / max(1, len(dataloader)),
        "cls": running_cls / max(1, len(dataloader)),
        "reg": running_reg / max(1, len(dataloader)),
        "acc": running_acc / max(1, len(dataloader)),
        "mae": running_reg_mae / max(1, len(dataloader)),
    }

    return epoch_stats


def save_checkpoint(state: Dict[str, Any], is_best: bool, out_dir: Path, filename: str = "last.pth"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / filename)
    if is_best:
        torch.save(state, out_dir / "best.pth")


def main():
    parser = argparse.ArgumentParser(description="Multi-task Facial Expression Training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root containing images/ and annotations/")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18","resnet34","resnet50","resnet101","efficientnet-b0"], help="Backbone model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--alpha_cls", type=float, default=1.0)
    parser.add_argument("--beta_reg_start", type=float, default=0.5)
    parser.add_argument("--beta_reg_end", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_augment", action="store_true", help="Disable train-time augmentations")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_loader, val_loader = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        augment=not args.no_augment,
    )

    # Model
    if args.model == "efficientnet-b0":
        model = MultiTaskEfficientNet(variant="efficientnet-b0", pretrained=True, freeze_until_block=8)
    else:
        model = MultiTaskResNet(backbone=args.model, pretrained=True, freeze_up_to="layer2")
    model = model.to(device)

    # Loss: compute class weights from training distribution if possible (optional)
    # Here we leave class_weights=None as default. You can optionally compute and pass.
    scheduler = LossWeightScheduler(
        start=LossWeights(alpha_cls=args.alpha_cls, beta_reg=args.beta_reg_start),
        end=LossWeights(alpha_cls=args.alpha_cls, beta_reg=args.beta_reg_end),
        mode="linear",
    )
    criterion = MultiTaskLoss(
        num_classes=8,
        class_weights=None,
        alpha_cls=args.alpha_cls,
        beta_reg=args.beta_reg_end,
        scheduler=scheduler,
        device=device,
    ).to(device)

    # Optimizer with differential LRs
    head_params = list(model.classifier.parameters()) + list(model.regressor.parameters())
    backbone_params = [p for n, p in model.named_parameters() if ("classifier" not in n and "regressor" not in n and p.requires_grad)]
    optimizer = Adam([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)

    # LR scheduler on validation loss
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Checkpointing
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_val = math.inf
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.epochs)
        val_stats = validate_one_epoch(model, criterion, val_loader, device, epoch, args.epochs)

        lr_scheduler.step(val_stats["loss"])

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        # Save history incrementally
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Checkpointing: best only
        is_best = val_stats["loss"] < best_val
        if is_best:
            best_val = val_stats["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
            "args": vars(args),
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        save_checkpoint(save_state, is_best=is_best, out_dir=out_dir)

        # Early stopping
        print(f"Epoch {epoch+1}/{args.epochs} - train: {train_stats} - val: {val_stats}")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"Training complete. Best val loss: {best_val:.4f}. Checkpoints at: {out_dir}")


if __name__ == "__main__":
    main()
