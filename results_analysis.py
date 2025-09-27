import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from facial_expression_dataset import build_dataloaders
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet

EMOTION_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]


def load_history(history_path: Path) -> Dict[str, Any]:
    with open(history_path, "r") as f:
        return json.load(f)


def plot_losses(hist_res: Dict[str, Any], hist_eff: Dict[str, Any], out_dir: Path, dpi: int = 300):
    # Common keys: loss, cls, reg, acc, mae
    def extract(hist: Dict[str, Any], split: str, key: str):
        return [e.get(key, None) for e in hist.get(split, [])]

    metrics = ["loss", "cls", "reg", "acc", "mae"]
    titles = {
        "loss": "Total Loss",
        "cls": "Classification Loss",
        "reg": "Regression Loss",
        "acc": "Accuracy",
        "mae": "Valence/Arousal MAE",
    }
    for key in metrics:
        plt.figure(figsize=(7, 4.5))
        # Train
        plt.plot(extract(hist_res, "train", key), label=f"ResNet train", color="#1f77b4", linestyle="-")
        plt.plot(extract(hist_res, "val", key), label=f"ResNet val", color="#1f77b4", linestyle="--")
        plt.plot(extract(hist_eff, "train", key), label=f"EffNet train", color="#ff7f0e", linestyle="-")
        plt.plot(extract(hist_eff, "val", key), label=f"EffNet val", color="#ff7f0e", linestyle="--")
        plt.title(f"{titles[key]} Progression")
        plt.xlabel("Epoch")
        plt.ylabel(titles[key])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{key}.png", dpi=dpi)
        plt.close()


def load_model(model_type: str, checkpoint: Path, device: torch.device) -> nn.Module:
    if model_type.lower() == "efficientnet-b0":
        model = MultiTaskEfficientNet(variant="efficientnet-b0", pretrained=False)
    else:
        model = MultiTaskResNet(backbone=model_type, pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.to(device)
    model.eval()
    return model


def collect_predictions(model: nn.Module, data_root: str, batch_size: int, num_workers: int, device: torch.device):
    _, val_loader = build_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=0.2,
        seed=42,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=False,
    )

    all_probs = []
    all_true = []
    all_va_pred = []
    all_va_true = []

    start = time.time()
    with torch.no_grad():
        for images, emotions, valences, arousals in val_loader:
            images = images.to(device)
            logits, va_pred = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_true.append(emotions.cpu())
            all_va_pred.append(va_pred.cpu())
            all_va_true.append(torch.stack([valences, arousals], dim=1))
    end = time.time()

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_true).numpy()
    va_pred = torch.cat(all_va_pred).numpy()
    va_true = torch.cat(all_va_true).numpy()

    elapsed = end - start
    n_samples = len(y_true)
    throughput = n_samples / max(1e-6, elapsed)
    return probs, y_true, va_pred, va_true, elapsed, throughput


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path, dpi: int = 300):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTION_NAMES)))
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def scatter_plot(vt: np.ndarray, vp: np.ndarray, name: str, title: str, out_path: Path, dpi: int = 300):
    plt.figure(figsize=(5.5, 5.5))
    lims = [-1.05, 1.05]
    plt.scatter(vt, vp, s=10, alpha=0.5)
    plt.plot(lims, lims, 'r--')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Pred {name}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def parameter_summary(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Results Analysis for ResNet vs EfficientNet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--resnet_history", type=str, required=True)
    parser.add_argument("--effnet_history", type=str, required=True)
    parser.add_argument("--resnet_ckpt", type=str, required=True)
    parser.add_argument("--effnet_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="analysis_outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load histories
    hist_res = load_history(Path(args.resnet_history))
    hist_eff = load_history(Path(args.effnet_history))
    plot_losses(hist_res, hist_eff, out_dir, dpi=300)

    # Load models
    resnet = load_model("resnet50", Path(args.resnet_ckpt), device)
    effnet = load_model("efficientnet-b0", Path(args.effnet_ckpt), device)

    # Parameter comparison
    res_total, res_train = parameter_summary(resnet)
    eff_total, eff_train = parameter_summary(effnet)
    with open(out_dir / "param_summary.json", "w") as f:
        json.dump({
            "resnet50": {"total": int(res_total), "trainable": int(res_train)},
            "efficientnet_b0": {"total": int(eff_total), "trainable": int(eff_train)},
        }, f, indent=2)

    # Collect predictions and timing
    res_probs, res_true, res_vp, res_vt, res_time, res_through = collect_predictions(
        resnet, args.data_root, args.batch_size, args.num_workers, device
    )
    eff_probs, eff_true, eff_vp, eff_vt, eff_time, eff_through = collect_predictions(
        effnet, args.data_root, args.batch_size, args.num_workers, device
    )

    # Sanity: ensure true labels match across runs (same val split builder)
    if not np.array_equal(res_true, eff_true):
        # Align by length if slight drift; otherwise trust res_true
        min_len = min(len(res_true), len(eff_true))
        res_true = res_true[:min_len]
        eff_true = eff_true[:min_len]
        res_probs = res_probs[:min_len]
        eff_probs = eff_probs[:min_len]
        res_vp, eff_vp = res_vp[:min_len], eff_vp[:min_len]
        res_vt, eff_vt = res_vt[:min_len], eff_vt[:min_len]

    # Accuracy progression from histories (final values noted)
    def final_metric(hist, split, key):
        arr = [e.get(key, None) for e in hist.get(split, [])]
        return arr[-1] if len(arr) else None

    # Confusion matrices
    plot_confusion(res_true, res_probs.argmax(axis=1), "Confusion Matrix - ResNet50", out_dir / "cm_resnet.png")
    plot_confusion(eff_true, eff_probs.argmax(axis=1), "Confusion Matrix - EfficientNet-B0", out_dir / "cm_effnet.png")

    # Valence/Arousal scatter plots (filter uncertain -2)
    mask_res = (res_vt.min(axis=1) > -2)
    mask_eff = (eff_vt.min(axis=1) > -2)
    if mask_res.sum() > 0:
        vt_res = res_vt[mask_res]
        vp_res = res_vp[mask_res]
        scatter_plot(vt_res[:,0], vp_res[:,0], "Valence", "Valence Scatter - ResNet50", out_dir / "scatter_valence_resnet.png")
        scatter_plot(vt_res[:,1], vp_res[:,1], "Arousal", "Arousal Scatter - ResNet50", out_dir / "scatter_arousal_resnet.png")
    if mask_eff.sum() > 0:
        vt_eff = eff_vt[mask_eff]
        vp_eff = eff_vp[mask_eff]
        scatter_plot(vt_eff[:,0], vp_eff[:,0], "Valence", "Valence Scatter - EfficientNet-B0", out_dir / "scatter_valence_effnet.png")
        scatter_plot(vt_eff[:,1], vp_eff[:,1], "Arousal", "Arousal Scatter - EfficientNet-B0", out_dir / "scatter_arousal_effnet.png")

    # Timing & accuracy comparison summary
    res_acc = accuracy_score(res_true, res_probs.argmax(axis=1))
    eff_acc = accuracy_score(eff_true, eff_probs.argmax(axis=1))
    summary = {
        "resnet50": {
            "val_accuracy": float(res_acc),
            "inference_time_sec": float(res_time),
            "throughput_samples_per_sec": float(res_through),
            "hist_final": {
                "train_acc": final_metric(hist_res, "train", "acc"),
                "val_acc": final_metric(hist_res, "val", "acc"),
                "train_loss": final_metric(hist_res, "train", "loss"),
                "val_loss": final_metric(hist_res, "val", "loss"),
            }
        },
        "efficientnet_b0": {
            "val_accuracy": float(eff_acc),
            "inference_time_sec": float(eff_time),
            "throughput_samples_per_sec": float(eff_through),
            "hist_final": {
                "train_acc": final_metric(hist_eff, "train", "acc"),
                "val_acc": final_metric(hist_eff, "val", "acc"),
                "train_loss": final_metric(hist_eff, "train", "loss"),
                "val_loss": final_metric(hist_eff, "val", "loss"),
            }
        }
    }
    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
