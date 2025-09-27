import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from scipy import stats

from facial_expression_dataset import build_dataloaders
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet

EMOTION_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def parameter_summary(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def checkpoint_size_mb(ckpt_path: Path) -> float:
    if ckpt_path.exists():
        return ckpt_path.stat().st_size / (1024 * 1024)
    return float("nan")


def run_inference(model: nn.Module, data_root: str, batch_size: int, num_workers: int, device: torch.device):
    """Run through the validation split, collect predictions and measure throughput."""
    _, val_loader = build_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=0.2,
        seed=42,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=False,
    )

    all_probs, all_true = [], []
    warmup = 3
    start = None

    with torch.no_grad():
        for i, (images, emotions, _, _) in enumerate(tqdm(val_loader, desc="Inference", ncols=100)):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i == warmup:
                # start timing after a few warmup batches
                start = time.time()
            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_true.append(emotions.cpu())
        if start is None:
            start = time.time()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_true).numpy()
    elapsed = end - start
    n_timed = probs.shape[0] - warmup * batch_size
    n_timed = max(1, n_timed)
    throughput = n_timed / max(1e-6, elapsed)
    return probs, y_true, elapsed, throughput


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, name in enumerate(EMOTION_NAMES):
        mask = (y_true == i)
        if mask.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float((y_pred[mask] == y_true[mask]).mean())
    return out


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test for paired nominal outcomes (significance of accuracy difference)."""
    # Contingency for disagreements
    both_correct = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    a_correct_b_wrong = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    b_correct_a_wrong = np.sum((y_pred_b == y_true) & (y_pred_a != y_true))
    both_wrong = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))

    b = a_correct_b_wrong
    c = b_correct_a_wrong

    # Exact binomial test on b vs c (two-sided)
    n = b + c
    if n == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        # null: b == c, P(success)=0.5
        res = stats.binomtest(k=min(b, c), n=n, p=0.5, alternative='two-sided')
        p_value = float(res.pvalue)
        # also compute McNemar's chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0

    return {
        "a_correct_b_wrong": int(a_correct_b_wrong),
        "b_correct_a_wrong": int(b_correct_a_wrong),
        "both_correct": int(both_correct),
        "both_wrong": int(both_wrong),
        "chi2_cc": float(chi2),
        "p_value": float(p_value),
    }


def summarize_table(res_metrics: Dict[str, Any], eff_metrics: Dict[str, Any], out_dir: Path):
    # Save a CSV and Markdown table focused on accuracy & time
    import pandas as pd
    rows = []
    rows.append({
        "Model": "ResNet50",
        "Val Accuracy": res_metrics["val_accuracy"],
        "Throughput (img/s)": res_metrics["throughput"],
        "Inference Time (s)": res_metrics["inference_time"],
        "Params (M)": res_metrics["params_total"]/1e6,
        "Trainable (M)": res_metrics["params_trainable"]/1e6,
        "Model Size (MB)": res_metrics["model_size_mb"],
        "Epochs to Converge": res_metrics.get("epochs_to_converge", None),
    })
    rows.append({
        "Model": "EfficientNet-B0",
        "Val Accuracy": eff_metrics["val_accuracy"],
        "Throughput (img/s)": eff_metrics["throughput"],
        "Inference Time (s)": eff_metrics["inference_time"],
        "Params (M)": eff_metrics["params_total"]/1e6,
        "Trainable (M)": eff_metrics["params_trainable"]/1e6,
        "Model Size (MB)": eff_metrics["model_size_mb"],
        "Epochs to Converge": eff_metrics.get("epochs_to_converge", None),
    })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_table.csv", index=False)
    with open(out_dir / "comparison_table.md", "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".3f"))


def epochs_to_convergence(history_path: Path, key: str = "val", metric: str = "loss") -> int:
    try:
        with open(history_path, "r") as f:
            hist = json.load(f)
        vals = [e.get(metric, None) for e in hist.get(key, [])]
        if not vals:
            return -1
        best_idx = int(np.nanargmin(vals))
        return best_idx + 1  # epochs are 1-indexed
    except Exception:
        return -1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model comparison: ResNet vs EfficientNet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--resnet_ckpt", type=str, required=True)
    parser.add_argument("--effnet_ckpt", type=str, required=True)
    parser.add_argument("--resnet_history", type=str, required=True)
    parser.add_argument("--effnet_history", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="model_comparison_outputs")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    resnet = load_model("resnet50", Path(args.resnet_ckpt), device)
    effnet = load_model("efficientnet-b0", Path(args.effnet_ckpt), device)

    # Complexity
    res_total, res_train = parameter_summary(resnet)
    eff_total, eff_train = parameter_summary(effnet)

    res_size = checkpoint_size_mb(Path(args.resnet_ckpt))
    eff_size = checkpoint_size_mb(Path(args.effnet_ckpt))

    # Inference speed and accuracy
    res_probs, res_true, res_time, res_through = run_inference(resnet, args.data_root, args.batch_size, args.num_workers, device)
    eff_probs, eff_true, eff_time, eff_through = run_inference(effnet, args.data_root, args.batch_size, args.num_workers, device)

    # Align labels if any mismatch (shouldn't happen with same builder/seed)
    if not np.array_equal(res_true, eff_true):
        min_len = min(len(res_true), len(eff_true))
        res_true = res_true[:min_len]
        eff_true = eff_true[:min_len]
        res_probs = res_probs[:min_len]
        eff_probs = eff_probs[:min_len]

    res_pred = res_probs.argmax(axis=1)
    eff_pred = eff_probs.argmax(axis=1)

    res_acc = accuracy_score(res_true, res_pred)
    eff_acc = accuracy_score(eff_true, eff_pred)

    # Per-class analysis
    res_per_class = per_class_accuracy(res_true, res_pred)
    eff_per_class = per_class_accuracy(eff_true, eff_pred)

    # Significance test (McNemar)
    sig = mcnemar_test(res_true, res_pred, eff_pred)

    # Epochs to convergence from histories
    res_epochs_conv = epochs_to_convergence(Path(args.resnet_history))
    eff_epochs_conv = epochs_to_convergence(Path(args.effnet_history))

    # Save metrics summary
    summary = {
        "resnet50": {
            "val_accuracy": float(res_acc),
            "per_class_accuracy": res_per_class,
            "inference_time": float(res_time),
            "throughput": float(res_through),
            "params_total": int(res_total),
            "params_trainable": int(res_train),
            "model_size_mb": float(res_size),
            "epochs_to_converge": int(res_epochs_conv),
        },
        "efficientnet_b0": {
            "val_accuracy": float(eff_acc),
            "per_class_accuracy": eff_per_class,
            "inference_time": float(eff_time),
            "throughput": float(eff_through),
            "params_total": int(eff_total),
            "params_trainable": int(eff_train),
            "model_size_mb": float(eff_size),
            "epochs_to_converge": int(eff_epochs_conv),
        },
        "significance": sig,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Produce performance comparison table (CSV + Markdown)
    summarize_table(summary["resnet50"], summary["efficientnet_b0"], out_dir)

    # Save per-class comparison table
    import pandas as pd
    df_pc = pd.DataFrame({
        "Class": EMOTION_NAMES,
        "ResNet50": [summary["resnet50"]["per_class_accuracy"].get(n, np.nan) for n in EMOTION_NAMES],
        "EfficientNet-B0": [summary["efficientnet_b0"]["per_class_accuracy"].get(n, np.nan) for n in EMOTION_NAMES],
    })
    df_pc["Delta (Eff-Res)"] = df_pc["EfficientNet-B0"] - df_pc["ResNet50"]
    df_pc.to_csv(out_dir / "per_class_accuracy.csv", index=False)
    with open(out_dir / "per_class_accuracy.md", "w") as f:
        f.write(df_pc.to_markdown(index=False, floatfmt=".3f"))

    print(f"Model comparison complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
