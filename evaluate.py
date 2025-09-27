import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr

from facial_expression_dataset import build_dataloaders
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet

EMOTION_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def corr_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r)
    except Exception:
        return float("nan")


def sagr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Sign Agreement Metric: proportion of samples where sign matches
    # Use sgn(x) = +1 if x >= 0, -1 otherwise
    s_true = np.where(y_true >= 0, 1, -1)
    s_pred = np.where(y_pred >= 0, 1, -1)
    return float((s_true == s_pred).mean())


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Concordance Correlation Coefficient
    x = y_true.astype(np.float64)
    y = y_pred.astype(np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    rho = np.corrcoef(x, y)[0, 1]
    return float((2 * rho * sx * sy) / (vx + vy + (mx - my) ** 2))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Use sklearn if available
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


def krippendorffs_alpha_nominal(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 8) -> float:
    # Krippendorff's alpha for nominal data with 2 raters
    # Build coincidence matrix C
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1.0
        if t != p:
            # symmetric contribution for pair counting across raters
            C[int(p), int(t)] += 1.0
    # Observed disagreement Do
    Do = C.sum() - np.trace(C)
    # Expected disagreement De from marginals
    n_c = C.sum(axis=1)
    N = n_c.sum()
    De = (N * N - (n_c @ n_c))
    if De == 0:
        return float("nan")
    alpha = 1.0 - (Do / De)
    return float(alpha)


def load_model(model_type: str, checkpoint: Path, device: torch.device) -> nn.Module:
    if model_type.lower() == "efficientnet-b0":
        model = MultiTaskEfficientNet(variant="efficientnet-b0", pretrained=False)
    else:
        # default to resnet50 family names
        model = MultiTaskResNet(backbone=model_type, pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.to(device)
    model.eval()
    return model


def collect_predictions(model: nn.Module, data_root: str, batch_size: int, num_workers: int, device: torch.device):
    # Use validation split from our dataloader builder
    train_loader, val_loader = build_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=0.2,
        seed=42,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        augment=False,
    )
    loader = val_loader

    all_logits = []
    all_probs = []
    all_cls = []
    all_va_pred = []
    all_va_true = []

    with torch.no_grad():
        for images, emotions, valences, arousals in loader:
            images = images.to(device)
            logits, va_pred = model(images)
            probs = torch.softmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_cls.append(emotions.cpu())

            va_targets = torch.stack([valences, arousals], dim=1)
            all_va_pred.append(va_pred.cpu())
            all_va_true.append(va_targets.cpu())

    logits = torch.cat(all_logits).numpy()
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_cls).numpy()
    va_pred = torch.cat(all_va_pred).numpy()
    va_true = torch.cat(all_va_true).numpy()

    return logits, probs, y_true, va_pred, va_true


def evaluate_classification(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(len(EMOTION_NAMES)))

    # per-class accuracy
    per_class_acc = []
    for c in range(len(EMOTION_NAMES)):
        idx = (y_true == c)
        if idx.sum() == 0:
            per_class_acc.append(float("nan"))
        else:
            per_class_acc.append(float((y_pred[idx] == y_true[idx]).mean()))

    # Cohen's Kappa
    kappa = cohens_kappa(y_true, y_pred)

    # Krippendorff's Alpha (nominal)
    alpha = krippendorffs_alpha_nominal(y_true, y_pred, n_classes=len(EMOTION_NAMES))

    # AUC-ROC one-vs-rest (needs probabilities)
    y_true_bin = label_binarize(y_true, classes=np.arange(len(EMOTION_NAMES)))
    try:
        auc_roc_macro = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc_roc_macro = float("nan")

    # AUC-PR (Average Precision)
    try:
        auc_pr_macro = average_precision_score(y_true_bin, probs, average="macro")
    except Exception:
        auc_pr_macro = float("nan")

    # Confusion Matrix and Report
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTION_NAMES)))
    report = classification_report(y_true, y_pred, target_names=EMOTION_NAMES, digits=4)

    return {
        "accuracy": float(acc),
        "per_class_accuracy": per_class_acc,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_per_class.tolist(),
        "cohen_kappa": float(kappa),
        "krippendorff_alpha": float(alpha),
        "auc_roc_macro_ovr": float(auc_roc_macro),
        "auc_pr_macro": float(auc_pr_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_continuous(va_true: np.ndarray, va_pred: np.ndarray) -> Dict:
    # Handle -2 (uncertain) if present
    mask = np.all(va_true > -2, axis=1)
    if mask.sum() == 0:
        return {
            "rmse_valence": None,
            "rmse_arousal": None,
            "pearson_valence": None,
            "pearson_arousal": None,
            "sagr_valence": None,
            "sagr_arousal": None,
            "ccc_valence": None,
            "ccc_arousal": None,
            "n_samples": 0,
        }

    vt = va_true[mask]
    vp = va_pred[mask]

    rv = rmse(vt[:, 0], vp[:, 0])
    ra = rmse(vt[:, 1], vp[:, 1])
    pv = corr_pearson(vt[:, 0], vp[:, 0])
    pa = corr_pearson(vt[:, 1], vp[:, 1])
    sv = sagr(vt[:, 0], vp[:, 0])
    sa = sagr(vt[:, 1], vp[:, 1])
    cv = ccc(vt[:, 0], vp[:, 0])
    ca = ccc(vt[:, 1], vp[:, 1])

    return {
        "rmse_valence": rv,
        "rmse_arousal": ra,
        "pearson_valence": pv,
        "pearson_arousal": pa,
        "sagr_valence": sv,
        "sagr_arousal": sa,
        "ccc_valence": cv,
        "ccc_arousal": ca,
        "n_samples": int(mask.sum()),
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curves(y_true: np.ndarray, probs: np.ndarray, out_path: Path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_true_bin = label_binarize(y_true, classes=np.arange(len(EMOTION_NAMES)))

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(EMOTION_NAMES):
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
            ap = average_precision_score(y_true_bin[:, i], probs[:, i])
            plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (OvR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, out_path: Path):
    from sklearn.metrics import roc_curve, auc
    y_true_bin = label_binarize(y_true, classes=np.arange(len(EMOTION_NAMES)))

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(EMOTION_NAMES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (OvR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, name: str, out_path: Path, stats: Dict):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    lims = [-1.05, 1.05]
    plt.plot(lims, lims, 'r--', label='y=x')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Pred {name}")
    plt.title(f"{name} Scatter: RMSE={stats[f'rmse_{name.lower()}']:.3f}, r={stats[f'pearson_{name.lower()}']:.3f}, CCC={stats[f'ccc_{name.lower()}']:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Script")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="resnet50", choices=["resnet18","resnet34","resnet50","resnet101","efficientnet-b0"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (best.pth or last.pth)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_type, Path(args.checkpoint), device)

    # Collect predictions
    logits, probs, y_true, va_pred, va_true = collect_predictions(
        model, args.data_root, args.batch_size, args.num_workers, device
    )

    # Classification metrics
    cls_metrics = evaluate_classification(y_true, probs)

    # Continuous metrics
    cont_metrics = evaluate_continuous(va_true, va_pred)

    # Save metrics JSON
    metrics = {
        "classification": cls_metrics,
        "continuous": cont_metrics,
        "n_samples_cls": int(len(y_true)),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(cls_metrics["classification_report"])  # type: ignore

    # Plots
    plot_confusion_matrix(np.array(cls_metrics["confusion_matrix"]), out_dir / "confusion_matrix.png")
    plot_pr_curves(y_true, probs, out_dir / "pr_curves.png")
    plot_roc_curves(y_true, probs, out_dir / "roc_curves.png")

    # Scatter plots for VA
    if cont_metrics["n_samples"] > 0:
        vt = va_true[va_true.min(axis=1) > -2]
        vp = va_pred[va_pred.shape[0]-vt.shape[0]:]  # align lengths safely
        plot_scatter(vt[:, 0], vp[:, 0], "Valence", out_dir / "scatter_valence.png", cont_metrics)
        plot_scatter(vt[:, 1], vp[:, 1], "Arousal", out_dir / "scatter_arousal.png", cont_metrics)

    print(f"Evaluation complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
