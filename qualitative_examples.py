import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from facial_expression_dataset import FacialExpressionDataset
from multitask_resnet import MultiTaskResNet
from multitask_efficientnet import MultiTaskEfficientNet

EMOTION_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]


def set_seed(seed: int = 42):
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


def pick_examples(ds: FacialExpressionDataset, model: nn.Module, device: torch.device, n_correct: int, n_incorrect: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(ds))
    rng.shuffle(idxs)

    correct, incorrect = [], []

    for idx in idxs:
        img_path, true_emotion, valence, arousal = ds.items[idx]
        with Image.open(img_path) as img:
            rgb = img.convert('RGB').resize((224, 224))
            # tensor for inference (normalize like dataset)
            x = torch.from_numpy(np.asarray(rgb).transpose(2,0,1)).float()/255.0
            mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            x = (x - mean) / std
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, va_pred = model(x)
                pred = int(torch.argmax(logits, dim=1).item())
                va = va_pred.squeeze(0).cpu().numpy().tolist()
        rec = {
            "path": str(img_path),
            "img": rgb.copy(),
            "true": int(true_emotion),
            "pred": pred,
            "true_name": EMOTION_NAMES[int(true_emotion)] if 0 <= int(true_emotion) < len(EMOTION_NAMES) else str(true_emotion),
            "pred_name": EMOTION_NAMES[pred] if 0 <= pred < len(EMOTION_NAMES) else str(pred),
            "va_pred": va,
        }
        if pred == int(true_emotion) and len(correct) < n_correct:
            correct.append(rec)
        elif pred != int(true_emotion) and len(incorrect) < n_incorrect:
            incorrect.append(rec)
        if len(correct) >= n_correct and len(incorrect) >= n_incorrect:
            break

    return correct, incorrect


def save_grid(examples: List[Dict], title: str, out_path: Path, cols: int = 4, dpi: int = 300):
    if not examples:
        return
    rows = (len(examples) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, ex in enumerate(examples):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(ex["img"])  # already RGB
        ax.axis('off')
        ax.set_title(f"T:{ex['true_name']}\nP:{ex['pred_name']}", fontsize=9)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Save grids of correct and incorrect classifications")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="resnet50", choices=["resnet18","resnet34","resnet50","resnet101","efficientnet-b0"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="qualitative_outputs")
    parser.add_argument("--n_correct", type=int, default=8)
    parser.add_argument("--n_incorrect", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build validation dataset (no augmentation), reusing your dataset split
    val_ds = FacialExpressionDataset(args.data_root, split='val', augment=False)

    # Load model
    model = load_model(args.model_type, Path(args.checkpoint), device)

    # Pick examples
    correct, incorrect = pick_examples(val_ds, model, device, args.n_correct, args.n_incorrect, args.seed)

    out_dir = Path(args.out_dir)
    save_grid(correct, f"Correct ({args.model_type})", out_dir / f"correct_{args.model_type}.png")
    save_grid(incorrect, f"Incorrect ({args.model_type})", out_dir / f"incorrect_{args.model_type}.png")

    print(f"Saved: {out_dir / f'correct_{args.model_type}.png'}")
    print(f"Saved: {out_dir / f'incorrect_{args.model_type}.png'}")


if __name__ == "__main__":
    main()
