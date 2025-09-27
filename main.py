"""
Main single-file pipeline for facial expression multi-task learning.
Includes:
- Dataset with augmentation and .npy annotation parsing
- Multi-task ResNet and EfficientNet models
- Training loop with validation and AMP
- Evaluation metrics (classification + continuous VA)
- Visualization utilities (curves, confusion matrix, ROC/PR, VA scatter)
- Model comparison analysis and simple report generation
- CLI phases: data_exploration | train_models | evaluate_models | generate_report

Usage examples:
  python main.py --phase data_exploration --data_root path/to/Dataset
  python main.py --phase train_models --data_root path/to/Dataset --epochs 10
  python main.py --phase evaluate_models --data_root path/to/Dataset --resnet_ckpt checkpoints/.../resnet50/best.pth --effnet_ckpt checkpoints/.../efficientnet_b0/best.pth
  python main.py --phase generate_report --data_root path/to/Dataset --resnet_ckpt ... --effnet_ckpt ...
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Optional libs (guarded)
try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None  # type: ignore

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None  # type: ignore

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
        auc,
    )
    from sklearn.preprocessing import label_binarize
    from scipy.stats import pearsonr
    from scipy import stats as scipy_stats
except Exception:
    accuracy_score = None  # type: ignore

EMOTION_NAMES = [
    "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"
]

# =============================
# Dataset
# =============================
class FacialExpressionDataset(Dataset):
    """Dataset reading 224x224 RGB images and .npy annotations.

    Expects structure:
      root/images/<id>.jpg
      root/annotations/<id>_exp.npy, <id>_val.npy, <id>_aro.npy
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform: Optional[Callable] = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.split = split
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.user_transform = transform
        self.augment = augment

        if not self.images_dir.exists() or not self.ann_dir.exists():
            raise FileNotFoundError("Missing images/ or annotations/ directory under root")

        items = self._collect_items()
        self.train_items, self.val_items = self._split_items(items, self.val_ratio, self.seed)
        self.items = self.train_items if self.split == "train" else self.val_items

        self.transform = self._build_transforms()

    def _collect_items(self) -> List[Tuple[Path, int, float, float]]:
        image_paths: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(self.images_dir.rglob(ext))
        items: List[Tuple[Path, int, float, float]] = []
        for img in sorted(image_paths, key=lambda p: p.stem):
            sid = img.stem
            exp_p = self.ann_dir / f"{sid}_exp.npy"
            val_p = self.ann_dir / f"{sid}_val.npy"
            aro_p = self.ann_dir / f"{sid}_aro.npy"
            if not (exp_p.exists() and val_p.exists() and aro_p.exists()):
                continue
            try:
                emotion = self._parse_emotion(exp_p)
                valence = self._parse_float(val_p)
                arousal = self._parse_float(aro_p)
            except Exception:
                continue
            # filter uncertain
            if valence == -2 or arousal == -2:
                continue
            valence = float(np.clip(valence, -1.0, 1.0))
            arousal = float(np.clip(arousal, -1.0, 1.0))
            if not (0 <= emotion <= 7):
                continue
            items.append((img, emotion, valence, arousal))
        if not items:
            raise RuntimeError("No valid samples found")
        return items

    @staticmethod
    def _load_npy(path: Path):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        return data

    def _parse_emotion(self, path: Path) -> int:
        d = self._load_npy(path)
        if isinstance(d, (int, np.integer)):
            return int(d)
        if isinstance(d, (float, np.floating)):
            return int(round(float(d)))
        if isinstance(d, str):
            return int(d.split(";")[0])
        if isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0:
            return int(str(d[0]).split(";")[0])
        raise ValueError("Unsupported emotion format")

    def _parse_float(self, path: Path) -> float:
        d = self._load_npy(path)
        if isinstance(d, (int, float, np.integer, np.floating)):
            return float(d)
        if isinstance(d, str):
            return float(d.split(";")[0].replace(",", "."))
        if isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0:
            return float(str(d[0]).split(";")[0].replace(",", "."))
        raise ValueError("Unsupported float format")

    def _split_items(self, items: List[Tuple[Path, int, float, float]], val_ratio: float, seed: int):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_val = int(round(len(items) * val_ratio))
        val_idx = set(idx[:n_val].tolist())
        train_items, val_items = [], []
        for i, it in enumerate(items):
            (val_items if i in val_idx else train_items).append(it)
        return train_items, val_items

    def _build_transforms(self) -> Callable:
        if self.user_transform:
            return self.user_transform
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        if self.split == "train" and self.augment:
            return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, float]:
        img_path, emotion, valence, arousal = self.items[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        return x, int(emotion), float(valence), float(arousal)


def collate_with_targets(batch):
    imgs, emos, vals, aros = zip(*batch)
    return torch.stack(imgs,0), torch.as_tensor(emos), torch.as_tensor(vals), torch.as_tensor(aros)


def build_dataloaders(root: str, batch_size: int=32, val_ratio: float=0.2, seed: int=42, num_workers: int=0, pin_memory: bool=False, augment: bool=True):
    train_ds = FacialExpressionDataset(root, split='train', val_ratio=val_ratio, seed=seed, augment=augment)
    val_ds = FacialExpressionDataset(root, split='val', val_ratio=val_ratio, seed=seed, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_with_targets)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_with_targets)
    return train_loader, val_loader

# =============================
# Models
# =============================
class MultiTaskResNet(nn.Module):
    def __init__(self, backbone: str='resnet50', pretrained: bool=True, freeze_up_to: str='layer2', cls_dropout: float=0.3, reg_dropout: float=0.2) -> None:
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for ResNet")
        name = backbone.lower()
        ctor = getattr(tv_models, name)
        # handle weights API
        try:
            weights_attr = getattr(tv_models, f"{name}_Weights", None)
            weights = getattr(weights_attr, 'DEFAULT', None) if (pretrained and weights_attr) else None
            backbone_m = ctor(weights=weights)
        except TypeError:
            backbone_m = ctor(pretrained=pretrained)
        in_feat = backbone_m.fc.in_features
        self.backbone = nn.Sequential(*list(backbone_m.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(in_feat,512), nn.ReLU(True), nn.Dropout(cls_dropout), nn.Linear(512,8))
        self.regressor = nn.Sequential(nn.Linear(in_feat,256), nn.ReLU(True), nn.Dropout(reg_dropout), nn.Linear(256,2))
        self._freeze_until(backbone_m, freeze_up_to)

    def _freeze_until(self, backbone_m: nn.Module, up_to: Optional[str]) -> None:
        stages = [
            ("conv1", backbone_m.conv1), ("bn1", backbone_m.bn1),
            ("layer1", backbone_m.layer1), ("layer2", backbone_m.layer2), ("layer3", backbone_m.layer3)
        ]
        freeze = True
        for nm, mod in stages:
            for p in mod.parameters():
                p.requires_grad = freeze
            if up_to is not None and nm == up_to:
                freeze = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x).flatten(1)
        return self.classifier(f), self.regressor(f)

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, variant: str='efficientnet-b0', pretrained: bool=True, freeze_until_block: int=8, cls_dropout: float=0.3, reg_dropout: float=0.2) -> None:
        super().__init__()
        if EfficientNet is None:
            raise ImportError("efficientnet_pytorch is required for EfficientNet")
        self.backbone = EfficientNet.from_pretrained(variant) if pretrained else EfficientNet.from_name(variant)
        in_feat = self.backbone._fc.in_features  # 1280 for B0
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.Linear(in_feat,512), nn.ReLU(True), nn.Dropout(cls_dropout), nn.Linear(512,8))
        self.regressor = nn.Sequential(nn.Linear(in_feat,256), nn.ReLU(True), nn.Dropout(reg_dropout), nn.Linear(256,2))
        # freeze stem + early blocks
        for p in self.backbone._conv_stem.parameters(): p.requires_grad=False
        for p in self.backbone._bn0.parameters(): p.requires_grad=False
        for i, blk in enumerate(self.backbone._blocks):
            if i < freeze_until_block:
                for p in blk.parameters(): p.requires_grad=False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone.extract_features(x)
        f = self.pool(f).flatten(1)
        return self.classifier(f), self.regressor(f)

# =============================
# Loss & Metrics
# =============================
@dataclass
class LossWeights:
    alpha_cls: float = 1.0
    beta_reg: float = 1.0

class LossWeightScheduler:
    def __init__(self, start: LossWeights, end: LossWeights, mode: str='linear') -> None:
        self.start, self.end, self.mode = start, end, mode
    def __call__(self, p: float) -> LossWeights:
        p = float(min(1.0, max(0.0, p)))
        if self.mode == 'linear':
            a = self.start.alpha_cls + p*(self.end.alpha_cls - self.start.alpha_cls)
            b = self.start.beta_reg + p*(self.end.beta_reg - self.start.beta_reg)
            return LossWeights(a,b)
        # default
        return self.start

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes: int=8, class_weights: Optional[torch.Tensor]=None, alpha_cls: float=1.0, beta_reg: float=1.0, scheduler: Optional[LossWeightScheduler]=None, device: Optional[torch.device]=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('class_weights', class_weights.float() if class_weights is not None else None, persistent=False)
        self.alpha_cls = float(alpha_cls)
        self.beta_reg = float(beta_reg)
        self.scheduler = scheduler
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
    def forward(self, logits: torch.Tensor, va_pred: torch.Tensor, targets_cls: torch.Tensor, targets_va: torch.Tensor, progress: Optional[float]=None) -> Dict[str, torch.Tensor]:
        dev = logits.device
        w = self.scheduler(progress) if (self.scheduler is not None and progress is not None) else LossWeights(self.alpha_cls, self.beta_reg)
        cw = self.class_weights.to(dev) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, targets_cls, weight=cw)
        mse = self.mse(va_pred, targets_va)
        total = w.alpha_cls*ce + w.beta_reg*mse
        return { 'total': total, 'cls': ce.detach(), 'reg': mse.detach(), 'alpha': torch.tensor(w.alpha_cls, device=dev), 'beta': torch.tensor(w.beta_reg, device=dev) }

# Utility metrics

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(1) == targets).float().mean().item())

def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        return float((a-b).abs().mean().item())

# =============================
# Training / Validation
# =============================

def train_one_epoch(model: nn.Module, criterion: MultiTaskLoss, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, total_epochs: int, scaler: Optional[torch.cuda.amp.GradScaler]=None) -> Dict[str,float]:
    model.train()
    run = { 'loss':0.0, 'cls':0.0, 'reg':0.0, 'acc':0.0, 'mae':0.0 }
    for i, (imgs, emos, vals, aros) in enumerate(loader):
        imgs, emos = imgs.to(device), emos.to(device)
        va = torch.stack([vals, aros],1).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logits, vp = model(imgs)
            prog = (epoch + i/max(1,len(loader)))/max(1,total_epochs)
            losses = criterion(logits, vp, emos, va, progress=prog)
            loss = losses['total']
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type=='cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        run['loss']+=loss.item(); run['cls']+=losses['cls'].item(); run['reg']+=losses['reg'].item(); run['acc']+=accuracy(logits,emos); run['mae']+=mae(vp,va)
    n = max(1,len(loader)); return {k:v/n for k,v in run.items()}

def validate_one_epoch(model: nn.Module, criterion: MultiTaskLoss, loader: DataLoader, device: torch.device, epoch: int, total_epochs: int) -> Dict[str,float]:
    model.eval(); run={ 'loss':0.0, 'cls':0.0, 'reg':0.0, 'acc':0.0, 'mae':0.0 }
    with torch.no_grad():
        for imgs, emos, vals, aros in loader:
            imgs, emos = imgs.to(device), emos.to(device)
            va = torch.stack([vals, aros],1).to(device)
            logits, vp = model(imgs)
            losses = criterion(logits, vp, emos, va, progress=None)
            run['loss']+=losses['total'].item(); run['cls']+=losses['cls'].item(); run['reg']+=losses['reg'].item(); run['acc']+=accuracy(logits,emos); run['mae']+=mae(vp,va)
    n=max(1,len(loader)); return {k:v/n for k,v in run.items()}

# =============================
# Evaluation Utilities
# =============================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def corr_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if pearsonr is None or y_true.size<2: return float('nan')
    r,_ = pearsonr(y_true, y_pred); return float(r)

def sagr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = np.where(y_true>=0,1,-1); s_pred = np.where(y_pred>=0,1,-1); return float((s_true==s_pred).mean())

def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    x,y = y_true.astype(np.float64), y_pred.astype(np.float64)
    mx,my=x.mean(),y.mean(); vx,vy=x.var(),y.var(); sx,sy=x.std(),y.std()
    if sx==0 or sy==0: return float('nan')
    rho = np.corrcoef(x,y)[0,1]
    return float((2*rho*sx*sy)/(vx+vy+(mx-my)**2))

# =============================
# Phases
# =============================

def phase_data_exploration(data_root: str) -> None:
    root = Path(data_root)
    print(f"Dataset root: {root}")
    total_images = sum(1 for _ in (root/"images").rglob("*.jpg"))
    print(f"Images found: {total_images}")
    npy_files = list((root/"annotations").glob("*.npy"))
    print(f"NPY files in annotations/: {len(npy_files)} (show 5)")
    for p in npy_files[:5]:
        try:
            arr = np.load(p, allow_pickle=True)
            print(f"- {p.name}: shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}")
        except Exception as e:
            print(f"- {p.name}: error {e}")


def phase_train_models(data_root: str, out_dir: str, epochs: int=10, batch_size: int=32, lr: float=1e-3, seed: int=42, num_workers: int=0) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = build_dataloaders(data_root, batch_size=batch_size, val_ratio=0.2, seed=seed, num_workers=num_workers, pin_memory=(device.type=='cuda'), augment=True)

    # Models
    resnet = MultiTaskResNet(backbone='resnet50', pretrained=True).to(device)
    effnet = MultiTaskEfficientNet(variant='efficientnet-b0', pretrained=True).to(device)

    # Loss & sched
    scheduler_weights = LossWeightScheduler(LossWeights(1.0,0.5), LossWeights(1.0,1.0), mode='linear')
    criterion = MultiTaskLoss(alpha_cls=1.0, beta_reg=1.0, scheduler=scheduler_weights).to(device)

    def train_model(name: str, model: nn.Module):
        head_params = list(model.classifier.parameters()) + list(model.regressor.parameters())
        backbone_params = [p for n,p in model.named_parameters() if ('classifier' not in n and 'regressor' not in n and p.requires_grad)]
        optim = torch.optim.Adam([
            { 'params': backbone_params, 'lr': 1e-4 },
            { 'params': head_params, 'lr': lr },
        ], weight_decay=1e-4)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

        run_name = time.strftime('%Y%m%d_%H%M%S')
        save_dir = Path(out_dir)/run_name/name
        save_dir.mkdir(parents=True, exist_ok=True)
        history = { 'train':[], 'val':[] }
        best = float('inf')

        for ep in range(epochs):
            tr = train_one_epoch(model, criterion, train_loader, optim, device, ep, epochs, scaler)
            va = validate_one_epoch(model, criterion, val_loader, device, ep, epochs)
            lr_sched.step(va['loss'])
            history['train'].append(tr); history['val'].append(va)
            with open(save_dir/'history.json','w') as f: json.dump(history,f,indent=2)
            state = { 'epoch': ep+1, 'model_state': model.state_dict(), 'optimizer_state': optim.state_dict(), 'args': { 'epochs':epochs,'batch_size':batch_size,'lr':lr } }
            torch.save(state, save_dir/'last.pth')
            if va['loss'] < best:
                best = va['loss']; torch.save(state, save_dir/'best.pth')
            print(f"[{name}] Epoch {ep+1}/{epochs} train={tr} val={va}")
        print(f"[{name}] Done. Best val loss {best:.4f}. Folder: {save_dir}")

    train_model('resnet50', resnet)
    train_model('efficientnet_b0', effnet)


def _collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); logits_all=[]; probs_all=[]; cls_all=[]; va_p_all=[]; va_t_all=[]
    with torch.no_grad():
        for imgs, emos, vals, aros in loader:
            imgs = imgs.to(device)
            l, vp = model(imgs)
            p = torch.softmax(l,1)
            logits_all.append(l.cpu()); probs_all.append(p.cpu()); cls_all.append(emos); va_p_all.append(vp.cpu()); va_t_all.append(torch.stack([vals,aros],1))
    return (torch.cat(logits_all).numpy(), torch.cat(probs_all).numpy(), torch.cat(cls_all).numpy(), torch.cat(va_p_all).numpy(), torch.cat(va_t_all).numpy())


def phase_evaluate_models(data_root: str, resnet_ckpt: str, effnet_ckpt: str, out_dir: str, batch_size: int=64, num_workers: int=0) -> None:
    if accuracy_score is None:
        raise ImportError("scikit-learn and scipy are required for evaluation metrics")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = build_dataloaders(data_root, batch_size=batch_size, val_ratio=0.2, seed=42, num_workers=num_workers, pin_memory=(device.type=='cuda'), augment=False)

    def load_model_ckpt(kind: str, ckpt: str) -> nn.Module:
        if kind=='resnet50':
            m = MultiTaskResNet(backbone='resnet50', pretrained=False)
        else:
            m = MultiTaskEfficientNet(variant='efficientnet-b0', pretrained=False)
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state.get('model_state', state)); m.to(device); m.eval(); return m

    resnet = load_model_ckpt('resnet50', resnet_ckpt)
    effnet = load_model_ckpt('efficientnet-b0', effnet_ckpt)

    def eval_model(name: str, model: nn.Module, out: Path) -> Dict[str,Any]:
        logits, probs, y_true, va_pred, va_true = _collect_predictions(model, val_loader, device)
        y_pred = probs.argmax(1)
        # classification metrics
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(EMOTION_NAMES)))
        report = classification_report(y_true, y_pred, target_names=EMOTION_NAMES, digits=4)
        # AUCs
        y_true_bin = label_binarize(y_true, classes=np.arange(len(EMOTION_NAMES)))
        try:
            auc_ovr = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
        except Exception:
            auc_ovr = float('nan')
        try:
            auc_pr = average_precision_score(y_true_bin, probs, average='macro')
        except Exception:
            auc_pr = float('nan')
        # continuous
        mask = (va_true.min(axis=1) > -2)
        vt, vp = va_true[mask], va_pred[mask]
        rv = rmse(vt[:,0], vp[:,0]); ra = rmse(vt[:,1], vp[:,1])
        pv = corr_pearson(vt[:,0], vp[:,0]); pa = corr_pearson(vt[:,1], vp[:,1])
        sv = sagr(vt[:,0], vp[:,0]); sa = sagr(vt[:,1], vp[:,1])
        cv = ccc(vt[:,0], vp[:,0]); ca = ccc(vt[:,1], vp[:,1])
        metrics = {
            'classification': {
                'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted), 'auc_roc_macro_ovr': float(auc_ovr), 'auc_pr_macro': float(auc_pr), 'confusion_matrix': cm.tolist(), 'classification_report': report
            },
            'continuous': {
                'rmse_valence': rv, 'rmse_arousal': ra, 'pearson_valence': pv, 'pearson_arousal': pa, 'sagr_valence': sv, 'sagr_arousal': sa, 'ccc_valence': cv, 'ccc_arousal': ca, 'n_samples': int(mask.sum())
            }
        }
        out.mkdir(parents=True, exist_ok=True)
        with open(out/f"metrics_{name}.json",'w') as f: json.dump(metrics,f,indent=2)
        # plots
        plt.figure(figsize=(7,5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix - {name}'); plt.tight_layout(); plt.savefig(out/f"cm_{name}.png", dpi=300); plt.close()
        # ROC/PR
        try:
            plt.figure(figsize=(7,5))
            for i, n in enumerate(EMOTION_NAMES):
                fpr,tpr,_=roc_curve(y_true_bin[:,i], probs[:,i]); rauc=auc(fpr,tpr); plt.plot(fpr,tpr,label=f"{n} (AUC={rauc:.2f})")
            plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC OvR - {name}'); plt.legend(); plt.tight_layout(); plt.savefig(out/f"roc_{name}.png", dpi=300); plt.close()
        except Exception: pass
        try:
            plt.figure(figsize=(7,5))
            for i, n in enumerate(EMOTION_NAMES):
                pr, rc, _ = precision_recall_curve(y_true_bin[:,i], probs[:,i]); ap = average_precision_score(y_true_bin[:,i], probs[:,i]); plt.plot(rc,pr,label=f"{n} (AP={ap:.2f})")
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR OvR - {name}'); plt.legend(); plt.tight_layout(); plt.savefig(out/f"pr_{name}.png", dpi=300); plt.close()
        except Exception: pass
        # VA scatter
        if metrics['continuous']['n_samples']>0:
            plt.figure(figsize=(5.5,5.5)); lim=[-1.05,1.05]; plt.scatter(vt[:,0], vp[:,0], s=10, alpha=0.5); plt.plot(lim,lim,'r--'); plt.xlim(lim); plt.ylim(lim); plt.xlabel('True Valence'); plt.ylabel('Pred Valence'); plt.title(f'Valence Scatter - {name}'); plt.tight_layout(); plt.savefig(out/f"scatter_val_{name}.png", dpi=300); plt.close()
            plt.figure(figsize=(5.5,5.5)); plt.scatter(vt[:,1], vp[:,1], s=10, alpha=0.5); plt.plot(lim,lim,'r--'); plt.xlim(lim); plt.ylim(lim); plt.xlabel('True Arousal'); plt.ylabel('Pred Arousal'); plt.title(f'Arousal Scatter - {name}'); plt.tight_layout(); plt.savefig(out/f"scatter_aro_{name}.png", dpi=300); plt.close()
        return metrics

    out = Path(out_dir)/time.strftime('%Y%m%d_%H%M%S')
    res_metrics = eval_model('resnet50', resnet, out)
    eff_metrics = eval_model('efficientnet_b0', effnet, out)
    with open(out/'summary.json','w') as f: json.dump({'resnet50':res_metrics, 'efficientnet_b0':eff_metrics}, f, indent=2)
    print(f"Evaluation done. Outputs at: {out}")


def phase_generate_report(data_root: str, resnet_ckpt: str, effnet_ckpt: str, eval_dir: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    md = out/f'report_{timestamp}.md'
    # find latest metrics files
    eval_path = Path(eval_dir)
    metrics_files = sorted(eval_path.rglob('metrics_*.json'))
    res_eval = next((p for p in metrics_files if 'resnet50' in p.name), None)
    eff_eval = next((p for p in metrics_files if 'efficientnet_b0' in p.name), None)
    # Load models to get param counts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = MultiTaskResNet(backbone='resnet50', pretrained=False).to(device)
    effnet = MultiTaskEfficientNet(variant='efficientnet-b0', pretrained=False).to(device)
    res_total=sum(p.numel() for p in resnet.parameters()); eff_total=sum(p.numel() for p in effnet.parameters())
    content = [
        '# Report Summary',
        '## Network Details',
        f"ResNet-50 total params: {res_total:,}",
        f"EfficientNet-B0 total params: {eff_total:,}",
        '## Metrics Overview',
        f"Loaded metrics from: {res_eval} and {eff_eval}",
        'Please see generated figures and tables in evaluation outputs.',
    ]
    with open(md,'w',encoding='utf-8') as f: f.write('\n'.join(content))
    print(f"Report generated: {md}")

# =============================
# CLI
# =============================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Single-file pipeline for facial expression multi-task learning')
    parser.add_argument('--phase', type=str, required=True, choices=['data_exploration','train_models','evaluate_models','generate_report'])
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root containing images/ and annotations/')
    # training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    # evaluation args
    parser.add_argument('--resnet_ckpt', type=str, default='')
    parser.add_argument('--effnet_ckpt', type=str, default='')
    parser.add_argument('--eval_out', type=str, default='eval_outputs')
    # report args
    parser.add_argument('--report_out', type=str, default='report_outputs')

    args = parser.parse_args()

    if args.phase == 'data_exploration':
        phase_data_exploration(args.data_root)
    elif args.phase == 'train_models':
        phase_train_models(args.data_root, out_dir='checkpoints_both', epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)
    elif args.phase == 'evaluate_models':
        if not args.resnet_ckpt or not args.effnet_ckpt:
            raise ValueError('Please provide --resnet_ckpt and --effnet_ckpt for evaluation phase')
        phase_evaluate_models(args.data_root, args.resnet_ckpt, args.effnet_ckpt, out_dir=args.eval_out, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.phase == 'generate_report':
        if not args.resnet_ckpt or not args.effnet_ckpt:
            print('Note: generate_report uses only param counts if metrics not provided in eval dir')
        phase_generate_report(args.data_root, args.resnet_ckpt, args.effnet_ckpt, eval_dir=args.eval_out, out_dir=args.report_out)


if __name__ == '__main__':
    main()
