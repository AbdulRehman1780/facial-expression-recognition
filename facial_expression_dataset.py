
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class FacialExpressionDataset(Dataset):
    """
    Comprehensive PyTorch Dataset for facial expression data with .npy annotations.

    Expected structure:
        dataset_root/
        ├── images/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── ... (224x224 RGB)
        └── annotations/
            ├── 0_exp.npy
            ├── 0_val.npy
            ├── 0_aro.npy
            └── ...

    Annotation files are .npy containing semicolon-separated string values.
    This class parses emotion, valence, and arousal values, filters uncertain (-2)
    and returns (image_tensor, emotion_label, valence, arousal).
    """

    EMOTION_NAMES = [
        "Neutral",    # 0
        "Happy",      # 1
        "Sad",        # 2
        "Surprise",   # 3
        "Fear",       # 4
        "Disgust",    # 5
        "Anger",      # 6
        "Contempt",   # 7
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        transform: Optional[Callable] = None,
        augment: bool = True,
    ) -> None:
        """
        Args:
            root: Path to dataset root containing `images/` and `annotations/`.
            split: One of {"train", "val"}.
            val_ratio: Fraction of data to reserve for validation.
            seed: Random seed for deterministic split.
            transform: Optional torchvision transform to apply.
            augment: If True and split=="train", apply default augmentations when no transform is provided.
        """
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.split = split.lower()
        assert self.split in {"train", "val"}, "split must be 'train' or 'val'"
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.user_transform = transform
        self.augment = augment

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.ann_dir}")

        # Build index of valid samples
        all_items = self._collect_items()
        train_items, val_items = self._split_items(all_items, self.val_ratio, self.seed)
        self.items = train_items if self.split == "train" else val_items

        # Define default transforms if none provided
        self.transform = self._build_default_transforms()

    # -------------------------------
    # Building the index
    # -------------------------------
    def _collect_items(self) -> List[Tuple[Path, int, float, float]]:
        """
        Collect items by scanning images and reading corresponding annotations.
        Returns a list of tuples: (image_path, emotion:int, valence:float, arousal:float)
        Filters out samples where valence or arousal equals -2 (uncertain).
        """
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(self.images_dir.rglob(ext))

        items: List[Tuple[Path, int, float, float]] = []
        missing = 0
        skipped_uncertain = 0
        for img_path in sorted(image_paths, key=lambda p: p.stem):
            sample_id = img_path.stem  # assumes '123.jpg' -> '123'

            exp_path = self.ann_dir / f"{sample_id}_exp.npy"
            val_path = self.ann_dir / f"{sample_id}_val.npy"
            aro_path = self.ann_dir / f"{sample_id}_aro.npy"

            if not (exp_path.exists() and val_path.exists() and aro_path.exists()):
                missing += 1
                continue

            try:
                emotion = self._parse_emotion(exp_path)
                valence = self._parse_float(val_path)
                arousal = self._parse_float(aro_path)
            except Exception:
                # Skip malformed entries
                continue

            # Filter uncertain values (-2)
            if valence == -2 or arousal == -2:
                skipped_uncertain += 1
                continue

            # Clamp to [-1, 1] if needed
            valence = float(np.clip(valence, -1.0, 1.0))
            arousal = float(np.clip(arousal, -1.0, 1.0))

            # Ensure emotion within [0,7]
            if not (0 <= emotion <= 7):
                continue

            items.append((img_path, int(emotion), valence, arousal))

        if len(items) == 0:
            raise RuntimeError(
                "No valid samples were found. Please verify the dataset structure and annotations."
            )
        return items

    def _split_items(
        self, items: List[Tuple[Path, int, float, float]], val_ratio: float, seed: int
    ) -> Tuple[List[Tuple[Path, int, float, float]], List[Tuple[Path, int, float, float]]]:
        """
        Deterministic split into train and validation subsets based on sorted IDs.
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(len(items))
        rng.shuffle(indices)

        n_val = int(round(len(items) * val_ratio))
        val_idx = set(indices[:n_val].tolist())

        train_items, val_items = [], []
        for i, it in enumerate(items):
            (val_items if i in val_idx else train_items).append(it)
        return train_items, val_items

    # -------------------------------
    # Parsers
    # -------------------------------
    @staticmethod
    def _load_npy_scalar(path: Path):
        data = np.load(path, allow_pickle=True)
        # Accept scalar numpy types, strings with semicolons, or small arrays
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        return data

    def _parse_emotion(self, path: Path) -> int:
        data = self._load_npy_scalar(path)
        # Cases:
        # - int already
        # - string like "6" or "6;confidence;..."
        # - array of strings
        if isinstance(data, (int, np.integer)):
            return int(data)
        if isinstance(data, (float, np.floating)):
            return int(round(float(data)))
        if isinstance(data, str):
            token = data.split(";")[0].strip()
            return int(token)
        if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 0:
            # take first entry
            token = str(data[0]).split(";")[0].strip()
            return int(token)
        raise ValueError(f"Unsupported emotion format in {path}")

    def _parse_float(self, path: Path) -> float:
        data = self._load_npy_scalar(path)
        # Accept numeric or semicolon-separated string, e.g., "0.123;conf" or "-2"
        if isinstance(data, (int, float, np.integer, np.floating)):
            return float(data)
        if isinstance(data, str):
            token = data.split(";")[0].strip().replace(",", ".")
            return float(token)
        if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 0:
            token = str(data[0]).split(";")[0].strip().replace(",", ".")
            return float(token)
        raise ValueError(f"Unsupported float format in {path}")

    # -------------------------------
    # Transforms
    # -------------------------------
    def _build_default_transforms(self) -> Callable:
        if self.user_transform is not None:
            return self.user_transform

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.split == "train" and self.augment:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

    # -------------------------------
    # PyTorch Dataset protocol
    # -------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, float]:
        img_path, emotion, valence, arousal = self.items[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img)
        return image_tensor, int(emotion), float(valence), float(arousal)

    # -------------------------------
    # Utility
    # -------------------------------
    @classmethod
    def emotion_name(cls, label: int) -> str:
        if 0 <= label < len(cls.EMOTION_NAMES):
            return cls.EMOTION_NAMES[label]
        return f"Unknown({label})"


def build_dataloaders(
    root: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment: bool = True,
):
    """
    Convenience function to create train/val dataloaders.
    Returns: train_loader, val_loader
    """
    train_ds = FacialExpressionDataset(
        root=root, split="train", val_ratio=val_ratio, seed=seed, augment=augment
    )
    val_ds = FacialExpressionDataset(
        root=root, split="val", val_ratio=val_ratio, seed=seed, augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_with_targets
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_with_targets
    )
    return train_loader, val_loader


def collate_with_targets(batch):
    """
    Custom collate function to handle multi-output targets.
    Batch is a list of tuples: (image_tensor, emotion:int, valence:float, arousal:float)
    Returns: images[B,3,224,224], emotions[B], valences[B], arousals[B]
    """
    images, emotions, valences, arousals = zip(*batch)
    images = torch.stack(images, dim=0)
    emotions = torch.as_tensor(emotions, dtype=torch.long)
    valences = torch.as_tensor(valences, dtype=torch.float32)
    arousals = torch.as_tensor(arousals, dtype=torch.float32)
    return images, emotions, valences, arousals


def visualize_augmentations(
    root: str,
    split: str = "train",
    n_samples: int = 8,
    cols: int = 4,
    seed: int = 0,
    augment: bool = True,
):
    """
    Visualize dataset samples to verify augmentations.
    For train split, augment=True shows effects of augmentation; for val, augment is ignored.
    """
    rng = np.random.default_rng(seed)
    ds = FacialExpressionDataset(root=root, split=split, augment=augment)
    n_samples = min(n_samples, len(ds))

    idxs = rng.choice(len(ds), size=n_samples, replace=False)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(rows, cols)

    for ax, idx in zip(axes.flatten(), idxs):
        img_t, emo, val, aro = ds[idx]
        # Convert tensor to HWC for plotting
        img_np = img_t.permute(1, 2, 0).detach().cpu().numpy()
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.set_title(f"{FacialExpressionDataset.emotion_name(emo)}\nV:{val:.2f} A:{aro:.2f}")
        ax.axis('off')

    # Hide unused axes
    for ax in axes.flatten()[n_samples:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
