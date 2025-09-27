from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Container for multi-task loss weights."""
    alpha_cls: float = 1.0  # weight for classification loss
    beta_reg: float = 1.0   # weight for regression loss


class LossWeightScheduler:
    """
    Simple scheduler to adjust loss weights over training.

    Supports:
    - linear: linearly interpolate from (alpha0, beta0) to (alpha1, beta1)
    - cosine: cosine interpolation between start and end
    - custom: user-provided callable(progress: float) -> (alpha, beta)

    progress is in [0, 1], typically step / total_steps or epoch / total_epochs.
    """

    def __init__(
        self,
        start: LossWeights = LossWeights(1.0, 0.5),
        end: LossWeights = LossWeights(1.0, 1.0),
        mode: str = "linear",
        custom_fn: Optional[Callable[[float], LossWeights]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.mode = mode
        self.custom_fn = custom_fn

    def __call__(self, progress: float) -> LossWeights:
        p = float(max(0.0, min(1.0, progress)))
        if self.custom_fn is not None:
            return self.custom_fn(p)
        if self.mode == "linear":
            a = self.start.alpha_cls + p * (self.end.alpha_cls - self.start.alpha_cls)
            b = self.start.beta_reg + p * (self.end.beta_reg - self.start.beta_reg)
            return LossWeights(a, b)
        if self.mode == "cosine":
            # cosine interpolation
            cp = 0.5 * (1 - torch.cos(torch.tensor(p * 3.1415926535))).item()
            a = self.start.alpha_cls + cp * (self.end.alpha_cls - self.start.alpha_cls)
            b = self.start.beta_reg + cp * (self.end.beta_reg - self.start.beta_reg)
            return LossWeights(a, b)
        # default: no change
        return self.start


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss:
      - CrossEntropyLoss for emotion classification (with optional class weights)
      - MSELoss for continuous valence/arousal targets

    Returns total loss and individual components for monitoring, with adjustable
    weights that can be scheduled during training.
    """

    def __init__(
        self,
        num_classes: int = 8,
        class_weights: Optional[torch.Tensor] = None,
        alpha_cls: float = 1.0,
        beta_reg: float = 1.0,
        scheduler: Optional[LossWeightScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "class_weights",
            class_weights.clone().float() if class_weights is not None else None,
            persistent=False,
        )
        self.alpha_cls = float(alpha_cls)
        self.beta_reg = float(beta_reg)
        self.scheduler = scheduler
        self._device = device  # optional preferred device for buffers

        # Define losses
        # Note: class weights moved to correct device inside forward
        self.ce = nn.CrossEntropyLoss(weight=None, reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def to(self, *args, **kwargs):  # ensure buffer device updated on .to()
        module = super().to(*args, **kwargs)
        self._device = next(self.parameters(), torch.tensor(0, device=kwargs.get("device", None))).device
        return module

    def _get_weights(self, progress: Optional[float] = None) -> LossWeights:
        if self.scheduler is None or progress is None:
            return LossWeights(self.alpha_cls, self.beta_reg)
        return self.scheduler(progress)

    def forward(
        self,
        logits: torch.Tensor,
        va_pred: torch.Tensor,
        targets_cls: torch.Tensor,
        targets_va: torch.Tensor,
        progress: Optional[float] = None,  # in [0,1], e.g., step/total_steps
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, num_classes]
            va_pred: [B, 2] (valence, arousal)
            targets_cls: [B] LongTensor with class indices
            targets_va: [B, 2] FloatTensor with valence/arousal in [-1, 1]
            progress: optional [0,1] value for scheduling loss weights
            device: optional device override for loss computation
        Returns:
            dict with keys: total, cls, reg, alpha, beta
        """
        dev = device or logits.device

        # Ensure buffers on correct device
        if self.class_weights is not None and self.class_weights.device != dev:
            cw = self.class_weights.to(dev)
        else:
            cw = self.class_weights

        # Compute component losses
        if cw is not None:
            ce = F.cross_entropy(logits, targets_cls.to(dev), weight=cw, reduction="mean")
        else:
            ce = self.ce(logits.to(dev), targets_cls.to(dev))

        mse = self.mse(va_pred.to(dev), targets_va.to(dev))

        # Get current weights (optionally scheduled)
        w = self._get_weights(progress)
        alpha = torch.as_tensor(w.alpha_cls, dtype=torch.float32, device=dev)
        beta = torch.as_tensor(w.beta_reg, dtype=torch.float32, device=dev)

        total = alpha * ce + beta * mse

        return {
            "total": total,
            "cls": ce.detach(),
            "reg": mse.detach(),
            "alpha": alpha.detach(),
            "beta": beta.detach(),
        }


if __name__ == "__main__":
    # Smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example class weights to address imbalance (heavier weight for minority classes)
    class_weights = torch.tensor([1.0, 0.8, 1.2, 1.1, 1.3, 1.0, 0.9, 1.5])

    scheduler = LossWeightScheduler(
        start=LossWeights(alpha_cls=1.0, beta_reg=0.5),
        end=LossWeights(alpha_cls=1.0, beta_reg=1.0),
        mode="linear",
    )

    criterion = MultiTaskLoss(
        num_classes=8,
        class_weights=class_weights,
        alpha_cls=1.0,
        beta_reg=1.0,
        scheduler=scheduler,
        device=device,
    ).to(device)

    B = 16
    logits = torch.randn(B, 8, device=device)
    va_pred = torch.randn(B, 2, device=device)
    targets_cls = torch.randint(0, 8, (B,), device=device)
    targets_va = torch.randn(B, 2, device=device).clamp(-1, 1)

    out = criterion(logits, va_pred, targets_cls, targets_va, progress=0.3)
    print({k: (v.item() if torch.is_tensor(v) else v) for k, v in out.items()})
