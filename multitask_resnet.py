import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models


_BACKBONE_MAP = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


def _load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in _BACKBONE_MAP:
        raise ValueError(f"Unsupported backbone '{name}'. Choose from {list(_BACKBONE_MAP.keys())}.")

    ctor = _BACKBONE_MAP[name]
    # Handle torchvision API differences (weights vs pretrained)
    try:
        if pretrained:
            # Use default pretrained weights per model (torchvision >= 0.13)
            weights_attr = getattr(models, f"{name}_Weights", None)
            if weights_attr is not None:
                weights = getattr(weights_attr, "IMAGENET1K_V1", None) or getattr(weights_attr, "DEFAULT", None)
                return ctor(weights=weights)
        return ctor(weights=None)
    except TypeError:
        # Fallback for older versions
        return ctor(pretrained=pretrained)


class MultiTaskResNet(nn.Module):
    """
    Multi-task ResNet backbone with dual heads:
      - Classification head (8 emotions)
      - Regression head (valence, arousal)

    For ResNet-50/101, the backbone feature dim is 2048; for ResNet-18/34 it's 512.
    The heads are sized dynamically based on the backbone's feature dimension.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_up_to: str = "layer2",  # freeze conv1, bn1, layer1, layer2
        cls_dropout: float = 0.3,
        reg_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone.lower()
        self.backbone = _load_backbone(self.backbone_name, pretrained=pretrained)

        # Build feature extractor (all layers except final fc)
        in_features = self.backbone.fc.in_features  # 512 for 18/34, 2048 for 50/101
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # outputs [B, C, 1, 1]

        # Multi-task heads
        # Classification: Linear(C, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 8)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(512, 8),
        )
        # Regression: Linear(C, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 2)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=reg_dropout),
            nn.Linear(256, 2),
        )

        # Initialize newly added heads
        self._init_head(self.classifier)
        self._init_head(self.regressor)

        # Freeze early layers for transfer learning
        self.freeze_until(freeze_up_to)

    @staticmethod
    def _init_head(head: nn.Sequential) -> None:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def freeze_until(self, up_to: Optional[str] = "layer2") -> None:
        """
        Freeze backbone parameters up to and including a given stage.
        up_to in {None, 'conv1', 'bn1', 'layer1', 'layer2', 'layer3'}
        Use None to keep everything trainable.
        """
        stages = [
            ("conv1", self.backbone.conv1),
            ("bn1", self.backbone.bn1),
            ("layer1", self.backbone.layer1),
            ("layer2", self.backbone.layer2),
            ("layer3", self.backbone.layer3),
        ]
        freeze = True
        for name, module in stages:
            for p in module.parameters():
                p.requires_grad = freeze
            if up_to is not None and name == up_to:
                freeze = False  # unfreeze subsequent layers
        # layer4 remains trainable by default

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        feats = self.feature_extractor(x)  # [B, C, 1, 1]
        feats = torch.flatten(feats, 1)    # [B, C]
        logits = self.classifier(feats)    # [B, 8]
        contin = self.regressor(feats)     # [B, 2] -> (valence, arousal)
        return logits, contin

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"MultiTaskResNet(backbone={self.backbone_name})\n"
            f"Total params: {total:,}\n"
            f"Trainable params: {trainable:,}\n"
        )


if __name__ == "__main__":
    # Quick smoke test and parameter summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet(backbone="resnet50", pretrained=True, freeze_up_to="layer2").to(device)
    print(model.summary())

    x = torch.randn(2, 3, 224, 224, device=device)
    cls_logits, reg_out = model(x)
    print("cls_logits:", cls_logits.shape)  # [2, 8]
    print("reg_out   :", reg_out.shape)      # [2, 2]
