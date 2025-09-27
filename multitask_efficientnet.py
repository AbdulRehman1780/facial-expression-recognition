from typing import Tuple, Optional

import torch
import torch.nn as nn
try:
    from efficientnet_pytorch import EfficientNet
except ImportError as e:
    raise ImportError(
        "efficientnet_pytorch is required. Install with: pip install efficientnet_pytorch"
    ) from e


class MultiTaskEfficientNet(nn.Module):
    """
    Multi-task EfficientNet-B0 backbone with dual heads:
      - Classification head (8 emotions)
      - Regression head (valence, arousal)

    Architecture notes vs ResNet:
    - EfficientNet uses MBConv blocks with depthwise separable convolutions and
      compound scaling; feature dimension before classifier is 1280 for B0.
    - Global pooling followed by lightweight heads yields fewer parameters than
      ResNet-50 while maintaining strong efficiency.
    """

    def __init__(
        self,
        variant: str = "efficientnet-b0",
        pretrained: bool = True,
        freeze_until_block: Optional[int] = 8,
        cls_dropout: float = 0.3,
        reg_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.variant = variant
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(variant)
        else:
            self.backbone = EfficientNet.from_name(variant)

        # EfficientNet outputs features after self.backbone.extract_features(x)
        # Output channels for B0 are 1280
        self.feature_dim = self.backbone._fc.in_features

        # We'll use the backbone up to features and drop its classifier
        self.backbone._dropout = nn.Identity()
        self.backbone._fc = nn.Identity()

        # Multi-task heads (match requested structure but sized for 1280)
        in_features = self.feature_dim  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(512, 8),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=reg_dropout),
            nn.Linear(256, 2),
        )

        self._init_head(self.classifier)
        self._init_head(self.regressor)

        # Transfer learning: freeze early MBConv blocks
        if freeze_until_block is not None:
            self.freeze_blocks(up_to=freeze_until_block)

        # Global pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _init_head(head: nn.Sequential) -> None:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def freeze_blocks(self, up_to: int = 8) -> None:
        """
        Freeze parameters of the stem and the first `up_to` blocks of EfficientNet.
        up_to refers to indices in self.backbone._blocks list (len depends on variant).
        """
        # Freeze stem
        for p in self.backbone._conv_stem.parameters():
            p.requires_grad = False
        for p in self.backbone._bn0.parameters():
            p.requires_grad = False

        # Freeze early MBConv blocks
        n_blocks = len(self.backbone._blocks)
        k = max(0, min(up_to, n_blocks))
        for i, block in enumerate(self.backbone._blocks):
            req = (i < k)
            for p in block.parameters():
                p.requires_grad = not (req is False)  # if req True -> freeze
            if i < k:
                for p in block.parameters():
                    p.requires_grad = False

        # Keep head conv/bn trainable by default
        # self.backbone._conv_head, self.backbone._bn1 stay trainable

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        feats = self.backbone.extract_features(x)  # [B, C, H, W]
        feats = self.pool(feats)                   # [B, C, 1, 1]
        feats = feats.view(feats.size(0), -1)      # [B, C]
        logits = self.classifier(feats)            # [B, 8]
        contin = self.regressor(feats)             # [B, 2]
        return logits, contin

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"MultiTaskEfficientNet(variant={self.variant})\n"
            f"Total params: {total:,}\n"
            f"Trainable params: {trainable:,}\n"
        )


def complexity_analysis(model: nn.Module, input_size=(1, 3, 224, 224)):
    """
    Optional FLOPs/MACs estimation using thop if available.
    """
    try:
        from thop import profile
    except Exception:
        print("[complexity] Install thop for FLOPs estimation: pip install thop")
        return None
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    print(f"MACs: {macs/1e9:.2f} G, Params: {params/1e6:.2f} M")
    return macs, params


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskEfficientNet(variant="efficientnet-b0", pretrained=True, freeze_until_block=8).to(device)
    print(model.summary())
    x = torch.randn(2, 3, 224, 224, device=device)
    cls_logits, reg_out = model(x)
    print("cls_logits:", cls_logits.shape)  # [2, 8]
    print("reg_out   :", reg_out.shape)      # [2, 2]
    complexity_analysis(model, input_size=(1,3,224,224))
