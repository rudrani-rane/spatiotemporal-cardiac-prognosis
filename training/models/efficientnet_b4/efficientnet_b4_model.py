"""
efficientnet_b4_model.py
────────────────────────
EfficientNet-B4 adapted for LVEF regression.

EfficientNet overview:
    Unlike ResNet (which manually tuned depth/width) or our custom CNN
    (fixed stages), EfficientNet scales all three dimensions together using
    a compound coefficient φ:
        depth   d = 1.2^φ
        width   w = 1.1^φ
        resolution r = 1.15^φ

    B4 uses φ=4:
        depth   ≈ 1.2^4 = 2.07×
        width   ≈ 1.1^4 = 1.46×
        resolution = 380×380 (we resize to 112 to match our pipeline)

    The core building block is the MBConv (Mobile Inverted Bottleneck Conv):
        Expand channels → Depthwise conv → Squeeze-and-Excitation → Project
    This is more parameter-efficient than standard convolutions.

Input:
    [B, 3, 112, 112]  —  ED frame / ES frame / motion map  (same as CNN/ResNet)

Architecture:
    EfficientNet-B4 backbone (pretrained on ImageNet, 1000-class head removed)
    └── AdaptiveAvgPool → flatten → [B, 1792]
    Regression head: 1792 → 512 → 256 → 128 → 1

Output:
    [B, 1]  —  predicted LVEF value
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import EfficientNet_B4_Weights


class EfficientNetB4EFModel(nn.Module):
    """
    EfficientNet-B4 for cardiac LVEF regression.

    Input  : [B, 3, 112, 112]  —  ED / ES / motion tensor
    Output : [B, 1]            —  predicted LVEF (%)

    Construction modes:
        pretrained=True  (recommended) — loads ImageNet weights, replaces
                         the 1000-class classifier with a regression head.
        pretrained=False — random init, useful for ablation studies.
    """

    # EfficientNet-B4 feature extractor outputs 1792 channels
    _FEATURE_DIM = 1792

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        self.pretrained = pretrained

        # ── Load backbone ──────────────────────────────────────────────────
        if pretrained:
            weights  = EfficientNet_B4_Weights.IMAGENET1K_V1
            backbone = tv_models.efficientnet_b4(weights=weights)
        else:
            backbone = tv_models.efficientnet_b4(weights=None)

        # ── Extract feature layers (everything except the classifier) ──────
        # EfficientNet torchvision layout:
        #   backbone.features  — MBConv stages (the conv backbone)
        #   backbone.avgpool   — AdaptiveAvgPool2d(1,1)
        #   backbone.classifier — [Dropout, Linear(1792, 1000)] ← we replace this
        self.features  = backbone.features    # MBConv stages → [B, 1792, H', W']
        self.pool      = backbone.avgpool     # AdaptiveAvgPool2d(1,1) → [B, 1792, 1, 1]

        # ── Regression head (replaces the 1000-class classifier) ───────────
        # 1792 → 512 → 256 → 128 → 1
        # Deeper than ResNet-18 head (512 input) because B4 has richer features
        self.regressor = nn.Sequential(
            nn.Linear(self._FEATURE_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.1),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 1),          # scalar EF output — no activation
        )

        # Initialise only the regression head (backbone already has good weights)
        self._init_head()

    # ── Weight init (head only) ────────────────────────────────────────────
    def _init_head(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : [B, 3, 112, 112]  — stacked ED / ES / motion tensor

        Returns:
            out : [B, 1]            — predicted LVEF value
        """
        x = self.features(x)   # [B, 1792, H', W']  (H'≈3 for 112 input)
        x = self.pool(x)       # [B, 1792, 1, 1]
        x = x.flatten(1)       # [B, 1792]
        x = self.regressor(x)  # [B, 1]
        return x


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for mode, pt in [("pretrained", True), ("from-scratch", False)]:
        model = EfficientNetB4EFModel(pretrained=pt)
        dummy = torch.randn(4, 3, 112, 112)

        model.eval()
        with torch.no_grad():
            out = model(dummy)

        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mode:12s}]  Output : {out.shape}  |  Params : {total:,}")
