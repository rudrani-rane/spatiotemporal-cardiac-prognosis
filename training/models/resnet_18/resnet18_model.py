import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── Residual Block (standard ResNet-18 building block) ────────────────────
class ResidualBlock(nn.Module):
    """
    Basic residual block used in ResNet-18 / ResNet-34.

    Two 3×3 conv layers with BN + ReLU, plus an identity (or projection)
    skip connection.  No bottleneck — keeps the channel count constant.
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut when spatial size or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)          # residual addition
        out = self.relu(out)
        return out


# ── ResNet-18 for EF Regression ───────────────────────────────────────────
class ResNet18EFModel(nn.Module):
    """
    ResNet-18 adapted for cardiac EF regression.

    Input:
        3-channel tensor [B, 3, 112, 112]
        Channel 0 = ED frame  (End-Diastole)
        Channel 1 = ES frame  (End-Systole)
        Channel 2 = Motion map (|ED – ES|)

    Architecture (ResNet-18 layer layout):
        stem  : Conv(3→64, 7×7, s=2) → BN → ReLU → MaxPool(3×3, s=2)
        layer1: 2 × ResidualBlock(64→64,  s=1)
        layer2: 2 × ResidualBlock(64→128, s=2)
        layer3: 2 × ResidualBlock(128→256, s=2)
        layer4: 2 × ResidualBlock(256→512, s=2)
        AdaptiveAvgPool(1) → flatten → FC regression head

    Output:
        EF value (regression scalar) — shape [B, 1]

    Two construction modes
    ─────────────────────
    pretrained=True  : Load ImageNet weights from torchvision, replace the
                       final FC with a regression head. First conv is kept
                       as-is (still accepts 3-channel input — ED / ES /
                       motion maps map naturally to RGB channels).

    pretrained=False : Build from scratch using the manual ResidualBlock
                       implementation above (useful for ablations or when
                       network access is unavailable).
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        self.pretrained = pretrained

        if pretrained:
            self._build_pretrained(dropout_rate)
        else:
            self._build_from_scratch(dropout_rate)

    # ── Pretrained variant (torchvision backbone) ──────────────────────────
    def _build_pretrained(self, dropout_rate: float):
        """
        Load ImageNet ResNet-18, strip its classifier, add regression head.
        """
        weights  = tv_models.ResNet18_Weights.IMAGENET1K_V1
        backbone = tv_models.resnet18(weights=weights)

        # Remove the final FC layer (keep everything up to avgpool)
        # backbone.fc = nn.Identity() would work but we rebuild the head
        self.stem   = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # [B, 64,  H/4,  W/4]
        self.layer2 = backbone.layer2   # [B, 128, H/8,  W/8]
        self.layer3 = backbone.layer3   # [B, 256, H/16, W/16]
        self.layer4 = backbone.layer4   # [B, 512, H/32, W/32]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.1),

            nn.Linear(128, 1),
        )

    # ── From-scratch variant (manual residual blocks) ──────────────────────
    def _build_from_scratch(self, dropout_rate: float):
        """
        Build ResNet-18 from scratch — no torchvision dependency at runtime.
        """
        def _make_layer(in_ch, out_ch, blocks, stride=1):
            layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*layers)

        # Stem: 3 → 64  (112 → 28 after pool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # → [B, 64, 28, 28]
        )

        self.layer1 = _make_layer(64,  64,  blocks=2, stride=1)  # [B, 64,  28, 28]
        self.layer2 = _make_layer(64,  128, blocks=2, stride=2)  # [B, 128, 14, 14]
        self.layer3 = _make_layer(128, 256, blocks=2, stride=2)  # [B, 256,  7,  7]
        self.layer4 = _make_layer(256, 512, blocks=2, stride=2)  # [B, 512,  4,  4]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))           # [B, 512,  1,  1]

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.1),

            nn.Linear(128, 1),
        )

        self._init_weights()

    # ── Weight init (from-scratch only) ───────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 112, 112]

        Returns:
            out : [B, 1]  — predicted EF value
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.flatten(1)          # [B, 512]
        x = self.regressor(x)     # [B, 1]
        return x


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for mode, pt in [("pretrained", True), ("from-scratch", False)]:
        model = ResNet18EFModel(pretrained=pt)
        dummy = torch.randn(4, 3, 112, 112)
        out   = model(dummy)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mode}]  Output : {out.shape}  |  Params : {total:,}")
