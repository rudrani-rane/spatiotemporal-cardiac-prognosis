import torch
import torch.nn as nn


class CardioCNNModel(nn.Module):
    """
    Cardiac EF Prediction – Pure CNN Model

    Input:
        3-channel tensor [B, 3, 112, 112]
        Channel 0 = ED frame  (End-Diastole)
        Channel 1 = ES frame  (End-Systole)
        Channel 2 = Motion map (|ED - ES|)

    Architecture:
        5-stage CNN with increasing depth: 3 → 32 → 64 → 128 → 256 → 256
        Each stage: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
        (double-conv per stage for richer feature extraction vs hybrid's single-conv)
        AdaptiveAvgPool(1) → flatten → FC regression head

    Output:
        EF value (regression scalar per sample) — shape [B, 1]
    """

    def __init__(self, dropout_rate: float = 0.4):
        super(CardioCNNModel, self).__init__()

        self.features = nn.Sequential(
            # ── Stage 1: 3 → 32  (112 → 56) ───────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # → [B, 32, 56, 56]

            # ── Stage 2: 32 → 64  (56 → 28) ────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # → [B, 64, 28, 28]

            # ── Stage 3: 64 → 128  (28 → 14) ───────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # → [B, 128, 14, 14]

            # ── Stage 4: 128 → 256  (14 → 7) ───────────────────────────
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # → [B, 256, 7, 7]

            # ── Stage 5: 256 → 256  (7 → 3) ────────────────────────────
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # → [B, 256, 3, 3]
        )

        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))     # → [B, 256, 1, 1]

        self.regressor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate - 0.1),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 1),          
        )

        # Weight initialisation (He for Conv, zero-bias for BN)
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 112, 112]  — stacked ED / ES / motion tensor

        Returns:
            out : [B, 1]           — predicted EF value
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)            # [B, 256]
        x = self.regressor(x)       # [B, 1]
        return x



if __name__ == "__main__":
    model = CardioCNNModel()
    dummy = torch.randn(4, 3, 112, 112)
    out   = model(dummy)
    print("Output shape       :", out.shape)          # expect [4, 1]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params   : {total:,}")
