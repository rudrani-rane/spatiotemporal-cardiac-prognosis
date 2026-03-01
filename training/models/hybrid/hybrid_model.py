import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridEFModel(nn.Module):
    """
    Hybrid EF Prediction Model (Improved)

    Input:
        3-channel tensor [B, 3, 112, 112]
        Channel 0 = ED frame
        Channel 1 = ES frame
        Channel 2 = Motion (|ED - ES|)

    Architecture:
        4-stage CNN: 3 → 16 → 32 → 64 → 128
        Each stage: Conv → BN → ReLU → MaxPool
        AdaptiveAvgPool(1) → flatten → FC head

    Output:
        EF value (regression scalar)
    """

    def __init__(self):
        super(HybridEFModel, self).__init__()

        self.features = nn.Sequential(
            # Stage 1: 3 → 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 2: 16 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 4: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Adaptive pool removes dependency on input spatial size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        x: [B, 3, 112, 112]
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    model = HybridEFModel()
    dummy = torch.randn(4, 3, 112, 112)
    out = model(dummy)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")