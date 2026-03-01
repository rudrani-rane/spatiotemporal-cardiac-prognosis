import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Deeper CNN Feature Extractor Block
    Structure: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout2d
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)


class MotionDifferenceModel(nn.Module):
    """
    Motion Difference EF Prediction Model (Improved)

    Input:
        Tensor shape = [B, 2, 112, 112]
        Channel 0 = ED frame
        Channel 1 = ES frame

    Architecture:
        Three parallel ConvBlock branches (ED, ES, Motion)
        Each: 1 → 32 → 64 → 128 channels over 3 downsampling stages
        AdaptiveAvgPool → concat → MLP regressor

    Output:
        EF value (regression scalar)
    """

    def __init__(self):
        super(MotionDifferenceModel, self).__init__()

        # Three parallel branches: ED, ES, motion map
        self.ed_encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.es_encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.motion_encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # Pool each branch to [B, 128, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP regression head: 128*3 → 256 → 128 → 1
        self.regressor = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        x: [B, 2, 112, 112]
        """
        # Split channels
        ed = x[:, 0:1, :, :]
        es = x[:, 1:2, :, :]
        motion = ed - es  # pixel-wise difference

        # Encode each branch
        ed_feat     = self.global_pool(self.ed_encoder(ed)).flatten(1)
        es_feat     = self.global_pool(self.es_encoder(es)).flatten(1)
        motion_feat = self.global_pool(self.motion_encoder(motion)).flatten(1)

        # Concat and regress
        combined = torch.cat([ed_feat, es_feat, motion_feat], dim=1)
        ef = self.regressor(combined)
        return ef


if __name__ == "__main__":
    model = MotionDifferenceModel()
    dummy = torch.randn(4, 2, 112, 112)
    out = model(dummy)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")