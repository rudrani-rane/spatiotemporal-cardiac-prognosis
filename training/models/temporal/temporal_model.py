"""
Temporal EF Model  –  CNN + LSTM
=================================
Processes the full cardiac cycle as a sequence of frames.

Input:
    [B, T, 1, 112, 112]
    T = 16 uniformly sampled frames from the AVI video

Architecture:
    FrameEncoder (shared weights across all T frames):
        Conv block  1 →  16  (Conv→BN→ReLU→MaxPool)
        Conv block 16 →  32  (Conv→BN→ReLU→MaxPool)
        Conv block 32 →  64  (Conv→BN→ReLU→MaxPool)
        AdaptiveAvgPool(1,1) → flatten → 64-d feature per frame

    LSTM temporal aggregator:
        input_size  = 64
        hidden_size = 128
        num_layers  = 2
        dropout     = 0.3  (between layers)
        → last hidden state  [B, 128]

    MLP regressor:
        128 → 64 → Dropout(0.3) → 1

Output:
    EF value (regression scalar)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# Per-frame CNN encoder (shared weights)
# ─────────────────────────────────────────────────────────────────────────

class FrameEncoder(nn.Module):
    """
    Lightweight CNN that maps a single grayscale frame [B, 1, H, W]
    to a 64-d feature vector [B, 64].
    """

    def __init__(self):
        super(FrameEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1: 1 → 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 112 → 56

            # Block 2: 16 → 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 56 → 28

            # Block 3: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28 → 14
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → [B, 64, 1, 1]

    def forward(self, x):
        """x: [B, 1, H, W]  →  [B, 64]"""
        x = self.encoder(x)
        x = self.pool(x)
        return x.flatten(1)          # [B, 64]


# ─────────────────────────────────────────────────────────────────────────
# Full temporal model
# ─────────────────────────────────────────────────────────────────────────

class TemporalEFModel(nn.Module):
    """
    Temporal EF prediction model.

    Args:
        seq_len     : number of frames in each input sequence (default 16)
        hidden_size : LSTM hidden dimension (default 128)
        num_layers  : LSTM depth (default 2)
    """

    def __init__(self, seq_len: int = 16, hidden_size: int = 128, num_layers: int = 2):
        super(TemporalEFModel, self).__init__()

        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Shared per-frame CNN encoder
        self.frame_encoder = FrameEncoder()

        # Temporal aggregator
        self.lstm = nn.LSTM(
            input_size  = 64,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3 if num_layers > 1 else 0.0,
        )

        # MLP regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x: [B, T, 1, H, W]

        Steps:
            1. Reshape to [B*T, 1, H, W] → FrameEncoder → [B*T, 64]
            2. Reshape to [B, T, 64]
            3. LSTM → last hidden state [B, 128]
            4. Regressor → [B, 1]
        """
        B, T, C, H, W = x.shape

        # Encode all frames in one batch pass
        x_flat   = x.view(B * T, C, H, W)             # [B*T, 1, H, W]
        feats    = self.frame_encoder(x_flat)           # [B*T, 64]
        feats    = feats.view(B, T, -1)                 # [B, T, 64]

        # LSTM: output is (output, (h_n, c_n))
        # h_n: [num_layers, B, hidden_size]
        _, (h_n, _) = self.lstm(feats)

        # Use the last layer's hidden state
        context = h_n[-1]                               # [B, hidden_size]

        return self.regressor(context)                  # [B, 1]


if __name__ == "__main__":
    model = TemporalEFModel()
    dummy = torch.randn(4, 16, 1, 112, 112)
    out   = model(dummy)
    print("Output shape:", out.shape)                   # [4, 1]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
