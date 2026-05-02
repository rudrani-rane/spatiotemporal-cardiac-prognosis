"""
efnet_model.py
──────────────
EF-Net: A hybrid 3D ResNet-18 + Transformer model for LVEF prediction
from echocardiogram video clips.

Reference:
    Ali W, Alsabban W, Shahbaz M, Al-Laith A, Almogadwy B.
    "EFNet: estimation of left ventricular ejection fraction from cardiac
    ultrasound videos using deep learning."
    PeerJ Computer Science 11:e2506 (2025).
    https://doi.org/10.7717/peerj-cs.2506

Architecture overview:
    Input : [B, 1, T, H, W]  — grayscale video clip (T=16 frames, H=W=112)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 3D Stem  : Conv3d(1→64, 7×7×7, s=1×2×2) → BN → ReLU → MaxPool3d  │
    │ Layer 1  : 2 × ResBlock3D(64→64,   s=1)                            │
    │ Layer 2  : 2 × RTM(64→128,  s=2)   ← Residual Transformer Module  │
    │ Layer 3  : 2 × RTM(128→256, s=2)   ← Residual Transformer Module  │
    │ Layer 4  : 2 × ResBlock3D(256→512, s=2)                            │
    │ AdaptiveAvgPool3d(1,1,1) → flatten → FC head → EF scalar           │
    └─────────────────────────────────────────────────────────────────────┘

Key innovation — Residual Transformer Module (RTM):
    Replaces the second residual block in layers 2 and 3 with a block that
    runs Multi-Head Self-Attention (MHSA) in parallel with the convolution
    and fuses both via addition. Relative Positional Encoding (RPE) encodes
    height (Rh), width (Rw), and time (Rt) dimensions separately so the
    attention is spatiotemporally aware.

Input shape note:
    The dataset loader returns clips of shape [B, T, H, W] (no channel dim).
    The model unsqueezes a channel dim internally → [B, 1, T, H, W].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  3D Residual Block  (standard — used in layers 1 and 4)
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3D(nn.Module):
    """
    Standard 3D residual block (two 3×3×3 convolutions + skip connection).

    Identical in spirit to the 2D ResidualBlock in resnet18_model.py but
    operates on volumetric tensors [B, C, T, H, W].
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # temporal stride=1 always — we only stride spatially
        t_stride = 1
        s_stride = stride

        self.conv1 = nn.Conv3d(in_ch, out_ch,
                               kernel_size=3,
                               stride=(t_stride, s_stride, s_stride),
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch,
                               kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # Projection shortcut: needed when channels or spatial size change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch,
                          kernel_size=1,
                          stride=(t_stride, s_stride, s_stride),
                          bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Relative Positional Encoding (RPE)
# ══════════════════════════════════════════════════════════════════════════════

class RelativePositionalEncoding3D(nn.Module):
    """
    3D Relative Positional Encoding as described in EF-Net.

    Learns separate bias tables for height (Rh), width (Rw), and time (Rt).
    For an input volume of shape (T, H, W), the bias added to each attention
    logit is:
        RPE(i, j) = Rh(dh) + Rw(dw) + Rt(dt)
    where (dh, dw, dt) are the relative offsets between positions i and j.

    The bias is pre-computed for the expected (T, H, W) and cached.
    """

    def __init__(self, num_heads: int, T: int, H: int, W: int):
        super().__init__()
        self.num_heads = num_heads
        self.T = T
        self.H = H
        self.W = W

        # Learnable bias tables — one entry per relative offset per head
        self.rpe_h = nn.Parameter(torch.zeros(num_heads, 2 * H - 1))
        self.rpe_w = nn.Parameter(torch.zeros(num_heads, 2 * W - 1))
        self.rpe_t = nn.Parameter(torch.zeros(num_heads, 2 * T - 1))

        nn.init.trunc_normal_(self.rpe_h, std=0.02)
        nn.init.trunc_normal_(self.rpe_w, std=0.02)
        nn.init.trunc_normal_(self.rpe_t, std=0.02)

        # Pre-compute relative index tables (registered as buffers — not params)
        self._build_index_tables(T, H, W)

    def _build_index_tables(self, T, H, W):
        # Height offsets: shape [H*H]
        coords_h = torch.arange(H)
        rel_h    = coords_h[:, None] - coords_h[None, :]   # [H, H]
        rel_h   += H - 1                                    # shift to [0, 2H-2]
        self.register_buffer("idx_h", rel_h.reshape(-1))

        # Width offsets: shape [W*W]
        coords_w = torch.arange(W)
        rel_w    = coords_w[:, None] - coords_w[None, :]
        rel_w   += W - 1
        self.register_buffer("idx_w", rel_w.reshape(-1))

        # Temporal offsets: shape [T*T]
        coords_t = torch.arange(T)
        rel_t    = coords_t[:, None] - coords_t[None, :]
        rel_t   += T - 1
        self.register_buffer("idx_t", rel_t.reshape(-1))

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        """
        Returns additive bias tensor of shape [num_heads, T*H*W, T*H*W].
        Uses stored index tables (assumes T, H, W match constructor args).
        """
        # Gather biases for each axis   [num_heads, N*N]
        bias_h = self.rpe_h[:, self.idx_h]   # [nh, H*H]
        bias_w = self.rpe_w[:, self.idx_w]   # [nh, W*W]
        bias_t = self.rpe_t[:, self.idx_t]   # [nh, T*T]

        N_s = H * W   # spatial tokens per frame
        N   = T * N_s  # total tokens

        # Expand spatial bias to (T*H*W) × (T*H*W) by broadcasting across T
        # bias_h/w apply within each frame, repeated across all frame pairs
        bias_h = bias_h.reshape(self.num_heads, H, H)       # [nh, H, H]
        bias_w = bias_w.reshape(self.num_heads, W, W)       # [nh, W, W]

        # Outer-sum of h and w biases  → [nh, H*W, H*W]
        spatial_bias = (
            bias_h.unsqueeze(-1).unsqueeze(-3) +             # [nh,H,1,H,1]
            bias_w.unsqueeze(-2).unsqueeze(-4)               # [nh,1,W,1,W]
        ).reshape(self.num_heads, N_s, N_s)

        # Expand to full temporal sequence: same spatial bias for every (t1,t2) pair
        spatial_bias = spatial_bias.unsqueeze(1).unsqueeze(3)   # [nh,1,HW,1,HW]
        spatial_bias = spatial_bias.expand(-1, T, -1, T, -1)    # [nh,T,HW,T,HW]
        spatial_bias = spatial_bias.reshape(self.num_heads, N, N)

        # Temporal bias: for each pair of tokens at different times
        bias_t = bias_t.reshape(self.num_heads, T, T)            # [nh, T, T]
        # Expand: same temporal bias for all spatial positions
        temporal_bias = bias_t.unsqueeze(-1).unsqueeze(-3)       # [nh,T,1,T,1]
        temporal_bias = temporal_bias.expand(-1, -1, N_s, -1, N_s)
        temporal_bias = temporal_bias.reshape(self.num_heads, N, N)

        return spatial_bias + temporal_bias   # [nh, N, N]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Residual Transformer Module (RTM)
# ══════════════════════════════════════════════════════════════════════════════

class RTM(nn.Module):
    """
    Residual Transformer Module — the core EF-Net building block.

    Replaces the second ResBlock3D in layers 2 and 3 with:
        1.  A standard 3D convolution path  (local features)
        2.  A Multi-Head Self-Attention path (global spatiotemporal features)
        3.  Both paths added together + residual skip connection

    The MHSA uses relative positional encoding (RPE) across height, width,
    and time so the model knows not just *what* is attended but *where/when*.

    Input shape:  [B, in_ch, T, H, W]
    Output shape: [B, out_ch, T, H', W']  (H', W' halved if stride=2)
    """

    def __init__(self,
                 in_ch:     int,
                 out_ch:    int,
                 stride:    int = 1,
                 num_heads: int = 4,
                 clip_T:    int = 16,
                 clip_H:    int = 14,    # spatial size AFTER stem + layer1
                 clip_W:    int = 14):
        super().__init__()
        self.stride   = stride
        self.out_ch   = out_ch
        self.num_heads = num_heads

        # ── Convolutional path (same as ResBlock3D) ────────────────────────
        t_stride = 1
        s_stride = stride

        self.conv1 = nn.Conv3d(in_ch, out_ch,
                               kernel_size=3,
                               stride=(t_stride, s_stride, s_stride),
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch,
                               kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # ── Skip connection ────────────────────────────────────────────────
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch,
                          kernel_size=1,
                          stride=(t_stride, s_stride, s_stride),
                          bias=False),
                nn.BatchNorm3d(out_ch),
            )

        # ── Attention path ─────────────────────────────────────────────────
        # Compute spatial dims after this block's stride
        attn_H = clip_H // stride
        attn_W = clip_W // stride
        attn_T = clip_T             # temporal dim never strided

        self.attn_H = attn_H
        self.attn_W = attn_W
        self.attn_T = attn_T

        head_dim  = max(out_ch // num_heads, 1)
        self.head_dim = head_dim

        # Project to Q, K, V
        self.qkv_proj = nn.Linear(out_ch, 3 * num_heads * head_dim, bias=False)
        self.out_proj  = nn.Linear(num_heads * head_dim, out_ch, bias=False)
        self.attn_norm = nn.LayerNorm(out_ch)
        self.scale     = math.sqrt(head_dim)

        # Relative positional encoding
        self.rpe = RelativePositionalEncoding3D(num_heads, attn_T, attn_H, attn_W)

        # Gate: blend conv output and attention output
        self.gate = nn.Parameter(torch.tensor(0.5))

    def _mhsa(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-Head Self-Attention over the full T×H×W volume.

        Args:
            x : [B, C, T, H, W]  (post-conv2 features)
        Returns:
            [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        N = T * H * W

        # Flatten spatial + temporal dims into sequence
        x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)  # [B, N, C]
        x_norm = self.attn_norm(x_seq)                       # LayerNorm

        # Q, K, V projections
        qkv = self.qkv_proj(x_norm)                          # [B, N, 3*nh*hd]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # [3, B, nh, N, hd]
        q, k, v = qkv.unbind(0)                              # each [B, nh, N, hd]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, nh, N, N]

        # Add relative positional bias
        rpe_bias = self.rpe(T, H, W)                         # [nh, N, N]
        attn     = attn + rpe_bias.unsqueeze(0)              # [B, nh, N, N]
        attn     = F.softmax(attn, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v)                          # [B, nh, N, hd]
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        out = self.out_proj(out)                             # [B, N, C]

        # Reshape back to volume
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)  # [B,C,T,H,W]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Conv path ─────────────────────────────────────────────────────
        conv_out = self.relu(self.bn1(self.conv1(x)))
        conv_out = self.bn2(self.conv2(conv_out))

        # ── Attention path (runs on conv output) ──────────────────────────
        attn_out = self._mhsa(conv_out)

        # ── Gated fusion of conv + attention ──────────────────────────────
        gate     = torch.sigmoid(self.gate)
        fused    = gate * conv_out + (1 - gate) * attn_out

        # ── Residual add ──────────────────────────────────────────────────
        out      = self.relu(fused + self.shortcut(x))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EF-Net  (full model)
# ══════════════════════════════════════════════════════════════════════════════

class EFNet(nn.Module):
    """
    EF-Net: 3D ResNet-18 with Residual Transformer Modules in layers 2 & 3.

    Input:
        x : [B, T, H, W]  — grayscale video clip (no explicit channel dim)
            T = NUM_FRAMES (default 16)
            H = W = 112

    Architecture:
        stem    : Conv3d(1→64, 7×7×7, s=(1,2,2)) → BN → ReLU → MaxPool3d
        layer1  : 2 × ResBlock3D(64→64,   stride=1)    spatial: 28×28
        layer2  : 2 × RTM(64→128,         stride=2)    spatial: 14×14  ← transformer
        layer3  : 2 × RTM(128→256,        stride=2)    spatial:  7×7   ← transformer
        layer4  : 2 × ResBlock3D(256→512, stride=2)    spatial:  4×4
        AdaptiveAvgPool3d(1,1,1) → [B, 512]
        FC head : 512 → 256 → 128 → 1

    Output:
        [B, 1] — predicted LVEF percentage
    """

    # Spatial size at each stage (input 112×112 after stem MaxPool → 28×28)
    _STAGE_H = {1: 28, 2: 14, 3: 7}

    def __init__(self,
                 num_frames:   int   = 16,
                 dropout_rate: float = 0.4,
                 num_heads:    int   = 4):
        super().__init__()
        self.num_frames = num_frames

        # ── Stem ──────────────────────────────────────────────────────────
        # 7×7×7 conv, temporal stride=1 (preserve frames), spatial stride=2
        # Input:  [B, 1, T, 112, 112]
        # Output: [B, 64, T, 56, 56] → after MaxPool3d → [B, 64, T, 28, 28]
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64,
                      kernel_size=(7, 7, 7),
                      stride=(1, 2, 2),
                      padding=(3, 3, 3),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1)),   # [B, 64, T, 28, 28]
        )

        # ── Layer 1 — standard 3D ResBlocks (spatial: 28×28) ──────────────
        self.layer1 = nn.Sequential(
            ResBlock3D(64,  64, stride=1),
            ResBlock3D(64,  64, stride=1),
        )   # [B, 64, T, 28, 28]

        # ── Layer 2 — RTM (spatial: 28→14) ────────────────────────────────
        self.layer2 = nn.Sequential(
            ResBlock3D(64,  128, stride=2),   # first block: standard downsample
            RTM(128, 128, stride=1,           # second block: RTM
                num_heads=num_heads,
                clip_T=num_frames,
                clip_H=14, clip_W=14),
        )   # [B, 128, T, 14, 14]

        # ── Layer 3 — RTM (spatial: 14→7) ─────────────────────────────────
        self.layer3 = nn.Sequential(
            ResBlock3D(128, 256, stride=2),   # first block: standard downsample
            RTM(256, 256, stride=1,           # second block: RTM
                num_heads=num_heads,
                clip_T=num_frames,
                clip_H=7, clip_W=7),
        )   # [B, 256, T, 7, 7]

        # ── Layer 4 — standard 3D ResBlocks (spatial: 7→4) ────────────────
        self.layer4 = nn.Sequential(
            ResBlock3D(256, 512, stride=2),
            ResBlock3D(512, 512, stride=1),
        )   # [B, 512, T, 4, 4]

        # ── Global pool: collapses T, H, W to 1 ───────────────────────────
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # [B, 512, 1, 1, 1]

        # ── Regression head ────────────────────────────────────────────────
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

    # ── Weight initialisation ──────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, H, W]  — batch of grayscale video clips

        Returns:
            out : [B, 1]  — predicted LVEF values
        """
        # Add channel dim: [B, T, H, W] → [B, 1, T, H, W]
        x = x.unsqueeze(1)

        x = self.stem(x)      # [B, 64,  T, 28, 28]
        x = self.layer1(x)    # [B, 64,  T, 28, 28]
        x = self.layer2(x)    # [B, 128, T, 14, 14]
        x = self.layer3(x)    # [B, 256, T,  7,  7]
        x = self.layer4(x)    # [B, 512, T,  4,  4]

        x = self.global_pool(x)        # [B, 512, 1, 1, 1]
        x = x.flatten(1)               # [B, 512]
        x = self.regressor(x)          # [B, 1]
        return x


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    T, H, W = 16, 112, 112
    model    = EFNet(num_frames=T, dropout_rate=0.4, num_heads=4)
    dummy    = torch.randn(2, T, H, W)

    model.eval()
    with torch.no_grad():
        out = model(dummy)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Params       : {total:,}")
