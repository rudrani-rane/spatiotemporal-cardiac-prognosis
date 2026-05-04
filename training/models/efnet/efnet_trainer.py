"""
efnet_trainer.py
────────────────
Training loop for EF-Net.

Run from project root:
    python -m training.models.efnet.efnet_trainer
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.models.efnet.efnet_dataset_loader import EFNetEchoDataset
from training.models.efnet.efnet_model import EFNet
from training.metrics import compute_all_metrics

# ── Paths ──────────────────────────────────────────────────────────────────
SAVE_DIR        = os.path.dirname(__file__)
MODEL_BEST_PATH = os.path.join(SAVE_DIR, "efnet_model.pth")
LOG_PATH        = os.path.join(SAVE_DIR, "efnet_epoch_log.csv")

# ══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS  — edit these freely
# ══════════════════════════════════════════════════════════════════════════
EPOCHS       = 60
BATCH_SIZE   = 4       # EF-Net processes 16 frames — needs more GPU memory
                       # reduce to 2 if you get CUDA OOM
NUM_FRAMES   = 16      # clip length (must match efnet_model.py)
NUM_HEADS    = 4       # attention heads in RTM layers
LR_HEAD      = 3e-4   # regression head (new weights)
LR_BACKBONE  = 5e-5   # 3D conv + RTM layers

# Huber loss delta — same reasoning as CNN/ResNet trainers:
# delta=5.0 → quadratic for errors <5 EF points, linear beyond
HUBER_DELTA  = 5.0
# ══════════════════════════════════════════════════════════════════════════

# ── CSV Logger ─────────────────────────────────────────────────────────────
CSV_HEADER = [
    "epoch",
    "train_loss", "train_MAE", "train_RMSE", "train_MAPE", "train_R2",
    "train_Acc5", "train_Acc3", "train_Acc2",
    "val_loss",   "val_MAE",   "val_RMSE",   "val_MAPE",   "val_R2",
    "val_Acc5",   "val_Acc3",  "val_Acc2",
    "lr",
]


def init_log(path: str):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_log(path, epoch, tr_loss, tr_m, vl_loss, vl_m, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            round(tr_loss, 5),
            round(tr_m["MAE"],   4), round(tr_m["RMSE"],  4),
            round(tr_m["MAPE"],  4), round(tr_m["R2"],    4),
            round(tr_m["Acc@5"], 4), round(tr_m["Acc@3"], 4), round(tr_m["Acc@2"], 4),
            round(vl_loss, 5),
            round(vl_m["MAE"],   4), round(vl_m["RMSE"],  4),
            round(vl_m["MAPE"],  4), round(vl_m["R2"],    4),
            round(vl_m["Acc@5"], 4), round(vl_m["Acc@3"], 4), round(vl_m["Acc@2"], 4),
            f"{lr:.2e}",
        ])


# ── Differential parameter groups ─────────────────────────────────────────
def get_param_groups(model: EFNet):
    """
    Backbone layers (3D convs, RTM attention) get LR_BACKBONE.
    Regression head gets LR_HEAD.
    This mirrors the ResNet-18 trainer strategy — head needs faster updates
    since its weights are random; backbone needs gentler updates.
    """
    head_ids        = {id(p) for p in model.regressor.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params     = list(model.regressor.parameters())
    return [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ]


# ── One training epoch ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x).squeeze(-1)          # [B]
        loss  = criterion(preds, y)
        loss.backward()

        # Gradient clipping — important for attention layers which can
        # produce larger gradients than pure conv models
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.cpu())

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return total_loss / len(loader), compute_all_metrics(preds_cat, targets_cat)


# ── Validation ─────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            preds   = model(x).squeeze(-1)
            loss    = criterion(preds, y)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return total_loss / len(loader), compute_all_metrics(preds_cat, targets_cat)


# ── Main ───────────────────────────────────────────────────────────────────
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  EF-Net — Training")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {EPOCHS}  |  Batch : {BATCH_SIZE}")
    print(f"  Frames/clip : {NUM_FRAMES}  |  Heads : {NUM_HEADS}")
    print(f"  LR head     : {LR_HEAD}  |  LR backbone : {LR_BACKBONE}")
    print(f"  Loss        : HuberLoss(delta={HUBER_DELTA})")
    print(f"  Scheduler   : ReduceLROnPlateau (factor=0.5, patience=6)")
    print(f"{'='*62}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    # Train: random window sampling (temporal augmentation)
    # Val/Test: uniform sampling (deterministic, reproducible)
    train_ds = EFNetEchoDataset(split=0, sample_mode="random")
    val_ds   = EFNetEchoDataset(split=1, sample_mode="uniform")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)} samples\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = EFNet(num_frames=NUM_FRAMES,
                      dropout_rate=0.4,
                      num_heads=NUM_HEADS).to(device)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = optim.Adam(get_param_groups(model), weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 6,
        min_lr   = 1e-6,
    )

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params : {total_params:,}\n")

    best_val_mae = float("inf")
    os.makedirs(SAVE_DIR, exist_ok=True)
    init_log(LOG_PATH)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_m = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]["lr"]

        # ── Console output ─────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.2e})")
        print(f"  Train Loss : {train_loss:.4f}   Val Loss : {val_loss:.4f}")
        print(f"  {'Metric':<10} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*34}")
        for k in ("MAE", "RMSE", "MAPE", "R2"):
            print(f"  {k:<10} {train_m[k]:>10.3f} {val_m[k]:>10.3f}")
        for k in ("Acc@5", "Acc@3", "Acc@2"):
            print(f"  {k:<10} {train_m[k]*100:>9.2f}% {val_m[k]*100:>9.2f}%")

        # ── Log ───────────────────────────────────────────────────────
        append_log(LOG_PATH, epoch, train_loss, train_m,
                   val_loss, val_m, current_lr)

        # ── Checkpoint ────────────────────────────────────────────────
        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"  ✔ Best model saved  (val_MAE = {best_val_mae:.3f})")

    print(f"\n{'='*62}")
    print(f"  Training complete.")
    print(f"  Best Val MAE : {best_val_mae:.3f}")
    print(f"  Model saved  : {MODEL_BEST_PATH}")
    print(f"  Epoch log    : {LOG_PATH}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run_training()
