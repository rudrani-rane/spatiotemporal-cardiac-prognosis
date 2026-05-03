import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.models.resnet_18.resnet18_dataset_loader import ResNetEchoDataset
from training.models.resnet_18.resnet18_model import ResNet18EFModel
from training.metrics import compute_all_metrics

# ── Paths ──────────────────────────────────────────────────────────────────
SAVE_DIR        = os.path.dirname(__file__)
MODEL_BEST_PATH = os.path.join(SAVE_DIR, "resnet18_model.pth")
LOG_PATH        = os.path.join(SAVE_DIR, "resnet18_epoch_log.csv")

# ══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS  — edit these freely
# ══════════════════════════════════════════════════════════════════════════
EPOCHS         = 60
BATCH_SIZE     = 8
LR_HEAD        = 3e-4   # regression head  (newly initialised weights)
LR_BACKBONE    = 5e-5   # pretrained layers (needs to adapt to echo images)
PRETRAINED     = True   # False = train from random init
USE_AUGMENT    = False  # keep False — augmentation hurt convergence here

# Huber loss delta — boundary between MSE and MAE behaviour:
#   errors < delta  → squared loss  (strongly penalises small errors)
#   errors >= delta → linear loss   (robust to large outliers)
# EF values range ~10–85%, typical prediction error ~5–10 EF points.
# delta=5.0: quadratic for errors <5 EF pts, linear (outlier-robust) beyond.
HUBER_DELTA    = 5.0
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


def init_log(path):
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


# ── Parameter groups: lower LR for backbone, higher for head ──────────────
def get_param_groups(model):
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
        preds = model(x).squeeze(-1)
        loss  = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            x, y  = x.to(device), y.to(device)
            preds  = model(x).squeeze(-1)
            loss   = criterion(preds, y)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    return total_loss / len(loader), compute_all_metrics(preds_cat, targets_cat)


# ── Main ───────────────────────────────────────────────────────────────────
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  ResNet-18 EF Model — Training  (pretrained={PRETRAINED})")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {EPOCHS}  |  Batch : {BATCH_SIZE}")
    print(f"  LR head    : {LR_HEAD}  |  LR backbone : {LR_BACKBONE}")
    print(f"  Scheduler  : ReduceLROnPlateau (factor=0.5, patience=6)")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = ResNetEchoDataset(split=0, augment=USE_AUGMENT)
    val_ds   = ResNetEchoDataset(split=1, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)} samples\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ResNet18EFModel(pretrained=PRETRAINED, dropout_rate=0.4).to(device)
    # Huber loss with tuned delta — see HUBER_DELTA hyperparameter above
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    # Single Adam with differential LRs — NO warm-up, NO cosine annealing
    optimizer = optim.Adam(get_param_groups(model), weight_decay=1e-4)

    # ReduceLROnPlateau: halve LR when val_loss stops improving for 6 epochs
    # This is the same scheduler the CNN used — proven to work on this dataset
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "min",
        factor    = 0.5,
        patience  = 6,
        min_lr    = 1e-6,
    )

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

        # Step scheduler on val_loss (NOT val_MAE — keeps it consistent)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[-1]["lr"]   # head LR

        # ── Print ─────────────────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.2e})")
        print(f"  Train Loss : {train_loss:.4f}   Val Loss : {val_loss:.4f}")
        print(f"  {'Metric':<10} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*34}")
        for k in ("MAE", "RMSE", "MAPE", "R2"):
            print(f"  {k:<10} {train_m[k]:>10.3f} {val_m[k]:>10.3f}")
        for k in ("Acc@5", "Acc@3", "Acc@2"):
            print(f"  {k:<10} {train_m[k]*100:>9.2f}% {val_m[k]*100:>9.2f}%")

        # ── Log ───────────────────────────────────────────────────────────
        append_log(LOG_PATH, epoch, train_loss, train_m,
                   val_loss, val_m, current_lr)

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_m["MAE"] < best_val_mae:
            best_val_mae = val_m["MAE"]
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"  ✔ Best model saved  (val_MAE = {best_val_mae:.3f})")

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best Val MAE : {best_val_mae:.3f}")
    print(f"  Model saved  : {MODEL_BEST_PATH}")
    print(f"  Epoch log    : {LOG_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_training()