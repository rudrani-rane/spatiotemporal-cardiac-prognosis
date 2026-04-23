import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.models.cardio_cnn.cnn_dataset_loader import CNNEchoDataset
from training.models.cardio_cnn.cnn_model import CardioCNNModel
from training.metrics import compute_all_metrics


SAVE_DIR        = os.path.dirname(__file__)
MODEL_BEST_PATH = os.path.join(SAVE_DIR, "cnn_model.pth")
LOG_PATH        = os.path.join(SAVE_DIR, "cnn_epoch_log.csv")

# ══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS 
# ══════════════════════════════════════════════════════════════════════════
EPOCHS      = 60
BATCH_SIZE  = 8
LR          = 1e-4


HUBER_DELTA = 5.0



CSV_HEADER = [
    "epoch",
    "train_loss", "train_MAE", "train_RMSE", "train_MAPE", "train_R2",
    "train_Acc5", "train_Acc3", "train_Acc2",
    "val_loss",   "val_MAE",   "val_RMSE",   "val_MAPE",   "val_R2",
    "val_Acc5",   "val_Acc3",  "val_Acc2",
]


def init_log(path: str):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_log(path, epoch, tr_loss, tr_m, vl_loss, vl_m):
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
        ])



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x).squeeze(-1)          # [B]
        loss  = criterion(preds, y)
        loss.backward()

        # Gradient clipping stabilises training on small datasets
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss  += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics     = compute_all_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds  = model(x).squeeze(-1)
            loss   = criterion(preds, y)
            total_loss  += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics     = compute_all_metrics(all_preds, all_targets)
    return total_loss / len(loader), metrics, all_preds, all_targets


# ── Main ───────────────────────────────────────────────────────────────────
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Cardio-CNN EF Model – Training")
    print(f"  Device : {device}  |  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}")
    print(f"{'='*55}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = CNNEchoDataset(split=0)
    val_dataset   = CNNEchoDataset(split=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = CardioCNNModel(dropout_rate=0.4).to(device)

    # Huber loss: quadratic for |error| < delta, linear beyond.
    # delta=5.0 matches EF error scale (~5-10 EF points typical).
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # ReduceLROnPlateau: halve LR when val_loss stagnates for 6 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6
    )

    best_val_mae = float("inf")
    os.makedirs(SAVE_DIR, exist_ok=True)
    init_log(LOG_PATH)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics, _, _ = validate(
            model, val_loader, criterion, device
        )

        # ReduceLROnPlateau needs the metric it monitors
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Console output ─────────────────────────────────────────────
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.2e}  huber_delta={HUBER_DELTA})")
        print(f"  Train Loss : {train_loss:.4f}   Val Loss : {val_loss:.4f}")
        print(f"  {'Metric':<10} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*34}")
        for k in ("MAE", "RMSE", "MAPE", "R2"):
            print(f"  {k:<10} {train_metrics[k]:>10.3f} {val_metrics[k]:>10.3f}")
        for k in ("Acc@5", "Acc@3", "Acc@2"):
            print(f"  {k:<10} {train_metrics[k]*100:>9.2f}% {val_metrics[k]*100:>9.2f}%")

        # ── Log & checkpoint ───────────────────────────────────────────
        append_log(LOG_PATH, epoch, train_loss, train_metrics, val_loss, val_metrics)

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"  ✔ Best model saved  (val_MAE = {best_val_mae:.3f})")

    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Best Val MAE : {best_val_mae:.3f}")
    print(f"  Model saved  : {MODEL_BEST_PATH}")
    print(f"  Epoch log    : {LOG_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_training()