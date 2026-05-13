"""
Training loop for AAPL CNN direction predictor
-----------------------------------------------
Trains the CNN on the train split, evaluates on val each epoch,
saves the best checkpoint based on val accuracy.

Usage:
    python src/training/train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.dataset import get_splits
from src.models.cnn import CNN

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-3    # L2 regularisation
PATIENCE     = 5        # early stopping — stop if val acc doesn't improve
CHECKPOINT   = Path("models/checkpoints/cnn_best.pt")

# ── Device ─────────────────────────────────────────────────────────────────────
device = (
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ── Data ───────────────────────────────────────────────────────────────────────
train_ds, val_ds, _ = get_splits()

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(device == "cuda")
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(device == "cuda")
)

# ── Model, loss, optimiser ─────────────────────────────────────────────────────
model     = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Reduce LR by 0.5 if val loss plateaus for 3 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=3
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimiser.zero_grad()

            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            total_loss += loss.item() * y.size(0)

    return total_loss / total, correct / total

# ── Training loop ──────────────────────────────────────────────────────────────
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

best_val_acc  = 0.0
patience_ctr  = 0
history       = []

print(f"\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})\n")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)

    scheduler.step(val_loss)
    current_lr = optimiser.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss,     "val_acc": val_acc,
    })

    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ctr = 0
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_acc":    val_acc,
            "val_loss":   val_loss,
        }, CHECKPOINT)
        marker = "  ← best"
    else:
        patience_ctr += 1

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
          f"{val_loss:>8.4f}  {val_acc:>7.4f}  {current_lr:>8.2e}{marker}")

    if patience_ctr >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nBest val accuracy : {best_val_acc:.4f}")
print(f"Checkpoint saved  : {CHECKPOINT.resolve()}")

# Save training history
import json
hist_path = CHECKPOINT.parent / "cnn_history.json"
with open(hist_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"History saved     : {hist_path.resolve()}")