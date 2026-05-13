"""
PyTorch Dataset for AAPL 15-minute direction prediction
--------------------------------------------------------
Builds sliding 30-bar windows of features for the CNN.

Each sample:
    X : float32 tensor of shape (13, 30)  — features × time steps
    y : long tensor — 1 (up) or 0 (down)

Normalization is per-window per-feature (z-score), so the CNN
sees relative patterns regardless of absolute price level.

Usage:
    from src.training.dataset import AAPLDataset, get_splits
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/AAPL_labeled_sentiment.csv")

WINDOW      = 60    # bars of history fed to the CNN
FEATURE_COLS = [
    "return", "hl_range", "close_position",
    "rsi", "macd", "macd_signal", "macd_hist", "roc",
    "bb_pct", "atr",
    "rel_volume", "vwap_dev", "rel_transactions",
]

# Chronological split boundaries
TRAIN_END = "2021-12-31"
VAL_END   = "2022-12-31"
# test = everything after VAL_END

def get_sentiment_scores(self) -> np.ndarray:
    """Return sentiment score for each valid sample index."""
    return np.array([
        self.df["sentiment_score"].iloc[i]
        for i in self.indices
    ])

# ── Dataset ────────────────────────────────────────────────────────────────────
class AAPLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int = WINDOW):
        """
        df     : labeled dataframe for a single split (train/val/test)
        window : number of bars per input window
        """
        self.window = window
        self.features = df[FEATURE_COLS].values.astype(np.float32)  # (N, 13)
        self.labels   = df["label"].values.astype(np.int64)         # (N,)
        self.dates    = df["Date"].values
        self.df       = df
        self.sent_col = "sentiment_score" if "sentiment_score" in df.columns else None 

        # Pre-compute valid indices — bars that:
        #   1. have a full window of history available
        #   2. are labeled 1 or 0 (not discarded -1)
        self.indices = [
            i for i in range(window, len(df))
            if self.labels[i] != -1
        ]
        print(f"  Dataset: {len(self.indices):,} samples "
              f"(window={window}, features={len(FEATURE_COLS)})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Window: bars from (i - window) to (i - 1), shape (window, features)
        window = self.features[i - self.window : i]  # (30, 13)

        # Normalize per feature within this window (z-score)
        # Each feature is centred and scaled independently
        mean = window.mean(axis=0, keepdims=True)   # (1, 13)
        std  = window.std(axis=0, keepdims=True)     # (1, 13)
        std  = np.where(std < 1e-8, 1.0, std)        # avoid division by zero
        window = (window - mean) / std               # (30, 13)

        # CNN expects (channels, length) = (features, time)
        x = torch.tensor(window.T, dtype=torch.float32)  # (13, 30)
        y = torch.tensor(self.labels[i], dtype=torch.long)

        return x, y
    def get_sentiment_scores(self) -> np.ndarray:
        """Return sentiment score for each valid sample index."""
        if self.sent_col is None:
            return np.zeros(len(self.indices), dtype=np.float32)
        return np.array([
            self.df[self.sent_col].iloc[i]
            for i in self.indices
        ], dtype=np.float32)

# ── Split helper ───────────────────────────────────────────────────────────────
def get_splits(data_path: Path = DATA_PATH):
    """
    Load the labeled CSV and return three AAPLDataset objects:
        train : 2015 – end of 2021
        val   : 2022
        test  : 2023 – 2025
    Split is strictly chronological — no data leakage.
    """
    print("Loading labeled data...")
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  {len(df):,} total bars")

    train_df = df[df["Date"] <= TRAIN_END].copy()
    val_df   = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)].copy()
    test_df  = df[df["Date"] > VAL_END].copy()

    # Reset indices so window lookback works correctly within each split
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train_df):>7,} bars  ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"  Val   : {len(val_df):>7,} bars  ({val_df['Date'].min().date()} → {val_df['Date'].max().date()})")
    print(f"  Test  : {len(test_df):>7,} bars  ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

    print("\nBuilding datasets...")
    print("  Train ", end="")
    train_ds = AAPLDataset(train_df)
    print("  Val   ", end="")
    val_ds   = AAPLDataset(val_df)
    print("  Test  ", end="")
    test_ds  = AAPLDataset(test_df)

    return train_ds, val_ds, test_ds


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_splits()

    # Check a single sample
    x, y = train_ds[0]
    print(f"\nSample check:")
    print(f"  x shape : {x.shape}   (features × time)")
    print(f"  y       : {y.item()}  (1=up, 0=down)")
    print(f"  x mean  : {x.mean().item():.4f}  (should be ~0 after normalisation)")
    print(f"  x std   : {x.std().item():.4f}   (should be ~1 after normalisation)")

    # Label balance per split
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        labels = [ds[i][1].item() for i in range(len(ds))]
        up   = sum(l == 1 for l in labels)
        down = sum(l == 0 for l in labels)
        print(f"  {name} balance — up: {up:,}  down: {down:,}  ratio: {up/down:.3f}")