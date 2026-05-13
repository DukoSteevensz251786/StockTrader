"""
Label generation for AAPL 15-minute direction prediction
---------------------------------------------------------
Reads AAPL_features.csv and adds a direction label per bar.

Label logic (filtered):
    1  = close[t+15] > close[t] by more than THRESHOLD
    0  = close[t+15] < close[t] by more than THRESHOLD
   -1  = move is within threshold (discard — too flat to learn from)

Also discards bars where t+15 crosses into the next trading day
(we never want to predict across the close).

Usage:
    python src/training/labels.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data/processed/AAPL_features.csv")
OUTPUT_PATH  = Path("data/processed/AAPL_labeled.csv")
LOOKAHEAD    = 15       # minutes ahead to predict
THRESHOLD    = 0.001    # 0.10% minimum move to label (not discard)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading feature data...")
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"  {len(df):,} bars loaded")

# ── Compute forward return ─────────────────────────────────────────────────────
# shift(-15) gives the close price 15 bars ahead
# This only works correctly because we forward-filled missing minutes —
# every bar is exactly 1 minute apart within a trading day
df["fwd_close"]  = df["Close"].shift(-LOOKAHEAD)
df["fwd_return"] = df["fwd_close"] / df["Close"] - 1

# ── Drop cross-day lookaheads ──────────────────────────────────────────────────
# If t and t+15 are on different dates, the prediction crosses the close —
# we don't want the model learning overnight gaps as intraday signals
df["date_only"]     = df["Date"].dt.date
df["fwd_date_only"] = df["Date"].shift(-LOOKAHEAD).dt.date

cross_day = df["date_only"] != df["fwd_date_only"]
df.loc[cross_day, "fwd_return"] = np.nan
print(f"  Dropped {cross_day.sum():,} cross-day bars")

# ── Assign labels ──────────────────────────────────────────────────────────────
def assign_label(ret):
    if pd.isna(ret):
        return -1
    if ret > THRESHOLD:
        return 1
    if ret < -THRESHOLD:
        return 0
    return -1   # flat — discard

df["label"] = df["fwd_return"].apply(assign_label)

# ── Stats ──────────────────────────────────────────────────────────────────────
total     = len(df)
up        = (df["label"] == 1).sum()
down      = (df["label"] == 0).sum()
discarded = (df["label"] == -1).sum()

print(f"\nLabel distribution (threshold={THRESHOLD*100:.2f}%):")
print(f"  UP       (1): {up:>7,}  ({up/total*100:.1f}%)")
print(f"  DOWN     (0): {down:>7,}  ({down/total*100:.1f}%)")
print(f"  DISCARD (-1): {discarded:>7,}  ({discarded/total*100:.1f}%)")
print(f"  Class balance (up/down): {up/down:.3f}  (1.0 = perfect)")

# ── Save full dataset (including discarded rows marked -1) ────────────────────
# We keep discarded rows in the file so we can filter them at training time.
# This makes it easy to experiment with different thresholds later.
df = df.drop(columns=["fwd_close", "fwd_date_only", "date_only"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH.resolve()}")
print(f"Final shape: {df.shape}")
print(f"\nSample (first 5 labeled bars):")
print(df[df["label"] != -1][["Date", "Close", "fwd_return", "label"]].head())