"""
Feature engineering for AAPL minute data
-----------------------------------------
Reads the preprocessed CSV and computes technical indicators.
Output is saved to data/processed/AAPL_features.csv

Indicators computed:
    Price-based   : returns, high-low range, close position in bar
    Momentum      : RSI, MACD, MACD signal, MACD histogram, ROC
    Volatility    : Bollinger %B, ATR
    Volume        : relative volume, VWAP deviation, transactions ratio

Usage:
    python src/data/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/AAPL_preprocessed.csv")
OUTPUT_PATH = Path("data/processed/AAPL_features.csv")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"  {len(df):,} bars loaded")

# ── Price-based features ───────────────────────────────────────────────────────
print("Computing price-based features...")

# Percentage return close-to-close
df["return"] = df["Close"].pct_change()

# High-low range as % of close — volatility proxy per bar
df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

# Where did close land within the bar? 0 = at the low, 1 = at the high
# Avoids division by zero on flat bars (high == low)
hl = df["High"] - df["Low"]
df["close_position"] = np.where(
    hl > 0,
    (df["Close"] - df["Low"]) / hl,
    0.5
)

# ── Momentum indicators ────────────────────────────────────────────────────────
print("Computing momentum indicators...")

# RSI (14 period) normalised to 0–1
df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi() / 100.0

# MACD: fast=12, slow=26, signal=9
macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
df["macd"]        = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"]   = macd.macd_diff()

# Rate of change over 10 bars
df["roc"] = ROCIndicator(close=df["Close"], window=10).roc() / 100.0

# ── Volatility indicators ──────────────────────────────────────────────────────
print("Computing volatility indicators...")

# Bollinger Bands (20 period, 2 std) — %B: 0=lower band, 1=upper band
bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
df["bb_pct"] = bb.bollinger_pband()

# ATR (14 period) normalised by close price
df["atr"] = AverageTrueRange(
    high=df["High"], low=df["Low"], close=df["Close"], window=14
).average_true_range() / df["Close"]

# ── Volume indicators ──────────────────────────────────────────────────────────
print("Computing volume indicators...")

# Relative volume vs 20-bar rolling average
rolling_vol = df["Volume"].rolling(20).mean()
df["rel_volume"] = np.where(rolling_vol > 0, df["Volume"] / rolling_vol, 1.0)

# VWAP deviation — resets every trading day
df["date_only"]     = df["Date"].dt.date
df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3

def compute_vwap(group):
    cum_tp_vol = (group["typical_price"] * group["Volume"]).cumsum()
    cum_vol    = group["Volume"].cumsum()
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, group["typical_price"])
    return pd.Series(vwap, index=group.index)

print("  Computing VWAP per day (this takes a moment)...")
df["vwap"]     = df.groupby("date_only", group_keys=False).apply(compute_vwap)
df["vwap_dev"] = (df["Close"] - df["vwap"]) / df["vwap"]

# Transactions ratio vs 20-bar rolling average
rolling_tx = df["Transactions"].rolling(20).mean()
df["rel_transactions"] = np.where(rolling_tx > 0, df["Transactions"] / rolling_tx, 1.0)

# ── Clean up intermediate columns ─────────────────────────────────────────────
df = df.drop(columns=["date_only", "typical_price", "vwap"])

# ── Drop NaN rows ──────────────────────────────────────────────────────────────
before = len(df)
df = df.dropna().reset_index(drop=True)
after  = len(df)
print(f"\nDropped {before - after:,} warmup/NaN rows ({before:,} → {after:,})")

# ── Select final feature columns ───────────────────────────────────────────────
FEATURE_COLS = [
    "Date", "Open", "High", "Low", "Close", "Volume", "Transactions",
    "return", "hl_range", "close_position",
    "rsi", "macd", "macd_signal", "macd_hist", "roc",
    "bb_pct", "atr",
    "rel_volume", "vwap_dev", "rel_transactions",
]

df = df[FEATURE_COLS]

# ── Sanity check ───────────────────────────────────────────────────────────────
print(f"\nFeature summary:")
print(df[FEATURE_COLS[7:]].describe().round(4))
print(f"\nNull check:\n{df.isnull().sum()}")

# ── Save ───────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH.resolve()}")
print(f"Final shape: {df.shape}")