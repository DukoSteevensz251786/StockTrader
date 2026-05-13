"""
Paper trading bot — AAPL 15-minute direction predictor
-------------------------------------------------------
Runs every minute during market hours.
Makes BUY/SELL decisions based on CNN + XGBoost + sentiment.
Logs all decisions to data/live/trades.csv — no real execution.

Usage:
    python src/live/bot.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.live.feed import AlpacaFeed
from src.live.marketaux import NewsFetcher
from src.models.cnn import CNN
from src.training.dataset import FEATURE_COLS, WINDOW
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ── Config ─────────────────────────────────────────────────────────────────────
CNN_CHECKPOINT = Path("models/checkpoints/cnn_best.pt")
XGB_CHECKPOINT = Path("models/checkpoints/xgb_pipeline.json")
TRADES_LOG     = Path("data/live/trades.csv")
SYMBOL         = "AAPL"
PROB_THRESHOLD = 0.60       # minimum XGBoost confidence to trade
CAPITAL        = 10_000.0   # virtual capital per trade ($)
POLL_SECONDS   = 60         # check every 60 seconds

ET = ZoneInfo("America/New_York")

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
cnn = CNN().to(device)
ckpt = torch.load(CNN_CHECKPOINT, map_location=device)
cnn.load_state_dict(ckpt["model_state"])
cnn.eval()
print(f"  CNN loaded (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_CHECKPOINT)
print(f"  XGBoost loaded")

# ── Feature engineering (mirrors src/data/feature_engineering.py) ─────────────
def compute_features(bars: pd.DataFrame) -> np.ndarray | None:
    """
    Takes a DataFrame of raw OHLCV bars and returns a
    (WINDOW, 13) numpy array of features — or None if not enough data.
    """
    if len(bars) < WINDOW + 30:   # need extra bars for indicator warmup
        return None

    df = bars.copy().reset_index(drop=True)

    # Price
    df["return"]         = df["Close"].pct_change()
    df["hl_range"]       = (df["High"] - df["Low"]) / df["Close"]
    hl = df["High"] - df["Low"]
    df["close_position"] = np.where(hl > 0, (df["Close"] - df["Low"]) / hl, 0.5)

    # Momentum
    df["rsi"]         = RSIIndicator(close=df["Close"], window=14).rsi() / 100.0
    macd              = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["roc"]         = ROCIndicator(close=df["Close"], window=10).roc() / 100.0

    # Volatility
    bb             = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"]      = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range() / df["Close"]

    # Volume
    rolling_vol        = df["Volume"].rolling(20).mean()
    df["rel_volume"]   = np.where(rolling_vol > 0, df["Volume"] / rolling_vol, 1.0)
    df["typical_price"]= (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol         = (df["typical_price"] * df["Volume"]).cumsum()
    cum_vol            = df["Volume"].cumsum()
    vwap               = np.where(cum_vol > 0, cum_tp_vol / cum_vol, df["typical_price"])
    df["vwap_dev"]     = (df["Close"] - vwap) / vwap
    rolling_tx         = df["Transactions"].rolling(20).mean()
    df["rel_transactions"] = np.where(rolling_tx > 0, df["Transactions"] / rolling_tx, 1.0)

    df = df.dropna()
    if len(df) < WINDOW:
        return None

    # Take the last WINDOW bars
    window = df[FEATURE_COLS].values[-WINDOW:].astype(np.float32)

    # Normalize per feature (z-score) — mirrors dataset.py
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    window = (window - mean) / std

    return window   # (WINDOW, 13)


def get_embedding(window: np.ndarray) -> np.ndarray:
    """Run window through CNN, return 128-dim embedding."""
    x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = cnn.extract_features(x)
    return emb.cpu().numpy()   # (1, 128)


def get_signal(embedding: np.ndarray, sentiment: float):
    """
    Combine CNN embedding + sentiment, run through XGBoost.
    Returns (direction, probability) or (None, None) if below threshold.

    direction: 'BUY' or 'SELL'
    probability: XGBoost confidence (0.6 to 1.0)
    """
    # ── ADD SENTIMENT HERE ─────────────────────────────────────────────────
    # Uncomment when sentiment pipeline is ready:
    #features = np.hstack([embedding, [[sentiment]]])
    features = embedding 
    # ──────────────────────────────────────────────────────────────────────

    proba = xgb_model.predict_proba(features)[0]   # [p_down, p_up]
    p_up, p_down = proba[1], proba[0]

    if p_up >= PROB_THRESHOLD:
        return "BUY", float(p_up)
    if p_down >= PROB_THRESHOLD:
        return "SELL", float(p_down)
    return None, None


def size_position(probability: float) -> float:
    """Scale position size linearly with confidence above threshold."""
    return CAPITAL * (probability - PROB_THRESHOLD) / (1.0 - PROB_THRESHOLD)


def log_trade(record: dict):
    """Append a trade record to the CSV log."""
    TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not TRADES_LOG.exists()
    df.to_csv(TRADES_LOG, mode="a", header=write_header, index=False)


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    feed    = AlpacaFeed()
    news    = NewsFetcher()

    print(f"\nBot started — symbol={SYMBOL}, threshold={PROB_THRESHOLD}, capital=${CAPITAL:,.0f}")
    print(f"Logging trades to {TRADES_LOG.resolve()}")
    print(f"Press Ctrl+C to stop.\n")

    while True:
        now = datetime.now(ET)

        # Only run during market hours
        if not feed.is_market_open():
            print(f"[{now.strftime('%H:%M:%S')}] Market closed — waiting...")
            time.sleep(60)
            continue

        print(f"[{now.strftime('%H:%M:%S')}] Fetching data...", end=" ", flush=True)

        # 1. Get latest bars
        bars = feed.get_latest_bars(SYMBOL, limit=WINDOW + 40)
        if bars.empty:
            print("no bars returned, skipping.")
            time.sleep(POLL_SECONDS)
            continue

        # 2. Compute features
        window = compute_features(bars)
        if window is None:
            print("not enough data, skipping.")
            time.sleep(POLL_SECONDS)
            continue

        # 3. Get CNN embedding
        embedding = get_embedding(window)

        # 4. Get sentiment
        sentiment = news.get_current_sentiment()

        # 5. Get signal
        direction, probability = get_signal(embedding, sentiment)

        current_price = float(bars["Close"].iloc[-1])

        if direction is None:
            print(f"no signal (max prob below {PROB_THRESHOLD})")
        else:
            position_value = size_position(probability)
            shares         = position_value / current_price

            record = {
                "timestamp"      : now.isoformat(),
                "symbol"         : SYMBOL,
                "direction"      : direction,
                "probability"    : round(probability, 4),
                "price"          : round(current_price, 4),
                "position_value" : round(position_value, 2),
                "shares"         : round(shares, 4),
                "sentiment"      : round(sentiment, 4),
                "outcome"        : None,   # filled in 15 min later
                "pnl"            : None,
            }

            log_trade(record)
            print(f"{direction} @ ${current_price:.2f} | "
                  f"prob={probability:.3f} | "
                  f"size=${position_value:.0f} | "
                  f"sentiment={sentiment:.3f}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
