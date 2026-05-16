"""
Paper trading bot — AAPL 15-minute direction predictor
-------------------------------------------------------
Runs every minute during market hours.
Uses CNN directly for predictions (no XGBoost layer).
Submits real orders to Alpaca paper trading account.
Logs all decisions to data/live/trades.csv.

Usage:
    python src/live/bot.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

# Alpaca trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── Config ─────────────────────────────────────────────────────────────────────
CNN_CHECKPOINT   = Path("models/checkpoints/cnn_best.pt")
TRADES_LOG       = Path("data/live/trades.csv")
SYMBOL           = "AAPL"
PROB_THRESHOLD   = 0.55
CAPITAL          = 10_000.0
POLL_SECONDS     = 60
NEWS_REFRESH_MIN = 5
HOLD_MINUTES     = 15
MAX_BAR_AGE_MIN  = 3

ET = ZoneInfo("America/New_York")

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Load CNN ───────────────────────────────────────────────────────────────────
print("Loading models...")
cnn = CNN().to(device)
ckpt = torch.load(CNN_CHECKPOINT, map_location=device)
cnn.load_state_dict(ckpt["model_state"])
cnn.eval()
print(f"  CNN loaded (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})")

# ── Alpaca trading client ──────────────────────────────────────────────────────
trading_client = TradingClient(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    paper=True
)
account = trading_client.get_account()
print(f"  Alpaca paper trading connected")
print(f"  Buying power: ${float(account.buying_power):,.2f}")

# ── Alpaca helpers ─────────────────────────────────────────────────────────────
def submit_order(direction: str, shares: float) -> str | None:
    try:
        side  = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol        = SYMBOL,
                qty           = round(max(shares, 1.0), 2),
                side          = side,
                time_in_force = TimeInForce.DAY,
            )
        )
        print(f"  → Alpaca order: {side.value} {round(shares,2)} shares (id={order.id})")
        return str(order.id)
    except Exception as e:
        print(f"  → Order failed: {e}")
        return None


def close_position() -> bool:
    try:
        for pos in trading_client.get_all_positions():
            if pos.symbol == SYMBOL:
                trading_client.close_position(SYMBOL)
                print(f"  → Closed {SYMBOL} "
                      f"({pos.qty} shares, P&L=${float(pos.unrealized_pl):+.2f})")
                return True
        return False
    except Exception as e:
        print(f"  → Failed to close: {e}")
        return False


def get_open_position() -> dict | None:
    try:
        for pos in trading_client.get_all_positions():
            if pos.symbol == SYMBOL:
                return {
                    "qty"          : float(pos.qty),
                    "side"         : str(pos.side),
                    "entry_price"  : float(pos.avg_entry_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                }
        return None
    except Exception:
        return None


# ── Macro features ─────────────────────────────────────────────────────────────
def get_macro_features() -> dict:
    """
    Fetch today's macro context from yfinance.
    Returns daily SPY, QQQ, VIX features.
    Cached per day — only fetches once per trading day.
    """
    import yfinance as yf
    from ta.momentum import RSIIndicator as RSI

    result = {}
    try:
        for name, symbol in [("spy", "SPY"), ("qqq", "QQQ"), ("vix", "^VIX")]:
            df = yf.Ticker(symbol).history(period="30d", interval="1d")
            if df.empty:
                continue
            df = df.reset_index()

            if name == "vix":
                result["vix_level"]  = float(df["Close"].iloc[-1]) / 20.0
                result["vix_return"] = float(df["Close"].pct_change().iloc[-1])
            else:
                close  = df["Close"]
                volume = df["Volume"]
                result[f"{name}_return"]     = float(close.pct_change().iloc[-1])
                result[f"{name}_rsi"]        = float(RSI(close=close, window=14).rsi().iloc[-1]) / 100.0
                rolling_vol = volume.rolling(20).mean()
                result[f"{name}_rel_volume"] = float(
                    volume.iloc[-1] / rolling_vol.iloc[-1]
                    if rolling_vol.iloc[-1] > 0 else 1.0
                )
    except Exception as e:
        print(f"  [Macro fetch error] {e}")

    # Defaults if fetch fails
    defaults = {
        "spy_return": 0.0, "spy_rsi": 0.5, "spy_rel_volume": 1.0,
        "qqq_return": 0.0, "qqq_rsi": 0.5, "qqq_rel_volume": 1.0,
        "vix_level" : 0.9, "vix_return": 0.0,
    }
    for k, v in defaults.items():
        result.setdefault(k, v)

    return result


# ── Feature engineering ────────────────────────────────────────────────────────
MACRO_COLS = [
    "spy_return", "spy_rsi", "spy_rel_volume",
    "qqq_return", "qqq_rsi", "qqq_rel_volume",
    "vix_level",  "vix_return",
]

def compute_features(bars: pd.DataFrame, macro: dict) -> np.ndarray | None:
    if len(bars) < WINDOW + 30:
        return None

    df = bars.copy().reset_index(drop=True)

    # AAPL price features
    df["return"]         = df["Close"].pct_change()
    df["hl_range"]       = (df["High"] - df["Low"]) / df["Close"]
    hl = df["High"] - df["Low"]
    df["close_position"] = np.where(hl > 0, (df["Close"] - df["Low"]) / hl, 0.5)
    df["rsi"]            = RSIIndicator(close=df["Close"], window=14).rsi() / 100.0
    macd                 = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]           = macd.macd()
    df["macd_signal"]    = macd.macd_signal()
    df["macd_hist"]      = macd.macd_diff()
    df["roc"]            = ROCIndicator(close=df["Close"], window=10).roc() / 100.0
    bb                   = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_pct"]         = bb.bollinger_pband()
    df["atr"]            = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range() / df["Close"]
    rolling_vol            = df["Volume"].rolling(20).mean()
    df["rel_volume"]       = np.where(rolling_vol > 0, df["Volume"] / rolling_vol, 1.0)
    df["typical_price"]    = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol             = (df["typical_price"] * df["Volume"]).cumsum()
    cum_vol                = df["Volume"].cumsum()
    vwap                   = np.where(cum_vol > 0, cum_tp_vol / cum_vol, df["typical_price"])
    df["vwap_dev"]         = (df["Close"] - vwap) / vwap
    rolling_tx             = df["Transactions"].rolling(20).mean()
    df["rel_transactions"] = np.where(rolling_tx > 0, df["Transactions"] / rolling_tx, 1.0)

    # Add macro features as constant columns for this window
    for col in MACRO_COLS:
        df[col] = macro.get(col, 0.0)

    df = df.dropna()
    if len(df) < WINDOW:
        return None

    # All 21 features
    all_features = FEATURE_COLS + MACRO_COLS
    window = df[all_features].values[-WINDOW:].astype(np.float32)

    # Normalise AAPL features per window (z-score)
    # Keep macro features unnormalised — they're already scaled
    aapl_window  = window[:, :len(FEATURE_COLS)]
    macro_window = window[:, len(FEATURE_COLS):]

    mean = aapl_window.mean(axis=0, keepdims=True)
    std  = aapl_window.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    aapl_window = (aapl_window - mean) / std

    return np.concatenate([aapl_window, macro_window], axis=1)  # (60, 21)


def get_signal(window: np.ndarray):
    """
    Run window through CNN directly.
    Returns (direction, probability) or (None, None).
    """
    x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 21, 60)
    with torch.no_grad():
        logits = cnn(x)                          # (1, 2)
        proba  = F.softmax(logits, dim=1)[0]     # (2,)

    p_up   = float(proba[1])
    p_down = float(proba[0])

    if p_up >= PROB_THRESHOLD:
        return "BUY", p_up
    if p_down >= PROB_THRESHOLD:
        return "SELL", p_down
    return None, None


def size_position(probability: float, price: float) -> float:
    value = CAPITAL * (probability - PROB_THRESHOLD) / (1.0 - PROB_THRESHOLD)
    return max(1.0, value / price)


def log_trade(record: dict):
    TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not TRADES_LOG.exists()
    df.to_csv(TRADES_LOG, mode="a", header=write_header, index=False)


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    feed = AlpacaFeed()
    news = NewsFetcher()

    print(f"\nBot started — symbol={SYMBOL}, threshold={PROB_THRESHOLD}, "
          f"capital=${CAPITAL:,.0f}, hold={HOLD_MINUTES}min")
    print(f"Logging to {TRADES_LOG.resolve()}")
    print(f"Press Ctrl+C to stop.\n")

    open_order_time   = None
    last_news_time    = None
    last_macro_date   = None
    current_sentiment = 0.0
    current_macro     = {}

    while True:
        now = datetime.now(ET)

        # ── Market closed ──────────────────────────────────────────────────────
        if not feed.is_market_open():
            print(f"[{now.strftime('%H:%M:%S')}] Market closed — waiting...")
            open_order_time = None  # reset tracking, don't try to close
            time.sleep(60)
            continue

        print(f"[{now.strftime('%H:%M:%S')}] Fetching data...", end=" ", flush=True)

        # ── Fetch bars ─────────────────────────────────────────────────────────
        bars = feed.get_latest_bars(SYMBOL, limit=WINDOW + 60)
        if bars.empty:
            print("no bars returned, skipping.")
            time.sleep(POLL_SECONDS)
            continue

        latest_time  = bars["Date"].iloc[-1]
        latest_price = float(bars["Close"].iloc[-1])
        print(f"bar={latest_time.strftime('%H:%M')} price=${latest_price:.2f}", end=" ")

        # ── Staleness check ────────────────────────────────────────────────────
        bar_age = (datetime.now(ET).replace(tzinfo=None) - latest_time).total_seconds() / 60
        if bar_age > MAX_BAR_AGE_MIN:
            print(f"— stale ({bar_age:.0f}min old), skipping.")
            time.sleep(POLL_SECONDS)
            continue

        # ── Auto-close after HOLD_MINUTES ─────────────────────────────────────
        if open_order_time and (now - open_order_time) >= timedelta(minutes=HOLD_MINUTES):
            print(f"\n  {HOLD_MINUTES}min complete — closing position...")
            close_position()
            open_order_time = None

        # ── Skip if in position ────────────────────────────────────────────────
        if open_order_time is not None:
            pos = get_open_position()
            if pos:
                print(f"— holding ({pos['side']} {pos['qty']} shares, "
                      f"P&L=${pos['unrealized_pl']:+.2f})")
                time.sleep(POLL_SECONDS)
                continue
            else:
                open_order_time = None

        # ── Refresh macro once per day ─────────────────────────────────────────
        today = now.date()
        if last_macro_date != today:
            print(f"\n  Refreshing macro features...", end=" ")
            current_macro  = get_macro_features()
            last_macro_date = today
            print(f"VIX={current_macro.get('vix_level',0)*20:.1f} "
                  f"SPY_rsi={current_macro.get('spy_rsi',0):.2f}")

        # ── Compute features ───────────────────────────────────────────────────
        window = compute_features(bars, current_macro)
        if window is None:
            print("— not enough data.")
            time.sleep(POLL_SECONDS)
            continue

        # ── Refresh sentiment every 5 minutes ─────────────────────────────────
        if last_news_time is None or (now - last_news_time).seconds >= NEWS_REFRESH_MIN * 60:
            current_sentiment = news.get_current_sentiment()
            last_news_time    = now

        # ── Get signal from CNN directly ───────────────────────────────────────
        direction, probability = get_signal(window)

        if direction is None:
            # Show probabilities so we can see how close to threshold
            x   = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                proba = F.softmax(cnn(x), dim=1)[0]
            print(f"— no signal (p_up={float(proba[1]):.3f} p_down={float(proba[0]):.3f})")
        else:
            shares   = size_position(probability, latest_price)
            order_id = submit_order(direction, shares)

            if order_id:
                open_order_time = now
                record = {
                    "timestamp"      : now.isoformat(),
                    "symbol"         : SYMBOL,
                    "direction"      : direction,
                    "probability"    : round(probability, 4),
                    "price"          : round(latest_price, 4),
                    "position_value" : round(shares * latest_price, 2),
                    "shares"         : round(shares, 4),
                    "sentiment"      : round(current_sentiment, 4),
                    "vix"            : round(current_macro.get("vix_level", 0) * 20, 2),
                    "alpaca_order_id": order_id,
                    "outcome"        : None,
                    "pnl"            : None,
                }
                log_trade(record)
                print(f"— {direction} @ ${latest_price:.2f} | "
                      f"prob={probability:.3f} | "
                      f"shares={shares:.2f} | "
                      f"vix={current_macro.get('vix_level',0)*20:.1f}  ← LIVE ORDER")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        if get_open_position():
            print("Closing open position...")
            close_position()
        print("Bot stopped.")