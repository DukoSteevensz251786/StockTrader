"""
Outcome tracker — fills in trade results after 15 minutes
----------------------------------------------------------
Reads data/live/trades.csv, finds trades where outcome is
still None and enough time has passed, fetches the actual
price 15 minutes later from Alpaca, and fills in:

    outcome : 'correct' or 'incorrect'
    pnl     : profit/loss in $ based on position size

Run this periodically (e.g. every 15 minutes) alongside the bot,
or manually after a trading session.

Usage:
    python src/live/outcome_tracker.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.live.feed import AlpacaFeed

# ── Config ─────────────────────────────────────────────────────────────────────
TRADES_LOG  = Path("data/live/trades.csv")
LOOKAHEAD   = 15   # minutes — must match training
ET          = ZoneInfo("America/New_York")

# ── Load trades ────────────────────────────────────────────────────────────────
def load_trades() -> pd.DataFrame:
    if not TRADES_LOG.exists():
        print("No trades log found.")
        return pd.DataFrame()

    df = pd.read_csv(TRADES_LOG, parse_dates=["timestamp"])
    df["outcome"]    = df["outcome"].astype(object)
    df["exit_price"] = df["exit_price"].astype(object) if "exit_price" in df.columns else None
    print(f"Loaded {len(df)} trades — {df['outcome'].isna().sum()} pending outcomes")
    return df


def get_price_at(feed, symbol: str, target_time: datetime) -> float | None:
    """
    Fetch the closing price at or just after target_time using yfinance.
    """
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        bars   = ticker.history(period="5d", interval="1m")

        if bars.empty:
            return None

        bars = bars.reset_index()
        bars["Datetime"] = pd.to_datetime(bars["Datetime"])

        # Convert to naive ET
        if bars["Datetime"].dt.tz is not None:
            from zoneinfo import ZoneInfo
            bars["Datetime"] = bars["Datetime"].dt.tz_convert(
                ZoneInfo("America/New_York")
            ).dt.tz_localize(None)

        # Find bar closest to target time
        target_naive = pd.Timestamp(target_time).tz_localize(None)
        bars["diff"] = (bars["Datetime"] - target_naive).abs()
        closest = bars.loc[bars["diff"].idxmin()]
        return float(closest["Close"])

    except Exception as e:
        print(f"  [Error fetching price] {e}")
        return None


def compute_pnl(direction: str, entry_price: float, exit_price: float,
                position_value: float) -> float:
    """
    Compute P&L for a paper trade.
    BUY : profit if price went up
    SELL: profit if price went down
    """
    price_change_pct = (exit_price - entry_price) / entry_price
    if direction == "SELL":
        price_change_pct = -price_change_pct
    return round(position_value * price_change_pct, 2)


def fill_outcomes():
    df = load_trades()
    if df.empty:
        return

    feed    = AlpacaFeed()
    now     = datetime.now(ET)
    updated = 0

    for idx, row in df.iterrows():
        # Skip already resolved trades
        if pd.notna(row["outcome"]):
            continue

        trade_time = pd.Timestamp(row["timestamp"]).tz_localize(ET) \
            if row["timestamp"].tzinfo is None \
            else pd.Timestamp(row["timestamp"]).tz_convert(ET)

        # Only resolve if 15+ minutes have passed
        if now - trade_time < timedelta(minutes=LOOKAHEAD + 1):
            continue

        target_time = trade_time + timedelta(minutes=LOOKAHEAD)
        print(f"  Resolving trade at {trade_time.strftime('%H:%M')} "
              f"→ checking price at {target_time.strftime('%H:%M')}...",
              end=" ", flush=True)

        exit_price = get_price_at(feed, row["symbol"], target_time)
        if exit_price is None:
            print("price not found, skipping.")
            continue

        entry_price    = float(row["price"])
        direction      = row["direction"]
        position_value = float(row["position_value"])

        # Was the prediction correct?
        price_went_up = exit_price > entry_price
        correct = (direction == "BUY" and price_went_up) or \
                  (direction == "SELL" and not price_went_up)

        pnl = compute_pnl(direction, entry_price, exit_price, position_value)

        df.at[idx, "outcome"]    = "correct" if correct else "incorrect"
        df.at[idx, "pnl"]        = pnl
        df.at[idx, "exit_price"] = exit_price

        print(f"{'✓' if correct else '✗'} | "
              f"entry=${entry_price:.2f} exit=${exit_price:.2f} | "
              f"pnl=${pnl:+.2f}")
        updated += 1

    if updated > 0:
        # Add exit_price column if it doesn't exist
        if "exit_price" not in df.columns:
            df["exit_price"] = None

        df.to_csv(TRADES_LOG, index=False)
        print(f"\nUpdated {updated} trades.")
    else:
        print("No new trades to resolve.")

    # ── Session summary ────────────────────────────────────────────────────────
    resolved = df[df["outcome"].notna()]
    if len(resolved) == 0:
        return

    total     = len(resolved)
    correct   = (resolved["outcome"] == "correct").sum()
    total_pnl = resolved["pnl"].sum()
    win_rate  = correct / total * 100

    print(f"\n{'='*40}")
    print(f"Session summary ({total} resolved trades)")
    print(f"  Win rate  : {win_rate:.1f}%  ({correct}/{total})")
    print(f"  Total P&L : ${total_pnl:+,.2f}")
    print(f"  Avg P&L   : ${resolved['pnl'].mean():+.2f} per trade")

    by_direction = resolved.groupby("direction").agg(
        trades  = ("pnl", "count"),
        win_pct = ("outcome", lambda x: (x == "correct").mean() * 100),
        pnl     = ("pnl", "sum"),
    ).round(2)
    print(f"\nBy direction:\n{by_direction}")


if __name__ == "__main__":
    fill_outcomes()
