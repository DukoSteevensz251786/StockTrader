"""
Live data feed using yfinance
------------------------------
Fetches real-time AAPL minute bars via Yahoo Finance.
More reliable than Alpaca IEX free tier for prototyping.

Usage:
    from src.live.feed import AlpacaFeed
    feed = AlpacaFeed()
    bars = feed.get_latest_bars("AAPL", limit=100)
"""

import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import yfinance as yf
from alpaca.trading.client import TradingClient

load_dotenv()

ET = ZoneInfo("America/New_York")


class AlpacaFeed:
    def __init__(self):
        # Trading client only used for market open check
        self.trading_client = TradingClient(
            os.getenv("APCA_API_KEY_ID"),
            os.getenv("APCA_API_SECRET_KEY"),
            paper=True
        )
        print("AlpacaFeed connected (yfinance data, Alpaca execution).")

    def get_latest_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch the last `limit` minute bars using yfinance.
        Returns a DataFrame with columns matching our training format.
        """
        ticker = yf.Ticker(symbol)
        bars   = ticker.history(period="1d", interval="1m")

        if bars.empty:
            return pd.DataFrame()

        # Reset index — Datetime becomes a column
        bars = bars.reset_index()
        bars = bars.rename(columns={
            "Datetime"     : "Date",
            "Open"         : "Open",
            "High"         : "High",
            "Low"          : "Low",
            "Close"        : "Close",
            "Volume"       : "Volume",
        })

        # Add Transactions column (not in yfinance — use 0)
        bars["Transactions"] = 0

        # Keep only needed columns
        bars = bars[["Date", "Open", "High", "Low", "Close", "Volume", "Transactions"]]

        # Convert timezone to ET and strip for consistency with training data
        bars["Date"] = pd.to_datetime(bars["Date"])
        if bars["Date"].dt.tz is not None:
            bars["Date"] = bars["Date"].dt.tz_convert(ET).dt.tz_localize(None)

        bars = bars.sort_values("Date").reset_index(drop=True)

        # Drop last bar if volume is 0 — incomplete bar
        if len(bars) > 1 and bars["Volume"].iloc[-1] == 0:
            bars = bars.iloc[:-1]

        return bars.tail(limit).reset_index(drop=True)

    def is_market_open(self) -> bool:
        """Check if US market is currently open via Alpaca."""
        clock = self.trading_client.get_clock()
        return clock.is_open


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feed = AlpacaFeed()

    print("\nFetching last 5 AAPL bars...")
    bars = feed.get_latest_bars("AAPL", limit=5)
    print(bars[["Date", "Open", "Close", "Volume"]])

    print(f"\nLatest price : ${bars['Close'].iloc[-1]:.2f}")
    print(f"Latest bar   : {bars['Date'].iloc[-1]}")
    print(f"Market open  : {feed.is_market_open()}")