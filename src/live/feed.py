"""
Live data feed from Alpaca
---------------------------
Fetches real-time and historical AAPL minute bars.
Used by the bot to build the 60-bar feature window.

Usage:
    from src.live.feed import AlpacaFeed
    feed = AlpacaFeed()
    bars = feed.get_latest_bars("AAPL", limit=60)
"""

import os
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

ET = ZoneInfo("America/New_York")

class AlpacaFeed:
    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key    = os.getenv("APCA_API_KEY_ID"),
            secret_key = os.getenv("APCA_API_SECRET_KEY"),
        )
        print("AlpacaFeed connected.")

    def get_latest_bars(self, symbol: str, limit: int = 60) -> pd.DataFrame:
        """
        Fetch the last `limit` completed minute bars for a symbol.
        Returns a DataFrame with columns: Date, Open, High, Low, Close,
        Volume, Transactions — matching our training data format.
        """
        request = StockBarsRequest(
            symbol_or_symbols = symbol,
            timeframe         = TimeFrame.Minute,
            limit             = limit,
        )
        bars = self.client.get_stock_bars(request).df

        if bars.empty:
            return pd.DataFrame()

        # Alpaca returns a multi-index (symbol, timestamp) — drop symbol level
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")

        bars = bars.reset_index()
        bars = bars.rename(columns={
            "timestamp"     : "Date",
            "open"          : "Open",
            "high"          : "High",
            "low"           : "Low",
            "close"         : "Close",
            "volume"        : "Volume",
            "trade_count"   : "Transactions",
        })

        # Keep only the columns our pipeline expects
        bars = bars[["Date", "Open", "High", "Low", "Close", "Volume", "Transactions"]]

        # Convert to ET and strip timezone for consistency with training data
        bars["Date"] = pd.to_datetime(bars["Date"]).dt.tz_convert(ET).dt.tz_localize(None)
        bars = bars.sort_values("Date").reset_index(drop=True)

        return bars

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        from alpaca.trading.client import TradingClient
        trading = TradingClient(
            os.getenv("APCA_API_KEY_ID"),
            os.getenv("APCA_API_SECRET_KEY"),
            paper=True
        )
        clock = trading.get_clock()
        return clock.is_open


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feed = AlpacaFeed()

    print("\nFetching last 5 AAPL bars...")
    bars = feed.get_latest_bars("AAPL", limit=5)
    print(bars)

    print(f"\nMarket open: {feed.is_market_open()}")
