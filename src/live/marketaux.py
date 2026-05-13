"""
Live news fetcher from Marketaux
----------------------------------
Fetches recent AAPL news and caches it to avoid hitting
the daily API limit. GPT scores each article once and
caches the result.

Usage:
    from src.live.marketaux import NewsFetcher
    fetcher = NewsFetcher()
    score = fetcher.get_current_sentiment()
"""

import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ET       = ZoneInfo("America/New_York")
CACHE    = {}   # in-memory cache: url → {"score": float, "timestamp": datetime}
DECAY    = 0.05 # recency decay lambda — matches training


def _gpt_score(headline: str, summary: str) -> float:
    """
    Ask GPT to score the short-term price impact of a news article.
    Returns a float from -1.0 (very bearish) to +1.0 (very bullish).
    Cached per article URL so we never score the same article twice.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        f"News about Apple Inc (AAPL):\n"
        f"Headline: {headline}\n"
        f"Summary: {summary}\n\n"
        f"Rate the likely SHORT-TERM (next 15-60 minutes) price impact "
        f"on AAPL stock. Respond with ONLY a single float between "
        f"-1.0 (very bearish) and 1.0 (very bullish). No explanation."
    )
    try:
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",   # cheap and fast
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 10,
        )
        return float(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  [GPT error] {e}")
        return 0.0   # neutral fallback


class NewsFetcher:
    def __init__(self):
        self.api_key  = os.getenv("MARKETAUX_API_KEY")
        self.base_url = "https://api.marketaux.com/v1/news/all"
        print("NewsFetcher ready.")

    def fetch_recent(self, lookback_hours: int = 4) -> list[dict]:
        """
        Fetch AAPL news from the last `lookback_hours` hours.
        Returns a list of dicts with keys: url, headline, summary, published_at
        """
        params = {
            "symbols"        : "AAPL",
            "filter_entities": "true",
            "language"       : "en",
            "api_token"      : self.api_key,
        }
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("data", [])
        except Exception as e:
            print(f"  [Marketaux error] {e}")
            return []

        now     = datetime.now(ET)
        cutoff  = now - timedelta(hours=lookback_hours)
        results = []

        for a in articles:
            try:
                pub = datetime.fromisoformat(
                    a["published_at"].replace("Z", "+00:00")
                ).astimezone(ET)
            except Exception:
                continue

            if pub < cutoff:
                continue

            results.append({
                "url"          : a.get("url", ""),
                "headline"     : a.get("title", ""),
                "summary"      : a.get("description", ""),
                "published_at" : pub,
            })

        return results

    def get_current_sentiment(self) -> float:
        """
        Main method called by the bot every minute.
        1. Fetches recent articles
        2. Scores uncached articles with GPT
        3. Applies recency decay
        4. Returns a single weighted sentiment score (-1 to 1)
        """
        articles = self.fetch_recent(lookback_hours=4)

        if not articles:
            return 0.0   # neutral when no news

        now     = datetime.now(ET)
        scores  = []
        weights = []

        for article in articles:
            url = article["url"]

            # Score with GPT if not cached
            if url not in CACHE:
                score = _gpt_score(article["headline"], article["summary"])
                CACHE[url] = {
                    "score"     : score,
                    "published" : article["published_at"],
                }
                print(f"  [GPT scored] {article['headline'][:60]}... → {score:.2f}")

            cached = CACHE[url]
            age_minutes = (now - cached["published"]).total_seconds() / 60
            weight = np.exp(-DECAY * age_minutes)

            scores.append(cached["score"])
            weights.append(weight)

        if sum(weights) == 0:
            return 0.0

        weighted_score = np.dot(scores, weights) / sum(weights)
        return float(np.clip(weighted_score, -1.0, 1.0))


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = NewsFetcher()
    print("\nFetching current AAPL sentiment...")
    score = fetcher.get_current_sentiment()
    print(f"\nCurrent sentiment score: {score:.4f}")
    print("(-1.0 = very bearish, 0.0 = neutral, +1.0 = very bullish)")
