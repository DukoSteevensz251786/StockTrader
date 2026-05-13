"""
Sentiment merger — joins GDELT tone scores with minute price bars
-----------------------------------------------------------------
Uses GDELT's built-in tone_score instead of FinBERT scoring.
GDELT tone is computed from actual article text, more reliable
than scoring theme tags with FinBERT.

Tone score range: roughly -10 (very negative) to +10 (very positive)
We normalise to -1 to +1 to match the live GPT scoring range.

For each trading day, computes one sentiment score by averaging
all tone scores for that date. Then joins on date so every
minute bar gets the sentiment score for its trading day.

Missing days (no news) default to 0.0 (neutral).

Output: data/processed/AAPL_labeled_sentiment.csv

Usage:
    python src/sentiment/merger.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DIR      = Path("data/raw/sentiment")
LABELED_PATH = Path("data/processed/AAPL_labeled.csv")
OUTPUT_PATH  = Path("data/processed/AAPL_labeled_sentiment.csv")
OUTPUT_DIR   = Path("data/sentiment")
YEARS        = list(range(2015, 2026))

# GDELT tone is roughly -10 to +10, clip and normalise to -1 to +1
TONE_CLIP = 10.0

# ── Load all yearly GDELT files ────────────────────────────────────────────────
def load_all_gdelt() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"gdelt_aapl_{year}.csv"
        if not path.exists():
            print(f"  {year}: not found, skipping")
            continue
        df = pd.read_csv(path, dtype={"date": str})
        df["year"] = year
        frames.append(df)
        print(f"  {year}: {len(df):,} rows")

    if not frames:
        print("No GDELT files found in data/raw/sentiment/")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ── Compute daily sentiment ────────────────────────────────────────────────────
def compute_daily_scores(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tone scores per day.
    Normalises GDELT tone (-10 to +10) to (-1 to +1).
    """
    # Drop rows with missing tone
    gdelt_df = gdelt_df.dropna(subset=["tone_score"]).copy()
    gdelt_df["tone_score"] = pd.to_numeric(gdelt_df["tone_score"], errors="coerce")
    gdelt_df = gdelt_df.dropna(subset=["tone_score"])

    # Normalise to -1 to +1
    gdelt_df["sentiment_score"] = (
        gdelt_df["tone_score"].clip(-TONE_CLIP, TONE_CLIP) / TONE_CLIP
    )

    # Parse date — GDELT format is YYYYMMDD
    gdelt_df["date_parsed"] = pd.to_datetime(
        gdelt_df["date"].astype(str).str[:8],
        format="%Y%m%d",
        errors="coerce"
    )
    gdelt_df = gdelt_df.dropna(subset=["date_parsed"])

    # Daily average
    daily = (
        gdelt_df
        .groupby("date_parsed")["sentiment_score"]
        .agg(
            sentiment_score = "mean",
            article_count   = "count",
        )
        .reset_index()
        .rename(columns={"date_parsed": "date"})
    )

    print(f"\nDaily sentiment computed:")
    print(f"  Days with news   : {len(daily):,}")
    print(f"  Avg articles/day : {daily['article_count'].mean():.1f}")
    print(f"  Score mean       : {daily['sentiment_score'].mean():.4f}")
    print(f"  Score std        : {daily['sentiment_score'].std():.4f}")
    print(f"  Positive days    : {(daily['sentiment_score'] > 0.05).mean()*100:.1f}%")
    print(f"  Negative days    : {(daily['sentiment_score'] < -0.05).mean()*100:.1f}%")
    print(f"  Neutral days     : {(daily['sentiment_score'].abs() <= 0.05).mean()*100:.1f}%")

    return daily


# ── Merge with price bars ──────────────────────────────────────────────────────
def merge_with_prices(daily_scores: pd.DataFrame) -> pd.DataFrame:
    print(f"\nLoading labeled price data...")
    price_df = pd.read_csv(LABELED_PATH, parse_dates=["Date"])
    print(f"  {len(price_df):,} bars loaded")

    # Extract date only from timestamp for joining
    price_df["date"] = price_df["Date"].dt.normalize()

    # Left join
    merged = price_df.merge(
        daily_scores[["date", "sentiment_score", "article_count"]],
        on  = "date",
        how = "left",
    )

    # Fill missing days with 0.0 neutral
    missing = merged["sentiment_score"].isna().sum()
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged["article_count"]   = merged["article_count"].fillna(0).astype(int)
    merged = merged.drop(columns=["date"])

    print(f"  Bars with sentiment : {(merged['sentiment_score'] != 0).sum():,}")
    print(f"  Bars without news   : {missing:,} (set to 0.0 neutral)")

    return merged


# ── Save daily scores separately too ──────────────────────────────────────────
def save_daily_scores(daily_scores: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "AAPL_daily_sentiment.csv"
    daily_scores.to_csv(path, index=False)
    print(f"\nDaily scores saved to {path.resolve()}")


# ── Sanity check ───────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame):
    print(f"\nSanity check:")
    print(f"  Shape            : {df.shape}")
    print(f"  Null check       : {df.isnull().sum().sum()} total nulls")
    print(f"  Sentiment range  : [{df['sentiment_score'].min():.4f}, "
          f"{df['sentiment_score'].max():.4f}]")
    print(f"\nSample labeled rows:")
    sample = df[df["label"] != -1][
        ["Date", "Close", "label", "sentiment_score", "article_count"]
    ].head(5)
    print(sample.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading GDELT files...")
    gdelt_df = load_all_gdelt()
    print(f"  Total rows: {len(gdelt_df):,}")

    daily_scores = compute_daily_scores(gdelt_df)
    save_daily_scores(daily_scores)

    merged_df = merge_with_prices(daily_scores)
    sanity_check(merged_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH.resolve()}")
    print(f"\nNext steps:")
    print(f"  1. python src/training/train.py")
    print(f"  2. python src/models/pipeline.py")


if __name__ == "__main__":
    main()