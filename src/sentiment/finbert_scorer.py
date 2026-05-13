"""
FinBERT sentiment scorer for GDELT AAPL news data
--------------------------------------------------
Loads each yearly GDELT CSV from data/raw/sentiment/,
scores each article using FinBERT, and saves the results
to data/sentiment/AAPL_sentiment_YYYY.csv

FinBERT outputs: positive, negative, neutral probabilities
We convert to a single score: positive - negative (-1 to +1)

The output is one row per article with:
    date, sentiment_score, weight (placeholder, set to 1.0)

The sentiment merge script will later apply recency decay
when joining with the minute price bars.

Usage:
    python src/sentiment/finbert_scorer.py

Requirements:
    pip install transformers torch
    (model downloads automatically on first run, ~500MB)
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/sentiment")
OUTPUT_DIR = Path("data/sentiment")
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LENGTH = 128   # FinBERT max token length

# Keywords to filter irrelevant GDELT rows before scoring
APPLE_KEYWORDS = [
    "AAPL", "APPLE INC", "IPHONE", "TIM COOK",
    "APP STORE", "MACBOOK", "APPLE WATCH"
]

# ── Device ─────────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ── Load FinBERT ───────────────────────────────────────────────────────────────
print(f"\nLoading FinBERT ({MODEL_NAME})...")
print("  (downloading ~500MB on first run, cached after that)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("  FinBERT loaded.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_themes(themes_str: str) -> str:
    """
    Convert GDELT themes string to readable text for FinBERT.
    e.g. 'APPLE_INC;ECON_STOCKMARKET;IPHONE' → 'Apple Inc stock market iPhone'
    """
    if not isinstance(themes_str, str):
        return ""

    replacements = {
        "APPLE_INC"       : "Apple Inc",
        "AAPL"            : "AAPL stock",
        "IPHONE"          : "iPhone",
        "TIM_COOK"        : "Tim Cook",
        "APP_STORE"       : "App Store",
        "ECON_STOCKMARKET": "stock market",
        "ECON_BANKRUPTCY" : "bankruptcy",
        "BUS_MARKET"      : "business market",
        "TAX_FNCACT_CEO"  : "CEO",
        "LEGISLATION"     : "legislation",
        "REGULATION"      : "regulation",
        "COURT_CASE"      : "court case",
    }

    tokens = themes_str.split(";")
    words  = []
    for token in tokens:
        token = token.strip()
        # Use replacement if available, otherwise clean underscores
        if token in replacements:
            words.append(replacements[token])
        else:
            words.append(token.replace("_", " ").title())

    return " ".join(words) if words else themes_str.replace(";", " ").replace("_", " ")


def is_relevant(themes_str: str) -> bool:
    """Check if a GDELT row is relevant to AAPL."""
    if not isinstance(themes_str, str):
        return False
    themes_upper = themes_str.upper()
    return any(kw in themes_upper for kw in APPLE_KEYWORDS)


def score_batch(texts: list[str]) -> list[float]:
    """
    Score a batch of texts with FinBERT.
    Returns a list of floats: positive_prob - negative_prob
    Range: -1.0 (very negative) to +1.0 (very positive)
    """
    inputs = tokenizer(
        texts,
        padding    = True,
        truncation = True,
        max_length = MAX_LENGTH,
        return_tensors = "pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # FinBERT label order: positive=0, negative=1, neutral=2
    probs = softmax(outputs.logits, dim=1).cpu().numpy()
    scores = probs[:, 0] - probs[:, 1]   # positive - negative
    return scores.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────
def score_year(year: int) -> pd.DataFrame | None:
    input_path  = RAW_DIR / f"gdelt_aapl_{year}.csv"
    output_path = OUTPUT_DIR / f"AAPL_sentiment_{year}.csv"

    if not input_path.exists():
        print(f"  {year}: file not found, skipping")
        return None

    if output_path.exists():
        print(f"  {year}: already scored, skipping")
        return None

    print(f"\n{'='*50}")
    print(f"Scoring {year}...")

    df = pd.read_csv(input_path, dtype=str)
    print(f"  Loaded {len(df):,} rows")

    # Filter to relevant rows
    # Data is already Apple-filtered from download step
    print(f"  Skipping relevance filter (data pre-filtered at download)")

    if df.empty:
        print(f"  No relevant rows found for {year}")
        return None

    # Clean themes to readable text
    df["text"] = df["themes"].apply(clean_themes)

    # Drop rows where text is too short to be meaningful
    df = df[df["text"].str.len() > 10].copy()
    print(f"  After text filter: {len(df):,} rows")

    # Score in batches
    all_scores = []
    total      = len(df)

    for i in range(0, total, BATCH_SIZE):
        batch = df["text"].iloc[i:i+BATCH_SIZE].tolist()
        scores = score_batch(batch)
        all_scores.extend(scores)

        if (i // BATCH_SIZE) % 10 == 0:
            pct = min(i + BATCH_SIZE, total) / total * 100
            print(f"  Progress: {pct:.0f}% ({min(i+BATCH_SIZE, total):,}/{total:,})")

    df["sentiment_score"] = all_scores
    df["weight"]          = 1.0   # recency decay applied later at merge time

    # Keep only what we need
    output_df = df[["date", "sentiment_score", "weight", "themes"]].copy()
    output_df = output_df.reset_index(drop=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"  Saved {len(output_df):,} scored rows to {output_path.name}")

    # Quick stats
    print(f"  Score stats:")
    print(f"    mean  : {output_df['sentiment_score'].mean():.4f}")
    print(f"    std   : {output_df['sentiment_score'].std():.4f}")
    print(f"    pos   : {(output_df['sentiment_score'] > 0.1).mean()*100:.1f}%")
    print(f"    neg   : {(output_df['sentiment_score'] < -0.1).mean()*100:.1f}%")
    print(f"    neut  : {(output_df['sentiment_score'].abs() <= 0.1).mean()*100:.1f}%")

    return output_df


def main():
    years = list(range(2015, 2026))
    print(f"FinBERT scorer — processing {len(years)} years")
    print(f"Input  : {RAW_DIR.resolve()}")
    print(f"Output : {OUTPUT_DIR.resolve()}")

    for year in years:
        score_year(year)

    print(f"\n{'='*50}")
    print("Done! All years scored.")
    print(f"Next step: run src/sentiment/merger.py to join with price data.")


if __name__ == "__main__":
    main()
