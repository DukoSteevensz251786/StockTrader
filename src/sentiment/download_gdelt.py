"""
GDELT GKG downloader for AAPL sentiment data
---------------------------------------------
- Downloads daily GKG files from data.gdeltproject.org
- Filters rows mentioning Apple/AAPL
- Saves one CSV per year to data/raw/sentiment/
- Skips weekends and already-completed days
- Safe to stop and restart — resumes where it left off

Usage:
    python src/sentiment/download_gdelt.py
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
START_DATE   = date(2015, 1, 1)
END_DATE     = date(2025, 2, 1)
OUTPUT_DIR   = Path("data/raw/sentiment")
KEYWORDS     = ["AAPL", "APPLE INC", "IPHONE", "TIM COOK", "APP STORE"]
SLEEP_SEC    = 1.0      # be polite, don't hammer the server
MAX_RETRIES  = 3

# GKG column names (tab-separated, no header in file)
# Full spec: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook.pdf
GKG_COLUMNS = [
    "date", "numarts", "counts", "themes", "locations",
    "persons", "organizations", "tone", "cameoeventids",
    "sources", "sourceurls"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research project, non-commercial)"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def date_range(start: date, end: date):
    """Yield every calendar date from start to end, skipping weekends."""
    current = start
    while current < end:
        if current.weekday() < 5:   # 0=Mon, 4=Fri
            yield current
        current += timedelta(days=1)


def download_day(d: date) -> pd.DataFrame | None:
    """Download, unzip, and return the GKG dataframe for a single day."""
    datestr = d.strftime("%Y%m%d")
    url = f"http://data.gdeltproject.org/gkg/{datestr}.gkg.csv.zip"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                return None     # some days simply don't exist
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [!] Failed after {MAX_RETRIES} attempts: {e}")
                return None
            time.sleep(2 ** attempt)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            try:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=GKG_COLUMNS,
                    encoding="latin-1",
                    on_bad_lines="skip",
                    dtype=str,
                )
            except Exception as e:
                print(f"  [!] Parse error: {e}")
                return None

    return df


def filter_apple(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where themes or organizations mention Apple."""
    mask = pd.Series(False, index=df.index)
    for col in ["themes", "organizations", "sourceurls"]:
        if col in df.columns:
            mask |= df[col].str.upper().str.contains(
                "|".join(KEYWORDS), na=False
            )
    return df[mask][["date", "sources", "sourceurls", "themes", "tone"]].copy()


def parse_tone(tone_str: str) -> float | None:
    """
    Tone is a comma-separated string: overall, positive, negative, polarity, ...
    We want the first value (overall tone score).
    Negative = bearish, positive = bullish.
    """
    try:
        return float(str(tone_str).split(",")[0])
    except (ValueError, AttributeError):
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track progress: one line per completed date in a log file
    log_path = OUTPUT_DIR / "completed_dates.txt"
    completed = set()
    if log_path.exists():
        completed = set(log_path.read_text().splitlines())

    # Accumulate rows per year
    yearly_buffers: dict[int, list[pd.DataFrame]] = {}

    all_dates = list(date_range(START_DATE, END_DATE))
    total = len(all_dates)

    print(f"GDELT AAPL downloader — {total} trading days to process")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    for i, d in enumerate(all_dates, 1):
        datestr = d.strftime("%Y%m%d")

        if datestr in completed:
            print(f"[{i}/{total}] {datestr} — skipped (already done)")
            continue

        print(f"[{i}/{total}] {datestr} ", end="", flush=True)

        df = download_day(d)
        if df is None or df.empty:
            print("— no data")
        else:
            filtered = filter_apple(df)
            if filtered.empty:
                print(f"— 0 Apple rows")
            else:
                # Parse tone to a float
                filtered["tone_score"] = filtered["tone"].apply(parse_tone)
                filtered = filtered.drop(columns=["tone"])
                filtered["date"] = datestr

                year = d.year
                yearly_buffers.setdefault(year, []).append(filtered)
                print(f"— {len(filtered)} rows")

        # Mark as done
        with open(log_path, "a") as log:
            log.write(datestr + "\n")

        # Flush to disk every 30 days to avoid losing everything on crash
        if i % 30 == 0:
            flush_buffers(yearly_buffers, OUTPUT_DIR)
            yearly_buffers.clear()

        time.sleep(SLEEP_SEC)

    # Final flush
    flush_buffers(yearly_buffers, OUTPUT_DIR)
    print("\nDone! Files saved to", OUTPUT_DIR.resolve())


def flush_buffers(buffers: dict, output_dir: Path):
    """Append accumulated rows to per-year CSV files."""
    for year, frames in buffers.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = output_dir / f"gdelt_aapl_{year}.csv"
        write_header = not out_path.exists()
        combined.to_csv(out_path, mode="a", header=write_header, index=False)
        print(f"  → flushed {len(combined)} rows to {out_path.name}")


if __name__ == "__main__":
    main()