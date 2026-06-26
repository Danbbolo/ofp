"""
run_research_futures.py — Run the sweep on futures data (1 day smoke test).

Wraps run_research.py with a different RAW_DIR and output filename.
Patches the _prepare_liq function to handle futures liq data format
(event_time, quantity, side=BUY/SELL).
"""
import sys
from pathlib import Path
import pandas as pd
import run_research


def _prepare_liq_futures(df: pd.DataFrame) -> pd.DataFrame:
    """Patched for futures liq format: event_time, quantity, side=BUY/SELL.

    Convention in feature_extractor:
        side="SELL" → long liquidation (taker sold, forced the long out)
        side="BUY"  → short liquidation (taker bought, forced the short out)

    cryptohftdata convention:
        order side "BUY"  = taker BUY  = short liquidation
        order side "SELL" = taker SELL = long liquidation

    So NO MAPPING is needed — pass through.
    """
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = pd.DataFrame()
    out["timestamp_ms"] = df["event_time"].astype("int64")
    out["price"] = df["price"].astype(float)
    out["size"] = df["quantity"].astype(float)
    out["side"] = df["side"].str.lower()  # BUY -> buy, SELL -> sell
    return out[["timestamp_ms", "side", "price", "size"]]


# Override config and helper BEFORE main() runs
run_research.RAW_DIR = Path("data/raw_futures")
run_research.OUTPUT_FILE = Path("data/research_dataset_futures.parquet")
run_research._prepare_liq = _prepare_liq_futures  # monkey-patch

if __name__ == "__main__":
    # Parse date range from argv or default to single-day from the futures dir
    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        # Find the only date in raw_futures/
        dates = sorted(p.name for p in Path("data/raw_futures").iterdir() if p.is_dir())
        if not dates:
            print("No dates in data/raw_futures/")
            sys.exit(1)
        start = end = dates[0]
        print(f"Using single day: {start}")

    run_research.main(start, end)
