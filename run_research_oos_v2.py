"""
run_research_oos_v2.py — OOS sweep using the proven run_research.py pipeline.

Monkey-patches run_research.py to point at raw_futures_oos/ and use
smaller sweep params (W=60 only, H=1800/3600).
"""
import sys
from pathlib import Path
import pandas as pd
import run_research


def _prepare_liq_futures(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = pd.DataFrame()
    out["timestamp_ms"] = df["event_time"].astype("int64")
    out["price"] = df["price"].astype(float)
    out["size"] = df["quantity"].astype(float)
    out["side"] = df["side"].str.lower()
    return out[["timestamp_ms", "side", "price", "size"]]


# Override config BEFORE main() runs
run_research.RAW_DIR = Path("data/raw_futures_oos")
run_research.OUTPUT_FILE = Path("data/research_dataset_oos.parquet")
run_research._prepare_liq = _prepare_liq_futures

# Smaller sweep for OOS
run_research.WINDOW_SIZES_SEC = [60]
run_research.HORIZONS_SEC = [1800, 3600]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        dates = sorted(p.name for p in Path("data/raw_futures_oos").iterdir() if p.is_dir())
        if not dates:
            print("No dates in data/raw_futures_oos/")
            sys.exit(1)
        start, end = dates[0], dates[-1]
        print(f"Using date range: {start} → {end}")

    run_research.main(start, end)
