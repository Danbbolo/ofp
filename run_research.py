"""
run_research.py — Multi-day grid sweep from local parquet files.

Reads raw data from ``data/raw/YYYY-MM-DD/``, builds book snapshots
incrementally across days, sweeps all days in one continuous pass.

Usage::

    python run_research.py 2026-06-01 2026-06-30

Output: ``data/research_dataset.parquet``
"""

from __future__ import annotations

import gc
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ofp.book_reconstructor import OrderBookReconstructor
from ofp.grid_sweeper import GridSweeper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "research_dataset.parquet"

WINDOW_SIZES_SEC = [60, 120, 180, 300, 600]
HORIZONS_SEC = [300, 900, 1800, 3600, 14400]
PROGRESS_EVERY = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]


def _prepare_liq(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = df.rename(columns={"timestamp": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out.get("price", pd.Series([0.0] * len(out))).astype(float)
    out["size"] = out.get("size", pd.Series([0.0] * len(out))).astype(float)
    return out[["timestamp_ms", "side", "price", "size"]]


def _build_book_snapshots_multi(
    start_str: str, end_str: str,
) -> dict[int, tuple[list, list]]:
    """
    Build 1-second book snapshots incrementally across multiple days.
    OrderBookReconstructor state persists across day boundaries.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    recon = OrderBookReconstructor()
    snapshots: dict[int, tuple[list, list]] = {}
    current_bucket_ms = -1
    total_rows = 0

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "book.parquet"
        if not fpath.exists():
            print(f"    {date_str}: no book file, skipping", flush=True)
            d += timedelta(days=1)
            continue

        df = pd.read_parquet(fpath)
        ev = df["event_time"].values.astype("int64")
        tp = df["event_type"].values
        sd = df["side"].values
        px = df["price"].values.astype(float)
        qt = df["quantity"].values.astype(float)
        n = len(ev)

        for i in range(n):
            bucket = int(ev[i]) // 1000
            if bucket != current_bucket_ms and current_bucket_ms != -1:
                snapshots[current_bucket_ms * 1000] = recon.top_n(20)
            current_bucket_ms = bucket
            if tp[i] == "snapshot":
                recon.clear()
            recon.apply(side=str(sd[i]), price=float(px[i]), quantity=float(qt[i]))

        total_rows += n
        print(f"    {date_str}: {n:,} rows → {len(snapshots):,} snapshots",
              flush=True)
        d += timedelta(days=1)

    if current_bucket_ms != -1:
        snapshots[current_bucket_ms * 1000] = recon.top_n(20)

    print(f"  Total: {total_rows:,} book rows → {len(snapshots):,} snapshots",
          flush=True)
    return snapshots


def _load_trades_multi(start_str: str, end_str: str) -> pd.DataFrame:
    """Load and concatenate trades from all days."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "trades.parquet"
        if fpath.exists():
            df = _prepare_trades(pd.read_parquet(fpath))
            chunks.append(df)
        d += timedelta(days=1)

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def _load_liq_multi(start_str: str, end_str: str) -> pd.DataFrame:
    """Load and concatenate liquidations from all days."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "liq.parquet"
        if fpath.exists():
            df = _prepare_liq(pd.read_parquet(fpath))
            if not df.empty:
                chunks.append(df)
        d += timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_str: str, end_str: str) -> None:
    print(f"Multi-day sweep: {start_str} → {end_str}")
    print(f"  Windows: {WINDOW_SIZES_SEC}")
    print(f"  Horizons: {HORIZONS_SEC}")

    # --- Load trades ---
    print("Loading trades …", flush=True)
    trades_df = _load_trades_multi(start_str, end_str)
    print(f"  Trades: {len(trades_df):,} rows")
    if trades_df.empty:
        print("ERROR: No trade data found.")
        sys.exit(1)

    # --- Load liquidations ---
    print("Loading liquidations …", flush=True)
    liq_df = _load_liq_multi(start_str, end_str)
    print(f"  Liquidations: {len(liq_df):,} rows")

    # --- Build book snapshots incrementally ---
    print("Building book snapshots …", flush=True)
    book_snapshots = _build_book_snapshots_multi(start_str, end_str)

    # --- Compute globals ---
    rolling_avg_volume = float(trades_df["size"].sum() / max(len(trades_df), 1))
    _24h_stats = {
        "_24h_avg_range": float(trades_df["price"].max() - trades_df["price"].min()),
        "_24h_low": float(trades_df["price"].min()),
        "_24h_high": float(trades_df["price"].max()),
    }
    print(f"  Rolling avg volume: {rolling_avg_volume:.4f}")
    print(f"  Range: {_24h_stats['_24h_low']:.2f} – {_24h_stats['_24h_high']:.2f}")
    print("  Sweeping …", flush=True)

    # --- Sweep ---
    sweeper = GridSweeper(window_sizes_sec=WINDOW_SIZES_SEC,
                          horizons_sec=HORIZONS_SEC)
    gen = sweeper.sweep(
        trades_df=trades_df,
        book_snapshots=book_snapshots,
        liq_df=liq_df,
        rolling_avg_volume=rolling_avg_volume,
        _24h_stats=_24h_stats,
    )

    def _progress(iterator):
        n = 0
        for row in iterator:
            yield row
            n += 1
            if n % PROGRESS_EVERY == 0:
                print(f"    Processed {n:,} windows …", flush=True)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GridSweeper.save_to_disk(_progress(gen), str(OUTPUT_FILE))

    result = pd.read_parquet(OUTPUT_FILE)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Done.")
    print(f"  File:  {OUTPUT_FILE.resolve()}")
    print(f"  Size:  {size_mb:.2f} MB")
    print(f"  Rows:  {len(result):,}")
    print()
    print(result.head(3))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
