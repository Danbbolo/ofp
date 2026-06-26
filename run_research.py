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

WINDOW_SIZES_SEC = [60, 120, 180]
# Horizons chosen for ride-style trading: 30m, 1h, 2h, 4h.
# 5/15 min were too short to capture "the move" — see audit notes.
HORIZONS_SEC = [1800, 3600, 7200, 14400]
PROGRESS_EVERY = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    # Drop zero/negative-price trades (liquidation prints, bad data)
    out = out[out["price"] > 0].reset_index(drop=True)
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

    Uses **running book state** between seconds — at every second boundary
    the current top-20 is snapshotted, then deltas in the new second are
    applied on top.  This is O(n) total applies, and most importantly
    the snapshot reflects the cumulative book state at that second, not
    a rebuild from just that second's events.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    recon = OrderBookReconstructor()
    snapshots: dict[int, tuple[list, list]] = {}
    total_rows = 0
    current_sec: int = -1  # last second we have applied at least one delta of

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "book.parquet"
        if not fpath.exists():
            print(f"    {date_str}: no book file, skipping", flush=True)
            d += timedelta(days=1)
            continue

        import pyarrow.parquet as pq
        pf = pq.ParquetFile(fpath)
        day_rows = 0
        # Reset the running book at day boundary — book state doesn't persist
        # across days because the trading day starts with a fresh order book.
        recon.clear()
        current_sec = -1

        for batch in pf.iter_batches(batch_size=200_000):
            ev = batch.column("event_time").to_numpy(zero_copy_only=False)
            sd = batch.column("side").to_pylist()
            px = batch.column("price").to_pylist()
            qt = batch.column("quantity").to_pylist()
            m = len(ev)

            for i in range(m):
                sec = int(ev[i]) // 1000

                if sec != current_sec and current_sec >= 0:
                    # Crossed into a new second — snapshot the state at the
                    # END of the previous second (cumulative book).
                    key = current_sec * 1000
                    if key not in snapshots:
                        # Evict stale levels before snapshotting.  This
                        # prevents ghost levels (bids/asks whose placer
                        # never explicitly cancelled) from producing
                        # crossed-book states when the market moves away.
                        # See docs/orderbook_data_audit.md for details.
                        recon.evict_stale(current_time_ms=key,
                                           max_age_ms=30_000)
                        snapshots[key] = recon.top_n(20)

                # Apply this delta to the running book.
                recon.apply(side=sd[i], price=float(px[i]),
                             quantity=float(qt[i]),
                             timestamp_ms=int(ev[i]))
                current_sec = sec
            day_rows += m

        # End-of-day: snapshot the final second too.
        if current_sec >= 0:
            key = current_sec * 1000
            if key not in snapshots:
                recon.evict_stale(current_time_ms=key, max_age_ms=30_000)
                snapshots[key] = recon.top_n(20)

        total_rows += day_rows
        print(f"    {date_str}: {day_rows:,} rows → {len(snapshots):,} snapshots",
              flush=True)
        # Free per-day memory (per spec section 7)
        gc.collect()
        d += timedelta(days=1)

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

    # --- Compute per-zoom rolling baselines (fixes context leak) ---
    # For each zoom, the baseline is the average volume in a window of that
    # zoom's size.  Computed from the entire dataset so it is a stable
    # reference.  Each zoom MUST get its own baseline — using a single
    # global value means the macro feature is just a scaled version of
    # the micro feature, which leaks context.
    data_duration_ms = int(trades_df["timestamp_ms"].iloc[-1]) - int(trades_df["timestamp_ms"].iloc[0])
    total_volume = float(trades_df["size"].sum())
    if data_duration_ms <= 0:
        print("ERROR: data duration is non-positive.")
        sys.exit(1)

    def _avg_volume_per_window(window_ms: int) -> float:
        n_windows = max(data_duration_ms // window_ms, 1)
        return total_volume / n_windows

    MESO_MS = 300_000
    MACRO_MS = 1_800_000
    micro_window_ms = WINDOW_SIZES_SEC[0] * 1000

    rolling_stats_per_zoom = {
        "micro": {"rolling_avg_volume": _avg_volume_per_window(micro_window_ms)},
        "meso":  {"rolling_avg_volume": _avg_volume_per_window(MESO_MS)},
        "macro": {"rolling_avg_volume": _avg_volume_per_window(MACRO_MS)},
    }
    print(f"  Per-zoom baselines:")
    for z, rs in rolling_stats_per_zoom.items():
        print(f"    {z:>5}: rolling_avg_volume = {rs['rolling_avg_volume']:.4f}")
    print("  Sweeping …", flush=True)

    # --- Sweep ---
    sweeper = GridSweeper(window_sizes_sec=WINDOW_SIZES_SEC,
                          horizons_sec=HORIZONS_SEC)
    gen = sweeper.sweep(
        trades_df=trades_df,
        book_snapshots=book_snapshots,
        liq_df=liq_df,
        rolling_stats_per_zoom=rolling_stats_per_zoom,
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
