"""
relabel_dataset.py — Re-label research_dataset.parquet with the new ride-style horizons.

GLM 5.2 plan:
1. Horizons: 30m, 1h, 2h, 4h (drop 5/15 min).
2. For each row, look up the future price at each new horizon and compute outcome_pct / outcome_binary.
3. Add target_hit_1pct: 1 if max price during next 4h reaches +1% from entry.
4. Explode to 4 rows per entry (one per new horizon) and save.

We DO NOT re-sweep (saves 90 min) — we only re-label the existing 110,679 rows.

Usage:
    python relabel_dataset.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
INPUT_FILE = Path("data/research_dataset.parquet")
OUTPUT_FILE = Path("data/research_dataset_relabel.parquet")

# New horizons for ride-style trading
NEW_HORIZONS_SEC = [1800, 3600, 7200, 14400]
MAX_HORIZON_SEC = max(NEW_HORIZONS_SEC)  # 4h
TARGET_PCT = 0.01  # 1% target_hit threshold


def _load_trades_for_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load and concatenate trades from data/raw/ for the date range covering [start_ms, end_ms]."""
    start = datetime.utcfromtimestamp(start_ms / 1000)
    end = datetime.utcfromtimestamp(end_ms / 1000)
    chunks = []
    d = start
    while d.date() <= end.date():
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "trades.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
            df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
            df["price"] = df["price"].astype(float)
            df["size"] = df["size"].astype(float)
            chunks.append(df[["timestamp_ms", "price"]])
        d += timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["timestamp_ms", "price"])
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def main() -> None:
    print(f"Loading {INPUT_FILE} …", flush=True)
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} cols", flush=True)
    print(f"  Date range: window_end_ms [{df['window_end_ms'].min()} .. {df['window_end_ms'].max()}]", flush=True)
    print(f"  Existing horizons: {sorted(df['horizon'].unique().tolist())}", flush=True)
    print(f"  Existing window_sizes: {sorted(df['window_size'].unique().tolist())}", flush=True)
    print()

    # Determine date range from window_end_ms — need trades up to max(end) + MAX_HORIZON + slack
    start_ms = int(df["window_end_ms"].min())
    end_ms = int(df["window_end_ms"].max()) + MAX_HORIZON_SEC * 1000 + 60_000  # +1 min slack

    print(f"Loading trades from {RAW_DIR} covering ms [{start_ms} .. {end_ms}] …", flush=True)
    trades = _load_trades_for_range(start_ms, end_ms)
    print(f"  {len(trades):,} trade rows", flush=True)
    if len(trades) == 0:
        print("ERROR: no trades found")
        sys.exit(1)
    print()

    trade_ts = trades["timestamp_ms"].values
    trade_px = trades["price"].values

    # Drop the OLD outcome columns
    drop_cols = [c for c in ("outcome_pct", "outcome_binary") if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop existing per-horizon outcome columns if any (defensive)
    for col in list(df.columns):
        if col.startswith("outcome_pct_") or col.startswith("outcome_binary_"):
            df = df.drop(columns=[col])

    we = df["window_end_ms"].values  # entry times
    n = len(df)
    print(f"Computing outcomes for {len(NEW_HORIZONS_SEC)} new horizons + target_hit_1pct …", flush=True)

    # Add the per-horizon outcome columns
    for hz in NEW_HORIZONS_SEC:
        df[f"outcome_pct_{hz}"] = 0.0
        df[f"outcome_binary_{hz}"] = 0.0

    out_target = np.zeros(n, dtype=np.float64)
    out_mae = np.zeros(n, dtype=np.float64)

    # Sort by entry time so the searchsorted bounds are amortized
    order = np.argsort(we, kind="stable")
    sorted_we = we[order]
    lo_idx_arr = np.searchsorted(trade_ts, sorted_we, side="left")
    hi_idx_arr = np.searchsorted(trade_ts, sorted_we + MAX_HORIZON_SEC * 1000, side="right")

    pct_col_idx = {hz: df.columns.get_loc(f"outcome_pct_{hz}") for hz in NEW_HORIZONS_SEC}
    bin_col_idx = {hz: df.columns.get_loc(f"outcome_binary_{hz}") for hz in NEW_HORIZONS_SEC}

    for i in range(n):
        lo, hi = int(lo_idx_arr[i]), int(hi_idx_arr[i])
        if hi <= lo:
            continue
        seg = trade_px[lo:hi]
        entry_idx = int(np.searchsorted(trade_ts, sorted_we[i], side="left"))
        entry_idx = min(entry_idx, len(trade_ts) - 1)
        entry_px = trade_px[entry_idx]
        if entry_px <= 0:
            continue
        max_px = seg.max()
        min_px = seg.min()

        out_target[order[i]] = 1.0 if (max_px / entry_px - 1.0) >= TARGET_PCT else 0.0
        out_mae[order[i]] = min_px / entry_px - 1.0

        for hz in NEW_HORIZONS_SEC:
            fut_ts = sorted_we[i] + hz * 1000
            fut_idx = int(np.searchsorted(trade_ts, fut_ts, side="left"))
            fut_idx = min(fut_idx, len(trade_ts) - 1)
            fut_px = trade_px[fut_idx]
            op = (fut_px - entry_px) / entry_px
            df.iat[order[i], pct_col_idx[hz]] = op
            df.iat[order[i], bin_col_idx[hz]] = 1.0 if op > 0 else 0.0

    df["target_hit_1pct"] = out_target
    df["mae_pct"] = out_mae

    print(f"  target_hit_1pct: {int(out_target.sum()):,} hits / {n:,} "
          f"({out_target.mean():.4f} hit rate)")
    print(f"  mae_pct: mean={out_mae.mean():.6f}, "
          f"q25={np.quantile(out_mae, 0.25):.6f}, "
          f"q50={np.quantile(out_mae, 0.5):.6f}")
    for hz in NEW_HORIZONS_SEC:
        wr = df[f"outcome_binary_{hz}"].mean()
        print(f"  horizon {hz}s ({hz // 60} min): win rate = {wr:.4f}")

    # Explode — 4 rows per entry
    print()
    print(f"Exploding to {len(NEW_HORIZONS_SEC)} horizons per entry …", flush=True)
    out_pieces = []
    for hz in NEW_HORIZONS_SEC:
        chunk = df.copy()
        chunk["horizon"] = hz
        chunk["outcome_pct"] = chunk[f"outcome_pct_{hz}"].values
        chunk["outcome_binary"] = chunk[f"outcome_binary_{hz}"].values
        chunk = chunk.drop(columns=[f"outcome_pct_{hz}", f"outcome_binary_{hz}"])
        out_pieces.append(chunk)
    exploded = pd.concat(out_pieces, ignore_index=True)
    # Drop any per-horizon outcome columns
    exploded = exploded.drop(columns=[c for c in exploded.columns
                                      if c.startswith("outcome_pct_") or c.startswith("outcome_binary_")])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    exploded.to_parquet(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print()
    print(f"Saved {len(exploded):,} rows to {OUTPUT_FILE} ({size_mb:.2f} MB)")
    print(f"  Final horizons: {sorted(exploded['horizon'].unique().tolist())}")


if __name__ == "__main__":
    main()
