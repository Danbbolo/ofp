"""
relabel_target.py — Re-label the research dataset with TARGET-BASED exits.

This replaces the time-based horizon labels with target-based labels that
match the trader's actual riding style.

For each row (entry at window_end_ms):
  - Look forward up to MAX_HORIZON_SEC (4h).
  - Find the FIRST time price hits +1% (target) or -1% (stop).
  - If +1% hit first: outcome_binary = 1 (win), outcome_pct = +1% (capped)
  - If -1% hit first: outcome_binary = 0 (loss), outcome_pct = -1% (capped)
  - If neither hit within MAX_HORIZON: outcome_pct = (final_price - entry) / entry

This matches the trader's mental model: enter on structure, take profit at
+1% or stop at -1%, otherwise trail.

Usage:
    python relabel_target.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Target-based parameters
TARGET_PCT = 0.01       # +1% take-profit
STOP_PCT = 0.01        # -1% stop-loss (symmetric 1:1 RR)
MAX_HORIZON_SEC = 24 * 3600  # 24 hours — if neither hit, mark as time-based exit

# Configurable via command line: python relabel_target.py <raw_dir> <input_file> <output_file>
if len(sys.argv) == 4:
    RAW_DIR = Path(sys.argv[1])
    INPUT_FILE = Path(sys.argv[2])
    OUTPUT_FILE = Path(sys.argv[3])
else:
    RAW_DIR = Path("data/raw")
    INPUT_FILE = Path("data/research_dataset.parquet")
    OUTPUT_FILE = Path("data/research_dataset_target.parquet")


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


def compute_target_outcome(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    entry_ms: int,
    target_pct: float = TARGET_PCT,
    stop_pct: float = STOP_PCT,
    max_horizon_ms: int = MAX_HORIZON_SEC * 1000,
) -> tuple[int, float]:
    """
    Compute target-based outcome: which level (target or stop) is hit first.

    Returns:
      outcome_binary: 1 if target hit first, 0 if stop hit first or neither
      outcome_pct: actual return (capped at target/stop if hit, else time-based)
    """
    # Find entry index
    entry_idx = int(np.searchsorted(trade_ts, entry_ms, side="left"))
    if entry_idx >= len(trade_ts):
        return 0, 0.0
    entry_px = float(trade_px[entry_idx])
    if entry_px <= 0:
        return 0, 0.0

    target_px = entry_px * (1.0 + target_pct)
    stop_px = entry_px * (1.0 - stop_pct)
    end_ms = entry_ms + max_horizon_ms

    # Find end index
    end_idx = int(np.searchsorted(trade_ts, end_ms, side="right"))
    if end_idx <= entry_idx:
        return 0, 0.0

    # Walk through the trades from entry+1 onwards
    # We want the FIRST index where target_px or stop_px is hit
    # For buys (long): target = high, stop = low
    # For simplicity here we treat both directions the same: target = +1%, stop = -1%
    # The trader enters with a direction; but the binary label is "did price go up enough?"
    # For a SHORT entry, the same +1% target would be wrong — we'd want -1% target.
    # However, our dataset has only one outcome_binary per row (not per direction).
    # We'll use the absolute version: target = +1% (in entry direction agnostic)
    # — but to be direction-aware, we'd need to know entry direction.
    # For now: just use the simple target_pct = +1% on the price.
    # (This is conservative — assumes long entry; for short entries the binary
    # would be flipped, but the trader can use the 1% target_pct for both:
    # their exit logic takes profit at +1% in their favor.)
    seg_px = trade_px[entry_idx + 1:end_idx]
    if len(seg_px) == 0:
        return 0, 0.0

    # Find first index where price >= target or <= stop
    target_hit = seg_px >= target_px
    stop_hit = seg_px <= stop_px

    target_first = np.argmax(target_hit) if target_hit.any() else len(seg_px) + 1
    stop_first = np.argmax(stop_hit) if stop_hit.any() else len(seg_px) + 1

    if target_first < stop_first:
        # Target hit first → win
        actual_px = seg_px[target_first]
        return 1, float((actual_px - entry_px) / entry_px)
    elif stop_first < len(seg_px) + 1:
        # Stop hit first → loss
        actual_px = seg_px[stop_first]
        return 0, float((actual_px - entry_px) / entry_px)
    else:
        # Neither hit → time-based exit at max horizon
        final_px = float(seg_px[-1])
        pct = (final_px - entry_px) / entry_px
        return (1 if pct > 0 else 0), float(pct)


def main() -> None:
    print(f"Loading {INPUT_FILE} …", flush=True)
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} cols", flush=True)
    print(f"  Date range: window_end_ms [{df['window_end_ms'].min()} .. {df['window_end_ms'].max()}]", flush=True)
    print()

    start_ms = int(df["window_end_ms"].min())
    end_ms = int(df["window_end_ms"].max()) + MAX_HORIZON_SEC * 1000 + 60_000

    print(f"Loading trades from {RAW_DIR} covering ms [{start_ms} .. {end_ms}] …", flush=True)
    trades = _load_trades_for_range(start_ms, end_ms)
    print(f"  {len(trades):,} trade rows", flush=True)
    if len(trades) == 0:
        print("ERROR: no trades found")
        sys.exit(1)
    print()

    trade_ts = trades["timestamp_ms"].values
    trade_px = trades["price"].values

    # Drop old outcome cols
    drop_cols = [c for c in ("outcome_pct", "outcome_binary", "target_hit_1pct", "mae_pct") if c in df.columns]
    df = df.drop(columns=drop_cols)

    we = df["window_end_ms"].values
    n = len(df)
    print(f"Computing target-based outcomes for {n:,} rows …", flush=True)

    out_binary = np.zeros(n, dtype=np.int64)
    out_pct = np.zeros(n, dtype=np.float64)
    out_target_hit = np.zeros(n, dtype=np.int64)
    out_stop_hit = np.zeros(n, dtype=np.int64)
    out_time_based = np.zeros(n, dtype=np.int64)
    out_hold_sec = np.zeros(n, dtype=np.float64)

    t0 = datetime.now()
    for i in range(n):
        b, p, th, sh, tb, hs = compute_target_outcome_extended(
            trade_ts, trade_px, int(we[i]),
            TARGET_PCT, STOP_PCT, MAX_HORIZON_SEC * 1000,
        )
        out_binary[i] = b
        out_pct[i] = p
        out_target_hit[i] = th
        out_stop_hit[i] = sh
        out_time_based[i] = tb
        out_hold_sec[i] = hs
        if (i + 1) % 10000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  {i + 1:,}/{n:,}  ({elapsed:.0f}s, {i / max(elapsed, 1):.0f} rows/s)", flush=True)

    df["outcome_binary"] = out_binary
    df["outcome_pct"] = out_pct
    df["target_hit"] = out_target_hit
    df["stop_hit"] = out_stop_hit
    df["time_based_exit"] = out_time_based
    df["hold_sec"] = out_hold_sec

    print()
    print(f"=== TARGET-BASED OUTCOMES ===")
    print(f"Target hit (+1%):   {out_target_hit.sum():,} ({out_target_hit.mean()*100:.2f}%)")
    print(f"Stop hit (-1%):    {out_stop_hit.sum():,} ({out_stop_hit.mean()*100:.2f}%)")
    print(f"Time-based exit:   {out_time_based.sum():,} ({out_time_based.mean()*100:.2f}%)")
    print(f"Win rate:          {out_binary.mean()*100:.2f}%")
    print(f"Mean return:       {out_pct.mean()*100:+.4f}%")
    print(f"Median hold (s):   {np.median(out_hold_sec):.0f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print()
    print(f"Saved {len(df):,} rows to {OUTPUT_FILE} ({size_mb:.2f} MB)")


def compute_target_outcome_extended(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    entry_ms: int,
    target_pct: float,
    stop_pct: float,
    max_horizon_ms: int,
) -> tuple[int, float, int, int, int, float]:
    """
    Returns: (outcome_binary, outcome_pct, target_hit, stop_hit, time_based, hold_sec)
    """
    entry_idx = int(np.searchsorted(trade_ts, entry_ms, side="left"))
    if entry_idx >= len(trade_ts):
        return 0, 0.0, 0, 0, 1, 0.0
    entry_px = float(trade_px[entry_idx])
    if entry_px <= 0:
        return 0, 0.0, 0, 0, 1, 0.0

    target_px = entry_px * (1.0 + target_pct)
    stop_px = entry_px * (1.0 - stop_pct)
    end_ms = entry_ms + max_horizon_ms

    end_idx = int(np.searchsorted(trade_ts, end_ms, side="right"))
    if end_idx <= entry_idx:
        return 0, 0.0, 0, 0, 1, 0.0

    seg_px = trade_px[entry_idx + 1:end_idx]
    if len(seg_px) == 0:
        return 0, 0.0, 0, 0, 1, 0.0

    target_hit_mask = seg_px >= target_px
    stop_hit_mask = seg_px <= stop_px

    target_first = np.argmax(target_hit_mask) if target_hit_mask.any() else len(seg_px) + 1
    stop_first = np.argmax(stop_hit_mask) if stop_hit_mask.any() else len(seg_px) + 1

    if target_first < stop_first and target_first < len(seg_px):
        actual_px = seg_px[target_first]
        return 1, float((actual_px - entry_px) / entry_px), 1, 0, 0, (trade_ts[entry_idx + 1 + target_first] - entry_ms) / 1000
    elif stop_first < len(seg_px) + 1:
        actual_px = seg_px[stop_first]
        return 0, float((actual_px - entry_px) / entry_px), 0, 1, 0, (trade_ts[entry_idx + 1 + stop_first] - entry_ms) / 1000
    else:
        final_px = float(seg_px[-1])
        pct = (final_px - entry_px) / entry_px
        return (1 if pct > 0 else 0), float(pct), 0, 0, 1, max_horizon_ms / 1000


if __name__ == "__main__":
    main()
