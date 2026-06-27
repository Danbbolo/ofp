"""
relabel_hadmove.py — Re-label the research dataset with HAD_MOVE (V9 methodology).

For each row (entry at window_end_ms):
  - Look forward up to 1 hour (3600s).
  - If price reaches +0.5% OR -0.5% at ANY point within 1h → had_move = 1
  - If neither threshold hit within 1h → had_move = 0

This is a volatility/move label: "did a 0.5% move happen in either direction?"

Outcome columns:
  - outcome_binary = had_move (1 if 0.5% move happened, 0 otherwise)
  - outcome_pct = actual return at exit point (+0.5% if target hit, -0.5% if stop hit,
                  endpoint return at 1h if neither)
  - had_move = same as outcome_binary
  - move_direction = +1 if up, -1 if down, 0 if flat
  - move_pct = same as outcome_pct

Usage:
    python relabel_hadmove.py <raw_dir> <input_file> <output_file>

Examples:
    python relabel_hadmove.py data/raw_futures data/research_dataset_futures.parquet data/research_dataset_futures_hadmove.parquet
    python relabel_hadmove.py data/raw_futures_oos data/research_dataset_oos.parquet data/research_dataset_oos_hadmove.parquet
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Had-move parameters (V9 methodology)
MOVE_PCT = 0.005        # 0.5% threshold (either direction)
MAX_HORIZON_SEC = 3600  # 1 hour

# Configurable via command line
if len(sys.argv) == 4:
    RAW_DIR = Path(sys.argv[1])
    INPUT_FILE = Path(sys.argv[2])
    OUTPUT_FILE = Path(sys.argv[3])
else:
    RAW_DIR = Path("data/raw_futures")
    INPUT_FILE = Path("data/research_dataset_futures.parquet")
    OUTPUT_FILE = Path("data/research_dataset_futures_hadmove.parquet")


def _load_trades_for_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load and concatenate trades from RAW_DIR for the date range covering [start_ms, end_ms]."""
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
            # Drop zero/negative-price trades
            df = df[df["price"] > 0].reset_index(drop=True)
            chunks.append(df[["timestamp_ms", "price"]])
        d += timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["timestamp_ms", "price"])
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def compute_had_move(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    entry_ms: int,
    move_pct: float = MOVE_PCT,
    max_horizon_ms: int = MAX_HORIZON_SEC * 1000,
) -> tuple[int, float, int]:
    """
    Compute had_move outcome: did price reach +/- move_pct within max_horizon?

    Returns:
      had_move: 1 if +move_pct or -move_pct hit within horizon, 0 otherwise
      outcome_pct: actual return at exit (+move_pct if up hit, -move_pct if down hit,
                   endpoint return at 1h if neither)
      move_direction: +1 if up, -1 if down, 0 if flat
    """
    entry_idx = int(np.searchsorted(trade_ts, entry_ms, side="left"))
    if entry_idx >= len(trade_ts):
        return 0, 0.0, 0
    entry_px = float(trade_px[entry_idx])
    if entry_px <= 0:
        return 0, 0.0, 0

    up_px = entry_px * (1.0 + move_pct)
    down_px = entry_px * (1.0 - move_pct)
    end_ms = entry_ms + max_horizon_ms

    end_idx = int(np.searchsorted(trade_ts, end_ms, side="right"))
    if end_idx <= entry_idx:
        return 0, 0.0, 0

    seg_px = trade_px[entry_idx + 1:end_idx]
    if len(seg_px) == 0:
        return 0, 0.0, 0

    up_hit_mask = seg_px >= up_px
    down_hit_mask = seg_px <= down_px

    up_first = np.argmax(up_hit_mask) if up_hit_mask.any() else len(seg_px) + 1
    down_first = np.argmax(down_hit_mask) if down_hit_mask.any() else len(seg_px) + 1

    if up_first < down_first and up_first < len(seg_px):
        # Up move hit first → had_move = 1, return = +move_pct
        return 1, move_pct, 1
    elif down_first < len(seg_px) + 1:
        # Down move hit first → had_move = 1, return = -move_pct
        return 1, -move_pct, -1
    else:
        # Neither hit → had_move = 0, return = endpoint return at 1h
        final_px = float(seg_px[-1])
        pct = (final_px - entry_px) / entry_px
        return 0, float(pct), (1 if pct > 0 else (-1 if pct < 0 else 0))


def main() -> None:
    print(f"=== RELABEL WITH HAD_MOVE (V9 methodology) ===")
    print(f"  Move threshold: +-{MOVE_PCT*100:.1f}%")
    print(f"  Horizon: {MAX_HORIZON_SEC}s ({MAX_HORIZON_SEC/3600:.0f}h)")
    print(f"  Raw dir: {RAW_DIR}")
    print(f"  Input:   {INPUT_FILE}")
    print(f"  Output:  {OUTPUT_FILE}")
    print()

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
    drop_cols = [c for c in ("outcome_pct", "outcome_binary", "had_move",
                             "move_direction", "move_pct", "target_hit",
                             "stop_hit", "time_based_exit", "hold_sec")
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    we = df["window_end_ms"].values
    n = len(df)
    print(f"Computing had_move outcomes for {n:,} rows …", flush=True)

    out_binary = np.zeros(n, dtype=np.int64)
    out_pct = np.zeros(n, dtype=np.float64)
    out_had_move = np.zeros(n, dtype=np.int64)
    out_move_dir = np.zeros(n, dtype=np.int64)
    out_move_pct = np.zeros(n, dtype=np.float64)

    t0 = datetime.now()
    for i in range(n):
        hm, pct, direction = compute_had_move(
            trade_ts, trade_px, int(we[i]), MOVE_PCT, MAX_HORIZON_SEC * 1000,
        )
        out_binary[i] = hm
        out_pct[i] = pct
        out_had_move[i] = hm
        out_move_dir[i] = direction
        out_move_pct[i] = pct
        if (i + 1) % 10000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  {i + 1:,}/{n:,}  ({elapsed:.0f}s, {i / max(elapsed, 1):.0f} rows/s)", flush=True)

    df["outcome_binary"] = out_binary
    df["outcome_pct"] = out_pct
    df["had_move"] = out_had_move
    df["move_direction"] = out_move_dir
    df["move_pct"] = out_move_pct

    print()
    print(f"=== HAD_MOVE OUTCOMES ===")
    print(f"had_move = 1 (0.5% move hit):  {out_had_move.sum():,} ({out_had_move.mean()*100:.2f}%)")
    print(f"had_move = 0 (no move):         {(n - out_had_move.sum()):,} ({(1 - out_had_move.mean())*100:.2f}%)")
    print(f"Win rate (outcome_binary):     {out_binary.mean()*100:.2f}%")
    print(f"Mean return (outcome_pct):     {out_pct.mean()*100:+.4f}%")
    up_moves = (out_move_dir == 1).sum()
    down_moves = (out_move_dir == -1).sum()
    print(f"Up moves:    {up_moves:,} ({up_moves/n*100:.2f}%)")
    print(f"Down moves:  {down_moves:,} ({down_moves/n*100:.2f}%)")

    # Per-horizon base rates
    print()
    print("=== BASE RATE BY HORIZON ===")
    for hz in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == hz]
        print(f"  H={int(hz):>5d}: had_move rate = {sub['had_move'].mean()*100:.2f}%  ({len(sub):,} rows)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print()
    print(f"Saved {len(df):,} rows to {OUTPUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()