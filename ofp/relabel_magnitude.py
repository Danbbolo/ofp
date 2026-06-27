"""
relabel_magnitude.py — Magnitude-based relabeling for volume bars.

For each volume bar, looks forward 24 hours from bar close (timestamp_ms):
  - Label 1 (Target hit): +1% (100 bps) reached before -1% (-100 bps)
  - Label 0 (Stop hit):   -1% (-100 bps) reached before +1% (100 bps)
  - Label 2 (No Trade):   neither hits within 24h — exclude from training

The label is determined by which threshold is hit FIRST in time.
Tiebreaker: if both hit in the same trade tick, price above bar_close →
target first (label 1), price below → stop first (label 0).

Usage:
    python -m ofp.relabel_magnitude 2026-06-17 --threshold 50
    python -m ofp.relabel_magnitude 2026-06-17 2026-06-23 --threshold 50
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Label constants
LABEL_TARGET = 1   # +1% hit first
LABEL_STOP = 0     # -1% hit first
LABEL_NO_TRADE = 2 # neither hit within 24h

# Default parameters
TARGET_BPS = 100      # +1% = 100 basis points
STOP_BPS = 100        # -1% = 100 basis points
MAX_HORIZON_MS = 24 * 3600 * 1000  # 24 hours in ms

RAW_DIR = Path("data/raw_futures")
OUTPUT_FILE = Path("data/research_dataset_v2_magnitude.parquet")


# ---------------------------------------------------------------------------
# Core labeling function (pure, testable)
# ---------------------------------------------------------------------------

def compute_magnitude_label(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    bar_close_ms: int,
    bar_close_px: float,
    target_bps: int = TARGET_BPS,
    stop_bps: int = STOP_BPS,
    max_horizon_ms: int = MAX_HORIZON_MS,
) -> tuple[int, float, float]:
    """
    Compute magnitude label for a single bar.

    Parameters
    ----------
    trade_ts : np.ndarray[int64]
        Sorted trade timestamps in ms.
    trade_px : np.ndarray[float]
        Trade prices aligned to trade_ts.
    bar_close_ms : int
        Bar close timestamp (entry point).
    bar_close_px : float
        Bar close price (entry price).
    target_bps : int
        Target threshold in basis points (default 100 = +1%).
    stop_bps : int
        Stop threshold in basis points (default 100 = -1%).
    max_horizon_ms : int
        Maximum forward window in ms (default 24h).

    Returns
    -------
    (label, max_return_bps, max_drawdown_bps)
        label: 1 (target hit), 0 (stop hit), 2 (no trade)
        max_return_bps: highest favorable excursion in bps
        max_drawdown_bps: lowest adverse excursion in bps (negative)
    """
    if bar_close_px <= 0:
        return LABEL_NO_TRADE, 0.0, 0.0

    # Find entry index: first trade at or after bar_close_ms
    entry_idx = int(np.searchsorted(trade_ts, bar_close_ms, side="left"))
    if entry_idx >= len(trade_ts):
        return LABEL_NO_TRADE, 0.0, 0.0

    entry_px = bar_close_px
    target_px = entry_px * (1.0 + target_bps / 10_000.0)
    stop_px = entry_px * (1.0 - stop_bps / 10_000.0)

    end_ms = bar_close_ms + max_horizon_ms
    end_idx = int(np.searchsorted(trade_ts, end_ms, side="right"))

    if end_idx <= entry_idx:
        return LABEL_NO_TRADE, 0.0, 0.0

    # Segment of forward prices (entry+1 to end, exclusive of entry itself)
    seg_px = trade_px[entry_idx + 1:end_idx]
    if len(seg_px) == 0:
        return LABEL_NO_TRADE, 0.0, 0.0

    # Compute max return and max drawdown in bps
    max_return_bps = float((seg_px.max() - entry_px) / entry_px * 10_000)
    max_drawdown_bps = float((seg_px.min() - entry_px) / entry_px * 10_000)

    # Find first index where target or stop is hit
    target_hit_mask = seg_px >= target_px
    stop_hit_mask = seg_px <= stop_px

    target_first = int(np.argmax(target_hit_mask)) if target_hit_mask.any() else len(seg_px) + 1
    stop_first = int(np.argmax(stop_hit_mask)) if stop_hit_mask.any() else len(seg_px) + 1

    if target_first < stop_first and target_first < len(seg_px):
        # Target hit first
        return LABEL_TARGET, max_return_bps, max_drawdown_bps
    elif stop_first < target_first and stop_first < len(seg_px):
        # Stop hit first
        return LABEL_STOP, max_return_bps, max_drawdown_bps
    elif target_first == stop_first and target_first < len(seg_px):
        # Both hit at same index — tiebreaker: check price vs entry
        hit_px = float(seg_px[target_first])
        if hit_px >= entry_px:
            return LABEL_TARGET, max_return_bps, max_drawdown_bps
        else:
            return LABEL_STOP, max_return_bps, max_drawdown_bps
    else:
        # Neither hit within horizon
        return LABEL_NO_TRADE, max_return_bps, max_drawdown_bps


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_trades_for_range(start_ms: int, end_ms: int, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """
    Load and concatenate trades from raw_dir for the date range covering [start_ms, end_ms].
    Handles cross-day loading.
    """
    start = datetime.utcfromtimestamp(start_ms / 1000)
    end = datetime.utcfromtimestamp(end_ms / 1000)
    chunks = []
    d = start
    while d.date() <= end.date():
        date_str = d.strftime("%Y-%m-%d")
        fpath = raw_dir / date_str / "trades.parquet"
        if fpath.exists():
            df = pl.read_parquet(str(fpath))
            # Normalize column names
            if "trade_time" in df.columns:
                df = df.rename({"trade_time": "timestamp_ms", "quantity": "size"})
            # Cast types — price and size come as strings
            df = df.with_columns([
                pl.col("timestamp_ms").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("size").cast(pl.Float64),
            ])
            # Drop zero/negative-price trades
            df = df.filter(pl.col("price") > 0)
            chunks.append(df.select(["timestamp_ms", "price"]))
        d += timedelta(days=1)

    if not chunks:
        return pl.DataFrame({
            "timestamp_ms": pl.Series([], dtype=pl.Int64),
            "price": pl.Series([], dtype=pl.Float64),
        })

    result = pl.concat(chunks, how="vertical")
    result = result.sort("timestamp_ms")
    return result


# ---------------------------------------------------------------------------
# Main relabeling pipeline
# ---------------------------------------------------------------------------

def relabel_bars(
    bars: pl.DataFrame,
    raw_dir: Path = RAW_DIR,
    target_bps: int = TARGET_BPS,
    stop_bps: int = STOP_BPS,
    max_horizon_ms: int = MAX_HORIZON_MS,
) -> pl.DataFrame:
    """
    Relabel volume bars with magnitude-based targets.

    Parameters
    ----------
    bars : pl.DataFrame
        Volume bars from volume_clock (must have timestamp_ms, close).
    raw_dir : Path
        Directory containing raw futures data (data/raw_futures/).
    target_bps, stop_bps, max_horizon_ms : int
        Label parameters.

    Returns
    -------
    pl.DataFrame
        Bars with added columns: label, max_return_bps, max_drawdown_bps.
    """
    # Determine forward data range needed
    bar_ts_min = int(bars["timestamp_ms"].min())
    bar_ts_max = int(bars["timestamp_ms"].max())
    forward_end_ms = bar_ts_max + max_horizon_ms + 60_000  # 1min buffer

    print(f"Loading forward trades from {raw_dir} covering ms [{bar_ts_min} .. {forward_end_ms}] …", flush=True)
    trades = _load_trades_for_range(bar_ts_min, forward_end_ms, raw_dir)
    print(f"  {len(trades):,} trade rows loaded", flush=True)

    if len(trades) == 0:
        print("ERROR: no trades found for forward window")
        sys.exit(1)

    trade_ts = trades["timestamp_ms"].to_numpy()
    trade_px = trades["price"].to_numpy()

    # Extract bar data to numpy for fast iteration
    bar_close_ts = bars["timestamp_ms"].to_numpy()
    bar_close_px = bars["close"].to_numpy()
    n = len(bars)

    print(f"Computing magnitude labels for {n:,} bars …", flush=True)

    labels = np.zeros(n, dtype=np.int64)
    max_returns = np.zeros(n, dtype=np.float64)
    max_drawdowns = np.zeros(n, dtype=np.float64)

    t0 = datetime.now()
    for i in range(n):
        label, mr, md = compute_magnitude_label(
            trade_ts, trade_px,
            int(bar_close_ts[i]), float(bar_close_px[i]),
            target_bps, stop_bps, max_horizon_ms,
        )
        labels[i] = label
        max_returns[i] = mr
        max_drawdowns[i] = md

        if (i + 1) % 1000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  {i + 1:,}/{n:,}  ({elapsed:.0f}s, {i / max(elapsed, 1):.0f} bars/s)", flush=True)

    # Add columns to bars DataFrame
    bars = bars.with_columns([
        pl.Series("label", labels),
        pl.Series("max_return_bps", max_returns),
        pl.Series("max_drawdown_bps", max_drawdowns),
    ])

    return bars


def main() -> None:
    """CLI entry: build volume bars, relabel, save, print distribution."""
    # Parse args
    # Usage: python -m ofp.relabel_magnitude [start_date] [end_date] [--threshold N]
    start_date = None
    end_date = None
    threshold = 50.0

    args = sys.argv[1:]
    dates = [a for a in args if not a.startswith("--")]
    if len(dates) >= 1:
        start_date = dates[0]
    if len(dates) >= 2:
        end_date = dates[1]

    if "--threshold" in args:
        idx = args.index("--threshold")
        if idx + 1 < len(args):
            threshold = float(args[idx + 1])

    # Default to all dates in raw_futures if not specified
    if start_date is None:
        dates_avail = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
        if not dates_avail:
            print(f"No dates in {RAW_DIR}")
            sys.exit(1)
        start_date = dates_avail[0]
        end_date = dates_avail[-1]
    elif end_date is None:
        end_date = start_date

    print(f"=== MAGNITUDE RELABEL ===")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  Volume threshold: {threshold} BTC")
    print(f"  Target: +{TARGET_BPS} bps (+1%)")
    print(f"  Stop:   -{STOP_BPS} bps (-1%)")
    print(f"  Horizon: {MAX_HORIZON_MS / 3600000:.0f}h")
    print()

    # Build volume bars for the date range
    from ofp.volume_clock import build_volume_bars

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_bars = []
    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        trades_path = RAW_DIR / date_str / "trades.parquet"
        if trades_path.exists():
            print(f"Building volume bars for {date_str} …", flush=True)
            bars = build_volume_bars(trades_path, volume_threshold=threshold)
            print(f"  {len(bars)} bars", flush=True)
            all_bars.append(bars)
        else:
            print(f"  {date_str}: no trades file, skipping")
        d += timedelta(days=1)

    if not all_bars:
        print("ERROR: no volume bars generated")
        sys.exit(1)

    bars = pl.concat(all_bars, how="vertical")
    print(f"\nTotal bars: {len(bars):,}")
    print()

    # Relabel
    bars = relabel_bars(bars, raw_dir=RAW_DIR)

    # Print label distribution
    print()
    print(f"=== LABEL DISTRIBUTION ===")
    n = len(bars)
    labels = bars["label"].to_numpy()

    n_target = int((labels == LABEL_TARGET).sum())
    n_stop = int((labels == LABEL_STOP).sum())
    n_no_trade = int((labels == LABEL_NO_TRADE).sum())

    print(f"  Label 1 (Target hit):  {n_target:>8,}  ({n_target/n*100:.2f}%)")
    print(f"  Label 0 (Stop hit):    {n_stop:>8,}  ({n_stop/n*100:.2f}%)")
    print(f"  Label 2 (No Trade):    {n_no_trade:>8,}  ({n_no_trade/n*100:.2f}%)")
    print()

    tradeable = n_target + n_stop
    if tradeable > 0:
        base_rate = n_target / tradeable
        print(f"  Base rate (Label 1 / (Label 0 + Label 1)): {base_rate:.4f} ({base_rate*100:.2f}%)")
    else:
        print(f"  Base rate: N/A (no tradeable bars)")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(str(OUTPUT_FILE))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print()
    print(f"Saved {len(bars):,} rows to {OUTPUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()