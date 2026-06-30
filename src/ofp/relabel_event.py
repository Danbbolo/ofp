"""
relabel_event.py — Event-based relabeling for volume bars (Task 6).

Binary, direction-agnostic target: does ANY 1% move (up OR down) occur
within a forward window from bar close?

  - Label 1 (Event):    max_abs_move >= move_bps within horizon
  - Label 0 (No Event): max_abs_move <  move_bps for full horizon

where max_abs_move = max(|max_price - bar_close|, |min_price - bar_close|)
                    / bar_close * 10_000  (in bps)

Every bar gets a label (no "no-trade" exclusion).  We predict volatility
ignition, not direction.

Usage:
    python -m src.ofp.relabel_event 2026-06-17 2026-06-23
    python -m src.ofp.relabel_event 2026-06-17 --threshold 50 --horizon 7200 --move_bps 100
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_EVENT = 1      # a >=move_bps move occurred within horizon
LABEL_NO_EVENT = 0   # no such move

DEFAULT_MOVE_BPS = 100          # 1%
DEFAULT_HORIZON_S = 7200        # 2 hours
MAX_HORIZON_MS = DEFAULT_HORIZON_S * 1000

RAW_DIR = Path("data/raw_futures")

# ---------------------------------------------------------------------------
# Core labeling function (pure, testable)
# ---------------------------------------------------------------------------

def compute_event_label(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    bar_close_ms: int,
    bar_close_px: float,
    move_bps: int = DEFAULT_MOVE_BPS,
    max_horizon_ms: int = MAX_HORIZON_MS,
) -> tuple[int, float]:
    """
    Compute event label for a single bar.

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
    move_bps : int
        Move threshold in basis points (default 100 = 1%).
    max_horizon_ms : int
        Forward window in ms (default 2h).

    Returns
    -------
    (label, max_abs_move_bps)
        label: 1 (event), 0 (no event)
        max_abs_move_bps: largest absolute excursion in bps
    """
    if bar_close_px <= 0:
        return LABEL_NO_EVENT, 0.0

    # First trade at or after bar close
    entry_idx = int(np.searchsorted(trade_ts, bar_close_ms, side="left"))
    if entry_idx >= len(trade_ts):
        return LABEL_NO_EVENT, 0.0

    end_ms = bar_close_ms + max_horizon_ms
    end_idx = int(np.searchsorted(trade_ts, end_ms, side="right"))

    if end_idx <= entry_idx:
        return LABEL_NO_EVENT, 0.0

    # Forward prices (entry+1 .. end)
    seg_px = trade_px[entry_idx + 1:end_idx]
    if len(seg_px) == 0:
        return LABEL_NO_EVENT, 0.0

    # Direction-agnostic: largest absolute excursion
    max_abs_move_bps = float(
        max(abs(seg_px.max() - bar_close_px), abs(seg_px.min() - bar_close_px))
        / bar_close_px * 10_000
    )

    if max_abs_move_bps >= move_bps:
        return LABEL_EVENT, max_abs_move_bps
    return LABEL_NO_EVENT, max_abs_move_bps


# ---------------------------------------------------------------------------
# Data loading (handles cross-day forward windows)
# ---------------------------------------------------------------------------

def _load_trades_for_range(
    start_ms: int, end_ms: int, raw_dir: Path = RAW_DIR
) -> pl.DataFrame:
    """Load and concatenate trades covering [start_ms, end_ms]."""
    start = datetime.utcfromtimestamp(start_ms / 1000)
    end = datetime.utcfromtimestamp(end_ms / 1000)
    chunks = []
    d = start
    while d.date() <= end.date():
        date_str = d.strftime("%Y-%m-%d")
        fpath = raw_dir / date_str / "trades.parquet"
        if fpath.exists():
            df = pl.read_parquet(str(fpath))
            if "trade_time" in df.columns:
                df = df.rename({"trade_time": "timestamp_ms", "quantity": "size"})
            df = df.with_columns([
                pl.col("timestamp_ms").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("size").cast(pl.Float64),
            ])
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

def relabel_bars_event(
    bars: pl.DataFrame,
    raw_dir: Path = RAW_DIR,
    move_bps: int = DEFAULT_MOVE_BPS,
    max_horizon_ms: int = MAX_HORIZON_MS,
) -> pl.DataFrame:
    """
    Relabel volume bars with event-based binary target.

    Returns bars with added columns: label, max_abs_move_bps.
    """
    bar_ts_min = int(bars["timestamp_ms"].min())
    bar_ts_max = int(bars["timestamp_ms"].max())
    forward_end_ms = bar_ts_max + max_horizon_ms + 60_000

    print(
        f"Loading forward trades from {raw_dir} covering "
        f"ms [{bar_ts_min} .. {forward_end_ms}] …",
        flush=True,
    )
    trades = _load_trades_for_range(bar_ts_min, forward_end_ms, raw_dir)
    print(f"  {len(trades):,} trade rows loaded", flush=True)

    if len(trades) == 0:
        print("ERROR: no trades found for forward window")
        sys.exit(1)

    trade_ts = trades["timestamp_ms"].to_numpy()
    trade_px = trades["price"].to_numpy()

    bar_close_ts = bars["timestamp_ms"].to_numpy()
    bar_close_px = bars["close"].to_numpy()
    n = len(bars)

    print(f"Computing event labels for {n:,} bars …", flush=True)

    labels = np.zeros(n, dtype=np.int64)
    max_abs_moves = np.zeros(n, dtype=np.float64)

    t0 = datetime.now()
    for i in range(n):
        label, mam = compute_event_label(
            trade_ts, trade_px,
            int(bar_close_ts[i]), float(bar_close_px[i]),
            move_bps, max_horizon_ms,
        )
        labels[i] = label
        max_abs_moves[i] = mam

        if (i + 1) % 1000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print(
                f"  {i + 1:,}/{n:,}  "
                f"({elapsed:.0f}s, {i / max(elapsed, 1):.0f} bars/s)",
                flush=True,
            )

    bars = bars.with_columns([
        pl.Series("label", labels),
        pl.Series("max_abs_move_bps", max_abs_moves),
    ])
    return bars


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    start_date = None
    end_date = None
    threshold = 50.0
    horizon_s = DEFAULT_HORIZON_S
    move_bps = DEFAULT_MOVE_BPS
    raw_dir = RAW_DIR

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

    if "--horizon" in args:
        idx = args.index("--horizon")
        if idx + 1 < len(args):
            horizon_s = int(args[idx + 1])

    if "--move_bps" in args:
        idx = args.index("--move_bps")
        if idx + 1 < len(args):
            move_bps = int(args[idx + 1])

    if start_date is None:
        avail = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
        if not avail:
            print(f"No dates in {raw_dir}")
            sys.exit(1)
        start_date = avail[0]
        end_date = avail[-1]
    elif end_date is None:
        end_date = start_date

    horizon_ms = horizon_s * 1000
    output_file = Path(
        f"data/research_dataset_v2_event_{horizon_s}s_{move_bps}bps.parquet"
    )

    print(f"=== EVENT RELABEL ===")
    print(f"  Date range: {start_date} -> {end_date}")
    print(f"  Volume threshold: {threshold} BTC")
    print(f"  Move threshold: +/-{move_bps} bps (+/-{move_bps/100:.2f}%)")
    print(f"  Horizon: {horizon_s/3600:.1f}h ({horizon_s}s)")
    print(f"  Raw dir: {raw_dir}")
    print(f"  Output: {output_file}")
    print()

    from ofp.volume_clock import build_volume_bars

    # Build volume bars for all dates in range
    all_bars = []
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    while d.date() <= end_dt.date():
        date_str = d.strftime("%Y-%m-%d")
        trades_path = raw_dir / date_str / "trades.parquet"
        if trades_path.exists():
            print(f"  Building bars for {date_str} ...", flush=True)
            bars = build_volume_bars(trades_path, volume_threshold=threshold)
            print(f"    {len(bars):,} bars", flush=True)
            all_bars.append(bars)
        else:
            print(f"  {date_str}: no trades file, skipping")
        d += timedelta(days=1)

    if not all_bars:
        print("ERROR: no bars generated")
        sys.exit(1)

    bars = pl.concat(all_bars, how="vertical")
    print(f"\n  Total bars: {len(bars):,}")

    # Relabel
    bars = relabel_bars_event(
        bars, raw_dir=raw_dir, move_bps=move_bps, max_horizon_ms=horizon_ms
    )

    # Distribution
    labels = bars["label"].to_numpy()
    n = len(bars)
    n_event = int((labels == LABEL_EVENT).sum())
    n_no_event = int((labels == LABEL_NO_EVENT).sum())
    base_rate = n_event / n if n > 0 else 0.0

    print(f"\n=== LABEL DISTRIBUTION ===")
    print(f"  Label 1 (Event):    {n_event:>8,}  ({n_event/n*100:.2f}%)")
    print(f"  Label 0 (No Event): {n_no_event:>8,}  ({n_no_event/n*100:.2f}%)")
    print(f"  Base rate (P(event)): {base_rate:.4f} ({base_rate*100:.2f}%)")

    # Save
    bars.write_parquet(str(output_file))
    print(f"\n  Saved to {output_file}")


if __name__ == "__main__":
    main()