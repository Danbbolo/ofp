"""
feature_extractor_v2.py — Microstructure feature extractor for volume bars.

Extracts 10 features at each volume bar close. No time-based aggregates.
Only microstructure mechanics.

Base Features (per bar):
  1. vpin: Volume-Synchronized Probability of Informed Trading.
           50-bar rolling mean of |buy_vol - sell_vol| / total_vol.
  2. ofi: Order Flow Imbalance. Delta best-bid-size - delta best-ask-size
          from L2 book snapshots at consecutive bar closes.
  3. book_delta: Net change in total depth (top 20 levels) between
                 consecutive bar closes. Sum of bid+ask size changes.
  4. trade_arrival_rate: num_trades / duration_ms (guarded).
  5. liq_volume: Liquidation volume during the bar window.

Interaction Features:
  6. vpin_x_arrival:    vpin * trade_arrival_rate
  7. ofi_x_book_delta:  ofi * book_delta
  8. liq_x_vpin:         liq_volume * vpin
  9. vpin_x_duration:    vpin * duration_ms
 10. liq_x_return:       liq_volume * bar_return

All division is guarded: if denominator == 0, return 0. No NaN, no Inf.

Usage:
    python -m ofp.feature_extractor_v2 2026-06-17 --threshold 50
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ofp.book_reconstructor import OrderBookReconstructor
from ofp.volume_clock import build_volume_bars

VPIN_WINDOW = 50  # bars


# ---------------------------------------------------------------------------
# Book snapshot building
# ---------------------------------------------------------------------------

def _build_book_snapshots_at_bar_closes(
    book_path: str | Path,
    bar_close_ts: list[int],
) -> dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]]]]:
    """
    Replay book events and snapshot the top-20 book at each bar close.

    Walks through book deltas in event-time order.  When an event's
    timestamp exceeds the next bar-close timestamp, the current book
    state (all events up to that bar close applied) is snapshotted.

    Returns
    -------
    dict[int, (bids, asks)]
        Mapping bar_close_ms -> top-20 (price, size) lists.
    """
    recon = OrderBookReconstructor()
    snapshots: dict[int, tuple[list, list]] = {}

    target_ts = np.array(sorted(bar_close_ts), dtype=np.int64)
    snap_idx = 0
    n_targets = len(target_ts)

    pf = pq.ParquetFile(str(book_path))
    total_rows = pf.metadata.num_rows
    processed = 0

    for batch in pf.iter_batches(batch_size=200_000):
        ev = batch.column("event_time").to_numpy(zero_copy_only=False)
        sd = batch.column("side").to_pylist()
        px = batch.column("price").to_pylist()
        qt = batch.column("quantity").to_pylist()
        m = len(ev)

        for i in range(m):
            ts = int(ev[i])

            # Snapshot any bar closes that fall before this event.
            # At this point all events with event_time < ts have been
            # applied, which includes all events <= target_ts[snap_idx].
            while snap_idx < n_targets and int(target_ts[snap_idx]) < ts:
                recon.evict_stale(
                    current_time_ms=int(target_ts[snap_idx]),
                    max_age_ms=30_000,
                )
                snapshots[int(target_ts[snap_idx])] = recon.top_n(20)
                snap_idx += 1

            recon.apply(
                side=sd[i],
                price=float(px[i]),
                quantity=float(qt[i]),
                timestamp_ms=ts,
            )

        processed += m
        if processed % 10_000_000 < 200_000:
            print(
                f"    book replay: {processed:,}/{total_rows:,} "
                f"({processed / max(total_rows, 1) * 100:.1f}%), "
                f"{snap_idx}/{n_targets} snapshots",
                flush=True,
            )

    # Snapshot any remaining bar closes (past the last book event)
    while snap_idx < n_targets:
        recon.evict_stale(
            current_time_ms=int(target_ts[snap_idx]),
            max_age_ms=30_000,
        )
        snapshots[int(target_ts[snap_idx])] = recon.top_n(20)
        snap_idx += 1

    return snapshots


# ---------------------------------------------------------------------------
# OFI and book_delta
# ---------------------------------------------------------------------------

def _compute_ofi_and_book_delta(
    snapshots: dict[int, tuple[list, list]],
    bar_close_ts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OFI and book_delta from consecutive book snapshots.

    OFI = delta(best_bid_size) - delta(best_ask_size)
    book_delta = delta(total_bid_depth_top20) + delta(total_ask_depth_top20)

    First bar: both = 0 (no previous snapshot).
    """
    n = len(bar_close_ts)
    ofi = np.zeros(n, dtype=np.float64)
    book_delta = np.zeros(n, dtype=np.float64)

    prev_bid_size = 0.0
    prev_ask_size = 0.0
    prev_total_bid = 0.0
    prev_total_ask = 0.0

    for i, ts in enumerate(bar_close_ts):
        snap = snapshots.get(ts)
        if snap is None:
            # No snapshot for this bar — keep zeros, don't update prev
            continue

        bids, asks = snap

        # Best bid/ask size (bids sorted desc, asks sorted asc)
        curr_bid_size = float(bids[0][1]) if len(bids) > 0 else 0.0
        curr_ask_size = float(asks[0][1]) if len(asks) > 0 else 0.0

        # Total depth across top 20
        curr_total_bid = float(sum(s for _, s in bids))
        curr_total_ask = float(sum(s for _, s in asks))

        if i > 0:
            ofi[i] = (curr_bid_size - prev_bid_size) - (
                curr_ask_size - prev_ask_size
            )
            book_delta[i] = (curr_total_bid - prev_total_bid) + (
                curr_total_ask - prev_total_ask
            )

        prev_bid_size = curr_bid_size
        prev_ask_size = curr_ask_size
        prev_total_bid = curr_total_bid
        prev_total_ask = curr_total_ask

    return ofi, book_delta


# ---------------------------------------------------------------------------
# Liquidation volume
# ---------------------------------------------------------------------------

def _compute_liq_volume(
    liq_path: str | Path,
    bar_close_ts: np.ndarray,
    bar_start_ts: np.ndarray,
) -> np.ndarray:
    """
    Compute liquidation volume per bar.

    Sums liq quantity where event_time falls within [bar_start, bar_close].
    Uses searchsorted on sorted liq timestamps for O(n log m) efficiency.
    """
    n = len(bar_close_ts)
    liq_vol = np.zeros(n, dtype=np.float64)

    liq_path = Path(liq_path)
    if not liq_path.exists():
        return liq_vol

    liq = pl.read_parquet(str(liq_path))
    if len(liq) == 0:
        return liq_vol

    # Use event_time as the timestamp (trade_time is close but event_time
    # is the actual liquidation event time)
    ts_col = "event_time" if "event_time" in liq.columns else "trade_time"
    liq = liq.with_columns([
        pl.col(ts_col).cast(pl.Int64).alias("liq_ts"),
        pl.col("quantity").cast(pl.Float64).alias("liq_qty"),
    ])
    liq = liq.filter(pl.col("liq_qty") > 0)

    if len(liq) == 0:
        return liq_vol

    liq_ts = liq["liq_ts"].to_numpy()
    liq_qty = liq["liq_qty"].to_numpy()

    # Sort by timestamp for searchsorted
    sort_idx = np.argsort(liq_ts)
    liq_ts_sorted = liq_ts[sort_idx]
    liq_qty_sorted = liq_qty[sort_idx]
    # Cumulative sum for O(1) range-sum queries
    liq_cumsum = np.concatenate([[0.0], np.cumsum(liq_qty_sorted)])

    for i in range(n):
        lo = int(np.searchsorted(liq_ts_sorted, bar_start_ts[i], side="left"))
        hi = int(np.searchsorted(liq_ts_sorted, bar_close_ts[i], side="right"))
        liq_vol[i] = liq_cumsum[hi] - liq_cumsum[lo]

    return liq_vol


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features(
    bars: pl.DataFrame,
    raw_dir: str | Path,
) -> pl.DataFrame:
    """
    Extract 10 microstructure features from volume bars.

    Parameters
    ----------
    bars : pl.DataFrame
        Volume bars from volume_clock.build_volume_bars().
    raw_dir : str | Path
        Directory containing book.parquet and liq.parquet for the date.

    Returns
    -------
    pl.DataFrame
        timestamp_ms + 10 feature columns.
    """
    raw_dir = Path(raw_dir)
    book_path = raw_dir / "book.parquet"
    liq_path = raw_dir / "liq.parquet"

    n = len(bars)
    bar_close_ts = bars["timestamp_ms"].to_list()
    bar_start_ts = (bars["timestamp_ms"] - bars["duration_ms"]).to_list()
    duration_ms = bars["duration_ms"].to_numpy().astype(np.float64)
    num_trades = bars["num_trades"].to_numpy().astype(np.float64)
    buy_vol = bars["buy_volume"].to_numpy().astype(np.float64)
    sell_vol = bars["sell_volume"].to_numpy().astype(np.float64)
    open_px = bars["open"].to_numpy().astype(np.float64)
    close_px = bars["close"].to_numpy().astype(np.float64)

    # === 1. VPIN (50-bar rolling) ===
    total_vol = buy_vol + sell_vol
    flow_imbalance = np.abs(buy_vol - sell_vol)
    # Guard: if total_vol == 0, use 0 for that bar
    bar_vpin = np.where(total_vol > 0, flow_imbalance / total_vol, 0.0)

    # Rolling mean over 50 bars (causal: only uses past + current)
    vpin = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - VPIN_WINDOW + 1)
        vpin[i] = float(np.mean(bar_vpin[start : i + 1]))

    # === 2. OFI and 3. book_delta ===
    print(f"  Building book snapshots at {n} bar closes …", flush=True)
    snapshots = _build_book_snapshots_at_bar_closes(book_path, bar_close_ts)
    print(f"  {len(snapshots)} snapshots built", flush=True)
    ofi, book_delta = _compute_ofi_and_book_delta(snapshots, bar_close_ts)

    # === 4. trade_arrival_rate ===
    trade_arrival_rate = np.where(duration_ms > 0, num_trades / duration_ms, 0.0)

    # === 5. liq_volume ===
    print(f"  Computing liquidation volume …", flush=True)
    liq_volume = _compute_liq_volume(
        liq_path, np.array(bar_close_ts), np.array(bar_start_ts)
    )

    # === bar_return (for interaction feature) ===
    bar_return = np.where(
        np.abs(open_px) > 1e-9, (close_px - open_px) / open_px, 0.0
    )

    # === Interaction features ===
    vpin_x_arrival = vpin * trade_arrival_rate
    ofi_x_book_delta = ofi * book_delta
    liq_x_vpin = liq_volume * vpin
    vpin_x_duration = vpin * duration_ms
    liq_x_return = liq_volume * bar_return

    # Build output DataFrame
    result = pl.DataFrame({
        "timestamp_ms": bar_close_ts,
        "vpin": vpin,
        "ofi": ofi,
        "book_delta": book_delta,
        "trade_arrival_rate": trade_arrival_rate,
        "liq_volume": liq_volume,
        "vpin_x_arrival": vpin_x_arrival,
        "ofi_x_book_delta": ofi_x_book_delta,
        "liq_x_vpin": liq_x_vpin,
        "vpin_x_duration": vpin_x_duration,
        "liq_x_return": liq_x_return,
    })

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry: extract features from volume bars for a given date."""
    if len(sys.argv) < 2:
        print(
            f"Usage: python -m ofp.feature_extractor_v2 <date> [--threshold N]"
        )
        print(
            f"Example: python -m ofp.feature_extractor_v2 2026-06-17 --threshold 50"
        )
        sys.exit(1)

    date = sys.argv[1]
    threshold = 50.0
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])

    raw_dir = Path(f"data/raw_futures/{date}")
    trades_path = raw_dir / "trades.parquet"

    print(f"=== FEATURE EXTRACTOR v2 ===")
    print(f"  Date:      {date}")
    print(f"  Threshold: {threshold} BTC")
    print()

    # Build volume bars
    print("Building volume bars …")
    bars = build_volume_bars(trades_path, volume_threshold=threshold)
    print(f"  {len(bars)} bars")
    print()

    # Extract features
    print("Extracting features …")
    features = extract_features(bars, raw_dir)
    print()

    # Print results
    print(f"=== FEATURE MATRIX ===")
    print(f"  Shape: {features.shape}")
    print()

    print("=== FIRST 5 ROWS ===")
    print(features.head(5))
    print()

    # NaN/Inf check
    print("=== NaN/Inf CHECK ===")
    for col in features.columns:
        if col == "timestamp_ms":
            continue
        s = features[col]
        n_nan = int(s.is_nan().sum())
        n_inf = int(s.is_infinite().sum())
        print(f"  {col:25s}: NaN={n_nan}, Inf={n_inf}")

    # VPIN stats
    print()
    print("=== VPIN STATS ===")
    print(f"  min:  {float(features['vpin'].min()):.6f}")
    print(f"  max:  {float(features['vpin'].max()):.6f}")
    print(f"  mean: {float(features['vpin'].mean()):.6f}")

    # OFI stats
    print()
    print("=== OFI STATS ===")
    print(f"  min:  {float(features['ofi'].min()):.6f}")
    print(f"  max:  {float(features['ofi'].max()):.6f}")
    print(f"  mean: {float(features['ofi'].mean()):.6f}")


if __name__ == "__main__":
    main()