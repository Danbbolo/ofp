"""
feature_extractor_v2.py — Microstructure feature extractor for volume bars.

Extracts 15 features at each volume bar close. No time-based aggregates.
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

Extended Features (from V10 engine):
 11. cvd_momentum: Rate of change of cumulative volume delta over last 10 bars.
 12. wall_lifecycle: Large orders (>=5 BTC) appearing minus disappearing, last 10 bars.
 13. volume_profile_entropy: Shannon entropy of volume across price levels, last 50 bars.
 14. large_trade_count: Trades with quantity >= 1 BTC in current bar.
 15. macro_trade_size_skew: Skewness of trade sizes in last 50 bars.

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
# CVD momentum
# ---------------------------------------------------------------------------

def _compute_cvd_momentum(
    buy_vol: np.ndarray,
    sell_vol: np.ndarray,
    lookback: int = 10,
) -> np.ndarray:
    """
    Rate of change of cumulative volume delta over last 10 bars.

    Formula: (current_cvd - cvd_10_bars_ago) / cvd_10_bars_ago
    Guard: if cvd_10_bars_ago == 0, return 0.
    """
    n = len(buy_vol)
    cvd = np.cumsum(buy_vol - sell_vol)
    cvd_momentum = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        prev = cvd[i - lookback]
        if abs(prev) > 1e-9:
            cvd_momentum[i] = (cvd[i] - prev) / abs(prev)
    return cvd_momentum


# ---------------------------------------------------------------------------
# Wall lifecycle
# ---------------------------------------------------------------------------

def _compute_wall_lifecycle(
    snapshots: dict[int, tuple[list, list]],
    bar_close_ts: list[int],
    wall_threshold: float = 5.0,
    lookback: int = 10,
) -> np.ndarray:
    """
    Track large orders (>= wall_threshold BTC) in top 20 levels.

    For each bar, count walls that appeared minus walls that disappeared
    over the last `lookback` bars. Positive = walls building (support),
    negative = walls collapsing (resistance breaking).
    """
    n = len(bar_close_ts)
    wall_lifecycle = np.zeros(n, dtype=np.float64)

    prev_walls: set[float] = set()
    appeared_counts = np.zeros(n, dtype=np.float64)
    disappeared_counts = np.zeros(n, dtype=np.float64)

    for i, ts in enumerate(bar_close_ts):
        snap = snapshots.get(ts)
        if snap is None:
            continue

        bids, asks = snap
        curr_walls: set[float] = set()
        for price, size in bids:
            if size >= wall_threshold:
                curr_walls.add(round(price, 2))
        for price, size in asks:
            if size >= wall_threshold:
                curr_walls.add(round(price, 2))

        appeared_counts[i] = float(len(curr_walls - prev_walls))
        disappeared_counts[i] = float(len(prev_walls - curr_walls))
        prev_walls = curr_walls

    # Rolling sum over last `lookback` bars
    for i in range(n):
        start = max(0, i - lookback + 1)
        wall_lifecycle[i] = float(
            appeared_counts[start : i + 1].sum()
            - disappeared_counts[start : i + 1].sum()
        )

    return wall_lifecycle


# ---------------------------------------------------------------------------
# Trade-level features (volume profile entropy, large trade count, size skew)
# ---------------------------------------------------------------------------

def _load_trades_for_features(
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load trades for a date and return sorted (ts, price, quantity) arrays.
    Returns None if no trades file.
    """
    trades_path = raw_dir / "trades.parquet"
    if not trades_path.exists():
        return None

    df = pl.read_parquet(str(trades_path))
    if "trade_time" in df.columns:
        df = df.rename({"trade_time": "timestamp_ms", "quantity": "size"})
    df = df.with_columns([
        pl.col("timestamp_ms").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
    ])
    df = df.filter(pl.col("price") > 0)
    df = df.sort("timestamp_ms")

    return (
        df["timestamp_ms"].to_numpy(),
        df["price"].to_numpy(),
        df["size"].to_numpy(),
    )


def _compute_volume_profile_entropy(
    trades_ts: np.ndarray,
    trades_px: np.ndarray,
    trades_qty: np.ndarray,
    bar_close_ts: np.ndarray,
    bar_start_ts: np.ndarray,
    lookback: int = 50,
) -> np.ndarray:
    """
    Shannon entropy of volume distribution across price levels in last 50 bars.

    Low entropy = volume concentrated at specific prices (breakout setup).
    High entropy = volume scattered (chop).
    """
    n = len(bar_close_ts)
    entropy = np.zeros(n, dtype=np.float64)

    for i in range(n):
        window_start = bar_start_ts[max(0, i - lookback + 1)]
        window_end = bar_close_ts[i]

        lo = int(np.searchsorted(trades_ts, window_start, side="left"))
        hi = int(np.searchsorted(trades_ts, window_end, side="right"))

        if hi <= lo:
            continue

        window_qty = trades_qty[lo:hi]
        total_vol = float(window_qty.sum())
        if total_vol <= 0:
            continue

        # Group by price (round to 0.1 to avoid float noise)
        rounded_px = np.round(trades_px[lo:hi], 1)
        unique_px, inverse = np.unique(rounded_px, return_inverse=True)
        vol_at_price = np.zeros(len(unique_px), dtype=np.float64)
        np.add.at(vol_at_price, inverse, window_qty)

        # Entropy: -sum(p_i * log(p_i))
        p = vol_at_price / total_vol
        p = p[p > 0]
        entropy[i] = float(-np.sum(p * np.log(p)))

    return entropy


def _compute_large_trade_count(
    trades_ts: np.ndarray,
    trades_qty: np.ndarray,
    bar_close_ts: np.ndarray,
    bar_start_ts: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Number of trades with quantity >= threshold BTC in the current bar.
    Proxy for institutional footprints.
    """
    n = len(bar_close_ts)
    large_count = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = int(np.searchsorted(trades_ts, bar_start_ts[i], side="left"))
        hi = int(np.searchsorted(trades_ts, bar_close_ts[i], side="right"))

        if hi <= lo:
            continue

        large_count[i] = float(np.sum(trades_qty[lo:hi] >= threshold))

    return large_count


def _compute_macro_trade_size_skew(
    trades_ts: np.ndarray,
    trades_qty: np.ndarray,
    bar_close_ts: np.ndarray,
    bar_start_ts: np.ndarray,
    lookback: int = 50,
) -> np.ndarray:
    """
    Skewness of trade sizes in last 50 bars.

    Positive skew = a few large trades dominating (informed flow).
    Negative skew = many small trades (retail flow).
    """
    n = len(bar_close_ts)
    skew = np.zeros(n, dtype=np.float64)

    for i in range(n):
        window_start = bar_start_ts[max(0, i - lookback + 1)]
        window_end = bar_close_ts[i]

        lo = int(np.searchsorted(trades_ts, window_start, side="left"))
        hi = int(np.searchsorted(trades_ts, window_end, side="right"))

        if hi - lo < 3:  # need at least 3 samples
            continue

        sizes = trades_qty[lo:hi]
        mean = float(sizes.mean())
        std = float(sizes.std())

        if std < 1e-9:
            continue

        skew[i] = float(np.mean((sizes - mean) ** 3) / (std ** 3))

    return skew


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features(
    bars: pl.DataFrame,
    raw_dir: str | Path,
) -> pl.DataFrame:
    """
    Extract 15 microstructure features from volume bars.

    Parameters
    ----------
    bars : pl.DataFrame
        Volume bars from volume_clock.build_volume_bars().
    raw_dir : str | Path
        Directory containing book.parquet, liq.parquet, and trades.parquet.

    Returns
    -------
    pl.DataFrame
        timestamp_ms + 15 feature columns.
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
    # Suppress divide-by-zero warning — np.where guards the result,
    # but numpy still evaluates the division before selecting.
    with np.errstate(divide="ignore", invalid="ignore"):
        trade_arrival_rate = np.where(
            duration_ms > 0, num_trades / duration_ms, 0.0
        )

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

    # === 11. CVD momentum (10-bar lookback) ===
    print(f"  Computing CVD momentum …", flush=True)
    cvd_momentum = _compute_cvd_momentum(buy_vol, sell_vol, lookback=10)

    # === 12. Wall lifecycle (10-bar lookback, >= 5 BTC walls) ===
    print(f"  Computing wall lifecycle …", flush=True)
    wall_lifecycle = _compute_wall_lifecycle(snapshots, bar_close_ts)

    # === 13-15. Trade-level features (entropy, large count, size skew) ===
    print(f"  Computing trade-level features …", flush=True)
    trades_data = _load_trades_for_features(raw_dir)
    if trades_data is not None:
        t_ts, t_px, t_qty = trades_data
        bar_close_arr = np.array(bar_close_ts, dtype=np.int64)
        bar_start_arr = np.array(bar_start_ts, dtype=np.int64)

        volume_profile_entropy = _compute_volume_profile_entropy(
            t_ts, t_px, t_qty, bar_close_arr, bar_start_arr, lookback=50,
        )
        large_trade_count = _compute_large_trade_count(
            t_ts, t_qty, bar_close_arr, bar_start_arr, threshold=1.0,
        )
        macro_trade_size_skew = _compute_macro_trade_size_skew(
            t_ts, t_qty, bar_close_arr, bar_start_arr, lookback=50,
        )
    else:
        volume_profile_entropy = np.zeros(n, dtype=np.float64)
        large_trade_count = np.zeros(n, dtype=np.float64)
        macro_trade_size_skew = np.zeros(n, dtype=np.float64)

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
        "cvd_momentum": cvd_momentum,
        "wall_lifecycle": wall_lifecycle,
        "volume_profile_entropy": volume_profile_entropy,
        "large_trade_count": large_trade_count,
        "macro_trade_size_skew": macro_trade_size_skew,
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