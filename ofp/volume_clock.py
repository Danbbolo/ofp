"""
volume_clock.py — Convert time-based tick data to volume-based bars.

Volume bars group trades by a fixed volume threshold rather than fixed time
intervals. This produces more stationary features: bars are denser during
high-activity periods and sparser during quiet periods, aligning the sampling
rate with information flow rather than clock time.

Each bar outputs:
  - timestamp_ms: close time of the bar (last trade timestamp in the bucket)
  - open:  first trade price in the bucket
  - high:  max trade price in the bucket
  - low:   min trade price in the bucket
  - close: last trade price in the bucket
  - volume: total quantity in the bucket
  - buy_volume:  quantity from taker-buy trades (is_buyer_maker=False)
  - sell_volume: quantity from taker-sell trades (is_buyer_maker=True)
  - num_trades:  count of trades in the bucket
  - duration_ms: time span from first to last trade in the bucket

Usage:
    from ofp.volume_clock import build_volume_bars
    bars = build_volume_bars("data/raw_futures/2026-06-17/trades.parquet", volume_threshold=50.0)

    # Or run as a script:
    # python -m ofp.volume_clock data/raw_futures/2026-06-17/trades.parquet --threshold 50
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def build_volume_bars(
    trades_path: str | Path,
    volume_threshold: float = 50.0,
) -> pl.DataFrame:
    """
    Convert tick trades to volume bars using Polars.

    Parameters
    ----------
    trades_path : str | Path
        Path to the trades parquet file (columns: trade_time, price,
        quantity, is_buyer_maker).
    volume_threshold : float
        Volume threshold per bar in base currency (default 50 BTC).

    Returns
    -------
    pl.DataFrame
        Columns: timestamp_ms, open, high, low, close, volume,
        buy_volume, sell_volume, num_trades, duration_ms.
    """
    # Load and normalize — price/quantity are StringDtype in raw data
    df = pl.read_parquet(str(trades_path))

    # Rename to standard schema
    if "trade_time" in df.columns:
        df = df.rename({"trade_time": "timestamp_ms", "quantity": "size"})
    elif "timestamp_ms" not in df.columns:
        raise ValueError(f"Expected 'trade_time' or 'timestamp_ms' column, got {df.columns}")

    # Cast types — price and size come as strings
    df = df.with_columns([
        pl.col("timestamp_ms").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Boolean),
    ])

    # Drop zero/negative-price trades (liquidation prints, bad data)
    df = df.filter(pl.col("price") > 0)
    df = df.filter(pl.col("size") > 0)

    # Sort by timestamp (data should already be sorted, but enforce)
    df = df.sort("timestamp_ms")

    # Compute cumulative volume
    df = df.with_columns([
        pl.col("size").cum_sum().alias("cum_volume"),
    ])

    # Assign bar IDs: floor(cum_volume / threshold)
    # Bar 0 = first `threshold` BTC, bar 1 = next `threshold`, etc.
    df = df.with_columns([
        (pl.col("cum_volume") / volume_threshold).floor().cast(pl.Int64).alias("bar_id"),
    ])

    # Aggregate per bar
    bars = df.group_by("bar_id", maintain_order=True).agg([
        pl.col("timestamp_ms").last().alias("timestamp_ms"),   # close time
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").sum().alias("volume"),
        pl.col("timestamp_ms").first().alias("_first_ts"),
        pl.col("timestamp_ms").last().alias("_last_ts"),
        pl.col("is_buyer_maker").sum().cast(pl.Int64).alias("_maker_count"),
        pl.len().alias("num_trades"),
    ])

    # Compute derived columns
    # buy_volume = size where is_buyer_maker=False (taker buys)
    # sell_volume = size where is_buyer_maker=True (taker sells = maker)
    # We need to compute these separately since group_by agg above doesn't
    # easily do conditional sums. Recompute with a join-free approach.
    df = df.with_columns([
        pl.when(pl.col("is_buyer_maker") == False)
          .then(pl.col("size"))
          .otherwise(0.0)
          .alias("buy_vol"),
        pl.when(pl.col("is_buyer_maker") == True)
          .then(pl.col("size"))
          .otherwise(0.0)
          .alias("sell_vol"),
    ])

    buy_sell = df.group_by("bar_id", maintain_order=True).agg([
        pl.col("buy_vol").sum().alias("buy_volume"),
        pl.col("sell_vol").sum().alias("sell_volume"),
    ])

    bars = bars.join(buy_sell, on="bar_id", how="left")

    # Duration: last_ts - first_ts
    bars = bars.with_columns([
        (pl.col("_last_ts") - pl.col("_first_ts")).alias("duration_ms"),
    ])

    # Select final columns in order
    bars = bars.select([
        "timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "sell_volume",
        "num_trades",
        "duration_ms",
    ])

    return bars


def main() -> None:
    """CLI entry: build volume bars from a trades file and print summary."""
    # Parse args
    if len(sys.argv) < 2:
        print(f"Usage: python -m ofp.volume_clock <trades.parquet> [--threshold N]")
        print(f"Example: python -m ofp.volume_clock data/raw_futures/2026-06-17/trades.parquet --threshold 50")
        sys.exit(1)

    trades_path = sys.argv[1]
    threshold = 50.0

    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])

    print(f"=== VOLUME CLOCK ===")
    print(f"  Input:    {trades_path}")
    print(f"  Threshold: {threshold} BTC per bar")
    print()

    # Build bars
    bars = build_volume_bars(trades_path, volume_threshold=threshold)

    # Print summary
    print(f"Number of bars generated: {len(bars)}")
    print()

    print("=== FIRST 5 ROWS ===")
    print(bars.head(5))
    print()

    print("=== LAST 5 ROWS ===")
    print(bars.tail(5))
    print()

    # Verification: sum of bar volumes == total volume in raw data
    raw = pl.read_parquet(trades_path)
    if "quantity" in raw.columns:
        total_raw = raw["quantity"].cast(pl.Float64).filter(
            pl.col("price").cast(pl.Float64) > 0
        ).sum()
    else:
        total_raw = raw["size"].cast(pl.Float64).filter(
            pl.col("price").cast(pl.Float64) > 0
        ).sum()

    total_bars = bars["volume"].sum()
    print(f"=== VERIFICATION ===")
    print(f"  Total volume (raw):  {total_raw:.6f} BTC")
    print(f"  Total volume (bars): {total_bars:.6f} BTC")
    print(f"  Match: {abs(float(total_raw) - float(total_bars)) < 1e-6}")
    print()

    # Extra stats
    print(f"=== BAR STATS ===")
    print(f"  Avg volume/bar:  {float(bars['volume'].mean()):.4f} BTC")
    print(f"  Avg duration:    {float(bars['duration_ms'].mean()):.0f} ms")
    print(f"  Avg trades/bar:  {float(bars['num_trades'].mean()):.1f}")
    print(f"  Buy/sell ratio:  {float(bars['buy_volume'].sum()) / max(float(bars['sell_volume'].sum()), 1e-9):.4f}")


if __name__ == "__main__":
    main()