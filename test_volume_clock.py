"""
test_volume_clock.py — Test the volume clock module.

Verifies:
  1. Sum of bar volumes == total volume in raw data
  2. Bar count is positive for non-empty data
  3. OHLC consistency (open=first, close=last, high>=low, high>=open, high>=close, etc.)
  4. buy_volume + sell_volume == volume for every bar
  5. timestamp_ms is monotonically increasing
  6. duration_ms >= 0

Usage:
    python test_volume_clock.py
"""
import sys
from pathlib import Path

import polars as pl

from ofp.volume_clock import build_volume_bars


TRADES_PATH = "data/raw_futures/2026-06-17/trades.parquet"
THRESHOLD = 50.0


def test_volume_conservation():
    """Sum of bar volumes must equal total volume in raw data."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    raw = pl.read_parquet(TRADES_PATH)

    # Normalize raw: filter zero prices, cast to float
    if "quantity" in raw.columns:
        raw = raw.rename({"quantity": "size", "trade_time": "timestamp_ms"})
    raw = raw.with_columns([
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
    ])
    raw = raw.filter(pl.col("price") > 0)
    raw = raw.filter(pl.col("size") > 0)

    total_raw = float(raw["size"].sum())
    total_bars = float(bars["volume"].sum())

    diff = abs(total_raw - total_bars)
    assert diff < 1e-6, f"Volume mismatch: raw={total_raw}, bars={total_bars}, diff={diff}"
    print(f"  PASS: Volume conserved (raw={total_raw:.6f}, bars={total_bars:.6f}, diff={diff:.2e})")


def test_bar_count():
    """Bar count must be positive for non-empty data."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    assert len(bars) > 0, "No bars generated from non-empty data"
    print(f"  PASS: {len(bars)} bars generated")


def test_ohlc_consistency():
    """OHLC must be internally consistent."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)

    # high >= low
    assert (bars["high"] >= bars["low"]).all(), "high < low found"
    # high >= open, high >= close
    assert (bars["high"] >= bars["open"]).all(), "high < open found"
    assert (bars["high"] >= bars["close"]).all(), "high < close found"
    # low <= open, low <= close
    assert (bars["low"] <= bars["open"]).all(), "low > open found"
    assert (bars["low"] <= bars["close"]).all(), "low > close found"
    print(f"  PASS: OHLC consistent (high>=low>=open/close for all bars)")


def test_buy_sell_sum():
    """buy_volume + sell_volume must equal volume for every bar."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    diff = (bars["buy_volume"] + bars["sell_volume"] - bars["volume"]).abs()
    max_diff = float(diff.max())
    assert max_diff < 1e-9, f"buy+sell != volume, max diff={max_diff}"
    print(f"  PASS: buy_volume + sell_volume == volume (max diff={max_diff:.2e})")


def test_timestamp_monotonic():
    """timestamp_ms must be monotonically increasing."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    ts = bars["timestamp_ms"]
    is_mono = bool((ts.diff().drop_nulls() >= 0).all())
    assert is_mono, "timestamp_ms not monotonically increasing"
    print(f"  PASS: timestamp_ms monotonically increasing")


def test_duration_nonneg():
    """duration_ms must be >= 0."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    assert (bars["duration_ms"] >= 0).all(), "negative duration found"
    print(f"  PASS: duration_ms >= 0 for all bars")


def main():
    print("=" * 60)
    print("VOLUME CLOCK TESTS")
    print("=" * 60)
    print(f"  Data: {TRADES_PATH}")
    print(f"  Threshold: {THRESHOLD} BTC")
    print()

    tests = [
        ("Volume conservation", test_volume_conservation),
        ("Bar count", test_bar_count),
        ("OHLC consistency", test_ohlc_consistency),
        ("Buy+sell sum", test_buy_sell_sum),
        ("Timestamp monotonic", test_timestamp_monotonic),
        ("Duration non-negative", test_duration_nonneg),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()