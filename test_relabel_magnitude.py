"""
test_relabel_magnitude.py — Tests for magnitude-based relabeling.

Tests:
  1. Price hits +1% before -1% → label=1 (Target)
  2. Price hits -1% before +1% → label=0 (Stop)
  3. Price hits neither → label=2 (No Trade)
  4. Price hits +1% and -1% in same trade → tiebreaker logic
  5. Bars at end of day with incomplete forward data → label=2
"""
import numpy as np
import polars as pl
import pytest

from ofp.relabel_magnitude import (
    compute_magnitude_label,
    relabel_bars,
    LABEL_TARGET,
    LABEL_STOP,
    LABEL_NO_TRADE,
    TARGET_BPS,
    STOP_BPS,
    MAX_HORIZON_MS,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic trade series
# ---------------------------------------------------------------------------

def make_trades(prices: list[float], start_ms: int = 1_000_000, step_ms: int = 1_000) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted trade timestamps and prices from a list of prices."""
    ts = np.array([start_ms + i * step_ms for i in range(len(prices))], dtype=np.int64)
    px = np.array(prices, dtype=np.float64)
    return ts, px


# ---------------------------------------------------------------------------
# Test 1: +1% hit before -1% → Label 1 (Target)
# ---------------------------------------------------------------------------

def test_target_hit_first():
    """Price rises +1% then drops -1% → label=1."""
    entry_px = 100.0
    target_px = entry_px * 1.01   # 101.0
    stop_px = entry_px * 0.99     # 99.0

    # Price goes: 100 → 101 (target) → 99 (stop)
    prices = [entry_px, 100.5, target_px, 100.0, stop_px]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_TARGET, f"Expected label=1 (Target), got {label}"
    assert mr > 0, f"max_return should be positive, got {mr}"
    assert md < 0, f"max_drawdown should be negative, got {md}"


# ---------------------------------------------------------------------------
# Test 2: -1% hit before +1% → Label 0 (Stop)
# ---------------------------------------------------------------------------

def test_stop_hit_first():
    """Price drops -1% then rises +1% → label=0."""
    entry_px = 100.0
    target_px = entry_px * 1.01
    stop_px = entry_px * 0.99

    # Price goes: 100 → 99 (stop) → 101 (target)
    prices = [entry_px, 99.5, stop_px, 100.0, target_px]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_STOP, f"Expected label=0 (Stop), got {label}"
    assert mr > 0, f"max_return should be positive (later rally), got {mr}"
    assert md < 0, f"max_drawdown should be negative, got {md}"


# ---------------------------------------------------------------------------
# Test 3: Neither hits → Label 2 (No Trade)
# ---------------------------------------------------------------------------

def test_neither_hit():
    """Price stays within ±0.5% → label=2."""
    entry_px = 100.0

    # Price oscillates within ±0.5% (50 bps), never reaches ±100 bps
    prices = [entry_px, 100.3, 99.8, 100.2, 99.9, 100.1, 100.0]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_NO_TRADE, f"Expected label=2 (No Trade), got {label}"
    assert abs(mr) < 100, f"max_return should be < 100 bps, got {mr}"
    assert abs(md) < 100, f"max_drawdown should be < 100 bps, got {md}"


# ---------------------------------------------------------------------------
# Test 4: Both hit in same trade → tiebreaker by price direction
# ---------------------------------------------------------------------------

def test_tiebreaker_target():
    """Both thresholds hit at same index, price above entry → label=1."""
    entry_px = 100.0
    # A single trade at 102 (both +1% and -1% are "hit" at same index)
    # Price is above entry → target first
    prices = [entry_px, 102.0]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_TARGET, f"Expected label=1 (tiebreaker: above entry), got {label}"


def test_tiebreaker_stop():
    """Both thresholds hit at same index, price below entry → label=0."""
    entry_px = 100.0
    # A single trade at 98 (both thresholds "hit" at same index)
    # Price is below entry → stop first
    prices = [entry_px, 98.0]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_STOP, f"Expected label=0 (tiebreaker: below entry), got {label}"


# ---------------------------------------------------------------------------
# Test 5: Incomplete forward data → Label 2
# ---------------------------------------------------------------------------

def test_incomplete_forward_data():
    """Bar at end of data with no forward trades → label=2."""
    entry_px = 100.0

    # Only one trade (the entry itself), no forward data
    prices = [entry_px]
    ts, px = make_trades(prices)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=entry_px)

    assert label == LABEL_NO_TRADE, f"Expected label=2 (no forward data), got {label}"


def test_bar_beyond_last_trade():
    """Bar timestamp is after all trades → label=2."""
    entry_px = 100.0
    prices = [100.0, 101.0, 99.0]
    ts, px = make_trades(prices, start_ms=1_000_000)

    # Bar close is 10s after last trade
    bar_close_ms = ts[-1] + 10_000

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=bar_close_ms, bar_close_px=entry_px)

    assert label == LABEL_NO_TRADE, f"Expected label=2 (bar beyond last trade), got {label}"


# ---------------------------------------------------------------------------
# Test 6: Guard — zero entry price
# ---------------------------------------------------------------------------

def test_zero_entry_price():
    """Bar close price of 0 → label=2 (guard against division by zero)."""
    ts = np.array([1_000_000, 1_001_000], dtype=np.int64)
    px = np.array([0.0, 100.0], dtype=np.float64)

    label, mr, md = compute_magnitude_label(ts, px, bar_close_ms=ts[0], bar_close_px=0.0)

    assert label == LABEL_NO_TRADE, f"Expected label=2 (zero entry price guard), got {label}"
    assert mr == 0.0 and md == 0.0, f"Expected 0,0 returns for zero entry, got {mr}, {md}"


# ---------------------------------------------------------------------------
# Test 7: Full relabel_bars pipeline with synthetic data
# ---------------------------------------------------------------------------

def test_relabel_bars_pipeline(tmp_path):
    """End-to-end test: create fake raw data, run relabel_bars, check labels."""
    # Create fake raw trades directory
    raw_dir = tmp_path / "raw_futures" / "2026-06-17"
    raw_dir.mkdir(parents=True)

    # Use realistic timestamps for 2026-06-17 00:00:00 UTC
    # 2026-06-17 00:00:00 UTC = 1781654400000 ms
    base_ms = 1_781_654_400_000

    # Build a price series: 100 → 101 (target) → 99 (stop)
    entry_px = 100.0
    n_trades = 200
    ts_arr = np.array([base_ms + i * 1_000 for i in range(n_trades)], dtype=np.int64)
    # First 100 trades: rise from 100 to 101 (+1%)
    # Next 100 trades: drop from 101 to 99 (-2% from entry)
    px_arr = np.zeros(n_trades, dtype=np.float64)
    px_arr[:100] = np.linspace(100.0, 101.0, 100)
    px_arr[100:] = np.linspace(101.0, 99.0, 100)

    trades_df = pl.DataFrame({
        "trade_time": ts_arr,
        "price": px_arr.astype(str),
        "quantity": ["0.1"] * n_trades,
        "is_buyer_maker": [False] * n_trades,
    })
    trades_df.write_parquet(str(raw_dir / "trades.parquet"))

    # Build fake volume bars (3 bars, close at 100.0, 100.5, 101.0)
    # Bar timestamps are before the trades so forward window covers them
    bars = pl.DataFrame({
        "timestamp_ms": pl.Series([base_ms - 500_000, base_ms - 200_000, base_ms + 100_000], dtype=pl.Int64),
        "open": pl.Series([100.0, 100.2, 100.8], dtype=pl.Float64),
        "high": pl.Series([100.5, 100.8, 101.0], dtype=pl.Float64),
        "low": pl.Series([99.8, 100.0, 100.5], dtype=pl.Float64),
        "close": pl.Series([100.0, 100.5, 101.0], dtype=pl.Float64),
        "volume": pl.Series([50.0, 50.0, 50.0], dtype=pl.Float64),
        "buy_volume": pl.Series([25.0, 25.0, 25.0], dtype=pl.Float64),
        "sell_volume": pl.Series([25.0, 25.0, 25.0], dtype=pl.Float64),
        "num_trades": pl.Series([10, 10, 10], dtype=pl.Int64),
        "duration_ms": pl.Series([1000, 1000, 1000], dtype=pl.Int64),
    })

    # Relabel
    result = relabel_bars(bars, raw_dir=tmp_path / "raw_futures")

    # Check that label column exists
    assert "label" in result.columns
    assert "max_return_bps" in result.columns
    assert "max_drawdown_bps" in result.columns

    labels = result["label"].to_numpy()
    # Bar 0 (close=100.0): forward prices rise to 101 (+1%) then drop to 99 (-1%)
    # Target should hit first
    assert labels[0] == LABEL_TARGET, f"Bar 0: expected label=1, got {labels[0]}"

    # Bar 1 (close=100.5): target = 101.5, stop = 99.5
    # Forward prices max = 101.0 (< 101.5, no target), min = 99.0 (< 99.5, stop hit)
    # Stop should hit
    assert labels[1] == LABEL_STOP, f"Bar 1: expected label=0, got {labels[1]}"

    # Bar 2 (close=101.0): target = 102.01, stop = 99.99
    # Forward prices max = 101.0 (no further rise), min = 99.0 (< 99.99, stop hit)
    # Stop should hit
    assert labels[2] == LABEL_STOP, f"Bar 2: expected label=0, got {labels[2]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])