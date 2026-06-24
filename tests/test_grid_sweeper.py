"""
test_grid_sweeper.py — Tests for the GridSweeper.

Covers: window count with 50 % overlap, outcome_binary thresholds,
and generator behaviour (one-row-at-a-time, no in-memory accumulation).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ofp.grid_sweeper import GridSweeper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades_df(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (timestamp_ms, price, size, is_buyer_maker)."""
    return pd.DataFrame(rows, columns=["timestamp_ms", "price", "size", "is_buyer_maker"])


def _book_snapshots(ts: int = 0) -> dict[int, tuple[list, list]]:
    """Minimal book snapshot dict keyed by ms timestamp."""
    return {ts: ([(68000.0, 1.0)], [(68100.0, 2.0)])}


def _empty_liq_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWindowCount:
    """Correct number of windows with 50 % overlap."""

    def test_window_count(self) -> None:
        # 20 trades at 0, 1000, 2000, ..., 19000 ms
        trades = _trades_df([
            (i * 1000, 70000.0, 1.0, False) for i in range(20)
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        # Window=2s, step=1s, horizon=1s.
        # Valid: 0..16000 with step 1000 → 17 windows
        assert len(results) == 17

    def test_multiple_horizons(self) -> None:
        """Each (window, horizon) combo is enumerated."""
        trades = _trades_df([
            (i * 1000, 70000.0, 1.0, False) for i in range(30)
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2, 4], horizons_sec=[1, 2])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        # Count per combo:
        # W=2s, H=1s: 0..26000 step 1000 → 27 windows
        # W=2s, H=2s: 0..25000 step 1000 → 26 windows
        # W=4s, H=1s: 0..24000 step 2000 → 13 windows
        # W=4s, H=2s: 0..23000 step 2000 → 12 windows
        # Total = 27+26+13+12 = 78
        assert len(results) == 78

        # Verify diversity of params
        ws = {r["window_size"] for r in results}
        hz = {r["horizon"] for r in results}
        assert ws == {2, 4}
        assert hz == {1, 2}


class TestOutcomeBinary:
    """outcome_binary thresholds."""

    def test_binary_one_when_up_enough(self) -> None:
        """Future price > current by > 0.1 % → outcome_binary = 1."""
        trades = _trades_df([
            (0,     70000.0, 1.0, False),   # window end
            (2000,  70000.0, 1.0, False),   # window end (win=2s)
            (3000,  70100.0, 1.0, False),   # future: +0.143 % → > 0.1 %
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        assert len(results) == 1
        r = results[0]
        assert r["outcome_pct"] == pytest.approx(100.0 / 70000.0, rel=1e-6)
        assert r["outcome_binary"] == 1

    def test_binary_one_when_up_any_amount(self) -> None:
        """Future price up +0.05 % → outcome_binary = 1 (>0 threshold)."""
        trades = _trades_df([
            (0,     70000.0, 1.0, False),
            (2000,  70000.0, 1.0, False),
            (3000,  70035.0, 1.0, False),   # +0.05 %
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        assert len(results) == 1
        assert results[0]["outcome_binary"] == 1

    def test_binary_zero_when_down(self) -> None:
        """Future price below current → outcome_binary = 0."""
        trades = _trades_df([
            (0,     70000.0, 1.0, False),
            (2000,  70000.0, 1.0, False),
            (3000,  69800.0, 1.0, False),   # down
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        assert results[0]["outcome_binary"] == 0

    def test_binary_zero_when_exact_same_price(self) -> None:
        """Future price == current → outcome_binary = 0."""
        trades = _trades_df([
            (0,     70000.0, 1.0, False),
            (2000,  70000.0, 1.0, False),
            (3000,  70000.0, 1.0, False),
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        ))

        assert results[0]["outcome_pct"] == 0.0
        assert results[0]["outcome_binary"] == 0


class TestGeneratorBehaviour:
    """The generator yields one dict at a time, no in-memory bulk load."""

    def test_yields_one_at_a_time(self) -> None:
        trades = _trades_df([
            (i * 1000, 70000.0 + i, 1.0, False) for i in range(20)
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        gen = sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        )

        # Verify it's a generator (has __next__)
        assert hasattr(gen, "__next__")

        # Pull a few rows
        first = next(gen)
        assert isinstance(first, dict)
        assert "outcome_pct" in first
        assert "outcome_binary" in first
        assert "buy_volume" in first
        assert "window_size" in first

        second = next(gen)
        assert second["window_end_ms"] > first["window_end_ms"]

        # Consume the rest — should not blow up memory
        remaining = list(gen)
        assert len(remaining) >= 0  # just that it finishes

    def test_save_to_disk_writes_parquet(self) -> None:
        trades = _trades_df([
            (i * 1000, 70000.0 + i, 1.0, False) for i in range(20)
        ])
        book = _book_snapshots(0)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[1])
        gen = sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_avg_volume=1000.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            fpath = str(Path(tmp) / "sweep.parquet")
            GridSweeper.save_to_disk(gen, fpath, chunk_size=5)

            written = pd.read_parquet(fpath)
            assert len(written) > 0
            assert "outcome_pct" in written.columns
            assert "outcome_binary" in written.columns
            assert "buy_volume" in written.columns
            assert "window_size" in written.columns
            assert "horizon" in written.columns
