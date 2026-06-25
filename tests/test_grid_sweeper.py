"""
test_grid_sweeper.py â€” Tests for the GridSweeper.

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
        """40 trades spanning 2_000_000..2_039_000 ms â€” enough for macro window."""
        trades = _trades_df([
            (2_000_000 + i * 1000, 70000.0, 1.0, False) for i in range(40)
        ])
        book = _book_snapshots(2_000_000)

        # micro=2s, step=1s, horizon=10s (10000ms)
        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        # Valid: win_start + 2000 + 10000 <= 2_039_000 â†’ win_start <= 2_027_000
        # win_start starts at 2_000_000, step=1000 â†’ 28 windows
        assert len(results) == 28

    def test_multiple_horizons(self) -> None:
        """Each (window, horizon) combo is enumerated."""
        trades = _trades_df([
            (2_000_000 + i * 1000, 70000.0, 1.0, False) for i in range(60)
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2, 4], horizons_sec=[10, 20])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        # W=2s, step=1s, H=10s (10000ms):
        #   win_start + 2000 + 10000 <= 2_059_000 â†’ ws <= 2_047_000
        #   start=2_000_000, step=1000 â†’ 48 windows
        # W=2s, step=1s, H=20s (20000ms):
        #   ws + 2000 + 20000 <= 2_059_000 â†’ ws <= 2_037_000 â†’ 38 windows
        # W=4s, step=2s, H=10s:
        #   ws + 4000 + 10000 <= 2_059_000 â†’ ws <= 2_045_000
        #   start=2_000_000, step=2000 â†’ 23 windows
        # W=4s, step=2s, H=20s:
        #   ws + 4000 + 20000 <= 2_059_000 â†’ ws <= 2_035_000 â†’ 18 windows
        # Total = 48+38+23+18 = 127
        assert len(results) == 127

        # Verify diversity of params
        ws = {r["window_size"] for r in results}
        hz = {r["horizon"] for r in results}
        assert ws == {2, 4}
        assert hz == {10, 20}


class TestOutcomeBinary:
    """outcome_binary thresholds."""

    def test_binary_one_when_up_enough(self) -> None:
        """Future price > current by > 0.1 % â†’ outcome_binary = 1."""
        trades = _trades_df([
            (2_000_000, 70000.0, 1.0, False),
            (2_001_000, 70000.0, 1.0, False),
            (2_002_000, 70000.0, 1.0, False),   # window end (micro=2s)
            (2_012_000, 70100.0, 1.0, False),   # future (horizon=10s): +0.143 %
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        assert len(results) == 1
        r = results[0]
        assert r["outcome_pct"] == pytest.approx(100.0 / 70000.0, rel=1e-6)
        assert r["outcome_binary"] == 1

    def test_binary_one_when_up_any_amount(self) -> None:
        """Future price up +0.05 % â†’ outcome_binary = 1 (>0 threshold)."""
        trades = _trades_df([
            (2_000_000, 70000.0, 1.0, False),
            (2_001_000, 70000.0, 1.0, False),
            (2_002_000, 70000.0, 1.0, False),
            (2_012_000, 70035.0, 1.0, False),   # +0.05 %
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        assert len(results) == 1
        assert results[0]["outcome_binary"] == 1

    def test_binary_zero_when_down(self) -> None:
        """Future price below current â†’ outcome_binary = 0."""
        trades = _trades_df([
            (2_000_000, 70000.0, 1.0, False),
            (2_001_000, 70000.0, 1.0, False),
            (2_002_000, 70000.0, 1.0, False),
            (2_012_000, 69800.0, 1.0, False),   # down
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        assert results[0]["outcome_binary"] == 0

    def test_binary_zero_when_exact_same_price(self) -> None:
        """Future price == current â†’ outcome_binary = 0."""
        trades = _trades_df([
            (2_000_000, 70000.0, 1.0, False),
            (2_001_000, 70000.0, 1.0, False),
            (2_002_000, 70000.0, 1.0, False),
            (2_012_000, 70000.0, 1.0, False),
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        results = list(sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        ))

        assert results[0]["outcome_pct"] == 0.0
        assert results[0]["outcome_binary"] == 0


class TestGeneratorBehaviour:
    """The generator yields one dict at a time, no in-memory bulk load."""

    def test_yields_one_at_a_time(self) -> None:
        trades = _trades_df([
            (2_000_000 + i * 1000, 70000.0 + i, 1.0, False) for i in range(40)
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        gen = sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        )

        # Verify it's a generator (has __next__)
        assert hasattr(gen, "__next__")

        # Pull a few rows
        first = next(gen)
        assert isinstance(first, dict)
        assert "outcome_pct" in first
        assert "outcome_binary" in first
        assert "micro_buy_volume" in first
        assert "meso_buy_volume" in first
        assert "macro_buy_volume" in first
        assert "window_size" in first

        second = next(gen)
        assert second["window_end_ms"] > first["window_end_ms"]

        # Consume the rest â€” should not blow up memory
        remaining = list(gen)
        assert len(remaining) >= 0  # just that it finishes

    def test_save_to_disk_writes_parquet(self) -> None:
        trades = _trades_df([
            (2_000_000 + i * 1000, 70000.0 + i, 1.0, False) for i in range(40)
        ])
        book = _book_snapshots(2_000_000)

        sweeper = GridSweeper(window_sizes_sec=[2], horizons_sec=[10])
        gen = sweeper.sweep(
            trades_df=trades,
            book_snapshots=book,
            liq_df=_empty_liq_df(),
            rolling_stats_per_zoom={
                "micro": {"rolling_avg_volume": 1000.0},
                "meso":  {"rolling_avg_volume": 1000.0},
                "macro": {"rolling_avg_volume": 1000.0},
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            fpath = str(Path(tmp) / "sweep.parquet")
            GridSweeper.save_to_disk(gen, fpath, chunk_size=5)

            written = pd.read_parquet(fpath)
            assert len(written) > 0
            assert "outcome_pct" in written.columns
            assert "outcome_binary" in written.columns
            assert "micro_buy_volume" in written.columns
            assert "window_size" in written.columns
            assert "horizon" in written.columns
