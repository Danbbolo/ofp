"""
test_feature_extractor.py — Tests for extract_features().

Covers: empty window, single buy, single sell, mixed trades, delta curve.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ofp.feature_extractor import extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades_df(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (timestamp_ms, price, size, is_buyer_maker)."""
    return pd.DataFrame(rows, columns=["timestamp_ms", "price", "size", "is_buyer_maker"])


def _empty_book() -> tuple[list, list]:
    return ([], [])


def _empty_liq_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])


# Shorthand for the long call
def _extract(
    trades,
    rolling_avg: float = 1000.0,
    window_start_ms: int = 0,
    window_end_ms: int = 60_000,
    **kwargs,
):
    return extract_features(
        trades_df=trades,
        book_snapshot_start=_empty_book(),
        book_snapshot_end=_empty_book(),
        liq_df=_empty_liq_df(),
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        rolling_avg_volume=rolling_avg,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyWindow:
    """No trades in window — all volume features should be 0.0, no crashes."""

    def test_all_volume_features_zero(self) -> None:
        trades = _trades_df([
            (70_000, 68000.0, 1.0, False),  # outside window (>= 60_000)
        ])
        feats = _extract(trades)

        assert feats["buy_volume"] == 0.0
        assert feats["sell_volume"] == 0.0
        assert feats["net_volume"] == 0.0
        assert feats["volume_vs_avg"] == 0.0
        assert feats["large_trade_net"] == 0.0
        assert feats["acceleration"] == 0.0
        assert feats["delta_1"] == 0.0
        assert feats["delta_2"] == 0.0
        assert feats["delta_3"] == 0.0
        assert feats["delta_4"] == 0.0
        assert feats["delta_5"] == 0.0

    def test_buy_sell_ratio_is_finite(self) -> None:
        """Even with 0 sell_volume, buy_sell_ratio must not be inf."""
        trades = _trades_df([])
        feats = _extract(trades)
        assert feats["buy_sell_ratio"] == 0.0  # 0 / 1e-9

    def test_context_features_still_produced(self) -> None:
        trades = _trades_df([])
        feats = _extract(trades)
        assert "hour_sin" in feats
        assert "hour_cos" in feats
        assert isinstance(feats["hour_sin"], float)

    def test_exactly_28_keys(self) -> None:
        trades = _trades_df([])
        feats = _extract(trades)
        assert len(feats) == 28


class TestSingleBuy:
    """One buy trade in the window."""

    def test_buy_volume_equals_size(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 0.5, False),  # buyer aggressed
        ])
        feats = _extract(trades)

        assert feats["buy_volume"] == 0.5
        assert feats["sell_volume"] == 0.0
        assert feats["net_volume"] == 0.5

    def test_buy_sell_ratio_large_but_finite(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 2.0, False),
        ])
        feats = _extract(trades)

        assert feats["buy_sell_ratio"] > 1e8  # huge
        assert not math.isinf(feats["buy_sell_ratio"])
        assert not math.isnan(feats["buy_sell_ratio"])

    def test_delta_5_equals_net_volume(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 1.0, False),
        ])
        feats = _extract(trades)

        assert feats["delta_5"] == feats["net_volume"] == 1.0
        assert feats["delta_1"] == 1.0  # only trade is within first 20%


class TestSingleSell:
    """One sell trade in the window."""

    def test_sell_volume_equals_size(self) -> None:
        trades = _trades_df([
            (10_000, 68400.0, 0.3, True),  # seller aggressed
        ])
        feats = _extract(trades)

        assert feats["sell_volume"] == 0.3
        assert feats["buy_volume"] == 0.0
        assert feats["net_volume"] == -0.3

    def test_buy_sell_ratio_is_zero(self) -> None:
        trades = _trades_df([
            (10_000, 68400.0, 0.3, True),
        ])
        feats = _extract(trades)
        assert feats["buy_sell_ratio"] == 0.0


class TestMixedTrades:
    """Both buys and sells in the window."""

    def test_net_volume_correct(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 1.0, False),  # buy  +1.0
            (20_000, 68510.0, 0.3, True),   # sell −0.3
            (30_000, 68490.0, 0.5, False),  # buy  +0.5
        ])
        feats = _extract(trades)

        assert feats["buy_volume"] == 1.5
        assert feats["sell_volume"] == 0.3
        assert feats["net_volume"] == 1.2

    def test_buy_sell_ratio(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 2.0, False),
            (20_000, 68510.0, 1.0, True),
        ])
        feats = _extract(trades)
        assert feats["buy_sell_ratio"] == pytest.approx(2.0, rel=1e-6)

    def test_volume_vs_avg(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 100.0, False),
            (20_000, 68510.0, 50.0, True),
        ])
        feats = _extract(trades, rolling_avg=300.0)
        assert feats["volume_vs_avg"] == pytest.approx(150.0 / (300.0 + 1e-9), rel=1e-6)

    def test_large_trade_net(self) -> None:
        """Avg size = (5+0.5+0.5)/3 = 2.0, threshold = 4.0 → only the 5.0 trade qualifies."""
        trades = _trades_df([
            (10_000, 68500.0, 5.0, False),   # buy, large  (5.0 > 4.0)
            (20_000, 68510.0, 0.5, True),    # sell, small
            (30_000, 68490.0, 0.5, False),   # buy, small
        ])
        feats = _extract(trades)
        # Only the 5.0 buy qualifies → net = +5.0
        assert feats["large_trade_net"] == 5.0

    def test_acceleration(self) -> None:
        """First half net vs second half net."""
        trades = _trades_df([
            (10_000, 68500.0, 1.0, False),   # buy  → first half
            (20_000, 68510.0, 0.5, True),    # sell → first half
            (35_000, 68490.0, 2.0, False),   # buy  → second half
            (45_000, 68520.0, 0.3, False),   # buy  → second half
        ])
        feats = _extract(trades)
        # First half  (0–30K): +1.0 − 0.5 = +0.5
        # Second half (30K–60K): +2.0 + 0.3 = +2.3
        # acceleration = 2.3 − 0.5 = 1.8
        assert feats["acceleration"] == pytest.approx(1.8, rel=1e-9)


class TestDeltaCurve:
    """Cumulative net volume at 20/40/60/80/100 % marks."""

    def test_delta_5_equals_net_volume(self) -> None:
        trades = _trades_df([
            (5_000,  68500.0, 0.5, False),
            (15_000, 68510.0, 0.2, True),
            (25_000, 68490.0, 1.0, False),
            (55_000, 68520.0, 0.1, True),
        ])
        feats = _extract(trades)
        # net = 0.5 − 0.2 + 1.0 − 0.1 = 1.2
        assert feats["net_volume"] == 1.2
        assert feats["delta_5"] == 1.2

    def test_delta_boundaries(self) -> None:
        """Exact placement at 20/40/60/80 % marks."""
        # Window 0–60000 → boundaries at 12K, 24K, 36K, 48K
        trades = _trades_df([
            (6_000,  68500.0, 1.0, False),   # +1.0  (0–20%)
            (12_000, 68510.0, 0.5, True),    # −0.5  (at 20% boundary)
            (24_000, 68490.0, 2.0, False),   # +2.0  (at 40% boundary)
            (36_000, 68520.0, 1.0, True),    # −1.0  (at 60% boundary)
            (48_000, 68500.0, 0.5, False),   # +0.5  (at 80% boundary)
            (59_000, 68530.0, 0.3, False),   # +0.3  (80–100%)
        ])
        feats = _extract(trades)

        # delta_1 (≤20%): +1.0 − 0.5 = 0.5
        assert feats["delta_1"] == 0.5
        # delta_2 (≤40%): 0.5 + 2.0 = 2.5
        assert feats["delta_2"] == 2.5
        # delta_3 (≤60%): 2.5 − 1.0 = 1.5
        assert feats["delta_3"] == 1.5
        # delta_4 (≤80%): 1.5 + 0.5 = 2.0
        assert feats["delta_4"] == 2.0
        # delta_5 (≤100%): 2.0 + 0.3 = 2.3
        assert feats["delta_5"] == 2.3


class TestContextFeatures:
    """Group D features."""

    def test_hour_cyclicals(self) -> None:
        """window_end_ms at 14:00 UTC → hour=14."""
        # 14 * 3_600_000 = 50_400_000
        feats = _extract(_trades_df([]), window_end_ms=50_400_000)
        assert feats["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 14 / 24), rel=1e-9)
        assert feats["hour_cos"] == pytest.approx(math.cos(2 * math.pi * 14 / 24), rel=1e-9)

    def test_trend_slope(self) -> None:
        trades = _trades_df([
            (10_000, 68000.0, 1.0, False),
            (20_000, 68100.0, 0.5, False),
        ])
        feats = _extract(trades)
        expected = (68100.0 - 68000.0) / (68000.0 + 1e-9)
        assert feats["trend_slope"] == pytest.approx(expected, rel=1e-9)
