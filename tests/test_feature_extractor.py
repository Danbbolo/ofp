"""
test_feature_extractor.py — Tests for extract_features().

Covers: empty window, single buy, single sell, mixed trades, delta curve.
"""

from __future__ import annotations

import math

import numpy as np
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


def _book(bids=None, asks=None) -> tuple[list, list]:
    return (bids or [], asks or [])


def _liq_df(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (timestamp_ms, side, price, size)."""
    return pd.DataFrame(rows, columns=["timestamp_ms", "side", "price", "size"])


# Shorthand for the long call  (pre-slices DataFrames as GridSweeper does)
def _extract(
    trades,
    rolling_avg: float = 1000.0,
    window_start_ms: int = 0,
    window_end_ms: int = 60_000,
    book_start: tuple | None = None,
    book_end: tuple | None = None,
    liq: pd.DataFrame | None = None,
    **kwargs,
):
    # Pre-slice trades to window
    ts = trades["timestamp_ms"].values
    lo = int(np.searchsorted(ts, window_start_ms, side="left"))
    hi = int(np.searchsorted(ts, window_end_ms, side="left"))
    trades_win = trades.iloc[lo:hi]

    # Pre-slice liq
    liq_df = liq if liq is not None else _empty_liq_df()
    if not liq_df.empty:
        lq_ts = liq_df["timestamp_ms"].values
        lq_lo = int(np.searchsorted(lq_ts, window_start_ms, side="left"))
        lq_hi = int(np.searchsorted(lq_ts, window_end_ms, side="left"))
        liq_df = liq_df.iloc[lq_lo:lq_hi]

    return extract_features(
        trades_df=trades_win,
        book_snapshot_start=book_start if book_start is not None else _empty_book(),
        book_snapshot_end=book_end if book_end is not None else _empty_book(),
        liq_df=liq_df,
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

    def test_exactly_30_keys(self) -> None:
        trades = _trades_df([])
        feats = _extract(trades)
        assert len(feats) == 30


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


# ---------------------------------------------------------------------------
# Group B — Order Book Depth
# ---------------------------------------------------------------------------

class TestBookDepth:
    """bid_ask_imbalance, wall detection, spread, depth_trend."""

    @staticmethod
    def _simple_book() -> tuple[list, list]:
        bids = [(68500.0, 1.0), (68490.0, 2.0), (68480.0, 0.5), (68470.0, 3.0), (68460.0, 1.5)]
        asks = [(68600.0, 2.0), (68610.0, 1.0), (68620.0, 0.5), (68630.0, 4.0), (68640.0, 2.5)]
        return (bids, asks)

    def test_bid_ask_imbalance(self) -> None:
        book = self._simple_book()
        feats = _extract(_trades_df([]), book_end=book)
        # total bids top-5 = 1+2+0.5+3+1.5 = 8.0
        # total asks top-5 = 2+1+0.5+4+2.5 = 10.0
        # imbalance = 8/10 = 0.8
        assert feats["bid_ask_imbalance"] == pytest.approx(0.8, rel=1e-9)

    def test_bid_wall(self) -> None:
        book = self._simple_book()
        feats = _extract(_trades_df([]), book_end=book)
        # largest bid in top 5 = 3.0 at 68470
        assert feats["bid_wall"] == 3.0

    def test_ask_wall(self) -> None:
        book = self._simple_book()
        feats = _extract(_trades_df([]), book_end=book)
        # largest ask in top 5 = 4.0 at 68630
        assert feats["ask_wall"] == 4.0

    def test_wall_asymmetry(self) -> None:
        book = self._simple_book()
        feats = _extract(_trades_df([]), book_end=book)
        assert feats["wall_asymmetry"] == pytest.approx(3.0 / 4.0, rel=1e-9)

    def test_depth_trend(self) -> None:
        """Start imbalance > end imbalance → depth_trend positive."""
        start = _book(
            bids=[(68500.0, 10.0), (68490.0, 1.0)],
            asks=[(68600.0, 2.0), (68610.0, 1.0)],
        )  # imbalance = 11/3 ≈ 3.667
        end = _book(
            bids=[(68500.0, 2.0), (68490.0, 1.0)],
            asks=[(68600.0, 8.0), (68610.0, 2.0)],
        )  # imbalance = 3/10 = 0.3
        feats = _extract(_trades_df([]), book_start=start, book_end=end)
        # depth_trend = start_imbalance - end_imbalance ≈ 3.667 - 0.3
        assert feats["depth_trend"] == pytest.approx(11.0 / 3.0 - 3.0 / 10.0, rel=1e-9)
        assert feats["depth_trend"] > 0.0

    def test_spread_bps(self) -> None:
        book = _book(
            bids=[(68500.0, 1.0)],
            asks=[(68510.0, 1.0)],
        )
        feats = _extract(_trades_df([]), book_end=book)
        # best_bid=68500, best_ask=68510, mid=68505
        # spread_bps = (10/68505)*10000 ≈ 1.459
        expected = (10.0 / 68505.0) * 10000.0
        assert feats["spread_bps"] == pytest.approx(expected, rel=1e-6)

    def test_spread_change(self) -> None:
        start = _book(bids=[(68500.0, 1.0)], asks=[(68510.0, 1.0)])  # narrow
        end = _book(bids=[(68500.0, 1.0)], asks=[(68520.0, 1.0)])    # wider
        feats = _extract(_trades_df([]), book_start=start, book_end=end)
        # spread widened → change > 0
        assert feats["spread_change"] > 0.0

    def test_book_depth_slope(self) -> None:
        """Cumulative combined depth: [2, 5, 8, 12, 17] → positive slope."""
        book = _book(
            bids=[(68500.0, 1.0), (68490.0, 2.0), (68480.0, 3.0), (68470.0, 4.0), (68460.0, 5.0)],
            asks=[(68600.0, 1.0), (68610.0, 1.0), (68620.0, 0.0), (68630.0, 0.0), (68640.0, 0.0)],
        )
        feats = _extract(_trades_df([]), book_end=book)
        # combined per level: [2, 3, 3, 4, 5]
        # cumulative:        [2, 5, 8, 12, 17]
        # slope > 0
        assert feats["book_depth_slope"] > 0.0

    def test_empty_book_all_zeros(self) -> None:
        feats = _extract(_trades_df([]), book_end=_book())
        assert feats["bid_ask_imbalance"] == 0.0
        assert feats["bid_wall"] == 0.0
        assert feats["ask_wall"] == 0.0
        assert feats["wall_asymmetry"] == 0.0
        assert feats["spread_bps"] == 0.0
        assert feats["book_depth_slope"] == 0.0

    def test_spread_empty_book_zero(self) -> None:
        feats = _extract(_trades_df([]), book_end=_book(bids=[(68500.0, 1.0)], asks=[]))
        assert feats["spread_bps"] == 0.0

    def test_depth_slope_fewer_than_two_levels_zero(self) -> None:
        feats = _extract(_trades_df([]), book_end=_book(
            bids=[(68500.0, 1.0)],
            asks=[],
        ))
        assert feats["book_depth_slope"] == 0.0


# ---------------------------------------------------------------------------
# Group C — Liquidations
# ---------------------------------------------------------------------------

class TestLiquidations:
    """long_liq_vol, short_liq_vol, net_liq, liq_climax, liq_timing."""

    def test_long_liq_vol(self) -> None:
        liq = _liq_df([
            (10_000, "SELL", 68000.0, 1.5),  # long liquidated
            (20_000, "SELL", 67900.0, 0.5),  # long liquidated
        ])
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["long_liq_vol"] == 2.0
        assert feats["short_liq_vol"] == 0.0

    def test_short_liq_vol(self) -> None:
        liq = _liq_df([
            (10_000, "BUY", 68200.0, 0.8),  # short liquidated
            (25_000, "BUY", 68300.0, 0.2),  # short liquidated
        ])
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["short_liq_vol"] == 1.0
        assert feats["long_liq_vol"] == 0.0

    def test_net_liq(self) -> None:
        liq = _liq_df([
            (10_000, "SELL", 68000.0, 3.0),
            (20_000, "BUY",  68200.0, 1.0),
            (30_000, "BUY",  68300.0, 0.5),
        ])
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["long_liq_vol"] == 3.0
        assert feats["short_liq_vol"] == 1.5
        assert feats["net_liq"] == -1.5  # short − long = 1.5 − 3.0

    def test_liq_climax(self) -> None:
        trades = _trades_df([
            (10_000, 68500.0, 5.0, False),
            (20_000, 68400.0, 3.0, True),
        ])  # total trade volume = 8.0
        liq = _liq_df([
            (15_000, "SELL", 68000.0, 2.0),
        ])  # total liq = 2.0
        feats = _extract(trades, liq=liq)
        assert feats["liq_climax"] == pytest.approx(2.0 / (8.0 + 1e-9), rel=1e-9)

    def test_liq_timing_early_returns_zero(self) -> None:
        """All liquidations in first half → liq_timing = 0."""
        liq = _liq_df([
            (5_000, "SELL", 68000.0, 1.0),
            (29_000, "BUY", 68200.0, 0.5),
        ])  # both < 30_000 (mid of 0–60_000)
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["liq_timing"] == 0.0

    def test_liq_timing_late_returns_one(self) -> None:
        """>70 % of liq vol in second half → liq_timing = 1."""
        liq = _liq_df([
            (10_000, "SELL", 68000.0, 1.0),   # first half
            (40_000, "SELL", 67800.0, 5.0),   # second half
        ])  # total=6, second=5, ratio=5/6 ≈ 0.833 > 0.70
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["liq_timing"] == 1.0

    def test_liq_timing_no_liquidations_zero(self) -> None:
        feats = _extract(_trades_df([]))
        assert feats["liq_timing"] == 0.0
        assert feats["long_liq_vol"] == 0.0
        assert feats["short_liq_vol"] == 0.0
        assert feats["net_liq"] == 0.0
        assert feats["liq_climax"] == 0.0

    def test_liq_filtered_by_window(self) -> None:
        """Liquidations outside the window are ignored."""
        liq = _liq_df([
            (70_000, "SELL", 68000.0, 100.0),  # outside (>= 60_000)
        ])
        feats = _extract(_trades_df([]), liq=liq)
        assert feats["long_liq_vol"] == 0.0
