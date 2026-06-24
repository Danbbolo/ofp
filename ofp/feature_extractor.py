"""
feature_extractor.py — Extract 28 features from a trading window.

Consumes validated trades, L2 book snapshots, and liquidation DataFrames.
No models, no indicators — just deterministic feature arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def extract_features(
    trades_df: pd.DataFrame,
    book_snapshot_start: tuple[list[tuple[float, float]], list[tuple[float, float]]],
    book_snapshot_end: tuple[list[tuple[float, float]], list[tuple[float, float]]],
    liq_df: pd.DataFrame,
    window_start_ms: int,
    window_end_ms: int,
    rolling_avg_volume: float,
    *,
    _24h_avg_range: float = 0.0,
    _24h_low: float = 0.0,
    _24h_high: float = 0.0,
    current_price: float = 0.0,
) -> dict[str, float]:
    """
    Extract exactly 28 features from one time window.

    Parameters
    ----------
    trades_df : DataFrame
        Columns: ``timestamp_ms``, ``price``, ``size``, ``is_buyer_maker``.
    book_snapshot_start : (bids, asks)
        Each is ``[(price, size), ...]`` — top-20 book at window start.
    book_snapshot_end : (bids, asks)
        Top-20 book at window end.
    liq_df : DataFrame
        Columns: ``timestamp_ms``, ``side``, ``price``, ``size``.
    window_start_ms : int
    window_end_ms : int
    rolling_avg_volume : float
        Average volume over a larger lookback for normalisation.
    _24h_avg_range, _24h_low, _24h_high, current_price : float
        Contextual 24h stats (default 0.0 — pass real values when available).

    Returns
    -------
    dict[str, float]
        28 keys: ``buy_volume`` through ``trend_slope``.
        Groups B (book) and C (liquidation) are reserved at 0.0.
    """
    # ------------------------------------------------------------------
    # trades_df is assumed pre-sliced to [window_start_ms, window_end_ms)
    # by the caller (GridSweeper).  No internal filtering here.
    # ------------------------------------------------------------------
    win = trades_df.copy()

    # Pre-compute signed size: + for buys (aggressor=BUYER → is_buyer_maker=False),
    #                          − for sells (aggressor=SELLER → is_buyer_maker=True)
    win["signed_size"] = win["size"].where(~win["is_buyer_maker"], -win["size"])

    # ------------------------------------------------------------------
    # Group A — The Attack (Market Trades)  [keys  1–12]
    # ------------------------------------------------------------------

    # 1. buy_volume
    buy_volume = float(win.loc[~win["is_buyer_maker"], "size"].sum())

    # 2. sell_volume
    sell_volume = float(win.loc[win["is_buyer_maker"], "size"].sum())

    # 3. net_volume
    net_volume = buy_volume - sell_volume

    # 4. buy_sell_ratio
    buy_sell_ratio = buy_volume / (sell_volume + 1e-9)

    # 5. volume_vs_avg
    total_volume = buy_volume + sell_volume
    volume_vs_avg = total_volume / (rolling_avg_volume + 1e-9)

    # 6. large_trade_net
    n_trades = len(win)
    if n_trades > 0:
        avg_trade_size = total_volume / n_trades
        threshold = 2.0 * avg_trade_size
        large_mask = win["size"] > threshold
        large_trade_net = float(win.loc[large_mask, "signed_size"].sum())
    else:
        large_trade_net = 0.0

    # 7. acceleration  (second half net − first half net)
    if n_trades > 0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        first_half = win.loc[win["timestamp_ms"] < mid_ms, "signed_size"].sum()
        second_half = win.loc[win["timestamp_ms"] >= mid_ms, "signed_size"].sum()
        acceleration = float(second_half - first_half)
    else:
        acceleration = 0.0

    # 8–12.  Delta curve  (cumulative net volume at 20/40/60/80/100 % marks)
    window_dur = window_end_ms - window_start_ms

    if n_trades > 0 and window_dur > 0:
        win_sorted = win.sort_values("timestamp_ms")
        cum_net = win_sorted["signed_size"].cumsum().values

        # Compute fractional position [0, 1] for each trade within the window
        frac = (win_sorted["timestamp_ms"].values - window_start_ms) / window_dur

        def _cum_at(limit: float) -> float:
            """Last cumulative value where frac <= limit, else 0.0."""
            idx = -1
            for i, f in enumerate(frac):
                if f <= limit:
                    idx = i
                else:
                    break
            if idx >= 0:
                return float(cum_net[idx])
            return 0.0

        delta_1 = _cum_at(0.20)
        delta_2 = _cum_at(0.40)
        delta_3 = _cum_at(0.60)
        delta_4 = _cum_at(0.80)
        delta_5 = _cum_at(1.00)
    else:
        delta_1 = delta_2 = delta_3 = delta_4 = delta_5 = 0.0

    # ------------------------------------------------------------------
    # Group B — The Defence (Book Depth)     [keys 13–20]
    # ------------------------------------------------------------------
    bids_end, asks_end = book_snapshot_end
    bids_start, asks_start = book_snapshot_start

    # 13.  bid_ask_imbalance  (top-5 total bid / top-5 total ask, end snapshot)
    bid_ask_imbalance = _bid_ask_imbalance(bids_end, asks_end, n=5)

    # 14.  bid_wall  (largest single bid size in top 5)
    bid_wall = _max_size(bids_end, n=5)

    # 15.  ask_wall  (largest single ask size in top 5)
    ask_wall = _max_size(asks_end, n=5)

    # 16.  wall_asymmetry
    wall_asymmetry = bid_wall / (ask_wall + 1e-9)

    # 17.  depth_trend  (start imbalance − end imbalance)
    depth_trend = _bid_ask_imbalance(bids_start, asks_start, n=5) - bid_ask_imbalance

    # 18.  spread_bps  (end snapshot)
    spread_bps = _spread_bps(bids_end, asks_end)

    # 19.  spread_change  (end − start)
    spread_change = spread_bps - _spread_bps(bids_start, asks_start)

    # 20.  book_depth_slope  (linear slope of cumulative combined depth, top 5)
    book_depth_slope = _depth_slope(bids_end, asks_end, n=5)

    # ------------------------------------------------------------------
    # Group C — The Forced Errors (Liquidations)  [keys 21–25]
    # (liq_df is assumed pre-sliced by the caller)
    # ------------------------------------------------------------------
    liq_win = liq_df

    # 21.  long_liq_vol  (side == "SELL" → long was liquidated)
    long_liq_vol = float(liq_win.loc[liq_win["side"] == "SELL", "size"].sum())

    # 22.  short_liq_vol  (side == "BUY" → short was liquidated)
    short_liq_vol = float(liq_win.loc[liq_win["side"] == "BUY", "size"].sum())

    # 23.  net_liq  (short − long)
    net_liq = short_liq_vol - long_liq_vol

    # 24.  liq_climax  (total liq / total trade volume)
    total_liq_vol = long_liq_vol + short_liq_vol
    liq_climax = total_liq_vol / (total_volume + 1e-9)

    # 25.  liq_timing  (1 if >70 % of liq vol in second half, else 0)
    if total_liq_vol > 0.0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        second_half_liq = float(
            liq_win.loc[liq_win["timestamp_ms"] >= mid_ms, "size"].sum()
        )
        liq_timing = 1.0 if (second_half_liq / total_liq_vol) > 0.70 else 0.0
    else:
        liq_timing = 0.0

    # ------------------------------------------------------------------
    # Group D — Context                      [keys 26–30]
    # ------------------------------------------------------------------

    # 26–27.  Hour cyclicals
    hour = (window_end_ms // 3_600_000) % 24
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    # 28.  vol_ratio  (window range / 24h average range)
    if n_trades > 0:
        max_price = float(win["price"].max())
        min_price = float(win["price"].min())
        vol_ratio = (max_price - min_price) / (_24h_avg_range + 1e-9)
    else:
        vol_ratio = 0.0

    # 29.  price_position  ((current − 24h_low) / (24h_high − 24h_low))
    price_position = (current_price - _24h_low) / (_24h_high - _24h_low + 1e-9)

    # 30.  trend_slope  ((last − first) / first)
    if n_trades > 0:
        first_price = float(win["price"].iloc[0])
        last_price = float(win["price"].iloc[-1])
        trend_slope = (last_price - first_price) / (first_price + 1e-9)
    else:
        trend_slope = 0.0

    # ------------------------------------------------------------------
    # Assemble exactly 30 keys
    # ------------------------------------------------------------------
    return {
        # Group A  (1–12)
        "buy_volume":          buy_volume,
        "sell_volume":         sell_volume,
        "net_volume":          net_volume,
        "buy_sell_ratio":      buy_sell_ratio,
        "volume_vs_avg":       volume_vs_avg,
        "large_trade_net":     large_trade_net,
        "acceleration":        acceleration,
        "delta_1":             delta_1,
        "delta_2":             delta_2,
        "delta_3":             delta_3,
        "delta_4":             delta_4,
        "delta_5":             delta_5,
        # Group B  (13–20)
        "bid_ask_imbalance":   bid_ask_imbalance,
        "bid_wall":            bid_wall,
        "ask_wall":            ask_wall,
        "wall_asymmetry":      wall_asymmetry,
        "depth_trend":         depth_trend,
        "spread_bps":          spread_bps,
        "spread_change":       spread_change,
        "book_depth_slope":    book_depth_slope,
        # Group C  (21–25)
        "long_liq_vol":        long_liq_vol,
        "short_liq_vol":       short_liq_vol,
        "net_liq":             net_liq,
        "liq_climax":          liq_climax,
        "liq_timing":           liq_timing,
        # Group D  (26–30)
        "hour_sin":            hour_sin,
        "hour_cos":            hour_cos,
        "vol_ratio":           vol_ratio,
        "price_position":      price_position,
        "trend_slope":         trend_slope,
    }


# ---------------------------------------------------------------------------
# Book depth helpers
# ---------------------------------------------------------------------------

def _top_sizes(
    levels: list[tuple[float, float]], n: int
) -> list[float]:
    """Return the *size* of the first min(n, len(levels)) entries."""
    return [sz for _, sz in levels[:n]]


def _max_size(levels: list[tuple[float, float]], n: int) -> float:
    """Largest size among the top *n* levels (0.0 if empty)."""
    sizes = _top_sizes(levels, n)
    return max(sizes) if sizes else 0.0


def _bid_ask_imbalance(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    n: int = 5,
) -> float:
    """Total top-N bid size / total top-N ask size."""
    total_bid = sum(_top_sizes(bids, n))
    total_ask = sum(_top_sizes(asks, n))
    return total_bid / (total_ask + 1e-9)


def _spread_bps(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> float:
    """(best_ask - best_bid) / mid * 10000.  0.0 if either side is empty."""
    if not bids or not asks:
        return 0.0
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    return (best_ask - best_bid) / (mid + 1e-9) * 10000.0


def _depth_slope(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    n: int = 5,
) -> float:
    """
    Linear slope of cumulative combined depth across levels 0..n-1.

    x = [0, 1, ..., n-1]
    y[i] = sum(combined_depth[j] for j=0..i)

    Returns 0.0 if fewer than 2 levels exist on both sides combined.
    """
    bid_sizes = _top_sizes(bids, n)
    ask_sizes = _top_sizes(asks, n)
    max_len = max(len(bid_sizes), len(ask_sizes))
    if max_len < 2:
        return 0.0

    # Pad shorter side with zeros
    combined = [
        (bid_sizes[i] if i < len(bid_sizes) else 0.0)
        + (ask_sizes[i] if i < len(ask_sizes) else 0.0)
        for i in range(max_len)
    ]
    cumulative = np.cumsum(combined, dtype=np.float64)
    x = np.arange(len(cumulative), dtype=np.float64)
    slope, _ = np.polyfit(x, cumulative, deg=1)
    return float(slope)
