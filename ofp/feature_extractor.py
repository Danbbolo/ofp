"""
feature_extractor.py — Extract 28 features from a trading window.

Consumes validated trades, L2 book snapshots, and liquidation DataFrames.
No models, no indicators — just deterministic feature arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

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
    # Filter to window
    # ------------------------------------------------------------------
    mask = (trades_df["timestamp_ms"] >= window_start_ms) & (
        trades_df["timestamp_ms"] < window_end_ms
    )
    win = trades_df.loc[mask].copy()

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
    # Group B — The Defence (Book Depth)     [keys 13–18]  RESERVED
    # ------------------------------------------------------------------
    b_bid_imbalance = 0.0
    b_ask_imbalance = 0.0
    b_spread = 0.0
    b_bid_depth_change = 0.0
    b_ask_depth_change = 0.0
    b_wall_pressure = 0.0

    # ------------------------------------------------------------------
    # Group C — The Wreckage (Liquidations)  [keys 19–23]  RESERVED
    # ------------------------------------------------------------------
    c_liq_net = 0.0
    c_long_liq_volume = 0.0
    c_short_liq_volume = 0.0
    c_liq_intensity = 0.0
    c_liq_price_deviation = 0.0

    # ------------------------------------------------------------------
    # Group D — Context                      [keys 24–28]
    # ------------------------------------------------------------------

    # 24–25.  Hour cyclicals
    hour = (window_end_ms // 3_600_000) % 24
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    # 26.  vol_ratio  (window range / 24h average range)
    if n_trades > 0:
        max_price = float(win["price"].max())
        min_price = float(win["price"].min())
        vol_ratio = (max_price - min_price) / (_24h_avg_range + 1e-9)
    else:
        vol_ratio = 0.0

    # 27.  price_position  ((current − 24h_low) / (24h_high − 24h_low))
    price_position = (current_price - _24h_low) / (_24h_high - _24h_low + 1e-9)

    # 28.  trend_slope  ((last − first) / first)
    if n_trades > 0:
        first_price = float(win["price"].iloc[0])
        last_price = float(win["price"].iloc[-1])
        trend_slope = (last_price - first_price) / (first_price + 1e-9)
    else:
        trend_slope = 0.0

    # ------------------------------------------------------------------
    # Assemble exactly 28 keys
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
        # Group B  (13–18)  reserved
        "b_bid_imbalance":     b_bid_imbalance,
        "b_ask_imbalance":     b_ask_imbalance,
        "b_spread":            b_spread,
        "b_bid_depth_change":  b_bid_depth_change,
        "b_ask_depth_change":  b_ask_depth_change,
        "b_wall_pressure":     b_wall_pressure,
        # Group C  (19–23)  reserved
        "c_liq_net":           c_liq_net,
        "c_long_liq_volume":   c_long_liq_volume,
        "c_short_liq_volume":  c_short_liq_volume,
        "c_liq_intensity":     c_liq_intensity,
        "c_liq_price_deviation": c_liq_price_deviation,
        # Group D  (24–28)
        "hour_sin":            hour_sin,
        "hour_cos":            hour_cos,
        "vol_ratio":           vol_ratio,
        "price_position":      price_position,
        "trend_slope":         trend_slope,
    }
