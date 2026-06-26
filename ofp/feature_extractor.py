"""
feature_extractor.py - Extract 36 features from a trading window.

Consumes validated trades, L2 book snapshots, and liquidation DataFrames.
No models, no indicators - just deterministic feature arithmetic.
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
    Extract exactly 36 features from one time window.

    Parameters
    ----------
    trades_df : DataFrame
        Columns: ``timestamp_ms``, ``price``, ``size``, ``is_buyer_maker``.
    book_snapshot_start : (bids, asks)
        Each is ``[(price, size), ...]`` - top-20 book at window start.
    book_snapshot_end : (bids, asks)
        Top-20 book at window end.
    liq_df : DataFrame
        Columns: ``timestamp_ms``, ``side``, ``price``, ``size``.
    window_start_ms : int
    window_end_ms : int
    rolling_avg_volume : float
        Average volume over a larger lookback for normalisation.
    _24h_avg_range, _24h_low, _24h_high, current_price : float
        Contextual 24h stats (default 0.0; pass real values when available).

    Returns
    -------
    dict[str, float]
        36 keys: ``buy_volume`` through ``volume_profile_entropy``.
        Groups B (book) and C (liquidation) are reserved at 0.0.
    """
    # ------------------------------------------------------------------
    # trades_df is assumed pre-sliced to [window_start_ms, window_end_ms)
    # by the caller (GridSweeper).  No internal filtering, no copy.
    # Work on numpy views to avoid per-window DataFrame allocation.
    # ------------------------------------------------------------------
    trade_ts = trades_df["timestamp_ms"].values
    trade_px = trades_df["price"].values
    trade_sz = trades_df["size"].values
    # Force bool dtype: source columns may be object (Python bool) and
    # numpy refuses to use object arrays as boolean indices.
    trade_bm = trades_df["is_buyer_maker"].astype(bool).values

    n_trades = len(trade_ts)
    # Pre-compute signed size: + for buys (aggressor=BUYER → bm=False),
    #                          − for sells (aggressor=SELLER → bm=True)
    signed_size = np.where(trade_bm, -trade_sz, trade_sz)
    buy_mask = ~trade_bm
    sell_mask = trade_bm

    # ------------------------------------------------------------------
    # Group A - The Attack (Market Trades)  [keys  1-12]
    # ------------------------------------------------------------------

    # 1. buy_volume
    buy_volume = float(trade_sz[buy_mask].sum())

    # 2. sell_volume
    sell_volume = float(trade_sz[sell_mask].sum())

    # 3. net_volume
    net_volume = buy_volume - sell_volume

    # 4. buy_sell_ratio
    buy_sell_ratio = buy_volume / (sell_volume + 1e-9)

    # 5. volume_vs_avg
    total_volume = buy_volume + sell_volume
    volume_vs_avg = total_volume / (rolling_avg_volume + 1e-9)

    # 6. large_trade_net
    if n_trades > 0:
        avg_trade_size = total_volume / n_trades
        threshold = 2.0 * avg_trade_size
        large_mask = trade_sz > threshold
        large_trade_net = float(signed_size[large_mask].sum())
    else:
        large_trade_net = 0.0

    # 7. acceleration  (second half net − first half net)
    if n_trades > 0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        first_half = float(signed_size[trade_ts < mid_ms].sum())
        second_half = float(signed_size[trade_ts >= mid_ms].sum())
        acceleration = second_half - first_half
    else:
        acceleration = 0.0

    # 8-12.  Delta curve  (cumulative net volume at 20/40/60/80/100 % marks)
    window_dur = window_end_ms - window_start_ms

    if n_trades > 0 and window_dur > 0:
        # Sort by timestamp once, get the permutation indices.
        order = np.argsort(trade_ts, kind="stable")
        sorted_ts = trade_ts[order]
        sorted_signed = signed_size[order]
        cum_net = np.cumsum(sorted_signed)

        # Compute fractional position [0, 1] for each trade within the window
        frac = (sorted_ts - window_start_ms) / window_dur

        # Vectorized: index of last frac <= limit  (O(log n) per query)
        def _cum_at(limit: float) -> float:
            idx = int(np.searchsorted(frac, limit, side="right")) - 1
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
    # Group B - The Defence (Book Depth)     [keys 13-20]
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
    # Group C - The Forced Errors (Liquidations)  [keys 21-25]
    # (liq_df is assumed pre-sliced by the caller)
    # ------------------------------------------------------------------
    liq_win = liq_df

    # 21.  long_liq_vol  (side == "sell" → long was liquidated)
    long_liq_vol = float(liq_win.loc[liq_win["side"].str.lower() == "sell", "size"].sum())

    # 22.  short_liq_vol  (side == "buy" → short was liquidated)
    short_liq_vol = float(liq_win.loc[liq_win["side"].str.lower() == "buy", "size"].sum())

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
    # Group D - Context                      [keys 26-30]
    # ------------------------------------------------------------------

    # 26-27.  Hour cyclicals
    hour = (window_end_ms // 3_600_000) % 24
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    # 28.  vol_ratio  (window range / average range over the same window)
    # NOTE: using the window's own range as its own "average range" gives
    # vol_ratio = 1.0, which is uninformative.  Use the **previous** window
    # of the same size (if available) as the reference, or fall back to
    # the passed-in 24h baseline.  Computing a true rolling 24h average
    # requires streaming price history; for now use the passed-in
    # _24h_avg_range as a stable reference.
    if n_trades > 0:
        max_price = float(trade_px.max())
        min_price = float(trade_px.min())
        vol_ratio = (max_price - min_price) / (_24h_avg_range + 1e-9)
    else:
        vol_ratio = 0.0

    # 29.  price_position  ((current − local_low) / (local_high − local_low))
    # IMPORTANT: low/high must be LOCAL to this window, not a single global
    # 24h value, otherwise all three zoom levels see the exact same number
    # and the correlation between micro/meso/macro price_position = 1.0
    # (a context leak).  Use the window's own min/max.
    if n_trades > 0:
        local_low = float(trade_px.min())
        local_high = float(trade_px.max())
    else:
        local_low = _24h_low
        local_high = _24h_high
    price_position = (current_price - local_low) / (local_high - local_low + 1e-9)

    # 30.  trend_slope  ((last − first) / first)
    if n_trades > 0:
        first_price = float(trade_px[0])
        last_price = float(trade_px[-1])
        trend_slope = (last_price - first_price) / (first_price + 1e-9)
    else:
        trend_slope = 0.0

    # ------------------------------------------------------------------
    # Group E - Advanced Orderflow  [keys 31-36]
    # ------------------------------------------------------------------
    # 31.  cvd - Cumulative Volume Delta
    # Running sum of signed_size (buy_vol - sell_vol) across the window.
    # Positive = net buying pressure across the window.
    if n_trades > 0:
        cvd = float(signed_size.sum())
    else:
        cvd = 0.0

    # 32.  cvd_momentum - slope of CVD in 2nd half vs 1st half
    # Positive = flow is accelerating upward. Negative = flow drying up.
    if n_trades > 0 and window_dur > 0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        first_half_cvd = float(signed_size[trade_ts < mid_ms].sum())
        second_half_cvd = float(signed_size[trade_ts >= mid_ms].sum())
        # Normalize by total window volume so it's scale-independent
        cvd_momentum = (second_half_cvd - first_half_cvd) / (total_volume + 1e-9)
    else:
        cvd_momentum = 0.0

    # 33.  large_trade_count - whale detection
    # Count of trades where size > mean + 2*std. Right-tailed trade size
    # distribution = institutional footprint.
    if n_trades > 1:
        mean_size = float(trade_sz.mean())
        std_size = float(trade_sz.std())
        threshold = mean_size + 2.0 * std_size
        large_trade_count = int((trade_sz > threshold).sum())
    else:
        large_trade_count = 0

    # 34.  trade_size_skew - skewness of trade size distribution
    # Positive = right-skewed (few large trades among many small ones).
    if n_trades > 2:
        m = float(trade_sz.mean())
        s = float(trade_sz.std())
        if s > 0:
            n_f = float(n_trades)
            m3 = float(((trade_sz - m) ** 3).mean())
            trade_size_skew = m3 / (s ** 3) * (n_f * (n_f - 1)) ** 0.5 / (n_f - 2)
        else:
            trade_size_skew = 0.0
    else:
        trade_size_skew = 0.0

    # 35.  wall_lifecycle - track the largest bid/ask wall over the window.
    # Categories: 0=no wall, 1=appeared, 2=grew, 3=shrank, 4=cancelled.
    # We use top-5 book at start vs end. A "wall" is any level with size
    # >= 2x the median of top-5.
    def _find_wall(bids, asks):
        # Return (side, price, size) of largest wall, or (None, 0, 0)
        if not bids and not asks:
            return (None, 0.0, 0.0)
        best = (None, 0.0, 0.0)
        all_levels = [("bid", p, s) for p, s in bids] + [("ask", p, s) for p, s in asks]
        if not all_levels:
            return (None, 0.0, 0.0)
        sizes = [s for _, _, s in all_levels]
        med = float(np.median(sizes))
        if med <= 0:
            return (None, 0.0, 0.0)
        for side, p, s in all_levels:
            if s >= 2.0 * med and s > best[2]:
                best = (side, p, s)
        return best

    wall_start = _find_wall(bids_start, asks_start)
    wall_end = _find_wall(bids_end, asks_end)
    if wall_start[0] is None and wall_end[0] is None:
        wall_lifecycle = 0  # no wall in either
    elif wall_start[0] is None and wall_end[0] is not None:
        wall_lifecycle = 1  # appeared
    elif wall_start[0] is not None and wall_end[0] is None:
        wall_lifecycle = 4  # cancelled
    else:
        # both have a wall - check if same price or grew/shrank
        if wall_start[1] == wall_end[1]:
            if wall_end[2] > wall_start[2] * 1.1:
                wall_lifecycle = 2  # grew (>=10% larger)
            elif wall_end[2] < wall_start[2] * 0.9:
                wall_lifecycle = 3  # shrank
            else:
                wall_lifecycle = 2  # stable - count as grow (it persisted)
        else:
            wall_lifecycle = 1  # different level - appeared (and old gone)

    # 36.  volume_profile_entropy - Shannon entropy of volume distribution
    # across price bins. Low entropy = concentrated = S/R. High entropy = thin.
    if n_trades > 5 and len(bids_end) > 0 and len(asks_end) > 0:
        # Build a price range from local min/max
        pmin = float(trade_px.min())
        pmax = float(trade_px.max())
        if pmax > pmin:
            n_bins = 10
            # Vectorized bucketing — np.searchsorted + np.bincount replaces
            # the Python for-loop.  Was the main sweep bottleneck.
            edges = np.linspace(pmin, pmax, n_bins + 1)
            bin_idx = np.searchsorted(edges, trade_px, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            bin_vol = np.bincount(bin_idx, weights=trade_sz, minlength=n_bins)
            total = bin_vol.sum()
            if total > 0:
                p = bin_vol / total
                p = p[p > 0]
                entropy = float(-np.sum(p * np.log(p + 1e-12)))
                # Normalize to [0, 1] using log(n_bins) as max
                volume_profile_entropy = entropy / (np.log(n_bins) + 1e-12)
            else:
                volume_profile_entropy = 0.0
        else:
            volume_profile_entropy = 0.0
    else:
        volume_profile_entropy = 0.0

    # ------------------------------------------------------------------
    # Assemble exactly 36 keys
    # ------------------------------------------------------------------
    return {
        # Group A  (1-12)
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
        # Group B  (13-20)
        "bid_ask_imbalance":   bid_ask_imbalance,
        "bid_wall":            bid_wall,
        "ask_wall":            ask_wall,
        "wall_asymmetry":      wall_asymmetry,
        "depth_trend":         depth_trend,
        "spread_bps":          spread_bps,
        "spread_change":       spread_change,
        "book_depth_slope":    book_depth_slope,
        # Group C  (21-25)
        "long_liq_vol":        long_liq_vol,
        "short_liq_vol":       short_liq_vol,
        "net_liq":             net_liq,
        "liq_climax":          liq_climax,
        "liq_timing":           liq_timing,
        # Group D  (26-30)
        "hour_sin":            hour_sin,
        "hour_cos":            hour_cos,
        "vol_ratio":           vol_ratio,
        "price_position":      price_position,
        "trend_slope":         trend_slope,
        # Group E  (31-36) - advanced orderflow
        "cvd":                  cvd,
        "cvd_momentum":         cvd_momentum,
        "large_trade_count":    large_trade_count,
        "trade_size_skew":      trade_size_skew,
        "wall_lifecycle":       wall_lifecycle,
        "volume_profile_entropy": volume_profile_entropy,
    }


# ---------------------------------------------------------------------------
# Multi-Zoom feature extraction
# ---------------------------------------------------------------------------

def extract_multi_zoom_features(
    trades_df: pd.DataFrame,
    book_snapshots: dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]]]],
    liq_df: pd.DataFrame,
    micro_window_ms: int,
    meso_window_ms: int,
    macro_window_ms: int,
    end_time_ms: int,
    rolling_stats_per_zoom: dict[str, dict[str, float]] | None = None,
    current_price: float = 0.0,
    prior_24h_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """
    Extract features at 3 zoom levels sharing the same *end_time_ms*.

    Parameters
    ----------
    rolling_stats_per_zoom : dict
        ``{"micro": {...}, "meso": {...}, "macro": {...}}`` - each holds
        ``rolling_avg_volume`` (typical volume in a window of that zoom's
        size).  ``_24h_low`` / ``_24h_high`` / ``_24h_avg_range`` are
        computed LOCALLY per zoom from the prior 24h of trades (not a
        single global value - that would be a context leak).
    current_price : float
        Forwarded to every zoom (it's the same trade timestamp).
    prior_24h_cache : (sec_arr, low_arr, high_arr) | None
        Pre-computed per-second prior-24h stats.  Build this ONCE per
        sweep (not per row) - building it per row is O(n²) and
        kills performance.  If None, this function builds the cache
        itself (slow on large datasets; only used for unit tests).

    Returns
    -------
    dict[str, float]
        Keys prefixed ``micro_``, ``meso_``, ``macro_`` (36 × 3 = 108 total).
    """
    if rolling_stats_per_zoom is None:
        rolling_stats_per_zoom = {"micro": {}, "meso": {}, "macro": {}}

    # Book lookup helper
    snap_ts = np.array(sorted(book_snapshots.keys()), dtype=np.int64)

    def _book_at(ms: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if len(snap_ts) == 0:
            return ([], [])
        idx = int(np.searchsorted(snap_ts, ms, side="right")) - 1
        if idx < 0:
            return ([], [])
        return book_snapshots[int(snap_ts[idx])]

    result: dict[str, float] = {}

    # Pre-index timestamps for fast slicing (no DataFrame copy per zoom)
    trade_ts = trades_df["timestamp_ms"].values
    trade_px = trades_df["price"].values
    trade_sz = trades_df["size"].values
    trade_bm = trades_df["is_buyer_maker"].values
    liq_ts = liq_df["timestamp_ms"].values if len(liq_df) > 0 else None
    liq_sd = liq_df["side"].values if len(liq_df) > 0 else None
    liq_px = liq_df["price"].values if len(liq_df) > 0 else None
    liq_sz = liq_df["size"].values if len(liq_df) > 0 else None

    zooms = [("micro", micro_window_ms), ("meso", meso_window_ms), ("macro", macro_window_ms)]

    # ----------------------------------------------------------------
    # Pre-compute prior-24h rolling low/high/range ONCE per second.
    # ----------------------------------------------------------------
    # The prior 24h range for an arbitrary end_time_ms requires a slice
    # over the full prior 24h of trades (~3M trades).  Doing that
    # min/max scan per row is O(n²) over the whole sweep.  Instead we
    # build a cached lookup using a monotonic deque over per-second
    # price min/max, in O(n) total.
    #
    # IMPORTANT: if `prior_24h_cache` is passed in (the normal case
    # from `GridSweeper.sweep`), use it as-is.  Building per call is
    # O(n) and turns the sweep into O(n²).
    # ----------------------------------------------------------------
    if prior_24h_cache is not None:
        d24_sec_arr, d24_low_arr, d24_high_arr = prior_24h_cache
    elif len(trade_ts) == 0:
        d24_sec_arr = np.zeros(1, dtype=np.int64)
        d24_low_arr = np.zeros(1, dtype=np.float64)
        d24_high_arr = np.zeros(1, dtype=np.float64)
    else:
        # Bucket trades by their ms-timestamp floored to 1 second
        sec_ts = trade_ts // 1000
        unique_secs, inv = np.unique(sec_ts, return_inverse=True)
        n_secs = len(unique_secs)

        # Per-second min/max price
        sec_min = np.full(n_secs, np.inf)
        sec_max = np.full(n_secs, -np.inf)
        np.minimum.at(sec_min, inv, trade_px)
        np.maximum.at(sec_max, inv, trade_px)

        # Sliding-window 24h min/max using monotonic deques - O(n) total
        from collections import deque
        day_secs = 86_400
        d24_low = np.empty(n_secs, dtype=np.float64)
        d24_high = np.empty(n_secs, dtype=np.float64)
        min_q: deque[int] = deque()  # indices with strictly increasing sec_min
        max_q: deque[int] = deque()  # indices with strictly decreasing sec_max
        lo = 0
        for hi in range(n_secs):
            # Push hi onto min deque: pop while back has >= sec_min[hi]
            while min_q and sec_min[min_q[-1]] >= sec_min[hi]:
                min_q.pop()
            min_q.append(hi)
            # Push hi onto max deque: pop while back has <= sec_max[hi]
            while max_q and sec_max[max_q[-1]] <= sec_max[hi]:
                max_q.pop()
            max_q.append(hi)
            # Advance lo while the window is wider than 24h
            while unique_secs[hi] - unique_secs[lo] >= day_secs:
                lo += 1
                if min_q[0] < lo:
                    min_q.popleft()
                if max_q[0] < lo:
                    max_q.popleft()
            d24_low[hi] = sec_min[min_q[0]]
            d24_high[hi] = sec_max[max_q[0]]

        d24_sec_arr = unique_secs
        d24_low_arr = d24_low
        d24_high_arr = d24_high

    def _prior_24h_stats(lookback_end_ms: int) -> tuple[float, float, float]:
        """Return (low, high, range) for the 24h before lookback_end_ms."""
        lookback_sec = lookback_end_ms // 1000
        idx = int(np.searchsorted(d24_sec_arr, lookback_sec, side="right")) - 1
        if idx < 0 or idx >= len(d24_sec_arr):
            return 0.0, 0.0, 0.0
        lo = float(d24_low_arr[idx])
        hi = float(d24_high_arr[idx])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0 or lo <= 0:
            return 0.0, 0.0, 0.0
        return lo, hi, hi - lo

    for prefix, window_ms in zooms:
        win_start = end_time_ms - window_ms

        # Slice trades using numpy (zero-copy views into the underlying arrays)
        t_start = int(np.searchsorted(trade_ts, win_start, side="left"))
        t_end = int(np.searchsorted(trade_ts, end_time_ms, side="left"))
        sliced_trades = pd.DataFrame({
            "timestamp_ms": trade_ts[t_start:t_end],
            "price": trade_px[t_start:t_end],
            "size": trade_sz[t_start:t_end],
            "is_buyer_maker": trade_bm[t_start:t_end],
        })

        if liq_ts is not None and len(liq_df) > 0:
            l_start = int(np.searchsorted(liq_ts, win_start, side="left"))
            l_end = int(np.searchsorted(liq_ts, end_time_ms, side="left"))
            sliced_liq = pd.DataFrame({
                "timestamp_ms": liq_ts[l_start:l_end],
                "side": liq_sd[l_start:l_end],
                "price": liq_px[l_start:l_end],
                "size": liq_sz[l_start:l_end],
            })
        else:
            sliced_liq = liq_df

        # Per-zoom rolling stats - this is the fix for the context leak
        rs = rolling_stats_per_zoom.get(prefix, {})

        # Per-zoom prior-24h context - computed locally from the trades
        # array, NOT a single global value.  Without this, vol_ratio and
        # price_position are the same number across all 3 zooms, which
        # is a context leak (caught by verify_dataset).
        d24_low, d24_high, d24_range = _prior_24h_stats(end_time_ms)

        feats = extract_features(
            trades_df=sliced_trades,
            book_snapshot_start=_book_at(win_start),
            book_snapshot_end=_book_at(end_time_ms),
            liq_df=sliced_liq,
            window_start_ms=win_start,
            window_end_ms=end_time_ms,
            rolling_avg_volume=rs.get("rolling_avg_volume", 0.0),
            current_price=current_price,
            _24h_avg_range=d24_range if d24_range > 0 else rs.get("_24h_avg_range", 0.0),
            _24h_low=d24_low if d24_low > 0 else rs.get("_24h_low", 0.0),
            _24h_high=d24_high if d24_high > 0 else rs.get("_24h_high", 0.0),
        )
        for k, v in feats.items():
            result[f"{prefix}_{k}"] = v

    return result

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
