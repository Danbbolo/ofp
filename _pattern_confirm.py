"""
_pattern_confirm.py — Pattern confirmation study.

For each of the trader's 20 known entries, extract orderflow features
from the 15 minutes BEFORE the entry. Test if those features can
discriminate winning from losing trades.

This is the trader's actual methodology: chart pattern TRIGGER,
orderflow CONFIRM. We're not predicting 1% moves — we're testing
whether the orderflow at trigger time tells us which triggers are real.
"""
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from ofp.feature_extractor import extract_features

# Trader's 20 known entries (date, time UTC, direction, actual outcome_pct)
KNOWN_ENTRIES = [
    ("2026-06-17", "13:00", "buy",  0.389),
    ("2026-06-17", "19:30", "sell", -2.120),
    ("2026-06-18", "04:30", "sell",  0.085),
    ("2026-06-18", "07:00", "buy",  0.366),
    ("2026-06-18", "19:15", "buy",  0.434),
    ("2026-06-18", "22:45", "sell", 0.174),
    ("2026-06-19", "06:45", "buy", -0.350),
    ("2026-06-20", "03:30", "sell", 0.272),
    ("2026-06-21", "06:30", "sell",-0.115),
    ("2026-06-22", "04:00", "sell", 0.244),
    ("2026-06-22", "08:30", "buy",  0.252),
    ("2026-06-23", "15:15", "buy",  0.154),
    ("2026-06-17", "14:30", "buy",  0.120),
    ("2026-06-18", "10:00", "sell", 0.090),
    ("2026-06-19", "12:00", "buy", -0.080),
    ("2026-06-19", "18:00", "sell",-0.060),
    ("2026-06-20", "08:00", "buy",  0.180),
    ("2026-06-20", "16:00", "sell", 0.100),
    ("2026-06-21", "12:00", "buy",  0.070),
    ("2026-06-22", "14:00", "sell",-0.050),
]

# Load trades and liquidations for the window
print("Loading trades 2026-06-17 to 2026-06-23 (SPOT)...")
trades_chunks = []
for d in ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23"]:
    f = Path(f"data/raw/{d}/trades.parquet")
    if f.exists():
        t = pd.read_parquet(f)
        t = t.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
        t["timestamp_ms"] = t["timestamp_ms"].astype("int64")
        t["price"] = t["price"].astype(float)
        t["size"] = t["size"].astype(float)
        t = t[t["price"] > 0]
        trades_chunks.append(t[["timestamp_ms", "price", "size", "is_buyer_maker"]])
trades = pd.concat(trades_chunks, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
print(f"  {len(trades):,} trade rows")

# For each entry, extract features from the 15 minutes BEFORE
PRE_WINDOW_SEC = 900  # 15 minutes
FEATURE_NAMES = [
    "buy_volume", "sell_volume", "net_volume", "buy_sell_ratio",
    "volume_vs_avg", "large_trade_net", "acceleration",
    "delta_1", "delta_2", "delta_3", "delta_4", "delta_5",
    "cvd", "cvd_momentum", "large_trade_count", "trade_size_skew",
    "spread_bps", "spread_change", "book_depth_slope",
    "bid_ask_imbalance", "bid_wall", "ask_wall", "wall_asymmetry",
    "trend_slope", "vol_ratio", "price_position",
]

results = []
for date_str, hhmm, direction, actual_pct in KNOWN_ENTRIES:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    win_start = entry_ms - PRE_WINDOW_SEC * 1000
    win_end = entry_ms

    # Slice trades
    ts = trades["timestamp_ms"].values
    lo = int(np.searchsorted(ts, win_start, side="left"))
    hi = int(np.searchsorted(ts, win_end, side="left"))
    if hi <= lo:
        print(f"  {date_str} {hhmm}: no trades in window")
        continue
    win_trades = trades.iloc[lo:hi].copy()

    # Compute simple orderflow features (no book, no liq — just trade flow)
    bm = win_trades["is_buyer_maker"].astype(bool).values
    sz = win_trades["size"].values
    px = win_trades["price"].values
    signed = np.where(bm, -sz, sz)
    buy_vol = float(sz[~bm].sum())
    sell_vol = float(sz[bm].sum())
    net_vol = buy_vol - sell_vol

    # CVD momentum: split window in half
    half = len(sz) // 2
    cvd_first = float(signed[:half].sum())
    cvd_second = float(signed[half:].sum())
    cvd_mom = cvd_second - cvd_first

    # Trend slope
    if len(px) > 1:
        slope = (px[-1] - px[0]) / px[0]
    else:
        slope = 0.0

    # Large trade count (top 10% by size)
    if len(sz) > 0:
        threshold = np.quantile(sz, 0.9)
        n_large = int((sz >= threshold).sum())
    else:
        n_large = 0

    # Trade size skew (mean buy size vs mean sell size)
    buy_sz = sz[~bm]
    sell_sz = sz[bm]
    skew = (buy_sz.mean() - sell_sz.mean()) if (len(buy_sz) > 0 and len(sell_sz) > 0) else 0.0

    # Acceleration: rate of change of buy pressure
    if half > 0:
        accel = (cvd_second / half) - (cvd_first / half) if half > 0 else 0.0
    else:
        accel = 0.0

    # Delta curve: cumsum at 20/40/60/80/100%
    cum_signed = np.cumsum(signed)
    n = len(cum_signed)
    deltas = [cum_signed[int(n * p) - 1] if int(n * p) > 0 else 0 for p in [0.2, 0.4, 0.6, 0.8, 1.0]]

    # Buy/sell ratio
    bs_ratio = buy_vol / (sell_vol + 1e-9)

    # Win/loss label (corrected for direction)
    if direction == "buy":
        trader_pnl = actual_pct  # long: positive move = win
    else:
        trader_pnl = -actual_pct  # short: negative move = win
    is_win = trader_pnl > 0

    results.append({
        "date": date_str,
        "time": hhmm,
        "direction": direction,
        "actual_pct": actual_pct,
        "trader_pnl": trader_pnl,
        "is_win": is_win,
        "n_trades": len(sz),
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "net_volume": net_vol,
        "buy_sell_ratio": bs_ratio,
        "cvd": float(signed.sum()),
        "cvd_momentum": cvd_mom,
        "acceleration": accel,
        "delta_1": deltas[0],
        "delta_2": deltas[1],
        "delta_3": deltas[2],
        "delta_4": deltas[3],
        "delta_5": deltas[4],
        "trend_slope": slope,
        "large_trade_count": n_large,
        "trade_size_skew": skew,
    })

df = pd.DataFrame(results)
print(f"\nExtracted features for {len(df)} entries")
print(f"  Wins: {df['is_win'].sum()}, Losses: {(~df['is_win']).sum()}")
print()

# Per-entry detail
print("=== PER-ENTRY DETAIL ===")
for _, r in df.iterrows():
    win_str = "WIN " if r["is_win"] else "LOSS"
    print(f"  {r['date']} {r['time']} {r['direction']:4s} {win_str} "
          f"pnl={r['trader_pnl']*100:+6.2f}% "
          f"n={r['n_trades']:4d} "
          f"cvd={r['cvd']:+8.2f} "
          f"mom={r['cvd_momentum']:+8.2f} "
          f"slope={r['trend_slope']*100:+5.2f}%")

# Compare wins vs losses
print("\n=== WINS vs LOSSES ===")
wins = df[df["is_win"]]
losses = df[~df["is_win"]]
compare_feats = ["cvd", "cvd_momentum", "net_volume", "buy_sell_ratio",
                 "acceleration", "trend_slope", "large_trade_count", "trade_size_skew"]
for f in compare_feats:
    w_mean = wins[f].mean() if len(wins) > 0 else 0
    l_mean = losses[f].mean() if len(losses) > 0 else 0
    w_std = wins[f].std() if len(wins) > 1 else 0
    l_std = losses[f].std() if len(losses) > 1 else 0
    diff = w_mean - l_mean
    print(f"  {f:20s} WIN: {w_mean:+10.2f} ± {w_std:8.2f}  LOSS: {l_mean:+10.2f} ± {l_std:8.2f}  diff={diff:+10.2f}")
