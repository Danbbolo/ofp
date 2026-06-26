"""
_pattern_filter.py — Test the filter rule: CVD momentum > threshold
AND large_trade_count > threshold → trade is a winner.

This is the real test of whether orderflow features can CONFIRM
chart pattern entries.

The trader's 20 entries: 12 wins, 8 losses. We use the same 15-min
pre-entry window. Now extended to support any number of trader entries
and a simple "would this filter have helped" backtest.
"""
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Trader's 20 known entries
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

# Load spot trades
print("Loading spot trades 2026-06-17 to 2026-06-23 ...")
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

# Extract features for each entry
PRE_WINDOW_SEC = 900
results = []
for date_str, hhmm, direction, actual_pct in KNOWN_ENTRIES:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    win_start = entry_ms - PRE_WINDOW_SEC * 1000
    win_end = entry_ms

    ts = trades["timestamp_ms"].values
    lo = int(np.searchsorted(ts, win_start, side="left"))
    hi = int(np.searchsorted(ts, win_end, side="left"))
    if hi <= lo:
        continue
    win = trades.iloc[lo:hi]

    bm = win["is_buyer_maker"].astype(bool).values
    sz = win["size"].values
    px = win["price"].values
    signed = np.where(bm, -sz, sz)
    buy_vol = float(sz[~bm].sum())
    sell_vol = float(sz[bm].sum())
    net_vol = buy_vol - sell_vol

    half = max(len(sz) // 2, 1)
    cvd_first = float(signed[:half].sum())
    cvd_second = float(signed[half:].sum())
    cvd_mom = cvd_second - cvd_first
    slope = (px[-1] - px[0]) / px[0] if len(px) > 1 else 0.0

    if len(sz) > 0:
        threshold = np.quantile(sz, 0.9)
        n_large = int((sz >= threshold).sum())
    else:
        n_large = 0

    # Trader PnL: sign by direction
    if direction == "buy":
        trader_pnl = actual_pct
    else:
        trader_pnl = -actual_pct
    is_win = trader_pnl > 0

    results.append({
        "date": date_str, "time": hhmm, "direction": direction,
        "actual_pct": actual_pct, "trader_pnl": trader_pnl, "is_win": is_win,
        "n_trades": len(sz), "buy_vol": buy_vol, "sell_vol": sell_vol,
        "net_volume": net_vol, "cvd": float(signed.sum()),
        "cvd_momentum": cvd_mom, "trend_slope": slope,
        "large_trade_count": n_large, "buy_sell_ratio": buy_vol / (sell_vol + 1e-9),
    })

df = pd.DataFrame(results)
wins = df[df["is_win"]]
losses = df[~df["is_win"]]
print(f"\nTotal: {len(df)} entries, {len(wins)} wins, {len(losses)} losses")

# Statistical tests
print("\n" + "=" * 70)
print("STATISTICAL TESTS (Mann-Whitney U, two-sided)")
print("=" * 70)
feats = ["cvd", "cvd_momentum", "net_volume", "buy_sell_ratio",
         "trend_slope", "large_trade_count"]
for f in feats:
    w = wins[f].values
    l = losses[f].values
    if len(w) < 2 or len(l) < 2:
        print(f"  {f:20s}  too few samples")
        continue
    try:
        u_stat, p_val = stats.mannwhitneyu(w, l, alternative="two-sided")
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        print(f"  {f:20s}  W={w.mean():+10.2f} ± {w.std():8.2f}  "
              f"L={l.mean():+10.2f} ± {l.std():8.2f}  p={p_val:.4f} {sig}")
    except Exception as e:
        print(f"  {f:20s}  error: {e}")

# Direction-aware test: for each entry, sign the features by direction
# (i.e., for sells, positive cvd_momentum = against the trade)
print("\n" + "=" * 70)
print("DIRECTION-AWARE TEST (CVD momentum in TRADE direction)")
print("=" * 70)
df["cvd_mom_aligned"] = df.apply(
    lambda r: r["cvd_momentum"] if r["direction"] == "buy" else -r["cvd_momentum"],
    axis=1
)
df["cvd_aligned"] = df.apply(
    lambda r: r["cvd"] if r["direction"] == "buy" else -r["cvd"],
    axis=1
)
df["slope_aligned"] = df.apply(
    lambda r: r["trend_slope"] if r["direction"] == "buy" else -r["trend_slope"],
    axis=1
)
wins = df[df["is_win"]]
losses = df[~df["is_win"]]
for f in ["cvd_mom_aligned", "cvd_aligned", "slope_aligned"]:
    w = wins[f].values
    l = losses[f].values
    if len(w) < 2 or len(l) < 2:
        continue
    u_stat, p_val = stats.mannwhitneyu(w, l, alternative="two-sided")
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
    print(f"  {f:20s}  W={w.mean():+10.2f} ± {w.std():8.2f}  "
          f"L={l.mean():+10.2f} ± {l.std():8.2f}  p={p_val:.4f} {sig}")

# Filter rule backtest
print("\n" + "=" * 70)
print("FILTER RULE BACKTEST")
print("=" * 70)

# Rule 1: take only if direction-aligned CVD momentum > 0
for thresh in [-50, -20, 0, 20, 50, 100]:
    keep = df["cvd_mom_aligned"] > thresh
    n_keep = keep.sum()
    if n_keep == 0:
        continue
    wr_kept = df[keep]["is_win"].mean()
    pnl_kept = df[keep]["trader_pnl"].sum()
    n_reject = (~keep).sum()
    wr_rejected = df[~keep]["is_win"].mean() if n_reject > 0 else 0
    print(f"  CVD_mom_aligned > {thresh:+4d}: kept {n_keep:2d}/{len(df)} ({n_keep/len(df)*100:.0f}%) "
          f"WR_kept={wr_kept:.1%}  WR_rejected={wr_rejected:.1%}  PnL_kept={pnl_kept*100:+.1f}%")

# Rule 2: take only if large_trade_count > median
median_large = df["large_trade_count"].median()
print(f"\n  Median large_trade_count: {median_large:.0f}")
for thresh_pct in [0, 25, 50, 75, 100]:
    thresh_val = df["large_trade_count"].quantile(thresh_pct / 100)
    keep = df["large_trade_count"] >= thresh_val
    n_keep = keep.sum()
    if n_keep == 0:
        continue
    wr_kept = df[keep]["is_win"].mean()
    pnl_kept = df[keep]["trader_pnl"].sum()
    n_reject = (~keep).sum()
    wr_rejected = df[~keep]["is_win"].mean() if n_reject > 0 else 0
    print(f"  large_trades >= q{thresh_pct:2d} ({thresh_val:.0f}): kept {n_keep:2d}/{len(df)} "
          f"WR_kept={wr_kept:.1%}  WR_rejected={wr_rejected:.1%}  PnL_kept={pnl_kept*100:+.1f}%")

# Combined rule: CVD momentum aligned > 0 AND large trades > median
print(f"\n  === COMBINED: CVD_mom_aligned > 0 AND large_trades >= median ===")
keep = (df["cvd_mom_aligned"] > 0) & (df["large_trade_count"] >= median_large)
n_keep = keep.sum()
wr_kept = df[keep]["is_win"].mean() if n_keep > 0 else 0
pnl_kept = df[keep]["trader_pnl"].sum() if n_keep > 0 else 0
print(f"  Kept: {n_keep}/{len(df)} ({n_keep/len(df)*100:.0f}%)")
print(f"  WR_kept: {wr_kept:.1%}")
print(f"  PnL_kept: {pnl_kept*100:+.1f}%")
print(f"  Baseline (no filter): WR={df['is_win'].mean():.1%} PnL={df['trader_pnl'].sum()*100:+.1f}%")
