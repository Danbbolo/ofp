"""
_pattern_confirm_futures.py — Re-run pattern confirmation on 3-day FUTURES
data, this time with REAL liquidation features (post-bugfix).

Compares wins vs losses on the trader's 20 entries, using orderflow
features from the 15min BEFORE each entry.
"""
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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

# Load FUTURES trades (3-day)
print("Loading FUTURES trades 2026-06-21 to 2026-06-23 ...")
trades_chunks = []
for d in ["2026-06-21", "2026-06-22", "2026-06-23"]:
    f = Path(f"data/raw_futures/{d}/trades.parquet")
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

# Load liquidations
print("Loading FUTURES liquidations ...")
liq_chunks = []
for d in ["2026-06-21", "2026-06-22", "2026-06-23"]:
    f = Path(f"data/raw_futures/{d}/liq.parquet")
    if f.exists():
        l = pd.read_parquet(f)
        l = l.rename(columns={"event_time": "timestamp_ms", "quantity": "size"})
        l["timestamp_ms"] = l["timestamp_ms"].astype("int64")
        l["price"] = l["price"].astype(float)
        l["size"] = l["size"].astype(float)
        # SELL = long liquidation, BUY = short liquidation (taker side)
        l["long_liq"] = (l["side"] == "SELL").astype(float) * l["size"]
        l["short_liq"] = (l["side"] == "BUY").astype(float) * l["size"]
        liq_chunks.append(l[["timestamp_ms", "long_liq", "short_liq"]])
liq = pd.concat(liq_chunks, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
print(f"  {len(liq):,} liq rows")

# Extract features for each entry
PRE_WINDOW_SEC = 900
results = []
for date_str, hhmm, direction, actual_pct in KNOWN_ENTRIES:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    win_start = entry_ms - PRE_WINDOW_SEC * 1000
    win_end = entry_ms

    # Orderflow features
    ts = trades["timestamp_ms"].values
    lo = int(np.searchsorted(ts, win_start, side="left"))
    hi = int(np.searchsorted(ts, win_end, side="left"))
    if hi <= lo:
        results.append({"date": date_str, "time": hhmm, "direction": direction,
                        "actual_pct": actual_pct, "is_win": False, "in_data": False})
        continue
    win = trades.iloc[lo:hi]
    bm = win["is_buyer_maker"].astype(bool).values
    sz = win["size"].values
    px = win["price"].values
    signed = np.where(bm, -sz, sz)
    buy_vol = float(sz[~bm].sum())
    sell_vol = float(sz[bm].sum())

    half = max(len(sz) // 2, 1)
    cvd_first = float(signed[:half].sum())
    cvd_second = float(signed[half:].sum())
    cvd_mom = cvd_second - cvd_first

    threshold = np.quantile(sz, 0.9) if len(sz) > 0 else 0
    n_large = int((sz >= threshold).sum())

    # Liquidation features
    liq_ts = liq["timestamp_ms"].values
    llo = int(np.searchsorted(liq_ts, win_start, side="left"))
    lhi = int(np.searchsorted(liq_ts, win_end, side="left"))
    win_liq = liq.iloc[llo:lhi]
    long_liq_vol = float(win_liq["long_liq"].sum())
    short_liq_vol = float(win_liq["short_liq"].sum())
    total_liq = long_liq_vol + short_liq_vol

    # Trader PnL
    if direction == "buy":
        trader_pnl = actual_pct
    else:
        trader_pnl = -actual_pct
    is_win = trader_pnl > 0

    results.append({
        "date": date_str, "time": hhmm, "direction": direction,
        "actual_pct": actual_pct, "trader_pnl": trader_pnl, "is_win": is_win,
        "in_data": True, "n_trades": int(len(sz)),
        "buy_vol": buy_vol, "sell_vol": sell_vol,
        "cvd_momentum": cvd_mom, "large_trade_count": int(n_large),
        "long_liq": long_liq_vol, "short_liq": short_liq_vol, "total_liq": total_liq,
    })

df = pd.DataFrame(results)
print(f"\nTotal entries: {len(df)}, in data: {df['in_data'].sum()}, "
      f"wins: {df[df['in_data']]['is_win'].sum()}")

# Filter to in-data entries only
df_in = df[df["in_data"]].copy()
if len(df_in) == 0:
    print("No entries in 3-day futures data!")
    sys.exit(1)

wins = df_in[df_in["is_win"]]
losses = df_in[~df_in["is_win"]]
print(f"  In data: {len(df_in)} entries ({len(wins)} wins, {len(losses)} losses)")

# Per-entry detail
print("\n=== PER-ENTRY DETAIL (3-day futures) ===")
for _, r in df_in.iterrows():
    win_str = "WIN " if r["is_win"] else "LOSS"
    in_data = "IN" if r["in_data"] else "OUT"
    print(f"  {r['date']} {r['time']} {r['direction']:4s} {win_str} {in_data} "
          f"pnl={r['trader_pnl']*100:+6.2f}% "
          f"n={r['n_trades']:5d} "
          f"mom={r['cvd_momentum']:+8.2f} "
          f"large={r['large_trade_count']:5d} "
          f"liq(L/S)={r['long_liq']:.2f}/{r['short_liq']:.2f}")

# Statistical tests
print("\n" + "=" * 70)
print("STATISTICAL TESTS (Mann-Whitney U, 3-day futures)")
print("=" * 70)
feats = ["cvd_momentum", "large_trade_count", "long_liq", "short_liq", "total_liq"]
for f in feats:
    w = wins[f].values
    l = losses[f].values
    if len(w) < 2 or len(l) < 2:
        print(f"  {f:20s}  too few samples (w={len(w)}, l={len(l)})")
        continue
    try:
        u_stat, p_val = stats.mannwhitneyu(w, l, alternative="two-sided")
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        print(f"  {f:20s}  W={w.mean():+10.2f} ± {w.std():8.2f}  "
              f"L={l.mean():+10.2f} ± {l.std():8.2f}  p={p_val:.4f} {sig}")
    except Exception as e:
        print(f"  {f:20s}  error: {e}")

# Combined filter: large_trade_count > median AND liq activity
print("\n" + "=" * 70)
print("FILTER RULE BACKTEST (3-day futures)")
print("=" * 70)
median_large = df_in["large_trade_count"].median()
print(f"  Median large_trade_count: {median_large:.0f}")
print(f"  Median total_liq:         {df_in['total_liq'].median():.3f}")
print(f"  Baseline (no filter): WR={df_in['is_win'].mean():.1%} "
      f"PnL={df_in['trader_pnl'].sum()*100:+.1f}%")
print()

# Try several filters
for desc, mask in [
    ("large_trade_count >= median",
     df_in["large_trade_count"] >= median_large),
    ("total_liq > 0",
     df_in["total_liq"] > 0),
    ("cvd_momentum > 0",
     df_in["cvd_momentum"] > 0),
    ("large_trade >= median AND total_liq > 0",
     (df_in["large_trade_count"] >= median_large) & (df_in["total_liq"] > 0)),
    ("large_trade >= median AND cvd_mom > 0",
     (df_in["large_trade_count"] >= median_large) & (df_in["cvd_momentum"] > 0)),
    ("large_trade >= q75 (4,000+)",
     df_in["large_trade_count"] >= 4000),
]:
    kept = df_in[mask]
    n_k = len(kept)
    n_r = len(df_in) - n_k
    if n_k == 0:
        continue
    wr_k = kept["is_win"].mean()
    pnl_k = kept["trader_pnl"].sum() * 100
    wr_r = df_in[~mask]["is_win"].mean() if n_r > 0 else 0
    print(f"  {desc}:")
    print(f"    Kept: {n_k}/{len(df_in)} ({n_k/len(df_in)*100:.0f}%)  "
          f"WR_kept={wr_k:.1%}  WR_rejected={wr_r:.1%}  PnL_kept={pnl_k:+.1f}%")
