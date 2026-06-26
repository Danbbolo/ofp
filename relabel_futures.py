"""
relabel_futures.py — Re-label the futures dataset with target-based exits.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw_futures")
INPUT_FILE = Path("data/research_dataset_futures.parquet")
OUTPUT_FILE = Path("data/research_dataset_futures_relabel.parquet")

NEW_HORIZONS_SEC = [1800, 3600, 7200, 14400]
MAX_HORIZON_SEC = max(NEW_HORIZONS_SEC)
TARGET_PCT = 0.01

print(f"Loading {INPUT_FILE} …")
df = pd.read_parquet(INPUT_FILE)
print(f"  {len(df):,} rows, {len(df.columns)} cols")

# Load futures trades
print("Loading futures trades …")
trades_files = sorted(RAW_DIR.glob("*/trades.parquet"))
trades_chunks = []
for f in trades_files:
    t = pd.read_parquet(f)
    t = t.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    t["timestamp_ms"] = t["timestamp_ms"].astype("int64")
    t["price"] = t["price"].astype(float)
    t["size"] = t["size"].astype(float)
    t = t[t["price"] > 0]
    trades_chunks.append(t[["timestamp_ms", "price", "size"]])
trades = pd.concat(trades_chunks, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
print(f"  {len(trades):,} trade rows")

# Get unique window_end_ms values
unique_ends = df["window_end_ms"].unique()
unique_ends.sort()
print(f"  {len(unique_ends):,} unique window_end_ms values")

trade_ts = trades["timestamp_ms"].values
trade_px = trades["price"].values

# Vectorized: for each window_end, find first trade index, then scan forward
print("Computing target_hit_1pct for each window_end …")
end_indices = np.searchsorted(trade_ts, unique_ends, side="left")
end_indices_4h = np.searchsorted(trade_ts, unique_ends + MAX_HORIZON_SEC * 1000, side="left")

n_windows = len(unique_ends)
target_hit = np.zeros(n_windows, dtype=np.int8)
mae_pct = np.zeros(n_windows, dtype=np.float64)

for i in range(n_windows):
    if i % 2000 == 0:
        print(f"    {i:,}/{n_windows:,}")
    lo = end_indices[i]
    hi = end_indices_4h[i]
    if hi <= lo:
        continue
    entry = trade_px[lo]
    target = entry * (1 + TARGET_PCT)
    stop = entry * (1 - TARGET_PCT)
    px = trade_px[lo:hi]
    crossed = (px >= target) | (px <= stop)
    if crossed.any():
        first_cross = int(np.argmax(crossed))
        if px[first_cross] >= target:
            target_hit[i] = 1
    mae_pct[i] = (px[-1] - entry) / entry

res_df = pd.DataFrame({
    "window_end_ms": unique_ends,
    "target_hit_1pct": target_hit,
    "mae_pct": mae_pct,
})
print(f"\n  target_hit rate: {res_df['target_hit_1pct'].mean():.3f}")
print(f"  mae_pct mean:    {res_df['mae_pct'].mean()*100:+.3f}%")

# Merge back
df = df.merge(res_df, on="window_end_ms", how="left")
df["outcome_binary"] = df["target_hit_1pct"].astype(int)
df["outcome_pct"] = df["mae_pct"]

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df):,} rows to {OUTPUT_FILE}")
print(f"  Final win rate: {df['outcome_binary'].mean():.3f}")
print(f"  Pair counts:")
for (ws, hz), grp in df.groupby(["window_size", "horizon"]):
    print(f"    W={ws} H={hz}: n={len(grp):,}, wr={grp['outcome_binary'].mean():.3f}")
