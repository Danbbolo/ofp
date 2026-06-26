"""
relabel_move_start_oos.py — Relabel OOS dataset with had_move.

Same logic as relabel_move_start.py but reads from OOS data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = "data/research_dataset_oos.parquet"
OUTPUT_FILE = "data/research_dataset_oos_move_start.parquet"
RAW_DIR = Path("data/raw_futures_oos")

MOVE_THRESHOLD = 0.005  # 0.5%
LOOKAHEAD_SEC = 3600    # 1 hour

print(f"Loading {INPUT_FILE} …")
df = pd.read_parquet(INPUT_FILE)
print(f"  {len(df):,} rows, {len(df.columns)} cols")

# Load futures trades for forward price lookup
print("Loading OOS futures trades …")
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

trade_ts = trades["timestamp_ms"].values
trade_px = trades["price"].values

# For each unique window_end_ms, compute had_move and move_direction
unique_ends = df["window_end_ms"].unique()
unique_ends.sort()
print(f"  {len(unique_ends):,} unique window_end_ms values")

print("Computing had_move and move_direction …")
n_windows = len(unique_ends)
had_move = np.zeros(n_windows, dtype=np.int8)
move_dir = np.zeros(n_windows, dtype=np.int8)
move_pct = np.zeros(n_windows, dtype=np.float64)

for i in range(n_windows):
    if i % 2000 == 0:
        print(f"    {i:,}/{n_windows:,}")
    win_end = unique_ends[i]
    lo = int(np.searchsorted(trade_ts, win_end, side="left"))
    if lo >= len(trade_ts):
        continue
    entry = trade_px[lo]
    hi = int(np.searchsorted(trade_ts, win_end + LOOKAHEAD_SEC * 1000, side="left"))
    if hi <= lo:
        continue
    px = trade_px[lo:hi]
    max_px = px.max()
    min_px = px.min()
    max_pct = (max_px - entry) / entry
    min_pct = (min_px - entry) / entry
    up_hit = max_pct >= MOVE_THRESHOLD
    dn_hit = min_pct <= -MOVE_THRESHOLD
    if up_hit and dn_hit:
        up_idx = int(np.argmax(px >= entry * (1 + MOVE_THRESHOLD)))
        dn_idx = int(np.argmax(px <= entry * (1 - MOVE_THRESHOLD)))
        if up_idx < dn_idx:
            had_move[i] = 1
            move_dir[i] = 1
            move_pct[i] = max_pct
        else:
            had_move[i] = 1
            move_dir[i] = -1
            move_pct[i] = min_pct
    elif up_hit:
        had_move[i] = 1
        move_dir[i] = 1
        move_pct[i] = max_pct
    elif dn_hit:
        had_move[i] = 1
        move_dir[i] = -1
        move_pct[i] = min_pct
    else:
        had_move[i] = 0
        move_dir[i] = 0
        move_pct[i] = (px[-1] - entry) / entry

res_df = pd.DataFrame({
    "window_end_ms": unique_ends,
    "had_move": had_move,
    "move_direction": move_dir,
    "move_pct": move_pct,
})

print(f"\n  had_move rate: {res_df['had_move'].mean():.3f}")
print(f"  move_direction: up={((res_df['move_direction']==1).sum())}, "
      f"down={((res_df['move_direction']==-1).sum())}, "
      f"none={((res_df['move_direction']==0).sum())}")
print(f"  move_pct mean: {res_df['move_pct'].mean()*100:+.3f}%")

# Merge back to dataset
df = df.merge(res_df, on="window_end_ms", how="left")
df["outcome_binary"] = df["had_move"].astype(int)
df["outcome_pct"] = df["move_pct"]

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df):,} rows to {OUTPUT_FILE}")
print(f"  Final had_move rate: {df['outcome_binary'].mean():.3f}")
print(f"  Pair counts:")
for (ws, hz), grp in df.groupby(["window_size", "horizon"]):
    print(f"    W={ws} H={hz}: n={len(grp):,}, had_move={grp['outcome_binary'].mean():.3f}")
