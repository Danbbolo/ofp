"""
_check_zoom_features.py — Check if micro/meso/macro features are actually different.
"""
import pandas as pd
df = pd.read_parquet("data/research_dataset_futures_relabel.parquet")

# Look at first 5 rows for one window_end
sample = df[df["window_end_ms"] == df["window_end_ms"].iloc[0]].sort_values(["window_size", "horizon"])
print(f"Rows for window_end_ms = {df['window_end_ms'].iloc[0]}")
print(f"  window_size,horizon pairs: {len(sample)}")
print()

# Pick a few features
test_features = ["buy_volume", "cvd", "net_volume", "trend_slope", "delta_1", "bid_wall", "spread_bps", "hour_cos"]
for feat in test_features:
    print(f"=== {feat} ===")
    for prefix in ["micro", "meso", "macro"]:
        col = f"{prefix}_{feat}"
        if col in sample.columns:
            print(f"  {col:25s} = {sample[col].values}")
    print()
