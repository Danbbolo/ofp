"""
_test_zoom_extraction.py — Test extract_multi_zoom_features directly
to see if micro/meso/macro features differ.
"""
import numpy as np
import pandas as pd
from ofp.feature_extractor import extract_multi_zoom_features

# Load futures trades
trades = pd.read_parquet("data/raw_futures/2026-06-23/trades.parquet")
trades = trades.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
trades["timestamp_ms"] = trades["timestamp_ms"].astype("int64")
trades["price"] = trades["price"].astype(float)
trades["size"] = trades["size"].astype(float)
trades = trades[trades["price"] > 0]
trades = trades[["timestamp_ms", "price", "size", "is_buyer_maker"]]
trades = trades.sort_values("timestamp_ms").reset_index(drop=True)
print(f"Trades: {len(trades):,}")

# Load liq
liq = pd.read_parquet("data/raw_futures/2026-06-23/liq.parquet")
liq = liq.rename(columns={"event_time": "timestamp_ms", "quantity": "size"})
liq["timestamp_ms"] = liq["timestamp_ms"].astype("int64")
liq["price"] = liq["price"].astype(float)
liq["size"] = liq["size"].astype(float)
liq["side"] = liq["side"].str.lower()  # FIX
liq = liq[["timestamp_ms", "side", "price", "size"]].sort_values("timestamp_ms").reset_index(drop=True)
print(f"LiQ: {len(liq):,} rows")

# Empty book snapshots
book_snapshots = {}

# End time: pick a time in the middle
end_time_ms = int(trades["timestamp_ms"].iloc[1_000_000])  # somewhere in the middle

# Test with different micro window sizes
for micro_s in [60, 120, 180]:
    print(f"\n=== micro_window = {micro_s}s ===")
    feats = extract_multi_zoom_features(
        trades_df=trades,
        book_snapshots=book_snapshots,
        liq_df=liq,
        micro_window_ms=micro_s * 1000,
        meso_window_ms=300 * 1000,
        macro_window_ms=1800 * 1000,
        end_time_ms=end_time_ms,
    )
    for k in ["buy_volume", "cvd", "net_volume", "delta_1", "trend_slope", "long_liq_vol", "short_liq_vol"]:
        print(f"  {k:20s} micro={feats[f'micro_{k}']:12.4f} meso={feats[f'meso_{k}']:12.4f} macro={feats[f'macro_{k}']:12.4f}")
