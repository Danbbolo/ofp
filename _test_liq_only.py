"""
_test_liq_only.py — Minimal test of liq feature extraction.
"""
import numpy as np
import pandas as pd
from ofp.feature_extractor import extract_features

# Tiny test
liq = pd.DataFrame({
    "timestamp_ms": [100, 200, 300, 400],
    "side": ["sell", "buy", "sell", "buy"],
    "price": [100.0, 100.0, 100.0, 100.0],
    "size": [1.0, 2.0, 3.0, 4.0],
})
trades = pd.DataFrame({
    "timestamp_ms": [100, 200, 300],
    "price": [100.0, 100.0, 100.0],
    "size": [0.5, 0.5, 0.5],
    "is_buyer_maker": [False, True, False],
})
feats = extract_features(
    trades_df=trades,
    book_snapshot_start=([], []),
    book_snapshot_end=([], []),
    liq_df=liq,
    window_start_ms=0,
    window_end_ms=500,
    rolling_avg_volume=0.5,
    current_price=100.0,
)
print("Features:")
for k, v in feats.items():
    if "liq" in k:
        print(f"  {k}: {v}")
