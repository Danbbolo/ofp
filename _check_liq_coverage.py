"""
_check_liq_coverage.py — Check liq data coverage in test window.
"""
import datetime as dt
import pandas as pd

trades = pd.read_parquet("data/raw_futures/2026-06-23/trades.parquet")
end_time_ms = int(trades["trade_time"].iloc[1_000_000])
d = dt.datetime.fromtimestamp(end_time_ms/1000, dt.timezone.utc)
print(f"end_time_ms = {end_time_ms}")
print(f"UTC = {d}")

# Check liq coverage around this time
liq = pd.read_parquet("data/raw_futures/2026-06-23/liq.parquet")
liq_ms = liq["event_time"]
print(f"\nLiq range: {liq_ms.min()} to {liq_ms.max()}")
print(f"\nLiQ in [end-1800s, end]:")
mask = (liq_ms >= end_time_ms - 1800*1000) & (liq_ms <= end_time_ms)
print(f"  count: {mask.sum()}")
print(f"\nLiQ in [end-300s, end]:")
mask = (liq_ms >= end_time_ms - 300*1000) & (liq_ms <= end_time_ms)
print(f"  count: {mask.sum()}")
print(f"\nLiQ in [end-60s, end]:")
mask = (liq_ms >= end_time_ms - 60*1000) & (liq_ms <= end_time_ms)
print(f"  count: {mask.sum()}")

# Find a window WITH liquidations
# Pick a liq timestamp and use that as the center
liq_times_with_data = liq_ms[liq_ms.between(end_time_ms - 4*3600*1000, end_time_ms)]
if len(liq_times_with_data) > 0:
    test_end = int(liq_times_with_data.iloc[0])
    print(f"\n=== Testing with end_time_ms = {test_end} ({dt.datetime.fromtimestamp(test_end/1000, dt.timezone.utc)}) ===")
    for win_s in [60, 300, 1800]:
        mask = (liq_ms >= test_end - win_s*1000) & (liq_ms <= test_end)
        print(f"  LiQ in last {win_s}s: {mask.sum()}")
