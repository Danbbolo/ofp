"""
_check_futures_liq.py — Check the futures liquidation data columns.
"""
import pandas as pd

df = pd.read_parquet("data/raw_futures/2026-06-23/liq.parquet")
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nSide values: {df['side'].value_counts().to_dict()}")
print(f"Order type: {df['order_type'].value_counts().to_dict()}")
print(f"Time range: {df['event_time'].min()} → {df['event_time'].max()}")
