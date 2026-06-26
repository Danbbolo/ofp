"""
_check_futures_prices.py — Check for zero/negative prices in futures trades.
"""
import pandas as pd

df = pd.read_parquet("data/raw_futures/2026-06-23/trades.parquet")
print(f"Total rows: {len(df):,}")
print(f"Price stats:")
print(df["price"].describe())
print(f"\nZero prices: {(df['price'].astype(float) == 0).sum()}")
print(f"Negative prices: {(df['price'].astype(float) < 0).sum()}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
