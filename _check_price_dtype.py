import pandas as pd
df = pd.read_parquet("data/raw_futures/2026-06-23/trades.parquet")
print(f"price dtype: {df['price'].dtype}")
print(f"price[0] type: {type(df['price'].iloc[0])}")
print(f"price[0] value: {df['price'].iloc[0]}")
# Try converting
try:
    p = float(df['price'].iloc[0])
    print(f"As float: {p}")
except Exception as e:
    print(f"Convert error: {e}")
