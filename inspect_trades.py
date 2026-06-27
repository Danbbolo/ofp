"""Inspect trades data schema."""
import pandas as pd

df = pd.read_parquet("data/raw_futures/2026-06-17/trades.parquet")
print("cols:", list(df.columns))
print("dtypes:", {c: str(t) for c, t in df.dtypes.items()})
print("rows:", len(df))
print(df.head(3))
print("---")
print("total volume:", df["quantity"].sum())
print("is_buyer_maker vals:", df["is_buyer_maker"].unique()[:5])
print("price sample:", df["price"].head(3).tolist())
print("trade_time sample:", df["trade_time"].head(3).tolist())