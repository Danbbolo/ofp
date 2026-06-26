"""Quick check: what data do we have on Hetzner?"""
import pandas as pd
import glob
import os

raw_dir = "/root/ofp/data/raw"
dates = sorted(os.listdir(raw_dir))
print(f"Dates available: {dates}")
print()

for d in dates[:2]:  # just check first 2
    trades_path = os.path.join(raw_dir, d, "trades.parquet")
    book_path = os.path.join(raw_dir, d, "book.parquet")
    
    if os.path.exists(trades_path):
        df = pd.read_parquet(trades_path)
        print(f"=== {d} trades ===")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df):,}")
        print(f"  First row: {df.iloc[0].to_dict()}")
        print()
    
    if os.path.exists(book_path):
        df = pd.read_parquet(book_path)
        print(f"=== {d} book ===")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df):,}")
        print(f"  First row: {df.iloc[0].to_dict()}")
        print()

# Check if research_dataset exists
for f in ["/root/ofp/data/research_dataset.parquet", "/root/ofp/data/research_dataset_target.parquet"]:
    if os.path.exists(f):
        df = pd.read_parquet(f)
        print(f"=== {os.path.basename(f)} ===")
        print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")
        print()
