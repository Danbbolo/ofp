"""
download_futures_1d.py — Download 1 day of BTCUSDT PERPETUAL FUTURES data.

Saves to data/raw_futures/YYYY-MM-DD/ to avoid clobbering spot data.
"""
import sys
import gc
from pathlib import Path
import cryptohftdata as chd
import pandas as pd

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures")

if len(sys.argv) != 2:
    print("Usage: python download_futures_1d.py YYYY-MM-DD")
    sys.exit(1)

date_str = sys.argv[1]
day_dir = OUTPUT_ROOT / date_str
day_dir.mkdir(parents=True, exist_ok=True)

client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

# Trades
print(f"[{date_str}] Downloading trades (FUTURES) …", end=" ", flush=True)
try:
    df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                             start_date=date_str, end_date=date_str)
    df_t.to_parquet(day_dir / "trades.parquet", index=False)
    print(f"OK ({len(df_t):,} rows)")
    print(f"  Columns: {list(df_t.columns)}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
del df_t; gc.collect()

# Book
print(f"[{date_str}] Downloading book (FUTURES) …", end=" ", flush=True)
try:
    df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                start_date=date_str, end_date=date_str)
    df_b.to_parquet(day_dir / "book.parquet", index=False)
    print(f"OK ({len(df_b):,} rows)")
    print(f"  Columns: {list(df_b.columns)}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
del df_b; gc.collect()

# Liquidations
print(f"[{date_str}] Downloading liquidations (FUTURES) …", end=" ", flush=True)
try:
    df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                   start_date=date_str, end_date=date_str)
    if len(df_l) > 0:
        df_l.to_parquet(day_dir / "liq.parquet", index=False)
        print(f"OK ({len(df_l):,} rows)")
        print(f"  Columns: {list(df_l.columns)}")
    else:
        print("no liquidations in this window")
except Exception as e:
    print(f"ERROR: {e}")
    # not fatal - liq is optional

print(f"\nDone. Data in {day_dir}/")
