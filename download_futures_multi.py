"""
download_futures_multi.py — Download multiple days of BTCUSDT futures.
"""
import sys
from pathlib import Path
import cryptohftdata as chd
import pandas as pd
import gc

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures")

if len(sys.argv) != 3:
    print("Usage: python download_futures_multi.py START_DATE END_DATE")
    sys.exit(1)

start_date = sys.argv[1]
end_date = sys.argv[2]
client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

from datetime import datetime, timedelta
d = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")
while d <= end:
    date_str = d.strftime("%Y-%m-%d")
    day_dir = OUTPUT_ROOT / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    if (day_dir / "trades.parquet").exists():
        print(f"[{date_str}] already downloaded, skipping")
        d += timedelta(days=1)
        continue

    # Trades
    print(f"[{date_str}] trades …", end=" ", flush=True)
    try:
        df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                 start_date=date_str, end_date=date_str)
        df_t.to_parquet(day_dir / "trades.parquet", index=False)
        print(f"OK ({len(df_t):,})")
    except Exception as e:
        print(f"ERROR: {e}")
    del df_t; gc.collect()

    # Book
    print(f"[{date_str}] book …", end=" ", flush=True)
    try:
        df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                    start_date=date_str, end_date=date_str)
        df_b.to_parquet(day_dir / "book.parquet", index=False)
        print(f"OK ({len(df_b):,})")
    except Exception as e:
        print(f"ERROR: {e}")
    del df_b; gc.collect()

    # Liquidations
    print(f"[{date_str}] liq …", end=" ", flush=True)
    try:
        df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                       start_date=date_str, end_date=date_str)
        if len(df_l) > 0:
            df_l.to_parquet(day_dir / "liq.parquet", index=False)
            print(f"OK ({len(df_l):,})")
        else:
            print("none")
    except Exception as e:
        print(f"ERROR: {e}")

    d += timedelta(days=1)

print("\nDone.")
