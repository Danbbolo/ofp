"""
download_futures_seq.py — Sequential day-by-day downloader.
No multiprocessing, no resource_tracker issues.
"""
import sys
import time
from pathlib import Path
import cryptohftdata as chd
import pandas as pd
import gc

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures")

if len(sys.argv) != 3:
    print("Usage: python download_futures_seq.py START_DATE END_DATE")
    sys.exit(1)

from datetime import datetime, timedelta
start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
end = datetime.strptime(sys.argv[2], "%Y-%m-%d")

d = start
while d <= end:
    date_str = d.strftime("%Y-%m-%d")
    day_dir = OUTPUT_ROOT / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    if (day_dir / "trades.parquet").exists() and (day_dir / "book.parquet").exists() and (day_dir / "liq.parquet").exists():
        print(f"[{date_str}] already complete, skipping")
        d += timedelta(days=1)
        continue

    # Fresh client per day to avoid resource_tracker warnings
    client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

    # Trades
    if not (day_dir / "trades.parquet").exists():
        print(f"[{date_str}] trades ...", end=" ", flush=True)
        try:
            df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                     start_date=date_str, end_date=date_str)
            df_t.to_parquet(day_dir / "trades.parquet", index=False)
            print(f"OK ({len(df_t):,})")
        except Exception as e:
            print(f"ERROR: {e}")
        del df_t; gc.collect()
    else:
        print(f"[{date_str}] trades already done")

    # Book
    if not (day_dir / "book.parquet").exists():
        print(f"[{date_str}] book ...", end=" ", flush=True)
        try:
            df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                        start_date=date_str, end_date=date_str)
            df_b.to_parquet(day_dir / "book.parquet", index=False)
            print(f"OK ({len(df_b):,})")
        except Exception as e:
            print(f"ERROR: {e}")
        del df_b; gc.collect()
    else:
        print(f"[{date_str}] book already done")

    # Liquidations
    if not (day_dir / "liq.parquet").exists():
        print(f"[{date_str}] liq ...", end=" ", flush=True)
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
    else:
        print(f"[{date_str}] liq already done")

    d += timedelta(days=1)
    del client; gc.collect()
    time.sleep(2)  # brief pause between days

print("\nDone.")
