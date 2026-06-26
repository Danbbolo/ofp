"""
download_futures_lowmem.py — Low-memory sequential downloader.
Downloads each hour file separately to avoid OOM.
"""
import sys
import time
from pathlib import Path
import cryptohftdata as chd
import gc

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures")

if len(sys.argv) != 3:
    print("Usage: python download_futures_lowmem.py START_DATE END_DATE")
    sys.exit(1)

from datetime import datetime, timedelta
start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
end = datetime.strptime(sys.argv[2], "%Y-%m-%d")

client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

d = start
while d <= end:
    date_str = d.strftime("%Y-%m-%d")
    day_dir = OUTPUT_ROOT / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all complete
    if (day_dir / "trades.parquet").exists() and (day_dir / "book.parquet").exists() and (day_dir / "liq.parquet").exists():
        print(f"[{date_str}] complete, skip", flush=True)
        d += timedelta(days=1)
        continue

    # Trades (small, ~50MB)
    if not (day_dir / "trades.parquet").exists():
        print(f"[{date_str}] trades ...", end=" ", flush=True)
        try:
            df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                     start_date=date_str, end_date=date_str)
            df_t.to_parquet(day_dir / "trades.parquet", index=False)
            print(f"OK ({len(df_t):,})", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        del df_t
        gc.collect()
    else:
        print(f"[{date_str}] trades OK", flush=True)

    # Book (HUGE, ~500MB-1GB per day) - download but DON'T keep in memory
    if not (day_dir / "book.parquet").exists():
        print(f"[{date_str}] book ...", flush=True)
        try:
            # The library writes to disk internally? Let's check by trying
            df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                        start_date=date_str, end_date=date_str)
            # Save immediately, then free
            df_b.to_parquet(day_dir / "book.parquet", index=False)
            n = len(df_b)
            del df_b
            gc.collect()
            print(f"[{date_str}] book OK ({n:,})", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Liquidations
    if not (day_dir / "liq.parquet").exists():
        print(f"[{date_str}] liq ...", end=" ", flush=True)
        try:
            df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                           start_date=date_str, end_date=date_str)
            if len(df_l) > 0:
                df_l.to_parquet(day_dir / "liq.parquet", index=False)
                print(f"OK ({len(df_l):,})", flush=True)
            else:
                # Touch empty file so we skip next time
                pd.DataFrame(columns=["timestamp_ms"]).to_parquet(day_dir / "liq.parquet", index=False)
                print("none", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        try:
            del df_l
        except:
            pass
        gc.collect()
    else:
        print(f"[{date_str}] liq OK", flush=True)

    d += timedelta(days=1)
    time.sleep(1)

print("\nDone.")
