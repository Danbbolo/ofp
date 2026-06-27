"""
download_all_futures.py — Download all IS + OOS futures data to Azure.

IS dates: 2026-06-17 to 2026-06-23 (7 dates) -> data/raw_futures/
OOS dates: 2026-06-24 to 2026-06-26 (3 dates) -> data/raw_futures_oos/
"""
import sys
import gc
import time
from pathlib import Path
import cryptohftdata as chd

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"

IS_DATES = ["2026-06-17", "2026-06-18", "2026-06-19",
            "2026-06-20", "2026-06-21", "2026-06-22", "2026-06-23"]
OOS_DATES = ["2026-06-24", "2026-06-25", "2026-06-26"]


def download_date(date_str, output_root):
    """Download trades + book + liq for one date."""
    day_dir = Path(output_root) / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    trades_path = day_dir / "trades.parquet"
    book_path = day_dir / "book.parquet"
    liq_path = day_dir / "liq.parquet"
    if trades_path.exists() and book_path.exists():
        print(f"[{date_str}] SKIP (already exists)")
        return True

    client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

    # Trades
    t0 = time.time()
    print(f"[{date_str}] Downloading trades (FUTURES) …", end=" ", flush=True)
    try:
        df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                 start_date=date_str, end_date=date_str)
        df_t.to_parquet(trades_path, index=False)
        print(f"OK ({len(df_t):,} rows, {time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    del df_t; gc.collect()

    # Book
    t0 = time.time()
    print(f"[{date_str}] Downloading book (FUTURES) …", end=" ", flush=True)
    try:
        df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                    start_date=date_str, end_date=date_str)
        df_b.to_parquet(book_path, index=False)
        print(f"OK ({len(df_b):,} rows, {time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    del df_b; gc.collect()

    # Liquidations
    t0 = time.time()
    print(f"[{date_str}] Downloading liquidations (FUTURES) …", end=" ", flush=True)
    try:
        df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                       start_date=date_str, end_date=date_str)
        if len(df_l) > 0:
            df_l.to_parquet(liq_path, index=False)
            print(f"OK ({len(df_l):,} rows, {time.time()-t0:.0f}s)")
        else:
            print("no liquidations in this window")
    except Exception as e:
        print(f"ERROR (non-fatal): {e}")

    return True


def main():
    print("=" * 60)
    print("Downloading IS futures data (7 dates) -> data/raw_futures/")
    print("=" * 60)
    for date_str in IS_DATES:
        ok = download_date(date_str, "data/raw_futures")
        if not ok:
            print(f"FAILED on IS date {date_str}, continuing...")

    print()
    print("=" * 60)
    print("Downloading OOS futures data (3 dates) -> data/raw_futures_oos/")
    print("=" * 60)
    for date_str in OOS_DATES:
        ok = download_date(date_str, "data/raw_futures_oos")
        if not ok:
            print(f"FAILED on OOS date {date_str}, continuing...")

    print()
    print("=" * 60)
    print("DONE. Summary:")
    print("=" * 60)
    for d in IS_DATES:
        p = Path(f"data/raw_futures/{d}")
        files = list(p.glob("*.parquet")) if p.exists() else []
        print(f"  IS  {d}: {len(files)} files")
    for d in OOS_DATES:
        p = Path(f"data/raw_futures_oos/{d}")
        files = list(p.glob("*.parquet")) if p.exists() else []
        print(f"  OOS {d}: {len(files)} files")


if __name__ == "__main__":
    main()