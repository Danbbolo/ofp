"""
download_raw_data.py — Download 30 days of BTCUSDT data, validate, save locally.

Usage::

    python download_raw_data.py 2026-06-01 2026-06-30

Saves to ``data/raw/YYYY-MM-DD/{trades,book,liq}.parquet``.
Validates each download; retries once on validation failure.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import cryptohftdata as chd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_SPOT
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw")
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_trades(df: pd.DataFrame, date_str: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        errors.append("empty DataFrame")
        return errors

    if df["trade_time"].isna().any():
        errors.append("null trade_time")
    if df["price"].isna().any():
        errors.append("null price")
    if df["quantity"].isna().any():
        errors.append("null quantity")

    # is_buyer_maker must be boolean-like (True/False or 0/1)
    bm = df["is_buyer_maker"]
    if not bm.isin([True, False, 0, 1]).all():
        errors.append("is_buyer_maker not boolean")

    # Timestamps should be sortable (not necessarily monotonic, but not garbage)
    ts = df["trade_time"].dropna().values
    if len(ts) > 1:
        if (ts.max() - ts.min()) > 86_400_000 * 2:  # >2 days span
            errors.append(f"trade_time span too large: {ts.max() - ts.min()} ms")

    return errors


def _validate_book(df: pd.DataFrame, date_str: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        errors.append("empty DataFrame")
        return errors

    if df["event_time"].isna().any():
        errors.append("null event_time")
    if df["price"].isna().any():
        errors.append("null price")
    if df["quantity"].isna().any():
        errors.append("null quantity")
    if not df["side"].isin(["bid", "ask"]).all():
        errors.append("invalid side values")
    if not df["event_type"].isin(["snapshot", "update"]).all():
        errors.append("invalid event_type values")

    ts = df["event_time"].dropna().values
    if len(ts) > 1:
        if (ts.max() - ts.min()) > 86_400_000 * 2:
            errors.append(f"event_time span too large: {ts.max() - ts.min()} ms")

    return errors


def _validate_liq(df: pd.DataFrame, date_str: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return errors  # empty liq is OK

    if df["timestamp"].isna().any():
        errors.append("null timestamp")
    if df["price"].isna().any():
        errors.append("null price")
    if df["quantity"].isna().any():
        errors.append("null quantity")

    return errors


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_day(date_str: str) -> bool:
    """Download one day.  Returns True on success."""
    day_dir = OUTPUT_ROOT / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        ok = True
        client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)

        # --- Trades ---
        print(f"    Trades …", end=" ", flush=True)
        try:
            df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                     start_date=date_str, end_date=date_str)
        except Exception as e:
            print(f"DOWNLOAD ERROR: {e}")
            ok = False
            df_t = pd.DataFrame()

        if ok:
            errs = _validate_trades(df_t, date_str)
            if errs:
                print(f"VALIDATION FAILED: {errs}")
                ok = False
            else:
                df_t.to_parquet(day_dir / "trades.parquet", index=False)
                print(f"OK ({len(df_t):,} rows)", end="  ")
        del df_t; gc.collect()  # free before book download

        if not ok and attempt < MAX_RETRIES:
            print("  RETRYING …")
            continue

        # --- Book ---
        print(f"Book …", end=" ", flush=True)
        try:
            df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                        start_date=date_str, end_date=date_str)
        except Exception as e:
            print(f"DOWNLOAD ERROR: {e}")
            if attempt < MAX_RETRIES:
                continue
            return False

        errs = _validate_book(df_b, date_str)
        if errs:
            print(f"VALIDATION FAILED: {errs}")
            if attempt < MAX_RETRIES:
                print("  RETRYING …")
                ok = False
                continue
            return False
        df_b.to_parquet(day_dir / "book.parquet", index=False)
        print(f"OK ({len(df_b):,} rows)", end="  ")
        del df_b; gc.collect()

        # --- Liquidations ---
        print(f"Liq …", end=" ", flush=True)
        try:
            df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                           start_date=date_str, end_date=date_str)
        except Exception as e:
            print(f"DOWNLOAD ERROR: {e}")
            df_l = pd.DataFrame()

        errs = _validate_liq(df_l, date_str)
        if errs:
            print(f"VALIDATION FAILED: {errs}")
            if attempt < MAX_RETRIES:
                ok = False
                continue
        else:
            df_l.to_parquet(day_dir / "liq.parquet", index=False)
            print(f"OK ({len(df_l):,} rows)")

        if ok:
            return True

    print(f"  FAILED after {MAX_RETRIES} attempts — skipping {date_str}")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_str: str, end_str: str) -> None:
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    print(f"Downloading {SYMBOL} from {start_str} to {end_str}")
    print(f"  Output: {OUTPUT_ROOT.resolve()}")
    print()

    success = 0
    total = 0

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        total += 1
        print(f"[{date_str}]", flush=True)

        if download_day(date_str):
            success += 1

        # Free memory between days
        import gc
        gc.collect()

        d += timedelta(days=1)

    print(f"\nDone.  {success}/{total} days downloaded successfully.")
    print(f"Files in {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
