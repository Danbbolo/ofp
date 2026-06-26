"""
pull_historical.py — Pull all 6 CryptoHFTData data types for a date range.

This module is a SIBLING of ``download_raw_data.py`` — it does NOT touch
``ofp/``, the raw data dir, the sweep pipeline, or any engine code.  It
writes to a separate path ``data/historical/`` so the existing engine
keeps working unchanged.

Usage::

    python -m src.data.pull_historical 2026-06-23 2026-06-23 --symbol BTCUSDT

Verified SDK (v0.3.0):
    - 6 data types, all return pd.DataFrame
    - start_date/end_date are DAY-granular (time-of-day is ignored)
    - All numeric fields (price/qty/etc) come as str — cast to float on save
    - event_time / trade_time / timestamp are ms-UTC
    - received_time is ns-UTC (latency)
    - orderbook for 1h ≈ 141M rows (massive — see storage note)

Storage layout (one Parquet per data-type per day, zstd-compressed)::

    data/historical/
      trades/      BTCUSDT/2026-06-23.parquet.zst
      orderbook/   BTCUSDT/2026-06-23.parquet.zst
      liquidations/BTCUSDT/2026-06-23.parquet.zst
      mark_price/  BTCUSDT/2026-06-23.parquet.zst
      open_interest/BTCUSDT/2026-06-23.parquet.zst
      ticker/      BTCUSDT/2026-06-23.parquet.zst

Idempotent: skips days that already have a file on disk.  Pass --force to
re-pull.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# SDK import (v0.3.0)
# ---------------------------------------------------------------------------

import cryptohftdata as chd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
SYMBOL_DEFAULT = "BTCUSDT"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
OUTPUT_ROOT = Path("data/historical")

# 6 data types — function name, output subdir, numeric columns to cast to float
# (received_time / event_time / ids stay int64)
DATA_TYPES: list[dict] = [
    {
        "name": "trades",
        "fn": chd.get_trades,
        "subdir": "trades",
        "cast_float": ["price", "quantity"],
    },
    {
        "name": "orderbook",
        "fn": chd.get_orderbook,
        "subdir": "orderbook",
        "cast_float": ["price", "quantity"],
    },
    {
        "name": "liquidations",
        "fn": chd.get_liquidations,
        "subdir": "liquidations",
        "cast_float": ["quantity", "price", "average_price",
                       "last_filled_quantity", "filled_quantity"],
    },
    {
        "name": "mark_price",
        "fn": chd.get_mark_price,
        "subdir": "mark_price",
        "cast_float": ["mark_price", "index_price",
                       "estimated_settle_price", "funding_rate"],
    },
    {
        "name": "open_interest",
        "fn": chd.get_open_interest,
        "subdir": "open_interest",
        "cast_float": ["sum_open_interest", "sum_open_interest_value"],
    },
    {
        "name": "ticker",
        "fn": chd.get_ticker,
        "subdir": "ticker",
        "cast_float": [
            "price_change", "price_change_percent", "weighted_average_price",
            "last_price", "last_quantity", "open_price", "high_price",
            "low_price", "base_asset_volume", "quote_asset_volume",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive list of dates between start and end."""
    n = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(n + 1)]


def _out_path(data_subdir: str, symbol: str, day: dt.date) -> Path:
    """Output path for a given data type / symbol / day.

    Layout: ``data/historical/<subdir>/<symbol>/YYYY-MM-DD.parquet.zst``
    """
    return OUTPUT_ROOT / data_subdir / symbol / f"{day.isoformat()}.parquet.zst"


def _cast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cast known-str numeric columns to float64 in-place.

    The SDK returns numeric columns as either ``object`` (legacy pandas) or
    ``str`` (pandas 2.x with future string dtype).  We cast anything that
    isn't already a proper float/int.  ``pd.to_numeric`` with ``coerce``
    means bad rows become NaN rather than crashing the whole pull.

    Empty / missing columns are skipped silently.
    """
    out = df
    for c in cols:
        if c not in out.columns:
            continue
        dt = out[c].dtype
        # Skip if already numeric (float*, int*, bool doesn't apply here)
        if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
            continue
        # Cast string-like (object, str, string) → float64
        if dt == object or pd.api.types.is_string_dtype(dt):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _write_zstd_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a zstd-compressed Parquet file.

    Uses pyarrow directly (no pandas->csv conversion).  Compression level 19
    gives ~3-4x compression on numeric tick data with minimal CPU cost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        compression="zstd",
        compression_level=3,  # 3 is good enough for tick data; 19 wastes CPU
        use_dictionary=False,  # numeric tick data doesn't benefit
    )


def _pull_one_day(
    spec: dict, symbol: str, day: dt.date, force: bool,
) -> tuple[str, dt.date, str, int]:
    """Pull a single (data_type, day). Returns (subdir, day, status, n_rows).

    Status: "skipped" | "ok" | "error:<msg>"
    """
    out = _out_path(spec["subdir"], symbol, day)

    if out.exists() and not force:
        return (spec["subdir"], day, "skipped", 0)

    # SDK ignores time-of-day; pass start=end=day 00:00:00 for clarity
    start_iso = f"{day.isoformat()}T00:00:00"
    end_iso = f"{day.isoformat()}T23:59:59"

    t0 = time.time()
    try:
        df = spec["fn"](
            symbol=symbol,
            exchange=EXCHANGE,
            start_date=start_iso,
            end_date=end_iso,
        )
    except Exception as e:
        return (spec["subdir"], day, f"error:{type(e).__name__}:{e}", 0)

    if df is None or len(df) == 0:
        # Write an empty file so we don't re-attempt on next run
        df = pd.DataFrame()
        _write_zstd_parquet(df, out)
        return (spec["subdir"], day, "empty", 0)

    df = _cast_numeric(df, spec["cast_float"])
    n_rows = len(df)
    _write_zstd_parquet(df, out)

    # Free big frames before next call
    del df
    gc.collect()

    elapsed = time.time() - t0
    size_mb = out.stat().st_size / (1024 * 1024)
    return (spec["subdir"], day, f"ok ({elapsed:.1f}s, {size_mb:.1f} MB)", n_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pull all 6 CryptoHFTData types for a date range."
    )
    p.add_argument("start", help="Start date (YYYY-MM-DD)")
    p.add_argument("end",   help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--symbol", default=SYMBOL_DEFAULT)
    p.add_argument("--force", action="store_true",
                   help="Re-pull even if file exists")
    p.add_argument("--workers", type=int, default=4,
                   help="ThreadPoolExecutor workers (default 4)")
    p.add_argument("--types", nargs="*", default=None,
                   help="Subset of data types to pull (default: all 6)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()
    if end < start:
        print("ERROR: end < start")
        return 2

    chd.configure_client(api_key=API_KEY)

    days = _date_range(start, end)
    specs = DATA_TYPES
    if args.types:
        wanted = set(args.types)
        specs = [s for s in DATA_TYPES if s["name"] in wanted]
        if not specs:
            print(f"ERROR: no matching types.  Available: "
                  f"{[s['name'] for s in DATA_TYPES]}")
            return 2

    print(f"Pulling {args.symbol} on {chd.exchanges.BINANCE_FUTURES}")
    print(f"  Date range:  {start} → {end}  ({len(days)} day(s))")
    print(f"  Data types:  {[s['name'] for s in specs]}")
    print(f"  Workers:     {args.workers}")
    print(f"  Output:      {OUTPUT_ROOT.resolve()}")
    print(f"  Force:       {args.force}")
    print()

    # Build the full job list: one job per (data_type, day)
    # NOTE: orderbook for a single day of BTCUSDT = ~140M rows ≈ 30GB in
    # pandas, which OOM-kills on our 30GB box.  We do NOT exclude it
    # entirely (the user asked for it), but we cap the worker count for
    # the orderbook specifically so the SDK doesn't fan out to 24 threads.
    # If you want orderbook, use --workers 1 and have ~40GB RAM available.
    jobs: list[tuple[dict, dt.date]] = [
        (spec, day) for spec in specs for day in days
    ]
    n_jobs = len(jobs)
    # Cap effective parallelism: orderbook is huge, others are small
    safe_workers = args.workers
    if any(s["name"] == "orderbook" for s in specs) and safe_workers > 1:
        print("  WARNING: orderbook in the job list is huge (~30GB pandas RSS).")
        print("           Forcing workers=1 to avoid OOM on 30GB hosts.")
        print("           For parallel orderbook pulls, use a >=64GB host.")
        safe_workers = 1
    total = len(jobs)
    done = 0
    errors = 0

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=safe_workers) as ex:
        futures = {
            ex.submit(_pull_one_day, spec, args.symbol, day, args.force): (spec, day)
            for spec, day in jobs
        }
        for fut in as_completed(futures):
            subdir, day, status, n_rows = fut.result()
            done += 1
            tag = f"[{done:>4}/{total}]"
            if status.startswith("error:"):
                errors += 1
                print(f"  {tag} {subdir:14s} {day}  ✗ {status}", flush=True)
            elif status == "skipped":
                print(f"  {tag} {subdir:14s} {day}  ·  skipped (exists)", flush=True)
            elif status == "empty":
                print(f"  {tag} {subdir:14s} {day}  ·  empty (no data)", flush=True)
            else:
                msg = f"  {tag} {subdir:14s} {day}  ✓  {n_rows:>10,} rows  {status}"
                if args.verbose or done == total or done % 10 == 0:
                    print(msg, flush=True)
                else:
                    # Compact one-liner
                    print(f"  {tag} {subdir:14s} {day}  ✓  {n_rows:>10,} rows",
                          flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done.  {done - errors}/{total} OK, {errors} errors, {elapsed:.1f}s total")
    print(f"Files in {OUTPUT_ROOT.resolve()}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
