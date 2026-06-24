"""
run_research.py — Fetch 1 day of BTCUSDT data and run a full grid sweep.

Uses the ``cryptohftdata`` Python client to pull historical trades,
order-book deltas, and liquidations, then feeds them through the
``GridSweeper``.

Usage::

    python run_research.py 2026-06-23

Output: ``data/research_dataset.parquet``
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cryptohftdata as chd
import pandas as pd

from ofp.grid_sweeper import GridSweeper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_SPOT
SYMBOL = "BTCUSDT"

WINDOW_SIZES_SEC = [60, 120, 180, 300, 600]
HORIZONS_SEC = [300, 900, 1800, 3600, 14400]
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "research_dataset.parquet"
PROGRESS_EVERY = 10_000  # rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cryptohftdata trades DataFrame to ``extract_features`` format.

    Input:  received_time, event_time, symbol, trade_id, price(str),
            quantity(str), trade_time, is_buyer_maker, order_type
    Output: timestamp_ms, price, size, is_buyer_maker

    NOTE: ``trade_time`` is already in **milliseconds** (13-digit Unix ms).
    """
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]


def _prepare_book(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cryptohftdata orderbook DataFrame for OrderBookReconstructor.

    Keeps only the columns needed: event_time, event_type, side, price, quantity.
    Converts price/quantity from string to float.

    NOTE: ``event_time`` arrives in **milliseconds** but ``OrderBookReconstructor``
    expects nanoseconds, so we multiply by 1_000_000.
    """
    out = df[["event_time", "event_type", "side", "price", "quantity"]].copy()
    out["event_time"] = (out["event_time"] * 1_000_000).astype("int64")
    out["price"] = out["price"].astype(float)
    out["quantity"] = out["quantity"].astype(float)
    return out


def _prepare_liq(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cryptohftdata liquidations DataFrame for extract_features.

    Input may have columns: timestamp, side, price, quantity, order_id
    Output: timestamp_ms, side, price, size
    """
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = df.rename(columns={"timestamp": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    if "price" in out.columns:
        out["price"] = out["price"].astype(float)
    else:
        out["price"] = 0.0
    if "size" in out.columns:
        out["size"] = out["size"].astype(float)
    else:
        out["size"] = 0.0
    return out[["timestamp_ms", "side", "price", "size"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(date_str: str) -> None:
    print(f"Fetching {SYMBOL} data for {date_str} via cryptohftdata …")
    print(f"  Window sizes (sec): {WINDOW_SIZES_SEC}")
    print(f"  Horizons     (sec): {HORIZONS_SEC}")

    client = chd.CryptoHFTDataClient(api_key=API_KEY)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    print("  Downloading trades …")
    trades_raw = client.get_trades(
        symbol=SYMBOL, exchange=EXCHANGE,
        start_date=date_str, end_date=date_str,
    )

    print("  Downloading orderbook …")
    book_raw = client.get_orderbook(
        symbol=SYMBOL, exchange=EXCHANGE,
        start_date=date_str, end_date=date_str,
    )

    print("  Downloading liquidations …")
    liq_raw = client.get_liquidations(
        symbol=SYMBOL, exchange=EXCHANGE,
        start_date=date_str, end_date=date_str,
    )

    print(f"  Trades:       {len(trades_raw):,} rows")
    print(f"  Book deltas:  {len(book_raw):,} rows")
    print(f"  Liquidations: {len(liq_raw):,} rows")

    if trades_raw.empty:
        print("ERROR: No trade data fetched.  Check the date and API key.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------
    print("  Preparing DataFrames …")
    trades_df = _prepare_trades(trades_raw)
    liq_df = _prepare_liq(liq_raw)

    # Book data is passed as-is — GridSweeper accesses columns directly
    # and converts price/quantity to float on-the-fly in the streaming loop.

    rolling_avg_volume = float(trades_df["size"].sum() / max(len(trades_df), 1))

    _24h_stats = {
        "_24h_avg_range": float(trades_df["price"].max() - trades_df["price"].min()),
        "_24h_low": float(trades_df["price"].min()),
        "_24h_high": float(trades_df["price"].max()),
    }

    print(f"  Rolling avg volume: {rolling_avg_volume:.4f}")
    print(f"  24h high/low/range: {_24h_stats['_24h_high']:.2f} / "
          f"{_24h_stats['_24h_low']:.2f} / {_24h_stats['_24h_avg_range']:.2f}")
    print("  Sweeping …")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    sweeper = GridSweeper(
        window_sizes_sec=WINDOW_SIZES_SEC,
        horizons_sec=HORIZONS_SEC,
    )
    gen = sweeper.sweep(
        trades_df=trades_df,
        book_df=book_raw,
        liq_df=liq_df,
        rolling_avg_volume=rolling_avg_volume,
        _24h_stats=_24h_stats,
    )

    # Progress reporter
    def _progress_wrapper(iterator):
        count = 0
        for row in iterator:
            yield row
            count += 1
            if count % PROGRESS_EVERY == 0:
                print(f"    Processed {count:,} windows …")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GridSweeper.save_to_disk(_progress_wrapper(gen), str(OUTPUT_FILE))

    # Report
    result = pd.read_parquet(OUTPUT_FILE)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Done.")
    print(f"  File:  {OUTPUT_FILE.resolve()}")
    print(f"  Size:  {file_size_mb:.2f} MB")
    print(f"  Rows:  {len(result):,}")
    print(f"  Cols:  {list(result.columns)}")
    print()
    print(result.head(5))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD")
        sys.exit(1)
    main(sys.argv[1])
