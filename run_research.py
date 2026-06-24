"""
run_research.py — Fetch 1 day of BTCUSDT data and run a full grid sweep.

- ONE datatype conversion pass at the start.
- Pre-builds 1-second book snapshots into a dict (ms-keyed).
- GridSweeper does binary-search slicing only.

Usage::

    python run_research.py 2026-06-23

Output: ``data/research_dataset.parquet``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cryptohftdata as chd
import pandas as pd

from ofp.book_reconstructor import OrderBookReconstructor
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
PROGRESS_EVERY = 10_000


# ---------------------------------------------------------------------------
# Pre-conversion helpers  (ONE pass, string→float/int64)
# ---------------------------------------------------------------------------

def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    """trade_time (ms) → timestamp_ms.  price/quantity str → float."""
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]


def _prepare_liq(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp → timestamp_ms.  quantity → size.  price str → float."""
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = df.rename(columns={"timestamp": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out.get("price", pd.Series([0.0] * len(out))).astype(float)
    out["size"] = out.get("size", pd.Series([0.0] * len(out))).astype(float)
    return out[["timestamp_ms", "side", "price", "size"]]


def _build_book_snapshots(book_raw: pd.DataFrame) -> dict[int, tuple[list, list]]:
    """
    ONE streaming pass: apply all 28M book deltas → 1-second snapshots.

    Returns a dict keyed by **millisecond** timestamp.
    Uses OrderBookReconstructor directly (avoids pandas groupby overhead).
    """
    print("  Building 1s book snapshots …", flush=True)

    # Subset + convert types in one go
    book = book_raw[["event_time", "event_type", "side", "price", "quantity"]].copy()
    book["price"] = book["price"].astype(float)
    book["quantity"] = book["quantity"].astype(float)

    recon = OrderBookReconstructor()
    snapshots: dict[int, tuple[list, list]] = {}
    current_bucket_ms = -1

    for row in book.itertuples(index=False):
        ts_ms = int(row.event_time)          # already ms
        bucket = ts_ms // 1000               # 1-second bucket

        if bucket != current_bucket_ms and current_bucket_ms != -1:
            snapshots[current_bucket_ms * 1000] = recon.top_n(20)
        current_bucket_ms = bucket

        if row.event_type == "snapshot":
            recon.clear()
        recon.apply(side=row.side, price=row.price, quantity=row.quantity)

    # Final bucket
    if current_bucket_ms != -1:
        snapshots[current_bucket_ms * 1000] = recon.top_n(20)

    print(f"  {len(snapshots):,} snapshots built.", flush=True)
    return snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(date_str: str) -> None:
    print(f"Fetching {SYMBOL} data for {date_str} …")
    print(f"  Windows: {WINDOW_SIZES_SEC}")
    print(f"  Horizons: {HORIZONS_SEC}")

    client = chd.CryptoHFTDataClient(api_key=API_KEY)

    # --- Fetch ---
    print("  Downloading trades …", flush=True)
    trades_raw = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE,
                                   start_date=date_str, end_date=date_str)

    print("  Downloading orderbook …", flush=True)
    book_raw = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE,
                                    start_date=date_str, end_date=date_str)

    print("  Downloading liquidations …", flush=True)
    liq_raw = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE,
                                      start_date=date_str, end_date=date_str)

    print(f"  Trades:       {len(trades_raw):,} rows")
    print(f"  Book deltas:  {len(book_raw):,} rows")
    print(f"  Liquidations: {len(liq_raw):,} rows")

    if trades_raw.empty:
        print("ERROR: No trade data fetched.")
        sys.exit(1)

    # --- ONE conversion pass ---
    print("  Converting types …", flush=True)
    trades_df = _prepare_trades(trades_raw)
    liq_df = _prepare_liq(liq_raw)

    # Sort (needed for binary search)
    trades_df = trades_df.sort_values("timestamp_ms").reset_index(drop=True)
    liq_df = liq_df.sort_values("timestamp_ms").reset_index(drop=True)

    # --- Pre-build book snapshots ---
    book_snapshots = _build_book_snapshots(book_raw)

    # --- Compute globals ---
    rolling_avg_volume = float(trades_df["size"].sum() / max(len(trades_df), 1))
    _24h_stats = {
        "_24h_avg_range": float(trades_df["price"].max() - trades_df["price"].min()),
        "_24h_low": float(trades_df["price"].min()),
        "_24h_high": float(trades_df["price"].max()),
    }
    print(f"  Rolling avg volume: {rolling_avg_volume:.4f}")
    print(f"  24h range: {_24h_stats['_24h_low']:.2f} – {_24h_stats['_24h_high']:.2f}")
    print("  Sweeping …", flush=True)

    # --- Sweep ---
    sweeper = GridSweeper(window_sizes_sec=WINDOW_SIZES_SEC,
                          horizons_sec=HORIZONS_SEC)
    gen = sweeper.sweep(
        trades_df=trades_df,
        book_snapshots=book_snapshots,
        liq_df=liq_df,
        rolling_avg_volume=rolling_avg_volume,
        _24h_stats=_24h_stats,
    )

    def _progress(iterator):
        n = 0
        for row in iterator:
            yield row
            n += 1
            if n % PROGRESS_EVERY == 0:
                print(f"    Processed {n:,} windows …", flush=True)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GridSweeper.save_to_disk(_progress(gen), str(OUTPUT_FILE))

    result = pd.read_parquet(OUTPUT_FILE)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Done.")
    print(f"  File:  {OUTPUT_FILE.resolve()}")
    print(f"  Size:  {size_mb:.2f} MB")
    print(f"  Rows:  {len(result):,}")
    print(f"  Cols:  {list(result.columns)}")
    print()
    print(result.head(5))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD")
        sys.exit(1)
    main(sys.argv[1])
