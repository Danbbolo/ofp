"""
run_research.py — Fetch 1 day of BTCUSDT data and run a full grid sweep.

Usage::

    python run_research.py 2026-06-23

Set ``CRYPTOHFT_API_KEY`` in the environment before running.
Output is written to ``data/research_dataset.parquet``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ofp.api_streamer import CryptoHFTStreamer
from ofp.grid_sweeper import GridSweeper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_SIZES_SEC = [60, 120, 180, 300, 600]
HORIZONS_SEC = [300, 900, 1800, 3600, 14400]
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "research_dataset.parquet"
PROGRESS_EVERY = 10_000  # rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_to_ns(date_str: str) -> tuple[int, int]:
    """Parse ``YYYY-MM-DD`` and return (start_ns, end_ns) for that UTC day."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ns = int(dt.timestamp() * 1_000_000_000)
    end_ns = start_ns + 86_400_000_000_000  # +24 h in ns
    return start_ns, end_ns


async def _fetch_all(
    streamer: CryptoHFTStreamer,
    symbol: str,
    data_type: str,
    start_ns: int,
    end_ns: int,
) -> pd.DataFrame:
    """Fetch all pages for *data_type* and concatenate into one DataFrame."""
    chunks: list[pd.DataFrame] = []
    async for df in streamer.fetch_data(
        symbol=symbol,
        data_type=data_type,  # type: ignore[arg-type]
        start_time=start_ns,
        end_time=end_ns,
    ):
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match ``extract_features`` expectations."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp_ms", "price", "size", "is_buyer_maker"])
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = (out["timestamp_ms"] // 1_000_000).astype("int64")
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]


def _prepare_liq(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match ``extract_features`` expectations."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = (out["timestamp_ms"] // 1_000_000).astype("int64")
    return out[["timestamp_ms", "side", "price", "size"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(date_str: str) -> None:
    api_key = os.environ.get("CRYPTOHFT_API_KEY")
    if not api_key:
        print("ERROR: CRYPTOHFT_API_KEY environment variable is not set.")
        sys.exit(1)

    start_ns, end_ns = _day_to_ns(date_str)

    print(f"Fetching BTCUSDT data for {date_str} …")
    print(f"  Window sizes (sec): {WINDOW_SIZES_SEC}")
    print(f"  Horizons     (sec): {HORIZONS_SEC}")

    async with CryptoHFTStreamer(api_key=api_key) as streamer:
        # Fetch all three data types
        trades_raw, book_raw, liq_raw = await asyncio.gather(
            _fetch_all(streamer, "BTCUSDT", "trades", start_ns, end_ns),
            _fetch_all(streamer, "BTCUSDT", "book_snapshot", start_ns, end_ns),
            _fetch_all(streamer, "BTCUSDT", "liquidations", start_ns, end_ns),
        )

    print(f"  Trades:       {len(trades_raw):,} rows")
    print(f"  Book deltas:  {len(book_raw):,} rows")
    print(f"  Liquidations: {len(liq_raw):,} rows")

    if trades_raw.empty:
        print("ERROR: No trade data fetched.  Check the date and API key.")
        sys.exit(1)

    # Prepare DataFrames
    trades_df = _prepare_trades(trades_raw)
    liq_df = _prepare_liq(liq_raw)

    # Compute rolling average volume (simple: mean trade size × trade count per second)
    rolling_avg_volume = float(trades_df["size"].sum() / max(len(trades_df), 1))

    # 24h stats from the full day
    _24h_stats = {
        "_24h_avg_range": float(trades_df["price"].max() - trades_df["price"].min()),
        "_24h_low": float(trades_df["price"].min()),
        "_24h_high": float(trades_df["price"].max()),
    }

    print(f"  Rolling avg volume: {rolling_avg_volume:.4f}")
    print(f"  24h high/low/range: {_24h_stats['_24h_high']:.2f} / "
          f"{_24h_stats['_24h_low']:.2f} / {_24h_stats['_24h_avg_range']:.2f}")
    print("Sweeping …")

    # Sweep
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

    # Wrap with progress reporter
    def _progress_wrapper(iterator):
        count = 0
        for row in iterator:
            yield row
            count += 1
            if count % PROGRESS_EVERY == 0:
                print(f"  Processed {count:,} windows …")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GridSweeper.save_to_disk(_progress_wrapper(gen), str(OUTPUT_FILE))
    print(f"Done.  Output written to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
