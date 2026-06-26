"""
run_research_oos.py — Out-of-sample sweep runner.

Downloads + sweeps one day at a time, frees memory between days,
concatenates results at the end.

For 06-24 to 06-30 (7 days unseen).
"""
import sys
import time
import gc
from pathlib import Path
import pandas as pd
import cryptohftdata as chd
import numpy as np

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures_oos")  # SEPARATE from old data
COMBINED_FILE = Path("data/research_dataset_oos.parquet")

from ofp.book_reconstructor import OrderBookReconstructor
from ofp.grid_sweeper import GridSweeper
import ofp.grid_sweeper as gs_module
import ofp.run_research_oos_helpers as helpers

WINDOW_SIZES_SEC = [60, 120, 180]
HORIZONS_SEC = [1800, 3600, 7200, 14400]

def _prepare_trades(df):
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    out = out[out["price"] > 0].reset_index(drop=True)
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]

def _prepare_liq_futures(df):
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = pd.DataFrame()
    out["timestamp_ms"] = df["event_time"].astype("int64")
    out["price"] = df["price"].astype(float)
    out["size"] = df["quantity"].astype(float)
    out["side"] = df["side"].str.lower()
    return out[["timestamp_ms", "side", "price", "size"]]

def download_one_day(client, date_str, day_dir):
    day_dir.mkdir(parents=True, exist_ok=True)
    if (day_dir / "trades.parquet").exists() and (day_dir / "book.parquet").exists() and (day_dir / "liq.parquet").exists():
        return True
    if not (day_dir / "trades.parquet").exists():
        print(f"  trades ...", end=" ", flush=True)
        df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
        df_t.to_parquet(day_dir / "trades.parquet", index=False)
        del df_t
        print("OK")
    if not (day_dir / "book.parquet").exists():
        print(f"  book ...", flush=True)
        df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
        df_b.to_parquet(day_dir / "book.parquet", index=False)
        del df_b
        gc.collect()
        print(f"  book OK")
    if not (day_dir / "liq.parquet").exists():
        print(f"  liq ...", end=" ", flush=True)
        try:
            df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
            if len(df_l) > 0:
                df_l.to_parquet(day_dir / "liq.parquet", index=False)
            else:
                pd.DataFrame(columns=["timestamp_ms"]).to_parquet(day_dir / "liq.parquet", index=False)
        except:
            pass
        del df_l
        print("OK")
    return True

def main(start_str, end_str):
    from datetime import datetime, timedelta
    client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)
    d = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    all_results = []
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        day_dir = OUTPUT_ROOT / date_str
        print(f"\n=== {date_str} ===")

        # Download
        download_one_day(client, date_str, day_dir)
        gc.collect()

        # Load
        trades = _prepare_trades(pd.read_parquet(day_dir / "trades.parquet"))
        book_df = pd.read_parquet(day_dir / "book.parquet")
        liq_df = _prepare_liq_futures(pd.read_parquet(day_dir / "liq.parquet"))
        print(f"  Loaded: {len(trades):,} trades, {len(book_df):,} book rows, {len(liq_df):,} liq")

        # Build book snapshots
        print("  Building book snapshots ...", flush=True)
        book_snapshots = helpers.build_book_snapshots(book_df, trades)
        del book_df
        gc.collect()
        print(f"  {len(book_snapshots):,} snapshots")

        # Per-zoom rolling baselines
        data_duration_ms = int(trades["timestamp_ms"].iloc[-1]) - int(trades["timestamp_ms"].iloc[0])
        total_volume = float(trades["size"].sum())
        baseline = total_volume * 1000 / data_duration_ms
        rolling_stats_per_zoom = {
            "micro": {"rolling_avg_volume": baseline * 60 / 1000},
            "meso":  {"rolling_avg_volume": baseline * 300 / 1000},
            "macro": {"rolling_avg_volume": baseline * 1800 / 1000},
        }

        # Sweep
        print("  Sweeping ...", flush=True)
        sweep = GridSweeper(WINDOW_SIZES_SEC, HORIZONS_SEC)
        for window_sec in WINDOW_SIZES_SEC:
            for horizon_sec in HORIZONS_SEC:
                count = 0
                for row in sweep.sweep(
                    trades_df=trades,
                    book_snapshots=book_snapshots,
                    liq_df=liq_df,
                    micro_window_ms=window_sec * 1000,
                    meso_window_ms=300 * 1000,
                    macro_window_ms=1800 * 1000,
                    rolling_stats_per_zoom=rolling_stats_per_zoom,
                ):
                    all_results.append(row)
                    count += 1
                    if count % 10000 == 0:
                        print(f"    W={window_sec} H={horizon_sec}: {count}", flush=True)
        del book_snapshots, trades, liq_df
        gc.collect()

        d += timedelta(days=1)
        time.sleep(1)

    # Save combined
    print(f"\nSaving {len(all_results):,} rows to {COMBINED_FILE}")
    df = pd.DataFrame(all_results)
    df.to_parquet(COMBINED_FILE, index=False)
    print(f"  Pair counts:")
    for (ws, hz), grp in df.groupby(["window_size", "horizon"]):
        print(f"    W={ws} H={hz}: n={len(grp):,}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main("2026-06-24", "2026-06-30")
