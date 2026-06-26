"""
run_research_daily.py — Download + sweep one day at a time, free memory aggressively.
"""
import sys
import gc
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cryptohftdata as chd

API_KEY = "2845d16a0479fc66dc89c01eccc8a3d3434e199828de1c8f168dacfca4a0e0ec"
EXCHANGE = chd.exchanges.BINANCE_FUTURES
SYMBOL = "BTCUSDT"
OUTPUT_ROOT = Path("data/raw_futures_oos")
DAILY_DIR = Path("data/oos_features")
COMBINED_FILE = Path("data/research_dataset_oos.parquet")

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


def download_day(client, date_str, day_dir):
    day_dir.mkdir(parents=True, exist_ok=True)
    if (day_dir / "trades.parquet").exists():
        print(f"  trades: cached", flush=True)
    else:
        print(f"  trades: downloading ...", end=" ", flush=True)
        df_t = client.get_trades(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
        df_t.to_parquet(day_dir / "trades.parquet", index=False)
        print(f"OK ({len(df_t):,})", flush=True)
        del df_t
        gc.collect()

    if (day_dir / "book.parquet").exists():
        print(f"  book: cached", flush=True)
    else:
        print(f"  book: downloading ...", flush=True)
        df_b = client.get_orderbook(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
        df_b.to_parquet(day_dir / "book.parquet", index=False)
        n = len(df_b)
        del df_b
        gc.collect()
        print(f"  book: OK ({n:,})", flush=True)

    if (day_dir / "liq.parquet").exists():
        print(f"  liq: cached", flush=True)
    else:
        print(f"  liq: downloading ...", end=" ", flush=True)
        try:
            df_l = client.get_liquidations(symbol=SYMBOL, exchange=EXCHANGE, start_date=date_str, end_date=date_str)
            if len(df_l) > 0:
                df_l.to_parquet(day_dir / "liq.parquet", index=False)
            else:
                pd.DataFrame(columns=["timestamp_ms"]).to_parquet(day_dir / "liq.parquet", index=False)
        except Exception as e:
            print(f"err {e}", flush=True)
            pd.DataFrame(columns=["timestamp_ms"]).to_parquet(day_dir / "liq.parquet", index=False)
        try:
            del df_l
        except:
            pass
        gc.collect()
        print("OK", flush=True)


def main(start_str, end_str):
    from datetime import datetime, timedelta
    from ofp.book_reconstructor import OrderBookReconstructor
    from ofp.grid_sweeper import GridSweeper

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    client = chd.CryptoHFTDataClient(api_key=API_KEY, max_workers=1)
    d = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    all_results = []
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        day_dir = OUTPUT_ROOT / date_str
        daily_feature_file = DAILY_DIR / f"{date_str}_features.parquet"

        print(f"\n{'='*60}")
        print(f"=== {date_str} ===")
        print(f"{'='*60}")

        if daily_feature_file.exists():
            print(f"  Features already extracted, loading...", flush=True)
            day_df = pd.read_parquet(daily_feature_file)
            all_results.append(day_df)
        else:
            # Download
            download_day(client, date_str, day_dir)

            # Load
            print(f"  Loading data...", flush=True)
            trades = _prepare_trades(pd.read_parquet(day_dir / "trades.parquet"))
            book_df = pd.read_parquet(day_dir / "book.parquet")
            try:
                liq_df = _prepare_liq_futures(pd.read_parquet(day_dir / "liq.parquet"))
            except:
                liq_df = pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
            print(f"  {len(trades):,} trades, {len(book_df):,} book rows, {len(liq_df):,} liq", flush=True)

            # Build book snapshots
            print(f"  Building book snapshots...", flush=True)
            recon = OrderBookReconstructor(max_depth=20)
            book_snapshots = {}
            sorted_book = book_df.sort_values("event_time").reset_index(drop=True)
            current_sec = None
            for _, row in sorted_book.iterrows():
                sec = int(row["event_time"]) // 1000
                if sec != current_sec:
                    if current_sec is not None:
                        book_snapshots[current_sec * 1000] = recon.snapshot()
                    current_sec = sec
                recon.apply(side=row["side"], price=float(row["price"]), quantity=float(row["quantity"]))
            if current_sec is not None:
                book_snapshots[current_sec * 1000] = recon.snapshot()
            del book_df, sorted_book, recon
            gc.collect()
            print(f"  {len(book_snapshots):,} snapshots", flush=True)

            # Per-zoom baselines
            data_duration_ms = int(trades["timestamp_ms"].iloc[-1]) - int(trades["timestamp_ms"].iloc[0])
            total_volume = float(trades["size"].sum())
            baseline = total_volume * 1000 / max(data_duration_ms, 1)
            rolling_stats_per_zoom = {
                "micro": {"rolling_avg_volume": baseline * 60 / 1000},
                "meso":  {"rolling_avg_volume": baseline * 300 / 1000},
                "macro": {"rolling_avg_volume": baseline * 1800 / 1000},
            }

            # Sweep
            print(f"  Sweeping...", flush=True)
            sweep = GridSweeper(WINDOW_SIZES_SEC, HORIZONS_SEC)
            day_results = []
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
                        day_results.append(row)
                        count += 1
                    if count > 0:
                        print(f"    W={window_sec} H={horizon_sec}: {count}", flush=True)
            day_df = pd.DataFrame(day_results)
            day_df.to_parquet(daily_feature_file, index=False)
            all_results.append(day_df)

            del book_snapshots, trades, liq_df, day_results
            gc.collect()

        d += timedelta(days=1)
        time.sleep(1)

    # Combine
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_parquet(COMBINED_FILE, index=False)
        print(f"\nCombined: {len(combined):,} rows saved to {COMBINED_FILE}")
        for (ws, hz), grp in combined.groupby(["window_size", "horizon"]):
            print(f"  W={ws} H={hz}: n={len(grp):,}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main("2026-06-24", "2026-06-30")
