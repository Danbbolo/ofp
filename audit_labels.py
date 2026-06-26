"""
audit_labels.py — verify that the dataset's outcome_pct matches the trader's records.

For each known entry, find the closest window_end_ms in the dataset, compute
the TRUE PnL using the raw trade data, and compare to what the dataset says.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
DATASET = Path("data/research_dataset.parquet")
TRADES_COLS = ["timestamp_ms", "price", "size", "is_buyer_maker"]


def load_trades_for_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load and concatenate trades from data/raw/ for the date range covering [start_ms, end_ms]."""
    start = dt.datetime.utcfromtimestamp(start_ms / 1000)
    end = dt.datetime.utcfromtimestamp(end_ms / 1000)
    chunks = []
    d = start
    while d.date() <= end.date():
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "trades.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
            df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
            df["price"] = df["price"].astype(float)
            df["size"] = df["size"].astype(float)
            chunks.append(df[TRADES_COLS])
        d += dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=TRADES_COLS)
    result = pd.concat(chunks, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    return result


def price_at(trades: pd.DataFrame, target_ms: int) -> float | None:
    """Return the trade price AT or BEFORE target_ms."""
    ts = trades["timestamp_ms"].values
    px = trades["price"].values
    idx = int(np.searchsorted(ts, target_ms, side="right")) - 1
    if idx < 0:
        return None
    return float(px[idx])


def future_pct_at(trades: pd.DataFrame, start_ms: int, max_horizon_ms: int) -> float | None:
    """Return the pct change from start_ms to the LAST trade price within [start_ms, start_ms+max_horizon_ms]."""
    ts = trades["timestamp_ms"].values
    px = trades["price"].values
    start_idx = int(np.searchsorted(ts, start_ms, side="left"))
    end_idx = int(np.searchsorted(ts, start_ms + max_horizon_ms, side="left"))
    if start_idx >= end_idx:
        return None
    p0 = float(px[start_idx])
    p1 = float(px[end_idx - 1])
    if p0 <= 0:
        return None
    return (p1 - p0) / p0


# Trader's known entries with the EXACT return he reported
KNOWN = [
    # (date, entry_hhmm, exit_hhmm, direction, trader_reported_pct)
    ("2026-06-17", "13:00", "18:15", "buy",  0.389,  "+38.9%"),
    ("2026-06-17", "04:30", "10:30", "sell", -0.115,  "sell: should be -11.5% if matched"),
    ("2026-06-17", "18:15", "21:30", "sell", -0.564,  "sell: should be -56.4% if matched"),
    ("2026-06-18", "00:00", "02:15", "buy",  -0.048,  "buy: trader was off, -4.8%"),
    ("2026-06-18", "04:30", "06:15", "sell", +0.236,  "sell: trader was off, +23.6%"),
    ("2026-06-18", "07:00", "10:00", "buy",  +0.366,  "+36.6%"),
    ("2026-06-18", "10:15", "17:45", "sell", -0.089,  "sell: -8.9%"),
    ("2026-06-18", "19:15", "22:00", "buy",  +0.434,  "+43.4%"),
    ("2026-06-18", "22:45", "23:59", "sell", +0.174,  "sell: was off, +17.4%"),
    ("2026-06-19", "06:45", "15:45", "buy",  -0.350,  "buy: was off, -35%"),
    ("2026-06-19", "18:00", "19:00", "sell", -0.191,  "sell: -19.1%"),
    ("2026-06-19", "20:30", "23:59", "buy",  +0.394,  "buy: +39.4%"),
    ("2026-06-20", "03:30", "04:45", "sell", +0.273,  "sell: was off, +27.3%"),
    ("2026-06-20", "16:00", "17:15", "buy",  -0.320,  "buy: was off, -32%"),
    ("2026-06-21", "06:30", "23:00", "sell", -0.115,  "sell: -11.5%"),
    ("2026-06-22", "01:30", "03:15", "buy",  -0.337,  "buy: was off, -33.7%"),
    ("2026-06-22", "04:00", "05:45", "sell", +0.244,  "sell: was off, +24.4%"),
    ("2026-06-22", "08:30", "15:00", "buy",  +0.252,  "buy: +25.2%"),
    ("2026-06-22", "17:15", "23:59", "sell", -0.296,  "sell: -29.6%"),
    ("2026-06-23", "15:15", "16:15", "buy",  +0.154,  "buy: +15.4%"),
]


def hhmm_to_ms(date_str: str, hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def main() -> None:
    print(f"Loading {DATASET} …", flush=True)
    df = pd.read_parquet(DATASET)
    print(f"  {len(df):,} rows", flush=True)
    print()

    # Determine date range for trade loading
    start_ms = int(df["window_end_ms"].min()) - 24 * 3_600_000
    end_ms = int(df["window_end_ms"].max()) + 24 * 3_600_000
    print(f"Loading trades from {RAW_DIR} for ms [{start_ms}, {end_ms}] …", flush=True)
    trades = load_trades_for_range(start_ms, end_ms)
    print(f"  {len(trades):,} trade rows", flush=True)
    print()

    print("=" * 100)
    print("LABEL AUDIT — comparing dataset outcome_pct against trader's records")
    print("=" * 100)
    print(f"{'Date':<11} {'Entry':<6} {'Exit':<6} {'Dir':<5} {'P0':>12} {'P1':>12} "
          f"{'TruePct':>9} {'DatasetPct':>11} {'Match':>6}")
    print("-" * 100)

    n_match = 0
    n_mismatch = 0
    for date_str, entry_hhmm, exit_hhmm, direction, expected_pct, note in KNOWN:
        entry_ms = hhmm_to_ms(date_str, entry_hhmm)
        exit_ms = hhmm_to_ms(date_str, exit_hhmm)
        if exit_ms < entry_ms:
            exit_ms += 24 * 3_600_000  # next day

        p0 = price_at(trades, entry_ms)
        p1 = price_at(trades, exit_ms)
        if p0 is None or p1 is None:
            print(f"{date_str:<11} {entry_hhmm:<6} {exit_hhmm:<6} {direction:<5} "
                  f"{'NA':>12} {'NA':>12} {'NA':>9} {'NA':>11} {'NA':>6}")
            continue
        true_pct = (p1 - p0) / p0

        # Find dataset row
        ds_rows = df[
            (df["window_end_ms"] >= entry_ms - 5 * 60 * 1000) &
            (df["window_end_ms"] <= entry_ms + 5 * 60 * 1000)
        ]
        if len(ds_rows) == 0:
            print(f"{date_str:<11} {entry_hhmm:<6} {exit_hhmm:<6} {direction:<5} "
                  f"{p0:>12.2f} {p1:>12.2f} {true_pct*100:>+8.3f}%  {'NO ROW':>11} {'NA':>6}  {note}")
            continue

        # The dataset has multiple (ws, hz) per entry. Show the W=60s, H=1800s (smallest) and H=14400s (largest)
        for _, row in ds_rows.iterrows():
            ws, hz = int(row["window_size"]), int(row["horizon"])
            if ws == 60 and hz == 1800:
                ds_pct = float(row["outcome_pct"]) * 100
                diff = ds_pct - true_pct * 100
                tag = "OK" if abs(diff) < 0.5 else "MISMATCH"
                if abs(diff) < 0.5:
                    n_match += 1
                else:
                    n_mismatch += 1
                print(f"{date_str:<11} {entry_hhmm:<6} {exit_hhmm:<6} {direction:<5} "
                      f"{p0:>12.2f} {p1:>12.2f} {true_pct*100:>+8.3f}%  "
                      f"{ds_pct:>+10.3f}%  {tag:>6}  W={ws}s H={hz}s  ({note})")
                break

    print()
    print(f"Match: {n_match}, Mismatch: {n_mismatch}")


if __name__ == "__main__":
    main()
