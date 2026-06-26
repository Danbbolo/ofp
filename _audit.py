"""Hard diagnostic of the OFP dataset to find what's broken.

Things to check:
1. Labels: are outcome_pct / outcome_binary actually correct for known moves?
2. Features: are they distinct across windows or all the same?
3. Windows: do they overlap, have leaks, or are they sparse?
4. Model: is the training correct? (e.g., target leakage between train and test?)
"""
import datetime as dt
import lightgbm as lgb
import numpy as np
import pandas as pd

print("=" * 70)
print("AUDIT 1: Basic dataset shape and label sanity")
print("=" * 70)

df = pd.read_parquet("data/research_dataset.parquet")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Pair counts:")
print(df.groupby(["window_size", "horizon"]).size())
print()

# Check label ranges
print("=" * 70)
print("AUDIT 2: Label distributions per (ws, hz)")
print("=" * 70)
for (ws, hz), grp in df.groupby(["window_size", "horizon"]):
    print(f"W={ws}s H={hz}s: n={len(grp):,}, "
          f"win_rate={grp['outcome_binary'].mean():.3f}, "
          f"avg_pct={grp['outcome_pct'].mean()*100:+.4f}%, "
          f"min_pct={grp['outcome_pct'].min()*100:+.4f}%, "
          f"max_pct={grp['outcome_pct'].max()*100:+.4f}%")
print()

# CRITICAL CHECK: are the labels correct?
# We know trader entry on 2026-06-17 13:00 (buy) had a +38.9% move.
# Find the row and check.
print("=" * 70)
print("AUDIT 3: Validate labels against known trader moves")
print("=" * 70)
KNOWN = [
    ("2026-06-17", "13:00", "buy", 0.389),   # +38.9% move
    ("2026-06-18", "04:30", "sell", -0.085),  # +8.5% (sell was wrong, but actual pct was +)
    ("2026-06-18", "07:00", "buy", 0.366),   # +36.6%
    ("2026-06-18", "19:15", "buy", 0.434),   # +43.4%
    ("2026-06-18", "22:45", "sell", 0.174),  # +17.4% (sell was wrong, but actual pct was +)
    ("2026-06-19", "06:45", "buy", -0.350),  # -35%
    ("2026-06-20", "03:30", "sell", 0.272),  # +27.2% (sell was wrong)
    ("2026-06-21", "06:30", "sell", -0.115), # -11.5%
    ("2026-06-22", "04:00", "sell", 0.244),  # +24.4% (sell was wrong)
    ("2026-06-22", "08:30", "buy", 0.252),   # +25.2%
    ("2026-06-23", "15:15", "buy", 0.154),   # +15.4%
]
for date_str, hhmm, dir, expected_pct in KNOWN:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    # Find rows within 5 min of this
    rows = df[
        (df["window_end_ms"] >= entry_ms - 5*60*1000) &
        (df["window_end_ms"] <= entry_ms + 5*60*1000)
    ]
    if len(rows) == 0:
        print(f"  {date_str} {hhmm} {dir}: NO ROWS FOUND (entry_ms={entry_ms})")
        continue
    # For each (ws, hz) match, show the outcome_pct
    print(f"  {date_str} {hhmm} {dir}: expected ≈{expected_pct*100:+.1f}%, found:")
    for (ws, hz), grp in rows.groupby(["window_size", "horizon"]):
        for _, row in grp.iterrows():
            actual = row["outcome_pct"] * 100
            ok = "OK" if abs(actual - expected_pct * 100) < 0.1 else "MISMATCH"
            print(f"    W={ws}s H={hz}s: actual={actual:+.4f}%  {ok}")
            break
    print()
