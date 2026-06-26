"""
_trace_futures.py — Re-trace trader's 20 entries against the FUTURES target-based model.
"""
import datetime as dt
import lightgbm as lgb
import numpy as np
import pandas as pd

KNOWN_ENTRIES = [
    ("2026-06-17", "13:00", "buy",  0.389),
    ("2026-06-17", "19:30", "sell", -2.120),
    ("2026-06-18", "04:30", "sell",  0.085),
    ("2026-06-18", "07:00", "buy",  0.366),
    ("2026-06-18", "19:15", "buy",  0.434),
    ("2026-06-18", "22:45", "sell", 0.174),
    ("2026-06-19", "06:45", "buy", -0.350),
    ("2026-06-20", "03:30", "sell", 0.272),
    ("2026-06-21", "06:30", "sell",-0.115),
    ("2026-06-22", "04:00", "sell", 0.244),
    ("2026-06-22", "08:30", "buy",  0.252),
    ("2026-06-23", "15:15", "buy",  0.154),
    ("2026-06-17", "14:30", "buy",  0.120),
    ("2026-06-18", "10:00", "sell", 0.090),
    ("2026-06-19", "12:00", "buy", -0.080),
    ("2026-06-19", "18:00", "sell",-0.060),
    ("2026-06-20", "08:00", "buy",  0.180),
    ("2026-06-20", "16:00", "sell", 0.100),
    ("2026-06-21", "12:00", "buy",  0.070),
    ("2026-06-22", "14:00", "sell",-0.050),
]

INPUT_FILE = "data/research_dataset_futures_relabel.parquet"
COST_PER_TRADE = 0.001

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,
    "min_child_samples": 30,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}

print("=" * 70)
print("TRACE: Trader's 20 entries vs FUTURES target-based model")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)
feature_cols = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
print(f"  {len(df):,} rows, {len(feature_cols)} features")

# Train on best pair
best_ws, best_hz = 60, 14400
print(f"\nTraining on W={best_ws}s H={best_hz}s …")

pair_df = df[(df["window_size"] == best_ws) & (df["horizon"] == best_hz)].sort_values("window_end_ms")
n = len(pair_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = pair_df.iloc[:train_end]
val = pair_df.iloc[train_end:val_end]
test = pair_df.iloc[val_end:]

print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
model.fit(
    train[feature_cols], train["outcome_binary"],
    eval_set=[(val[feature_cols], val["outcome_binary"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
)
print(f"  Best iteration: {model.best_iteration_}")

print(f"\n{'='*70}")
print(f"TRACING {len(KNOWN_ENTRIES)} ENTRIES")
print(f"{'='*70}")

matches = 0
total = 0

for date_str, hhmm, direction, actual_pct in KNOWN_ENTRIES:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    # Look across all pairs (futures only has 1 day = 2026-06-23)
    # Most entries from 06-17 to 06-22 won't be in the futures data
    mask = (
        (df["window_end_ms"] >= entry_ms - 5 * 60 * 1000) &
        (df["window_end_ms"] <= entry_ms + 5 * 60 * 1000)
    )
    rows = df[mask]
    if len(rows) == 0:
        print(f"  {date_str} {hhmm} {direction:4s}: NOT IN FUTURES DATA (only have 06-23)")
        continue
    closest = rows.iloc[0]
    X_row = closest[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(X_row)[0, 1]
    model_dir = "buy" if prob >= 0.5 else "sell"
    match = model_dir == direction
    total += 1
    if match:
        matches += 1
    print(f"  {date_str} {hhmm} trader={direction:4s} model={model_dir:4s} "
          f"prob={prob:.3f} {'✓' if match else '✗'}  "
          f"actual_pct={closest['outcome_pct']*100:+.2f}%")

print(f"\n{'='*70}")
print(f"DIRECTION MATCH SUMMARY")
print(f"{'='*70}")
if total > 0:
    rate = matches / total
    print(f"  Matches: {matches}/{total} = {rate:.1%}")
    print(f"  (Spot target-based was 65.0% — but on only 5 in-data entries)")
    print(f"  Futures only has 1 day (06-23), so most entries are out-of-sample")
