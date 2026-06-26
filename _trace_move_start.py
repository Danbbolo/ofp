"""
_trace_move_start.py — Re-trace trader's 20 entries on the move-start dataset.

For each entry, extract features from the 15min BEFORE entry.
Check: did the model predict had_move=1? What was the actual outcome?
What do winning entries have in common at entry time?
"""
import datetime as dt
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

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

INPUT_FILE = "data/research_dataset_futures_move_start.parquet"
COST_PER_TRADE = 0.001

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 50,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "had_move", "move_direction", "move_pct"}

print("=" * 70)
print("TRACE: Trader's 20 entries vs MOVE-START model")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)
feature_cols = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
print(f"  {len(df):,} rows, {len(feature_cols)} features")

# Train on best pair (W=60, H=3600 — the top performer)
best_ws, best_hz = 60, 3600
print(f"\nTraining on W={best_ws}s H={best_hz}s …")

pair_df = df[(df["window_size"] == best_ws) & (df["horizon"] == best_hz)].sort_values("window_end_ms")
n = len(pair_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = pair_df.iloc[:train_end]
val = pair_df.iloc[train_end:val_end]
test = pair_df.iloc[val_end:]

print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
model.fit(
    train[feature_cols], train["outcome_binary"],
    eval_set=[(val[feature_cols], val["outcome_binary"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
)
print(f"  Best iteration: {model.best_iteration_}")

print(f"\n{'='*70}")
print(f"TRACING {len(KNOWN_ENTRIES)} ENTRIES")
print(f"{'='*70}")

results = []
for date_str, hhmm, direction, actual_pct in KNOWN_ENTRIES:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    entry_ms = int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    # Find closest row in the pair's dataset
    mask = (
        (pair_df["window_end_ms"] >= entry_ms - 5 * 60 * 1000) &
        (pair_df["window_end_ms"] <= entry_ms + 5 * 60 * 1000)
    )
    rows = pair_df[mask]
    if len(rows) == 0:
        # Try all pairs
        mask2 = (
            (df["window_end_ms"] >= entry_ms - 5 * 60 * 1000) &
            (df["window_end_ms"] <= entry_ms + 5 * 60 * 1000)
        )
        rows = df[mask2]
        if len(rows) == 0:
            print(f"  {date_str} {hhmm} {direction:4s}: NOT IN DATA")
            continue
        rows = rows.assign(diff=(rows["window_end_ms"] - entry_ms).abs())
        closest = rows.loc[rows["diff"].idxmin()]
    else:
        closest = rows.iloc[0]

    X_row = closest[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(X_row)[0, 1]

    # Model prediction: had_move?
    model_says_move = prob >= 0.30  # use 0.30 threshold
    actual_had_move = closest["had_move"] == 1
    actual_dir = closest["move_direction"]

    # Trader was right if: model predicted move AND there was a move
    # OR model predicted no-move AND there was no move
    model_correct = (model_says_move == actual_had_move)

    results.append({
        "date": date_str, "time": hhmm, "direction": direction,
        "actual_pct": actual_pct, "prob": prob,
        "model_says_move": model_says_move, "actual_had_move": actual_had_move,
        "actual_dir": actual_dir, "model_correct": model_correct,
    })

    move_str = "MOVE" if actual_had_move else "calm"
    pred_str = "PRED_MOVE" if model_says_move else "PRED_CALM"
    correct_str = "✓" if model_correct else "✗"
    dir_str = f"dir={int(actual_dir):+d}" if actual_had_move else ""
    print(f"  {date_str} {hhmm} {direction:4s} prob={prob:.3f} {pred_str:10s} "
          f"actual={move_str:4s} {dir_str:6s} {correct_str}  "
          f"pnl={actual_pct*100:+6.2f}%")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
res_df = pd.DataFrame(results)
n_total = len(res_df)
n_correct = res_df["model_correct"].sum()
n_actual_move = res_df["actual_had_move"].sum()
n_pred_move = res_df["model_says_move"].sum()
print(f"  Entries traced: {n_total}")
print(f"  Model correct: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
print(f"  Actual moves: {n_actual_move}/{n_total}")
print(f"  Predicted moves: {n_pred_move}/{n_total}")

# What do winning entries have in common?
print(f"\n=== FEATURE VALUES AT ENTRY TIME ===")
top_feats = ["macro_trade_size_skew", "macro_large_trade_count", "micro_hour_sin",
             "macro_vol_ratio", "micro_spread_bps", "macro_liq_climax",
             "macro_sell_volume", "macro_buy_volume"]
for feat in top_feats:
    if feat in res_df.columns:
        vals = []
        for _, r in res_df.iterrows():
            # Find the row in pair_df
            mask = (pair_df["window_end_ms"] >= int(dt.datetime.strptime(r["date"], "%Y-%m-%d").replace(
                hour=int(r["time"].split(":")[0]), minute=int(r["time"].split(":")[1]),
                tzinfo=dt.timezone.utc).timestamp() * 1000) - 5*60*1000)
            rows = pair_df[mask]
            if len(rows) > 0:
                vals.append(rows.iloc[0][feat])
        if vals:
            print(f"  {feat:30s}: mean={np.mean(vals):+10.3f}  std={np.std(vals):8.3f}")
