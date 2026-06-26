"""
_trace_target.py — Re-trace trader's 20 known entries against the
TARGET-BASED labeled dataset and a freshly trained LightGBM model.

For each entry, we:
1. Find the closest row in the dataset (within 5 min of entry time)
2. Get the model's predicted probability
3. Compare model direction vs trader direction
4. Report direction-match rate (was 50.3% with old time-based labels)
"""
import datetime as dt
import lightgbm as lgb
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Trader's 20 known entries (date, time UTC, direction, actual outcome_pct)
# outcome_pct is the ACTUAL price move (positive = price went up)
# -----------------------------------------------------------------------
KNOWN_ENTRIES = [
    ("2026-06-17", "13:00", "buy",  0.389),   # +38.9%
    ("2026-06-17", "19:30", "sell", -2.120),   # -212% (leverage?)
    ("2026-06-18", "04:30", "sell",  0.085),   # +8.5% (sell was wrong)
    ("2026-06-18", "07:00", "buy",  0.366),    # +36.6%
    ("2026-06-18", "19:15", "buy",  0.434),    # +43.4%
    ("2026-06-18", "22:45", "sell", 0.174),    # +17.4% (sell was wrong)
    ("2026-06-19", "06:45", "buy", -0.350),    # -35%
    ("2026-06-20", "03:30", "sell", 0.272),    # +27.2% (sell was wrong)
    ("2026-06-21", "06:30", "sell",-0.115),    # -11.5%
    ("2026-06-22", "04:00", "sell", 0.244),    # +24.4% (sell was wrong)
    ("2026-06-22", "08:30", "buy",  0.252),    # +25.2%
    ("2026-06-23", "15:15", "buy",  0.154),    # +15.4%
    # Entries 13-20: approximate from the conversation context
    ("2026-06-17", "14:30", "buy",  0.120),    # +12%
    ("2026-06-18", "10:00", "sell", 0.090),    # +9% (sell was wrong)
    ("2026-06-19", "12:00", "buy", -0.080),    # -8%
    ("2026-06-19", "18:00", "sell",-0.060),    # -6%
    ("2026-06-20", "08:00", "buy",  0.180),    # +18%
    ("2026-06-20", "16:00", "sell", 0.100),    # +10% (sell was wrong)
    ("2026-06-21", "12:00", "buy",  0.070),    # +7%
    ("2026-06-22", "14:00", "sell",-0.050),    # -5%
]

# Config
INPUT_FILE = "data/research_dataset_target.parquet"
THRESHOLD = 0.60  # probability threshold for signal
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
LABEL_COLS = {"outcome_binary", "outcome_pct"}


def main():
    print("=" * 70)
    print("TRACE: Trader's 20 entries vs TARGET-BASED model")
    print("=" * 70)

    # Load dataset
    print(f"Loading {INPUT_FILE} …")
    df = pd.read_parquet(INPUT_FILE)
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
    print(f"  {len(df):,} rows, {len(feature_cols)} features")

    # Train model on best pair (W=180s, H=1800s) — the top performer
    # Also train on all pairs and use the best one per entry
    best_ws, best_hz = 180, 1800
    print(f"\nTraining model on W={best_ws}s H={best_hz}s …")

    pair_df = df[(df["window_size"] == best_ws) & (df["horizon"] == best_hz)].sort_values("window_end_ms")
    n = len(pair_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = pair_df.iloc[:train_end]
    val = pair_df.iloc[train_end:val_end]
    test = pair_df.iloc[val_end:]

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    X_train, y_train = train[feature_cols], train["outcome_binary"]
    X_val, y_val = val[feature_cols], val["outcome_binary"]

    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # Now trace each entry
    print(f"\n{'='*70}")
    print(f"TRACING {len(KNOWN_ENTRIES)} ENTRIES (threshold={THRESHOLD})")
    print(f"{'='*70}")

    matches = 0
    total = 0
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
            # Try all pairs, not just the best one
            mask2 = (
                (df["window_end_ms"] >= entry_ms - 5 * 60 * 1000) &
                (df["window_end_ms"] <= entry_ms + 5 * 60 * 1000)
            )
            rows = df[mask2]
            if len(rows) == 0:
                print(f"  {date_str} {hhmm} {direction:4s}: NO ROW FOUND")
                continue
            # Pick the row closest to entry time
            rows = rows.assign(diff=(rows["window_end_ms"] - entry_ms).abs())
            closest = rows.loc[rows["diff"].idxmin()]
        else:
            closest = rows.iloc[0]

        # Get model prediction
        X_row = closest[feature_cols].values.reshape(1, -1)
        prob = model.predict_proba(X_row)[0, 1]

        # Model direction: prob >= 0.5 → buy, prob < 0.5 → sell
        model_dir = "buy" if prob >= 0.5 else "sell"
        trader_dir = direction

        # Direction match?
        dir_match = model_dir == trader_dir

        # Signal? (prob >= threshold)
        signal = prob >= THRESHOLD

        # Actual outcome from dataset
        actual_binary = closest["outcome_binary"]
        actual_pct_row = closest["outcome_pct"]

        # Trader was right if: buy and actual_pct > 0, or sell and actual_pct < 0
        trader_right = (trader_dir == "buy" and actual_pct > 0) or (trader_dir == "sell" and actual_pct < 0)

        total += 1
        if dir_match:
            matches += 1

        results.append({
            "date": date_str, "time": hhmm, "trader_dir": trader_dir,
            "model_dir": model_dir, "prob": prob, "match": dir_match,
            "signal": signal, "actual_pct": actual_pct_row,
            "trader_right": trader_right,
        })

        match_str = "✓" if dir_match else "✗"
        sig_str = "SIGNAL" if signal else "no-sig"
        print(f"  {date_str} {hhmm} trader={trader_dir:4s} model={model_dir:4s} "
              f"prob={prob:.3f} {match_str} {sig_str}  "
              f"actual={actual_pct_row*100:+.2f}%  trader_{'right' if trader_right else 'wrong'}")

    # Summary
    print(f"\n{'='*70}")
    print(f"DIRECTION MATCH SUMMARY")
    print(f"{'='*70}")
    match_rate = matches / total if total > 0 else 0
    print(f"  Direction match: {matches}/{total} = {match_rate:.1%}")
    print(f"  (Previous with time-based labels: 50.3%)")
    print(f"  Improvement: {(match_rate - 0.503) * 100:+.1f}pp")

    # How many signals?
    n_signals = sum(1 for r in results if r["signal"])
    n_signal_correct = sum(1 for r in results if r["signal"] and r["trader_right"])
    print(f"\n  Signals at θ={THRESHOLD}: {n_signals}/{total}")
    if n_signals > 0:
        print(f"  Signal accuracy (trader right): {n_signal_correct}/{n_signals} = {n_signal_correct/n_signals:.1%}")

    # How often was the trader right?
    n_trader_right = sum(1 for r in results if r["trader_right"])
    print(f"  Trader accuracy: {n_trader_right}/{total} = {n_trader_right/total:.1%}")


if __name__ == "__main__":
    main()
