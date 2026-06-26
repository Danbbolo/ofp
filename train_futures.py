"""
train_futures.py — Train on futures target-based data.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = "data/research_dataset_futures_relabel.parquet"
OUTPUT_FILE = "data/expectancy_table_futures.csv"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
COST_PER_TRADE = 0.001

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,  # smaller for small dataset
    "min_child_samples": 30,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}

print(f"Loading {INPUT_FILE} …")
df = pd.read_parquet(INPUT_FILE)
print(f"  {len(df):,} rows, {len(df.columns)} cols")

FEATURE_COLS = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
print(f"  {len(FEATURE_COLS)} features")

# Per-pair chronological split
pair_splits = {}
for (ws, hz), grp in df.groupby(["window_size", "horizon"], sort=True):
    g = grp.sort_values("window_end_ms").reset_index(drop=True)
    n = len(g)
    if n < 3:
        continue
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    pair_splits[(int(ws), int(hz))] = (g.iloc[:train_end], g.iloc[train_end:val_end], g.iloc[val_end:])

print(f"  {len(pair_splits)} (ws, hz) pairs")
print()

# Train one model per pair
pairs = sorted(pair_splits.keys())
all_rows = []
importance_dfs = []

for ws, hz in pairs:
    train, val, test = pair_splits[(ws, hz)]
    if len(train) < 50 or len(val) < 20:
        print(f"  W={ws} H={hz}: SKIP (too few)")
        continue

    X_train, y_train = train[FEATURE_COLS], train["outcome_binary"]
    X_val, y_val = val[FEATURE_COLS], val["outcome_binary"]

    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )

    # Feature importance
    imp = model.booster_.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
    imp_df["window_size"] = ws
    imp_df["horizon"] = hz
    importance_dfs.append(imp_df)

    # Evaluate
    X_test = test[FEATURE_COLS]
    y_true = test["outcome_binary"].values
    y_pct = test["outcome_pct"].values
    n_days = (test["window_end_ms"].max() - test["window_end_ms"].min()) / 86_400_000 + 1

    probs = model.predict(X_test)

    for thresh in THRESHOLDS:
        sig = probs >= thresh
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        wr = y_true[sig].mean()
        wins = y_true[sig] == 1
        losses = y_true[sig] == 0
        avg_win = y_pct[sig][wins].mean() if wins.any() else 0
        avg_loss = abs(y_pct[sig][losses].mean()) if losses.any() else 0
        exp = wr * avg_win - (1 - wr) * avg_loss - COST_PER_TRADE
        all_rows.append({
            "window_size": ws, "horizon": hz, "threshold": thresh,
            "n_signals": n_sig,
            "signals_per_day": n_sig / max(n_days, 1),
            "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": exp,
        })

    best = max(
        [r for r in all_rows if r["window_size"] == ws and r["horizon"] == hz],
        key=lambda r: r["expectancy"], default=None
    )
    if best:
        print(f"  W={ws} H={hz}: best θ={best['threshold']:.2f} exp={best['expectancy']*100:+.3f}% "
              f"wr={best['win_rate']:.3f} n={best['n_signals']}")

table = pd.DataFrame(all_rows).sort_values("expectancy", ascending=False)
table.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")
print("\n=== TOP 10 BY EXPECTANCY ===")
print(table.head(10).to_string(index=False))

# Feature importance
if importance_dfs:
    all_imp = pd.concat(importance_dfs, ignore_index=True)
    avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(f"\n=== TOP 10 FEATURES BY AVG IMPORTANCE ===")
    for rank, (feat, imp_val) in enumerate(avg_imp.head(10).items(), 1):
        print(f"  {rank:2d}. {feat:40s}  {imp_val:.1f}")
