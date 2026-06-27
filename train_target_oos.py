"""
train_target_oos.py — OOS validation with TARGET-BASED labels.

Trains LightGBM on IS (06-17→06-23), evaluates on OOS (06-24→06-26).
Uses target-based labels: +1% hit before -1% stop in 24h.

Params: min_child_samples=50, num_leaves=31, early_stopping=20
Chronological 70/15/15 split within IS for train/val
Threshold: θ=0.6
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

IS_FILE = "data/research_dataset_futures_target.parquet"
OOS_FILE = "data/research_dataset_oos_target.parquet"
OUTPUT_FILE = "data/expectancy_table_target_oos.csv"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
COST_PER_TRADE = 0.001  # 0.1%

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 50,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit", "stop_hit",
              "time_based_exit", "hold_sec", "had_move", "move_direction", "move_pct"}

print("=" * 60)
print("OOS VALIDATION (TARGET-BASED): Train IS (06-17→06-23), Test OOS (06-24→06-26)")
print("=" * 60)

# Load IS data
print(f"\nLoading IS data: {IS_FILE} …")
df_is = pd.read_parquet(IS_FILE)
print(f"  {len(df_is):,} rows, {len(df_is.columns)} cols")
print(f"  IS base rate (outcome_binary): {df_is['outcome_binary'].mean():.3f}")
print(f"  IS target_hit rate: {df_is['target_hit'].mean():.3f}")
print(f"  IS stop_hit rate: {df_is['stop_hit'].mean():.3f}")

# Load OOS data
print(f"Loading OOS data: {OOS_FILE} …")
df_oos = pd.read_parquet(OOS_FILE)
print(f"  {len(df_oos):,} rows, {len(df_oos.columns)} cols")
print(f"  OOS base rate (outcome_binary): {df_oos['outcome_binary'].mean():.3f}")
print(f"  OOS target_hit rate: {df_oos['target_hit'].mean():.3f}")
print(f"  OOS stop_hit rate: {df_oos['stop_hit'].mean():.3f}")

# Get feature columns
FEATURE_COLS = [c for c in df_is.columns if c not in META_COLS and c not in LABEL_COLS]
oos_features = [c for c in df_oos.columns if c not in META_COLS and c not in LABEL_COLS]
COMMON_FEATURES = [f for f in FEATURE_COLS if f in set(oos_features)]
print(f"  {len(COMMON_FEATURES)} common features")

# Train per (W, H) pair
all_rows = []
importance_dfs = []
models = {}

for (ws, hz), grp_is in df_is.groupby(["window_size", "horizon"], sort=True):
    ws, hz = int(ws), int(hz)
    g_is = grp_is.sort_values("window_end_ms").reset_index(drop=True)
    n_is = len(g_is)
    if n_is < 100:
        print(f"  W={ws} H={hz}: SKIP (too few IS rows: {n_is})")
        continue

    # IS split: 70% train, 15% val (for early stopping)
    train_end = int(n_is * 0.70)
    val_end = int(n_is * 0.85)
    train = g_is.iloc[:train_end]
    val = g_is.iloc[train_end:val_end]

    if len(train) < 50 or len(val) < 20:
        print(f"  W={ws} H={hz}: SKIP (train/val too small)")
        continue

    X_train, y_train = train[COMMON_FEATURES], train["outcome_binary"]
    X_val, y_val = val[COMMON_FEATURES], val["outcome_binary"]

    print(f"\n  W={ws} H={hz}: training on {len(train)} IS rows, early-stopping on {len(val)} …")

    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    models[(ws, hz)] = model

    # Feature importance
    imp = model.booster_.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({"feature": COMMON_FEATURES, "importance": imp})
    imp_df["window_size"] = ws
    imp_df["horizon"] = hz
    importance_dfs.append(imp_df)

    # === EVALUATE ON OOS DATA ===
    grp_oos = df_oos[(df_oos["window_size"] == ws) & (df_oos["horizon"] == hz)]
    if len(grp_oos) == 0:
        print(f"    No OOS data for this pair, skipping eval")
        continue

    X_oos = grp_oos[COMMON_FEATURES].values
    y_true = grp_oos["outcome_binary"].values
    y_pct = grp_oos["outcome_pct"].values  # target-based: actual return
    n_days_oos = (grp_oos["window_end_ms"].max() - grp_oos["window_end_ms"].min()) / 86_400_000 + 1

    probs = model.predict(X_oos)

    print(f"    OOS: {len(grp_oos)} rows, {n_days_oos:.1f} days")
    print(f"    OOS base rate: {y_true.mean():.3f}")
    print(f"    OOS prob range: [{probs.min():.3f}, {probs.max():.3f}], mean={probs.mean():.3f}")

    for thresh in THRESHOLDS:
        sig = probs >= thresh
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        wr = y_true[sig].mean()
        wins = y_true[sig] == 1
        losses = y_true[sig] == 0
        avg_win = abs(y_pct[sig][wins].mean()) if wins.any() else 0
        avg_loss = abs(y_pct[sig][losses].mean()) if losses.any() else 0
        exp = wr * avg_win - (1 - wr) * avg_loss - COST_PER_TRADE

        all_rows.append({
            "window_size": ws, "horizon": hz, "threshold": thresh,
            "n_signals": n_sig,
            "signals_per_day": n_sig / max(n_days_oos, 1),
            "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": exp,
            "dataset": "OOS",
        })

# Save results
table = pd.DataFrame(all_rows).sort_values("expectancy", ascending=False)
table.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")

# Print OOS results
oos_table = table[table["dataset"] == "OOS"].sort_values("expectancy", ascending=False)
print("\n" + "=" * 60)
print("=== OOS RESULTS (TOP 15 BY EXPECTANCY) ===")
print("=" * 60)
if len(oos_table) > 0:
    print(oos_table.head(15).to_string(index=False))
else:
    print("  No OOS results!")

# Feature importance
if importance_dfs:
    all_imp = pd.concat(importance_dfs, ignore_index=True)
    avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(f"\n=== TOP 5 FEATURES BY AVG IMPORTANCE ===")
    for rank, (feat, imp_val) in enumerate(avg_imp.head(5).items(), 1):
        print(f"  {rank}. {feat:40s}  {imp_val:.1f}")

# === SUMMARY AT θ=0.6 ===
print("\n" + "=" * 60)
print("=== OOS SUMMARY AT θ=0.60 ===")
print("=" * 60)
theta_results = oos_table[oos_table["threshold"] == 0.60]
if len(theta_results) > 0:
    for _, row in theta_results.iterrows():
        print(f"  W={int(row['window_size'])} H={int(row['horizon'])}: "
              f"signals={int(row['n_signals'])}, sig/day={row['signals_per_day']:.1f}, "
              f"WR={row['win_rate']:.3f}, exp={row['expectancy']*100:+.3f}%")
    # Aggregate
    total_sig = int(theta_results["n_signals"].sum())
    total_days = theta_results["signals_per_day"].sum() / len(theta_results) if len(theta_results) > 0 else 0
    print(f"\n  TOTAL at θ=0.60: {total_sig} signals")
else:
    print("  No signals at θ=0.60")