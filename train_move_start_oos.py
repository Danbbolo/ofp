"""
train_move_start_oos.py — OOS validation: train on IS, evaluate on OOS.

Trains LightGBM on the in-sample data (06-21 to 06-23),
then evaluates on the out-of-sample data (06-24 to 06-26).

This is the CRITICAL test — did the model find a real pattern
or just memorize the training data?
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

IS_FILE = "data/research_dataset_futures_move_start.parquet"
OOS_FILE = "data/research_dataset_oos_move_start.parquet"
OUTPUT_FILE = "data/expectancy_table_oos.csv"
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
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
LABEL_COLS = {"outcome_binary", "outcome_pct", "had_move", "move_direction", "move_pct"}

print("=" * 60)
print("OOS VALIDATION: Train on IS (06-21→06-23), Test on OOS (06-24→06-26)")
print("=" * 60)

# Load IS data (for training)
print(f"\nLoading IS data: {IS_FILE} …")
df_is = pd.read_parquet(IS_FILE)
print(f"  {len(df_is):,} rows, {len(df_is.columns)} cols")

# Load OOS data (for testing)
print(f"Loading OOS data: {OOS_FILE} …")
df_oos = pd.read_parquet(OOS_FILE)
print(f"  {len(df_oos):,} rows, {len(df_oos.columns)} cols")

# Get feature columns from IS data (they should match)
FEATURE_COLS = [c for c in df_is.columns if c not in META_COLS and c not in LABEL_COLS]
print(f"  {len(FEATURE_COLS)} features")

# Check feature overlap
oos_features = [c for c in df_oos.columns if c not in META_COLS and c not in LABEL_COLS]
missing_in_oos = set(FEATURE_COLS) - set(oos_features)
extra_in_oos = set(oos_features) - set(FEATURE_COLS)
if missing_in_oos:
    print(f"  WARNING: {len(missing_in_oos)} features in IS but not OOS: {missing_in_oos}")
if extra_in_oos:
    print(f"  NOTE: {len(extra_in_oos)} features in OOS but not IS: {extra_in_oos}")
# Use only common features
COMMON_FEATURES = [f for f in FEATURE_COLS if f in set(oos_features)]
print(f"  {len(COMMON_FEATURES)} common features for training/eval")

# Train on IS data (chronological 70/15/15 split within IS)
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
    y_pct = grp_oos["move_pct"].values
    y_dir = grp_oos["move_direction"].values
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

        # Direction accuracy on signals that had a move
        had_move_sig = sig & (y_true == 1)
        if had_move_sig.any():
            dir_correct = (np.sign(probs[had_move_sig] - 0.5) == y_dir[had_move_sig]).mean()
        else:
            dir_correct = 0

        all_rows.append({
            "window_size": ws, "horizon": hz, "threshold": thresh,
            "n_signals": n_sig,
            "signals_per_day": n_sig / max(n_days_oos, 1),
            "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": exp,
            "direction_accuracy": dir_correct,
            "dataset": "OOS",
        })

    # Also evaluate on IS test set for comparison
    is_test = g_is.iloc[val_end:]
    if len(is_test) > 0:
        X_is_test = is_test[COMMON_FEATURES].values
        y_is_true = is_test["outcome_binary"].values
        y_is_pct = is_test["move_pct"].values
        is_probs = model.predict(X_is_test)
        n_days_is = (is_test["window_end_ms"].max() - is_test["window_end_ms"].min()) / 86_400_000 + 1

        for thresh in THRESHOLDS:
            sig = is_probs >= thresh
            n_sig = int(sig.sum())
            if n_sig == 0:
                continue
            wr = y_is_true[sig].mean()
            wins = y_is_true[sig] == 1
            losses = y_is_true[sig] == 0
            avg_win = abs(y_is_pct[sig][wins].mean()) if wins.any() else 0
            avg_loss = abs(y_is_pct[sig][losses].mean()) if losses.any() else 0
            exp = wr * avg_win - (1 - wr) * avg_loss - COST_PER_TRADE
            all_rows.append({
                "window_size": ws, "horizon": hz, "threshold": thresh,
                "n_signals": n_sig,
                "signals_per_day": n_sig / max(n_days_is, 1),
                "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
                "expectancy": exp,
                "direction_accuracy": 0,
                "dataset": "IS_TEST",
            })

# Save results
table = pd.DataFrame(all_rows).sort_values("expectancy", ascending=False)
table.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")

# Print OOS results
oos_table = table[table["dataset"] == "OOS"].sort_values("expectancy", ascending=False)
print("\n" + "=" * 60)
print("=== OOS RESULTS (TOP 10 BY EXPECTANCY) ===")
print("=" * 60)
if len(oos_table) > 0:
    print(oos_table.head(10).to_string(index=False))
else:
    print("  No OOS results!")

# Print IS test results for comparison
is_table = table[table["dataset"] == "IS_TEST"].sort_values("expectancy", ascending=False)
print("\n=== IS TEST RESULTS (TOP 10 BY EXPECTANCY) ===")
if len(is_table) > 0:
    print(is_table.head(10).to_string(index=False))
else:
    print("  No IS test results!")

# Feature importance
if importance_dfs:
    all_imp = pd.concat(importance_dfs, ignore_index=True)
    avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(f"\n=== TOP 20 FEATURES BY AVG IMPORTANCE ===")
    for rank, (feat, imp_val) in enumerate(avg_imp.head(20).items(), 1):
        print(f"  {rank:2d}. {feat:40s}  {imp_val:.1f}")

# === HIGH-PROBABILITY SIGNAL TRACE ===
print("\n" + "=" * 60)
print("=== HIGH-PROBABILITY SIGNAL TRACE (prob >= 0.65) ===")
print("=" * 60)

for (ws, hz), model in models.items():
    grp_oos = df_oos[(df_oos["window_size"] == ws) & (df_oos["horizon"] == hz)]
    if len(grp_oos) == 0:
        continue

    X_oos = grp_oos[COMMON_FEATURES].values
    probs = model.predict(X_oos)
    y_true = grp_oos["outcome_binary"].values
    y_pct = grp_oos["move_pct"].values
    y_dir = grp_oos["move_direction"].values
    timestamps = grp_oos["window_end_ms"].values

    high_prob = probs >= 0.65
    n_hp = int(high_prob.sum())
    if n_hp == 0:
        print(f"\n  W={ws} H={hz}: No signals at prob >= 0.65")
        # Try 0.50
        high_prob = probs >= 0.50
        n_hp = int(high_prob.sum())
        if n_hp == 0:
            print(f"  W={ws} H={hz}: No signals at prob >= 0.50 either")
            continue
        print(f"  W={ws} H={hz}: Showing prob >= 0.50 instead ({n_hp} signals)")

    print(f"\n  W={ws} H={hz}: {n_hp} high-prob signals")
    print(f"  Win rate: {y_true[high_prob].mean():.3f}")
    if y_true[high_prob].any():
        print(f"  Avg move_pct on winners: {y_pct[high_prob & (y_true==1)].mean()*100:+.3f}%")
    if (high_prob & (y_true == 0)).any():
        print(f"  Avg move_pct on losers: {y_pct[high_prob & (y_true==0)].mean()*100:+.3f}%")

    # Print individual signals
    from datetime import datetime, timezone
    print(f"\n  {'Time (UTC)':<22s} {'Prob':>6s} {'HadMove':>8s} {'Dir':>5s} {'Move%':>8s}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*5} {'-'*8}")
    hp_indices = np.where(high_prob)[0]
    # Sort by probability descending
    hp_sorted = hp_indices[np.argsort(-probs[high_prob])]
    for idx in hp_sorted[:30]:  # Show top 30
        ts = datetime.fromtimestamp(timestamps[idx] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        p = probs[idx]
        hm = "YES" if y_true[idx] == 1 else "NO"
        d = "UP" if y_dir[idx] == 1 else ("DN" if y_dir[idx] == -1 else "--")
        mp = y_pct[idx] * 100
        print(f"  {ts:<22s} {p:.3f}  {hm:>6s}  {d:>4s}  {mp:+.3f}%")

print("\n" + "=" * 60)
print("DONE")
