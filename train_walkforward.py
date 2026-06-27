"""
train_walkforward.py — Walk-forward validation with had_move labels.

Phase 1 validation overhaul:
  - Non-overlapping windows (fixed in grid_sweeper.py)
  - Walk-forward validation: train on 5 days, test on next day, roll forward
  - min_child_samples=200 (prevents overfit on smaller non-overlapping dataset)
  - Cost: 0.15% per trade (entry + exit)
  - Fixed stop-loss: 0.5% (caps max loss per trade)
  - Reports validation probability std (signal check)

IS data: June 17-23 (7 days)
OOS data: June 24-26 (3 days)

Walk-forward folds within IS:
  Fold 1: Train Jun 17-21 (5d), Val Jun 22 (1d)
  Fold 2: Train Jun 18-22 (5d), Val Jun 23 (1d)
  Final:  Train all IS (7d), Eval on OOS (3d)
"""
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd

IS_FILE = "data/research_dataset_futures_hadmove.parquet"
OOS_FILE = "data/research_dataset_oos_hadmove.parquet"
OUTPUT_FILE = "data/expectancy_walkforward.csv"

COST_PER_TRADE = 0.0015   # 0.15% — entry + exit
STOP_LOSS_PCT = 0.005     # 0.5% fixed stop-loss
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 200,   # up from 50 — prevents overfit on non-overlapping data
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "had_move", "move_direction", "move_pct"}

# Walk-forward: train on N days, validate on next 1 day, roll forward
TRAIN_DAYS = 5  # days per training fold


def _day_of(ms: int) -> int:
    """Convert ms timestamp to day number (days since epoch)."""
    return int(ms // 86_400_000)


def _apply_stop_loss(pct: np.ndarray, stop: float = STOP_LOSS_PCT) -> np.ndarray:
    """Cap losses at -stop_loss_pct."""
    return np.where(pct < -stop, -stop, pct)


print("=" * 70)
print("WALK-FORWARD VALIDATION (had_move / V9, non-overlapping, 0.15% cost)")
print("=" * 70)

# Load IS data
print(f"\nLoading IS data: {IS_FILE} …")
df_is = pd.read_parquet(IS_FILE)
print(f"  {len(df_is):,} rows, {len(df_is.columns)} cols")
print(f"  IS base rate (had_move): {df_is['had_move'].mean():.4f}")

# Load OOS data
print(f"Loading OOS data: {OOS_FILE} …")
df_oos = pd.read_parquet(OOS_FILE)
print(f"  {len(df_oos):,} rows, {len(df_oos.columns)} cols")
print(f"  OOS base rate (had_move): {df_oos['had_move'].mean():.4f}")

# Feature columns
FEATURE_COLS = [c for c in df_is.columns if c not in META_COLS and c not in LABEL_COLS]
oos_features = [c for c in df_oos.columns if c not in META_COLS and c not in LABEL_COLS]
COMMON_FEATURES = [f for f in FEATURE_COLS if f in set(oos_features)]
print(f"  {len(COMMON_FEATURES)} common features")

# Identify unique days in IS
is_days = sorted(df_is["window_end_ms"].apply(_day_of).unique())
n_is_days = len(is_days)
print(f"  IS days: {n_is_days} ({[pd.Timestamp(d * 86_400_000, unit='ms').strftime('%Y-%m-%d') for d in is_days]})")
oos_days = sorted(df_oos["window_end_ms"].apply(_day_of).unique())
print(f"  OOS days: {len(oos_days)} ({[pd.Timestamp(d * 86_400_000, unit='ms').strftime('%Y-%m-%d') for d in oos_days]})")

# Walk-forward folds
folds = []
for i in range(n_is_days - TRAIN_DAYS):
    train_days = is_days[i : i + TRAIN_DAYS]
    val_day = is_days[i + TRAIN_DAYS]
    folds.append((train_days, val_day))
    dates_str = [pd.Timestamp(d * 86_400_000, unit='ms').strftime('%m-%d') for d in train_days]
    val_str = pd.Timestamp(val_day * 86_400_000, unit='ms').strftime('%m-%d')
    print(f"  Fold {i+1}: Train {dates_str} → Val {val_str}")

if not folds:
    print(f"ERROR: Not enough IS days ({n_is_days}) for walk-forward with TRAIN_DAYS={TRAIN_DAYS}")
    sys.exit(1)

# Train per (W, H) pair
all_oos_rows = []
importance_dfs = []
val_results = []

for (ws, hz), grp_is in df_is.groupby(["window_size", "horizon"], sort=True):
    ws, hz = int(ws), int(hz)
    g_is = grp_is.sort_values("window_end_ms").reset_index(drop=True)
    n_is = len(g_is)
    if n_is < 100:
        print(f"  W={ws} H={hz}: SKIP (too few IS rows: {n_is})")
        continue

    g_is_days = g_is["window_end_ms"].apply(_day_of).values

    # === WALK-FORWARD FOLDS ===
    fold_val_stds = []
    fold_val_accs = []

    for fold_idx, (train_days, val_day) in enumerate(folds):
        train_mask = np.isin(g_is_days, train_days)
        val_mask = (g_is_days == val_day)

        train = g_is[train_mask]
        val = g_is[val_mask]

        if len(train) < 100 or len(val) < 20:
            print(f"  W={ws} H={hz} Fold {fold_idx+1}: SKIP (train={len(train)}, val={len(val)})")
            continue

        X_train, y_train = train[COMMON_FEATURES], train["outcome_binary"]
        X_val, y_val = val[COMMON_FEATURES], val["outcome_binary"]

        model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        val_pred = (val_probs >= 0.5).astype(int)
        val_acc = (val_pred == y_val.values).mean()
        val_std = float(np.std(val_probs))
        fold_val_stds.append(val_std)
        fold_val_accs.append(val_acc)

    avg_val_std = np.mean(fold_val_stds) if fold_val_stds else 0.0
    avg_val_acc = np.mean(fold_val_accs) if fold_val_accs else 0.0
    val_results.append({"W": ws, "H": hz, "val_prob_std": avg_val_std, "val_acc": avg_val_acc})

    print(f"\n  W={ws} H={hz}: {n_is} IS rows, {len(folds)} folds")
    print(f"    Avg val prob std: {avg_val_std:.4f}  {'<-- NO SIGNAL' if avg_val_std < 0.05 else ''}")
    print(f"    Avg val accuracy: {avg_val_acc:.4f}")

    # === FINAL MODEL: Train on ALL IS, eval on OOS ===
    X_all_is = g_is[COMMON_FEATURES]
    y_all_is = g_is["outcome_binary"]

    # Use last fold's val for early stopping
    last_train_days, last_val_day = folds[-1]
    last_train_mask = np.isin(g_is_days, last_train_days)
    last_val_mask = (g_is_days == last_val_day)
    last_train = g_is[last_train_mask]
    last_val = g_is[last_val_mask]

    final_model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
    final_model.fit(
        last_train[COMMON_FEATURES], last_train["outcome_binary"],
        eval_set=[(last_val[COMMON_FEATURES], last_val["outcome_binary"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    # Feature importance
    imp = final_model.booster_.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({"feature": COMMON_FEATURES, "importance": imp})
    imp_df["window_size"] = ws
    imp_df["horizon"] = hz
    importance_dfs.append(imp_df)

    # === EVALUATE ON OOS ===
    grp_oos = df_oos[(df_oos["window_size"] == ws) & (df_oos["horizon"] == hz)]
    if len(grp_oos) == 0:
        print(f"    No OOS data for W={ws} H={hz}, skipping eval")
        continue

    X_oos = grp_oos[COMMON_FEATURES].values
    y_true = grp_oos["outcome_binary"].values
    y_pct_raw = grp_oos["outcome_pct"].values
    y_pct = _apply_stop_loss(y_pct_raw)  # cap losses at -0.5%
    n_days_oos = (grp_oos["window_end_ms"].max() - grp_oos["window_end_ms"].min()) / 86_400_000 + 1

    probs = final_model.predict_proba(X_oos)[:, 1]

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

        all_oos_rows.append({
            "window_size": ws, "horizon": hz, "threshold": thresh,
            "n_signals": n_sig,
            "signals_per_day": n_sig / max(n_days_oos, 1),
            "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": exp,
            "dataset": "OOS",
        })

# === RESULTS ===
print("\n" + "=" * 70)
print("=== VALIDATION SUMMARY (WALK-FORWARD) ===")
print("=" * 70)
val_df = pd.DataFrame(val_results)
print(val_df.to_string(index=False))
avg_std = val_df["val_prob_std"].mean()
print(f"\n  OVERALL avg val prob std: {avg_std:.4f}  {'<-- NO SIGNAL' if avg_std < 0.05 else ''}")

# Save OOS results
if len(all_oos_rows) > 0:
    table = pd.DataFrame(all_oos_rows).sort_values("expectancy", ascending=False)
    table.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("=== OOS RESULTS (TOP 15 BY EXPECTANCY) ===")
    print("=" * 70)
    print(table.head(15).to_string(index=False))
else:
    print("\n*** NO SIGNALS at any threshold ***")

# Feature importance
if importance_dfs:
    all_imp = pd.concat(importance_dfs, ignore_index=True)
    avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
    print(f"\n=== TOP 5 FEATURES BY AVG IMPORTANCE ===")
    for rank, (feat, imp_val) in enumerate(avg_imp.head(5).items(), 1):
        print(f"  {rank}. {feat:40s}  {imp_val:.1f}")

# === SUMMARY AT θ=0.50 ===
print("\n" + "=" * 70)
print(f"=== OOS SUMMARY AT θ=0.50 (cost={COST_PER_TRADE*100:.2f}%, stop={STOP_LOSS_PCT*100:.1f}%) ===")
print("=" * 70)
if len(all_oos_rows) > 0:
    theta_results = table[table["threshold"] == 0.50]
    if len(theta_results) > 0:
        for _, row in theta_results.iterrows():
            print(f"  W={int(row['window_size'])} H={int(row['horizon'])}: "
                  f"signals={int(row['n_signals'])}, sig/day={row['signals_per_day']:.1f}, "
                  f"WR={row['win_rate']:.3f}, exp={row['expectancy']*100:+.3f}%")
        total_sig = int(theta_results["n_signals"].sum())
        print(f"\n  TOTAL at θ=0.50: {total_sig} signals")
    else:
        print("  No signals at θ=0.50")
else:
    print("  No signals at any threshold")