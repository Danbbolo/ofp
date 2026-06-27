"""
canary_test.py — Verify train_target_oos.py can learn a perfect signal.

Injects a synthetic feature `canary_signal` = target label directly.
If LightGBM can't learn this, the training pipeline is broken.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

IS_FILE = "data/research_dataset_futures_target.parquet"
OOS_FILE = "data/research_dataset_oos_target.parquet"

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

print("=" * 70)
print("CANARY TEST: Can LightGBM learn a perfect signal?")
print("=" * 70)

# Load data
df_is = pd.read_parquet(IS_FILE)
df_oos = pd.read_parquet(OOS_FILE)

# Get original features
FEATURE_COLS = [c for c in df_is.columns if c not in META_COLS and c not in LABEL_COLS]
oos_features = [c for c in df_oos.columns if c not in META_COLS and c not in LABEL_COLS]
COMMON_FEATURES = [f for f in FEATURE_COLS if f in set(oos_features)]
print(f"\nOriginal features: {len(COMMON_FEATURES)}")

# === INJECT CANARY ===
# canary_signal = target label directly (perfect predictor)
df_is["canary_signal"] = df_is["outcome_binary"].values
df_oos["canary_signal"] = df_oos["outcome_binary"].values

# Add canary to feature set
CANARY_FEATURES = COMMON_FEATURES + ["canary_signal"]
print(f"Features with canary: {len(CANARY_FEATURES)}")
print(f"Canary at index: {len(CANARY_FEATURES) - 1}")

# Verify canary correlation
canary_corr = df_is["canary_signal"].corr(df_is["outcome_binary"])
print(f"Canary correlation with target: {canary_corr:.6f} (should be 1.0)")

# === TRAIN (same as train_target_oos.py) ===
# Use W=60, H=1800 as representative
ws, hz = 60, 1800
grp_is = df_is[(df_is["window_size"] == ws) & (df_is["horizon"] == hz)].sort_values("window_end_ms").reset_index(drop=True)
grp_oos = df_oos[(df_oos["window_size"] == ws) & (df_oos["horizon"] == hz)].sort_values("window_end_ms").reset_index(drop=True)

n_is = len(grp_is)
train_end = int(n_is * 0.70)
val_end = int(n_is * 0.85)

train = grp_is.iloc[:train_end]
val = grp_is.iloc[train_end:val_end]

X_train = train[CANARY_FEATURES]
y_train = train["outcome_binary"]
X_val = val[CANARY_FEATURES]
y_val = val["outcome_binary"]
X_oos = grp_oos[CANARY_FEATURES]
y_oos = grp_oos["outcome_binary"]

print(f"\nTraining on W={ws} H={hz}: {len(train)} train, {len(val)} val, {len(grp_oos)} OOS")
print(f"Train label ratio: {y_train.mean():.3f}")
print(f"Val label ratio: {y_val.mean():.3f}")
print(f"OOS label ratio: {y_oos.mean():.3f}")

model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
)

# === CHECK RESULTS ===
print("\n" + "=" * 70)
print("CANARY TEST RESULTS")
print("=" * 70)

# a. Feature importance
imp = model.booster_.feature_importance(importance_type="split")
imp_df = pd.DataFrame({"feature": CANARY_FEATURES, "importance": imp})
imp_df = imp_df.sort_values("importance", ascending=False)

print("\n(a) TOP 10 FEATURE IMPORTANCE:")
for rank, (_, row) in enumerate(imp_df.head(10).iterrows(), 1):
    marker = " <--- CANARY" if row["feature"] == "canary_signal" else ""
    print(f"  {rank:2d}. {row['feature']:40s}  {row['importance']:.1f}{marker}")

canary_rank = imp_df[imp_df["feature"] == "canary_signal"].index[0]
canary_rank_pos = list(imp_df["feature"]).index("canary_signal") + 1
print(f"\n  Canary rank: #{canary_rank_pos}")

# b. Validation probability range
val_probs = model.predict_proba(X_val)[:, 1]
print(f"\n(b) VALIDATION PROBABILITY RANGE:")
print(f"  min={val_probs.min():.6f}, max={val_probs.max():.6f}, mean={val_probs.mean():.6f}")
print(f"  std={val_probs.std():.6f}")
print(f"  fraction > 0.9: {(val_probs > 0.9).mean():.4f}")
print(f"  fraction < 0.1: {(val_probs < 0.1).mean():.4f}")

# c. Validation accuracy
val_preds = model.predict(X_val)
val_acc = (val_preds == y_val.values).mean()
print(f"\n(c) VALIDATION ACCURACY: {val_acc*100:.2f}%")

# d. OOS probability range
oos_probs = model.predict_proba(X_oos)[:, 1]
print(f"\n(d) OOS PROBABILITY RANGE:")
print(f"  min={oos_probs.min():.6f}, max={oos_probs.max():.6f}, mean={oos_probs.mean():.6f}")
print(f"  std={oos_probs.std():.6f}")
print(f"  fraction > 0.9: {(oos_probs > 0.9).mean():.4f}")
print(f"  fraction < 0.1: {(oos_probs < 0.1).mean():.4f}")

# OOS accuracy
oos_preds = model.predict(X_oos)
oos_acc = (oos_preds == y_oos.values).mean()
print(f"  OOS accuracy: {oos_acc*100:.2f}%")

# === VERDICT ===
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

canary_is_top1 = canary_rank_pos == 1
val_acc_high = val_acc >= 0.95
val_probs_spread = val_probs.std() > 0.1

if canary_is_top1 and val_acc_high and val_probs_spread:
    print("✅ PASS: Training script works correctly.")
    print("   Canary is #1 importance, val accuracy ~100%, probs are spread.")
    print("   The 'no edge' result is REAL — features genuinely don't predict target.")
elif not canary_is_top1:
    print("❌ FAIL: Canary is NOT #1 importance — training script is broken.")
elif not val_acc_high:
    print(f"❌ FAIL: Val accuracy only {val_acc*100:.1f}% — should be ~100%.")
elif not val_probs_spread:
    print("❌ FAIL: Val probs still clustered at 0.5 — model not learning canary.")
else:
    print("⚠️  PARTIAL: Some checks passed, some failed.")