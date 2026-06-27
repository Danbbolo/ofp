"""
diagnose_model.py — Diagnose why LightGBM predicts 1.0 for every OOS row.

Checks:
1. Feature-target leakage (correlation > 0.9)
2. Train/val/test split date ranges and overlap
3. Label distribution per split
4. Model probability distribution on IS validation
5. IS vs OOS feature schema diff
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime

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
print("DIAGNOSTIC: Why does LightGBM predict 1.0 for every OOS row?")
print("=" * 70)

# Load data
df_is = pd.read_parquet(IS_FILE)
df_oos = pd.read_parquet(OOS_FILE)

FEATURE_COLS = [c for c in df_is.columns if c not in META_COLS and c not in LABEL_COLS]
oos_features = [c for c in df_oos.columns if c not in META_COLS and c not in LABEL_COLS]
COMMON_FEATURES = [f for f in FEATURE_COLS if f in set(oos_features)]

# =========================================================================
# 1. FEATURE-TARGET LEAKAGE CHECK
# =========================================================================
print("\n" + "=" * 70)
print("1. FEATURE-TARGET LEAKAGE CHECK (IS dataset)")
print("=" * 70)

target = df_is["outcome_binary"]
correlations = {}
for col in COMMON_FEATURES:
    if df_is[col].dtype in [np.float64, np.int64, np.int8, float, int]:
        corr = df_is[col].corr(target)
        if np.isnan(corr):
            corr = 0.0
        correlations[col] = abs(corr)

corr_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 features by |correlation| with target:")
for rank, (feat, corr_val) in enumerate(corr_sorted[:10], 1):
    flag = " *** LEAKAGE ***" if corr_val > 0.9 else (" ** high **" if corr_val > 0.5 else "")
    print(f"  {rank:2d}. {feat:40s}  |r|={corr_val:.6f}{flag}")

leaky = [(f, c) for f, c in corr_sorted if c > 0.9]
if leaky:
    print(f"\n  *** {len(leaky)} features with |correlation| > 0.9 (LEAKAGE): ***")
    for f, c in leaky:
        print(f"      {f}: {c:.6f}")
else:
    print(f"\n  No features with |correlation| > 0.9")

# =========================================================================
# 2. TRAIN/VAL/TEST SPLIT CHECK
# =========================================================================
print("\n" + "=" * 70)
print("2. TRAIN/VAL/TEST SPLIT CHECK")
print("=" * 70)

# Use W=60, H=1800 as representative (same pattern as training)
ws, hz = 60, 1800
grp_is = df_is[(df_is["window_size"] == ws) & (df_is["horizon"] == hz)].sort_values("window_end_ms").reset_index(drop=True)
grp_oos = df_oos[(df_oos["window_size"] == ws) & (df_oos["horizon"] == hz)].sort_values("window_end_ms").reset_index(drop=True)

n_is = len(grp_is)
train_end = int(n_is * 0.70)
val_end = int(n_is * 0.85)

train_ms = grp_is["window_end_ms"].iloc[:train_end]
val_ms = grp_is["window_end_ms"].iloc[train_end:val_end]
is_test_ms = grp_is["window_end_ms"].iloc[val_end:]
oos_ms = grp_oos["window_end_ms"]

def ms_to_str(ms_series):
    return f"{pd.to_datetime(ms_series.min(), unit='ms')} to {pd.to_datetime(ms_series.max(), unit='ms')}"

print(f"\n  W={ws} H={hz} (representative pair):")
print(f"  Train:      {ms_to_str(train_ms)}  ({len(train_ms)} rows)")
print(f"  Val:        {ms_to_str(val_ms)}  ({len(val_ms)} rows)")
print(f"  IS Test:    {ms_to_str(is_test_ms)}  ({len(is_test_ms)} rows)")
print(f"  OOS Test:   {ms_to_str(oos_ms)}  ({len(oos_ms)} rows)")

# Check overlap
train_set = set(train_ms.values)
val_set = set(val_ms.values)
is_test_set = set(is_test_ms.values)
oos_set = set(oos_ms.values)

print(f"\n  Overlap check:")
print(f"    Train ∩ Val:       {len(train_set & val_set)} rows")
print(f"    Train ∩ IS Test:   {len(train_set & is_test_set)} rows")
print(f"    Val ∩ IS Test:     {len(val_set & is_test_set)} rows")
print(f"    IS Test ∩ OOS:     {len(is_test_set & oos_set)} rows")
print(f"    Train ∩ OOS:       {len(train_set & oos_set)} rows")

# Check chronological
is_chrono = train_ms.max() <= val_ms.min() and val_ms.max() <= is_test_ms.min()
oos_after_is = is_test_ms.max() < oos_ms.min()
print(f"\n  Chronological check:")
print(f"    IS train < val < test: {is_chrono}")
print(f"    OOS after IS test:     {oos_after_is}")

# =========================================================================
# 3. LABEL DISTRIBUTION CHECK
# =========================================================================
print("\n" + "=" * 70)
print("3. LABEL DISTRIBUTION PER SPLIT")
print("=" * 70)

splits = {
    "Train": grp_is.iloc[:train_end],
    "Val": grp_is.iloc[train_end:val_end],
    "IS Test": grp_is.iloc[val_end:],
    "OOS": grp_oos,
}

for name, split_df in splits.items():
    total = len(split_df)
    n_pos = int(split_df["outcome_binary"].sum())
    n_neg = total - n_pos
    ratio = n_pos / total if total > 0 else 0
    flag = " *** DEGENERATE ***" if ratio > 0.8 or ratio < 0.2 else ""
    print(f"  {name:10s}: {total:6d} rows, label=1: {n_pos:6d}, label=0: {n_neg:6d}, ratio={ratio:.3f}{flag}")

# =========================================================================
# 4. MODEL PROBABILITY CHECK (on IS validation)
# =========================================================================
print("\n" + "=" * 70)
print("4. MODEL PROBABILITY CHECK")
print("=" * 70)

train = grp_is.iloc[:train_end]
val = grp_is.iloc[train_end:val_end]

X_train = train[COMMON_FEATURES]
y_train = train["outcome_binary"]
X_val = val[COMMON_FEATURES]
y_val = val["outcome_binary"]
X_oos = grp_oos[COMMON_FEATURES]

print(f"\n  Training model on W={ws} H={hz}...")
model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
)

val_probs = model.predict_proba(X_val)[:, 1]
oos_probs = model.predict_proba(X_oos)[:, 1]

print(f"\n  IS Validation probabilities:")
print(f"    min={val_probs.min():.6f}, max={val_probs.max():.6f}, mean={val_probs.mean():.6f}")
print(f"    std={val_probs.std():.6f}")
print(f"    percentiles: 1%={np.percentile(val_probs, 1):.6f}, "
      f"25%={np.percentile(val_probs, 25):.6f}, "
      f"50%={np.percentile(val_probs, 50):.6f}, "
      f"75%={np.percentile(val_probs, 75):.6f}, "
      f"99%={np.percentile(val_probs, 99):.6f}")
print(f"    fraction > 0.9: {(val_probs > 0.9).mean():.4f}")
print(f"    fraction < 0.1: {(val_probs < 0.1).mean():.4f}")

print(f"\n  OOS probabilities:")
print(f"    min={oos_probs.min():.6f}, max={oos_probs.max():.6f}, mean={oos_probs.mean():.6f}")
print(f"    std={oos_probs.std():.6f}")
print(f"    fraction > 0.9: {(oos_probs > 0.9).mean():.4f}")
print(f"    fraction < 0.1: {(oos_probs < 0.1).mean():.4f}")

if val_probs.std() < 0.01:
    print("\n  *** BUG IS IN TRAINING: model predicts constant on validation too ***")
elif oos_probs.std() < 0.01:
    print("\n  *** BUG IS IN OOS DATA: model varies on val but constant on OOS ***")
else:
    print("\n  Model varies on both val and OOS — bug may be elsewhere")

# =========================================================================
# 5. FEATURE SCHEMA CHECK
# =========================================================================
print("\n" + "=" * 70)
print("5. IS vs OOS FEATURE SCHEMA CHECK")
print("=" * 70)

print(f"\n  IS features:  {len(FEATURE_COLS)} columns")
print(f"  OOS features: {len(oos_features)} columns")
print(f"  Common:       {len(COMMON_FEATURES)} columns")

missing_in_oos = set(FEATURE_COLS) - set(oos_features)
extra_in_oos = set(oos_features) - set(FEATURE_COLS)
if missing_in_oos:
    print(f"\n  *** {len(missing_in_oos)} features in IS but NOT in OOS: ***")
    for f in sorted(missing_in_oos):
        print(f"      {f}")
if extra_in_oos:
    print(f"\n  *** {len(extra_in_oos)} features in OOS but NOT in IS: ***")
    for f in sorted(extra_in_oos):
        print(f"      {f}")

# Check dtype mismatches
print(f"\n  Dtype comparison (first 20 common features):")
dtype_mismatches = []
for col in COMMON_FEATURES[:20]:
    is_dt = str(df_is[col].dtype)
    oos_dt = str(df_oos[col].dtype)
    match = "OK" if is_dt == oos_dt else "*** MISMATCH ***"
    if is_dt != oos_dt:
        dtype_mismatches.append(col)
    print(f"    {col:40s}  IS={is_dt:12s}  OOS={oos_dt:12s}  {match}")

if dtype_mismatches:
    print(f"\n  *** {len(dtype_mismatches)} dtype mismatches found! ***")

# Check for NaN/Inf in features
print(f"\n  NaN/Inf check:")
for name, df_check in [("IS", df_is), ("OOS", df_oos)]:
    nan_count = df_check[COMMON_FEATURES].isna().sum().sum()
    inf_count = np.isinf(df_check[COMMON_FEATURES].select_dtypes(include=[np.number]).values).sum()
    print(f"    {name}: NaN={nan_count}, Inf={inf_count}")

# Check feature value ranges
print(f"\n  Feature value range comparison (first 10 features):")
for col in COMMON_FEATURES[:10]:
    is_min, is_max = df_is[col].min(), df_is[col].max()
    oos_min, oos_max = df_oos[col].min(), df_oos[col].max()
    print(f"    {col:40s}  IS=[{is_min:.4f}, {is_max:.4f}]  OOS=[{oos_min:.4f}, {oos_max:.4f}]")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)