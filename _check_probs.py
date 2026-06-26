"""
_check_probs.py — Check what probabilities the model actually outputs.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 50,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

df = pd.read_parquet("data/research_dataset_target.parquet")
feature_cols = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
print(f"{len(df):,} rows, {len(feature_cols)} features")

# Check W=60 H=1800
pair = df[(df["window_size"] == 60) & (df["horizon"] == 1800)].sort_values("window_end_ms")
n = len(pair)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = pair.iloc[:train_end]
val = pair.iloc[train_end:val_end]
test = pair.iloc[val_end:]

model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
model.fit(
    train[feature_cols], train["outcome_binary"],
    eval_set=[(val[feature_cols], val["outcome_binary"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
)

probs = model.predict_proba(test[feature_cols])[:, 1]
print(f"\nProbability distribution on test (W=60, H=1800):")
print(f"  min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}  median={np.median(probs):.4f}")
for q in [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
    print(f"  q{int(q*100):2d}: {np.quantile(probs, q):.4f}")

print(f"\n  Actual win rate in test: {test['outcome_binary'].mean():.4f}")
print(f"  Signals at θ=0.20: {(probs >= 0.20).sum()}")
print(f"  Signals at θ=0.15: {(probs >= 0.15).sum()}")
print(f"  Signals at θ=0.10: {(probs >= 0.10).sum()}")
print(f"  Signals at θ=0.05: {(probs >= 0.05).sum()}")

# Check expectancy at lower thresholds
print(f"\nExpectancy at low thresholds (W=60, H=1800):")
y_true = test["outcome_binary"].values
y_pct = test["outcome_pct"].values
for thresh in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    sig = probs >= thresh
    n_sig = sig.sum()
    if n_sig == 0:
        continue
    wr = y_true[sig].mean()
    avg_win = y_pct[sig][y_true[sig] == 1].mean() if (y_true[sig] == 1).any() else 0
    avg_loss = abs(y_pct[sig][y_true[sig] == 0].mean()) if (y_true[sig] == 0).any() else 0
    exp = wr * avg_win - (1 - wr) * avg_loss - 0.001
    print(f"  θ={thresh:.2f}: n={n_sig:4d}  wr={wr:.3f}  avg_win={avg_win*100:+.3f}%  avg_loss={avg_loss*100:.3f}%  exp={exp*100:+.3f}%")
