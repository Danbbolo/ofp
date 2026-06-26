import pandas as pd
df = pd.read_parquet("data/research_dataset_futures_relabel.parquet")
print(f"Rows: {len(df)}")
print(f"Pair counts:")
for (ws, hz), grp in df.groupby(["window_size", "horizon"]):
    print(f"  W={ws} H={hz}: n={len(grp)}, wr={grp['outcome_binary'].mean():.3f}")
print()
META = {"window_size", "horizon", "window_end_ms"}
LABEL = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}
feats = [c for c in df.columns if c not in META and c not in LABEL]
print(f"{len(feats)} features")

# Test one pair end-to-end
pair = df[(df["window_size"] == 60) & (df["horizon"] == 1800)].sort_values("window_end_ms")
n = len(pair)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train = pair.iloc[:train_end]
val = pair.iloc[train_end:val_end]
test = pair.iloc[val_end:]

import lightgbm as lgb
import numpy as np

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,
    "min_child_samples": 30,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}
model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
model.fit(
    train[feats], train["outcome_binary"],
    eval_set=[(val[feats], val["outcome_binary"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
)
probs = model.predict(test[feats])
print(f"\nProb distribution on test:")
print(f"  min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  q{int(q*100):2d}: {np.quantile(probs, q):.4f}")

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
print(f"\nSignals per threshold:")
for t in THRESHOLDS:
    n_sig = (probs >= t).sum()
    print(f"  θ={t:.2f}: {n_sig} signals")
