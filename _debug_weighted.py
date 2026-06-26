"""Test class-weighted training on 3-day futures."""
import lightgbm as lgb
import pandas as pd
import numpy as np

df = pd.read_parquet("data/research_dataset_futures_relabel.parquet")
META = {"window_size", "horizon", "window_end_ms"}
LABEL = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}
feats = [c for c in df.columns if c not in META and c not in LABEL]

pair = df[(df["window_size"] == 60) & (df["horizon"] == 1800)].sort_values("window_end_ms")
n = len(pair)
train = pair.iloc[:int(n*0.70)]
val = pair.iloc[int(n*0.70):int(n*0.85)]
test = pair.iloc[int(n*0.85):]

neg = (train["outcome_binary"] == 0).sum()
pos = (train["outcome_binary"] == 1).sum()
scale = neg / pos
print(f"Train: neg={neg}, pos={pos}, scale={scale:.1f}")

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,
    "min_child_samples": 30,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "scale_pos_weight": scale,
}
model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
model.fit(
    train[feats], train["outcome_binary"],
    eval_set=[(val[feats], val["outcome_binary"])],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
)
probs = model.predict_proba(test[feats])[:, 1]
print(f"\nWith scale_pos_weight={scale:.1f}:")
print(f"  min={probs.min():.4f}  max={probs.max():.4f}  mean={probs.mean():.4f}")
for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    n_sig = (probs >= t).sum()
    if n_sig > 0:
        wr = test["outcome_binary"].values[probs >= t].mean()
        print(f"  θ={t:.2f}: {n_sig:4d} signals, WR={wr:.3f}")
