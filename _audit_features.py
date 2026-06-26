"""
_audit_features.py — Check feature variance, uniqueness, and whether
features are actually informative or just noise/redundant.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/research_dataset_futures_relabel.parquet")
META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}
feature_cols = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]

print(f"Dataset: {len(df):,} rows, {len(feature_cols)} features")
print()

# Per-feature stats
stats = []
for c in feature_cols:
    col = df[c]
    stats.append({
        "feature": c,
        "mean": col.mean(),
        "std": col.std(),
        "min": col.min(),
        "max": col.max(),
        "zeros": (col == 0).sum(),
        "zeros_pct": (col == 0).sum() / len(col),
        "n_unique": col.nunique(),
        "cv": col.std() / abs(col.mean()) if col.mean() != 0 else np.inf,
    })

stats_df = pd.DataFrame(stats)

# Top features by zero percentage (likely useless)
print("=== FEATURES WITH MOST ZEROS (likely useless) ===")
print(stats_df.nlargest(15, "zeros_pct")[["feature", "zeros_pct", "mean", "std"]].to_string(index=False))
print()

# Bottom by zero percentage (always varying)
print("=== FEATURES WITH FEWEST ZEROS (always varying) ===")
print(stats_df.nsmallest(15, "zeros_pct")[["feature", "zeros_pct", "mean", "std"]].to_string(index=False))
print()

# Lowest coefficient of variation
print("=== FEATURES WITH LOW CV (constant-ish) ===")
print(stats_df.nsmallest(15, "cv")[["feature", "cv", "mean", "std"]].to_string(index=False))
print()

# Check correlation with target
print("=== TOP 20 FEATURES BY |CORRELATION WITH outcome_binary| ===")
correlations = []
for c in feature_cols:
    corr = df[c].corr(df["outcome_binary"])
    if not np.isnan(corr):
        correlations.append({"feature": c, "corr": corr, "abs_corr": abs(corr)})
corr_df = pd.DataFrame(correlations).sort_values("abs_corr", ascending=False)
print(corr_df.head(20).to_string(index=False))
