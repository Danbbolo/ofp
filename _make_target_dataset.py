"""
_make_target_dataset.py — Convert relabeled dataset to target-based labels.

For the OFP model, we want:
  outcome_binary = target_hit_1pct (1 = hit +1% before -1%, 0 = stop first or neither)
  outcome_pct    = mae_pct (the realized move at 24h or at stop)

This makes the training target match the trader's +1%/-1% exit style.
"""
import pandas as pd

df = pd.read_parquet("data/research_dataset_relabel.parquet")
print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

# Replace outcome_binary and outcome_pct with target-based versions
df["outcome_binary"] = df["target_hit_1pct"].astype(int)
df["outcome_pct"] = df["mae_pct"]

# Save
out = "data/research_dataset_target.parquet"
df.to_parquet(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(f"  outcome_binary mean: {df['outcome_binary'].mean():.3f}")
print(f"  outcome_pct mean:    {df['outcome_pct'].mean()*100:+.3f}%")
