import pandas as pd
df = pd.read_parquet("data/research_dataset_relabel.parquet")
print("Cols:", list(df.columns))
print("Shape:", df.shape)
print("\nPair counts:")
print(df.groupby(["window_size","horizon"]).size())
print("\nSample:")
print(df[["window_end_ms","window_size","horizon","outcome_binary","outcome_pct","target_hit_1pct","mae_pct"]].head(5).to_string())
print("\nTarget hit rate per (ws, hz):")
for (ws, hz), grp in df.groupby(["window_size","horizon"]):
    print(f"  W={ws} H={hz}: target_hit={grp['target_hit_1pct'].mean():.3f}, "
          f"mae_mean={grp['mae_pct'].mean()*100:+.3f}%, "
          f"win_rate={grp['outcome_binary'].mean():.3f}")
