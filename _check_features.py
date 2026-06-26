import pandas as pd
df = pd.read_parquet("data/research_dataset_target.parquet")

# Check if any liq features are non-zero
liq_cols = [c for c in df.columns if "liq" in c.lower()]
print("Liq columns:", liq_cols)
for c in liq_cols:
    print(f"  {c}: mean={df[c].mean():.6f}, max={df[c].max():.4f}, nonzero={(df[c] != 0).sum()}")
print()

# Check funding/OI features
fund_cols = [c for c in df.columns if "fund" in c.lower() or "oi" in c.lower() or "open_interest" in c.lower()]
print("Funding/OI columns:", fund_cols)

# Show all column names that contain "fund" or "open" or "interest"
for c in df.columns:
    if any(k in c.lower() for k in ["fund", "open_int", "oi_", "basis"]):
        print(f"  {c}: mean={df[c].mean():.6f}")
