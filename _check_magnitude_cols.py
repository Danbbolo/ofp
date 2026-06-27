import polars as pl
df = pl.read_parquet('data/research_dataset_v2_magnitude.parquet')
print("COLUMNS:", df.columns)
print("SHAPE:", df.shape)
print("LABEL DIST:")
print(df.group_by('label').agg(pl.len()).sort('label'))
print("DTYPES:")
for c in df.columns:
    print(f"  {c}: {df[c].dtype}")