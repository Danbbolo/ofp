"""Check what dates the IS and OOS move_start datasets cover."""
import pandas as pd

# IS futures dataset
df = pd.read_parquet('data/research_dataset_futures_move_start.parquet')
print('IS futures dataset:')
print(f'  Rows: {len(df)}')
ts = pd.to_datetime(df['window_end_ms'], unit='ms')
print(f'  Date range: {ts.min()} to {ts.max()}')
print(f'  Unique dates: {sorted(ts.dt.date.unique())}')
print(f'  W values: {sorted(df["window_size"].unique())}')
print(f'  H values: {sorted(df["horizon"].unique())}')
print(f'  had_move rate: {df["had_move"].mean():.3f}')

print()

# OOS futures dataset
df2 = pd.read_parquet('data/research_dataset_oos_move_start.parquet')
print('OOS futures dataset:')
print(f'  Rows: {len(df2)}')
ts2 = pd.to_datetime(df2['window_end_ms'], unit='ms')
print(f'  Date range: {ts2.min()} to {ts2.max()}')
print(f'  Unique dates: {sorted(ts2.dt.date.unique())}')
print(f'  W values: {sorted(df2["window_size"].unique())}')
print(f'  H values: {sorted(df2["horizon"].unique())}')
print(f'  had_move rate: {df2["had_move"].mean():.3f}')