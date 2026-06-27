"""Inspect pre-relabel dataset columns and label distributions."""
import pandas as pd

for name, path in [("IS", "data/research_dataset_futures.parquet"),
                   ("OOS", "data/research_dataset_oos.parquet")]:
    df = pd.read_parquet(path)
    print(f"=== {name}: {path} ===")
    print(f"  Rows: {len(df):,}  Cols: {len(df.columns)}")
    print(f"  Has had_move: {'had_move' in df.columns}")
    print(f"  Has move_direction: {'move_direction' in df.columns}")
    print(f"  Has move_pct: {'move_pct' in df.columns}")
    print(f"  Has outcome_binary: {'outcome_binary' in df.columns}")
    print(f"  Has outcome_pct: {'outcome_pct' in df.columns}")
    print(f"  outcome_binary rate: {df['outcome_binary'].mean():.4f}")
    print(f"  outcome_pct mean: {df['outcome_pct'].mean()*100:+.4f}%")
    print(f"  outcome_pct std: {df['outcome_pct'].std()*100:.4f}%")
    if "had_move" in df.columns:
        print(f"  had_move rate: {df['had_move'].mean():.4f}")
    print(f"  horizons: {sorted(df['horizon'].unique())}")
    print(f"  windows: {sorted(df['window_size'].unique())}")
    # Label cols present
    labelish = [c for c in df.columns if c in {"outcome_binary", "outcome_pct", "had_move", "move_direction", "move_pct", "target_hit", "stop_hit", "time_based_exit", "hold_sec"}]
    print(f"  Label-ish cols: {labelish}")
    print()