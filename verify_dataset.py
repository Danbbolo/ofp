"""
verify_dataset.py — Sanity-check the research dataset for nulls, leakage,
desync, and class balance before model training.

Usage::

    python verify_dataset.py [data/research_dataset.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(filepath: str) -> None:
    print(f"Loading {filepath} …")
    df = pd.read_parquet(filepath)
    print(f"  Rows: {len(df):,}   Cols: {len(df.columns)}")
    print()

    anomalies = 0
    label_cols = {"outcome_pct", "outcome_binary"}
    meta_cols = {"window_size", "horizon", "window_end_ms"}
    feature_cols = [c for c in df.columns if c not in label_cols and c not in meta_cols]

    # ==================================================================
    # 1. NULLS / INF
    # ==================================================================
    print("=" * 60)
    print("1. NULLS & INFINITIES")
    print("=" * 60)

    for col in df.columns:
        n_null = int(df[col].isna().sum())
        n_inf = int(np.isinf(df[col].values).sum()) if df[col].dtype.kind in "fc" else 0
        if n_null > 0 or n_inf > 0:
            _flag(f"{col}: {n_null} nulls, {n_inf} infs")
            anomalies += 1

    if anomalies == 0:
        _ok("No nulls or infinities found.")

    # ==================================================================
    # 2. LABEL LEAKAGE
    # ==================================================================
    print()
    print("=" * 60)
    print("2. LABEL LEAKAGE CHECKS")
    print("=" * 60)

    # 2a. outcome_binary ∈ {0, 1}
    bad_binary = df[~df["outcome_binary"].isin([0.0, 1.0])]
    if len(bad_binary) > 0:
        _flag(f"outcome_binary has {len(bad_binary)} rows not in {{0, 1}}")
        anomalies += 1
    else:
        _ok("outcome_binary is strictly 0 or 1")

    # 2b. Sign consistency: positive pct → binary 1, negative/non-positive → binary 0
    sign_mismatch = df[
        ((df["outcome_pct"] > 0) & (df["outcome_binary"] != 1))
        | ((df["outcome_pct"] <= 0) & (df["outcome_binary"] != 0))
    ]
    if len(sign_mismatch) > 0:
        _flag(f"outcome_pct / outcome_binary sign mismatch: {len(sign_mismatch)} rows")
        anomalies += 1
    else:
        _ok("outcome_pct sign matches outcome_binary in all rows")

    # 2c. Feature correlation with labels (>0.95 = suspicious)
    print("  Checking feature-label correlations …")
    high_corr = []
    for col in feature_cols:
        if df[col].dtype.kind not in "fc":
            continue
        if df[col].nunique() <= 1:
            continue
        corr_pct = abs(df[col].corr(df["outcome_pct"]))
        corr_bin = abs(df[col].corr(df["outcome_binary"]))
        if corr_pct > 0.95:
            high_corr.append((col, "outcome_pct", corr_pct))
        if corr_bin > 0.95:
            high_corr.append((col, "outcome_binary", corr_bin))

    if high_corr:
        for col, target, val in high_corr:
            _flag(f"{col} correlated with {target} at r={val:.4f} — possible leakage!")
        anomalies += len(high_corr)
    else:
        _ok("No feature has >0.95 correlation with outcome_pct or outcome_binary")

    # ==================================================================
    # 3. DATA SANITY
    # ==================================================================
    print()
    print("=" * 60)
    print("3. DATA SANITY")
    print("=" * 60)

    # 3a. Monotonic window_end_ms within each group
    groups = df.groupby(["window_size", "horizon"])
    non_mono = 0
    for (ws, hz), grp in groups:
        ts = grp["window_end_ms"].values
        if not (ts[1:] >= ts[:-1]).all():
            _flag(f"window_size={ws}, horizon={hz}: window_end_ms NOT monotonic")
            non_mono += 1
    if non_mono == 0:
        _ok("window_end_ms is monotonic within every (window_size, horizon) group")

    # 3b. No duplicate (window_end_ms, window_size, horizon)
    dups = df.duplicated(subset=["window_end_ms", "window_size", "horizon"]).sum()
    if dups > 0:
        _flag(f"{dups} duplicate (window_end_ms, window_size, horizon) rows found")
        anomalies += 1
    else:
        _ok("No duplicate (window_end_ms, window_size, horizon) rows")

    # 3c. outcome_binary distribution
    print()
    print("  outcome_binary distribution:")
    dist = df["outcome_binary"].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {int(label)}: {count:,}  ({pct:.1f}%)")

    if dist.get(0.0, 0) == 0 or dist.get(1.0, 0) == 0:
        _flag("Single class!  No variation in outcome_binary.")
        anomalies += 1
    elif min(dist.get(0.0, 0), dist.get(1.0, 0)) / len(df) < 0.01:
        _flag("Severe class imbalance (<1% minority class)")
        anomalies += 1
    else:
        _ok("Class balance is acceptable")

    # ==================================================================
    # 4. SUMMARY PER (window_size, horizon)
    # ==================================================================
    print()
    print("=" * 60)
    print("4. ROWS PER (window_size, horizon)")
    print("=" * 60)

    summary = (
        df.groupby(["window_size", "horizon"])
        .agg(rows=("outcome_binary", "count"), win_rate=("outcome_binary", "mean"))
        .sort_index()
    )
    for (ws, hz), row in summary.iterrows():
        print(f"  W={ws:>4}s  H={hz:>5}s  →  {int(row['rows']):>7,} rows  "
              f"win_rate={row['win_rate']:.4f}")

    # ==================================================================
    # FINAL
    # ==================================================================
    print()
    if anomalies == 0:
        print("✓  Dataset passes all checks. Ready for training.")
    else:
        print(f"⚠  {anomalies} anomalies found.  Review before training.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/research_dataset.parquet"
    main(path)
