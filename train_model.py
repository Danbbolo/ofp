"""
train_model.py — Train LightGBM per (window_size, horizon) and build
the Expectancy Table across probability thresholds.

Chronological split (70/15/15), no shuffling, no scaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE = "data/research_dataset_target.parquet"
OUTPUT_FILE = "data/expectancy_table_target.csv"
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
COST_PER_TRADE = 0.001  # 0.1 %

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 50,  # was 500, too aggressive for ~20k rows per pair
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

FEATURE_COLS: list[str] = []  # filled after loading
META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chronological_split_per_pair(df: pd.DataFrame) -> dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Chronological 70/15/15 split WITHIN each (window_size, horizon) pair.

    Why per-pair: ``window_end_ms`` repeats across pairs (e.g. (60s, 300s)
    and (60s, 900s) both have rows at the same time).  A global sort splits
    those rows together, which is fine, but per-pair splitting is stricter
    and ensures each pair's test set is strictly the latest 15 % of that
    pair's data, with no contamination from other pairs' late-window rows.
    """
    out: dict[tuple[int, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for (ws, hz), grp in df.groupby(["window_size", "horizon"], sort=True):
        g = grp.sort_values("window_end_ms").reset_index(drop=True)
        n = len(g)
        if n < 3:
            out[(int(ws), int(hz))] = (g, g.iloc[0:0], g.iloc[0:0])
            continue
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        out[(int(ws), int(hz))] = (
            g.iloc[:train_end],
            g.iloc[train_end:val_end],
            g.iloc[val_end:],
        )
    return out


def _evaluate(
    model: lgb.Booster, test: pd.DataFrame, window_size: int, horizon: int
) -> list[dict]:
    """Evaluate one model across all thresholds on its (ws, hz) test subset."""
    X_test = test[FEATURE_COLS]
    y_true_bin = test["outcome_binary"].values
    y_true_pct = test["outcome_pct"].values
    n_days = (test["window_end_ms"].max() - test["window_end_ms"].min()) / 86_400_000 + 1

    probs = model.predict(X_test)
    rows = []

    for thresh in THRESHOLDS:
        signals = probs >= thresh
        n_signals = int(signals.sum())
        if n_signals == 0:
            rows.append({
                "window_size": window_size, "horizon": horizon, "threshold": thresh,
                "signals_per_day": 0.0, "n_signals": 0, "n_test": len(test),
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
            })
            continue

        sig_bin = y_true_bin[signals]
        sig_pct = y_true_pct[signals]
        wins = sig_bin == 1
        n_wins = int(wins.sum())
        n_losses = n_signals - n_wins

        win_rate = n_wins / n_signals
        avg_win = float(sig_pct[wins].mean()) if n_wins > 0 else 0.0
        avg_loss = float(np.abs(sig_pct[~wins]).mean()) if n_losses > 0 else 0.0
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) - COST_PER_TRADE

        rows.append({
            "window_size": window_size, "horizon": horizon, "threshold": thresh,
            "signals_per_day": n_signals / max(n_days, 1),
            "n_signals": n_signals, "n_test": len(test),
            "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": expectancy,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INPUT_FILE} …")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
    print(f"  {len(FEATURE_COLS)} features")
    print()

    # Per-pair chronological split (70/15/15 within each (window_size, horizon))
    print("Per-pair chronological split (70/15/15) …")
    pair_splits = _chronological_split_per_pair(df)
    total_train = sum(len(t) for t, _, _ in pair_splits.values())
    total_val = sum(len(v) for _, v, _ in pair_splits.values())
    total_test = sum(len(t) for _, _, t in pair_splits.values())
    print(f"  Train: {total_train:,}  Val: {total_val:,}  Test: {total_test:,}")
    print()

    # Train one model per (window_size, horizon)
    pairs = sorted(pair_splits.keys())
    all_rows: list[dict] = []
    importance_dfs: list[pd.DataFrame] = []  # collect per-pair importances

    for i, (ws, hz) in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] W={ws}s  H={hz}s …", end=" ", flush=True)

        train, val, test = pair_splits[(ws, hz)]

        if len(train) < 100 or len(val) < 50:
            print(f"SKIP (train={len(train)}, val={len(val)} — too few)")
            continue

        X_train, y_train = train[FEATURE_COLS], train["outcome_binary"]
        X_val, y_val = val[FEATURE_COLS], val["outcome_binary"]

        model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )

        rows = _evaluate(model.booster_, test, ws, hz)
        all_rows.extend(rows)

        # Collect feature importance (split)
        imp = model.booster_.feature_importance(importance_type="split")
        imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
        imp_df["window_size"] = ws
        imp_df["horizon"] = hz
        importance_dfs.append(imp_df)

        best = max(rows, key=lambda r: r["expectancy"])
        print(f"best θ={best['threshold']:.2f} exp={best['expectancy']:.6f} "
              f"wr={best['win_rate']:.4f} sig/day={best['signals_per_day']:.1f}")

    # Save
    table = pd.DataFrame(all_rows)
    table = table.sort_values("expectancy", ascending=False)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")

    # Top 10
    print("\n=== TOP 10 BY EXPECTANCY ===")
    cols = ["window_size", "horizon", "threshold", "expectancy", "win_rate",
            "signals_per_day", "n_signals", "avg_win", "avg_loss"]
    print(table[cols].head(10).to_string(index=False))

    # Best combo
    best = table.iloc[0]
    print(f"\n=== BEST COMBO ===")
    print(f"  window_size = {int(best['window_size'])}s")
    print(f"  horizon     = {int(best['horizon'])}s")
    print(f"  threshold   = {best['threshold']:.2f}")
    print(f"  expectancy  = {best['expectancy']:.6f}")
    print(f"  win_rate    = {best['win_rate']:.4f}")
    print(f"  signals/day = {best['signals_per_day']:.1f}")

    # Feature importance (average across all pairs)
    if importance_dfs:
        all_imp = pd.concat(importance_dfs, ignore_index=True)
        avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
        print(f"\n=== TOP 10 FEATURES BY AVG IMPORTANCE (split) ===")
        for rank, (feat, imp_val) in enumerate(avg_imp.head(10).items(), 1):
            print(f"  {rank:2d}. {feat:40s}  {imp_val:.1f}")


if __name__ == "__main__":
    main()
