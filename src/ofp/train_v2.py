"""
train_v2.py — Walk-forward training on v2 features + magnitude labels.

Pipeline:
  1. Load magnitude-labeled volume bars (data/research_dataset_v2_magnitude.parquet)
  2. Extract v2 features per day (or load cached feature dataset)
  3. Join features + labels on timestamp_ms
  4. Walk-forward train LightGBM with 10-day purge
  5. Report OOS expectancy, win rate, signals/day, prob std, feature importance

If walk-forward can't produce valid folds (not enough data with 10-day purge),
falls back to chronological 70/15/15 split for this initial test.

Usage:
    python -m src.ofp.train_v2
    python -m src.ofp.train_v2 --skip-extraction  # use cached features
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import lightgbm as lgb

from src.ofp.walk_forward import WalkForwardSplitter, MS_PER_DAY, _day_of

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGNITUDE_FILE = Path("data/research_dataset_v2_magnitude.parquet")
FEATURE_FILE = Path("data/research_dataset_v2_features.parquet")
RAW_DIR = Path("data/raw_futures")

COST_PER_TRADE = 0.0015   # 15 bps round-trip
THETA = 0.60              # probability threshold
MIN_CHILD_SAMPLES = 500   # prevent overfit
NUM_LEAVES = 31
EARLY_STOPPING = 20
N_JOBS = 14

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": NUM_LEAVES,
    "min_child_samples": MIN_CHILD_SAMPLES,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": N_JOBS,
}

FEATURE_COLS = [
    "vpin", "ofi", "book_delta", "trade_arrival_rate", "liq_volume",
    "vpin_x_arrival", "ofi_x_book_delta", "liq_x_vpin",
    "vpin_x_duration", "liq_x_return",
]

LABEL_COL = "label"
TIMESTAMP_COL = "timestamp_ms"


# ---------------------------------------------------------------------------
# Feature extraction (or load cache)
# ---------------------------------------------------------------------------

def build_feature_dataset() -> pl.DataFrame:
    """
    Extract v2 features for all dates in the magnitude dataset.
    Joins features with magnitude labels on timestamp_ms.
    """
    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING FEATURE DATASET ===")

    # Load magnitude labels
    mag = pl.read_parquet(str(MAGNITUDE_FILE))
    print(f"  Magnitude labels: {len(mag):,} rows")

    # Get unique dates from timestamps
    ts = mag[TIMESTAMP_COL].to_numpy()
    days = sorted(set(_day_of(int(t)) for t in ts))
    date_strs = [datetime.utcfromtimestamp(d * MS_PER_DAY / 1000).strftime("%Y-%m-%d") for d in days]
    print(f"  Dates: {date_strs}")

    all_features = []
    for date_str in date_strs:
        raw = RAW_DIR / date_str
        trades_path = raw / "trades.parquet"
        if not trades_path.exists():
            print(f"  {date_str}: no raw data, skipping")
            continue

        print(f"  Extracting features for {date_str} …", flush=True)
        bars = build_volume_bars(trades_path, volume_threshold=50.0)
        feats = extract_features(bars, raw)
        all_features.append(feats)
        print(f"    {len(feats)} bars", flush=True)

    features = pl.concat(all_features, how="vertical")
    print(f"  Total feature rows: {len(features):,}")

    # Join with magnitude labels on timestamp_ms
    dataset = features.join(mag, on=TIMESTAMP_COL, how="inner")
    print(f"  Joined dataset: {len(dataset):,} rows")

    # Save cache
    dataset.write_parquet(str(FEATURE_FILE))
    print(f"  Saved to {FEATURE_FILE}")

    return dataset


def load_or_build_features(skip_extraction: bool = False) -> pl.DataFrame:
    """Load cached features or build from scratch."""
    if skip_extraction and FEATURE_FILE.exists():
        print(f"Loading cached features from {FEATURE_FILE} …")
        return pl.read_parquet(str(FEATURE_FILE))
    return build_feature_dataset()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_walkforward(df: pl.DataFrame) -> None:
    """Run walk-forward training across all valid folds."""
    ts = df[TIMESTAMP_COL].to_numpy()
    y = df[LABEL_COL].to_numpy()
    X = df.select(FEATURE_COLS).to_numpy()

    # Exclude Label 2 (No Trade)
    tradeable = y != 2
    X = X[tradeable]
    y = y[tradeable]
    ts = ts[tradeable]
    y = y.astype(np.int8)  # 0 or 1

    n = len(X)
    n_days = len(set(_day_of(int(t)) for t in ts))
    print(f"\n=== TRAINING DATA ===")
    print(f"  Tradeable rows: {n:,}")
    print(f"  Days: {n_days}")
    print(f"  Base rate (Label 1): {y.mean():.4f}")
    print(f"  Features: {len(FEATURE_COLS)}")

    # Try walk-forward with 10-day purge
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)

    if len(folds) == 0:
        print(f"\n  *** No valid walk-forward folds with 10-day purge ***")
        print(f"  *** Falling back to chronological 70/15/15 split ***")
        train_chronological(X, y, ts)
        return

    print(f"\n=== WALK-FORWARD FOLDS ({len(folds)}) ===")
    all_results = []
    all_importance = np.zeros(len(FEATURE_COLS))

    for fold in folds:
        info = splitter.get_fold_info(fold)
        print(f"\n--- Fold {fold.fold_idx} ---")
        print(f"  Train: {info['train_days']} ({len(fold.train_indices)} rows)")
        print(f"  Test:  {info['test_days']} ({len(fold.test_indices)} rows)")

        X_train, y_train = X[fold.train_indices], y[fold.train_indices]
        X_test, y_test = X[fold.test_indices], y[fold.test_indices]

        # Use last 20% of train as validation for early stopping
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # Predict on test
        probs = model.predict_proba(X_test)[:, 1]
        prob_std = float(np.std(probs))
        prob_min = float(probs.min())
        prob_max = float(probs.max())

        # Signals at theta
        signals = probs >= THETA
        n_sig = int(signals.sum())
        n_test_days = len(set(_day_of(int(t)) for t in ts[fold.test_indices]))

        if n_sig > 0:
            wr = float(y_test[signals].mean())
            # Expectancy: win = +1% (100 bps), loss = -1% (100 bps), minus cost
            # Actual magnitude varies, but label is binary target/stop
            # Use 100 bps win, 100 bps loss as approximation
            avg_win = 100.0   # bps
            avg_loss = 100.0  # bps
            expectancy_bps = wr * avg_win - (1 - wr) * avg_loss - COST_PER_TRADE * 10_000
            expectancy_pct = expectancy_bps / 100.0
        else:
            wr = 0.0
            expectancy_bps = 0.0
            expectancy_pct = 0.0

        sig_per_day = n_sig / max(n_test_days, 1)

        # Feature importance
        imp = model.booster_.feature_importance(importance_type="split")
        all_importance += imp

        # Print fold results
        print(f"  OOS prob range: [{prob_min:.4f}, {prob_max:.4f}], std={prob_std:.4f}")
        if prob_std < 0.05:
            print(f"  *** NO SIGNAL (prob std < 0.05) ***")
        print(f"  OOS signals at θ={THETA}: {n_sig} ({sig_per_day:.1f}/day)")
        print(f"  OOS win rate: {wr:.4f}")
        print(f"  OOS expectancy: {expectancy_pct:+.3f}% ({expectancy_bps:+.1f} bps)")

        # Top 5 features
        top5 = np.argsort(imp)[::-1][:5]
        print(f"  Top 5 features:")
        for rank, idx in enumerate(top5, 1):
            print(f"    {rank}. {FEATURE_COLS[idx]:25s}  {imp[idx]:.0f}")

        all_results.append({
            "fold": fold.fold_idx,
            "train_n": len(fold.train_indices),
            "test_n": len(fold.test_indices),
            "prob_std": prob_std,
            "n_signals": n_sig,
            "sig_per_day": sig_per_day,
            "win_rate": wr,
            "expectancy_bps": expectancy_bps,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"=== WALK-FORWARD SUMMARY ({len(folds)} folds) ===")
    print(f"{'=' * 70}")

    avg_std = np.mean([r["prob_std"] for r in all_results])
    avg_wr = np.mean([r["win_rate"] for r in all_results])
    total_sig = sum(r["n_signals"] for r in all_results)
    avg_exp = np.mean([r["expectancy_bps"] for r in all_results])

    print(f"  Avg prob std:     {avg_std:.4f}  {'<-- NO SIGNAL' if avg_std < 0.05 else ''}")
    print(f"  Avg win rate:     {avg_wr:.4f}")
    print(f"  Total signals:    {total_sig}")
    print(f"  Avg expectancy:   {avg_exp:+.1f} bps ({avg_exp/100:+.3f}%)")
    print(f"  Cost per trade:   {COST_PER_TRADE*10_000:.0f} bps")

    print(f"\n  === TOP 5 FEATURES (avg importance) ===")
    avg_imp = all_importance / len(folds)
    top5 = np.argsort(avg_imp)[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        print(f"    {rank}. {FEATURE_COLS[idx]:25s}  {avg_imp[idx]:.1f}")

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 70}")
    if avg_std < 0.05:
        print(f"  NO SIGNAL — model probabilities have std < 0.05")
        print(f"  The model is not discriminating between classes.")
    elif avg_exp <= 0:
        print(f"  NO EDGE — expectancy is {avg_exp:+.1f} bps (cost = {COST_PER_TRADE*10_000:.0f} bps)")
        print(f"  Signals may be real but not large enough to clear fees.")
    else:
        print(f"  POTENTIAL EDGE — expectancy = {avg_exp:+.1f} bps after {COST_PER_TRADE*10_000:.0f} bps cost")
        print(f"  WARNING: Only {len(folds)} folds. Need 60+ days to trust.")


def train_chronological(X: np.ndarray, y: np.ndarray, ts: np.ndarray) -> None:
    """Fallback: chronological 70/15/15 split."""
    n = len(X)
    split1 = int(n * 0.70)
    split2 = int(n * 0.85)

    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]
    ts_test = ts[split2:]

    print(f"\n=== CHRONOLOGICAL SPLIT (70/15/15) ===")
    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val:   {len(X_val):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    print(f"  Test base rate: {y_test.mean():.4f}")

    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=2000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    probs = model.predict_proba(X_test)[:, 1]
    prob_std = float(np.std(probs))
    prob_min = float(probs.min())
    prob_max = float(probs.max())

    signals = probs >= THETA
    n_sig = int(signals.sum())
    n_test_days = len(set(_day_of(int(t)) for t in ts_test))

    if n_sig > 0:
        wr = float(y_test[signals].mean())
        expectancy_bps = wr * 100.0 - (1 - wr) * 100.0 - COST_PER_TRADE * 10_000
        expectancy_pct = expectancy_bps / 100.0
    else:
        wr = 0.0
        expectancy_bps = 0.0
        expectancy_pct = 0.0

    sig_per_day = n_sig / max(n_test_days, 1)

    print(f"\n=== OOS RESULTS (θ={THETA}) ===")
    print(f"  Prob range: [{prob_min:.4f}, {prob_max:.4f}], std={prob_std:.4f}")
    if prob_std < 0.05:
        print(f"  *** NO SIGNAL (prob std < 0.05) ***")
    print(f"  Signals: {n_sig} ({sig_per_day:.1f}/day)")
    print(f"  Win rate: {wr:.4f}")
    print(f"  Expectancy: {expectancy_pct:+.3f}% ({expectancy_bps:+.1f} bps)")

    # Feature importance
    imp = model.booster_.feature_importance(importance_type="split")
    top5 = np.argsort(imp)[::-1][:5]
    print(f"\n  Top 5 features:")
    for rank, idx in enumerate(top5, 1):
        print(f"    {rank}. {FEATURE_COLS[idx]:25s}  {imp[idx]:.0f}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 70}")
    if prob_std < 0.05:
        print(f"  NO SIGNAL — model probabilities have std < 0.05")
    elif expectancy_bps <= 0:
        print(f"  NO EDGE — expectancy = {expectancy_bps:+.1f} bps (cost = {COST_PER_TRADE*10_000:.0f} bps)")
    else:
        print(f"  POTENTIAL EDGE — expectancy = {expectancy_bps:+.1f} bps after cost")
        print(f"  WARNING: Only 7 days of data. Need 60+ days to trust.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_extraction = "--skip-extraction" in sys.argv

    print("=" * 70)
    print("=== OFP v2 TRAINING (magnitude labels, walk-forward, 15 bps cost) ===")
    print("=" * 70)
    print(f"  min_child_samples: {MIN_CHILD_SAMPLES}")
    print(f"  num_leaves:        {NUM_LEAVES}")
    print(f"  theta:             {THETA}")
    print(f"  cost:              {COST_PER_TRADE*10_000:.0f} bps")
    print(f"  purge:             10 days")

    # Load or build feature dataset
    df = load_or_build_features(skip_extraction=skip_extraction)

    print(f"\n  Dataset: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Label distribution:")
    for label_val, count in df.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL).iter_rows():
        print(f"    Label {label_val}: {count:,}")

    # Train
    train_walkforward(df)


if __name__ == "__main__":
    main()