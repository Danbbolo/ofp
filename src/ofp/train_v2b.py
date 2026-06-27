"""
train_v2b.py — Multi-horizon training with fixed parameters.

Fixes from train_v2:
  - min_child_samples: 500 → 50 (was crippling the model)
  - Multiple horizons: 1h, 2h, 4h, 24h (24h was diluting signal)
  - Multiple thresholds: 0.50, 0.55, 0.60, 0.65
  - Same 10 features, same volume bars, same labels (just different horizons)

Pipeline:
  1. For each horizon, load the corresponding magnitude-labeled dataset
  2. Extract v2 features (or load cached)
  3. Chronological 70/15/15 split (walk-forward needs more data)
  4. Train LightGBM with min_child_samples=50
  5. Evaluate at 4 threshold levels
  6. Print summary table sorted by OOS expectancy

Usage:
    python -m src.ofp.train_v2b
    python -m src.ofp.train_v2b --skip-extraction  # use cached features
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import polars as pl
import lightgbm as lgb

from src.ofp.walk_forward import _day_of

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = [3600, 7200, 14400, 86400]  # 1h, 2h, 4h, 24h in seconds
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

RAW_DIR = Path("data/raw_futures")
FEATURE_CACHE = Path("data/research_dataset_v2_features.parquet")

COST_PER_TRADE = 0.0015   # 15 bps round-trip
MIN_CHILD_SAMPLES = 50    # FIXED: was 500, way too high for 18k rows
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
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HorizonResult:
    horizon_s: int
    theta: float
    prob_std: float
    prob_min: float
    prob_max: float
    n_signals: int
    sig_per_day: float
    win_rate: float
    expectancy_bps: float
    top5_features: list[tuple[str, int]]
    has_signal: bool


# ---------------------------------------------------------------------------
# Feature extraction (or load cache)
# ---------------------------------------------------------------------------

def build_feature_dataset(mag_file: Path) -> pl.DataFrame:
    """
    Extract v2 features for all dates in the magnitude dataset.
    Joins features with magnitude labels on timestamp_ms.
    """
    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING FEATURE DATASET ===")

    # Load magnitude labels
    mag = pl.read_parquet(str(mag_file))
    print(f"  Magnitude labels: {len(mag):,} rows")

    # Get unique dates from timestamps
    ts = mag[TIMESTAMP_COL].to_numpy()
    days = sorted(set(_day_of(int(t)) for t in ts))
    date_strs = [datetime.utcfromtimestamp(d * 86400000 / 1000).strftime("%Y-%m-%d") for d in days]
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

    return dataset


def load_or_build_features(mag_file: Path, skip_extraction: bool = False) -> pl.DataFrame:
    """Load cached features or build from scratch."""
    # Use horizon-specific cache
    cache_file = Path(str(mag_file).replace(".parquet", "_features.parquet"))
    if skip_extraction and cache_file.exists():
        print(f"Loading cached features from {cache_file} …")
        return pl.read_parquet(str(cache_file))

    dataset = build_feature_dataset(mag_file)

    # Save cache
    dataset.write_parquet(str(cache_file))
    print(f"  Saved to {cache_file}")
    return dataset


# ---------------------------------------------------------------------------
# Training for a single horizon
# ---------------------------------------------------------------------------

def train_single_horizon(
    df: pl.DataFrame,
    horizon_s: int,
) -> list[HorizonResult]:
    """
    Train LightGBM on a single horizon dataset.
    Returns results for all threshold levels.
    """
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
    base_rate = float(y.mean())

    print(f"\n=== HORIZON {horizon_s}s ({horizon_s / 3600:.0f}h) ===")
    print(f"  Tradeable rows: {n:,}")
    print(f"  Days: {n_days}")
    print(f"  Base rate (Label 1): {base_rate:.4f}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  min_child_samples: {MIN_CHILD_SAMPLES}")

    # Chronological 70/15/15 split
    split1 = int(n * 0.70)
    split2 = int(n * 0.85)

    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]
    ts_test = ts[split2:]

    print(f"  Train: {len(X_train):,} rows")
    print(f"  Val:   {len(X_val):,} rows")
    print(f"  Test:  {len(X_test):,} rows")
    print(f"  Test base rate: {y_test.mean():.4f}")

    # Train model
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

    # Predict on test
    probs = model.predict_proba(X_test)[:, 1]
    prob_std = float(np.std(probs))
    prob_min = float(probs.min())
    prob_max = float(probs.max())

    # Feature importance
    imp = model.booster_.feature_importance(importance_type="split")
    top5_idx = np.argsort(imp)[::-1][:5]
    top5_features = [(FEATURE_COLS[i], int(imp[i])) for i in top5_idx]

    n_test_days = len(set(_day_of(int(t)) for t in ts_test))

    print(f"  Prob range: [{prob_min:.4f}, {prob_max:.4f}], std={prob_std:.4f}")
    if prob_std < 0.05:
        print(f"  *** NO SIGNAL (prob std < 0.05) ***")

    # Evaluate at each threshold
    results = []
    for theta in THRESHOLDS:
        signals = probs >= theta
        n_sig = int(signals.sum())

        if n_sig > 0:
            wr = float(y_test[signals].mean())
            expectancy_bps = wr * 100.0 - (1 - wr) * 100.0 - COST_PER_TRADE * 10_000
        else:
            wr = 0.0
            expectancy_bps = 0.0

        sig_per_day = n_sig / max(n_test_days, 1)
        has_signal = prob_std >= 0.05

        results.append(HorizonResult(
            horizon_s=horizon_s,
            theta=theta,
            prob_std=prob_std,
            prob_min=prob_min,
            prob_max=prob_max,
            n_signals=n_sig,
            sig_per_day=sig_per_day,
            win_rate=wr,
            expectancy_bps=expectancy_bps,
            top5_features=top5_features,
            has_signal=has_signal,
        ))

        print(f"  θ={theta:.2f}: signals={n_sig} ({sig_per_day:.1f}/day), "
              f"WR={wr:.4f}, expectancy={expectancy_bps:+.1f} bps")

    # Print top 5 features once per horizon
    print(f"  Top 5 features:")
    for rank, (fname, fimp) in enumerate(top5_features, 1):
        print(f"    {rank}. {fname:25s}  {fimp}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[HorizonResult]) -> None:
    """Print a summary table sorted by OOS expectancy descending."""
    print(f"\n{'=' * 100}")
    print(f"=== SUMMARY TABLE (sorted by OOS expectancy) ===")
    print(f"{'=' * 100}")
    print(f"{'Horizon':>8s} {'θ':>6s} {'Prob Std':>9s} {'Prob Range':>18s} {'Signals':>8s} "
          f"{'Sig/Day':>8s} {'Win Rate':>9s} {'Expectancy':>12s} {'Signal?':>8s}")
    print(f"{'-' * 100}")

    sorted_results = sorted(all_results, key=lambda r: r.expectancy_bps, reverse=True)

    for r in sorted_results:
        horizon_label = f"{r.horizon_s // 3600}h"
        prob_range = f"[{r.prob_min:.4f}, {r.prob_max:.4f}]"
        signal_str = "YES" if r.has_signal else "NO"
        print(f"{horizon_label:>8s} {r.theta:>6.2f} {r.prob_std:>9.4f} {prob_range:>18s} "
              f"{r.n_signals:>8d} {r.sig_per_day:>8.1f} {r.win_rate:>9.4f} "
              f"{r.expectancy_bps:>+10.1f} bps {signal_str:>8s}")

    print(f"{'-' * 100}")
    print(f"  Cost per trade: {COST_PER_TRADE * 10_000:.0f} bps")
    print(f"  min_child_samples: {MIN_CHILD_SAMPLES}")
    print(f"  num_leaves: {NUM_LEAVES}")

    # Verdict
    any_signal = any(r.has_signal for r in all_results)
    any_positive = any(r.expectancy_bps > 0 and r.has_signal for r in all_results)

    print(f"\n{'=' * 100}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 100}")

    if not any_signal:
        print(f"  NO SIGNAL — ALL horizons have prob std < 0.05")
        print(f"  The 10 features carry zero discriminative power across all horizons.")
        print(f"  Features are genuinely dead. Rethink approach.")
    elif any_positive:
        best = max((r for r in all_results if r.has_signal), key=lambda r: r.expectancy_bps)
        print(f"  POTENTIAL SIGNAL — best result at {best.horizon_s // 3600}h horizon, θ={best.theta:.2f}")
        print(f"  Expectancy: {best.expectancy_bps:+.1f} bps after {COST_PER_TRADE * 10_000:.0f} bps cost")
        print(f"  Win rate: {best.win_rate:.4f}, Signals: {best.n_signals} ({best.sig_per_day:.1f}/day)")
        print(f"  WARNING: Only 7 days of data. Need 60+ days to trust.")
    else:
        print(f"  SIGNAL EXISTS but NO EDGE — model discriminates (std > 0.05)")
        print(f"  but expectancy is negative after {COST_PER_TRADE * 10_000:.0f} bps cost.")
        print(f"  Need stronger features or lower cost to clear fees.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_extraction = "--skip-extraction" in sys.argv

    print("=" * 100)
    print("=== OFP v2b TRAINING (multi-horizon, fixed min_child_samples=50, 15 bps cost) ===")
    print("=" * 100)
    print(f"  Horizons: {[f'{h // 3600}h' for h in HORIZONS]}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  min_child_samples: {MIN_CHILD_SAMPLES} (was 500)")
    print(f"  num_leaves:        {NUM_LEAVES}")
    print(f"  cost:              {COST_PER_TRADE * 10_000:.0f} bps")
    print(f"  split:             chronological 70/15/15")

    all_results: list[HorizonResult] = []

    for horizon_s in HORIZONS:
        mag_file = Path(f"data/research_dataset_v2_mag_{horizon_s}s.parquet")

        if not mag_file.exists():
            print(f"\n  *** {mag_file} not found — run relabel_magnitude with --horizon {horizon_s} first ***")
            continue

        # Load features
        df = load_or_build_features(mag_file, skip_extraction=skip_extraction)

        print(f"\n  Dataset: {len(df):,} rows, {len(df.columns)} cols")
        print(f"  Label distribution:")
        for label_val, count in df.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL).iter_rows():
            print(f"    Label {label_val}: {count:,}")

        # Train
        results = train_single_horizon(df, horizon_s)
        all_results.extend(results)

    if all_results:
        print_summary_table(all_results)
    else:
        print("\nNo results — no horizon datasets found.")


if __name__ == "__main__":
    main()
