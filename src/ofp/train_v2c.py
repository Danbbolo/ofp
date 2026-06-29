"""
train_v2c.py — Training with 15 features at 1h horizon, ±50 bps labels.

Fixes from train_v2b:
  - 5 new features added (cvd_momentum, wall_lifecycle, volume_profile_entropy,
    large_trade_count, macro_trade_size_skew) → 15 total
  - Label threshold lowered to ±50 bps (0.5%) to triple positive sample size
  - 1h horizon only (that's where the signal was)

Pipeline:
  1. Load ±50 bps / 1h magnitude-labeled dataset
  2. Extract 15 v2 features per day (or load cached)
  3. Join features + labels on timestamp_ms
  4. Chronological 70/15/15 split
  5. Train LightGBM with min_child_samples=50
  6. Evaluate at 4 threshold levels
  7. Report OOS results + NaN/Inf audit

Usage:
    python -m src.ofp.train_v2c
    python -m src.ofp.train_v2c --skip-extraction  # use cached features
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

HORIZON_S = 3600       # 1h only
TARGET_BPS = 50        # ±0.5%
STOP_BPS = 50
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]

RAW_DIR = Path("data/raw_futures")
MAG_FILE = Path(f"data/research_dataset_v2_mag_{HORIZON_S}s_{TARGET_BPS}bps.parquet")
FEATURE_CACHE = Path("data/research_dataset_v2c_15features.parquet")

COST_PER_TRADE = 0.0015   # 15 bps round-trip
MIN_CHILD_SAMPLES = 50
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

# 15 features (10 original + 5 new)
FEATURE_COLS = [
    # Original 10
    "vpin", "ofi", "book_delta", "trade_arrival_rate", "liq_volume",
    "vpin_x_arrival", "ofi_x_book_delta", "liq_x_vpin",
    "vpin_x_duration", "liq_x_return",
    # New 5
    "cvd_momentum", "wall_lifecycle", "volume_profile_entropy",
    "large_trade_count", "macro_trade_size_skew",
]

LABEL_COL = "label"
TIMESTAMP_COL = "timestamp_ms"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    theta: float
    prob_std: float
    prob_min: float
    prob_max: float
    n_signals: int
    sig_per_day: float
    win_rate: float
    expectancy_bps: float
    has_signal: bool


# ---------------------------------------------------------------------------
# Feature extraction (or load cache)
# ---------------------------------------------------------------------------

def build_feature_dataset(mag_file: Path) -> pl.DataFrame:
    """
    Extract 15 v2 features for all dates in the magnitude dataset.
    Joins features with magnitude labels on timestamp_ms.
    """
    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING FEATURE DATASET (15 features) ===")

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
        print(f"    {len(feats)} bars, {len(feats.columns)} cols", flush=True)

    features = pl.concat(all_features, how="vertical")
    print(f"  Total feature rows: {len(features):,}")

    # Join with magnitude labels on timestamp_ms
    dataset = features.join(mag, on=TIMESTAMP_COL, how="inner")
    print(f"  Joined dataset: {len(dataset):,} rows")

    return dataset


def load_or_build_features(mag_file: Path, skip_extraction: bool = False) -> pl.DataFrame:
    """Load cached features or build from scratch."""
    if skip_extraction and FEATURE_CACHE.exists():
        print(f"Loading cached features from {FEATURE_CACHE} …")
        return pl.read_parquet(str(FEATURE_CACHE))

    dataset = build_feature_dataset(mag_file)

    # Save cache
    dataset.write_parquet(str(FEATURE_CACHE))
    print(f"  Saved to {FEATURE_CACHE}")
    return dataset


# ---------------------------------------------------------------------------
# NaN/Inf audit
# ---------------------------------------------------------------------------

def audit_nan_inf(df: pl.DataFrame) -> None:
    """Print NaN/Inf count per feature column."""
    print(f"\n=== NaN/Inf AUDIT ===")
    total_issues = 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  {col:30s}: MISSING COLUMN!")
            total_issues += 1
            continue
        s = df[col]
        n_nan = int(s.is_nan().sum())
        n_inf = int(s.is_infinite().sum())
        status = "OK" if n_nan == 0 and n_inf == 0 else "*** PROBLEM ***"
        print(f"  {col:30s}: NaN={n_nan}, Inf={n_inf}  {status}")
        total_issues += n_nan + n_inf

    if total_issues == 0:
        print(f"  ALL CLEAN — 0 NaN, 0 Inf across {len(FEATURE_COLS)} features")
    else:
        print(f"  *** {total_issues} total NaN/Inf issues ***")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_horizon(df: pl.DataFrame) -> list[ThresholdResult]:
    """
    Train LightGBM on the 1h / ±50bps dataset.
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

    print(f"\n=== TRAINING DATA (1h, ±{TARGET_BPS}bps) ===")
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

    print(f"\n=== CHRONOLOGICAL SPLIT (70/15/15) ===")
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

    n_test_days = len(set(_day_of(int(t)) for t in ts_test))

    print(f"\n=== OOS PROBABILITY STATS ===")
    print(f"  Prob range: [{prob_min:.4f}, {prob_max:.4f}], std={prob_std:.4f}")
    if prob_std < 0.05:
        print(f"  *** NO SIGNAL (prob std < 0.05) ***")

    print(f"\n=== TOP 5 FEATURES ===")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  {rank}. {FEATURE_COLS[idx]:30s}  {imp[idx]:.0f}")

    # Evaluate at each threshold
    print(f"\n=== OOS RESULTS (per threshold) ===")
    results = []
    for theta in THRESHOLDS:
        signals = probs >= theta
        n_sig = int(signals.sum())

        if n_sig > 0:
            wr = float(y_test[signals].mean())
            # ±50 bps target/stop → 50 bps win, 50 bps loss
            expectancy_bps = wr * TARGET_BPS - (1 - wr) * STOP_BPS - COST_PER_TRADE * 10_000
        else:
            wr = 0.0
            expectancy_bps = 0.0

        sig_per_day = n_sig / max(n_test_days, 1)
        has_signal = prob_std >= 0.05

        results.append(ThresholdResult(
            theta=theta,
            prob_std=prob_std,
            prob_min=prob_min,
            prob_max=prob_max,
            n_signals=n_sig,
            sig_per_day=sig_per_day,
            win_rate=wr,
            expectancy_bps=expectancy_bps,
            has_signal=has_signal,
        ))

        print(f"  θ={theta:.2f}: signals={n_sig} ({sig_per_day:.1f}/day), "
              f"WR={wr:.4f}, expectancy={expectancy_bps:+.1f} bps")

    return results, top5_idx, imp


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    results: list[ThresholdResult],
    top5_idx: np.ndarray,
    imp: np.ndarray,
    label_dist: dict,
) -> None:
    """Print summary table and verdict."""
    print(f"\n{'=' * 100}")
    print(f"=== SUMMARY TABLE (1h horizon, ±{TARGET_BPS}bps, 15 features) ===")
    print(f"{'=' * 100}")
    print(f"{'θ':>6s} {'Prob Std':>9s} {'Prob Range':>18s} {'Signals':>8s} "
          f"{'Sig/Day':>8s} {'Win Rate':>9s} {'Expectancy':>12s} {'Signal?':>8s}")
    print(f"{'-' * 100}")

    for r in results:
        prob_range = f"[{r.prob_min:.4f}, {r.prob_max:.4f}]"
        signal_str = "YES" if r.has_signal else "NO"
        print(f"{r.theta:>6.2f} {r.prob_std:>9.4f} {prob_range:>18s} "
              f"{r.n_signals:>8d} {r.sig_per_day:>8.1f} {r.win_rate:>9.4f} "
              f"{r.expectancy_bps:>+10.1f} bps {signal_str:>8s}")

    print(f"{'-' * 100}")
    print(f"  Cost per trade: {COST_PER_TRADE * 10_000:.0f} bps")
    print(f"  Target/Stop: ±{TARGET_BPS} bps")
    print(f"  min_child_samples: {MIN_CHILD_SAMPLES}")
    print(f"  num_leaves: {NUM_LEAVES}")

    # Top 5 features
    print(f"\n=== TOP 5 FEATURES (by split importance) ===")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  {rank}. {FEATURE_COLS[idx]:30s}  {imp[idx]:.0f}")

    # Verdict
    any_signal = any(r.has_signal for r in results)
    any_positive = any(r.expectancy_bps > 0 and r.has_signal for r in results)

    print(f"\n{'=' * 100}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 100}")

    if not any_signal:
        print(f"  NO SIGNAL — prob std < 0.05")
        print(f"  The 15 features carry zero discriminative power at 1h / ±{TARGET_BPS}bps.")
    elif any_positive:
        best = max((r for r in results if r.has_signal), key=lambda r: r.expectancy_bps)
        print(f"  POTENTIAL EDGE — best at θ={best.theta:.2f}")
        print(f"  Expectancy: {best.expectancy_bps:+.1f} bps after {COST_PER_TRADE * 10_000:.0f} bps cost")
        print(f"  Win rate: {best.win_rate:.4f}, Signals: {best.n_signals} ({best.sig_per_day:.1f}/day)")
        print(f"  WARNING: Only 7 days of data. Need 60+ days to trust.")
    else:
        print(f"  SIGNAL EXISTS but NO EDGE — model discriminates (std > 0.05)")
        print(f"  but expectancy is negative after {COST_PER_TRADE * 10_000:.0f} bps cost.")
        # Check if win rate is below base rate (inverted signal)
        best_wr = max(r.win_rate for r in results if r.n_signals > 0)
        if best_wr < 0.40:
            print(f"  Win rate ({best_wr:.4f}) is well below 50% — signal may be INVERTED.")
            print(f"  Consider flipping predictions or investigating label/feature alignment.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_extraction = "--skip-extraction" in sys.argv

    print("=" * 100)
    print(f"=== OFP v2c TRAINING (1h, ±{TARGET_BPS}bps, 15 features, min_child_samples=50) ===")
    print("=" * 100)
    print(f"  Horizon:    {HORIZON_S}s ({HORIZON_S / 3600:.0f}h)")
    print(f"  Target/Stop: ±{TARGET_BPS} bps (±{TARGET_BPS/100:.2f}%)")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Features:   {len(FEATURE_COLS)}")
    print(f"  Cost:       {COST_PER_TRADE * 10_000:.0f} bps")
    print(f"  Split:      chronological 70/15/15")

    if not MAG_FILE.exists():
        print(f"\n  *** {MAG_FILE} not found ***")
        print(f"  Run: python -m ofp.relabel_magnitude 2026-06-17 2026-06-23 --threshold 50 --horizon {HORIZON_S} --target_bps {TARGET_BPS}")
        sys.exit(1)

    # Load features
    df = load_or_build_features(MAG_FILE, skip_extraction=skip_extraction)

    print(f"\n=== DATASET ===")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Feature matrix: {df.shape[0]:,} × {len(FEATURE_COLS)} features")

    # Label distribution
    print(f"\n=== LABEL DISTRIBUTION (±{TARGET_BPS}bps, {HORIZON_S}s horizon) ===")
    label_dist = {}
    for label_val, count in df.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL).iter_rows():
        label_dist[int(label_val)] = int(count)
        pct = count / len(df) * 100
        print(f"  Label {label_val}: {count:,} ({pct:.2f}%)")

    n_target = label_dist.get(1, 0)
    n_stop = label_dist.get(0, 0)
    n_no_trade = label_dist.get(2, 0)
    tradeable = n_target + n_stop
    if tradeable > 0:
        base_rate = n_target / tradeable
        print(f"\n  Tradeable: {tradeable:,} ({tradeable / len(df) * 100:.2f}%)")
        print(f"  Base rate (Label 1 / (Label 0 + Label 1)): {base_rate:.4f} ({base_rate * 100:.2f}%)")

    # NaN/Inf audit
    audit_nan_inf(df)

    # Train
    results, top5_idx, imp = train_single_horizon(df)

    # Summary
    print_summary(results, top5_idx, imp, label_dist)


if __name__ == "__main__":
    main()
