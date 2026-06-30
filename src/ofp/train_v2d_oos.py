"""
train_v2d_oos.py — True OOS Validation + Probability Calibration (Task 5).

Step 1: Reproduce Task 4c model (15 features, 1h, ±50bps), predict on
        true OOS data (2026-06-24 to 2026-06-26) from data/raw_futures_oos/.
Step 2: Train calibrated models (is_unbalance=True, scale_pos_weight),
        predict on same OOS.

The IS model is reproduced by training on the same 70% train / 15% val
chronological split used in Task 4c (seed=42, same params).  The OOS data
was never seen during training.

Usage:
    python -m src.ofp.train_v2d_oos
    python -m src.ofp.train_v2d_oos --skip-is-extraction    # use cached IS features
    python -m src.ofp.train_v2d_oos --skip-oos-extraction   # use cached OOS dataset
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import lightgbm as lgb

from src.ofp.walk_forward import _day_of
from ofp.relabel_magnitude import relabel_bars, LABEL_TARGET, LABEL_STOP, LABEL_NO_TRADE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON_S = 3600       # 1h
TARGET_BPS = 50        # ±0.5%
STOP_BPS = 50
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
COST_PER_TRADE = 0.0015   # 15 bps round-trip
MIN_CHILD_SAMPLES = 50
NUM_LEAVES = 31
EARLY_STOPPING = 20
N_JOBS = 14

IS_RAW_DIR = Path("data/raw_futures")
OOS_RAW_DIR = Path("data/raw_futures_oos")
MAG_FILE = Path(f"data/research_dataset_v2_mag_{HORIZON_S}s_{TARGET_BPS}bps.parquet")
IS_FEATURE_CACHE = Path("data/research_dataset_v2c_15features.parquet")
OOS_DATASET_CACHE = Path(
    f"data/research_dataset_v2d_oos_{HORIZON_S}s_{TARGET_BPS}bps.parquet"
)
MODEL_STD_FILE = Path("data/model_v2d_std.txt")
MODEL_CAL_FILE = Path("data/model_v2d_cal.txt")
MODEL_SPW_FILE = Path("data/model_v2d_spw.txt")

HORIZON_MS = HORIZON_S * 1000

LGB_PARAMS_BASE = {
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
    n_signals: int
    sig_per_day: float
    win_rate: float
    expectancy_bps: float


@dataclass
class OOSResult:
    label: str
    prob_std: float
    prob_min: float
    prob_max: float
    base_rate: float
    n_tradeable: int
    n_days: int
    threshold_results: list[ThresholdResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IS feature loading
# ---------------------------------------------------------------------------

def load_is_dataset(skip_extraction: bool = False) -> pl.DataFrame:
    """Load IS features (from Task 4c cache or rebuild from scratch)."""
    if skip_extraction and IS_FEATURE_CACHE.exists():
        print(f"Loading cached IS features from {IS_FEATURE_CACHE} ...")
        return pl.read_parquet(str(IS_FEATURE_CACHE))

    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING IS FEATURE DATASET (15 features) ===")
    mag = pl.read_parquet(str(MAG_FILE))
    print(f"  Magnitude labels: {len(mag):,} rows")

    ts = mag[TIMESTAMP_COL].to_numpy()
    days = sorted(set(_day_of(int(t)) for t in ts))
    date_strs = [
        datetime.utcfromtimestamp(d * 86400000 / 1000).strftime("%Y-%m-%d")
        for d in days
    ]
    print(f"  IS dates: {date_strs}")

    all_features = []
    for date_str in date_strs:
        raw = IS_RAW_DIR / date_str
        trades_path = raw / "trades.parquet"
        if not trades_path.exists():
            print(f"  {date_str}: no raw data, skipping")
            continue
        print(f"  Extracting features for {date_str} ...", flush=True)
        bars = build_volume_bars(trades_path, volume_threshold=50.0)
        feats = extract_features(bars, raw)
        all_features.append(feats)
        print(f"    {len(feats)} bars", flush=True)

    features = pl.concat(all_features, how="vertical")
    dataset = features.join(mag, on=TIMESTAMP_COL, how="inner")
    print(f"  Joined IS dataset: {len(dataset):,} rows")

    dataset.write_parquet(str(IS_FEATURE_CACHE))
    print(f"  Saved to {IS_FEATURE_CACHE}")
    return dataset


# ---------------------------------------------------------------------------
# OOS dataset building
# ---------------------------------------------------------------------------

def build_oos_dataset() -> pl.DataFrame:
    """
    Build OOS features + labels from data/raw_futures_oos/.

    Pipeline per OOS date:
      1. Build volume bars (50 BTC threshold)
      2. Extract 15 features
    Then relabel all OOS bars together (±50bps, 1h horizon) and join
    features + labels on timestamp_ms.
    """
    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING OOS DATASET (15 features) ===")

    if not OOS_RAW_DIR.exists():
        print(f"  ERROR: {OOS_RAW_DIR} not found")
        sys.exit(1)

    oos_dates = sorted(p.name for p in OOS_RAW_DIR.iterdir() if p.is_dir())
    print(f"  OOS dates: {oos_dates}")

    all_features = []
    all_bars = []
    for date_str in oos_dates:
        raw = OOS_RAW_DIR / date_str
        trades_path = raw / "trades.parquet"
        if not trades_path.exists():
            print(f"  {date_str}: no trades file, skipping")
            continue

        print(f"\n  --- {date_str} ---", flush=True)
        print(f"  Building volume bars ...", flush=True)
        bars = build_volume_bars(trades_path, volume_threshold=50.0)
        print(f"    {len(bars)} bars", flush=True)
        all_bars.append(bars)

        print(f"  Extracting 15 features ...", flush=True)
        feats = extract_features(bars, raw)
        all_features.append(feats)
        print(f"    {len(feats)} feature rows", flush=True)

    if not all_bars:
        print("  ERROR: no OOS bars generated")
        sys.exit(1)

    features = pl.concat(all_features, how="vertical")
    bars = pl.concat(all_bars, how="vertical")
    print(f"\n  Total OOS bars: {len(bars):,}")
    print(f"  Total OOS features: {len(features):,}")

    # Relabel all OOS bars together (handles cross-day forward windows)
    print(f"\n  Relabeling OOS bars (+/-{TARGET_BPS}bps, {HORIZON_S}s horizon) ...", flush=True)
    bars_labeled = relabel_bars(
        bars,
        raw_dir=OOS_RAW_DIR,
        target_bps=TARGET_BPS,
        stop_bps=STOP_BPS,
        max_horizon_ms=HORIZON_MS,
    )

    # Print OOS label distribution
    labels = bars_labeled[LABEL_COL].to_numpy()
    n = len(bars_labeled)
    n_target = int((labels == LABEL_TARGET).sum())
    n_stop = int((labels == LABEL_STOP).sum())
    n_no_trade = int((labels == LABEL_NO_TRADE).sum())
    print(f"\n  OOS LABEL DISTRIBUTION:")
    print(f"    Label 1 (Target):  {n_target:>8,}  ({n_target / n * 100:.2f}%)")
    print(f"    Label 0 (Stop):    {n_stop:>8,}  ({n_stop / n * 100:.2f}%)")
    print(f"    Label 2 (No Trade): {n_no_trade:>8,}  ({n_no_trade / n * 100:.2f}%)")
    tradeable = n_target + n_stop
    if tradeable > 0:
        base_rate = n_target / tradeable
        print(f"    OOS base rate: {base_rate:.4f} ({base_rate * 100:.2f}%)")

    # Join features + labels on timestamp_ms
    dataset = features.join(
        bars_labeled.select([TIMESTAMP_COL, LABEL_COL]),
        on=TIMESTAMP_COL,
        how="inner",
    )
    print(f"  Joined OOS dataset: {len(dataset):,} rows")

    # Save cache
    dataset.write_parquet(str(OOS_DATASET_CACHE))
    print(f"  Saved to {OOS_DATASET_CACHE}")
    return dataset


def load_or_build_oos(skip_extraction: bool = False) -> pl.DataFrame:
    """Load cached OOS dataset or build from scratch."""
    if skip_extraction and OOS_DATASET_CACHE.exists():
        print(f"Loading cached OOS dataset from {OOS_DATASET_CACHE} ...")
        return pl.read_parquet(str(OOS_DATASET_CACHE))
    return build_oos_dataset()


# ---------------------------------------------------------------------------
# NaN/Inf audit
# ---------------------------------------------------------------------------

def audit_nan_inf(df: pl.DataFrame, label: str) -> bool:
    """Print NaN/Inf count per feature. Returns True if all clean."""
    print(f"\n=== NaN/Inf AUDIT ({label}) ===")
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
        print(f"  ALL CLEAN -- 0 NaN, 0 Inf across {len(FEATURE_COLS)} features")
    else:
        print(f"  *** {total_issues} total NaN/Inf issues ***")
    return total_issues == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    save_path: Path | None = None,
) -> lgb.LGBMClassifier:
    """Train LightGBM with given params, optional early-stopping val set."""
    model = lgb.LGBMClassifier(**params, n_estimators=2000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    if save_path:
        model.booster_.save_model(str(save_path))
        print(f"  Model saved to {save_path}")
    return model


# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------

def evaluate_oos(
    model: lgb.LGBMClassifier,
    X_oos: np.ndarray,
    y_oos: np.ndarray,
    ts_oos: np.ndarray,
    label_str: str,
) -> OOSResult:
    """Predict on OOS data and report metrics at each threshold."""
    probs = model.predict_proba(X_oos)[:, 1]
    prob_std = float(np.std(probs))
    prob_min = float(probs.min())
    prob_max = float(probs.max())
    base_rate = float(y_oos.mean())
    n = len(X_oos)
    n_days = len(set(_day_of(int(t)) for t in ts_oos))

    # Feature importance
    imp = model.booster_.feature_importance(importance_type="split")
    top5_idx = np.argsort(imp)[::-1][:5]

    print(f"\n=== OOS RESULTS ({label_str}) ===")
    print(f"  Tradeable rows: {n:,}")
    print(f"  Days: {n_days}")
    print(f"  Base rate: {base_rate:.4f} ({base_rate * 100:.2f}%)")
    print(f"  Prob range: [{prob_min:.4f}, {prob_max:.4f}]")
    print(f"  Prob std: {prob_std:.4f}")
    if prob_std < 0.05:
        print(f"  *** NO SIGNAL (prob std < 0.05) ***")

    print(f"\n  TOP 5 FEATURES:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. {FEATURE_COLS[idx]:30s}  {imp[idx]:.0f}")

    print(f"\n  THRESHOLD RESULTS:")
    print(f"    {'theta':>6s} {'signals':>8s} {'sig/day':>8s} "
          f"{'win_rate':>9s} {'expectancy':>12s}")
    print(f"    {'-' * 50}")

    threshold_results = []
    for theta in THRESHOLDS:
        signals = probs >= theta
        n_sig = int(signals.sum())
        if n_sig > 0:
            wr = float(y_oos[signals].mean())
            expectancy_bps = (
                wr * TARGET_BPS - (1 - wr) * STOP_BPS - COST_PER_TRADE * 10_000
            )
        else:
            wr = 0.0
            expectancy_bps = 0.0
        sig_per_day = n_sig / max(n_days, 1)

        threshold_results.append(ThresholdResult(
            theta=theta,
            n_signals=n_sig,
            sig_per_day=sig_per_day,
            win_rate=wr,
            expectancy_bps=expectancy_bps,
        ))

        print(f"    {theta:>6.2f} {n_sig:>8d} {sig_per_day:>8.1f} "
              f"{wr:>9.4f} {expectancy_bps:>+10.1f} bps")

    return OOSResult(
        label=label_str,
        prob_std=prob_std,
        prob_min=prob_min,
        prob_max=prob_max,
        base_rate=base_rate,
        n_tradeable=n,
        n_days=n_days,
        threshold_results=threshold_results,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[OOSResult]) -> None:
    """Print comparison table across all model variants."""
    print(f"\n{'=' * 100}")
    print(f"=== FINAL SUMMARY: OOS VALIDATION (1h, +/-{TARGET_BPS}bps, 15 features) ===")
    print(f"{'=' * 100}")

    for r in results:
        print(f"\n  --- {r.label} ---")
        print(f"    Prob std: {r.prob_std:.4f}  "
              f"range: [{r.prob_min:.4f}, {r.prob_max:.4f}]")
        print(f"    OOS base rate: {r.base_rate:.4f}  "
              f"tradeable rows: {r.n_tradeable:,}  days: {r.n_days}")
        print(f"    {'theta':>6s} {'signals':>8s} {'sig/day':>8s} "
              f"{'win_rate':>9s} {'expectancy':>12s}")
        for tr in r.threshold_results:
            print(f"    {tr.theta:>6.2f} {tr.n_signals:>8d} {tr.sig_per_day:>8.1f} "
                  f"{tr.win_rate:>9.4f} {tr.expectancy_bps:>+10.1f} bps")

    # Verdict
    print(f"\n{'=' * 100}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 100}")

    best_exp = -999.0
    best_label = ""
    best_theta = 0.0
    for r in results:
        for tr in r.threshold_results:
            if tr.expectancy_bps > best_exp:
                best_exp = tr.expectancy_bps
                best_label = r.label
                best_theta = tr.theta

    if best_exp > 0:
        print(f"  OOS EDGE CONFIRMED")
        print(f"  Best: {best_label} at theta={best_theta:.2f}")
        print(f"  Expectancy: {best_exp:+.1f} bps after {COST_PER_TRADE * 10_000:.0f} bps cost")
        print(f"  WARNING: Only 3 days of OOS data. Need 60+ days to trust.")
    else:
        print(f"  NO OOS EDGE")
        print(f"  All expectancies negative or zero across all model variants.")
        print(f"  Best (least negative): {best_label} at theta={best_theta:.2f} "
              f"= {best_exp:+.1f} bps")
        print(f"  The +18.5 bps seen in Task 4c IS test was an illusion")
        print(f"  (uptrending test period, base rate = 83.5%).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_is = "--skip-is-extraction" in sys.argv
    skip_oos = "--skip-oos-extraction" in sys.argv

    print("=" * 100)
    print(f"=== TASK 5: TRUE OOS VALIDATION + CALIBRATION ===")
    print(f"    1h horizon, +/-{TARGET_BPS}bps, 15 features")
    print(f"    IS: {IS_RAW_DIR} (2026-06-17 to 2026-06-23)")
    print(f"    OOS: {OOS_RAW_DIR} (2026-06-24 to 2026-06-26)")
    print(f"    Cost: {COST_PER_TRADE * 10_000:.0f} bps/trade")
    print(f"    Thresholds: {THRESHOLDS}")
    print("=" * 100)

    # --- Load IS data ---
    print(f"\n[1/4] Loading IS data ...")
    is_df = load_is_dataset(skip_extraction=skip_is)

    # Prepare IS arrays (exclude Label 2 = No Trade)
    is_ts = is_df[TIMESTAMP_COL].to_numpy()
    is_y = is_df[LABEL_COL].to_numpy()
    is_X = is_df.select(FEATURE_COLS).to_numpy()

    tradeable = is_y != 2
    is_X = is_X[tradeable]
    is_y = is_y[tradeable]
    is_ts = is_ts[tradeable]
    is_y = is_y.astype(np.int8)

    n_is = len(is_X)
    split1 = int(n_is * 0.70)
    split2 = int(n_is * 0.85)

    X_train, y_train = is_X[:split1], is_y[:split1]
    X_val, y_val = is_X[split1:split2], is_y[split1:split2]

    base_rate_train = float(y_train.mean())
    print(f"\n=== IS TRAIN DATA ===")
    print(f"  Train: {len(X_train):,} rows (70%)")
    print(f"  Val:   {len(X_val):,} rows (15%)")
    print(f"  Train base rate: {base_rate_train:.4f} ({base_rate_train * 100:.2f}%)")

    # --- Load OOS data ---
    print(f"\n[2/4] Loading OOS data ...")
    oos_df = load_or_build_oos(skip_extraction=skip_oos)

    # NaN/Inf audit on OOS
    audit_nan_inf(oos_df, "OOS")

    # Prepare OOS arrays (exclude Label 2)
    oos_ts = oos_df[TIMESTAMP_COL].to_numpy()
    oos_y = oos_df[LABEL_COL].to_numpy()
    oos_X = oos_df.select(FEATURE_COLS).to_numpy()

    tradeable_oos = oos_y != 2
    oos_X = oos_X[tradeable_oos]
    oos_y = oos_y[tradeable_oos]
    oos_ts = oos_ts[tradeable_oos]
    oos_y = oos_y.astype(np.int8)

    print(f"\n=== OOS DATA ===")
    print(f"  Tradeable rows: {len(oos_X):,}")
    print(f"  OOS base rate: {float(oos_y.mean()):.4f}")

    # --- Step 1: Standard model (reproduce Task 4c) ---
    print(f"\n[3/4] Step 1: Standard model (reproduce Task 4c) ...")
    print(f"{'=' * 100}")
    print(f"=== STEP 1: STANDARD MODEL ===")
    print(f"{'=' * 100}")
    model_std = train_model(
        X_train, y_train, X_val, y_val,
        LGB_PARAMS_BASE, save_path=MODEL_STD_FILE,
    )
    result_std = evaluate_oos(model_std, oos_X, oos_y, oos_ts, "Standard (Task 4c)")

    # --- Step 2: Calibrated models ---
    print(f"\n[4/4] Step 2: Calibrated models ...")
    print(f"{'=' * 100}")
    print(f"=== STEP 2a: CALIBRATED MODEL (is_unbalance=True) ===")
    print(f"{'=' * 100}")
    params_cal = {**LGB_PARAMS_BASE, "is_unbalance": True}
    model_cal = train_model(
        X_train, y_train, X_val, y_val,
        params_cal, save_path=MODEL_CAL_FILE,
    )
    result_cal = evaluate_oos(
        model_cal, oos_X, oos_y, oos_ts, "Calibrated (is_unbalance=True)"
    )

    print(f"\n{'=' * 100}")
    print(f"=== STEP 2b: CALIBRATED MODEL (scale_pos_weight) ===")
    print(f"{'=' * 100}")
    spw = 1.0 / base_rate_train
    print(f"  scale_pos_weight = 1 / base_rate = 1 / {base_rate_train:.4f} = {spw:.4f}")
    params_spw = {**LGB_PARAMS_BASE, "scale_pos_weight": spw}
    model_spw = train_model(
        X_train, y_train, X_val, y_val,
        params_spw, save_path=MODEL_SPW_FILE,
    )
    result_spw = evaluate_oos(
        model_spw, oos_X, oos_y, oos_ts,
        f"Calibrated (scale_pos_weight={spw:.2f})",
    )

    # --- Summary ---
    print_summary([result_std, result_cal, result_spw])


if __name__ == "__main__":
    main()