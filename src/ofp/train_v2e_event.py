"""
train_v2e_event.py — Event Prediction Training + OOS Evaluation (Task 6).

Pivots the ML target from directional prediction to volatility-event
prediction: does ANY 1% move (up or down) occur within a 2-hour window?

Pipeline:
  1. Relabel IS volume bars (2026-06-17 to 2026-06-23) with event target.
  2. Join with cached 15-feature dataset.
  3. Train LightGBM (binary, min_child_samples=50, num_leaves=31).
  4. Chronological 70/15/15 split. Report IS prob std, win rate, signals/day.
  5. Build OOS features (2026-06-24 to 2026-06-26), relabel with event target.
  6. Predict on OOS without retraining. Report OOS prob std + event hit rate.

Usage:
    python -m src.ofp.train_v2e_event
    python -m src.ofp.train_v2e_event --skip-is-extraction    # use cached IS features
    python -m src.ofp.train_v2e_event --skip-oos-extraction   # use cached OOS dataset
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
from src.ofp.relabel_event import relabel_bars_event, LABEL_EVENT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZON_S = 7200        # 2 hours
MOVE_BPS = 100          # 1%
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
COST_PER_TRADE = 0.0015   # 15 bps round-trip (for reference)
MIN_CHILD_SAMPLES = 50
NUM_LEAVES = 31
EARLY_STOPPING = 20
N_JOBS = 14

IS_RAW_DIR = Path("data/raw_futures")
OOS_RAW_DIR = Path("data/raw_futures_oos")
EVENT_FILE = Path(
    f"data/research_dataset_v2_event_{HORIZON_S}s_{MOVE_BPS}bps.parquet"
)
IS_FEATURE_CACHE = Path("data/research_dataset_v2c_15features.parquet")
IS_DATASET_CACHE = Path(
    f"data/research_dataset_v2e_is_{HORIZON_S}s_{MOVE_BPS}bps.parquet"
)
OOS_DATASET_CACHE = Path(
    f"data/research_dataset_v2e_oos_{HORIZON_S}s_{MOVE_BPS}bps.parquet"
)
MODEL_FILE = Path("data/model_v2e_event.txt")

HORIZON_MS = HORIZON_S * 1000

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
    "vpin", "ofi", "book_delta", "trade_arrival_rate", "liq_volume",
    "vpin_x_arrival", "ofi_x_book_delta", "liq_x_vpin",
    "vpin_x_duration", "liq_x_return",
    "cvd_momentum", "wall_lifecycle", "volume_profile_entropy",
    "large_trade_count", "macro_trade_size_skew",
]

LABEL_COL = "label"
TIMESTAMP_COL = "timestamp_ms"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    theta: float
    n_signals: int
    sig_per_day: float
    event_hit_rate: float
    lift_over_base: float


@dataclass
class EvalResult:
    label: str
    prob_std: float
    prob_min: float
    prob_max: float
    base_rate: float
    n_rows: int
    n_days: int
    threshold_results: list[ThresholdResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IS dataset
# ---------------------------------------------------------------------------

def load_is_dataset(skip_extraction: bool = False) -> pl.DataFrame:
    """Load IS features + event labels."""
    if skip_extraction and IS_DATASET_CACHE.exists():
        print(f"Loading cached IS event dataset from {IS_DATASET_CACHE} ...")
        return pl.read_parquet(str(IS_DATASET_CACHE))

    print("=== BUILDING IS EVENT DATASET (15 features) ===")

    # Load cached 15-feature dataset (from Task 4c)
    if not IS_FEATURE_CACHE.exists():
        print(f"ERROR: IS feature cache not found: {IS_FEATURE_CACHE}")
        print("Run Task 4c first to generate features.")
        sys.exit(1)

    features = pl.read_parquet(str(IS_FEATURE_CACHE))
    print(f"  Cached features: {len(features):,} rows")

    # Build volume bars and relabel with event target
    from ofp.volume_clock import build_volume_bars

    ts = features[TIMESTAMP_COL].to_numpy()
    days = sorted(set(_day_of(int(t)) for t in ts))
    date_strs = [
        datetime.utcfromtimestamp(d * 86400000 / 1000).strftime("%Y-%m-%d")
        for d in days
    ]
    print(f"  IS dates: {date_strs}")

    all_bars = []
    for date_str in date_strs:
        raw = IS_RAW_DIR / date_str
        trades_path = raw / "trades.parquet"
        if not trades_path.exists():
            print(f"  {date_str}: no raw data, skipping")
            continue
        bars = build_volume_bars(trades_path, volume_threshold=50.0)
        all_bars.append(bars)

    bars = pl.concat(all_bars, how="vertical")
    print(f"  Total IS bars: {len(bars):,}")

    # Relabel with event target
    print(f"  Relabeling with event target (+/-{MOVE_BPS}bps, {HORIZON_S}s) ...", flush=True)
    bars_labeled = relabel_bars_event(
        bars, raw_dir=IS_RAW_DIR,
        move_bps=MOVE_BPS, max_horizon_ms=HORIZON_MS,
    )

    # Print label distribution
    labels = bars_labeled[LABEL_COL].to_numpy()
    n = len(bars_labeled)
    n_event = int((labels == LABEL_EVENT).sum())
    base_rate = n_event / n if n > 0 else 0.0
    print(f"\n  IS EVENT LABEL DISTRIBUTION:")
    print(f"    Label 1 (Event):    {n_event:>8,}  ({n_event/n*100:.2f}%)")
    print(f"    Label 0 (No Event): {n - n_event:>8,}  ({(n - n_event)/n*100:.2f}%)")
    print(f"    Base rate (P(event)): {base_rate:.4f} ({base_rate*100:.2f}%)")

    # Join features + labels
    dataset = features.join(
        bars_labeled.select([TIMESTAMP_COL, LABEL_COL]),
        on=TIMESTAMP_COL,
        how="inner",
    )
    print(f"  Joined IS dataset: {len(dataset):,} rows")

    dataset.write_parquet(str(IS_DATASET_CACHE))
    print(f"  Saved to {IS_DATASET_CACHE}")
    return dataset


# ---------------------------------------------------------------------------
# OOS dataset
# ---------------------------------------------------------------------------

def build_oos_dataset() -> pl.DataFrame:
    """Build OOS features + event labels from data/raw_futures_oos/."""
    from ofp.feature_extractor_v2 import extract_features
    from ofp.volume_clock import build_volume_bars

    print("=== BUILDING OOS EVENT DATASET (15 features) ===")

    if not OOS_RAW_DIR.exists():
        print(f"ERROR: {OOS_RAW_DIR} not found")
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
        print("ERROR: no OOS bars generated")
        sys.exit(1)

    features = pl.concat(all_features, how="vertical")
    bars = pl.concat(all_bars, how="vertical")
    print(f"\n  Total OOS bars: {len(bars):,}")
    print(f"  Total OOS features: {len(features):,}")

    # Relabel with event target
    print(
        f"\n  Relabeling OOS with event target "
        f"(+/-{MOVE_BPS}bps, {HORIZON_S}s) ...",
        flush=True,
    )
    bars_labeled = relabel_bars_event(
        bars, raw_dir=OOS_RAW_DIR,
        move_bps=MOVE_BPS, max_horizon_ms=HORIZON_MS,
    )

    labels = bars_labeled[LABEL_COL].to_numpy()
    n = len(bars_labeled)
    n_event = int((labels == LABEL_EVENT).sum())
    base_rate = n_event / n if n > 0 else 0.0
    print(f"\n  OOS EVENT LABEL DISTRIBUTION:")
    print(f"    Label 1 (Event):    {n_event:>8,}  ({n_event/n*100:.2f}%)")
    print(f"    Label 0 (No Event): {n - n_event:>8,}  ({(n - n_event)/n*100:.2f}%)")
    print(f"    Base rate (P(event)): {base_rate:.4f} ({base_rate*100:.2f}%)")

    dataset = features.join(
        bars_labeled.select([TIMESTAMP_COL, LABEL_COL]),
        on=TIMESTAMP_COL,
        how="inner",
    )
    print(f"  Joined OOS dataset: {len(dataset):,} rows")

    dataset.write_parquet(str(OOS_DATASET_CACHE))
    print(f"  Saved to {OOS_DATASET_CACHE}")
    return dataset


def load_or_build_oos(skip_extraction: bool = False) -> pl.DataFrame:
    if skip_extraction and OOS_DATASET_CACHE.exists():
        print(f"Loading cached OOS dataset from {OOS_DATASET_CACHE} ...")
        return pl.read_parquet(str(OOS_DATASET_CACHE))
    return build_oos_dataset()


# ---------------------------------------------------------------------------
# NaN/Inf audit
# ---------------------------------------------------------------------------

def audit_nan_inf(df: pl.DataFrame, label: str) -> bool:
    print(f"\n=== NaN/Inf AUDIT ({label}) ===")
    total = 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  {col:30s}: MISSING!")
            total += 1
            continue
        s = df[col]
        n_nan = int(s.is_nan().sum())
        n_inf = int(s.is_infinite().sum())
        status = "OK" if n_nan == 0 and n_inf == 0 else "*** PROBLEM ***"
        print(f"  {col:30s}: NaN={n_nan}, Inf={n_inf}  {status}")
        total += n_nan + n_inf
    if total == 0:
        print(f"  ALL CLEAN -- 0 NaN, 0 Inf across {len(FEATURE_COLS)} features")
    else:
        print(f"  *** {total} total NaN/Inf issues ***")
    return total == 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train, y_train, X_val, y_val, params, save_path=None
) -> lgb.LGBMClassifier:
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    label_str: str,
) -> EvalResult:
    """Predict and report event-prediction metrics at each threshold."""
    probs = model.predict_proba(X)[:, 1]
    prob_std = float(np.std(probs))
    prob_min = float(probs.min())
    prob_max = float(probs.max())
    base_rate = float(y.mean())
    n = len(X)
    n_days = len(set(_day_of(int(t)) for t in ts))

    imp = model.booster_.feature_importance(importance_type="split")
    top5_idx = np.argsort(imp)[::-1][:5]

    print(f"\n=== RESULTS ({label_str}) ===")
    print(f"  Rows: {n:,}")
    print(f"  Days: {n_days}")
    print(f"  Base rate (P(event)): {base_rate:.4f} ({base_rate*100:.2f}%)")
    print(f"  Prob range: [{prob_min:.4f}, {prob_max:.4f}]")
    print(f"  Prob std: {prob_std:.4f}")
    if prob_std < 0.05:
        print(f"  *** NO EVENT SIGNAL (prob std < 0.05) ***")
    else:
        print(f"  *** PROB STD > 0.05 — model is using features ***")

    print(f"\n  TOP 5 FEATURES:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. {FEATURE_COLS[idx]:30s}  {imp[idx]:.0f}")

    print(f"\n  THRESHOLD RESULTS:")
    print(
        f"    {'theta':>6s} {'signals':>8s} {'sig/day':>8s} "
        f"{'hit_rate':>9s} {'lift':>8s}"
    )
    print(f"    {'-' * 48}")

    threshold_results = []
    for theta in THRESHOLDS:
        signals = probs >= theta
        n_sig = int(signals.sum())
        if n_sig > 0:
            hit_rate = float(y[signals].mean())
            lift = hit_rate - base_rate
        else:
            hit_rate = 0.0
            lift = 0.0
        sig_per_day = n_sig / max(n_days, 1)

        threshold_results.append(ThresholdResult(
            theta=theta,
            n_signals=n_sig,
            sig_per_day=sig_per_day,
            event_hit_rate=hit_rate,
            lift_over_base=lift,
        ))

        print(
            f"    {theta:>6.2f} {n_sig:>8d} {sig_per_day:>8.1f} "
            f"{hit_rate:>9.4f} {lift:>+8.4f}"
        )

    return EvalResult(
        label=label_str,
        prob_std=prob_std,
        prob_min=prob_min,
        prob_max=prob_max,
        base_rate=base_rate,
        n_rows=n,
        n_days=n_days,
        threshold_results=threshold_results,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(is_result: EvalResult, oos_result: EvalResult) -> None:
    print(f"\n{'=' * 100}")
    print(
        f"=== FINAL SUMMARY: EVENT PREDICTION "
        f"({HORIZON_S/3600:.0f}h, +/-{MOVE_BPS}bps, 15 features) ==="
    )
    print(f"{'=' * 100}")

    for tag, r in [("IS", is_result), ("OOS", oos_result)]:
        print(f"\n  --- {tag} ({r.label}) ---")
        print(
            f"    Prob std: {r.prob_std:.4f}  "
            f"range: [{r.prob_min:.4f}, {r.prob_max:.4f}]"
        )
        print(
            f"    Base rate: {r.base_rate:.4f}  "
            f"rows: {r.n_rows:,}  days: {r.n_days}"
        )
        print(
            f"    {'theta':>6s} {'signals':>8s} {'sig/day':>8s} "
            f"{'hit_rate':>9s} {'lift':>8s}"
        )
        for tr in r.threshold_results:
            print(
                f"    {tr.theta:>6.2f} {tr.n_signals:>8d} {tr.sig_per_day:>8.1f} "
                f"{tr.event_hit_rate:>9.4f} {tr.lift_over_base:>+8.4f}"
            )

    # Verdict
    print(f"\n{'=' * 100}")
    print(f"=== VERDICT ===")
    print(f"{'=' * 100}")

    if oos_result.prob_std >= 0.05:
        # Check if hit rate lifts above base at any threshold
        best_lift = max(tr.lift_over_base for tr in oos_result.threshold_results)
        if best_lift > 0.02:
            print(f"  EVENT SIGNAL CONFIRMED")
            print(f"  OOS prob std = {oos_result.prob_std:.4f} (>= 0.05)")
            print(f"  Best lift over base: {best_lift:+.4f}")
            print(
                f"  WARNING: Only 3 OOS days. Need 60+ days to trust. "
                f"Also must clear {COST_PER_TRADE*10_000:.0f} bps fee."
            )
        else:
            print(f"  WEAK EVENT SIGNAL")
            print(f"  OOS prob std = {oos_result.prob_std:.4f} (>= 0.05)")
            print(
                f"  But lift over base is only {best_lift:+.4f} "
                f"(need >0.02 for meaningful separation)"
            )
    else:
        print(f"  NO EVENT SIGNAL")
        print(f"  OOS prob std = {oos_result.prob_std:.4f} (< 0.05)")
        print(
            f"  15 microstructure features cannot predict volatility "
            f"ignition at {HORIZON_S/3600:.0f}h / +/-{MOVE_BPS}bps."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_is = "--skip-is-extraction" in sys.argv
    skip_oos = "--skip-oos-extraction" in sys.argv

    print("=" * 100)
    print(f"=== TASK 6: EVENT PREDICTION PIVOT (VOLATILITY BREAKOUT) ===")
    print(f"    Horizon: {HORIZON_S/3600:.1f}h  Move: +/-{MOVE_BPS} bps")
    print(f"    IS:  {IS_RAW_DIR} (2026-06-17 to 2026-06-23)")
    print(f"    OOS: {OOS_RAW_DIR} (2026-06-24 to 2026-06-26)")
    print(f"    Features: {len(FEATURE_COLS)}  Cost: {COST_PER_TRADE*10_000:.0f} bps")
    print(f"    Thresholds: {THRESHOLDS}")
    print("=" * 100)

    # --- IS data ---
    print(f"\n[1/4] Loading IS data ...")
    is_df = load_is_dataset(skip_extraction=skip_is)
    audit_nan_inf(is_df, "IS")

    is_ts = is_df[TIMESTAMP_COL].to_numpy()
    is_y = is_df[LABEL_COL].to_numpy().astype(np.int8)
    is_X = is_df.select(FEATURE_COLS).to_numpy()

    n_is = len(is_X)
    split1 = int(n_is * 0.70)
    split2 = int(n_is * 0.85)

    X_train, y_train = is_X[:split1], is_y[:split1]
    X_val, y_val = is_X[split1:split2], is_y[split1:split2]
    X_test, y_test = is_X[split2:], is_y[split2:]
    ts_test = is_ts[split2:]

    base_rate_train = float(y_train.mean())
    print(f"\n=== IS TRAIN DATA ===")
    print(f"  Train: {len(X_train):,} rows (70%)")
    print(f"  Val:   {len(X_val):,} rows (15%)")
    print(f"  Test:  {len(X_test):,} rows (15%)")
    print(f"  Train base rate: {base_rate_train:.4f} ({base_rate_train*100:.2f}%)")

    # --- Train ---
    print(f"\n[2/4] Training LightGBM ...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        LGB_PARAMS, save_path=MODEL_FILE,
    )

    # --- IS test evaluation ---
    print(f"\n[3/4] IS test evaluation ...")
    is_result = evaluate(model, X_test, y_test, ts_test, "IS test (15% holdout)")

    # --- OOS data ---
    print(f"\n[4/4] Loading OOS data ...")
    oos_df = load_or_build_oos(skip_extraction=skip_oos)
    audit_nan_inf(oos_df, "OOS")

    oos_ts = oos_df[TIMESTAMP_COL].to_numpy()
    oos_y = oos_df[LABEL_COL].to_numpy().astype(np.int8)
    oos_X = oos_df.select(FEATURE_COLS).to_numpy()

    oos_result = evaluate(model, oos_X, oos_y, oos_ts, "OOS (3-day, no retrain)")

    # --- Summary ---
    print_summary(is_result, oos_result)


if __name__ == "__main__":
    main()