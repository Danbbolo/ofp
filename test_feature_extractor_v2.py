"""
test_feature_extractor_v2.py — Tests for the microstructure feature extractor.

Tests:
  1. No NaN or Inf in any of the 10 features
  2. All feature values are finite numbers
  3. VPIN is between 0 and 1
  4. Print feature matrix for first 100 bars

Usage:
    python test_feature_extractor_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ofp.volume_clock import build_volume_bars
from ofp.feature_extractor_v2 import extract_features


DATE = "2026-06-17"
THRESHOLD = 50.0
RAW_DIR = f"data/raw_futures/{DATE}"
TRADES_PATH = f"{RAW_DIR}/trades.parquet"


def get_features():
    """Build volume bars and extract features (cached at module level)."""
    if not hasattr(get_features, "_cache"):
        print(f"  Building volume bars ({THRESHOLD} BTC threshold) …")
        bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
        print(f"  {len(bars)} bars")
        print(f"  Extracting features …")
        get_features._cache = extract_features(bars, RAW_DIR)
    return get_features._cache


def test_no_nan_inf():
    """No NaN or Inf in any of the 10 features."""
    features = get_features()
    feature_cols = [c for c in features.columns if c != "timestamp_ms"]
    for col in feature_cols:
        s = features[col]
        n_nan = int(s.is_nan().sum())
        n_inf = int(s.is_infinite().sum())
        assert n_nan == 0, f"{col}: {n_nan} NaN values"
        assert n_inf == 0, f"{col}: {n_inf} Inf values"
    print(f"  PASS: No NaN or Inf in any of {len(feature_cols)} features")


def test_all_finite():
    """All feature values are finite numbers."""
    features = get_features()
    feature_cols = [c for c in features.columns if c != "timestamp_ms"]
    for col in feature_cols:
        s = features[col]
        # Check all values are finite (not NaN, not Inf)
        arr = s.to_numpy()
        assert np.all(np.isfinite(arr)), f"{col}: non-finite values found"
    print(f"  PASS: All {len(feature_cols)} features are finite")


def test_vpin_range():
    """VPIN must be between 0 and 1 (it's a probability)."""
    features = get_features()
    vpin = features["vpin"].to_numpy()
    assert vpin.min() >= 0.0, f"VPIN min < 0: {vpin.min()}"
    assert vpin.max() <= 1.0, f"VPIN max > 1: {vpin.max()}"
    print(f"  PASS: VPIN in [0, 1] (min={vpin.min():.6f}, max={vpin.max():.6f})")


def test_feature_count():
    """Must have exactly 10 feature columns + timestamp_ms."""
    features = get_features()
    feature_cols = [c for c in features.columns if c != "timestamp_ms"]
    assert len(feature_cols) == 10, f"Expected 10 features, got {len(feature_cols)}: {feature_cols}"
    print(f"  PASS: {len(feature_cols)} feature columns present")


def test_row_count():
    """Feature matrix must have same row count as volume bars."""
    bars = build_volume_bars(TRADES_PATH, volume_threshold=THRESHOLD)
    features = get_features()
    assert len(features) == len(bars), f"Row mismatch: bars={len(bars)}, features={len(features)}"
    print(f"  PASS: {len(features)} rows (matches volume bars)")


def main():
    print("=" * 60)
    print("FEATURE EXTRACTOR v2 TESTS")
    print("=" * 60)
    print(f"  Date:      {DATE}")
    print(f"  Threshold: {THRESHOLD} BTC")
    print()

    tests = [
        ("Feature count = 10", test_feature_count),
        ("Row count matches bars", test_row_count),
        ("No NaN or Inf", test_no_nan_inf),
        ("All finite", test_all_finite),
        ("VPIN in [0,1]", test_vpin_range),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    # Print first 100 bars
    print()
    print("=" * 60)
    print("FEATURE MATRIX (FIRST 100 BARS)")
    print("=" * 60)
    features = get_features()
    print(features.head(100))

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()