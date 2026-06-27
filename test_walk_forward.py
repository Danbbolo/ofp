"""
test_walk_forward.py — Tests for walk-forward splitter with purge gap.

Tests:
  1. No overlap between train and test in any fold.
  2. 10-day purge is enforced (no train data within 10 days of test start).
  3. Expanding window (train set grows each fold).
  4. Every bar appears in exactly one test set per fold.
"""
import numpy as np
import pytest

from src.ofp.walk_forward import WalkForwardSplitter, Fold, MS_PER_DAY, _day_of


def _make_timestamps(n_days: int, bars_per_day: int = 10, start_day: int = 1000) -> np.ndarray:
    """Generate n_days of timestamps with bars_per_day bars per day."""
    ts = []
    for d in range(n_days):
        for b in range(bars_per_day):
            ts.append((start_day + d) * MS_PER_DAY + b * 1000)
    return np.array(sorted(ts), dtype=np.int64)


# ---------------------------------------------------------------------------
# Test 1: No overlap between train and test in any fold
# ---------------------------------------------------------------------------

def test_no_train_test_overlap():
    """Train and test indices must not overlap in any fold."""
    ts = _make_timestamps(n_days=30, bars_per_day=10)
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)

    assert len(folds) > 0, "Expected at least one fold"

    for fold in folds:
        train_set = set(fold.train_indices.tolist())
        test_set = set(fold.test_indices.tolist())
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Fold {fold.fold_idx}: train/test overlap at indices {overlap}"
        )


# ---------------------------------------------------------------------------
# Test 2: 10-day purge is enforced
# ---------------------------------------------------------------------------

def test_purge_enforced():
    """No train data within 10 days of test start."""
    ts = _make_timestamps(n_days=30, bars_per_day=10)
    purge_days = 10
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=purge_days)
    folds = splitter.split(ts)

    assert len(folds) > 0, "Expected at least one fold"

    for fold in folds:
        # The last train day must be at least purge_days before test start day
        gap = fold.test_start_day - fold.train_end_day
        assert gap > purge_days, (
            f"Fold {fold.fold_idx}: gap={gap} but purge={purge_days}. "
            f"train_end=day{fold.train_end_day}, test_start=day{fold.test_start_day}"
        )


# ---------------------------------------------------------------------------
# Test 3: Expanding window (train set grows each fold)
# ---------------------------------------------------------------------------

def test_expanding_window():
    """Train set should grow (or stay same) each fold — never shrink."""
    ts = _make_timestamps(n_days=30, bars_per_day=10)
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)

    assert len(folds) >= 2, "Need at least 2 folds to test expanding"

    prev_train_n = 0
    for fold in folds:
        assert len(fold.train_indices) >= prev_train_n, (
            f"Fold {fold.fold_idx}: train_n={len(fold.train_indices)} "
            f"< prev={prev_train_n} — train set shrank!"
        )
        prev_train_n = len(fold.train_indices)


# ---------------------------------------------------------------------------
# Test 4: Every bar appears in exactly one test set per fold
# ---------------------------------------------------------------------------

def test_one_test_set_per_bar():
    """Each bar should appear in at most one test set across all folds."""
    ts = _make_timestamps(n_days=30, bars_per_day=10)
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)

    all_test_indices = []
    for fold in folds:
        all_test_indices.extend(fold.test_indices.tolist())

    # Check no duplicates
    assert len(all_test_indices) == len(set(all_test_indices)), (
        f"Duplicate test indices found — some bars in multiple test sets"
    )


# ---------------------------------------------------------------------------
# Test 5: Not enough data → no folds
# ---------------------------------------------------------------------------

def test_insufficient_data():
    """With only 15 days and 10-day purge, should get 0 folds (need 5+10+1=16)."""
    ts = _make_timestamps(n_days=15, bars_per_day=10)
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)
    assert len(folds) == 0, f"Expected 0 folds with 15 days, got {len(folds)}"


def test_sufficient_data():
    """With 20 days and 10-day purge, should get folds."""
    ts = _make_timestamps(n_days=20, bars_per_day=10)
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(ts)
    assert len(folds) > 0, f"Expected folds with 20 days, got 0"


# ---------------------------------------------------------------------------
# Test 6: Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    """Empty timestamps should return no folds."""
    splitter = WalkForwardSplitter(min_train_days=5, test_days=1, purge_days=10)
    folds = splitter.split(np.array([], dtype=np.int64))
    assert len(folds) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])