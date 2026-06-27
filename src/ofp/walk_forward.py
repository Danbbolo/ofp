"""
walk_forward.py — Walk-forward splitter with purge gap.

Expanding window: train set grows each fold.
  Fold 1: Train days 1-5, Test day 6
  Fold 2: Train days 1-6, Test day 7
  ...

A 10-day purge window is enforced between train end and test start.
If the purge window can't be satisfied (not enough data), the fold is skipped.

Usage:
    from src.ofp.walk_forward import WalkForwardSplitter

    splitter = WalkForwardSplitter(
        min_train_days=5,
        test_days=1,
        purge_days=10,
    )
    folds = splitter.split(timestamps_ms)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

MS_PER_DAY = 86_400_000


def _day_of(ms: int) -> int:
    """Convert ms timestamp to day number (days since epoch)."""
    return int(ms // MS_PER_DAY)


def _day_to_ms(day: int) -> int:
    """Convert day number to ms timestamp (start of day)."""
    return day * MS_PER_DAY


@dataclass
class Fold:
    """A single walk-forward fold."""
    fold_idx: int
    train_start_day: int
    train_end_day: int       # inclusive
    test_start_day: int      # inclusive
    test_end_day: int        # inclusive
    train_indices: np.ndarray
    test_indices: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Fold({self.fold_idx}: "
            f"train=day{self.train_start_day}..{self.train_end_day}, "
            f"test=day{self.test_start_day}..{self.test_end_day}, "
            f"train_n={len(self.train_indices)}, test_n={len(self.test_indices)})"
        )


class WalkForwardSplitter:
    """
    Walk-forward splitter with expanding window and purge gap.

    Parameters
    ----------
    min_train_days : int
        Minimum number of days in the first training window.
    test_days : int
        Number of days in each test window.
    purge_days : int
        Number of days between train end and test start (no data in this gap).
    """

    def __init__(
        self,
        min_train_days: int = 5,
        test_days: int = 1,
        purge_days: int = 10,
    ):
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.purge_days = purge_days

    def split(self, timestamps_ms: np.ndarray) -> List[Fold]:
        """
        Generate walk-forward folds.

        Parameters
        ----------
        timestamps_ms : np.ndarray[int64]
            Sorted timestamps in milliseconds.

        Returns
        -------
        List[Fold]
            List of folds with train/test indices.
        """
        if len(timestamps_ms) == 0:
            return []

        # Convert to day numbers
        days = np.array([_day_of(int(ts)) for ts in timestamps_ms])
        unique_days = sorted(set(days.tolist()))
        n_days = len(unique_days)

        if n_days < self.min_train_days + self.purge_days + self.test_days:
            return []

        folds: List[Fold] = []
        fold_idx = 0

        # Expanding window: train grows, test slides forward
        # Fold 0: train=days[0:min_train], purge, test=days[min_train+purge : min_train+purge+test]
        # Fold 1: train=days[0:min_train+1], purge, test=days[min_train+1+purge : ...]
        # ...
        test_start_idx = self.min_train_days + self.purge_days

        while test_start_idx + self.test_days <= n_days:
            # Expanding train: always from day 0 to test_start - purge - 1
            train_end_idx = test_start_idx - self.purge_days - 1

            if train_end_idx < self.min_train_days - 1:
                # Not enough train days
                test_start_idx += 1
                continue

            train_start_day = unique_days[0]
            train_end_day = unique_days[train_end_idx]
            test_start_day = unique_days[test_start_idx]
            test_end_idx = min(test_start_idx + self.test_days - 1, n_days - 1)
            test_end_day = unique_days[test_end_idx]

            # Get indices
            train_mask = (days >= train_start_day) & (days <= train_end_day)
            test_mask = (days >= test_start_day) & (days <= test_end_day)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) == 0 or len(test_indices) == 0:
                test_start_idx += 1
                continue

            folds.append(Fold(
                fold_idx=fold_idx,
                train_start_day=train_start_day,
                train_end_day=train_end_day,
                test_start_day=test_start_day,
                test_end_day=test_end_day,
                train_indices=train_indices,
                test_indices=test_indices,
            ))
            fold_idx += 1
            test_start_idx += 1

        return folds

    def get_fold_info(self, fold: Fold) -> dict:
        """Return human-readable fold info."""
        return {
            "fold": fold.fold_idx,
            "train_days": f"day{fold.train_start_day}..day{fold.train_end_day}",
            "test_days": f"day{fold.test_start_day}..day{fold.test_end_day}",
            "train_n": len(fold.train_indices),
            "test_n": len(fold.test_indices),
            "purge_days": self.purge_days,
        }