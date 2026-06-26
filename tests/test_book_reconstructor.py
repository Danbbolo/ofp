"""
test_book_reconstructor.py — Tests for the OrderBookReconstructor.

Covers: snapshot application, delta updates, zero-quantity removal,
sparse books (<20 levels), and 1-second bucketed iteration.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ofp.book_reconstructor import OrderBookReconstructor


# ---------------------------------------------------------------------------
# Helpers — build DataFrames matching BookSnapshotData columns
# ---------------------------------------------------------------------------

BOOK_COLUMNS = [
    "received_time", "event_time", "symbol", "event_type",
    "side", "price", "quantity",
]


def _make_book_df(rows: list[tuple]) -> pd.DataFrame:
    """Rows: (event_time_ns, event_type, side, price, quantity)."""
    data = []
    for tup in rows:
        rec = {
            "received_time": tup[0] + 1000,  # offset from event_time
            "event_time": tup[0],
            "symbol": "BTCUSDT",
            "event_type": tup[1],
            "side": tup[2],
            "price": float(tup[3]),
            "quantity": float(tup[4]),
        }
        data.append(rec)
    return pd.DataFrame(data, columns=BOOK_COLUMNS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSnapshot:
    """Processing a full snapshot yields correct top-20 bids and asks."""

    @staticmethod
    def _snapshot_df(n_bids: int = 25, n_asks: int = 25) -> pd.DataFrame:
        rows = []
        ts = 1_719_000_000_000_000_000
        for i in range(n_bids):
            rows.append((ts, "snapshot", "bid", 68500.0 - i * 10.0, 1.0 + i * 0.1))
        for i in range(n_asks):
            rows.append((ts, "snapshot", "ask", 68600.0 + i * 10.0, 1.0 + i * 0.1))
        return _make_book_df(rows)

    def test_top_20_bids_descending(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(self._snapshot_df(25, 5))
        bids, _ = recon.top_n(20)

        assert len(bids) == 20
        # Bids must be descending
        for i in range(len(bids) - 1):
            assert bids[i][0] > bids[i + 1][0]
        # Top bid = highest price = 68500.0
        assert bids[0][0] == 68500.0
        # 20th bid = 68500 - 19*10 = 68310
        assert bids[-1][0] == 68310.0

    def test_top_20_asks_ascending(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(self._snapshot_df(5, 25))
        _, asks = recon.top_n(20)

        assert len(asks) == 20
        # Asks must be ascending
        for i in range(len(asks) - 1):
            assert asks[i][0] < asks[i + 1][0]
        # Top ask = lowest price = 68600.0
        assert asks[0][0] == 68600.0
        # 20th ask = 68600 + 19*10 = 68790
        assert asks[-1][0] == 68790.0

    def test_snapshot_clears_previous_state(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(self._snapshot_df(5, 5))
        assert len(recon) == 10

        # Second snapshot with different data
        df2 = _make_book_df([
            (1_719_000_000_000_000_000, "snapshot", "bid", 70000.0, 5.0),
            (1_719_000_000_000_000_000, "snapshot", "ask", 70100.0, 3.0),
        ])
        recon.apply_snapshot(df2)
        assert len(recon) == 2
        bids, asks = recon.top_n(20)
        assert bids == [(70000.0, 5.0)]
        assert asks == [(70100.0, 3.0)]


class TestUpdates:
    """Delta updates modify existing levels correctly."""

    def test_update_existing_level_size(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(_make_book_df([
            (1, "snapshot", "bid", 68000.0, 1.0),
            (1, "snapshot", "ask", 68100.0, 2.0),
        ]))

        recon.apply_updates(_make_book_df([
            (2, "update", "bid", 68000.0, 3.5),   # size changed
            (2, "update", "ask", 68100.0, 0.8),   # size changed
        ]))

        bids, asks = recon.top_n(20)
        assert bids == [(68000.0, 3.5)]
        assert asks == [(68100.0, 0.8)]

    def test_update_adds_new_level(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(_make_book_df([
            (1, "snapshot", "bid", 68000.0, 1.0),
        ]))

        recon.apply_updates(_make_book_df([
            (2, "update", "bid", 67950.0, 2.0),  # new level below
            (2, "update", "bid", 68050.0, 3.0),  # new level above
        ]))

        bids, _ = recon.top_n(20)
        # Highest first: 68050, 68000, 67950
        assert bids == [(68050.0, 3.0), (68000.0, 1.0), (67950.0, 2.0)]


class TestZeroQuantityRemoval:
    """Zero-quantity updates remove the level; lower levels move up."""

    def test_removal_lets_lower_level_into_top_n(self) -> None:
        recon = OrderBookReconstructor()
        # Create 21 bids so the 21st is outside top-20
        rows = []
        for i in range(21):
            rows.append((1, "snapshot", "bid", 70000.0 - i * 10.0, 1.0))
        recon.apply_snapshot(_make_book_df(rows))

        bids_before, _ = recon.top_n(20)
        assert len(bids_before) == 20
        assert bids_before[0][0] == 70000.0       # highest
        assert bids_before[-1][0] == 69810.0       # 20th = 70000 - 19*10
        # The 21st level at 69800 is NOT in the top 20

        # Remove the top level
        recon.apply_updates(_make_book_df([
            (2, "update", "bid", 70000.0, 0.0),
        ]))

        bids_after, _ = recon.top_n(20)
        assert len(bids_after) == 20
        assert bids_after[0][0] == 69990.0          # was #2, now #1
        assert bids_after[-1][0] == 69800.0         # was #21, now #20

    def test_remove_only_level(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(_make_book_df([
            (1, "snapshot", "ask", 68100.0, 1.0),
        ]))

        recon.apply_updates(_make_book_df([
            (2, "update", "ask", 68100.0, 0.0),
        ]))

        _, asks = recon.top_n(20)
        assert asks == []

    def test_remove_nonexistent_level_no_error(self) -> None:
        """Removing a level not in the book should not raise."""
        recon = OrderBookReconstructor()
        recon.apply_updates(_make_book_df([
            (1, "update", "bid", 99999.0, 0.0),
        ]))
        bids, asks = recon.top_n(20)
        assert bids == []
        assert asks == []


class TestSparseBook:
    """Books with fewer than 20 levels return only what's available."""

    def test_fewer_than_20(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply_snapshot(_make_book_df([
            (1, "snapshot", "bid", 68000.0, 1.0),
            (1, "snapshot", "bid", 67900.0, 0.5),
            (1, "snapshot", "ask", 68100.0, 2.0),
        ]))

        bids, asks = recon.top_n(20)
        assert len(bids) == 2
        assert len(asks) == 1
        assert bids == [(68000.0, 1.0), (67900.0, 0.5)]
        assert asks == [(68100.0, 2.0)]

    def test_empty_book(self) -> None:
        recon = OrderBookReconstructor()
        bids, asks = recon.top_n(20)
        assert bids == []
        assert asks == []


class TestBucketedSnapshots:
    """1-second bucketed iteration yields one snapshot per bucket."""

    @staticmethod
    def _make_bucketed_df() -> pd.DataFrame:
        """
        Bucket 0 (ts=0): snapshot
        Bucket 1 (ts=1e9): two updates
        Bucket 2 (ts=2e9): one update
        """
        rows = [
            # Bucket 0 — snapshot
            (0,               "snapshot", "bid", 68500.0, 1.0),
            (0,               "snapshot", "ask", 68600.0, 2.0),
            # Bucket 1 — updates
            (1_000_000_000,   "update",   "bid", 68500.0, 3.0),
            (1_000_000_500,   "update",   "bid", 68400.0, 0.5),
            # Bucket 2 — update
            (2_000_000_000,   "update",   "ask", 68600.0, 0.0),
        ]
        return _make_book_df(rows)

    def test_three_buckets(self) -> None:
        recon = OrderBookReconstructor()
        df = self._make_bucketed_df()
        results = list(recon.iter_bucketed_snapshots(df, interval_ns=1_000_000_000, n=20))

        assert len(results) == 3

        # Bucket 0: snapshot applied
        ts0, bids0, asks0 = results[0]
        assert ts0 == 0
        assert bids0 == [(68500.0, 1.0)]
        assert asks0 == [(68600.0, 2.0)]

        # Bucket 1: two updates applied
        ts1, bids1, asks1 = results[1]
        assert ts1 == 1_000_000_000
        assert bids1 == [(68500.0, 3.0), (68400.0, 0.5)]
        assert asks1 == [(68600.0, 2.0)]

        # Bucket 2: ask removed
        ts2, bids2, asks2 = results[2]
        assert ts2 == 2_000_000_000
        assert bids2 == [(68500.0, 3.0), (68400.0, 0.5)]
        assert asks2 == []

    def test_empty_dataframe(self) -> None:
        recon = OrderBookReconstructor()
        df = pd.DataFrame(columns=BOOK_COLUMNS)
        results = list(recon.iter_bucketed_snapshots(df))
        assert results == []

    def test_mid_bucket_snapshot_resets(self) -> None:
        """A snapshot in bucket 2 resets the book for that bucket."""
        recon = OrderBookReconstructor()
        rows = [
            (0,               "snapshot", "bid", 70000.0, 5.0),
            (1_000_000_000,   "update",   "bid", 70000.0, 10.0),
            (2_000_000_000,   "snapshot", "bid", 71000.0, 1.0),  # resets
            (2_000_000_100,   "update",   "bid", 71000.0, 2.0),
        ]
        df = _make_book_df(rows)
        results = list(recon.iter_bucketed_snapshots(df, interval_ns=1_000_000_000, n=20))

        assert len(results) == 3

        # Bucket 0
        assert results[0][1] == [(70000.0, 5.0)]

        # Bucket 1 — update on top
        assert results[1][1] == [(70000.0, 10.0)]

        # Bucket 2 — snapshot cleared, then update applied
        assert results[2][1] == [(71000.0, 2.0)]


class TestStaleEviction:
    """Stale-level eviction prevents ghost levels from contaminating snapshots.

    See docs/orderbook_data_audit.md for the bug this fixes.
    """

    def test_evict_stale_removes_old_bid(self) -> None:
        recon = OrderBookReconstructor()
        # Place a bid at t=0, no further updates
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        # At t=60s, evict with max_age=30s → bid should be gone
        n = recon.evict_stale(current_time_ms=60_000, max_age_ms=30_000)
        assert n == 1
        bids, _ = recon.top_n(20)
        assert bids == []

    def test_evict_stale_keeps_recent_bid(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        # At t=10s, max_age=30s → bid is recent, keep
        n = recon.evict_stale(current_time_ms=10_000, max_age_ms=30_000)
        assert n == 0
        bids, _ = recon.top_n(20)
        assert bids == [(100.0, 1.0)]

    def test_evict_stale_keeps_updated_bid(self) -> None:
        """A bid that was placed long ago but updated recently survives."""
        recon = OrderBookReconstructor()
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        # Update at t=50s
        recon.apply("bid", 100.0, 1.5, timestamp_ms=50_000)
        # At t=60s, max_age=30s → updated at 50s, still recent
        n = recon.evict_stale(current_time_ms=60_000, max_age_ms=30_000)
        assert n == 0
        bids, _ = recon.top_n(20)
        assert bids == [(100.0, 1.5)]

    def test_evict_stale_clears_crossed_book(self) -> None:
        """The original bug: a stale bid above a recent ask creates a crossed book."""
        recon = OrderBookReconstructor()
        # Stale bid placed 6 hours ago, no updates
        recon.apply("bid", 100.0, 0.5, timestamp_ms=0)
        # Recent ask (current price)
        recon.apply("ask", 95.0, 1.0, timestamp_ms=6 * 3600 * 1000)
        # Before eviction: best bid (100) > best ask (95) → CROSSED
        bids, asks = recon.top_n(20)
        assert bids[0][0] > asks[0][0]
        # Evict
        n = recon.evict_stale(current_time_ms=6 * 3600 * 1000, max_age_ms=30_000)
        assert n == 1
        # After: book is no longer crossed
        bids, asks = recon.top_n(20)
        assert bids == []
        assert asks == [(95.0, 1.0)]

    def test_evict_stale_both_sides(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        recon.apply("ask", 105.0, 1.0, timestamp_ms=0)
        n = recon.evict_stale(current_time_ms=60_000, max_age_ms=30_000)
        assert n == 2
        assert recon.top_n(20) == ([], [])

    def test_clear_resets_timestamps(self) -> None:
        recon = OrderBookReconstructor()
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        recon.clear()
        # After clear, no levels to evict even at t=infinity
        n = recon.evict_stale(current_time_ms=10**12, max_age_ms=30_000)
        assert n == 0

    def test_apply_with_no_timestamp_preserves_old(self) -> None:
        """If a row has no timestamp_ms, the level's old ts is kept
        (and eventually evicts — which is the correct behavior)."""
        recon = OrderBookReconstructor()
        recon.apply("bid", 100.0, 1.0, timestamp_ms=0)
        # Re-apply quantity without timestamp — old ts=0 stays
        recon.apply("bid", 100.0, 2.0, timestamp_ms=None)
        # At t=60s, evicts because ts is still 0
        n = recon.evict_stale(current_time_ms=60_000, max_age_ms=30_000)
        assert n == 1
