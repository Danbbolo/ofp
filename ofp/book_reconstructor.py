"""
book_reconstructor.py — In-memory L2 order book maintained from delta updates.

Processes ``BookSnapshotData`` rows (snapshot + update events) and provides
top-N bid/ask snapshots on demand.  Supports 1-second bucketed iteration.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
from sortedcontainers import SortedDict


class OrderBookReconstructor:
    """
    Maintains a full in-memory limit order book from CryptoHFTData deltas.

    - Bids are stored in a ``SortedDict`` (ascending by price).  Top-N =
      reverse-iterated from the highest keys.
    - Asks are stored in a ``SortedDict`` (ascending by price).  Top-N =
      forward-iterated from the lowest keys.
    - All price levels are tracked, not only the top 20 — a level outside
      the top 20 now can become top-20 later.
    - Zero-quantity updates remove the level.
    """

    def __init__(self) -> None:
        self._bids: SortedDict[float, float] = SortedDict()
        self._asks: SortedDict[float, float] = SortedDict()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Empty both sides of the book."""
        self._bids.clear()
        self._asks.clear()

    def apply(self, side: str, price: float, quantity: float) -> None:
        """
        Apply a single price-level delta.

        Parameters
        ----------
        side : ``"bid"`` or ``"ask"``
        price : float
        quantity : float
            A value of ``0.0`` removes the level.
        """
        book = self._bids if side == "bid" else self._asks
        if quantity == 0.0:
            book.pop(price, None)
        else:
            book[price] = quantity

    def apply_snapshot(self, df: pd.DataFrame) -> None:
        """
        **Replace** the current book state with the rows in *df*.

        All existing levels are cleared before the snapshot rows are applied.
        Rows with ``event_type == "update"`` are still applied — the caller
        must ensure the first rows are genuinely a snapshot; this method does
        not inspect ``event_type``.
        """
        self.clear()
        self._apply_rows(df)

    def apply_updates(self, df: pd.DataFrame) -> None:
        """
        Apply delta updates on top of the existing book (no clear).

        Rows with ``quantity == 0`` remove the level.
        """
        self._apply_rows(df)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_n(self, n: int = 20) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Return ``(bids, asks)`` — each a list of ``(price, size)`` tuples.

        Bids are sorted **descending** (highest price first).
        Asks are sorted **ascending** (lowest price first).
        If fewer than *n* levels exist, all available levels are returned.
        """
        # Bids: SortedDict is ascending → take last n keys, reverse
        bid_items = list(self._bids.items())
        top_bids = bid_items[-n:] if len(bid_items) > n else bid_items
        bids = [(price, size) for price, size in reversed(top_bids)]

        # Asks: SortedDict is ascending → take first n keys
        ask_items = list(self._asks.items())[:n]
        asks = list(ask_items)

        return bids, asks

    def __len__(self) -> int:
        """Total number of levels across both sides."""
        return len(self._bids) + len(self._asks)

    # ------------------------------------------------------------------
    # Bucketed iteration
    # ------------------------------------------------------------------

    def iter_bucketed_snapshots(
        self,
        df: pd.DataFrame,
        interval_ns: int = 1_000_000_000,
        n: int = 20,
    ) -> Iterator[tuple[int, list[tuple[float, float]], list[tuple[float, float]]]]:
        """
        Process *df* in time-order, yielding one top-N snapshot per time
        bucket.

        The DataFrame is sorted by ``event_time``, then partitioned into
        buckets of duration *interval_ns* (default 1 second = 1e9 ns).

        Within each bucket:
        1. If **any** row has ``event_type == "snapshot"``, the book is
           cleared before applying that bucket's rows (the snapshot rows
           themselves rebuild it).
        2. All rows in the bucket are applied in order.
        3. At the bucket boundary, the top-N snapshot is yielded.

        Yields
        ------
        (bucket_start_ns, bids, asks)
            *bucket_start_ns* is the ``event_time`` of the first row in the
            bucket, floored to the bucket boundary.
        """
        if df.empty:
            return

        sorted_df = df.sort_values("event_time")
        start_epoch = int(sorted_df["event_time"].iloc[0])
        bucket_start = (start_epoch // interval_ns) * interval_ns

        def _bucket_key(et: int) -> int:
            return (et // interval_ns) * interval_ns

        for bucket_ts, group in sorted_df.groupby(
            sorted_df["event_time"].apply(_bucket_key), sort=True
        ):
            # Check for snapshot in this bucket
            if (group["event_type"] == "snapshot").any():
                self.clear()

            self._apply_rows(group)  # type: ignore[arg-type]

            yield (int(bucket_ts), *self.top_n(n))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_rows(self, df: pd.DataFrame) -> None:
        """Apply every row in *df* via ``self.apply()``."""
        for row in df.itertuples(index=False):
            self.apply(row.side, row.price, row.quantity)
