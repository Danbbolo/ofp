"""
grid_sweeper.py — Parametric grid sweep across historical data.

Slides windows over trades, book deltas, and liquidations, extracts
features via ``extract_features()``, computes forward returns, and
yields labelled rows one at a time (generator).  Supports chunked
parquet output via ``save_to_disk()``.
"""

from __future__ import annotations

import bisect
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ofp.book_reconstructor import OrderBookReconstructor
from ofp.feature_extractor import extract_features


class GridSweeper:
    """
    Slides feature-extraction windows across historical data.

    Parameters
    ----------
    window_sizes_sec : list[int]
        Window durations in seconds, e.g. ``[60, 120, 300]``.
    horizons_sec : list[int]
        Forward-return horizons in seconds, e.g. ``[300, 900, 3600]``.
    """

    def __init__(
        self,
        window_sizes_sec: list[int],
        horizons_sec: list[int],
    ) -> None:
        if not window_sizes_sec:
            raise ValueError("window_sizes_sec must not be empty")
        if not horizons_sec:
            raise ValueError("horizons_sec must not be empty")

        self._window_sizes_sec = window_sizes_sec
        self._horizons_sec = horizons_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sweep(
        self,
        trades_df: pd.DataFrame,
        book_df: pd.DataFrame,
        liq_df: pd.DataFrame,
        rolling_avg_volume: float,
        _24h_stats: dict[str, float] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield one feature+label dict per (window, horizon, step).

        Nothing is accumulated — the caller must consume the generator
        and persist rows as needed.

        Parameters
        ----------
        trades_df : DataFrame
            Columns: ``timestamp_ms``, ``price``, ``size``, ``is_buyer_maker``.
        book_df : DataFrame
            ``BookSnapshotData`` columns.  Will be reconstructed into 1 s
            snapshots via ``OrderBookReconstructor``.
        liq_df : DataFrame
            Columns: ``timestamp_ms``, ``side``, ``price``, ``size``.
        rolling_avg_volume : float
            Long-term average volume for normalisation.
        _24h_stats : dict | None
            Optional keys: ``_24h_avg_range``, ``_24h_low``, ``_24h_high``,
            ``current_price``.
        """
        if _24h_stats is None:
            _24h_stats = {}

        # ------------------------------------------------------------------
        # Sort source data
        # ------------------------------------------------------------------
        trades = trades_df.sort_values("timestamp_ms").reset_index(drop=True)
        liq = liq_df.sort_values("timestamp_ms").reset_index(drop=True)

        # ------------------------------------------------------------------
        # Pre-build 1-second book snapshots
        # ------------------------------------------------------------------
        recon = OrderBookReconstructor()
        book_snapshots: list[tuple[int, list, list]] = list(
            recon.iter_bucketed_snapshots(book_df, interval_ns=1_000_000_000, n=20)
        )
        snap_ts = np.array([sn[0] for sn in book_snapshots], dtype=np.int64)

        def _book_at(target_ns: int) -> tuple[list, list]:
            """Return (bids, asks) closest to *target_ns* (inclusive)."""
            if len(snap_ts) == 0:
                return ([], [])
            idx = int(np.searchsorted(snap_ts, target_ns, side="right")) - 1
            if idx < 0:
                return ([], [])
            return book_snapshots[idx][1], book_snapshots[idx][2]

        # ------------------------------------------------------------------
        # Price lookup helpers
        # ------------------------------------------------------------------
        trade_ts = trades["timestamp_ms"].values
        trade_px = trades["price"].values

        def _price_at(target_ms: int) -> float | None:
            """First trade price at or after *target_ms*, or None."""
            idx = int(np.searchsorted(trade_ts, target_ms, side="left"))
            if idx >= len(trade_ts):
                return None
            return float(trade_px[idx])

        # ------------------------------------------------------------------
        # Time range
        # ------------------------------------------------------------------
        data_start_ms = int(trades["timestamp_ms"].iloc[0])
        data_end_ms = int(trades["timestamp_ms"].iloc[-1])

        # ------------------------------------------------------------------
        # Sweep
        # ------------------------------------------------------------------
        for window_sec in self._window_sizes_sec:
            window_ms = window_sec * 1000
            step_ms = window_ms // 2  # 50 % overlap

            for horizon_sec in self._horizons_sec:
                horizon_ms = horizon_sec * 1000
                win_start = data_start_ms

                while win_start + window_ms + horizon_ms <= data_end_ms:
                    win_end = win_start + window_ms
                    future_target_ms = win_end + horizon_ms

                    # Book snapshots (ns → ms conversion)
                    book_start = _book_at(win_start * 1_000_000)
                    book_end = _book_at(win_end * 1_000_000)

                    # Current & future price
                    current_px = _price_at(win_end)
                    future_px = _price_at(future_target_ms)

                    if current_px is None or future_px is None:
                        win_start += step_ms
                        continue

                    # Features
                    feats = extract_features(
                        trades_df=trades,
                        book_snapshot_start=book_start,
                        book_snapshot_end=book_end,
                        liq_df=liq,
                        window_start_ms=win_start,
                        window_end_ms=win_end,
                        rolling_avg_volume=rolling_avg_volume,
                        current_price=current_px,
                        **{k: _24h_stats.get(k, 0.0) for k in (
                            "_24h_avg_range", "_24h_low", "_24h_high",
                        )},
                    )

                    # Labels
                    outcome_pct = (future_px - current_px) / current_px
                    outcome_binary = 1 if outcome_pct > 0 else 0

                    yield {
                        **feats,
                        "outcome_pct":   outcome_pct,
                        "outcome_binary": outcome_binary,
                        "window_size":   window_sec,
                        "horizon":       horizon_sec,
                        "window_end_ms": win_end,
                    }

                    win_start += step_ms

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_to_disk(
        data_gen: Iterator[dict[str, Any]],
        filepath: str,
        chunk_size: int = 100_000,
    ) -> None:
        """
        Consume *data_gen* and write to a single Parquet file in chunks.

        Parameters
        ----------
        data_gen : Iterator[dict]
            Typically the output of ``GridSweeper.sweep()``.
        filepath : str
            Path to the ``.parquet`` output file.
        chunk_size : int
            Rows per write batch.
        """
        writer: pq.ParquetWriter | None = None
        buffer: list[dict[str, Any]] = []

        for row in data_gen:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                batch = pa.RecordBatch.from_pylist(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(filepath, batch.schema)
                writer.write_batch(batch)
                buffer.clear()

        # Flush remainder
        if buffer:
            batch = pa.RecordBatch.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(filepath, batch.schema)
            writer.write_batch(batch)

        if writer is not None:
            writer.close()
