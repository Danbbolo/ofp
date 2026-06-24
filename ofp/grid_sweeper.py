"""
grid_sweeper.py — Parametric grid sweep across historical data.

Receives pre-converted DataFrames and pre-built book snapshot dict.
Slides windows, extracts features, computes labels, yields one dict at a time.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ofp.feature_extractor import extract_features


class GridSweeper:
    """Slides feature-extraction windows across pre-processed historical data."""

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
        book_snapshots: dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]]]],
        liq_df: pd.DataFrame,
        rolling_avg_volume: float,
        _24h_stats: dict[str, float] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield one feature+label dict per (window, horizon, step).

        All DataFrames must be pre-sorted and pre-typed (no string columns).
        *book_snapshots* is a dict keyed by **millisecond** timestamp.
        """
        if _24h_stats is None:
            _24h_stats = {}

        trades = trades_df
        liq = liq_df

        # Book lookup
        snap_ts = np.array(sorted(book_snapshots.keys()), dtype=np.int64)

        def _book_at(target_ms: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
            if len(snap_ts) == 0:
                return ([], [])
            idx = int(np.searchsorted(snap_ts, target_ms, side="right")) - 1
            if idx < 0:
                return ([], [])
            return book_snapshots[int(snap_ts[idx])]

        # Price lookup & slicing
        trade_ts = trades["timestamp_ms"].values
        trade_px = trades["price"].values

        def _price_at(target_ms: int) -> float | None:
            idx = int(np.searchsorted(trade_ts, target_ms, side="left"))
            if idx >= len(trade_ts):
                return None
            return float(trade_px[idx])

        def _slice_trades(lo_ms: int, hi_ms: int) -> pd.DataFrame:
            lo = int(np.searchsorted(trade_ts, lo_ms, side="left"))
            hi = int(np.searchsorted(trade_ts, hi_ms, side="left"))
            return trades.iloc[lo:hi]

        liq_ts = liq["timestamp_ms"].values

        def _slice_liq(lo_ms: int, hi_ms: int) -> pd.DataFrame:
            lo = int(np.searchsorted(liq_ts, lo_ms, side="left"))
            hi = int(np.searchsorted(liq_ts, hi_ms, side="left"))
            return liq.iloc[lo:hi]

        # Sweep
        data_start_ms = int(trade_ts[0])
        data_end_ms = int(trade_ts[-1])

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
                    future_ms = win_end + horizon_ms

                    current_px = _price_at(win_end)
                    future_px = _price_at(future_ms)

                    if current_px is None or future_px is None:
                        win_start += step_ms
                        continue

                    feats = extract_features(
                        trades_df=_slice_trades(win_start, win_end),
                        book_snapshot_start=_book_at(win_start),
                        book_snapshot_end=_book_at(win_end),
                        liq_df=_slice_liq(win_start, win_end),
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
