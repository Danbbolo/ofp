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

from ofp.feature_extractor import extract_multi_zoom_features


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
        rolling_stats_per_zoom: dict[str, dict[str, float]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Multi-zoom sweep: micro windows slide, meso (300s) + macro (1800s) fixed.
        Horizons: [300, 900, 1800].

        Parameters
        ----------
        rolling_stats_per_zoom
            ``{"micro": {...}, "meso": {...}, "macro": {...}}`` — each zoom
            must have its own ``rolling_avg_volume`` baseline.  Passing a
            single global value across zooms is a context leak.
        """
        if rolling_stats_per_zoom is None:
            rolling_stats_per_zoom = {"micro": {}, "meso": {}, "macro": {}}

        trades = trades_df
        liq = liq_df

        MESO_MS = 300_000
        MACRO_MS = 1_800_000

        trade_ts = trades["timestamp_ms"].values
        trade_px = trades["price"].values

        # ----------------------------------------------------------------
        # Build the prior-24h cache ONCE per sweep.  This is the same
        # 24h min/max that extract_multi_zoom_features would otherwise
        # recompute per row — making the sweep O(n²) and slow.
        # ----------------------------------------------------------------
        from collections import deque
        if len(trade_ts) == 0:
            d24_sec = np.zeros(1, dtype=np.int64)
            d24_low = np.zeros(1, dtype=np.float64)
            d24_high = np.zeros(1, dtype=np.float64)
        else:
            sec_ts = trade_ts // 1000
            unique_secs, inv = np.unique(sec_ts, return_inverse=True)
            n_secs = len(unique_secs)
            sec_min = np.full(n_secs, np.inf)
            sec_max = np.full(n_secs, -np.inf)
            np.minimum.at(sec_min, inv, trade_px)
            np.maximum.at(sec_max, inv, trade_px)

            day_secs = 86_400
            d24_low = np.empty(n_secs, dtype=np.float64)
            d24_high = np.empty(n_secs, dtype=np.float64)
            min_q: deque = deque()
            max_q: deque = deque()
            lo = 0
            for hi in range(n_secs):
                while min_q and sec_min[min_q[-1]] >= sec_min[hi]:
                    min_q.pop()
                min_q.append(hi)
                while max_q and sec_max[max_q[-1]] <= sec_max[hi]:
                    max_q.pop()
                max_q.append(hi)
                while unique_secs[hi] - unique_secs[lo] >= day_secs:
                    lo += 1
                    if min_q[0] < lo:
                        min_q.popleft()
                    if max_q[0] < lo:
                        max_q.popleft()
                d24_low[hi] = sec_min[min_q[0]]
                d24_high[hi] = sec_max[max_q[0]]
            d24_sec = unique_secs
        prior_24h_cache = (d24_sec, d24_low, d24_high)

        def _price_at(target_ms: int) -> float | None:
            idx = int(np.searchsorted(trade_ts, target_ms, side="left"))
            if idx >= len(trade_ts):
                return None
            return float(trade_px[idx])

        data_start_ms = int(trade_ts[0])
        data_end_ms = int(trade_ts[-1])

        for window_sec in self._window_sizes_sec:
            micro_ms = window_sec * 1000
            step_ms = micro_ms // 2

            for horizon_sec in self._horizons_sec:
                horizon_ms = horizon_sec * 1000
                win_start = data_start_ms

                while win_start + micro_ms + horizon_ms <= data_end_ms:
                    win_end = win_start + micro_ms
                    future_ms = win_end + horizon_ms

                    current_px = _price_at(win_end)
                    future_px = _price_at(future_ms)

                    if current_px is None or future_px is None or current_px <= 0 or future_px <= 0:
                        win_start += step_ms
                        continue

                    feats = extract_multi_zoom_features(
                        trades_df=trades,
                        book_snapshots=book_snapshots,
                        liq_df=liq,
                        micro_window_ms=micro_ms,
                        meso_window_ms=MESO_MS,
                        macro_window_ms=MACRO_MS,
                        end_time_ms=win_end,
                        rolling_stats_per_zoom=rolling_stats_per_zoom,
                        current_price=current_px,
                        prior_24h_cache=prior_24h_cache,
                    )

                    outcome_pct = (future_px - current_px) / current_px
                    outcome_binary = 1 if outcome_pct > 0 else 0

                    yield {
                        **feats,
                        "outcome_pct": outcome_pct,
                        "outcome_binary": outcome_binary,
                        "window_size": window_sec,
                        "horizon": horizon_sec,
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
        import gc
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
                gc.collect()  # free memory before next chunk (per spec)

        # Flush remainder
        if buffer:
            batch = pa.RecordBatch.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(filepath, batch.schema)
            writer.write_batch(batch)

        if writer is not None:
            writer.close()
