# --- ofp/data_schema.py ---
"""
data_schema.py — Strict Pydantic schemas for CryptoHFTData API responses.

Matches the CryptoHFTData REST API JSON format exactly.
All price/quantity fields arrive as strings and are coerced to float64.
"""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, BeforeValidator, Field


# ---------------------------------------------------------------------------
# Shared coercion helpers
# ---------------------------------------------------------------------------

def _parse_float(v: Any) -> float:
    """Coerce a string (or numeric) value to float64. Raises on failure."""
    if isinstance(v, str):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"Cannot coerce {type(v).__name__} to float: {v!r}")


def _parse_int64(v: Any) -> int:
    """Coerce a value to int64. Raises on failure."""
    if isinstance(v, str):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    raise TypeError(f"Cannot coerce {type(v).__name__} to int64: {v!r}")


# Reusable annotated types
Float64 = Annotated[float, BeforeValidator(_parse_float)]
Int64 = Annotated[int, BeforeValidator(_parse_int64)]


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

class TradesData(BaseModel):
    """
    A single trade from the CryptoHFTData /trades endpoint.

    CRITICAL: ``is_buyer_maker == True`` means the aggressor is a SELLER
    (market sell hitting the bid).  ``False`` means the aggressor is a BUYER
    (market buy lifting the ask).  This matches the Binance convention.
    """

    received_time: Int64
    event_time: Int64
    symbol: str
    trade_id: Int64
    price: Float64
    quantity: Float64
    trade_time: Int64
    is_buyer_maker: bool
    order_type: str

    @staticmethod
    def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse a list of raw dicts into a validated DataFrame."""
        validated = [TradesData(**r) for r in records]
        return pd.DataFrame([m.model_dump() for m in validated])


# ---------------------------------------------------------------------------
# Book Snapshot / L2 Delta
# ---------------------------------------------------------------------------

class BookSnapshotData(BaseModel):
    """
    A single L2 order-book delta row from CryptoHFTData /book_snapshot.

    ``event_type`` is ``"snapshot"`` or ``"update"``.
    ``side`` is ``"bid"`` or ``"ask"``.
    ``price`` and ``quantity`` are parsed from their string representations.
    A quantity of 0 on an update means the level was removed.
    """

    received_time: Int64
    event_time: Int64
    symbol: str
    event_type: str  # "snapshot" | "update"
    side: str        # "bid" | "ask"
    price: Float64
    quantity: Float64

    @staticmethod
    def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse a list of raw dicts into a validated DataFrame."""
        validated = [BookSnapshotData(**r) for r in records]
        return pd.DataFrame([m.model_dump() for m in validated])


# ---------------------------------------------------------------------------
# Liquidations
# ---------------------------------------------------------------------------

class LiquidationData(BaseModel):
    """
    A single liquidation event from CryptoHFTData /liquidations.

    ``side`` semantics (CryptoHFTData convention):
        * ``"SELL"`` — a long position was liquidated (forced market sell).
        * ``"BUY"``  — a short position was liquidated (forced market buy).
    """

    received_time: Int64
    event_time: Int64
    symbol: str
    side: str  # "BUY" | "SELL"
    price: Float64
    quantity: Float64
    trade_time: Int64

    @staticmethod
    def to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse a list of raw dicts into a validated DataFrame."""
        validated = [LiquidationData(**r) for r in records]
        return pd.DataFrame([m.model_dump() for m in validated])

# --- ofp/api_streamer.py ---
"""
api_streamer.py — Async streaming client for the CryptoHFTData REST API.

Fetches historical trades, L2 book snapshots, and liquidations in paginated
chunks, validates every record through Pydantic, and yields clean DataFrames.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Literal

import httpx

from ofp.data_schema import BookSnapshotData, LiquidationData, TradesData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.cryptohftdata.com/v1"
DEFAULT_CHUNK_SIZE = 10_000  # records per page
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # seconds
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

DataType = Literal["trades", "book_snapshot", "liquidations"]

SCHEMA_MAP = {
    "trades": TradesData,
    "book_snapshot": BookSnapshotData,
    "liquidations": LiquidationData,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CryptoHFTAPIError(Exception):
    """Raised when the CryptoHFTData API returns a non-retryable error."""


class CryptoHFTValidationError(Exception):
    """Raised when the API response fails Pydantic schema validation."""


class CryptoHFTTimeoutError(Exception):
    """Raised when all retries are exhausted due to timeouts / 5xx."""


# ---------------------------------------------------------------------------
# Streamer
# ---------------------------------------------------------------------------

class CryptoHFTStreamer:
    """
    Async streaming client for CryptoHFTData.

    Usage::

        async with CryptoHFTStreamer() as streamer:
            async for df in streamer.fetch_data(
                symbol="BTCUSDT",
                data_type="trades",
                start_time=1719000000000000000,
                end_time=1719100000000000000,
            ):
                process(df)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chunk_size = chunk_size
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CryptoHFTStreamer":
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_data(
        self,
        symbol: str,
        data_type: DataType,
        start_time: int,
        end_time: int,
    ) -> AsyncIterator["pd.DataFrame"]:  # noqa: F821
        """
        Fetch data from CryptoHFTData, yielding one validated DataFrame per
        page.  Never loads the entire dataset into memory at once.

        Parameters
        ----------
        symbol : str
            e.g. ``"BTCUSDT"``.
        data_type : Literal["trades", "book_snapshot", "liquidations"]
        start_time : int
            Start timestamp in **nanoseconds** (inclusive).
        end_time : int
            End timestamp in **nanoseconds** (exclusive).
        """
        if self._client is None:
            raise RuntimeError(
                "CryptoHFTStreamer must be used as an async context manager "
                "(`async with CryptoHFTStreamer() as s: ...`)"
            )

        if start_time >= end_time:
            raise ValueError(
                f"start_time ({start_time}) must be < end_time ({end_time})"
            )

        schema = SCHEMA_MAP[data_type]
        offset = 0

        while True:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
                "limit": self._chunk_size,
                "offset": offset,
            }

            records = await self._fetch_page(data_type, params)

            if not records:
                break  # no more data

            yield schema.to_dataframe(records)

            if len(records) < self._chunk_size:
                break  # last page

            offset += len(records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_page(
        self, endpoint: str, params: dict[str, str | int]
    ) -> list[dict[str, object]]:
        """Fetch a single page with retry logic."""
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                assert self._client is not None
                resp = await self._client.get(f"/{endpoint}", params=params)

                if resp.status_code == 200:
                    return self._parse_response(resp)

                if resp.status_code in RETRYABLE_STATUSES:
                    last_exc = CryptoHFTAPIError(
                        f"Retryable status {resp.status_code} from "
                        f"/{endpoint}: {resp.text[:500]}"
                    )
                else:
                    raise CryptoHFTAPIError(
                        f"Non-retryable status {resp.status_code} from "
                        f"/{endpoint}: {resp.text[:500]}"
                    )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc

            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Attempt %d/%d for /%s failed: %s.  "
                    "Retrying in %.1fs …",
                    attempt, MAX_RETRIES, endpoint, last_exc, backoff,
                )
                await asyncio.sleep(backoff)

        # All retries exhausted
        raise CryptoHFTTimeoutError(
            f"All {MAX_RETRIES} retries exhausted for /{endpoint}.  "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _parse_response(resp: httpx.Response) -> list[dict[str, object]]:
        """Parse JSON body, ensuring we got a list of dicts."""
        try:
            body = resp.json()
        except ValueError as exc:
            raise CryptoHFTAPIError(
                f"Response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(body, list):
            raise CryptoHFTAPIError(
                f"Expected a JSON array, got {type(body).__name__}: "
                f"{str(body)[:300]}"
            )

        return body

# --- ofp/book_reconstructor.py ---
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

# --- ofp/feature_extractor.py ---
"""
feature_extractor.py — Extract 28 features from a trading window.

Consumes validated trades, L2 book snapshots, and liquidation DataFrames.
No models, no indicators — just deterministic feature arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def extract_features(
    trades_df: pd.DataFrame,
    book_snapshot_start: tuple[list[tuple[float, float]], list[tuple[float, float]]],
    book_snapshot_end: tuple[list[tuple[float, float]], list[tuple[float, float]]],
    liq_df: pd.DataFrame,
    window_start_ms: int,
    window_end_ms: int,
    rolling_avg_volume: float,
    *,
    _24h_avg_range: float = 0.0,
    _24h_low: float = 0.0,
    _24h_high: float = 0.0,
    current_price: float = 0.0,
) -> dict[str, float]:
    """
    Extract exactly 28 features from one time window.

    Parameters
    ----------
    trades_df : DataFrame
        Columns: ``timestamp_ms``, ``price``, ``size``, ``is_buyer_maker``.
    book_snapshot_start : (bids, asks)
        Each is ``[(price, size), ...]`` — top-20 book at window start.
    book_snapshot_end : (bids, asks)
        Top-20 book at window end.
    liq_df : DataFrame
        Columns: ``timestamp_ms``, ``side``, ``price``, ``size``.
    window_start_ms : int
    window_end_ms : int
    rolling_avg_volume : float
        Average volume over a larger lookback for normalisation.
    _24h_avg_range, _24h_low, _24h_high, current_price : float
        Contextual 24h stats (default 0.0 — pass real values when available).

    Returns
    -------
    dict[str, float]
        28 keys: ``buy_volume`` through ``trend_slope``.
        Groups B (book) and C (liquidation) are reserved at 0.0.
    """
    # ------------------------------------------------------------------
    # trades_df is assumed pre-sliced to [window_start_ms, window_end_ms)
    # by the caller (GridSweeper).  No internal filtering here.
    # ------------------------------------------------------------------
    win = trades_df.copy()

    # Pre-compute signed size: + for buys (aggressor=BUYER → is_buyer_maker=False),
    #                          − for sells (aggressor=SELLER → is_buyer_maker=True)
    win["signed_size"] = win["size"].where(~win["is_buyer_maker"], -win["size"])

    # ------------------------------------------------------------------
    # Group A — The Attack (Market Trades)  [keys  1–12]
    # ------------------------------------------------------------------

    # 1. buy_volume
    buy_volume = float(win.loc[~win["is_buyer_maker"], "size"].sum())

    # 2. sell_volume
    sell_volume = float(win.loc[win["is_buyer_maker"], "size"].sum())

    # 3. net_volume
    net_volume = buy_volume - sell_volume

    # 4. buy_sell_ratio
    buy_sell_ratio = buy_volume / (sell_volume + 1e-9)

    # 5. volume_vs_avg
    total_volume = buy_volume + sell_volume
    volume_vs_avg = total_volume / (rolling_avg_volume + 1e-9)

    # 6. large_trade_net
    n_trades = len(win)
    if n_trades > 0:
        avg_trade_size = total_volume / n_trades
        threshold = 2.0 * avg_trade_size
        large_mask = win["size"] > threshold
        large_trade_net = float(win.loc[large_mask, "signed_size"].sum())
    else:
        large_trade_net = 0.0

    # 7. acceleration  (second half net − first half net)
    if n_trades > 0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        first_half = win.loc[win["timestamp_ms"] < mid_ms, "signed_size"].sum()
        second_half = win.loc[win["timestamp_ms"] >= mid_ms, "signed_size"].sum()
        acceleration = float(second_half - first_half)
    else:
        acceleration = 0.0

    # 8–12.  Delta curve  (cumulative net volume at 20/40/60/80/100 % marks)
    window_dur = window_end_ms - window_start_ms

    if n_trades > 0 and window_dur > 0:
        win_sorted = win.sort_values("timestamp_ms")
        cum_net = win_sorted["signed_size"].cumsum().values

        # Compute fractional position [0, 1] for each trade within the window
        frac = (win_sorted["timestamp_ms"].values - window_start_ms) / window_dur

        def _cum_at(limit: float) -> float:
            """Last cumulative value where frac <= limit, else 0.0."""
            idx = -1
            for i, f in enumerate(frac):
                if f <= limit:
                    idx = i
                else:
                    break
            if idx >= 0:
                return float(cum_net[idx])
            return 0.0

        delta_1 = _cum_at(0.20)
        delta_2 = _cum_at(0.40)
        delta_3 = _cum_at(0.60)
        delta_4 = _cum_at(0.80)
        delta_5 = _cum_at(1.00)
    else:
        delta_1 = delta_2 = delta_3 = delta_4 = delta_5 = 0.0

    # ------------------------------------------------------------------
    # Group B — The Defence (Book Depth)     [keys 13–20]
    # ------------------------------------------------------------------
    bids_end, asks_end = book_snapshot_end
    bids_start, asks_start = book_snapshot_start

    # 13.  bid_ask_imbalance  (top-5 total bid / top-5 total ask, end snapshot)
    bid_ask_imbalance = _bid_ask_imbalance(bids_end, asks_end, n=5)

    # 14.  bid_wall  (largest single bid size in top 5)
    bid_wall = _max_size(bids_end, n=5)

    # 15.  ask_wall  (largest single ask size in top 5)
    ask_wall = _max_size(asks_end, n=5)

    # 16.  wall_asymmetry
    wall_asymmetry = bid_wall / (ask_wall + 1e-9)

    # 17.  depth_trend  (start imbalance − end imbalance)
    depth_trend = _bid_ask_imbalance(bids_start, asks_start, n=5) - bid_ask_imbalance

    # 18.  spread_bps  (end snapshot)
    spread_bps = _spread_bps(bids_end, asks_end)

    # 19.  spread_change  (end − start)
    spread_change = spread_bps - _spread_bps(bids_start, asks_start)

    # 20.  book_depth_slope  (linear slope of cumulative combined depth, top 5)
    book_depth_slope = _depth_slope(bids_end, asks_end, n=5)

    # ------------------------------------------------------------------
    # Group C — The Forced Errors (Liquidations)  [keys 21–25]
    # (liq_df is assumed pre-sliced by the caller)
    # ------------------------------------------------------------------
    liq_win = liq_df

    # 21.  long_liq_vol  (side == "SELL" → long was liquidated)
    long_liq_vol = float(liq_win.loc[liq_win["side"] == "SELL", "size"].sum())

    # 22.  short_liq_vol  (side == "BUY" → short was liquidated)
    short_liq_vol = float(liq_win.loc[liq_win["side"] == "BUY", "size"].sum())

    # 23.  net_liq  (short − long)
    net_liq = short_liq_vol - long_liq_vol

    # 24.  liq_climax  (total liq / total trade volume)
    total_liq_vol = long_liq_vol + short_liq_vol
    liq_climax = total_liq_vol / (total_volume + 1e-9)

    # 25.  liq_timing  (1 if >70 % of liq vol in second half, else 0)
    if total_liq_vol > 0.0:
        mid_ms = window_start_ms + (window_end_ms - window_start_ms) / 2.0
        second_half_liq = float(
            liq_win.loc[liq_win["timestamp_ms"] >= mid_ms, "size"].sum()
        )
        liq_timing = 1.0 if (second_half_liq / total_liq_vol) > 0.70 else 0.0
    else:
        liq_timing = 0.0

    # ------------------------------------------------------------------
    # Group D — Context                      [keys 26–30]
    # ------------------------------------------------------------------

    # 26–27.  Hour cyclicals
    hour = (window_end_ms // 3_600_000) % 24
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    # 28.  vol_ratio  (window range / 24h average range)
    if n_trades > 0:
        max_price = float(win["price"].max())
        min_price = float(win["price"].min())
        vol_ratio = (max_price - min_price) / (_24h_avg_range + 1e-9)
    else:
        vol_ratio = 0.0

    # 29.  price_position  ((current − 24h_low) / (24h_high − 24h_low))
    price_position = (current_price - _24h_low) / (_24h_high - _24h_low + 1e-9)

    # 30.  trend_slope  ((last − first) / first)
    if n_trades > 0:
        first_price = float(win["price"].iloc[0])
        last_price = float(win["price"].iloc[-1])
        trend_slope = (last_price - first_price) / (first_price + 1e-9)
    else:
        trend_slope = 0.0

    # ------------------------------------------------------------------
    # Assemble exactly 30 keys
    # ------------------------------------------------------------------
    return {
        # Group A  (1–12)
        "buy_volume":          buy_volume,
        "sell_volume":         sell_volume,
        "net_volume":          net_volume,
        "buy_sell_ratio":      buy_sell_ratio,
        "volume_vs_avg":       volume_vs_avg,
        "large_trade_net":     large_trade_net,
        "acceleration":        acceleration,
        "delta_1":             delta_1,
        "delta_2":             delta_2,
        "delta_3":             delta_3,
        "delta_4":             delta_4,
        "delta_5":             delta_5,
        # Group B  (13–20)
        "bid_ask_imbalance":   bid_ask_imbalance,
        "bid_wall":            bid_wall,
        "ask_wall":            ask_wall,
        "wall_asymmetry":      wall_asymmetry,
        "depth_trend":         depth_trend,
        "spread_bps":          spread_bps,
        "spread_change":       spread_change,
        "book_depth_slope":    book_depth_slope,
        # Group C  (21–25)
        "long_liq_vol":        long_liq_vol,
        "short_liq_vol":       short_liq_vol,
        "net_liq":             net_liq,
        "liq_climax":          liq_climax,
        "liq_timing":           liq_timing,
        # Group D  (26–30)
        "hour_sin":            hour_sin,
        "hour_cos":            hour_cos,
        "vol_ratio":           vol_ratio,
        "price_position":      price_position,
        "trend_slope":         trend_slope,
    }


# ---------------------------------------------------------------------------
# Multi-Zoom feature extraction
# ---------------------------------------------------------------------------

def extract_multi_zoom_features(
    trades_df: pd.DataFrame,
    book_snapshots: dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]]]],
    liq_df: pd.DataFrame,
    micro_window_ms: int,
    meso_window_ms: int,
    macro_window_ms: int,
    end_time_ms: int,
    rolling_stats: dict[str, float],
) -> dict[str, float]:
    """
    Extract features at 3 zoom levels sharing the same *end_time_ms*.

    Returns a dict with keys prefixed ``micro_``, ``meso_``, ``macro_``
    (30 features × 3 = 90 keys total).
    """
    # Book lookup helper
    snap_ts = np.array(sorted(book_snapshots.keys()), dtype=np.int64)

    def _book_at(ms: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if len(snap_ts) == 0:
            return ([], [])
        idx = int(np.searchsorted(snap_ts, ms, side="right")) - 1
        if idx < 0:
            return ([], [])
        return book_snapshots[int(snap_ts[idx])]

    result: dict[str, float] = {}

    # Pre-index timestamps for fast slicing
    trade_ts = trades_df["timestamp_ms"].values
    trade_px = trades_df["price"].values
    trade_sz = trades_df["size"].values
    trade_bm = trades_df["is_buyer_maker"].values
    liq_ts = liq_df["timestamp_ms"].values if len(liq_df) > 0 else None
    liq_sd = liq_df["side"].values if len(liq_df) > 0 else None
    liq_px = liq_df["price"].values if len(liq_df) > 0 else None
    liq_sz = liq_df["size"].values if len(liq_df) > 0 else None

    for prefix, window_ms in [("micro", micro_window_ms), ("meso", meso_window_ms), ("macro", macro_window_ms)]:
        win_start = end_time_ms - window_ms

        # Slice trades using numpy (fast, no-copy views into columns)
        t_start = int(np.searchsorted(trade_ts, win_start, side="left"))
        t_end = int(np.searchsorted(trade_ts, end_time_ms, side="left"))
        sliced_trades = pd.DataFrame({
            "timestamp_ms": trade_ts[t_start:t_end],
            "price": trade_px[t_start:t_end],
            "size": trade_sz[t_start:t_end],
            "is_buyer_maker": trade_bm[t_start:t_end],
        })

        if liq_ts is not None and len(liq_df) > 0:
            l_start = int(np.searchsorted(liq_ts, win_start, side="left"))
            l_end = int(np.searchsorted(liq_ts, end_time_ms, side="left"))
            sliced_liq = pd.DataFrame({
                "timestamp_ms": liq_ts[l_start:l_end],
                "side": liq_sd[l_start:l_end],
                "price": liq_px[l_start:l_end],
                "size": liq_sz[l_start:l_end],
            })
        else:
            sliced_liq = liq_df

        feats = extract_features(
            trades_df=sliced_trades,
            book_snapshot_start=_book_at(win_start),
            book_snapshot_end=_book_at(end_time_ms),
            liq_df=sliced_liq,
            window_start_ms=win_start,
            window_end_ms=end_time_ms,
            rolling_avg_volume=rolling_stats.get("rolling_avg_volume", 0.0),
            current_price=rolling_stats.get("current_price", 0.0),
            _24h_avg_range=rolling_stats.get("_24h_avg_range", 0.0),
            _24h_low=rolling_stats.get("_24h_low", 0.0),
            _24h_high=rolling_stats.get("_24h_high", 0.0),
        )
        for k, v in feats.items():
            result[f"{prefix}_{k}"] = v

    return result

def _top_sizes(
    levels: list[tuple[float, float]], n: int
) -> list[float]:
    """Return the *size* of the first min(n, len(levels)) entries."""
    return [sz for _, sz in levels[:n]]


def _max_size(levels: list[tuple[float, float]], n: int) -> float:
    """Largest size among the top *n* levels (0.0 if empty)."""
    sizes = _top_sizes(levels, n)
    return max(sizes) if sizes else 0.0


def _bid_ask_imbalance(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    n: int = 5,
) -> float:
    """Total top-N bid size / total top-N ask size."""
    total_bid = sum(_top_sizes(bids, n))
    total_ask = sum(_top_sizes(asks, n))
    return total_bid / (total_ask + 1e-9)


def _spread_bps(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> float:
    """(best_ask - best_bid) / mid * 10000.  0.0 if either side is empty."""
    if not bids or not asks:
        return 0.0
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    return (best_ask - best_bid) / (mid + 1e-9) * 10000.0


def _depth_slope(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    n: int = 5,
) -> float:
    """
    Linear slope of cumulative combined depth across levels 0..n-1.

    x = [0, 1, ..., n-1]
    y[i] = sum(combined_depth[j] for j=0..i)

    Returns 0.0 if fewer than 2 levels exist on both sides combined.
    """
    bid_sizes = _top_sizes(bids, n)
    ask_sizes = _top_sizes(asks, n)
    max_len = max(len(bid_sizes), len(ask_sizes))
    if max_len < 2:
        return 0.0

    # Pad shorter side with zeros
    combined = [
        (bid_sizes[i] if i < len(bid_sizes) else 0.0)
        + (ask_sizes[i] if i < len(ask_sizes) else 0.0)
        for i in range(max_len)
    ]
    cumulative = np.cumsum(combined, dtype=np.float64)
    x = np.arange(len(cumulative), dtype=np.float64)
    slope, _ = np.polyfit(x, cumulative, deg=1)
    return float(slope)

# --- ofp/grid_sweeper.py ---
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
        rolling_avg_volume: float,
        _24h_stats: dict[str, float] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Multi-zoom sweep: micro windows slide, meso (300s) + macro (1800s) fixed.
        Horizons: [300, 900, 1800].
        """
        if _24h_stats is None:
            _24h_stats = {}

        trades = trades_df
        liq = liq_df

        MESO_MS = 300_000
        MACRO_MS = 1_800_000

        trade_ts = trades["timestamp_ms"].values
        trade_px = trades["price"].values

        def _price_at(target_ms: int) -> float | None:
            idx = int(np.searchsorted(trade_ts, target_ms, side="left"))
            if idx >= len(trade_ts):
                return None
            return float(trade_px[idx])

        data_start_ms = int(trade_ts[0])
        data_end_ms = int(trade_ts[-1])

        rolling_stats = {
            "rolling_avg_volume": rolling_avg_volume,
            **_24h_stats,
        }

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

                    if current_px is None or future_px is None:
                        win_start += step_ms
                        continue

                    rolling_stats["current_price"] = current_px

                    feats = extract_multi_zoom_features(
                        trades_df=trades,
                        book_snapshots=book_snapshots,
                        liq_df=liq,
                        micro_window_ms=micro_ms,
                        meso_window_ms=MESO_MS,
                        macro_window_ms=MACRO_MS,
                        end_time_ms=win_end,
                        rolling_stats=rolling_stats,
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

# --- run_research.py ---
"""
run_research.py — Multi-day grid sweep from local parquet files.

Reads raw data from ``data/raw/YYYY-MM-DD/``, builds book snapshots
incrementally across days, sweeps all days in one continuous pass.

Usage::

    python run_research.py 2026-06-01 2026-06-30

Output: ``data/research_dataset.parquet``
"""

from __future__ import annotations

import gc
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ofp.book_reconstructor import OrderBookReconstructor
from ofp.grid_sweeper import GridSweeper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "research_dataset.parquet"

WINDOW_SIZES_SEC = [60, 120, 180]
HORIZONS_SEC = [300, 900, 1800]
PROGRESS_EVERY = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"trade_time": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["size"] = out["size"].astype(float)
    return out[["timestamp_ms", "price", "size", "is_buyer_maker"]]


def _prepare_liq(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    out = df.rename(columns={"timestamp": "timestamp_ms", "quantity": "size"})
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    out["price"] = out.get("price", pd.Series([0.0] * len(out))).astype(float)
    out["size"] = out.get("size", pd.Series([0.0] * len(out))).astype(float)
    return out[["timestamp_ms", "side", "price", "size"]]


def _build_book_snapshots_multi(
    start_str: str, end_str: str,
) -> dict[int, tuple[list, list]]:
    """
    Build 1-second book snapshots incrementally across multiple days.
    OrderBookReconstructor state persists across day boundaries.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    recon = OrderBookReconstructor()
    snapshots: dict[int, tuple[list, list]] = {}
    total_rows = 0

    d = start
    day_idx = 0
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "book.parquet"
        if not fpath.exists():
            print(f"    {date_str}: no book file, skipping", flush=True)
            d += timedelta(days=1)
            continue

        df = pd.read_parquet(fpath)
        if day_idx == 0:
            print(f"    [debug] book columns: {list(df.columns)}")
            print(f"    [debug] event_type values: {df['event_type'].value_counts().to_dict()}")
        day_idx += 1

        # Process via numpy from Arrow batch (no pandas, minimal memory)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(fpath)
        day_rows = 0
        bucket_rows: list[dict] = []
        current_sec = -1

        for batch in pf.iter_batches(batch_size=200_000):
            ev = batch.column("event_time").to_numpy()
            sd = batch.column("side").to_pylist()
            px = batch.column("price").to_pylist()
            qt = batch.column("quantity").to_pylist()
            m = len(ev)

            for i in range(m):
                sec = int(ev[i]) // 1000

                if sec != current_sec and bucket_rows:
                    recon.clear()
                    for r in bucket_rows:
                        recon.apply(side=r["side"], price=r["price"], quantity=r["quantity"])
                    key = current_sec * 1000
                    if key not in snapshots:
                        snapshots[key] = recon.top_n(20)
                    bucket_rows.clear()

                current_sec = sec
                bucket_rows.append({
                    "side": sd[i],
                    "price": float(px[i]),
                    "quantity": float(qt[i]),
                })

            day_rows += m

        # Final second
        if bucket_rows:
            recon.clear()
            for r in bucket_rows:
                recon.apply(side=r["side"], price=r["price"], quantity=r["quantity"])
            key = current_sec * 1000
            if key not in snapshots:
                snapshots[key] = recon.top_n(20)

        total_rows += day_rows
        print(f"    {date_str}: {day_rows:,} rows -> {len(snapshots):,} snapshots",
              flush=True)
        d += timedelta(days=1)

    print(f"  Total: {total_rows:,} book rows -> {len(snapshots):,} snapshots",
          flush=True)
    return snapshots


def _load_trades_multi(start_str: str, end_str: str) -> pd.DataFrame:
    """Load and concatenate trades from all days."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "trades.parquet"
        if fpath.exists():
            df = _prepare_trades(pd.read_parquet(fpath))
            chunks.append(df)
        d += timedelta(days=1)

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def _load_liq_multi(start_str: str, end_str: str) -> pd.DataFrame:
    """Load and concatenate liquidations from all days."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []

    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        fpath = RAW_DIR / date_str / "liq.parquet"
        if fpath.exists():
            df = _prepare_liq(pd.read_parquet(fpath))
            if not df.empty:
                chunks.append(df)
        d += timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["timestamp_ms", "side", "price", "size"])
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_str: str, end_str: str) -> None:
    print(f"Multi-day sweep: {start_str} -> {end_str}")
    print(f"  Windows: {WINDOW_SIZES_SEC}")
    print(f"  Horizons: {HORIZONS_SEC}")

    # --- Load trades ---
    print("Loading trades ...", flush=True)
    trades_df = _load_trades_multi(start_str, end_str)
    print(f"  Trades: {len(trades_df):,} rows")
    if trades_df.empty:
        print("ERROR: No trade data found.")
        sys.exit(1)

    # --- Load liquidations ---
    print("Loading liquidations ...", flush=True)
    liq_df = _load_liq_multi(start_str, end_str)
    print(f"  Liquidations: {len(liq_df):,} rows")

    # --- Build book snapshots incrementally ---
    print("Building book snapshots ...", flush=True)
    book_snapshots = _build_book_snapshots_multi(start_str, end_str)

    # --- Compute globals ---
    rolling_avg_volume = float(trades_df["size"].sum() / max(len(trades_df), 1))
    _24h_stats = {
        "_24h_avg_range": float(trades_df["price"].max() - trades_df["price"].min()),
        "_24h_low": float(trades_df["price"].min()),
        "_24h_high": float(trades_df["price"].max()),
    }
    print(f"  Rolling avg volume: {rolling_avg_volume:.4f}")
    print(f"  Range: {_24h_stats['_24h_low']:.2f} - {_24h_stats['_24h_high']:.2f}")
    print("  Sweeping ...", flush=True)

    # --- Sweep ---
    sweeper = GridSweeper(window_sizes_sec=WINDOW_SIZES_SEC,
                          horizons_sec=HORIZONS_SEC)
    gen = sweeper.sweep(
        trades_df=trades_df,
        book_snapshots=book_snapshots,
        liq_df=liq_df,
        rolling_avg_volume=rolling_avg_volume,
        _24h_stats=_24h_stats,
    )

    def _progress(iterator):
        n = 0
        for row in iterator:
            yield row
            n += 1
            if n % PROGRESS_EVERY == 0:
                print(f"    Processed {n:,} windows ...", flush=True)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GridSweeper.save_to_disk(_progress(gen), str(OUTPUT_FILE))

    result = pd.read_parquet(OUTPUT_FILE)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Done.")
    print(f"  File:  {OUTPUT_FILE.resolve()}")
    print(f"  Size:  {size_mb:.2f} MB")
    print(f"  Rows:  {len(result):,}")
    print()
    print(result.head(3))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

# --- verify_dataset.py ---
"""
verify_dataset.py — Sanity-check the research dataset for nulls, leakage,
desync, and class balance before model training.

Usage::

    python verify_dataset.py [data/research_dataset.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag(msg: str) -> None:
    print(f"  !!  {msg}")


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(filepath: str) -> None:
    print(f"Loading {filepath} ...")
    df = pd.read_parquet(filepath)
    print(f"  Rows: {len(df):,}   Cols: {len(df.columns)}")
    print()

    anomalies = 0
    label_cols = {"outcome_pct", "outcome_binary"}
    meta_cols = {"window_size", "horizon", "window_end_ms"}
    feature_cols = [c for c in df.columns if c not in label_cols and c not in meta_cols]

    # ==================================================================
    # 1. NULLS / INF
    # ==================================================================
    print("=" * 60)
    print("1. NULLS & INFINITIES")
    print("=" * 60)

    for col in df.columns:
        n_null = int(df[col].isna().sum())
        n_inf = int(np.isinf(df[col].values).sum()) if df[col].dtype.kind in "fc" else 0
        if n_null > 0 or n_inf > 0:
            _flag(f"{col}: {n_null} nulls, {n_inf} infs")
            anomalies += 1

    if anomalies == 0:
        _ok("No nulls or infinities found.")

    # ==================================================================
    # 2. LABEL LEAKAGE
    # ==================================================================
    print()
    print("=" * 60)
    print("2. LABEL LEAKAGE CHECKS")
    print("=" * 60)

    # 2a. outcome_binary in {0, 1}
    bad_binary = df[~df["outcome_binary"].isin([0.0, 1.0])]
    if len(bad_binary) > 0:
        _flag(f"outcome_binary has {len(bad_binary)} rows not in {{0, 1}}")
        anomalies += 1
    else:
        _ok("outcome_binary is strictly 0 or 1")

    # 2b. Sign consistency
    sign_mismatch = df[
        ((df["outcome_pct"] > 0) & (df["outcome_binary"] != 1))
        | ((df["outcome_pct"] <= 0) & (df["outcome_binary"] != 0))
    ]
    if len(sign_mismatch) > 0:
        _flag(f"outcome_pct / outcome_binary sign mismatch: {len(sign_mismatch)} rows")
        anomalies += 1
    else:
        _ok("outcome_pct sign matches outcome_binary in all rows")

    # 2c. Feature correlation with labels
    print("  Checking feature-label correlations ...")
    high_corr = []
    for col in feature_cols:
        if df[col].dtype.kind not in "fc":
            continue
        if df[col].nunique() <= 1:
            continue
        corr_pct = abs(df[col].corr(df["outcome_pct"]))
        corr_bin = abs(df[col].corr(df["outcome_binary"]))
        if corr_pct > 0.95:
            high_corr.append((col, "outcome_pct", corr_pct))
        if corr_bin > 0.95:
            high_corr.append((col, "outcome_binary", corr_bin))

    if high_corr:
        for col, target, val in high_corr:
            _flag(f"{col} correlated with {target} at r={val:.4f} - possible leakage!")
        anomalies += len(high_corr)
    else:
        _ok("No feature has >0.95 correlation with outcome_pct or outcome_binary")

    # ==================================================================
    # 3. DATA SANITY
    # ==================================================================
    print()
    print("=" * 60)
    print("3. DATA SANITY")
    print("=" * 60)

    # 3a. Monotonic window_end_ms
    groups = df.groupby(["window_size", "horizon"])
    non_mono = 0
    for (ws, hz), grp in groups:
        ts = grp["window_end_ms"].values
        if not (ts[1:] >= ts[:-1]).all():
            _flag(f"window_size={ws}, horizon={hz}: window_end_ms NOT monotonic")
            non_mono += 1
    if non_mono == 0:
        _ok("window_end_ms is monotonic within every (window_size, horizon) group")

    # 3b. No duplicates
    dups = df.duplicated(subset=["window_end_ms", "window_size", "horizon"]).sum()
    if dups > 0:
        _flag(f"{dups} duplicate (window_end_ms, window_size, horizon) rows found")
        anomalies += 1
    else:
        _ok("No duplicate (window_end_ms, window_size, horizon) rows")

    # 3c. Class balance
    print()
    print("  outcome_binary distribution:")
    dist = df["outcome_binary"].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {int(label)}: {count:,}  ({pct:.1f}%)")

    if dist.get(0.0, 0) == 0 or dist.get(1.0, 0) == 0:
        _flag("Single class! No variation in outcome_binary.")
        anomalies += 1
    elif min(dist.get(0.0, 0), dist.get(1.0, 0)) / len(df) < 0.01:
        _flag("Severe class imbalance (<1% minority class)")
        anomalies += 1
    else:
        _ok("Class balance is acceptable")

    # ==================================================================
    # 4. SUMMARY PER (window_size, horizon)
    # ==================================================================
    print()
    print("=" * 60)
    print("4. ROWS PER (window_size, horizon)")
    print("=" * 60)

    summary = (
        df.groupby(["window_size", "horizon"])
        .agg(rows=("outcome_binary", "count"), win_rate=("outcome_binary", "mean"))
        .sort_index()
    )
    for (ws, hz), row in summary.iterrows():
        print(f"  W={ws:>4}s  H={hz:>5}s  ->  {int(row['rows']):>7,} rows  "
              f"win_rate={row['win_rate']:.4f}")

    # ==================================================================
    # FINAL
    # ==================================================================
    print()
    if anomalies == 0:
        print("OK  Dataset passes all checks. Ready for training.")
    else:
        print(f"!!  {anomalies} anomalies found. Review before training.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/research_dataset.parquet"
    main(path)

# --- train_model.py ---
"""
train_model.py — Train LightGBM per (window_size, horizon) and build
the Expectancy Table across probability thresholds.

Chronological split (70/15/15), no shuffling, no scaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE = "data/research_dataset.parquet"
OUTPUT_FILE = "data/expectancy_table.csv"
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
COST_PER_TRADE = 0.001  # 0.1 %

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 500,
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
}

FEATURE_COLS: list[str] = []  # filled after loading
META_COLS = {"window_size", "horizon", "window_end_ms"}
LABEL_COLS = {"outcome_binary", "outcome_pct"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by window_end_ms: first 70% train, next 15% val, last 15% test."""
    df = df.sort_values("window_end_ms").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _evaluate(
    model: lgb.Booster, test: pd.DataFrame, window_size: int, horizon: int
) -> list[dict]:
    """Evaluate one model across all thresholds on its (ws, hz) test subset."""
    X_test = test[FEATURE_COLS]
    y_true_bin = test["outcome_binary"].values
    y_true_pct = test["outcome_pct"].values
    n_days = (test["window_end_ms"].max() - test["window_end_ms"].min()) / 86_400_000 + 1

    probs = model.predict(X_test)
    rows = []

    for thresh in THRESHOLDS:
        signals = probs >= thresh
        n_signals = int(signals.sum())
        if n_signals == 0:
            rows.append({
                "window_size": window_size, "horizon": horizon, "threshold": thresh,
                "signals_per_day": 0.0, "n_signals": 0, "n_test": len(test),
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
            })
            continue

        sig_bin = y_true_bin[signals]
        sig_pct = y_true_pct[signals]
        wins = sig_bin == 1
        n_wins = int(wins.sum())
        n_losses = n_signals - n_wins

        win_rate = n_wins / n_signals
        avg_win = float(sig_pct[wins].mean()) if n_wins > 0 else 0.0
        avg_loss = float(np.abs(sig_pct[~wins]).mean()) if n_losses > 0 else 0.0
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) - COST_PER_TRADE

        rows.append({
            "window_size": window_size, "horizon": horizon, "threshold": thresh,
            "signals_per_day": n_signals / max(n_days, 1),
            "n_signals": n_signals, "n_test": len(test),
            "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "expectancy": expectancy,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns if c not in META_COLS and c not in LABEL_COLS]
    print(f"  {len(FEATURE_COLS)} features")
    print()

    # Chronological split
    print("Chronological split (70/15/15) ...")
    train_all, val_all, test_all = _chronological_split(df)
    print(f"  Train: {len(train_all):,}  Val: {len(val_all):,}  Test: {len(test_all):,}")
    print()

    # Train one model per (window_size, horizon)
    pairs = sorted(df[["window_size", "horizon"]].drop_duplicates().values.tolist())
    all_rows: list[dict] = []

    for i, (ws, hz) in enumerate(pairs):
        ws, hz = int(ws), int(hz)
        print(f"[{i + 1}/{len(pairs)}] W={ws}s  H={hz}s ...", end=" ", flush=True)

        train = train_all[(train_all["window_size"] == ws) & (train_all["horizon"] == hz)]
        val = val_all[(val_all["window_size"] == ws) & (val_all["horizon"] == hz)]
        test = test_all[(test_all["window_size"] == ws) & (test_all["horizon"] == hz)]

        if len(train) < 100 or len(val) < 50:
            print(f"SKIP (train={len(train)}, val={len(val)} - too few)")
            continue

        X_train, y_train = train[FEATURE_COLS], train["outcome_binary"]
        X_val, y_val = val[FEATURE_COLS], val["outcome_binary"]

        model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=1000)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        rows = _evaluate(model.booster_, test, ws, hz)
        all_rows.extend(rows)

        best = max(rows, key=lambda r: r["expectancy"])
        print(f"best th={best['threshold']:.2f} exp={best['expectancy']:.6f} "
              f"wr={best['win_rate']:.4f} sig/day={best['signals_per_day']:.1f}")

    # Save
    table = pd.DataFrame(all_rows)
    table = table.sort_values("expectancy", ascending=False)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(table)} rows to {OUTPUT_FILE}")

    # Top 10
    print("\n=== TOP 10 BY EXPECTANCY ===")
    cols = ["window_size", "horizon", "threshold", "expectancy", "win_rate",
            "signals_per_day", "n_signals", "avg_win", "avg_loss"]
    print(table[cols].head(10).to_string(index=False))

    # Best combo
    best = table.iloc[0]
    print(f"\n=== BEST COMBO ===")
    print(f"  window_size = {int(best['window_size'])}s")
    print(f"  horizon     = {int(best['horizon'])}s")
    print(f"  threshold   = {best['threshold']:.2f}")
    print(f"  expectancy  = {best['expectancy']:.6f}")
    print(f"  win_rate    = {best['win_rate']:.4f}")
    print(f"  signals/day = {best['signals_per_day']:.1f}")


if __name__ == "__main__":
    main()