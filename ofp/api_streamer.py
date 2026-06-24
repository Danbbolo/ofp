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
