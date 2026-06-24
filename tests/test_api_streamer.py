"""
test_api_streamer.py — Tests for the CryptoHFTStreamer using mocked HTTP.

No real API calls are made.  httpx is mocked via monkey-patching.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import httpx

from ofp.api_streamer import (
    CryptoHFTStreamer,
    CryptoHFTAPIError,
    CryptoHFTTimeoutError,
)
from ofp.data_schema import TradesData, BookSnapshotData, LiquidationData


# ---------------------------------------------------------------------------
# Dummy payloads (matching CryptoHFTData JSON wire format)
# ---------------------------------------------------------------------------

VALID_TRADES_PAYLOAD = [
    {
        "received_time": 1719000001000000000,
        "event_time": 1719000000900000000,
        "symbol": "BTCUSDT",
        "trade_id": 123456789,
        "price": "68432.50",
        "quantity": "0.015",
        "trade_time": 1719000000800000000,
        "is_buyer_maker": False,
        "order_type": "LIMIT",
    },
    {
        "received_time": 1719000002000000000,
        "event_time": 1719000001900000000,
        "symbol": "BTCUSDT",
        "trade_id": 123456790,
        "price": "68433.00",
        "quantity": "0.320",
        "trade_time": 1719000001800000000,
        "is_buyer_maker": True,
        "order_type": "MARKET",
    },
]

VALID_BOOK_PAYLOAD = [
    {
        "received_time": 1719000001000000000,
        "event_time": 1719000000900000000,
        "symbol": "BTCUSDT",
        "event_type": "update",
        "side": "bid",
        "price": "68400.00",
        "quantity": "1.500",
    },
    {
        "received_time": 1719000001000000000,
        "event_time": 1719000000900000000,
        "symbol": "BTCUSDT",
        "event_type": "update",
        "side": "ask",
        "price": "68450.00",
        "quantity": "0.000",
    },
]

VALID_LIQUIDATION_PAYLOAD = [
    {
        "received_time": 1719000001000000000,
        "event_time": 1719000000900000000,
        "symbol": "BTCUSDT",
        "side": "SELL",
        "price": "68200.00",
        "quantity": "2.400",
        "trade_time": 1719000000800000000,
    },
]


# ---------------------------------------------------------------------------
# Minimal mock for httpx.Response
# ---------------------------------------------------------------------------

class MockResponse:
    """Minimal httpx.Response stand-in for testing."""

    def __init__(self, status_code: int, json_body: object) -> None:
        self.status_code = status_code
        self._json = json_body

    def json(self) -> object:
        return self._json

    @property
    def text(self) -> str:
        if isinstance(self._json, (list, dict)):
            return json.dumps(self._json)
        return str(self._json)


# ---------------------------------------------------------------------------
# Pydantic schema unit tests
# ---------------------------------------------------------------------------

class TestTradesData:
    def test_valid_trade_parsing(self) -> None:
        record = VALID_TRADES_PAYLOAD[0]
        trade = TradesData(**record)
        assert trade.symbol == "BTCUSDT"
        assert trade.price == 68432.50
        assert trade.quantity == 0.015
        assert trade.is_buyer_maker is False
        assert trade.trade_id == 123456789
        assert isinstance(trade.received_time, int)

    def test_is_buyer_maker_semantics(self) -> None:
        """True = aggressor is SELLER, False = aggressor is BUYER."""
        buyer_is_taker = TradesData(**VALID_TRADES_PAYLOAD[0])
        seller_is_taker = TradesData(**VALID_TRADES_PAYLOAD[1])
        assert buyer_is_taker.is_buyer_maker is False  # buyer aggressed
        assert seller_is_taker.is_buyer_maker is True   # seller aggressed

    def test_price_quantity_string_coercion(self) -> None:
        trade = TradesData(
            received_time=1,
            event_time=1,
            symbol="BTCUSDT",
            trade_id=1,
            price="68500.123",
            quantity="0.001",
            trade_time=1,
            is_buyer_maker=False,
            order_type="LIMIT",
        )
        assert isinstance(trade.price, float)
        assert isinstance(trade.quantity, float)
        assert trade.price == 68500.123

    def test_missing_field_raises(self) -> None:
        bad = dict(VALID_TRADES_PAYLOAD[0])
        del bad["price"]
        with pytest.raises(Exception):  # Pydantic ValidationError
            TradesData(**bad)

    def test_bad_price_raises(self) -> None:
        bad = dict(VALID_TRADES_PAYLOAD[0])
        bad["price"] = "not_a_number"
        with pytest.raises(Exception):
            TradesData(**bad)

    def test_to_dataframe(self) -> None:
        df = TradesData.to_dataframe(VALID_TRADES_PAYLOAD)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            "received_time", "event_time", "symbol", "trade_id",
            "price", "quantity", "trade_time", "is_buyer_maker", "order_type",
        ]
        assert df["price"].dtype == "float64"
        assert df["quantity"].dtype == "float64"
        assert df["is_buyer_maker"].dtype == "bool"


class TestBookSnapshotData:
    def test_valid_snapshot_parsing(self) -> None:
        record = VALID_BOOK_PAYLOAD[0]
        snap = BookSnapshotData(**record)
        assert snap.event_type == "update"
        assert snap.side == "bid"
        assert snap.price == 68400.00
        assert snap.quantity == 1.500

    def test_zero_quantity_update(self) -> None:
        """Zero quantity = level removed (valid use case)."""
        snap = BookSnapshotData(**VALID_BOOK_PAYLOAD[1])
        assert snap.quantity == 0.0
        assert snap.side == "ask"

    def test_to_dataframe(self) -> None:
        df = BookSnapshotData.to_dataframe(VALID_BOOK_PAYLOAD)
        assert len(df) == 2
        assert df["side"].tolist() == ["bid", "ask"]


class TestLiquidationData:
    def test_valid_liquidation_parsing(self) -> None:
        record = VALID_LIQUIDATION_PAYLOAD[0]
        liq = LiquidationData(**record)
        assert liq.side == "SELL"  # long liquidated
        assert liq.price == 68200.00
        assert liq.quantity == 2.400

    def test_to_dataframe(self) -> None:
        df = LiquidationData.to_dataframe(VALID_LIQUIDATION_PAYLOAD)
        assert len(df) == 1
        assert df["side"].iloc[0] == "SELL"


# ---------------------------------------------------------------------------
# Streamer fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def streamer() -> CryptoHFTStreamer:
    return CryptoHFTStreamer(base_url="http://fake.test/v1")


# ---------------------------------------------------------------------------
# Streamer tests (mocked HTTP)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_trades_yields_dataframe(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Happy path: one page of trades → one DataFrame yielded."""
    called_params: list[dict] = []

    async def fake_get(url: str, params: dict) -> MockResponse:
        called_params.append(dict(params))
        return MockResponse(200, VALID_TRADES_PAYLOAD)

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="trades",
        start_time=1719000000000000000, end_time=1719100000000000000,
    ):
        dfs.append(df)

    assert len(dfs) == 1
    df = dfs[0]
    assert len(df) == 2
    assert df["symbol"].iloc[0] == "BTCUSDT"
    assert df["price"].iloc[0] == 68432.50
    assert "is_buyer_maker" in df.columns
    assert called_params[0]["symbol"] == "BTCUSDT"
    assert called_params[0]["limit"] == 10000
    assert called_params[0]["offset"] == 0


@pytest.mark.asyncio
async def test_fetch_multiple_pages(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Ensure pagination yields multiple DataFrames and stops correctly."""
    pages = [
        VALID_TRADES_PAYLOAD * 5000,  # 10_000 rows (full page)
        VALID_TRADES_PAYLOAD * 5000,  # another full page
        VALID_TRADES_PAYLOAD[:1],     # short final page (1 row)
    ]
    call_count = [0]
    streamer._chunk_size = 10_000

    async def fake_get(url: str, params: dict) -> MockResponse:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(pages):
            return MockResponse(200, pages[idx])
        return MockResponse(200, [])

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="trades",
        start_time=1, end_time=9999999999999999999,
    ):
        dfs.append(df)

    assert len(dfs) == 3
    assert len(dfs[0]) == 10_000
    assert len(dfs[1]) == 10_000
    assert len(dfs[2]) == 1


@pytest.mark.asyncio
async def test_empty_response(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Empty list → yields nothing."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(200, [])

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="trades",
        start_time=1, end_time=999,
    ):
        dfs.append(df)

    assert len(dfs) == 0


@pytest.mark.asyncio
async def test_non_list_response_raises(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """API returns a dict instead of a list → CryptoHFTAPIError."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(200, {"error": "bad"})

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    with pytest.raises(CryptoHFTAPIError, match="Expected a JSON array"):
        async for _ in streamer.fetch_data(
            symbol="BTCUSDT", data_type="trades",
            start_time=1, end_time=999,
        ):
            pass


@pytest.mark.asyncio
async def test_malformed_record_raises_validation_error(
    streamer: CryptoHFTStreamer, monkeypatch,
) -> None:
    """A record missing a required field → Pydantic ValidationError bubbles up."""
    bad_record = dict(VALID_TRADES_PAYLOAD[0])
    del bad_record["price"]

    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(200, [bad_record])

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    with pytest.raises(Exception):
        async for _ in streamer.fetch_data(
            symbol="BTCUSDT", data_type="trades",
            start_time=1, end_time=999,
        ):
            pass


@pytest.mark.asyncio
async def test_non_retryable_4xx_raises(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """A 400-level error (non-retryable) → immediate CryptoHFTAPIError."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(400, {"msg": "bad request"})

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    with pytest.raises(CryptoHFTAPIError, match="Non-retryable"):
        async for _ in streamer.fetch_data(
            symbol="BTCUSDT", data_type="trades",
            start_time=1, end_time=999,
        ):
            pass


@pytest.mark.asyncio
async def test_retry_on_5xx_then_succeed(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Retryable status → retries → eventually succeeds."""
    call_count = [0]

    async def fake_get(url: str, params: dict) -> MockResponse:
        call_count[0] += 1
        if call_count[0] < 3:
            return MockResponse(503, "Service Unavailable")
        return MockResponse(200, VALID_TRADES_PAYLOAD)

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="trades",
        start_time=1, end_time=999,
    ):
        dfs.append(df)

    assert len(dfs) == 1
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_retry_exhausted_raises(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """All retries exhausted on 5xx → CryptoHFTTimeoutError."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(503, "Service Unavailable")

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    with pytest.raises(CryptoHFTTimeoutError, match="retries exhausted"):
        async for _ in streamer.fetch_data(
            symbol="BTCUSDT", data_type="trades",
            start_time=1, end_time=999,
        ):
            pass


@pytest.mark.asyncio
async def test_invalid_time_range_raises(streamer: CryptoHFTStreamer) -> None:
    """start_time >= end_time → ValueError."""
    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    with pytest.raises(ValueError, match="start_time"):
        async for _ in streamer.fetch_data(
            symbol="BTCUSDT", data_type="trades",
            start_time=100, end_time=50,
        ):
            pass


@pytest.mark.asyncio
async def test_book_snapshot_data_type(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Ensure book_snapshot data_type uses BookSnapshotData schema."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(200, VALID_BOOK_PAYLOAD)

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="book_snapshot",
        start_time=1, end_time=999,
    ):
        dfs.append(df)

    assert len(dfs) == 1
    assert "event_type" in dfs[0].columns
    assert "side" in dfs[0].columns


@pytest.mark.asyncio
async def test_liquidation_data_type(streamer: CryptoHFTStreamer, monkeypatch) -> None:
    """Ensure liquidations data_type uses LiquidationData schema."""
    async def fake_get(url: str, params: dict) -> MockResponse:
        return MockResponse(200, VALID_LIQUIDATION_PAYLOAD)

    streamer._client = httpx.AsyncClient(base_url="http://fake.test/v1")
    monkeypatch.setattr(streamer._client, "get", fake_get)

    dfs = []
    async for df in streamer.fetch_data(
        symbol="BTCUSDT", data_type="liquidations",
        start_time=1, end_time=999,
    ):
        dfs.append(df)

    assert len(dfs) == 1
    assert "side" in dfs[0].columns
    assert dfs[0]["side"].iloc[0] == "SELL"
