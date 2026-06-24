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
