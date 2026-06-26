# CryptoHFTData SDK Reference

**SDK version tested:** 0.3.0
**Install:** `pip install cryptohftdata`
**PyPI:** https://pypi.org/project/cryptohftdata/
**Docs:** https://www.cryptohftdata.com/docs

> **Status:** All claims in this doc are verified by inspection of the installed
> package on the OFP project, not guessed from docs.

---

## 1. Configuration

```python
import cryptohftdata as chd
chd.configure_client(api_key="YOUR_KEY")  # required for downloads
```

`api_key` is **required** for all data fetches. The OFP project key is set in
`download_raw_data.py` — use that one or supply your own.

---

## 2. Common signature

All data-fetch functions share the same shape:

```python
get_<data_type>(
    symbol: str,
    exchange: str,         # e.g. chd.exchanges.BINANCE_FUTURES
    start_date: str | datetime,   # ISO format "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
    end_date:   str | datetime,
    **kwargs: Any,         # implementation-specific
) -> pd.DataFrame
```

### Date granularity: **DAY-LEVEL ONLY**

⚠️ **Gotcha:** `start_date` and `end_date` are interpreted as **whole-day
windows**. The SDK ignores time-of-day within a date range — passing
`start_date="2026-06-23T10:00:00"` returns the same data as
`start_date="2026-06-23T00:00:00"`. Both return the full 2026-06-23 day.

If you need sub-day slicing, do it client-side after download.

### Timestamps

All `event_time` / `timestamp` / `trade_time` columns are **milliseconds since
Unix epoch in UTC** (e.g. `1782172800217` = 2026-06-23 00:00:00.217 UTC).

`received_time` is in **nanoseconds** (different from `event_time` — used for
latency analysis, not for time-series).

---

## 3. Supported exchanges (futures)

`chd.exchanges.BINANCE_FUTURES` is what OFP uses. Other relevant ones:
- `BINANCE_FUTURES`, `BINANCE_SPOT`
- `BYBIT_FUTURES`, `BYBIT_SPOT`
- `OKX_FUTURES`, `OKX_SPOT`
- `BITGET_FUTURES`, `BITGET_SPOT`
- `HYPERLIQUID_FUTURES`, `HYPERLIQUID_SPOT`
- `ASTER_FUTURES`, `LIGHTER`

Use `chd.list_exchanges()` and `chd.list_symbols(exchange, data_type)` to enumerate.

---

## 4. The 6 data types (verified)

### 4.1 `get_trades` — aggTrades (per-trade tape)

**Call:** `chd.get_trades(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame`

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | **nanoseconds** (latency) |
| `event_time` | int64 | **ms UTC** (canonical) |
| `symbol` | str | e.g. "BTCUSDT" |
| `trade_id` | int64 | aggTrade ID |
| `price` | **str** | ⚠️ cast to float |
| `quantity` | **str** | ⚠️ cast to float |
| `trade_time` | int64 | **ms UTC** (Binance trade time) |
| `is_buyer_maker` | bool | True = sell-side taker, False = buy-side taker |
| `order_type` | str | usually "MARKET" |

**1h volume (2026-06-23 00:00-01:00 UTC):** ~3.99M rows

---

### 4.2 `get_orderbook` — L2 deltas (depth diff stream)

**Call:** `chd.get_orderbook(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame` of L2 depth-update events (NOT snapshots — apply deltas in order to build the book)

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | ns (latency) |
| `event_time` | int64 | **ms UTC** |
| `transaction_time` | int64 | ms UTC (Binance tx time) |
| `symbol` | str | |
| `event_type` | str | "snapshot" or "update" |
| `first_update_id` | float64 | NaN for snapshots |
| `final_update_id` | int64 | |
| `prev_final_update_id` | float64 | |
| `last_update_id` | float64 | |
| `side` | str | "bid" or "ask" |
| `price` | **str** | ⚠️ cast |
| `quantity` | **str** | ⚠️ cast — 0.0 means price level removed |
| `order_count` | float64 | NaN for many rows |

**1h volume (2026-06-23 00:00-01:00 UTC):** ~141M rows ⚠️ **massive — storage-heavy**

---

### 4.3 `get_liquidations` — forceOrder

**Call:** `chd.get_liquidations(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame`

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | ns |
| `event_time` | int64 | ms UTC |
| `symbol` | str | |
| `side` | str | "BUY" (long liq) or "SELL" (short liq) |
| `order_type` | str | usually "LIMIT" |
| `time_in_force` | str | |
| `quantity` | **str** | ⚠️ cast |
| `price` | **str** | ⚠️ cast — bankruptcy price |
| `average_price` | **str** | ⚠️ cast — actual fill avg |
| `order_status` | str | "FILLED" |
| `last_filled_quantity` | **str** | ⚠️ |
| `filled_quantity` | **str** | ⚠️ |
| `trade_time` | int64 | ms UTC |

**1h volume:** ~1,161 rows (sparse)

---

### 4.4 `get_mark_price` — mark price + funding rate

**Call:** `chd.get_mark_price(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame`

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | ns |
| `event_time` | int64 | ms UTC (1-sec granularity) |
| `symbol` | str | |
| `mark_price` | **str** | ⚠️ cast |
| `index_price` | **str** | ⚠️ cast |
| `estimated_settle_price` | **str** | ⚠️ cast |
| `funding_rate` | **str** | ⚠️ cast (decimal e.g. "0.00004081") |
| `next_funding_time` | int64 | ms UTC (every 8h on Binance) |

**Funding events:** every 8h (00:00, 08:00, 16:00 UTC) — `funding_rate` is
non-zero only on those events; in between it's a small carry.

**1h volume:** ~86K rows

---

### 4.5 `get_open_interest` — OI history

**Call:** `chd.get_open_interest(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame`

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | ns |
| `symbol` | str | |
| `sum_open_interest` | **str** | ⚠️ cast — in contracts (BTC) |
| `sum_open_interest_value` | **str** | ⚠️ cast — in USDT |
| `timestamp` | int64 | ms UTC (5-min granularity on Binance) |

**1h volume:** ~787 rows (5-min cadence)

---

### 4.6 `get_ticker` — bookTicker / 24h stats

**Call:** `chd.get_ticker(symbol, exchange, start_date, end_date)`
**Returns:** `pd.DataFrame`

| Column | dtype | Notes |
|---|---|---|
| `received_time` | int64 | ns |
| `event_time` | int64 | ms UTC (1-sec granularity) |
| `symbol` | str | |
| `price_change` | **str** | ⚠️ 24h delta |
| `price_change_percent` | **str** | ⚠️ % delta |
| `weighted_average_price` | **str** | ⚠️ |
| `last_price` | **str** | ⚠️ |
| `last_quantity` | **str** | ⚠️ |
| `open_price`, `high_price`, `low_price` | **str** | ⚠️ |
| `base_asset_volume`, `quote_asset_volume` | **str** | ⚠️ |
| `statistics_open_time`, `statistics_close_time` | int64 | ms UTC (24h rolling window) |
| `first_trade_id`, `last_trade_id` | int64 | |
| `total_trades` | int64 | |

**1h volume:** ~83K rows

---

## 5. Bonus: `get_klines` (candles)

`chd.get_klines(symbol, exchange, start_date, end_date)` — OHLCV candles
(useful for sanity-checking live price series against book data).

---

## 6. Common gotchas

1. **All numeric fields (except IDs) come as `str`.** You MUST cast with
   `pd.to_numeric(df["price"])` or `df["price"].astype(float)` before any math.
2. **`start_date`/`end_date` are day-granular.** Time-of-day is ignored.
3. **Timestamps:** `event_time`/`trade_time`/`timestamp` are **ms UTC**;
   `received_time` is **ns UTC** (latency only).
4. **`orderbook` is huge** — 141M rows for 1 hour. Plan for ~150GB/day if you
   store full L2 depth. Consider downsampling or storing only top-20.
5. **`get_trades` volume is high too** — ~4M rows/hour. Full day ≈ 100M rows.
6. **Rate limits** are not documented but the SDK does internal
   multi-threading per day. Don't hammer it with parallel manual calls.
7. **No `on_bad_data` parameter** — if a row is malformed it comes through
   as-is. Always validate (e.g. `df["price"] > 0`).
8. **`configure_client(api_key=...)` is global** — set it once at process start.

---

## 7. What OFP already uses

- `download_raw_data.py` — uses trades + orderbook + liquidations for the
  Binance Futures BTCUSDT perpetual. Saves to `data/raw/YYYY-MM-DD/`.
- The orderbook build pipeline (`run_research.py`) uses the depth-update
  deltas to reconstruct 1-second snapshots with a **running book state** that
  resets at day boundaries (book doesn't persist across days).

## 8. What we're adding

- `get_mark_price` for funding-rate signal
- `get_open_interest` for OI-divergence / cascade detection
- `get_ticker` for 24h-rolling baseline context (basis, vol proxy)
- `get_klines` for sanity-checking price series
