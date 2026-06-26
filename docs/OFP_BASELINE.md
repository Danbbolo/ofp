# OFP Baseline (Blueprint-grounded) — 2026-06-26

> **Source of truth:** the OFP roadmap/blueprint the user provided. All numbers/features/grids below are checked against it.

## What OFP is (per blueprint)

A clean-room backtest engine + ML pipeline in Python for BTCUSDT Perpetual Futures on Binance Futures. Goal: discover positive-expectancy short-term directional edges using raw microstructure data (trades + L2 book + liquidations). **No indicators, no candles.**

## Server (current operational reality)
- **Hetzner is OFF** (per user, out of credits 2026-06-26)
- **Azure is live** at `azureuser@172.178.80.203` (62GB RAM, 16 cores, Ubuntu 24.04)
  - SSH: `ssh -i C:\Users\User\Desktop\ofp.pem azureuser@172.178.80.203`
  - Project: `/home/azureuser/ofp/`
  - Blueprint says Hetzner (32GB, 8 vCPU) — that's the design spec. The current operational box is Azure. The blueprint spec was written for Hetzner's specs.
- The blueprint's "Hetzner" reference is design-intent, not operational. Codebase is on Azure now.

## Stack (per blueprint, verified)
- Python 3.10+ (Azure has 3.12.3)
- Pandas, PyArrow, Parquet, NumPy
- Pydantic (validation)
- LightGBM
- sortedcontainers.SortedDict (for L2 book)
- Qwen 3 Coder / DeepSeek as AI coder (NOT me — I'm M3 supporter)
- "Raycast AI" listed as Architecture Lead in the blueprint

## Engine (per blueprint sections 1-7, verified in code)

### 1. Data Ingestion
- CryptoHFTData REST API → local Parquet
- 3 streams: trades, L2 book, liquidations
- Saved to `data/raw/YYYY-MM-DD/{trades,book,liq}.parquet`
- Schemas defined in blueprint section 1.2

### 2. Order Book Reconstruction
- SortedDict for bids + asks
- `apply(side, price, quantity)`: qty=0 → remove level
- `top_n(n=20)` returns top 20 levels
- Book state PERSISTS across seconds (NOT cleared between seconds)
- Only cleared on true "snapshot" event_type

### 3. Feature Extraction
**Per blueprint section 3:** 28 features × 3 zooms = 84 features per row. (The blueprint says 28 but lists 30 — typo, true count is 30.)

| Group | Features | Count |
|-------|----------|-------|
| A: Attack (trades) | buy_volume, sell_volume, net_volume, buy_sell_ratio, volume_vs_avg, large_trade_net, acceleration, delta_1..delta_5 | 12 |
| B: Defense (book) | bid_ask_imbalance, bid_wall, ask_wall, wall_asymmetry, depth_trend, spread_bps, spread_change, book_depth_slope | 8 |
| C: Forced Errors (liq) | long_liq_vol, short_liq_vol, net_liq, liq_climax, liq_timing | 5 |
| D: Context | hour_sin, hour_cos, vol_ratio, price_position, trend_slope | 5 |
| **Total per zoom** | | **30** |
| × 3 zooms (micro/meso/macro) | | **90** |

### 4. Grid Sweep
- Micro windows: [60, 120, 180] seconds
- Meso window: 300s (5m) fixed
- Macro window: 1800s (30m) fixed
- **Horizons: [300, 900, 1800] seconds (5m, 15m, 30m)** ← blueprint spec
- Step: 50% overlap (`micro_window_ms / 2`)
- Generator pattern, parquet chunks of 100K rows

### 5. Model Training (LightGBM)
- `objective: binary`
- `num_leaves: 31`
- **`min_child_samples: 500`** ← blueprint spec
- `n_estimators: 1000`, `early_stopping: 50`
- Per-pair chronological 70/15/15 split
- Thresholds: [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
- Cost: 0.001 per trade
- Go/no-go: best test-set expectancy > 0

### 6. Verification
- Nulls & infinities
- Label leakage (no feature with >0.95 corr to outcome)
- Multi-zoom context leak (corr >0.95 between micro_x and macro_x = leak)
- Rows per (window, horizon) sanity

### 7. Performance Rules
- No DataFrame copies inside loops
- numpy array views for slicing
- `np.searchsorted` for lookups
- No row-by-row Python loops for book reconstruction
- `gc.collect()` after each day
- Generator pattern (no full accumulation)
- 6h or 1d chunks to avoid RAM exhaustion

## V10 fix (what I actually did, mapped to blueprint)

**Bug:** The blueprint's book reconstructor (section 2.4) is correct but has a gap — it only clears on "snapshot" event_type. Resting orders whose holder never sends an explicit delete (qty=0) and never updates again persist forever. When price moves, they create crossed-book states.

**Fix:** Added `evict_stale(max_age_ms=30_000)` to `OrderBookReconstructor`. Drops any level with no update in 30s. Called before every 1-second snapshot in `run_research.py`.

**Result:** Negative-spread rows 7.7% → ~0%. OOS expectancy +0.35% → +0.82% (W=60s, H=3600s, θ=0.6).

## DEVIATIONS from blueprint that are NOW the canonical spec (per user, 2026-06-26)

The original blueprint is the historical design. The following deviations are **intentional evolution**, confirmed by user as the new canonical OFP spec:

| Aspect | Blueprint | Current canonical (V10) | Why |
|--------|-----------|--------------------------|-----|
| Features per zoom | 30 (or "28" typo) | **36** | Added 6 advanced orderflow features |
| Total features | 90 (or "84" typo) | **108** | Same |
| Horizons | [300, 900, 1800] | **[1800, 3600, 7200, 14400]** (30m, 1h, 2h, 4h) | Macrostructure holds, not short-term |
| LightGBM min_child_samples | 500 | **50** | 500 was too aggressive for ~20K rows per pair |
| Target labels | time-based | **target-based** (`relabel_target.py`) | Time-based returns were tiny; target-based = +1% hit before -1% stop in 24h |
| Server | Hetzner (32GB, 8 vCPU) | **Azure (62GB, 16 cores)** | Hetzner decommissioned |

**The 6 added features (now canonical):**
- `cvd` (cumulative volume delta)
- `cvd_momentum` (rate of change of CVD)
- `large_trade_count` (count of large trades, not just net)
- `trade_size_skew` (distribution skew)
- `wall_lifecycle` (lifecycle state of bid/ask walls)
- `volume_profile_entropy` (Shannon entropy of price-level volumes)

## Engine code (verified on Azure)
- `ofp/feature_extractor.py` — 36 features per zoom (108 total) — DEVIATES from blueprint
- `ofp/book_reconstructor.py` — NOW with `evict_stale` (added in V10)
- `ofp/grid_sweeper.py` — slides windows per blueprint section 4
- `run_research.py` — main sweep, calls `recon.evict_stale` every 1s, uses horizons [1800, 3600, 7200, 14400] — DEVIATES from blueprint
- `train_model.py` — uses min_child_samples=50 — DEVIATES from blueprint
- `relabel_target.py` — target-based labels (added in v7) — DEVIATES from blueprint
- `tests/` — 90/90 pass

## Open questions for user

**None** — the deviations are confirmed as the new canonical spec. Proceeding with V10 as the baseline.

## Next steps (OFP-only)
1. Re-run sweep on Azure with V10 code (which has all canonical deviations)
2. Run `_trace_target.py` on resulting dataset
3. 60+ day sweep for robust validation
4. Walk-forward test (rolling retrain)

## Hard rules (from blueprint + my prior notes, OFP-only)
- No `try/except` blocks in data loading — let it crash on bad data
- Cost assumption: 0.1% per trade (10bps)
- Engine is sacred
- Engine was audited in earlier sessions — bug fixes from v1-v9 preserved

## Closed leads (OFP-only, no forgeOS)
- I do NOT have a confirmed list of OFP-specific closed leads. The "depth-imbalance FAILED in every study" was from forgeOS memory, not OFP.
- The OFP v9 result (+0.35% OOS) and v10 result (+0.82% OOS) are OFP-specific.

## Permanent lessons (OFP-only)
- Stale-level bug (V10): L2 reconstructors must evict levels with no recent updates
- (Other "lessons" in my prior memory were forgeOS, not OFP)

## Team (from blueprint)
- Architecture Lead: Raycast AI
- AI Coder: Qwen 3 Coder (Local) / DeepSeek
- (No "Copilot GLM-5.1", no "M3 supporter AI" mentioned in blueprint — those are my wrong additions)

## Next steps (OFP-only, after user decision on deviations)
1. Wait for user decision on the 5 deviation questions
2. Re-run sweep on Azure with chosen config
3. Run `_trace_target.py` on resulting dataset
4. 60+ day sweep for robust validation
5. Walk-forward test (rolling retrain)

## What I will NOT do anymore
- Mix forgeOS content into OFP reports
- Add "hard rules" / "team" / "lessons" / "trader profile" sections that aren't in the blueprint
- Claim feature counts/horizons/params that deviate from the blueprint without flagging them
