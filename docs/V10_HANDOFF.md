# OFP V10 Handoff — 2026-06-26

> **Use this doc to onboard any new AI session (or human) onto the OFP project.**
> It's the single source of truth for the current state and next steps.

## V10 headline result

- **OOS expectancy: +0.82%** (was +0.35% in v9, +135%)
- OOS win rate: 77.0% (was 68.7%)
- OOS signals/day: 87 (was 135)
- OOS avg win: 1.07% (was 0.67%)
- Negative-spread rows: 7.7% → ~0%
- W=60s, H=3600s, θ=0.6
- Sweep: 2026-06-17 → 2026-06-23 (7 days, 146,166 rows)

## What the fix was

**Root cause:** Orderbook levels whose holder never sends a delete (qty=0) and never updates them again sit in the book FOREVER. When the market moves 1%+ away, they become "ghost" levels (bid > ask → crossed book).

In v9, a bid at 63011.35 sat in the book for **6.22 hours** (last update 04:00, snapshot 09:13) because the placer never cancelled. Market moved down $675, ghost bid $675 above the new ask. Spread = -108 bps.

**Fix:** track per-level last-update time, call `recon.evict_stale(current_ms, max_age_ms=30_000)` before every 1-second snapshot. Any level not updated in 30s gets dropped.

**Files changed (committed on Hetzner, not yet pushed):**
- `ofp/book_reconstructor.py` — added `evict_stale` + `_bids_ts` / `_asks_ts` tracking
- `run_research.py` — calls `recon.evict_stale(...)` before every 1s snapshot
- `tests/test_book_reconstructor.py` — 7 new tests
- `docs/cryptoHFTData_sdk_reference.md` — SDK reference
- `docs/orderbook_data_audit.md` — V9 audit + V10 verification
- `docs/oos_v10_verification.md` — V10 headline results
- `src/data/pull_historical.py` — new pull module (isolated, not used by engine)
- `src/__init__.py`, `src/data/__init__.py`
- `final_verify.py` — verification helper

## Server (as of 2026-06-26)

- **Azure (CURRENT)**: `azureuser@172.178.80.203` (62GB RAM, 122GB disk, Ubuntu 24.04, 16 cores)
- SSH key: `C:\Users\User\Desktop\ofp.pem` (ed25519)
- SSH from Windows: `ssh -i C:\Users\User\Desktop\ofp.pem azureuser@172.178.80.203`
- Project path: `/home/azureuser/ofp/`
- Venv: `.venv/bin/python` (Python 3.12.3, lightgbm 4.6.0, all deps installed)
- V10 fix committed on Azure: `f6bfcbd` (was missing, now restored)
- **Hetzner is OFF** (out of credits). Do NOT use Hetzner paths in any new work.
- 90/90 tests pass on Azure

## Project summary

- Crypto microstructure edge discovery, BTCUSDT perpetuals (Binance Futures via cryptohftdata v0.3.0)
- 108 features × 3 zooms (micro 60-180s, meso 300s, macro 1800s) per `feature_extractor.py`
- LightGBM binary classifier (outcome = +1% target hit before -1% stop in 24h)
- 7-day sweep produces ~146k rows
- Sweep time: ~90 min on Hetzner (30 min book build + 60 min sweep)

## Engine architecture

- `ofp/feature_extractor.py` — extracts 36 features per zoom (108 total)
- `ofp/book_reconstructor.py` — L2 orderbook maintainer (SortedDict, NOW with `evict_stale`)
- `ofp/grid_sweeper.py` — slides windows, calls feature extractor, yields rows
- `run_research.py` — main sweep entry point, calls evict_stale every 1s
- `train_model.py` — per-pair chronological split (70/15/15), LightGBM
- `relabel_target.py` — target-based labels (hit +1% before -1% in 24h)
- `tests/` — **90 tests passing** (83 original + 7 new eviction tests)

## Key bug fixes (v1 → v10)

- v1-v6: context leak (per-zoom rolling stats), broken book reconstruction (running state), _cum_at O(n²), per-zoom prior-24h, monotonic deque
- v7: `relabel_target.py` (target-based labels instead of time-based)
- v8: `train_model.py` min_child_samples 500→50
- v9: 7-day sweep runs successfully
- **v10: stale-level eviction** (THE big win)

## CryptoHFTData SDK notes

- v0.3.0, all 6 data types: trades, orderbook, liquidations, mark_price, open_interest, ticker
- All numeric fields come as `str` (pandas 2.x `str` dtype, NOT `object`) — must use `pd.api.types.is_string_dtype` check, not `dtype == object`
- Timestamps: event_time = ms UTC, received_time = ns UTC (latency)
- start_date/end_date are DAY-granular (time-of-day ignored)
- Orderbook for 1 day = ~140M rows = ~30GB pandas RSS (OOMs on 32GB hosts)
- Full SDK reference: `docs/cryptoHFTData_sdk_reference.md`

## Data path (NOT yet integrated with engine)

- `src/data/pull_historical.py` pulls all 6 types to `data/historical/<type>/<symbol>/YYYY-MM-DD.parquet.zst`
- Engine still uses old `download_raw_data.py` writing to `data/raw/YYYY-MM-DD/`
- New module is isolated — engine untouched
- Funding rate + OI + ticker are NEW data types available for feature engineering

## What the trader wants

- Macrostructure holds (15min-4hr), NOT tick-level racing
- Method: chart pattern TRIGGER + orderflow CONFIRM
- Bad news first, simple terms
- ~EUR500 real account on Hyperliquid (HL)
- 10y discretionary macrostructure trader, NOT a coder

## Open leads (lessons preserved)

- All prior leads CLOSED (Lagshot, OI-cascade, sweep, VP LVN, ladder)
- Depth-imbalance confirm FAILED as orderflow confirm in EVERY study
- OOS result is real (+0.82% expectancy on 87 sigs/day) but needs:
  1. Longer OOS validation (60+ days)
  2. Trader-entry trace on v10 dataset (re-run `_trace_target.py`)
  3. Walk-forward test (train on rolling windows, not static 70/15/15)

## Hard rules (must follow)

- **Engine = SACRED** (do not touch without explicit sign-off)
- **Null-edge gate:** coinflip must LOSE ~fees; if it profits, STOP and fix
- **Knob-bite rule:** "no edge" only counts if the swept param moved trade behavior
- **Always report FILL-RATE + per-euro SIZE-WEIGHTED EUR**, never per-trade equal-weight for ladders
- **A 10-day separator MUST be re-checked at 60d+/OOS before trust**
- **Commit small**; changes touching fills/P&L carry a test catching their failure mode
- **Listen to trader frustration as a signal** (caught the liq case bug)

## Next steps (on Azure, can start immediately)

1. **Download data to Azure** (Hetzner data is GONE): `download_raw_data.py 2026-06-17 2026-06-23` (~30 min for 7 days)
2. Run sweep on Azure: `nohup .venv/bin/python -u run_research.py 2026-06-17 2026-06-23 > /tmp/sweep_v10_azure.log 2>&1 &` (~90 min)
3. Run `_trace_target.py` against v10 dataset — check if direction-match improved from 60% (v9)
4. Sweep 60+ days for OOS validation
5. Add funding rate + OI features (data is already there in `data/historical/`)
6. Walk-forward test (rolling retrain)
7. Look at combining with chart pattern trigger (the trader's method)

## Team structure

- **Trader**: direction, priorities, final call. 10y discretionary macrostructure trader, NOT a coder. Prefers 15min-4hr holds.
- **Copilot (GLM-5.1)**: engine architecture, Rust code, honesty gates, null-edge validation, documentation. The skeptic.
- **DeepSeek V4 Pro**: data analysis, feature prototyping, ML/statistical work, Python scripts, number crunching.
- **M3 (this AI)**: supporter AI for the OFP project, picked up after GLM 5.2 lost context on 2026-06-26

## Permanent lessons (apply forever)

- Per-trade can LIE (ladder/grid/DCA: losers are bigger than winners). Always judge size-weighted.
- Resting-depth imbalance is NOT predictive (failed 4+ times as orderflow confirm).
- Latency-vol correlation: latency blows out exactly when signals fire (Lagshot live: 766ms calm → 1.3-2.4s at triggers).
- Reversion half-life ≈ HL's consensus floor (~879ms server-side); retail taker structurally below it.
- The ~9bps taker fee is a structural wall for microstructure-bps edges. Next edge must clear >>9bps per trade.
- Exhaustion/flow-deceleration is a REAL directional read (+15-30pp P(reversal), both assets).
- Structural wick-stop fixes stop-bleed (dynamic beyond the sweep extreme).
- Reclaim entry is a real capture improvement (but not an edge by itself).
- OKX is the best reference venue (strictly better than Binance for basis reversion).
- **Stale-level bug** (v10): L2 orderbook reconstructors must evict levels with no recent updates; otherwise ghost levels persist for hours and produce crossed-book features.
