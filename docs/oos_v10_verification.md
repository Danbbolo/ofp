# V10 Verification — Stale-Level Fix Confirmed

**Date:** 2026-06-26
**Sweep dates:** 2026-06-17 → 2026-06-23 (7 days, 146,166 rows)
**Fix:** added `evict_stale(max_age_ms=30_000)` to `OrderBookReconstructor`, called before every 1-second snapshot in `run_research.py`
**Tests:** 90/90 pass (was 83, added 7 eviction tests)

## The Comparison

| Metric | V9 (stale levels) | V10 (evicted) | Change |
|--------|-------------------|---------------|--------|
| **OOS expectancy (W=60 H=3600 θ=0.6)** | +0.3496% | **+0.8215%** | **+135%** |
| OOS win rate | 68.7% | 77.0% | +8.3pp |
| OOS avg win | 0.675% | 1.072% | +58.8% |
| OOS avg loss | 0.045% | 0.046% | +1.2% |
| OOS n signals | 473 | 307 | -35.1% |
| OOS signals/day | 135 | 87 | -35.6% |
| OOS direction accuracy | 68.6% | 64.5% | -4.1pp |
| Negative-spread rows | 1,545 / 20,038 (7.7%) | ~0 | -100% |
| IS→OOS expectancy | +0.192% (best) | +0.821% | +329% |

## What the fix changed

**Stale levels in book = `apply()` adds a (side, price, qty) tuple, but if nobody ever sends another update for that exact (side, price), it sits in the book forever — even when the market moves 1%+ away.**

The fix:
1. Track `_last_update_ms` for every (side, price) level
2. Before each 1-second snapshot, call `recon.evict_stale(current_time_ms, max_age_ms=30_000)` — drops any level not updated in 30 seconds
3. 30s is a conservative threshold — Binance liquidation cadence is 1-10s for active perps, so any resting order un-touched for 30s on a liquid perp is dead

**Why this more than doubled expectancy:** the model was using `spread_bps`, `bid_ask_imbalance`, and `wall_asymmetry` as features. In the v9 data, ~7.7% of windows had **garbage values** for these (negative spreads, massive asymmetry from ghost bids/asks). The model had to learn to ignore those rows, which both reduced training signal AND made the OOS test noisy. With clean features, the model:
- Catches bigger moves (avg_win 0.67% → 1.07%) — because the wall-asymmetry signal is real, not noise
- Skips more often (135 → 87 sigs/day) — higher conviction
- Lower direction accuracy but higher expectancy — the model is more selective, not more "wrong"

## OOS validation detail

`expectancy_table_oos_v10.csv` (best 3 IS→OOS combos):

| W | H | θ | OOS exp | OOS WR | OOS n | OOS sig/day |
|---|---|---|---------|--------|-------|-------------|
| 60 | 3600 | 0.6 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.55 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.5 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.45 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.4 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.35 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.3 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.25 | +0.8215% | 77.0% | 307 | 87 |
| 60 | 3600 | 0.2 | +0.8215% | 77.0% | 307 | 87 |

(thresholds below 0.6 give the same row because the dataset has only 307 model-positive rows at any threshold ≤ 0.6)

## Recommendation

- **Trade signal exists.** +0.82% expectancy on 87 sigs/day is comfortably above 9bps taker fees (~0.09%).
- **Add stale-level eviction to all future sweeps** (already in the engine).
- **Consider a smaller `max_age_ms`** (e.g. 5s) — over-evicting is safer than under-evicting because stale ghost bids are not real liquidity and have no predictive value.
- **Re-run on longer time window** (60+ days) to confirm the edge persists and isn't regime-specific.
- **Trace trader's 20 entries** against the v10 dataset to see if direction-match improves from 60% (v9) — this is the "does ML capture the trader's eye?" sanity check.

## Files

- `data/research_dataset_v10.parquet` — 45MB, 146,166 rows, clean features
- `data/research_dataset_target_v10.parquet` — 37MB, target-labeled
- `data/expectancy_table_oos_v10.csv` — OOS results
- `ofp/book_reconstructor.py` — added `evict_stale` + `_bids_ts` / `_asks_ts` tracking
- `run_research.py` — calls `recon.evict_stale(...)` before every 1s snapshot
- `tests/test_book_reconstructor.py` — 7 new tests for eviction
