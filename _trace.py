"""
Trace trader's entries through the research dataset to see if features flagged
each move BEFORE it happened.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

INPUT = Path("data/research_dataset_relabel.parquet")
OUT_TXT = Path("data/trace_report.txt")

# Trader's entries: (day, HH:MM, direction)
# Times assumed UTC (server logs)
ENTRIES = [
    # Day 17 (2026-06-17)
    ("2026-06-17", "04:30", "sell"),
    ("2026-06-17", "13:00", "buy"),
    ("2026-06-17", "18:15", "sell"),
    # Day 18 (2026-06-18)
    ("2026-06-18", "00:00", "buy"),
    ("2026-06-18", "04:30", "sell"),
    ("2026-06-18", "07:00", "buy"),
    ("2026-06-18", "10:15", "sell"),
    ("2026-06-18", "19:15", "buy"),
    ("2026-06-18", "22:45", "sell"),
    # Day 19 (2026-06-19)
    ("2026-06-19", "06:45", "buy"),
    ("2026-06-19", "18:00", "sell"),
    ("2026-06-19", "20:30", "buy"),
    # Day 20 (2026-06-20)
    ("2026-06-20", "03:30", "sell"),
    ("2026-06-20", "16:00", "buy"),
    # Day 21 (2026-06-21)
    ("2026-06-21", "06:30", "sell"),
    # Day 22 (2026-06-22)
    ("2026-06-22", "01:30", "buy"),
    ("2026-06-22", "04:00", "sell"),
    ("2026-06-22", "08:30", "buy"),
    ("2026-06-22", "17:15", "sell"),
    # Day 23 (2026-06-23)
    ("2026-06-23", "15:15", "buy"),
]


def to_ms(date_str: str, hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=h, minute=m)
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def main() -> None:
    print(f"Loading {INPUT} …", flush=True)
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} rows, {len(df.columns)} cols", flush=True)

    # Pre-extract feature columns
    META = {"window_size", "horizon", "window_end_ms"}
    LABEL = {"outcome_binary", "outcome_pct", "target_hit_1pct", "mae_pct"}
    FEATURE_COLS = [
        c for c in df.columns
        if c not in META and c not in LABEL
        and not c.startswith("outcome_pct_") and not c.startswith("outcome_binary_")
    ]
    print(f"  {len(FEATURE_COLS)} feature columns", flush=True)

    # Train one model per (window_size, horizon) on train+val, then predict on test rows
    # Actually for trace, we want to know what the model WOULD HAVE predicted at entry time.
    # Strategy: train on rows with window_end_ms < entry_time, predict on rows at entry_time.
    # Simpler: train on the per-pair train+val rows (chronological), predict on test rows.
    # For each entry timestamp, find rows in any split and report the prediction.

    df_sorted = df.sort_values("window_end_ms").reset_index(drop=True)

    out_lines = []

    for i, (date_str, hhmm, direction) in enumerate(ENTRIES, 1):
        entry_ms = to_ms(date_str, hhmm)
        out_lines.append(f"\n{'=' * 70}")
        out_lines.append(f"#{i:>2}  {date_str} {hhmm}  ENTRY {direction.upper()}  (entry_ms={entry_ms})")
        out_lines.append(f"  Local time: {dt.datetime.fromtimestamp(entry_ms/1000, tz=dt.timezone.utc)}")

        # Find the closest row in the dataset for each (ws, hz)
        # We look for rows where window_end_ms is within +/- 30 seconds of entry
        # The 4 horizons × 3 window_sizes = 12 rows per entry
        rows_for_entry = df_sorted[
            (df_sorted["window_end_ms"] >= entry_ms - 30_000) &
            (df_sorted["window_end_ms"] <= entry_ms + 30_000)
        ]
        if len(rows_for_entry) == 0:
            out_lines.append(f"  ⚠ NO ROWS within ±30s of this entry")
            continue
        # For each (ws, hz) we should have at most 1 row
        out_lines.append(f"  Found {len(rows_for_entry)} matching rows")

        # For each matched row, train a quick model on rows with same (ws, hz) but
        # window_end_ms < entry_time (using train+val) and predict on this row.
        # Then also report actual outcome.
        for _, row in rows_for_entry.iterrows():
            ws = int(row["window_size"])
            hz = int(row["horizon"])
            we = int(row["window_end_ms"])
            offset_ms = we - entry_ms
            out_lines.append(f"\n  -- W={ws}s, H={hz}s  (offset = {offset_ms/1000:+.1f}s from entry) --")

            pair = df_sorted[(df_sorted["window_size"] == ws) & (df_sorted["horizon"] == hz)]
            train_pool = pair[pair["window_end_ms"] < entry_ms]
            n_train = len(train_pool)
            if n_train < 100:
                out_lines.append(f"     (only {n_train} training rows — too few to fit)")
                out_lines.append(f"     outcome_pct: {row['outcome_pct']:+.6f}  outcome_binary: {int(row['outcome_binary'])}")
                continue

            # 80/20 chronological split for early stopping
            n = len(train_pool)
            split = int(n * 0.85)
            train = train_pool.iloc[:split]
            val = train_pool.iloc[split:]

            try:
                model = lgb.LGBMClassifier(
                    objective="binary", num_leaves=31, min_child_samples=500,
                    metric="binary_logloss", verbosity=-1, seed=42, n_estimators=500,
                )
                model.fit(
                    train[FEATURE_COLS], train["outcome_binary"],
                    eval_set=[(val[FEATURE_COLS], val["outcome_binary"])],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
                )
                prob = float(model.predict_proba(row[FEATURE_COLS].to_frame().T)[:, 1][0])
            except Exception as e:
                prob = -1.0
                out_lines.append(f"     model fit error: {e}")

            actual_pct = float(row["outcome_pct"])
            actual_bin = int(row["outcome_binary"])
            direction_match = (
                (direction == "buy" and actual_pct > 0) or
                (direction == "sell" and actual_pct < 0)
            )
            target_hit = int(row.get("target_hit_1pct", 0))
            mae = float(row.get("mae_pct", 0))

            tag = "✅" if direction_match else "❌"
            out_lines.append(
                f"     model_prob={prob:.3f}  actual={actual_pct:+.4%}  "
                f"target_hit_1pct={target_hit}  mae={mae:+.4%}  {tag} {direction}"
            )

            # Top 5 features at this moment (largest absolute z-score across the pair)
            # Skip the per-feature comparison if it would slow this too much
            try:
                # Compute z-scores against training set
                means = train[FEATURE_COLS].mean()
                stds = train[FEATURE_COLS].std().replace(0, 1)
                z = (row[FEATURE_COLS] - means) / stds
                # Get top 5 by absolute z
                top5 = z.abs().sort_values(ascending=False).head(5)
                for feat, abs_z in top5.items():
                    actual_z = float(z[feat])
                    out_lines.append(
                        f"       {feat:>32}: z={actual_z:+.2f}  val={float(row[feat]):.4f}"
                    )
            except Exception as e:
                pass

    # Summary
    out_lines.append(f"\n{'=' * 70}")
    out_lines.append("SUMMARY")
    out_lines.append("=" * 70)
    n_total = len(ENTRIES)
    out_lines.append(f"Total entries: {n_total}")
    out_lines.append(f"(Full per-entry detail above; this is just the header.)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote report to {OUT_TXT}")


if __name__ == "__main__":
    main()
