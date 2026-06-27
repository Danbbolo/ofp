"""Inspect book and liq data schemas."""
import polars as pl

print("=== BOOK DATA ===")
book = pl.read_parquet("data/raw_futures/2026-06-17/book.parquet")
print("cols:", book.columns)
print("dtypes:", {c: str(t) for c, t in zip(book.columns, book.dtypes)})
print("rows:", len(book))
print("head:")
print(book.head(3))
print("event_type vals:", book["event_type"].unique().to_list() if "event_type" in book.columns else "N/A")
print("side vals:", book["side"].unique().to_list() if "side" in book.columns else "N/A")

print()
print("=== LIQ DATA ===")
liq = pl.read_parquet("data/raw_futures/2026-06-17/liq.parquet")
print("cols:", liq.columns)
print("dtypes:", {c: str(t) for c, t in zip(liq.columns, liq.dtypes)})
print("rows:", len(liq))
print("head:")
print(liq.head(3))
print("side vals:", liq["side"].unique().to_list() if "side" in liq.columns else "N/A")