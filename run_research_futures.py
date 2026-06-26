"""
run_research_futures.py — Run the sweep on futures data (1 day smoke test).

Wraps run_research.py with a different RAW_DIR and output filename.
"""
import sys
import run_research
from pathlib import Path

# Override paths BEFORE main() runs
run_research.RAW_DIR = Path("data/raw_futures")
run_research.OUTPUT_FILE = Path("data/research_dataset_futures.parquet")

if __name__ == "__main__":
    # Parse date range from argv or default to single-day from the futures dir
    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        # Find the only date in raw_futures/
        dates = sorted(p.name for p in Path("data/raw_futures").iterdir() if p.is_dir())
        if not dates:
            print("No dates in data/raw_futures/")
            sys.exit(1)
        start = end = dates[0]
        print(f"Using single day: {start}")

    run_research.main(start, end)
