#!/bin/bash
# backup_for_migration.sh — Backup OFP project (scripts + datasets, NO raw data)
#
# Excludes:
#   - .venv/                 (Python virtual env, can be recreated)
#   - __pycache__/           (Python bytecode cache)
#   - data/raw_futures/      (raw parquet files, re-downloadable)
#   - data/raw_futures_oos/  (raw parquet files, re-downloadable)
#   - *.pyc, .git/
#
# Includes:
#   - All Python scripts (run_research_*, relabel_*, train_*, download_*)
#   - Processed datasets (research_dataset_*.parquet)
#   - Research output (expectancy_table_*.csv)
#   - requirements.txt, README, etc.

set -e

cd /root
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/root/ofp_backup_${TIMESTAMP}.tar.gz"
LATEST_LINK="/root/ofp_backup.tar.gz"

echo "=== OFP Migration Backup ==="
echo "Excluding: .venv/, __pycache__/, data/raw_futures/, data/raw_futures_oos/"
echo "Output: $BACKUP_FILE"
echo ""

# Create tar excluding heavy/unnecessary files
tar -czf "$BACKUP_FILE" \
  --exclude='ofp/.venv' \
  --exclude='ofp/__pycache__' \
  --exclude='ofp/**/__pycache__' \
  --exclude='ofp/.git' \
  --exclude='ofp/data/raw_futures' \
  --exclude='ofp/data/raw_futures_oos' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  ofp/

# Update latest symlink
ln -sf "$BACKUP_FILE" "$LATEST_LINK"

# Report
FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo ""
echo "=== Backup Complete ==="
echo "File: $BACKUP_FILE"
echo "Size: $FILE_SIZE"
echo "Symlink: $LATEST_LINK -> $BACKUP_FILE"
echo ""
echo "To download to your local machine (from Windows):"
echo "  scp root@167.233.57.140:$BACKUP_FILE ."
echo ""
echo "To extract on new server:"
echo "  tar -xzf ofp_backup.tar.gz"
