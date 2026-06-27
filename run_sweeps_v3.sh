#!/bin/bash
# Re-sweep IS and OOS with non-overlapping windows (Phase 1 fix)
cd /home/azureuser/ofp
git pull --rebase
source .venv/bin/activate

echo "=== IS SWEEP (non-overlapping) ==="
python run_research_futures.py > is_sweep_v3_log.txt 2>&1 &
IS_PID=$!
echo "IS PID: $IS_PID"

echo "=== OOS SWEEP (non-overlapping) ==="
python run_research_oos_v2.py > oos_sweep_v3_log.txt 2>&1 &
OOS_PID=$!
echo "OOS PID: $OOS_PID"

echo "Waiting for both sweeps..."
wait $IS_PID
IS_EXIT=$?
echo "IS sweep done (exit $IS_EXIT)"
wait $OOS_PID
OOS_EXIT=$?
echo "OOS sweep done (exit $OOS_EXIT)"

echo "=== IS LOG TAIL ==="
tail -10 is_sweep_v3_log.txt
echo "=== OOS LOG TAIL ==="
tail -10 oos_sweep_v3_log.txt