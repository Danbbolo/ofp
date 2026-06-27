#!/bin/bash
cd /home/azureuser/ofp
source .venv/bin/activate
nohup python run_research_oos_v2.py > oos_sweep_log.txt 2>&1 &
echo "OOS_PID=$!"