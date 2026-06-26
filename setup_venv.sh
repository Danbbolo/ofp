#!/bin/bash
set -e
echo "=== Installing python3-venv ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip
echo "=== Creating venv ==="
cd /home/azureuser/ofp
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet cryptohftdata
echo "===VERIFY==="
python -c "
import ofp
from ofp.feature_extractor import FeatureExtractor
from ofp.book_reconstructor import OrderBookReconstructor
from ofp.grid_sweeper import GridSweeper
import lightgbm, pandas, cryptohftdata
print('all imports OK')
print(f'pandas {pandas.__version__}, lightgbm {lightgbm.__version__}')
"
