#!/usr/bin/env bash
# Sets up an isolated environment and runs the live TS-ICL demo notebook end to end.
# Every command in this script has been run and tested on macOS (Apple Silicon, CPU-only
# inference, no CUDA) as part of building this repository. Total time on that machine:
# ~30s for install, ~1s for model load (checkpoint already cached), well under a minute
# to execute the whole notebook once the environment exists.
#
# Usage:
#   bash demo/run_demo.sh
#
# Run this from the repository root (the folder containing this "demo/" directory).
set -euo pipefail

cd "$(dirname "$0")/.."  # repository root

VENV=.venv_tsicl_demo
if [ ! -d "$VENV" ]; then
    echo "Creating $VENV ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip -q
    echo "Installing tsicl (pulls in torch, scikit-learn, pandas, numpy, matplotlib) ..."
    "$VENV/bin/pip" install -q tsicl jupyter nbconvert ipykernel
fi

# Register this venv as a named Jupyter kernel so the notebook (which pins kernel
# name "tsicl-demo") resolves to it instead of any other Python/Jupyter install on
# this machine.
"$VENV/bin/python3" -m ipykernel install --user --name tsicl-demo --display-name "TS-ICL demo (venv)"

echo "Running the notebook (executes live, including TS-ICL inference) ..."
"$VENV/bin/python3" -m jupyter nbconvert \
    --to notebook --execute \
    --output gap_reconstruction_walkthrough_executed.ipynb \
    demo/gap_reconstruction_walkthrough.ipynb

echo "Done. Executed notebook: demo/gap_reconstruction_walkthrough_executed.ipynb"
echo "Figure and results table: demo/outputs/"
