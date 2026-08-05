#!/usr/bin/env bash
# Executes the live TS-ICL demo notebook end to end, using this repository's
# single authoritative TS-ICL environment (environments/tsicl/) -- the same
# locked environment the benchmark drivers use, not a separate demo-only one.
#
# Usage:
#   bash demo/run_demo.sh
#
# Run this from the repository root (the folder containing this "demo/" directory).
set -euo pipefail

cd "$(dirname "$0")/.."  # repository root

ENV_DIR=environments/tsicl
if [ ! -d "$ENV_DIR/.venv" ]; then
    echo "Setting up the locked TS-ICL environment ($ENV_DIR) ..."
    (cd "$ENV_DIR" && uv sync --locked)
fi

VENV_PY="$ENV_DIR/.venv/bin/python"

# Register this venv as a named Jupyter kernel so the notebook (which pins
# kernel name "tsicl-env") resolves to it instead of any other Python/Jupyter
# install on this machine.
"$VENV_PY" -m ipykernel install --user --name tsicl-env --display-name "TS-ICL (environments/tsicl)"

echo "Running the notebook (executes live, including TS-ICL inference) ..."
# PATH must put this venv's bin/ first: `jupyter nbconvert` dispatches to a
# `jupyter-nbconvert` executable resolved via PATH, not via Python import,
# so without this it silently falls back to any other Jupyter install on
# the machine.
PATH="$PWD/$ENV_DIR/.venv/bin:$PATH" "$VENV_PY" -m jupyter nbconvert \
    --to notebook --execute \
    --output gap_reconstruction_walkthrough_executed.ipynb \
    demo/gap_reconstruction_walkthrough.ipynb

echo "Done. Executed notebook: demo/gap_reconstruction_walkthrough_executed.ipynb"
echo "Figure and results table: demo/outputs/"
