# Live gap-reconstruction demo

Reconstructs a 14-day gap in a daily chlorophyll-a time series with eight methods,
including a **live, zero-shot run of TS-ICL** (no training or fine-tuning on this
data). This is the hands-on companion to the workflow described in the repository
root `README.md`.

## Quick start

```bash
bash demo/run_demo.sh
```

This creates an isolated virtual environment (`.venv_tsicl_demo/`, git-ignored),
installs everything needed, registers a Jupyter kernel, and executes the notebook
end to end. Tested on macOS (Apple Silicon, CPU-only inference -- TS-ICL has no
CUDA GPU on that machine and runs on CPU). Total time for a first run, including
installing packages: **under 2 minutes**. Once the environment exists, re-running
just the notebook takes well under a minute.

Output: `demo/gap_reconstruction_walkthrough_executed.ipynb` (the notebook with all
outputs filled in) and `demo/outputs/` (a figure, a results CSV, and a runtime
summary).

## What "live" means here

TS-ICL (https://github.com/EDF-Lab/ts-icl, PyPI package `tsicl`) is a pretrained,
zero-shot time-series foundation model. `run_demo.sh` installs the real `tsicl`
package and, on first use inside the notebook, the package downloads its public,
non-gated checkpoint (`tsicl-v1.ckpt`, ~209 MB, from the Hugging Face repository
`taharnbl/TS-ICL`) and caches it under `~/.cache/huggingface/hub/`. After that first
download, model loading takes under a second and each `impute()` call on a window
this size takes under 0.1s.

This needs internet access once, to install the package and fetch the checkpoint.
After that, everything (including re-running the notebook) works offline.

**If live TS-ICL genuinely cannot run** (no internet on first use, an incompatible
environment, etc.), the notebook detects the failure explicitly and falls back to
`data/cached_tsicl_predictions.csv` / `data/cached_tsicl_predictions_real_gap.csv` --
real TS-ICL output saved from an earlier live run on this same demo data, not
fabricated numbers. The notebook always prints, in the cell output and in the final
plot title, whether it is showing live or cached TS-ICL results. It never silently
substitutes one for the other.

## Manual setup (equivalent to `run_demo.sh`)

```bash
python3 -m venv .venv_tsicl_demo
.venv_tsicl_demo/bin/pip install --upgrade pip
.venv_tsicl_demo/bin/pip install tsicl jupyter nbconvert ipykernel
.venv_tsicl_demo/bin/python3 -m ipykernel install --user --name tsicl-demo --display-name "TS-ICL demo (venv)"

# headless execution:
.venv_tsicl_demo/bin/python3 -m jupyter nbconvert --to notebook --execute \
    --output gap_reconstruction_walkthrough_executed.ipynb \
    demo/gap_reconstruction_walkthrough.ipynb

# or interactively:
.venv_tsicl_demo/bin/python3 -m jupyter lab demo/gap_reconstruction_walkthrough.ipynb
# then select the "TS-ICL demo (venv)" kernel and Kernel -> Restart & Run All
```

`tsicl` pins `torch>=2.5.1,<2.10`; installing it into an unrelated existing
environment can force a torch downgrade there. Use a dedicated venv (as above), not
your general-purpose Python environment.

## Package versions this was tested with

Python 3.13, `tsicl==0.2.0`, `torch==2.9.1`, `scikit-learn==1.9.0`,
`pandas==3.0.5`, `numpy==2.5.1`, `matplotlib==3.11.1`. TS-ICL's own `pyproject.toml`
is the authoritative dependency spec; the versions above are what `pip install tsicl`
resolved to at the time this demo was built.

## License note

TS-ICL is released under the TS-ICL Non-Commercial License v1.0 (EDF SA).
Non-commercial research, evaluation, and benchmarking -- including on your own
data, as in this demo -- is an explicitly permitted use. You may not use its
outputs to train or distill a competing model, and you may not redistribute the
model or a derivative as a hosted/SaaS service. Neither restriction affects running
this notebook. Read the license yourself before other uses:
https://github.com/EDF-Lab/ts-icl.

## What the notebook does

1. Loads `data/chlorophyll_demo_series.csv` (daily in-situ chlorophyll-a, plus a
   satellite chlorophyll proxy, wind speed, and sea-surface temperature covariates
   -- all from this repository's public `data_public/`).
2. Audits missingness in the loaded window.
3. Hides a known, real 14-day interval on purpose (an *artificial gap*) and keeps
   the true values aside, only for scoring at the end.
4. Reconstructs that interval with: persistence, climatology, linear interpolation,
   a Gaussian process, an external-covariates-only tabular model
   (HistGradientBoostingRegressor), a gap-edge residual model, TS-ICL target-only,
   TS-ICL with a satellite-chlorophyll covariate, and TS-ICL with a physical
   covariate bundle (wind + SST).
5. Scores every method against the withheld truth (mean absolute error) and prints
   a runtime table.
6. Plots all reconstructions, including TS-ICL's q05-q95 predictive interval.
7. Applies the same live TS-ICL call to one real gap (2015-07-01 to 2015-07-14, no
   withheld truth) and labels the output explicitly as a candidate, not validation
   evidence.
8. Exports `outputs/demo_reconstruction_results.csv` with one row per date and
   method, carrying method name, reconstructed value, q05/q95 where available, gap
   type (artificial/real), and a `tsicl_mode` column recording `live` or
   `cached_fallback`.

## Adapting this to your own series

See the last markdown cell in the notebook ("Adapting this notebook to another
sensor or variable") for the full checklist. In short: swap `data/chlorophyll_demo_series.csv`
for your own `date` + target (+ optional covariate) table, update the column names
used in the notebook, and re-run. TS-ICL itself needs no retraining for a new
target or station -- that is the point of a zero-shot model -- but you still have
to choose, align, and quality-control whichever covariates you pass in; TS-ICL
does not do that for you.
