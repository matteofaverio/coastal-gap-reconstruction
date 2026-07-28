# Live gap-reconstruction demo

`gap_reconstruction_walkthrough.ipynb` is a step-by-step, visual walkthrough that
reconstructs a 14-day gap in a daily chlorophyll-a time series with eight methods,
including a **live, zero-shot run of TS-ICL** (no training or fine-tuning on this
data). Every step shows a figure inline before moving to the next one.

## Quick start

```bash
bash run_demo.sh
```

(or `bash demo/run_demo.sh` from the repository root.)

This creates an isolated virtual environment (`.venv_tsicl_demo/`, git-ignored),
installs everything needed, registers a Jupyter kernel, and executes the notebook
end to end. Tested on macOS (Apple Silicon, CPU-only inference -- TS-ICL has no
CUDA GPU on that machine and runs on CPU).

**Setup time** (first run, installing packages): under 2 minutes.
**Execution time** (once the environment exists): well under 30 seconds --
TS-ICL's three live calls together take well under a second; most of the wall
time is model/kernel startup, not inference.

Output: `gap_reconstruction_walkthrough_executed.ipynb` (the notebook with every
figure and value filled in) and `outputs/demo_reconstruction_results.csv` (the one
exported table -- see Section 11 of the notebook). No image files are written;
every figure lives inline in the executed notebook.

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
environment, etc.), the notebook detects the failure explicitly (Section 8) and
falls back to `data/cached_tsicl_predictions.csv` /
`data/cached_tsicl_predictions_real_gap.csv` -- real TS-ICL output saved from an
earlier live run on this same demo data, not fabricated numbers. The notebook
prints, in plain text, whether it is running live or falling back. It never
silently substitutes one for the other.

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

## Notebook sequence

Each numbered section states a short scientific question, runs a short code cell,
and immediately shows a figure. The full sequence:

| # | Section | Scientific purpose |
|---|---|---|
| 1 | Inspect the available data | Which series is the local sensor, which are external products |
| 2 | Select an observed interval | Confirm the demonstration window is fully observed, so it can be scored later |
| 3 | Create the artificial gap | Make masking and withheld truth visible, not just stated |
| 4 | Simple baselines | Persistence, climatology, interpolation -- the reference every other method must beat |
| 5 | Gaussian process | Target-only probabilistic reconstruction with a predictive interval |
| 6 | External tabular model | What a model that never sees target history receives instead |
| 7 | Gap-edge residual correction | Decompose the reconstruction into interpolation + learned correction |
| 8 | TS-ICL, live, zero-shot | Same pretrained model, three input configurations, no local fitting |
| 9 | Compare all methods | Small multiples + one MAE bar chart, one illustrative gap |
| 10 | Apply to a real gap | No withheld truth -- candidate output only |
| 11 | Export and reuse | One operational CSV, plus a minimal adaptation example |

## Mechanical vs. scientific cells

Cells tagged `hide-input` (the imports/path-setup cell near the top) are mechanical
setup, not a scientific step -- collapse them in JupyterLab via the cell toolbar
("Hide code") if you want a cleaner read. Every other cell is a deliberate
scientific step and is meant to stay visible; the notebook runs identically whether
cells are collapsed or not -- no JupyterLab extension is required.

## Where the implementation lives

The notebook calls into three small modules under `src/`, kept out of the notebook
so the walkthrough stays readable:

- `src/demo_helpers.py` -- data loading, artificial-gap construction, export.
- `src/methods.py` -- the eight reconstruction methods (each returns a
  `MethodResult`: prediction table + runtime + covariates used).
- `src/plotting.py` -- one function per figure; every function returns
  `(fig, axes)` and never calls `plt.show()` or `plt.savefig()` -- the notebook
  cell calls `plt.show()` explicitly after every plotting call, and no method here
  writes an image file. Only `demo_helpers.export_csv()` writes to disk, and only
  for the one operational CSV in Section 11.

Read these files directly to inspect exactly how each method is implemented --
the notebook shows what each method does, these modules show how.

## Output CSV schema

`outputs/demo_reconstruction_results.csv`, one row per date and method:

| Column | Meaning |
|---|---|
| date | Calendar date |
| original_target | True value, only for artificial-gap rows (empty for the real gap) |
| observed_or_missing | `artificially_hidden` or `really_missing` |
| method | Method name |
| reconstructed_median | Point/median reconstruction |
| q05, q95 | Predictive interval, where the method provides one (empty otherwise) |
| artificial_or_real_gap | `artificial` or `real` |
| validation_status | `single_illustrative_gap` or `candidate_not_validation_evidence` |
| covariates_used | Comma-separated covariate column names, or `none` |

## Adapting this to your own series

See the notebook's final section ("Use your own series") for a minimal code
example. In short:

1. Load your own `date` + target (+ optional covariate) table.
2. Pass your target column name to `dh.create_artificial_gap(..., target_column=...)`.
3. State explicitly whether you are modelling the raw unit or a transformed scale
   (this demo uses `log10`); do not silently mix the two.
4. Pick an artificial-gap interval with enough observed context on both sides --
   `create_artificial_gap` raises if the interval isn't fully observed.
5. The four "live" classical methods (persistence, climatology, interpolation, GP)
   need no code changes beyond the column name.
6. TS-ICL is called the same way regardless of target: `covariates=None` for
   target-only, or an aligned `(T, C)` array for any number of covariate channels
   (see `src/methods.py:run_tsicl`). There is no separate "installation-free" mode --
   if `tsicl` and its checkpoint are available in the active environment, it runs
   live; if not, the notebook falls back to cached output explicitly.
7. Do not evaluate a gap length your own validation has not covered without
   flagging it as unvalidated.
