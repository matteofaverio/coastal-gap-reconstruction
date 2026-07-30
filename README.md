# Coastal Gap Reconstruction

Historical reconstruction of gaps in coastal in-situ sensor time series, using
chlorophyll-a at a Chilean monitoring station as the primary case and
dissolved oxygen at the same station as a transfer case. The methodology is
diagnostic, not forecasting: it uses artificial-gap validation (withholding
known values and scoring the reconstruction) to compare classical baselines,
tabular machine learning, and a zero-shot time-series foundation model
(TS-ICL).

## Project outputs

| Output | Path |
|---|---|
| Final report (PDF) | [`manuscript/report/coastal_gap_reconstruction_report.pdf`](manuscript/report/coastal_gap_reconstruction_report.pdf) |
| English academic presentation (PDF) | [`manuscript/presentation/coastal_gap_reconstruction_presentation_en.pdf`](manuscript/presentation/coastal_gap_reconstruction_presentation_en.pdf) |
| Spanish colleague-facing presentation (PDF) | [`manuscript/presentation_colleagues_es/coastal_gap_reconstruction_presentation_es.pdf`](manuscript/presentation_colleagues_es/coastal_gap_reconstruction_presentation_es.pdf) |
| Scientific poster (PDF) | [`manuscript/poster/coastal_gap_reconstruction_poster.pdf`](manuscript/poster/coastal_gap_reconstruction_poster.pdf) |
| Live visual demo | [`demo/gap_reconstruction_walkthrough.ipynb`](demo/gap_reconstruction_walkthrough.ipynb) |
| Notebook index | [`notebooks/README.md`](notebooks/README.md) |
| Public data and result tables | [`data_public/`](data_public/), [`results_public/`](results_public/) |

## Workflow

```
daily target + QC  →  real-gap audit  →  artificial gaps
   →  common model comparison  →  stratified diagnostics
   →  candidate reconstruction of real gaps
```

Every result comes from the same pipeline, applied to both case studies.
Artificial gaps (known, withheld values) produce validation-grade evidence;
real gaps (naturally missing, no withheld truth) produce plausibility-only
candidate outputs. See `docs/evidence_hierarchy.md` for the full framing.

## Main findings

- Linear interpolation is a strong reference method, especially at short gap
  lengths.
- External-covariate-only tabular models (calendar, meteorology, satellite
  predictors, with no target-sensor history) do not improve on interpolation
  as direct reconstructors, for either case study.
- TS-ICL, conditioned on an appropriate covariate, gives the strongest pooled
  result across both case studies. The improvement over interpolation is
  resolved (confidence interval excludes zero) at short gaps and on
  high-chlorophyll event days; at mid-range gap lengths (7/14/30 days) the
  direction is consistent but not statistically resolved. See
  `docs/methodology/tsicl_usage.md` for the per-length breakdown.
- High-chlorophyll events and the oxygen distribution tails remain difficult
  for every method, including TS-ICL.

## Limitations

- Every method under-predicts the amplitude of high-chlorophyll event days;
  see `docs/methodology/event_limitation.md`.
- TS-ICL's pooled improvement on oxygen does not hold uniformly in either
  distribution tail; see `notebooks/10_oxygen_case_study.ipynb`.
- Real-gap candidate outputs are not validation evidence — there is no
  withheld ground truth for a naturally occurring gap. Treat them as
  plausible fill values only.

## Adapting to a new sensor

`notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` is a methodological
checklist, and `notebooks/10_oxygen_case_study.ipynb` is its worked result
(the actual released oxygen case study, not a reproduction produced by
notebook 09). The masking/scoring layer (`generate_gap_candidates`,
`apply_artificial_gap`, `run_all_baselines`, `compute_gap_metrics`) takes
`target_col`/`eligible_col` arguments rather than hardcoding chlorophyll's
column names, so it runs unchanged against a second target. That does not
mean adapting to a new sensor is a two-parameter change: eligibility
thresholds, event definitions, gap-length support, and predictor selection
are target-specific scientific decisions that still require human judgment
per target -- see the checklist notebook and
`docs/methodology/target_and_gap_construction.md`.

## Quick start

```bash
git clone https://github.com/matteofaverio/coastal-gap-reconstruction
cd coastal-gap-reconstruction
uv sync --extra notebooks --extra test   # installs the exact locked environment (uv.lock)
uv run jupyter lab notebooks/01_target_and_gap_audit.ipynb
uv run pytest tests/
```

This installs the notebook-executable core (pandas/numpy/matplotlib/scipy/
scikit-learn/jupyter) at the exact versions recorded in `uv.lock`; it does not
install TS-ICL or torch. CI runs against this same locked environment. For the
live zero-shot TS-ICL demo, see `demo/README.md` (`bash demo/run_demo.sh`),
which builds its own, separately (informally) pinned isolated environment,
kept out of `uv.lock` because `tsicl` pins a narrow `torch` range that would
otherwise force unrelated constraints onto the core lock.

If you don't have [uv](https://docs.astral.sh/uv/) installed, a plain-`pip`
equivalent works but is not version-pinned:
`python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[notebooks,test]"`.

## Repository map

```
data_public/              daily targets, predictor features, gap inventories
results_public/           benchmark scores, candidate reconstructions
figures/                  key figures referenced in the docs
manuscript/               report, presentations, poster (PDF + LaTeX source)
notebooks/                numbered, runnable notebooks (see notebooks/README.md)
demo/                     live, runnable TS-ICL demo
src/coastal_gap_reconstruction/  reusable Python utilities
tests/                    smoke tests for the demo and public notebooks
docs/                     methodology write-ups and data dictionary
```

## Reproducibility

- Python: `>=3.10` (tested on 3.11/3.12; not verified below 3.10).
- Environment: `pyproject.toml` optional-dependency groups
  (`notebooks`, `plotting`, `stats`, `tsicl-dev`, `test`); see
  `CONTRIBUTING.md` for the locked/pinned setup used for CI and the
  reported demo timings.
- Tests: `pytest tests/`.
- CI: `.github/workflows/ci.yml` runs lint, tests, notebook smoke tests, and
  document compilation on every push/PR (see that workflow file for exact
  steps; it has not been exercised on GitHub Actions itself since this
  repository is developed locally before push — see `CONTRIBUTING.md`).
- Documents: `manuscript/README.md` for the exact compilation command for
  each PDF (Tectonic).

## Citation and licenses

- [`CITATION.cff`](CITATION.cff) — how to cite this repository (code only).
- [`LICENSE`](LICENSE) — MIT, applies to code.
- [`DATA_LICENSE_AND_ATTRIBUTION.md`](DATA_LICENSE_AND_ATTRIBUTION.md) — data,
  results, and figures are not MIT licensed; required attribution for
  CEAZAMet/CEAZA, NASA/PO.DAAC MUR SST, and Copernicus/CMEMS.
- TS-ICL (used in `demo/` and `notebooks/06_tsicl_zero_shot_imputation.ipynb`)
  is separately licensed by its original authors — see `demo/README.md`.

Further detail — the full evidence hierarchy, per-notebook execution status,
the complete data dictionary, and the sensor-adaptation checklist — lives in
`docs/`, `notebooks/README.md`, and `manuscript/README.md` rather than here.
