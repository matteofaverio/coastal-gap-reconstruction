# Model families

This benchmark organizes reconstruction methods into a small ladder, from
simplest to most complex. Simpler methods are kept as a permanent floor:
more complex methods are only worth adopting if they beat them with
statistically significant, validation-grade evidence.

## Baselines (Model 0)

Three simple, fast, fully transparent baselines:

- **Monthly climatology** -- predict each missing day with the mean target
  value for its calendar month, computed from all other eligible days.
- **Persistence** -- predict every missing day with the last observed value
  before the gap, held flat.
- **Linear interpolation** -- linearly interpolate between the last
  pre-gap and first post-gap observations. This is a diagnostic
  reconstruction method only; it is not forecast-safe, since it requires a
  future (post-gap) observation that would not be available in a true
  forecasting setting.

See `src/coastal_gap_reconstruction/baseline_imputation.py` and
`notebooks/03_baselines.ipynb`.

## Engineered tabular and gap-edge models

Classical machine learning models (linear/ridge/lasso regression, random
forest, gradient boosting) trained on engineered feature tables. Two
sub-families were explored:

- Models using only external, robust predictors (calendar, meteorological
  forcing at a nearby station, satellite/reanalysis products) -- safe to use
  on any gap, including long ones, since they never depend on the target's
  own recent history.
- Models that additionally use gap-edge features -- diagnostic information
  about the target's value immediately before and/or after a gap. These can
  be more accurate for short gaps, but are only admissible when both edges
  of the gap are actually observed (and are explicitly excluded from
  forecast-style use cases).

**Public implementations**: `experiments/chlorophyll/tabular_models.py`
(external-only; canonical learner: `ExtraTreesRegressor` on the 47-column
`arm4`/`minimal_plus_wind_relaxation` feature set, `ARM4_COLUMNS`;
`HistGradientBoostingRegressor` is a diagnostic comparator only) and
`experiments/chlorophyll/gap_edge_models.py` (retrospective, residual-over-
interpolation; the exact feature families -- `meta`/`pre`/`post`/`edge`/
`interp` -- are enumerated in `gap_edge_models.build_feature_registry()`,
not just described in prose).

The released "engineered hybrid" reconstruction output combines the gap-edge
model with a Gaussian process and a state-space (Kalman-filter) model under
a single, deterministic, length-routed method-assignment rule ("Rule D" in
the private project's internal history): Gaussian process for gaps of
1-3 days, Kalman local-level smoother for 4-29 days, gap-edge residual model
for 30+ days. This is a fixed assignment by gap length, not a per-gap fitted
choice -- see `experiments/chlorophyll/engineered_hybrid.py::ASSIGNMENT_RULE`.
See `results_public/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv`
and the column documentation in `docs/data_dictionary.md`.

See `notebooks/04_engineered_tabular_models.ipynb` for a walkthrough of
how external predictor feature tables are built and why external predictors
alone did not clearly beat interpolation in this low-data local setting, and
`notebooks/05_gap_edge_residual_models.ipynb` for the gap-edge residual
correction approach: predicting a correction over linear interpolation from
both gap edges, and how it compares as a classical ML comparator to TS-ICL.

**Running the classical benchmark**: the four methods above, plus linear
interpolation and the engineered hybrid, can be re-run on the released
449-gap matched support (`experiments/chlorophyll/benchmark_contract.py` --
the exact gap-length-restricted subset every one of these methods is scored
on, distinct from the full 681-gap pool TS-ICL diagnostics use) with:

```bash
python -m experiments.chlorophyll.run_classical_benchmark
python -m experiments.chlorophyll.run_classical_benchmark --verify
```

Expect this to take on the order of tens of minutes on a laptop CPU (the
external-tabular and gap-edge arms each fit one 500-tree `ExtraTrees` model
per gap, ~450 fits; the GP arm fits a full Gaussian process per gap).
Outputs go to a gitignored `build/chlorophyll/classical_benchmark/`
directory by default and never overwrite the frozen `results_public/`
tables; `--verify` classifies each method's reproduction against the frozen
`chlorophyll_matched_support_method_metrics.csv` as bit-identical,
numerically exact, within tolerance, or not reproduced.

## Probabilistic sequence models

Two probabilistic time-series models were evaluated:

- **Gaussian process (time-only)** -- `coastal_gap_reconstruction.gaussian_process`
  (reusable across targets). An ARD Matern-3/2 kernel plus white noise, fit
  on a local context window (default 30 days pre/post) around each gap over
  `t_rel`/`doy_sin`/`doy_cos`. `run_gp_on_gap` returns the model's actual
  posterior predictive mean and standard deviation on the modelling
  (log10) scale -- a genuine per-point predictive uncertainty from the
  fitted kernel, not a generic "confidence interval" label. It outperforms
  linear interpolation at short gap lengths in the artificial-gap
  benchmark.
- **State-space / Kalman filter** -- `experiments/chlorophyll/probabilistic_models.py`,
  a linear-Gaussian local-level model fit to the target series (chlorophyll-
  specific, not published as a reusable core module the way GP is, because
  the finding below is evidence about this specific series). **Known
  limitation, reproduced exactly, not fixed**: the maximum-likelihood fit
  converges to a near-zero observation-noise standard deviation
  (`sigma_r` &approx; 1e-13 to 1e-15 depending on exact library/BLAS
  versions), which makes a random-walk state-space model's RTS-smoothed
  mean mathematically equal to linear interpolation between flanking
  observations -- confirmed bit-identical to interpolation in the large
  majority of gaps on the private project's full pool (632/681, 93%). This
  means the "Kalman smoothing" component of the engineered hybrid's L=4-29
  segment does not currently demonstrate skill distinct from interpolation,
  despite its historical rationale. `probabilistic_models.kalman_degeneracy_report`
  and `tests/test_probabilistic_models.py`'s pinned regression test make
  this an assertion, not only a claim in prose.

## Zero-shot foundation model (TS-ICL)

TS-ICL ("time-series in-context learning") is a pretrained time-series
foundation model applied here in a zero-shot setting -- no fine-tuning on
this station's data. It is given the gappy target series directly,
optionally alongside covariate channels (most notably a satellite
chlorophyll proxy), and produces both a point estimate and quantile-based
uncertainty bands for the missing region.

The TS-ICL run using a satellite chlorophyll proxy covariate is the leading
candidate in this benchmark under artificial-gap validation -- see
`docs/methodology/tsicl_usage.md` and `notebooks/06_tsicl_zero_shot_imputation.ipynb`.
