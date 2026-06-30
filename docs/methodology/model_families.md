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

The released "engineered hybrid" reconstruction output combines several of
these engineered components with a Gaussian process and a state-space
(Kalman-filter) model under a single validation-aware method-assignment
policy: for each real gap, the method assignment is chosen based on which
family had validated, statistically supported skill at gaps of that length.
See `results_public/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv`
and the column documentation in `docs/data_dictionary.md`.

See `notebooks/04_engineered_tabular_models.ipynb` for a walkthrough of
how external predictor feature tables are built and why external predictors
alone did not clearly beat interpolation in this low-data local setting, and
`notebooks/05_gap_edge_residual_models.ipynb` for the gap-edge residual
correction approach: predicting a correction over linear interpolation from
both gap edges, and how it compares as a classical ML comparator to TS-ICL.

## Probabilistic sequence models

Two probabilistic time-series models were evaluated:

- **Gaussian process (time-only)** -- a GP regression over time, used as a
  smooth interpolant with calibrated uncertainty. It outperforms linear
  interpolation at short gap lengths in the artificial-gap benchmark.
- **State-space / Kalman filter** -- a linear-Gaussian state-space model
  fit to the target series. A known limitation: in this benchmark, the
  fitted Kalman component converged to a degenerate parameterization that
  made its point predictions numerically near-identical to linear
  interpolation, rather than exploiting the state-space structure as
  intended. This is documented here as a caveat, not fixed in this release
  -- a known limitation of the Kalman component as currently configured,
  not a property of state-space models in general.

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
