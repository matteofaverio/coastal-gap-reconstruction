# Using TS-ICL in this benchmark

## What TS-ICL is

TS-ICL is a zero-shot, in-context-learning time-series foundation model: a
single pretrained model is given a series with missing values directly
(no task-specific fine-tuning on this station's data) and asked to fill
them in, optionally conditioned on additional covariate channels supplied
alongside the target series.

## How it was applied here

For each artificial or real gap, the target series (daily chlorophyll, in
log10 space) is passed to TS-ICL with the gap positions set to NaN. The
leading configuration in this benchmark, labeled "TS-ICL (satellite
chlorophyll proxy covariate)" throughout the result tables, additionally
supplies a single covariate channel: a satellite-derived chlorophyll proxy
aligned to the same dates. The model returns a point estimate and a set of
quantile predictions (5th-95th percentile) usable as an uncertainty band.

Several alternative covariate configurations were also evaluated --
wind/upwelling forcing, ocean current/transport, combined physical-forcing
sets -- and are summarized in
`results_public/chlorophyll/chlorophyll_covariate_mechanism_summary.csv`.
See `docs/methodology/model_families.md` for how the satellite-proxy
configuration compares to the others.

## Practical notes

- Covariates must be passed with an explicit batch dimension: shape
  `(1, T, C)`, not a bare `(T, C)` array -- see
  `src/coastal_gap_reconstruction/tsicl_helpers.py` for the exact reshaping
  helper and an explanation of why this matters.
- Covariates that are only sparsely available can still be used by setting
  `allow_auto_complete=True`, which lets the model fill small covariate
  gaps internally.
- Quantile outputs provide an uncertainty band around the point estimate;
  these were used for some exploratory calibration checks in this project
  but are not the primary reported metric (mean absolute error is).
- Evaluation must only use the artificially hidden positions described in
  `docs/methodology/validation_protocol.md` -- never score against
  positions the model could see.

## License note

TS-ICL's code and pretrained weights are distributed under the original
authors' license, separate from this repository's license. Review their
license and terms of use before installing or running TS-ICL in your own
work. See `notebooks/06_tsicl_zero_shot_imputation.ipynb` for an
installation and usage template.
