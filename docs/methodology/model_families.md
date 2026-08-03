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

**Two distinct interpolation formulas, not one.** This package's classical
benchmark (`experiments/chlorophyll/run_classical_benchmark.py`) uses linear
interpolation in two different roles, with two different, non-interchangeable
formulas:

- **Standalone `canonical_interpolation` baseline**
  (`experiments/chlorophyll/interpolation_baselines.py`): interpolates in
  **log10(chl_mean) space** between the bracketing observations, then
  back-transforms. This is the exact formula that produced the frozen
  `canonical_interpolation` row in
  `chlorophyll_matched_support_method_metrics.csv` (verified against the
  private source script, not just the aggregate number).
- **Gap-edge residual anchor** (`gap_edge_models.compute_interp`):
  interpolates in **physical (mg/m^3) space**, then takes log10 of the
  result. This is the anchor the gap-edge residual model predicts a
  correction against -- an internal detail of that model, never used as a
  standalone baseline.

These formulas agree only at the two bracketing observations and diverge at
every interior day. See `interpolation_baselines.py`'s module docstring for
the full explanation and `tests/test_interpolation_baselines.py` for the
regression test pinning the standalone baseline to the released per-length
numbers.

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

**Two external-tabular protocols, not one.** The private project evaluated
external-predictor tabular models under two distinct protocols, kept
separate here as separate methods rather than conflated:

- **Plain external-only protocol** (`experiments/chlorophyll/tabular_models.py`,
  method IDs `external_only_extratrees`/`external_only_hgb`): the 47-column
  `arm4`/`minimal_plus_wind_relaxation` feature set (`ARM4_COLUMNS`, 46
  numeric after dropping the categorical `sst_primary_source`) only, no
  gap-position information. Evaluated for its own sake -- it has **no row**
  in the released `chlorophyll_matched_support_method_metrics.csv`.
- **Matched-reference protocol** (`experiments/chlorophyll/gap_edge_models.py`'s
  `run_reference_arm_loco_evaluation`, method IDs `ext_tabular_extratrees`/
  `ext_tabular_hgb`): the same `arm4` columns plus 5 structural meta-features
  describing gap length and within-gap position (`gap_length`,
  `day_index_within_gap`, `gap_position_fraction`, `distance_to_left_edge`,
  `distance_to_right_edge` -- never a hidden target value), fit under
  leave-one-gap-out with a stricter pre-only dependency-window exclusion.
  **This is the protocol that actually produced the released
  `ext_tabular_extratrees`/`ext_tabular_hgb` rows** -- confirmed against the
  private project's own per-gap prediction table
  (`data/interim/models/tier_c_7a/predictions.csv`, `arm_name ==
  "tier_a_arm4_reference"`), not merely by matching an aggregate MAE. Because
  it conditions on gap length/position, it is not strictly external-only or
  forecast-safe in the same sense as the plain protocol: it assumes the
  gap's length and the hidden day's position are known in advance.

In both protocols, `ExtraTreesRegressor(n_estimators=500)` is the canonical
learner; `HistGradientBoostingRegressor` is a diagnostic comparator only.

`experiments/chlorophyll/gap_edge_models.py` also implements the retrospective,
residual-over-interpolation gap-edge model (`tier_ch_deployed`) -- the exact
feature families (`meta`/`pre`/`post`/`edge`/`interp`) are enumerated in
`gap_edge_models.build_feature_registry()`, not just described in prose.

The released "engineered hybrid" reconstruction output combines the gap-edge
model with a Gaussian process and a state-space (Kalman-filter) model under
a single, deterministic, length-routed method-assignment rule ("Rule D" in
the private project's internal history): Gaussian process for gaps of
1-3 days, Kalman local-level smoother for 4-29 days, gap-edge residual model
for 30+ days. This is a fixed assignment by gap length, not a per-gap fitted
choice -- see `experiments/chlorophyll/engineered_hybrid.py::ASSIGNMENT_RULE`.
See `results_public/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv`
and the column documentation in `docs/data_dictionary.md`. **Support status**:
the engineered hybrid has a released row on the full 681-gap pool
(`chlorophyll_benchmark_summary.csv`), but **not** on the 449-gap matched
support -- when this package's benchmark runs it on the matched support
(`benchmark_contract.METHODS["engineered_hybrid"].support_status ==
"new_evaluation_on_matched_449"`), that is a new consistency evaluation for
comparability with the other matched-support methods, not a reproduction of
a released matched-support number.

See `notebooks/04_engineered_tabular_models.ipynb` for a walkthrough of
how external predictor feature tables are built, why external predictors
alone did not clearly beat interpolation in this low-data local setting, and
a real run of both protocols side by side, and
`notebooks/05_gap_edge_residual_models.ipynb` for the gap-edge residual
correction approach: predicting a correction over linear interpolation from
both gap edges, and how it compares as a classical ML comparator to TS-ICL.

**Running the classical benchmark**: all methods above can be re-run on the
released 449-gap matched support (`experiments/chlorophyll/benchmark_contract.py`
-- the exact gap-length-restricted subset every one of these methods is
scored on, distinct from the full 681-gap pool TS-ICL diagnostics use) with:

```bash
python -m experiments.chlorophyll.run_classical_benchmark
python -m experiments.chlorophyll.run_classical_benchmark --verify
```

Expect this to take on the order of tens of minutes on a laptop CPU (the
external-tabular and gap-edge arms each fit one 500-tree `ExtraTrees` model
per gap, ~450 fits per method x 2 protocols; the GP arm fits a full Gaussian
process per gap). Outputs go to a gitignored
`build/chlorophyll/classical_benchmark/` directory by default and never
overwrite the frozen `results_public/` tables. A method's cached predictions
are only reused across re-runs if a metadata sidecar
(`predictions_<method>.meta.json`) confirms the exact same gap set, input
file hashes, and configuration produced them -- any drift forces a
recompute. `--verify` writes a structured, per-metric
`verification_report.csv`/`verification_summary.json` classifying each
metric (day-weighted/gap-weighted MAE, RMSE, bias, median/p90 absolute
error, aggregate and by gap length) as exact, numerically equal, within a
documented method-specific tolerance, mismatched, or not applicable (for
methods with no frozen row, e.g. the plain external-only protocol) --
**not** a single global tolerance on one aggregate number.

**Reproduction tolerance evidence, from an actual clean 449-gap run.**
`canonical_interpolation` is a closed-form deterministic formula and
reproduced the released numbers **exactly** (diff `0.0` on every aggregate
metric). `gp_m1`, `ext_tabular_extratrees`, `ext_tabular_hgb`, and
`tier_ch_deployed` all fit scikit-learn estimators with an internal source of
run-to-run numerical variability that is **not** bit-reproducible across
environments even under a fixed `random_state` -- GP's L-BFGS-B
hyperparameter optimizer reaching different local optima on a minority of
gaps; `ExtraTreesRegressor`/`HistGradientBoostingRegressor` with `n_jobs=-1`
parallel floating-point summation order; HGB's early-stopping validation
split adds a further source for that learner specifically. Observed
aggregate-metric diffs from the clean run: `gp_m1` up to 0.00105 (rmse),
`ext_tabular_extratrees` up to 0.00123 (rmse), `tier_ch_deployed` up to
0.00066 (bias), `ext_tabular_hgb` up to 0.00353 (p90, the diagnostic
comparator, largest gap). All aggregate metrics for all four methods land
`within_documented_method_specific_tolerance` except `gp_m1`'s p90 (a
tail-sensitive statistic, expected to be more sensitive to the same
small-outlier-gap pattern found in the per-day forensic comparison). A
handful of by-length cells (smaller subsets, as few as 50-100 gaps) remain
`mismatched` even though the aggregate is within tolerance -- expected
statistical behavior for smaller-n subsets, reported honestly rather than
hidden by widening the tolerance further. None of this pattern is
length-dependent/systematic in a way that would indicate a scientific-
protocol mismatch (checked directly, not assumed) -- see
`docs/methodology/validation_protocol.md` "Reproduction tolerance evidence"
and `build/chlorophyll/classical_benchmark_corrected/verification_report.csv`
for the full per-metric breakdown. `METRIC_TOLERANCE` in
`run_classical_benchmark.py` encodes each method's tolerance from this
evidence, not as one global number.

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
