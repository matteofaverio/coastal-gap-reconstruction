# Artificial-gap validation protocol

## Why artificial gaps

Real gaps in the chlorophyll record have no withheld ground truth: the
sensor was actually offline, so there is no "true" value to compare a
reconstruction against. To get any quantitative evidence about which
reconstruction method is more accurate, this benchmark instead carves
artificial gaps out of stretches of the record that ARE observed, hides the
true values from each candidate method, generates a reconstruction, and then
scores the prediction against the value that was secretly retained.

This is the only validation-grade evidence in this repository. Anything
computed on real gaps is a candidate output, not a score (see
`docs/evidence_hierarchy.md`).

## The shared method-comparison gap pool

`data_public/chlorophyll/chlorophyll_validation_gaps.csv` contains the
shared method-comparison pool of artificial gaps (hundreds of gaps, spanning
multiple lengths and seasons) used for all benchmark comparisons in this
repository -- every method scored in `docs/evidence_hierarchy.md`'s
validation-grade tier is evaluated against this same fixed set of hidden
gaps, so method comparisons share an identical evaluation set rather than
each method being scored on a different sample. Each row is one gap, with
its length, season, start/end date, the true mean/max chlorophyll value over
the hidden days (kept for scoring), and an event flag.

Gap lengths in the pool: 1, 3, 7, 10, 14, 21, 30, 45, and 60 days. Longer
lengths were not included in this pool because there are too few
non-overlapping eligible stretches of the record long enough to support
them with adequate statistical power -- this is also why the 256-day real
gap is explicitly out of scope for validated comparisons.

## Construction rules

- Every hidden day in a candidate gap must itself be eligible (see
  `docs/methodology/target_and_gap_construction.md`) -- otherwise we would
  be scoring against an untrustworthy "true" value.
- Gaps of a given length are sampled to be non-overlapping with a fixed
  random seed. `src/coastal_gap_reconstruction/artificial_gap_validation.py`
  implements this masking/sampling logic as a public illustration of the rule
  (it does not reproduce the released pool byte-for-byte). The exact,
  byte-for-byte reproduction of the full released 681-row pool -- including
  the sustained-event flags, context-availability checks, regime label, and
  checksum columns -- is `experiments/chlorophyll/target_and_gap_pool.py`; see
  that module's docstring for the two sampling procedures involved and
  `tests/test_gap_pool_regeneration.py` for the exact equality guarantees.
  Either way, treat `chlorophyll_validation_gaps.csv` itself as the
  authoritative pool definition.
- A gap is flagged as a high-chlorophyll "event" gap if any hidden day's
  true value exceeds the 90th percentile of all eligible target values.

## Matched support for the classical/probabilistic/gap-edge benchmark

The full 681-gap pool above (L=1,3,7,10,14,21,30,45,60) is used for TS-ICL
calibration/covariate diagnostics. The external-tabular, gap-edge residual,
Gaussian process, and engineered hybrid methods in
`experiments/chlorophyll/` are instead scored on a **449-gap matched
support** (`data_public/chlorophyll/chlorophyll_matched_support_449.csv`,
L=1,3,7,14,30 only -- an exact subset of the 681-gap pool, see
`experiments/chlorophyll/benchmark_contract.py`). This restriction exists
because the gap-edge residual model's hindcast feature construction only
ever ran on these five original ("core") gap lengths in the private
project's history -- a fair comparison across every method needs a support
every method actually has a prediction for. Run
`python -m experiments.chlorophyll.run_classical_benchmark` to reproduce it
(see `docs/methodology/model_families.md`).

## Oxygen gap-length support

The oxygen benchmark (`data_public/oxygen/oxygen_validation_gaps.csv`) uses
a different length set than chlorophyll, reflecting its shorter/differently
structured eligible-run history:

- primary support (used for the headline oxygen benchmark numbers):
  1, 3, 7, 10, 14, 21, 30 days;
- exploratory extended lengths (reported separately, smaller sample, wider
  uncertainty, not part of the primary comparison):
  45, 60, 90, 120 days.

`results_public/oxygen/oxygen_benchmark_by_length.csv` reports the primary
support only. See `notebooks/10_oxygen_case_study.ipynb` for how the two are
distinguished in practice.

## Leakage prevention (plain-language explanation)

"Leakage" here means accidentally letting a method see information it
should not have access to -- most importantly, the true value of a day that
is supposed to be hidden.

To prevent this:

1. Only the chlorophyll target column is masked for an artificial gap.
   External predictor data (satellite, meteorological, oceanographic
   covariates) at the same dates remain available -- this mirrors a
   realistic deployment, where the sensor itself failed but other data
   sources kept reporting.
2. Any feature derived FROM the target history (for example, "chlorophyll
   3 days ago" or "chlorophyll over the last 7 days") must be computed
   strictly from the masked series -- if any of those days fall inside the
   hidden gap, the feature must also become missing/NaN, never silently
   filled in from the true value.
3. Reconstruction methods are evaluated only on the secretly retained true
   values of the hidden days, never on days that were visible to the
   method during prediction.
4. A method's hyperparameters or thresholds (e.g. a climatology baseline's
   monthly averages) are always fit excluding the hidden days of the gap
   currently being scored.

## Forbidden scoring substitutes (chlorophyll)

The chlorophyll daily target table also carries satellite-derived chlorophyll
proxy columns (`chl_cons_*`, `chl_perm_*`, `chl_anom_*`, and
patchiness/patch-distance statistics). These are legitimate *predictors* --
they may be used as model features or TS-ICL covariates -- but must never be
used as the scoring ground truth for evaluating in-situ chlorophyll
reconstruction: they are satellite estimates, not the in-situ measurement
this benchmark reconstructs. The only valid scoring target is `chl_mean`
(physical) or `log10(chl_mean)` (log, the benchmark's actual scoring scale --
see `docs/methodology/target_and_gap_construction.md`), both from
`data_public/chlorophyll/chlorophyll_daily_target.csv`. This rule is stated
here as prose rather than enforced by code; nothing in this repository
checks it automatically.

## What gets reported

Errors are not only reported as a single overall number. This benchmark
tracks performance broken out by:

- gap length;
- season;
- event status (high-chlorophyll vs. background days);
- and, where available, predictor/covariate availability.

Comparisons between methods use paired bootstrap resampling over the gap
pool to produce confidence intervals on the difference in mean absolute
error, rather than relying on a single point estimate.
