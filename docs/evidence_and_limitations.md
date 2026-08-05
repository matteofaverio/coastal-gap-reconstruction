# Evidence and limitations

Not all numbers in this repository carry the same evidential weight. Read
results with this hierarchy in mind, and read the limitations below before
drawing a stronger conclusion than the evidence supports.

## Evidence hierarchy

### 1. Validation-grade: artificial-gap results

Results computed against the canonical artificial-gap pool -- chlorophyll:
681 gaps, 9 lengths, 1-60 days; oxygen: 406 primary gaps, 7 lengths, 1-30
days -- are the only results in this repository with a known, withheld
ground-truth value to score against. These are the only numbers that
should be used to rank methods or claim statistical significance:
`results/chlorophyll/chlorophyll_benchmark_summary.csv`,
`chlorophyll_artificial_gap_scores.csv`,
`chlorophyll_covariate_mechanism_summary.csv`,
`chlorophyll_event_performance_summary.csv`,
`results/oxygen/oxygen_benchmark_by_length.csv`,
`oxygen_paired_deltas_vs_tsicl_physical_covariates.csv`.

Maximum validated gap length: 60 days (chlorophyll), 30 days (oxygen
primary; 45-120 days is exploratory-extended, never a headline number).
Claims beyond these lengths are extrapolation, not validated performance.

### 2. Plausibility only: real-gap candidate outputs

`results/chlorophyll/chlorophyll_real_gap_candidate_outputs*.csv` and the
underlying per-day reconstruction tables are candidate outputs for the 128
real (naturally occurring) gaps in the observed chlorophyll record. There
is no withheld ground truth for any of these -- the sensor was genuinely
offline. Useful as plausible fill values (continuous-series plotting,
rough magnitude checks) and as a qualitative sanity check against similar
artificial-gap-length behavior; must **not** be used to claim one method is
more accurate than another on real gaps -- no such claim is falsifiable
without ground truth.

Two independent candidate methods exist and neither is presented as "the"
correct reconstruction: `engineered_hybrid` (a **method-selected**
candidate -- a deterministic length-routed rule, "Rule D," assigns exactly
one of three component methods per gap) and `tsicl_satellite_proxy`
(applied **uniformly** to every gap, no per-gap routing). See
`docs/methods.md` and `experiments/chlorophyll/real_gap_contract.py`
(`REAL_GAP_ARTIFACTS`) for the exact evidential status of every published
real-gap artifact.

Oxygen has a real-gap **inventory only** (125 real gaps, by length class) --
no reconstruction-candidate generator exists for oxygen; this is a
confirmed absence, not an oversight
(`experiments/oxygen/real_gap_contract.py`).

### 3. Scenario-only: the 256-day gap

The longest real gap in the chlorophyll record (256 days, 2020-02-11 to
2020-10-23, a genuine sensor outage) is far outside the validated
gap-length envelope (max validated: 60 days). It is a real, observed gap --
not a synthetic construction -- but its reconstruction is explicitly a
scenario/illustrative output, not even plausibility-checked in the same
sense as shorter real gaps. Flagged `scenario_only_256day` throughout
`results/chlorophyll/`.

### Summary table

| Evidence type | Ground truth available | Rank methods? | Plausible values? |
|---|---|---|---|
| Artificial-gap validation (within max validated length) | Yes (withheld) | Yes | Yes |
| Real-gap candidate, within validated length range | No | No | Yes, with caution |
| Real-gap candidate, beyond validated length range | No | No | Yes, with more caution |
| 256-day scenario gap | No | No | No -- illustrative only |

## Reproducing this classification, deterministically

`experiments/chlorophyll/real_gap_contract.py` (`REAL_GAP_ARTIFACTS`) is
the tested, machine-readable version of the table above.
`real_gap_inventory.py` detects real gaps from the daily target directly
(reproduces the released 128-row inventory exactly, contiguous-run
detection over the eligibility column only -- never reads a candidate
file). `select_real_gap_reconstruction.py` reproduces Rule D's routing;
`assemble_real_gap_candidates.py` joins the per-method candidate files with
validation. See `notebooks/05_real_gap_candidates.ipynb` for a runnable
walkthrough. None of this runs TS-ICL or fits a model -- pure detection/
lookup/joining over already-frozen inputs.

## Known limitations

### High-chlorophyll event days

Every reconstruction method under-predicts the amplitude of high-chlorophyll
("event," >90th percentile) days and shows measurably worse error there
than on background days -- an unresolved limitation, not a solved problem:

| Method | MAE, event days | MAE, non-event days | Event penalty | Amplitude bias |
|---|---|---|---|---|
| Linear interpolation | 0.323 | 0.230 | +0.093 | -0.004 |
| Gaussian process | 0.316 | 0.221 | +0.095 | -0.071 |
| Engineered hybrid | 0.309 | 0.222 | +0.087 | -0.046 |
| Gap-edge residual model | 0.281 | 0.205 | +0.076 | -0.096 |
| TS-ICL, target-only | 0.301 | 0.211 | +0.090 | -0.064 |
| TS-ICL, satellite proxy | 0.292 | 0.205 | +0.086 | -0.052 |

(log10 chlorophyll scale; negative amplitude bias = under-prediction.) The
satellite-proxy TS-ICL configuration has the best event-day MAE among
leading candidates but still under-predicts event amplitude meaningfully.
High-chlorophyll events are exactly the periods of greatest scientific
interest (harmful algal blooms, productivity pulses), so this is a material
limitation for downstream use -- see
`results/chlorophyll/chlorophyll_event_performance_summary.csv`.

### Oxygen distribution tails

TS-ICL's pooled +8.0% improvement over interpolation on oxygen does not
hold uniformly: significant gains in the p10-p50 range, a significant
*loss* in the high tail (above p90, -18.8%), and a non-significant loss in
the low tail (below p10, -9.8%). See
`results/oxygen/oxygen_tail_quantile_band_metrics.csv` and
`oxygen_tail_persistence_metrics.csv`.

### Kalman-filter degeneracy

The state-space (Kalman) model's maximum-likelihood observation-noise
estimate collapses to a near-zero value, making its smoothed output
numerically equal to linear interpolation on 93% of gaps in the full
chlorophyll pool (632/681). This means the "Kalman smoothing" component of
the engineered hybrid pipeline's L=4-29 segment does not currently
demonstrate skill distinct from interpolation, despite its historical
rationale -- documented as a reproduced finding, not fixed
(`experiments/chlorophyll/probabilistic_models.py::kalman_degeneracy_report`,
`tests/test_probabilistic_models.py`).

### External-tabular models did not beat interpolation

Classical tabular models trained only on external predictors (no target
history) did not show a clear, statistically significant improvement over
linear interpolation across most gap lengths, for either case study. Two
probabilistic sequence models and TS-ICL performed competitively or better.
See `docs/methods.md` for plausible reasons in this low-data setting.
