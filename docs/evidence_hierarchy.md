# Evidence hierarchy

Not all numbers in this repository carry the same evidential weight. Read
results with the following hierarchy in mind.

## 1. Validation-grade evidence: artificial-gap results

Results computed against the canonical artificial-gap pool (hundreds of
gaps spanning lengths from 1 to 60 days and multiple seasons;
`data_public/chlorophyll/chlorophyll_validation_gaps.csv`) are the only
results in this repository with a known, withheld ground-truth value to
score against. These are the only numbers that should be used to rank
methods, claim statistical significance, or argue that one method
outperforms another.

This includes:
- `results_public/chlorophyll/chlorophyll_benchmark_summary.csv`
- `results_public/chlorophyll/chlorophyll_artificial_gap_scores.csv`
- `results_public/chlorophyll/chlorophyll_covariate_mechanism_summary.csv`
- `results_public/chlorophyll/chlorophyll_event_performance_summary.csv`

Maximum validated gap length: 60 days. Claims about reconstruction quality
beyond this length are extrapolation, not validated performance.

## 2. Plausibility only: real-gap candidate outputs

`results_public/chlorophyll/chlorophyll_real_gap_candidate_outputs.csv` and
the underlying per-day reconstruction tables
(`chlorophyll_reconstruction_tsicl_satellite_proxy.csv`,
`chlorophyll_reconstruction_engineered_hybrid.csv`) are candidate outputs
for the real (naturally occurring) gaps in the observed record. There is
no withheld ground truth for any of these gaps -- the sensor was genuinely
offline, so there is nothing to score against.

These outputs are useful as plausible filled-in values for downstream
use (e.g. plotting a continuous series, rough magnitude checks), and as a
qualitative sanity check that a method's behavior on real gaps is
consistent with its behavior on similar-length artificial gaps. They must
not be used to claim a method is more accurate than another, since no such
claim is falsifiable without ground truth.

## 3. Scenario-only: the 256-day gap

The longest real gap in the record (256 days) is far outside the validated
gap-length envelope (maximum validated length: 60 days). Any reconstruction
produced for this gap is explicitly a scenario/illustrative output, not a
validated or even plausibility-checked reconstruction in the same sense as
shorter real gaps. Treat it as a demonstration of what the pipeline
produces when asked to extrapolate far beyond its tested range, not as a
trustworthy estimate of true conditions during that period.

## Summary table

| Evidence type | Ground truth available | Use for method ranking? | Use for plausible values? |
|---|---|---|---|
| Artificial-gap validation (L <= 60 days) | Yes (withheld) | Yes | Yes |
| Real-gap candidate outputs, `extrapolation_beyond_validation` = `no` or `interpolation_within_range` | No | No | Yes, with caution |
| Real-gap candidate outputs, `extrapolation_beyond_validation` = `yes` or `open_ended` | No | No | Yes, with more caution -- outside the discretely validated gap-length set |
| 256-day scenario gap (`scenario_only_256day` = `True`) | No | No | No -- illustrative only |

The `extrapolation_beyond_validation` and `scenario_only_256day` columns in
`chlorophyll_real_gap_candidate_outputs.csv` state each real gap's exact status
per-row; do not summarize this as a single day-length cutoff -- artificial-gap
validation was run at discrete lengths (1, 3, 7, 10, 14, 21, 30, 45, 60 days;
see `data_public/chlorophyll/chlorophyll_validation_gaps.csv` and
`docs/methodology/validation_protocol.md`), not as a continuous envelope up to
some maximum, so a real gap's length alone does not determine its validation
status.
