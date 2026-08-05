# Using the oxygen case study in this benchmark

This is the oxygen-specific companion to `docs/methodology/tsicl_usage.md`
(TS-ICL provenance/environment, shared verbatim) and `model_families.md`
(shared model-family definitions). It states only what genuinely differs for
oxygen -- target scale, support size, and predictor policy -- rather than
repeating the shared material.

## Target and QC contract

`BTGOXD2` (dissolved oxygen, mg/L), daily mean, eligibility >= 18 valid
hourly observations/day, **raw mg/L throughout -- no log10 transform**
(unlike chlorophyll). Oxygen legitimately approaches zero under hypoxia;
log10 is undefined/unstable exactly at that physically meaningful low tail.
No day/night eligibility rule is applied. See
`experiments/oxygen/benchmark_contract.py` for the tested constants.

## Support

- **Primary**: 406 gaps, L in {1(100), 3(100), 7(78), 10(55), 14(36), 21(22),
  30(15)}, 2912 hidden days. Every primary scientific claim is restricted to
  this range.
- **Exploratory extended**: 6 gaps, L in {45(3), 60(1), 90(1), 120(1)} --
  illustrative only, never a headline claim.

## Predictor policy

- External physical predictors (calendar, PLV meteorology incl. solar,
  SST, wind, upwelling) and ocean currents are admissible.
- Satellite chlorophyll is exploratory/ablation-only (`external_all_available`
  arm), never a primary predictor -- weak correlation with oxygen (r=0.045-0.135).
- In-situ chlorophyll is never admissible.
- Local BTG water temperature/pressure (BTGTA/BTGPA) are admissible **only**
  in the clearly labeled `local_btg_temp_pressure_diagnostic` arm -- same-
  station co-missingness risk with the oxygen sensor itself (see
  `benchmark_contract.SAME_STATION_AVAILABILITY_CAVEAT`).
- BTGSAL/BTGTUR/BTGCND are excluded outright.

## Four feature arms (`experiments/oxygen/feature_registry.py`)

`external_physical_core`, `external_physical_plus_currents`,
`external_all_available` (exploratory), `local_btg_temp_pressure_diagnostic`
(diagnostic) -- all built by column-family selection from the same
265-column external feature snapshot chlorophyll's current-transport arms
use (`coastal_gap_reconstruction.feature_tables.load_full_feature_table`);
no oxygen-specific external feature table exists or is needed, since both
case studies share the same site and external products.

## Classical/engineered comparators (`experiments/oxygen/classical_models.py`)

Model 0 (climatology/persistence/linear interpolation) reuses
`coastal_gap_reconstruction.baseline_imputation` directly, unmodified. The
GP (Matern, time-only) residual-over-interpolation comparator reuses
`coastal_gap_reconstruction.gaussian_process` directly, unmodified -- the
one classical/engineered arm that reaches a statistical tie with
interpolation (-0.65%, CI includes zero). External-only tabular LOCO models
(Ridge/ElasticNet/HGB/ExtraTrees/RandomForest, exact hyperparameters ported
from the private release) are published; **no engineered tabular model beats
interpolation** on the primary support (all 18 completed combinations
significantly worse, CI excludes zero).

The tree-ensemble gap-edge structural variants (`direct_hindcast`,
`pre_only_forecast_safe`, and the tree-ensemble learners under
`residual_interp_hindcast`) are **not** live-executable in this package --
reproducing them correctly requires the private project's own multi-gap
pooled edge-feature training design, and a simplified substitute would
silently misrepresent what those released numbers measure. They remain
`frozen_result_only`; the released `oxygen_benchmark_by_length.csv` rows are
authoritative for them.

## TS-ICL (`experiments/oxygen/tsicl_models.py`)

Reuses the shared chlorophyll TS-ICL calling layer
(`coastal_gap_reconstruction.tsicl_helpers`) and locked environment
(`environments/tsicl/`) entirely -- checkpoint provenance, resume state
(`experiments.chlorophyll.tsicl_run_state`), and configuration-bound run
manifests (`experiments.chlorophyll.tsicl_run_manifest`) are not duplicated.

Nine arms: 5 **audited-original** (`target_only`, `calendar_seasonal`,
`external_physical_core`, `external_physical_plus_currents`,
`local_btg_temp_pressure_diagnostic`) x 2 context modes (`full_series`,
`edge_balanced`) = 10 headline rows, plus 4 **exploratory family-ablation**
arms (`currents_only`, `sst_thermal_only`, `wind_upwelling_only`,
`radiation_only`), `full_series` only. No oxygen analogue of chlorophyll's
satellite-proxy arm exists; satellite chlorophyll is forbidden outright from
every oxygen TS-ICL covariate block
(`benchmark_contract.TSICL_FORBIDDEN_COVARIATE_SUBSTRINGS`).

**Frozen headline finding** (not re-derived from a bounded run --
`results_public/oxygen/oxygen_paired_deltas_vs_tsicl_physical_covariates.csv`):
TS-ICL is the first oxygen comparator that improves over interpolation on
the primary L1-L30 support. Best arm
`external_physical_plus_currents`/`full_series`, +8.0% relative improvement,
95% CI [4.5%, 11.4%] (excludes zero). 9 of the 10 audited-original arms beat
interpolation at 95% CI; the lone exception (`target_only`/`full_series`,
CI [-0.06%, 7.1%]) is borderline.

### Bounded live validation

`run_oxygen_benchmark.py --mode tsicl-bounded` runs a deterministic
stratified subset of at most 20 primary gaps (several lengths represented)
x `target_only` and the best arm x both context modes, capped at 60 live
calls (`--max-calls`). This validates real checkpoint loading, raw mg/L
input/output, tensor shapes, no-hidden-truth masking, finite/ordered
output, and same-process repeatability -- **it does not derive a new
headline performance estimate**: the subset is small and length-biased
(stratified evenly across 7 lengths means long gaps are over-represented
relative to the primary pool's own length distribution), so its own
aggregate MAE is descriptive-only, never compared to the frozen headline as
if on the same support (see `score_oxygen_benchmark.py`'s explicit
non-extrapolation design).

## Tail diagnostics (`experiments/oxygen/tail_diagnostics.py`)

No oxygen event/high-value label exists (unlike chlorophyll's
`is_high_chl_event`) -- every threshold is an **empirical quantile of the
eligible-day population** (2880 days), not an ecological threshold
(`oxygen_threshold_provenance_audit.md` explicitly rejects any locally-
extractable ecological threshold). Six quantile bands (below_p10 through
above_p90, edges at p10=3.776, p25=5.099, p50=6.429, p75=7.436, p90=8.303
mg/L, independently reproduced exactly from the public daily target table)
and three run-persistence categories (`isolated_tail_day`, `short_tail_run`
2-6 days, `sustained_tail_run` >=7 days, broken by any calendar-date gap).
TS-ICL's improvement is not uniform across the distribution: significant
gains in the p10-p50 range, a significant loss in the high tail
(above_p90, -18.8%), and a non-significant loss in the low tail
(below_p10, -9.8%) -- 3 of 6 bands individually clear the 95% bar at n=406.

## Reproducibility levels

**QUICK**: tests, notebooks, demo, frozen-result inspection
(`--mode frozen`) -- no TS-ICL checkpoint required.

**BOUNDED**: `--mode classical` (complete 406-gap primary support,
~30 seconds) and `--mode tsicl-bounded` (<=60 live calls) -- both well under
90 minutes.

**EXPENSIVE, optional**: the complete TS-ICL grid (10 audited-original rows
+ 4 exploratory rows, all 406 primary gaps, several thousand calls) is not
re-run by this package. The frozen tables are the authoritative complete
result; running the complete grid is never required to use or inspect this
repository.
