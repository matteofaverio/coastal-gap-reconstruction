# Data dictionary

Column definitions for every public CSV under `data_public/` and
`results_public/`. For the small standalone CSVs used by the live demo
(`demo/data/`), see the "What the notebook does" section of
`demo/README.md` -- they are subsets/derivatives of the tables documented
below, not a separate schema.

## data_public/oxygen/oxygen_daily_target.csv

One row per calendar day, 2015-07-01 to 2026-05-31 (BTGOXD2 sensor, mg/L).
Same construction logic as the chlorophyll daily target, applied to a
second sensor.

| Column | Description |
|---|---|
| date | Calendar date (daily resolution) |
| n_expected_hours, n_rows, valid_hours, coverage_fraction, quality_class | Same meaning as the chlorophyll target table |
| eligible_ge_12, eligible_ge_17, eligible_ge_18, eligible_ge_22 | Eligibility at alternative valid-hour thresholds; the oxygen benchmark uses `eligible_ge_18`, mirroring the chlorophyll >=18h rule |
| oxygen_mean_mgL | Daily mean dissolved oxygen, mg/L (the reconstruction target) |
| oxygen_median_mgL, oxygen_std_mgL, oxygen_min_mgL, oxygen_max_mgL | Other within-day summary statistics |
| oxygen_q10_mgL, oxygen_q25_mgL, oxygen_q75_mgL, oxygen_q90_mgL | Within-day quantiles |
| raw_prom_non_na_count, raw_prom_na_count | Count of non-missing / missing raw hourly readings |
| negative_oxygen_count, zero_oxygen_count, invalid_oxygen_count | QA counters for out-of-range or invalid hourly readings |
| first_valid_hour, last_valid_hour, longest_invalid_run_hours | Same meaning as the chlorophyll target table |
| source_variable | Sensor code (`BTGOXD2`); see `docs/data_sources_and_attribution.md` for why this sensor was selected over `BTGOXD`/`BTGOXSATPC` |
| unit | Physical unit (mg/L) |
| timezone_status | Timezone handling/conversion note |

## data_public/oxygen/oxygen_validation_gaps.csv

The artificial-gap validation pool for oxygen (L=1-30 days). Same schema
pattern as `chlorophyll_validation_gaps.csv` (one row per artificial gap).

## data_public/oxygen/oxygen_real_gap_inventory_by_class.csv

Naturally-occurring missing-data periods, grouped by length class
(short/medium/long), with gap counts and median/max length per class.

## results_public/oxygen/oxygen_benchmark_by_length.csv

One row per (method, gap length): gap-weighted MAE (mg/L) and gap count,
across the oxygen artificial-gap pool. Pivot on `method_label` to compare
methods.

## results_public/oxygen/oxygen_paired_deltas_vs_tsicl_physical_covariates.csv

One row per comparator method: paired MAE delta (TS-ICL physical
covariates minus comparator), 95% bootstrap CI, and whether the CI
excludes zero (statistically resolved).

## results_public/oxygen/oxygen_tail_quantile_band_metrics.csv, oxygen_tail_persistence_metrics.csv

Relative improvement of the strongest TS-ICL arm over interpolation,
broken out by empirical oxygen quantile band and by tail-run persistence
(isolated / short / sustained). Backs Figure 7 of the report
(`figures/oxygen/figure_oxygen_tail_diagnostics.pdf`).

## data_public/chlorophyll/chlorophyll_daily_target.csv

One row per calendar day, 2015-07-01 to 2026-05-31.

| Column | Description |
|---|---|
| date | Calendar date (daily resolution) |
| n_expected_hours | Expected number of hourly readings that day (usually 24) |
| n_rows | Number of raw hourly rows found for that day |
| valid_hours | Number of hourly readings that passed basic validity checks |
| coverage_fraction | valid_hours / n_expected_hours |
| quality_class | Categorical data-quality label for the day |
| target_eligible_default | True if the day meets the >=18-valid-hour eligibility threshold used throughout this benchmark |
| eligible_ge_12, eligible_ge_17, eligible_ge_22 | Alternative eligibility thresholds (12/17/22 valid hours), for sensitivity checks |
| chl_mean | Daily mean chlorophyll-a (the primary reconstruction target) |
| chl_median, chl_std, chl_min, chl_max | Other within-day summary statistics of chlorophyll-a |
| chl_q10, chl_q25, chl_q75, chl_q90 | Within-day quantiles of chlorophyll-a |
| raw_prom_non_na_count, raw_prom_na_count | Count of non-missing / missing raw hourly readings |
| negative_chl_count, invalid_chl_count | QA counters for out-of-range or invalid hourly readings |
| first_valid_hour, last_valid_hour | First/last hour of day with a valid reading |
| longest_invalid_run_hours | Longest consecutive run of invalid hourly readings that day |
| data_pc_min/median/mean/max | Data-percent-complete statistics reported by the source system |
| data_pc_zero_count, data_pc_nonzero_count, data_pc_100_count | Counts of hours with 0%/nonzero%/100% completeness flags |
| timezone_status | Note on timezone handling/conversion status for that day's source data |
| source_file_count_or_source_note | Provenance note: number of source files contributing, or a note about source |

## data_public/chlorophyll/chlorophyll_predictor_features_curated.csv

One row per calendar day. A curated set of external/spatial predictor
features (calendar, sea-surface temperature, wind, satellite chlorophyll
proxies, upwelling indices), used for engineered tabular models and as
TS-ICL covariate sources. Column naming convention: `<source>_<variable>`,
with suffixes `_lagNd` (value N days earlier), `_rollNd` (N-day rolling
mean), `_available` (boolean flag for whether the underlying source had
data that day). Key feature families:

| Prefix / column | Description |
|---|---|
| season, day_of_year, month, year, doy_sin, doy_cos | Calendar features |
| chl_cons_*, chl_perm_* | Satellite chlorophyll proxy variants (conservative/permissive masking), log10 scale |
| mur_sst_*, ostia_sst_*, sst_primary_* | Sea-surface temperature from MUR and OSTIA satellite products |
| wind_*, plv_wind_* | Wind components/speed from CMEMS reanalysis and a nearby meteorological station (PLV) |
| plv_temp_degC, plv_pressure_hPa, plv_humid_pct, plv_precip_daily_mm | PLV station meteorological variables |
| plv_solar_wm2 | PLV station solar irradiance |
| *_upwelling_ms, *_upwelling_cumulNd_ms_d | Upwelling-index variables (instantaneous and cumulative) from PLV and CMEMS |
| *_relaxation_index_14p3r | Wind relaxation index (upwelling-relaxation event detector) |
| mur_gradient_*, mur_front_*, mur_coastal_grad_* | SST gradient/frontal features derived from MUR SST |
| chl_patch*, chl_anom_* | Spatial patchiness and anomaly features derived from satellite chlorophyll |
| eval_event_spike | Internal QA flag for sudden spikes; not a predictor |

See `docs/data_sources_and_attribution.md` for the underlying satellite/
reanalysis product attributions.

## data_public/chlorophyll/chlorophyll_validation_gaps.csv

One row per artificial gap in the canonical chlorophyll validation pool
(hundreds of gaps spanning multiple lengths and seasons).

| Column | Description |
|---|---|
| gap_id | Unique identifier, encodes gap length and start date |
| gap_length | Number of hidden days |
| start_date, end_date | First/last hidden day |
| n_hidden_days | Same as gap_length |
| season | Meteorological season (DJF/MAM/JJA/SON) of the gap start |
| year | Calendar year of the gap start |
| target_mean_true, target_max_true | True mean/max chlorophyll over the hidden days (withheld from candidate methods, used only for scoring) |
| chl_90th_threshold | The 90th-percentile chlorophyll threshold used to flag high-chlorophyll events |
| is_high_chl_event | True if any hidden day's true value exceeds chl_90th_threshold |
| is_sustained_event | True if the event condition persists across multiple hidden days |
| is_background | True if the gap is not flagged as an event gap |
| pre_context_available_days, post_context_available_days | Number of eligible days immediately before/after the gap, available as context |
| context_constrained | True if pre/post context is limited (e.g. near the start/end of the record) |
| regime | Internal label for the gap-construction protocol version used |
| target_table_checksum | Checksum of the target table version used to build this gap pool, for reproducibility verification |

## data_public/chlorophyll/chlorophyll_real_gap_inventory.csv

One row per real (naturally occurring) gap in the observed record.

| Column | Description |
|---|---|
| gap_id | Unique identifier |
| start_date, end_date | First/last missing day |
| length_days | Gap length in days |
| gap_class | Categorical length bucket (e.g. short/medium/long) |
| seasons | Season(s) spanned by the gap |
| year_start, year_end | Calendar year(s) spanned |
| pre_edge_available, post_edge_available | Whether an eligible observation exists immediately before/after the gap |
| interpolation_admissible | Whether linear interpolation is technically computable for this gap (both edges available) |
| gap_edge_features_admissible | Whether gap-edge diagnostic features are admissible for this gap |
| nearest_val_lengths | The nearest validated artificial-gap length(s) used as a proxy for expected accuracy at this length |
| extrapolation_beyond_validation | Whether this gap's length exceeds the maximum validated gap length (60 days) |
| notes | Free-text notes on the gap, where applicable |

## results_public/chlorophyll/chlorophyll_reconstruction_tsicl_satellite_proxy.csv

Day-level TS-ICL reconstruction output (satellite chlorophyll proxy
covariate configuration), covering both real and artificial gap positions.

| Column | Description |
|---|---|
| gap_id | Gap identifier (links to a real or artificial gap) |
| date | Reconstructed day |
| pred_log10_chl | Point prediction, log10 chlorophyll scale |
| pred_chl | Point prediction, back-transformed to chlorophyll units |
| q05, q10, q25, q50, q75, q90, q95 | Quantile predictions (log10 scale), forming an uncertainty band |
| artificial_validation_supported | True if this gap length/configuration has validation-grade support from the artificial-gap pool |
| event_caveat | Free-text note on event-day handling; no event-specific bias correction is applied in this output |
| quantile_calibrated | Whether the quantile outputs have been calibration-checked |
| scenario_only_256day | True if this row belongs to the 256-day scenario-only gap |

## results_public/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv

Full daily series (observed + reconstructed) under the engineered hybrid
pipeline's validation-aware method assignment.

| Column | Description |
|---|---|
| date | Calendar date |
| chl_mean | Observed daily mean chlorophyll, where available (same as in the daily target table) |
| target_eligible_default | Eligibility flag, as in the daily target table |
| observed_chl | Observed chlorophyll value (NaN if not observed) |
| is_observed | True if the day was an eligible observation |
| reconstructed_chl | Reconstructed value for this day (NaN if observed or not reconstructed) |
| final_chl | Combined series: observed_chl where observed, else reconstructed_chl |
| is_reconstructed | True if this day's final_chl came from reconstruction |
| gap_id | Real gap identifier, if this day falls within a real gap |
| gap_length | Length of the gap this day belongs to |
| method | Which method family produced the reconstruction for this day (state_space_kalman / gaussian_process / gap_edge_residual_model) |
| method_sensitivity_variant | Alternate method used for a sensitivity comparison, where applicable |
| reconstructed_chl_sensitivity_variant | Reconstructed value under the sensitivity-variant method |
| uncertainty_lower, uncertainty_upper | Uncertainty band around the reconstruction |
| quality_flag | Categorical quality/confidence label for the reconstruction |
| extrapolation_flag | True if this reconstruction extrapolates beyond the validated gap-length envelope |
| notes | Free-text notes, including method assignment rationale |

## results_public/chlorophyll/chlorophyll_benchmark_summary.csv

One row per pairwise method comparison at the "all gaps" stratum, from
paired bootstrap testing over the artificial-gap pool.

| Column | Description |
|---|---|
| method_family | Coarse method family label (e.g. foundation_model_zero_shot, baseline) |
| method_public_name | Public name of the method being evaluated |
| compared_against_public_name | Public name of the comparison method |
| stratum | Subset of gaps used (e.g. "all_gaps") |
| n_gaps | Number of gaps contributing to the comparison |
| n_bootstrap_replicates | Number of bootstrap resamples used to build confidence intervals |
| metric | Metric being compared (day_weighted_mae) |
| method_value, comparison_value | Point estimate of the metric for each method |
| delta | method_value - comparison_value |
| ci_lo, ci_hi | Bootstrap confidence interval on delta |
| interpretation | Categorical summary (e.g. significant_improvement, directional_not_significant) |
| evidence_tier | Evidence-hierarchy label for this comparison |

## results_public/chlorophyll/chlorophyll_artificial_gap_scores.csv

Tidy long-format table of method performance by stratum (gap length,
season, or event status).

| Column | Description |
|---|---|
| method_public_name | Public method name |
| stratum_type | Which stratification this row belongs to (gap_length / season / event_status) |
| stratum_value | The specific stratum value (e.g. "L=7", "DJF", "event=True") |
| n_gaps | Number of gaps in this stratum |
| day_weighted_mae | Day-weighted mean absolute error |
| rmse | Root mean squared error |
| median_ae | Median absolute error |

## results_public/chlorophyll/chlorophyll_covariate_mechanism_summary.csv

Two sections in one file, distinguished by `table_section`.

`table_section == covariate_arm_performance`: one row per TS-ICL covariate
configuration.

| Column | Description |
|---|---|
| covariate_public_name | Public description of the covariate configuration |
| n_gaps | Number of gaps evaluated |
| mae_mean, rmse_mean, bias_mean | Mean error metrics across the artificial-gap pool |

`table_section == placebo_robustness_test`: results of a placebo/control
test, comparing a curated physical-forcing covariate configuration against
randomized/shuffled control covariates (to check the covariate's effect is
a real mechanism and not an artifact).

| Column | Description |
|---|---|
| comparison | Description of the comparison being made |
| context_mode | Context-window configuration used for the test |
| n_gaps | Number of gaps evaluated |
| delta_mae_mean, delta_mae_ci_lo, delta_mae_ci_hi | Mean and confidence interval of the MAE difference |
| significant | Whether the difference is statistically significant |

## results_public/chlorophyll/chlorophyll_event_performance_summary.csv

One row per method, comparing event-day vs. non-event-day performance.

| Column | Description |
|---|---|
| method_public_name | Public method name |
| n_event_days, n_nonevent_days | Number of event / non-event days evaluated |
| mae_event_days, mae_nonevent_days | MAE on event / non-event days |
| event_penalty_mae_event_minus_nonevent | mae_event_days - mae_nonevent_days |
| mae_sustained_event | MAE restricted to sustained (multi-day) events |
| p90_abs_error_event, p90_abs_error_nonevent | 90th-percentile absolute error on event / non-event days |
| amplitude_bias_log10_event_days | Mean signed bias (log10 scale) on event days; negative = under-prediction |
| false_flatten_rate_event_days | Fraction of event days where the method failed to predict any elevated value |
| false_bloom_rate_nonevent_days | Fraction of non-event days where the method falsely predicted an elevated value |

## results_public/chlorophyll/chlorophyll_real_gap_candidate_outputs.csv

One row per real gap, summarizing both candidate methods' outputs.

| Column | Description |
|---|---|
| gap_id, start_date, end_date, length_days, gap_class, seasons | As in the real gap inventory |
| extrapolation_beyond_validation | Whether this gap exceeds the validated gap-length envelope |
| scenario_only_256day | True for the 256-day scenario-only gap |
| tsicl_satellite_proxy_mean_pred_chl | Mean predicted chlorophyll over the gap, TS-ICL satellite-proxy configuration |
| tsicl_satellite_proxy_n_days | Number of days with a TS-ICL prediction for this gap |
| engineered_hybrid_mean_reconstructed_chl | Mean reconstructed chlorophyll over the gap, engineered hybrid pipeline |
| engineered_hybrid_method | Dominant method family used by the engineered hybrid pipeline for this gap |
| engineered_hybrid_n_days | Number of days reconstructed by the engineered hybrid pipeline for this gap |
| note_artificial_validation | Standing note: artificial-gap validation supports method ranking |
| note_real_gap_caveat | Standing note: real gaps have no withheld ground truth |
| note_256day_scenario | Note flagging the 256-day gap as scenario-only, where applicable |

## results_public/chlorophyll/chlorophyll_real_gap_candidate_outputs_daily.csv

Full per-day join of the daily target table against both candidate
reconstruction methods and the real-gap inventory, one row per calendar
day (3,988 rows, same date range as `chlorophyll_daily_target.csv`). This
is the day-level companion to `chlorophyll_real_gap_candidate_outputs.csv`
(which is one row per gap); use this table when a day-by-day comparison of
both candidate methods is needed, e.g. for plotting a continuous series.

| Column | Description |
|---|---|
| date | Calendar date |
| target_eligible_default | Eligibility flag, as in the daily target table |
| observed_chl_mean | Observed daily mean chlorophyll, where available (NaN if not observed/eligible) |
| is_observed | True if the day is an eligible observation (`target_eligible_default` True and a value present) |
| tsicl_gap_id | Real-gap identifier this day belongs to, as labeled in the TS-ICL output (NaN if outside any real gap) |
| tsicl_satellite_proxy_pred_chl | TS-ICL satellite-proxy candidate prediction, chlorophyll units (NaN if not predicted for this day) |
| tsicl_satellite_proxy_pred_log10_chl | Same prediction, log10 scale |
| tsicl_artificial_validation_supported | True if this gap length/configuration has validation-grade support from the artificial-gap pool |
| tsicl_scenario_only_256day | True if this row's TS-ICL prediction belongs to the 256-day scenario-only gap |
| hybrid_gap_id | Real-gap identifier this day belongs to, as labeled in the engineered hybrid output |
| hybrid_gap_length | Length (days) of the real gap this day belongs to, per the engineered hybrid output |
| engineered_hybrid_reconstructed_chl | Engineered hybrid candidate reconstruction, chlorophyll units (NaN if not reconstructed) |
| engineered_hybrid_final_chl | Engineered hybrid's combined series value (observed where observed, else reconstructed) |
| engineered_hybrid_is_reconstructed | True if this day's value came from reconstruction rather than observation, per the engineered hybrid output |
| engineered_hybrid_method | Method family used by the engineered hybrid pipeline for this day |
| engineered_hybrid_extrapolation_flag | True if this reconstruction extrapolates beyond the validated gap-length envelope |
| real_gap_id | Real-gap identifier this date falls within, joined from the real gap inventory (NaN if not in any real gap) |
| real_gap_length_days | Length (days) of that real gap, from the inventory |
| real_gap_class | Categorical length bucket of that real gap, from the inventory |
| real_gap_extrapolation_beyond_validation | Whether that real gap exceeds the validated gap-length envelope, from the inventory |
| evidence_label | One of `observed_day`, `real gap — no withheld truth`, or `scenario-only, outside validated gap-length envelope` -- see `docs/evidence_hierarchy.md` before using any non-observed row as if it were validated |
| scenario_only_256day_gap | True if this date falls within the single 256-day real gap (the longest in the record, far outside the validated gap-length envelope) |

Coverage notes: all 3,988 target dates are present as rows. The TS-ICL and
engineered hybrid candidate columns are populated only for the 976 days
that fall within a real gap (720 days in gaps up to 71 days long, plus 256
days in the single scenario-only gap); the engineered hybrid pipeline's
reconstructed-value column has 17 fewer non-null rows than TS-ICL's (959
vs. 976) for days where that pipeline did not assign a reconstruction.
`real_gap_id`, `tsicl_gap_id`, and `hybrid_gap_id` agree exactly everywhere
all three are populated -- there were no schema/date-range mismatches
requiring a partial or approximate join.
