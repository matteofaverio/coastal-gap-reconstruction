# Target and gap construction

## Source data

The reconstruction target is daily mean chlorophyll-a concentration at a
coastal in-situ monitoring station (Tongoy Balsa, Chile), built from hourly
sensor readings. The raw hourly data come from a Chilean coastal monitoring
network; see `docs/data_sources_and_attribution.md` for required attribution
language.

## Daily aggregation

Each calendar day is summarized from its (up to 24) hourly chlorophyll
readings into:

- `chl_mean` -- the primary reconstruction target, the daily mean of all
  valid hourly readings;
- `chl_median`, `chl_std`, `chl_min`, `chl_max`, and selected quantiles
  (`chl_q10`, `chl_q25`, `chl_q75`, `chl_q90`) -- preserved for later
  within-day variability analysis;
- `valid_hours` and `coverage_fraction` -- how many of the expected 24
  hourly readings were usable that day;
- a small set of QA counters (`negative_chl_count`, `invalid_chl_count`,
  `longest_invalid_run_hours`, etc.) describing within-day data quality.

## Eligibility rule

A day is only treated as a trustworthy daily-mean observation if it has at
least 18 valid hourly readings out of the expected 24 (`target_eligible_default
== True`). Days below this threshold are not used to compute "true" values
for validation scoring, and are not used to fit baselines or other
reconstruction methods -- they are treated the same as missing days even if
a partial daily mean could technically be computed.

Several alternative eligibility thresholds (`eligible_ge_12`,
`eligible_ge_17`, `eligible_ge_22`) are also carried in the target table for
sensitivity checks, but `target_eligible_default` (the 18-hour threshold) is
the convention used throughout this benchmark.

## Record-level numbers

As of the snapshot included in this repository:

- 3,988 daily rows, spanning 2015-07-01 to 2026-05-31;
- 3,012 eligible days (75.5% of the record);
- 128 real (naturally occurring) gaps in the eligible-day record, ranging
  from single missing days up to one 256-day gap (a known sensor outage,
  the longest in the record and well outside the validated gap-length
  envelope -- see `docs/evidence_hierarchy.md`).

## Scoring scale

`chl_mean` is strongly right-skewed (a small number of high-chlorophyll
event days dominate the physical-unit range). The report, the released
benchmark tables (`results_public/chlorophyll/`), and every chlorophyll
notebook that scores against them use `log10(chl_mean)` as the scoring
scale, not physical `chl_mean` (mg m^-3) directly. Physical-unit means,
medians, and quantiles are still carried in the daily target table for
display and diagnostics, but MAE/RMSE/bias figures quoted anywhere in this
repository are on the log10 scale unless a table or figure explicitly says
"physical units" or "mg m^-3". Do not compare a physical-scale MAE against
a log10-scale MAE — they are different metrics computed on different data,
not two views of the same number. `notebooks/03_baselines.ipynb` derives a
`chl_log10` column before scoring for exactly this reason.

## Gap definitions

Two distinct notions of "gap" are used throughout this repository:

- **Real gaps**: naturally occurring stretches of consecutive non-eligible
  days in the observed record (sensor outages, transmission failures,
  maintenance, etc.). These have no withheld ground truth -- by definition
  we never observed what the true values would have been.
- **Artificial gaps**: stretches of days that ARE observed and eligible,
  but whose target values are deliberately masked for validation purposes
  (see `docs/methodology/validation_protocol.md`). Because the true value
  is known but withheld from the candidate method, these can be scored.
