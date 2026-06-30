# Coastal Gap Reconstruction

A reproducible research archive and reusable benchmark workflow for
retrospective reconstruction (imputation) of sparse coastal sensor time
series.

## 1. What this repository is

This repository documents a methodology for filling gaps in coastal
sensor records where the sensor was offline, using everything from simple
statistical baselines to a zero-shot time-series foundation model. **Case
Study 1** is daily mean chlorophyll-a at a coastal in-situ monitoring
station (Tongoy Balsa, Chile). **Case Study 2**, dissolved oxygen at the
same site, is planned but not yet built -- see
`notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` for the template
that will be used to build it.

The core design principle throughout is to separate **validation-grade
evidence** (built from artificial gaps with known, withheld ground truth)
from **plausibility-only candidate outputs** (real gaps, where no ground
truth exists). See section 3 below.

### What this repository is NOT

- It is **not a forecasting tool**. Several methods here (linear
  interpolation, gap-edge residual correction, parts of the engineered
  hybrid pipeline) require an observation *after* the gap to work, which
  makes them retrospective/diagnostic only -- not usable for predicting an
  unobserved future. See `docs/methodology/model_families.md` and
  `notebooks/05_gap_edge_residual_models.ipynb`.
- It does **not** produce a single "correct" filled series. Several
  candidate reconstruction methods are released side by side
  (`chlorophyll_reconstruction_tsicl_satellite_proxy.csv`,
  `chlorophyll_reconstruction_engineered_hybrid.csv`); they sometimes
  disagree, and only the artificial-gap comparison is validation-grade
  evidence about which is more accurate. Treat every reconstructed value
  as a candidate, not a ground truth.

## 2. Key finding

Across the canonical 681-gap artificial-gap validation pool, a zero-shot
time-series foundation model (TS-ICL) conditioned on a satellite
chlorophyll proxy covariate is the first method in this benchmark to show
a statistically significant improvement in mean absolute error over linear
interpolation, a Gaussian process baseline, and a gap-edge engineered tree
model, across most gap lengths (3 to 45 days). The improvement is **not**
statistically significant at the shortest gap length (1 day, where
interpolation is already near-optimal) or the longest validated gap length
(60 days, where uncertainty widens). See
`results_public/chlorophyll/chlorophyll_benchmark_summary.csv` and
`results_public/chlorophyll/chlorophyll_artificial_gap_scores.csv`.

This result should be read alongside an important limitation: every method
in this benchmark, including the leading TS-ICL configuration,
systematically under-predicts high-chlorophyll ("event") days. See section
11 and `docs/methodology/event_limitation.md`.

## 3. Evidence hierarchy

Not every number in this repository carries the same weight. Read
`docs/evidence_hierarchy.md` before drawing conclusions from any table
here. In short:

1. **Artificial-gap validation results** (gap lengths 1-60 days) are the
   only validation-grade evidence -- the only place a true value was
   withheld and can be compared against.
2. **Real-gap candidate outputs** are plausibility-only -- there is no
   withheld ground truth for any of the 128 real gaps in this record.
3. The **256-day real gap** is scenario-only, far outside the validated
   gap-length envelope (max validated: 60 days).

## 4. Quick start

```bash
git clone https://github.com/matteofaverio/coastal-gap-reconstruction
cd coastal-gap-reconstruction
pip install -e ".[notebooks]"
jupyter lab notebooks/01_target_and_gap_audit.ipynb
```

## 5. Repository structure

```
config/contracts/        machine-checkable target/gap-pool definitions
data_public/chlorophyll/ daily target, predictor features, gap inventories
results_public/chlorophyll/  benchmark scores, reconstructions, mechanism tables
figures/chlorophyll/     key figures referenced in the docs
notebooks/               numbered, runnable notebooks (see below)
src/coastal_gap_reconstruction/  reusable Python utilities
docs/methodology/        plain-language write-ups of each methodological choice
docs/data_dictionary.md  column-by-column definitions for every public CSV
docs/evidence_hierarchy.md
docs/data_sources_and_attribution.md
```

## 6. Data included

- `chlorophyll_daily_target.csv` -- 3,988 daily rows, 2015-07-01 to
  2026-05-31, with eligibility flags, summary statistics, and QA counters.
- `chlorophyll_predictor_features_curated.csv` -- a curated external/
  spatial predictor feature table (calendar, SST, wind, satellite
  chlorophyll proxy, upwelling indices).
- `chlorophyll_validation_gaps.csv` -- the canonical 681-gap artificial
  validation pool.
- `chlorophyll_real_gap_inventory.csv` -- 128 real (naturally occurring)
  gaps in the record.

Result tables under `results_public/chlorophyll/` include per-gap and
per-day candidate reconstructions; in particular
`chlorophyll_real_gap_candidate_outputs_daily.csv` is a full day-by-day
join of the daily target table against both candidate methods and the
real-gap inventory (see section 9 below and `docs/data_dictionary.md`).

See `docs/data_dictionary.md` for full column definitions, and
`docs/data_sources_and_attribution.md` /
`DATA_LICENSE_AND_ATTRIBUTION.md` for required attribution of the
underlying CEAZAMet/CEAZA, NASA/PO.DAAC MUR SST, and Copernicus/CMEMS data.
Note that data and derived results are **not** covered by this
repository's MIT code license -- see section 13 below.

## 7. Reproducing the chlorophyll benchmark

Run the notebooks in order:

1. `01_target_and_gap_audit.ipynb` -- coverage and missingness audit.
2. `02_artificial_gap_validation.ipynb` -- validation protocol and gap pool.
3. `03_baselines.ipynb` -- climatology/persistence/interpolation, scored.
4. `04_engineered_tabular_models.ipynb` -- external-predictor tabular
   models: how engineered feature rows are built and why external
   predictors alone did not clearly beat interpolation in this low-data
   local setting.
5. `05_gap_edge_residual_models.ipynb` -- gap-edge residual correction
   models: predicting a correction over linear interpolation from both
   gap edges, and how this compares to TS-ICL as a classical comparator.
6. `06_tsicl_zero_shot_imputation.ipynb` -- TS-ICL usage template.
7. `07_benchmark_comparison_and_diagnostics.ipynb` -- full method comparison.
8. `08_real_gap_candidate_reconstructions.ipynb` -- real-gap candidate outputs.
9. `09_adapting_the_workflow_to_a_new_sensor.ipynb` -- checklist for a new case study.

Notebooks 01, 02, 03, 04, 07, and 08 are fully executable against the
public data in this repository as-is. Notebook 05's illustrative cells
(loading and plotting the public benchmark tables/figure) are also
executable; the full gap-edge residual training pipeline itself is not
republished, by design (see the notebook for why). Notebook 06 is a usage
template requiring a separate TS-ICL installation. Notebook 09 is a
markdown checklist.

## 8. Using TS-ICL

See `docs/methodology/tsicl_usage.md` and
`notebooks/06_tsicl_zero_shot_imputation.ipynb`. TS-ICL code and weights
are under the original authors' license -- review it before use. This
repository does not vendor TS-ICL itself, only a thin calling-convention
helper (`src/coastal_gap_reconstruction/tsicl_helpers.py`).

## 9. Candidate real-gap reconstructions

`results_public/chlorophyll/chlorophyll_real_gap_candidate_outputs.csv`
(one row per real gap) and
`chlorophyll_real_gap_candidate_outputs_daily.csv` (one row per calendar
day, joining the daily target table against both candidate methods and
the real-gap inventory) provide plausible filled values for all 128 real
gaps, from both the TS-ICL satellite-proxy configuration and the
engineered hybrid pipeline.

**These are explicitly not validation evidence.** There is no withheld
ground truth for any real gap -- the sensor was genuinely offline, so
there is nothing to score these candidate values against. Do not use them
to claim one method is more accurate than another, and do not treat them
as a "true" reconstructed series. They are useful only as plausible
filled-in values for plotting/rough magnitude checks, and as a qualitative
sanity check against artificial-gap behavior at similar lengths. See
section 3 and `docs/evidence_hierarchy.md` before using either table.

## 10. Adapting the workflow to another sensor

`notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` is a checklist
for applying this same workflow to a new sensor variable (e.g. dissolved
oxygen, the planned Case Study 2). It references
`docs/methodology/target_and_gap_construction.md` and
`docs/methodology/validation_protocol.md` as the scientific decisions that
must be re-made deliberately for a new variable, not inherited by default.

## 11. Limitations

- **Event/high-chlorophyll performance is unresolved.** Every method
  under-predicts on high-chlorophyll days; see
  `docs/methodology/event_limitation.md` for numbers.
- **Real-gap outputs are not validation evidence.** There is no withheld
  ground truth for any real gap; treat candidate reconstructions as
  plausible fill values only, with the 256-day gap as illustrative-only.

## 12. Citation and acknowledgments

See `CITATION.cff` for citing this repository, and
`docs/data_sources_and_attribution.md` for required attribution of
CEAZAMet/CEAZA in-situ data, NASA/PO.DAAC MUR SST, and Copernicus/CMEMS
products used as predictors in this benchmark.

## 13. License

Code in this repository (`src/`, notebook code cells, configuration files)
is MIT licensed -- see `LICENSE`. Data and derived results under
`data_public/`, `results_public/`, and `figures/` are **not** MIT licensed
and retain attribution obligations to their original providers; see
`DATA_LICENSE_AND_ATTRIBUTION.md` for the license-focused summary and
`docs/data_sources_and_attribution.md` for the detailed narrative version.
Including data here is not a claim of ownership over upstream datasets.
