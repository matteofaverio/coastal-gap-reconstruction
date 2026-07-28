# Coastal Gap Reconstruction

A reproducible research archive and reusable benchmark workflow for
retrospective reconstruction (imputation) of gaps in coastal sensor time
series.

## 1. What this repository is, and why it matters

Coastal monitoring sensors go offline -- biofouling, maintenance,
transmission failures, storms. When that happens, the resulting gap in the
record is a problem for anyone downstream who needs a continuous daily
series: trend analysis, event detection, model forcing, reporting. This
repository documents a general methodology for filling those gaps, and
applies it to two real sensors at a coastal in-situ monitoring station
(Tongoy Balsa, Chile):

- **Case Study 1: chlorophyll-a**, the primary case study, daily mean
  chlorophyll-a from 2015 to 2026.
- **Case Study 2: dissolved oxygen**, a second sensor at the same site,
  used to test whether the same workflow and findings transfer to a
  different variable. See `notebooks/10_oxygen_case_study.ipynb` and
  `notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` for the
  transfer template that was used to build it.

Methods evaluated range from simple statistical baselines (climatology,
persistence, linear interpolation) through classical machine learning to a
zero-shot time-series foundation model (TS-ICL). The core design principle
throughout is to separate **validation-grade evidence** (built from
artificial gaps with known, withheld ground truth) from
**plausibility-only candidate outputs** (real gaps, where no ground truth
exists). Section 3 below explains this distinction in detail -- it is the
single most important thing to understand before reading any result in
this repository.

### What this repository is NOT

- It is **not a forecasting tool**. Several methods here (linear
  interpolation, gap-edge residual correction, parts of the engineered
  hybrid pipeline) require an observation *after* the gap to work, which
  makes them retrospective/diagnostic only -- not usable for predicting an
  unobserved future. See `docs/methodology/model_families.md` and
  `notebooks/05_gap_edge_residual_models.ipynb`.
- It does **not** produce a single "correct" filled series. Several
  candidate reconstruction methods are released side by side; they
  sometimes disagree, and only the artificial-gap comparison is
  validation-grade evidence about which is more accurate. Treat every
  reconstructed value as a candidate, not a ground truth.

## 2. The pipeline (read this before looking at any result)

Every result in this repository comes from the same reusable, three-stage
pipeline. Understanding the pipeline first makes the findings in Section 4
meaningful instead of just a list of numbers.

**Stage 1 -- Artificial-gap validation.** Real gaps have no withheld truth
to score against (the sensor was actually offline). To get quantitative
evidence about which reconstruction method is more accurate, the pipeline
instead carves *artificial* gaps out of stretches of the record that are
actually observed: it deliberately hides the true values for a
contiguous run of days, keeps them secret, runs every candidate method
against the gappy series, and then scores each method's reconstruction
against the value that was withheld. This is repeated across many gap
lengths (from 1 day up to several weeks), seasons, and starting points, so
that performance can be broken down by gap length and regime rather than
reported as one aggregate number. See
`docs/methodology/validation_protocol.md` and
`notebooks/02_artificial_gap_validation.ipynb`.

**Stage 2 -- Method comparison.** Every method -- from a monthly-climatology
baseline up to TS-ICL -- is run through the exact same artificial-gap
protocol and scored with the same metric (mean absolute error), with
paired bootstrap confidence intervals on the difference between any two
methods. This is what makes the comparison in Section 4 below meaningful:
every method earns its ranking under the same test, not a cherry-picked
one. See `docs/methodology/model_families.md` and
`notebooks/07_benchmark_comparison_and_diagnostics.ipynb`.

**Stage 3 -- Applying the validated methods to real gaps.** Once a method's
accuracy profile is understood from Stage 2, it can be applied to the
record's actual (real) gaps to produce a filled series. Because there is
no withheld truth for a real gap, this step never produces new validation
evidence -- it produces a *candidate* reconstruction, whose plausibility is
judged only by comparison to the artificial-gap behavior of the same
method at a similar gap length. See Section 3 and
`notebooks/08_real_gap_candidate_reconstructions.ipynb`.

The live demo in Section 5 walks through all three stages end to end on a
single example gap, including a live run of TS-ICL.

## 3. Evidence hierarchy

Not every number in this repository carries the same weight. Read
`docs/evidence_hierarchy.md` before drawing conclusions from any table
here. In short:

1. **Artificial-gap validation results** are the only validation-grade
   evidence in this repository -- the only place a true value was withheld
   and can be compared against.
2. **Real-gap candidate outputs** are plausibility-only -- there is no
   withheld ground truth for any real (naturally occurring) gap in this
   record.
3. The single longest real gap in the chlorophyll record (256 days) is
   scenario-only, far outside the validated gap-length envelope
   (chlorophyll is validated up to 60 days).

## 4. Headline findings

These are qualitative summaries; see the linked tables and
`docs/methodology/` for exact numbers and confidence intervals. All
findings below are based on artificial-gap validation (Section 2, Stage 1
-- Section 3, tier 1), not on real-gap candidate outputs.

- **Linear interpolation is a strong baseline**, especially at short gap
  lengths, and is hard to beat without a genuinely better source of
  information.
- **External-covariate-only tabular models underperform interpolation as
  direct reconstructors**, for both chlorophyll and dissolved oxygen. Using
  only calendar/meteorological/satellite predictors -- without any
  information from the target sensor's own recent history -- was not
  enough to beat simple interpolation in this low-data, local setting. See
  `notebooks/04_engineered_tabular_models.ipynb`.
- **TS-ICL with the right covariate is the strongest candidate method
  overall.** A zero-shot run of the TS-ICL time-series foundation model,
  conditioned on an appropriate covariate (a satellite chlorophyll proxy
  for chlorophyll; a physical-forcing bundle of sea-surface temperature,
  wind, solar radiation, and current for oxygen), is the only method that
  clearly and consistently improves on linear interpolation across most
  validated gap lengths for both case studies. See
  `docs/methodology/tsicl_usage.md`,
  `results_public/chlorophyll/chlorophyll_benchmark_summary.csv`, and
  `results_public/oxygen/oxygen_benchmark_by_length.csv`.
- **Event and distribution-tail regimes remain hard for every method.**
  Every method under-predicts the amplitude of high-chlorophyll ("event")
  days -- see `docs/methodology/event_limitation.md`. For oxygen, TS-ICL's
  improvement over interpolation is not uniform across the distribution:
  it is worse than interpolation in both distribution tails, most
  severely during sustained high-oxygen runs. See
  `notebooks/10_oxygen_case_study.ipynb`.
- **Case Study 2 (oxygen) shows the methodology transfers, but not every
  finding transfers unchanged.** The pipeline, evidence hierarchy, and
  general ranking of method families reproduce on a second sensor, but the
  best covariate configuration and the tail behavior differ from
  chlorophyll -- see Section 12 below.

An "event" day (chlorophyll) is defined purely statistically, as a day
whose true value exceeds the 90th percentile of all eligible observed
values -- this is an empirical validation-stratification threshold, not an
ecological or regulatory bloom threshold. The same is true of the oxygen
distribution-tail bands (p10, p90, etc.) referenced above: they are
empirical percentiles of the observed oxygen distribution used to stratify
validation performance, not hypoxia thresholds or any biological/
regulatory criterion.

## 5. Live demo

The fastest way to see this pipeline work is to run it: `demo/` reconstructs
a 14-day chlorophyll gap with eight methods, including a **live, zero-shot
run of TS-ICL** (no training or fine-tuning on this data), in a self-contained
virtual environment.

```bash
bash demo/run_demo.sh
```

This creates an isolated, git-ignored virtual environment, installs
everything needed (including the real `tsicl` package and its pretrained
checkpoint), and executes the demo notebook end to end. Tested time for a
first run, including installing packages: under 2 minutes.

**Everything about the demo -- exact commands, what "live" vs. "cached
fallback" TS-ICL output means, package versions, and TS-ICL's license and
checkpoint details -- is documented once, in `demo/README.md`.** That file
is the single source of truth for running the demo; this section is
intentionally just a pointer to it, so there is nothing here that can
drift out of sync with what was actually tested.

In brief on TS-ICL, since colleagues reasonably ask about this before
running anything: TS-ICL (https://github.com/EDF-Lab/ts-icl) is installed
from PyPI (`pip install tsicl`), and on first use downloads its public,
non-gated pretrained checkpoint (~209 MB) from a public Hugging Face
repository, cached locally afterward. It is released under the TS-ICL
Non-Commercial License v1.0 (EDF SA), which explicitly permits
non-commercial research, evaluation, and benchmarking -- including on
third-party data, which is what this demo and this repository's benchmark
both do. It forbids using TS-ICL's outputs to train a competing model, and
forbids redistributing the model itself as a hosted/SaaS service. See
`demo/README.md` for the full license note and exact tested commands.

## 6. Repository structure

```
config/contracts/        machine-checkable target/gap-pool definitions
data_public/chlorophyll/ daily target, predictor features, gap inventories
data_public/oxygen/      daily target, validation gaps, real-gap inventory
demo/                    live, runnable demo (see Section 5)
results_public/chlorophyll/  benchmark scores, reconstructions, mechanism tables
results_public/oxygen/   benchmark scores, paired deltas, tail diagnostics
figures/chlorophyll/     key figures referenced in the docs
figures/oxygen/          key figures, reproduced from the published report
manuscript/report/       final report PDF + full LaTeX source (compilable)
manuscript/presentation/ final slide-deck PDF + full LaTeX source (compilable)
notebooks/               numbered, runnable notebooks (see below)
src/coastal_gap_reconstruction/  reusable Python utilities
tests/                   smoke tests that execute the demo and public notebooks
docs/methodology/        plain-language write-ups of each methodological choice
docs/data_dictionary.md  column-by-column definitions for every public CSV
docs/evidence_hierarchy.md
docs/data_sources_and_attribution.md
```

The intended reading path is: this README, then `demo/` for a hands-on
walkthrough, then the numbered `notebooks/` for full depth, then `docs/`
for the underlying methodological decisions and column-level definitions.

## 7. Installing the core package (without the demo)

If you only need the reusable Python utilities (`src/coastal_gap_reconstruction/`)
rather than the live TS-ICL demo:

```bash
git clone https://github.com/matteofaverio/coastal-gap-reconstruction
cd coastal-gap-reconstruction
pip install -e ".[notebooks]"
jupyter lab notebooks/01_target_and_gap_audit.ipynb
```

This installs pandas/numpy plus the notebook extras (matplotlib, scipy,
scikit-learn, jupyter) but does **not** install `tsicl` or `torch` -- for
the live TS-ICL demo, use `demo/run_demo.sh` (Section 5) instead, in its
own isolated environment.

## 8. Manuscript

`manuscript/report/REPORT.pdf` is the final written report (chlorophyll
Case Study 1 + oxygen Case Study 2), and
`manuscript/presentation/PRESENTATION.pdf` is the accompanying slide deck.
Both ship with their full LaTeX source (`main.tex`, sections, figures,
bibliography) so they can be recompiled from scratch -- see
`manuscript/README.md` for the exact commands. The manuscript is the
authoritative narrative synthesis of the data and results published
elsewhere in this repository.

## 9. Data included

- `chlorophyll_daily_target.csv` -- daily rows from 2015-07-01 to
  2026-05-31, with eligibility flags, summary statistics, and QA counters.
- `chlorophyll_predictor_features_curated.csv` -- a curated external/
  spatial predictor feature table (calendar, SST, wind, satellite
  chlorophyll proxy, upwelling indices).
- `chlorophyll_validation_gaps.csv` -- the canonical artificial-gap
  validation pool (hundreds of gaps across multiple lengths and seasons).
- `chlorophyll_real_gap_inventory.csv` -- the real (naturally occurring)
  gaps in the record.
- `data_public/oxygen/` -- the equivalent tables for the oxygen case
  study.

Result tables under `results_public/chlorophyll/` include per-gap and
per-day candidate reconstructions; in particular
`chlorophyll_real_gap_candidate_outputs_daily.csv` is a full day-by-day
join of the daily target table against both candidate methods and the
real-gap inventory (see Section 11 below and `docs/data_dictionary.md`).

See `docs/data_dictionary.md` for full column definitions, and
`docs/data_sources_and_attribution.md` /
`DATA_LICENSE_AND_ATTRIBUTION.md` for required attribution of the
underlying CEAZAMet/CEAZA, NASA/PO.DAAC MUR SST, and Copernicus/CMEMS data.
Note that data and derived results are **not** covered by this
repository's MIT code license -- see Section 15 below.

## 10. Reproducing the chlorophyll benchmark

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
10. `10_oxygen_case_study.ipynb` -- Case Study 2, the worked result of
    following notebook 9's checklist.

Notebooks 01, 02, 03, 04, 07, 08, and 10 are fully executable against the
public data in this repository as-is. Notebook 05's illustrative cells
(loading and plotting the public benchmark tables/figure) are also
executable; the full gap-edge residual training pipeline itself is not
republished, by design (see the notebook for why). Notebook 06 is a usage
template requiring a separate TS-ICL installation (see `demo/` for a
turnkey environment that provides this). Notebook 09 is a markdown
checklist. `tests/` contains smoke tests that confirm all directly
executable notebooks still run cleanly end to end.

## 11. Using TS-ICL

See `docs/methodology/tsicl_usage.md`,
`notebooks/06_tsicl_zero_shot_imputation.ipynb`, and `demo/README.md` (for
a fully turnkey, tested installation). TS-ICL code and weights are under
the original authors' license -- review it before use (summarized in
Section 5 above). This repository does not vendor TS-ICL itself, only a
thin calling-convention helper
(`src/coastal_gap_reconstruction/tsicl_helpers.py`).

## 12. Candidate real-gap reconstructions

`results_public/chlorophyll/chlorophyll_real_gap_candidate_outputs.csv`
(one row per real gap) and
`chlorophyll_real_gap_candidate_outputs_daily.csv` (one row per calendar
day, joining the daily target table against both candidate methods and
the real-gap inventory) provide plausible filled values for every real gap
in the chlorophyll record, from both the TS-ICL satellite-proxy
configuration and the engineered hybrid pipeline.

**These are explicitly not validation evidence.** There is no withheld
ground truth for any real gap -- the sensor was genuinely offline, so
there is nothing to score these candidate values against. Do not use them
to claim one method is more accurate than another, and do not treat them
as a "true" reconstructed series. They are useful only as plausible
filled-in values for plotting/rough magnitude checks, and as a qualitative
sanity check against artificial-gap behavior at similar lengths. See
Section 3 and `docs/evidence_hierarchy.md` before using either table.

## 13. Adapting the workflow to another sensor

`notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` is a checklist
for applying this same workflow to a new sensor variable, and
`notebooks/10_oxygen_case_study.ipynb` is the worked result: dissolved
oxygen (Case Study 2) at the same station. Both reference
`docs/methodology/target_and_gap_construction.md` and
`docs/methodology/validation_protocol.md` as the scientific decisions that
must be re-made deliberately for a new variable, not inherited by default
-- in particular, oxygen's eligibility threshold and gap-length range
(L=1-30 days vs. chlorophyll's L=1-60) were set independently based on its
own missingness structure.

## 14. Limitations

- **Event/high-chlorophyll performance is unresolved.** Every method
  under-predicts on high-chlorophyll days; see
  `docs/methodology/event_limitation.md` for numbers.
- **Oxygen's distribution tails are also unresolved.** TS-ICL's pooled
  improvement over interpolation on oxygen does not hold uniformly in
  either distribution tail; see `notebooks/10_oxygen_case_study.ipynb`.
- **Real-gap outputs are not validation evidence.** There is no withheld
  ground truth for any real gap; treat candidate reconstructions as
  plausible fill values only, with the chlorophyll record's 256-day gap
  as illustrative-only.

## 15. Citation and acknowledgments

See `CITATION.cff` for citing this repository, and
`docs/data_sources_and_attribution.md` for required attribution of
CEAZAMet/CEAZA in-situ data, NASA/PO.DAAC MUR SST, and Copernicus/CMEMS
products used as predictors in this benchmark.

## 16. License

Code in this repository (`src/`, notebook code cells, configuration files)
is MIT licensed -- see `LICENSE`. Data and derived results under
`data_public/`, `results_public/`, and `figures/` are **not** MIT licensed
and retain attribution obligations to their original providers; see
`DATA_LICENSE_AND_ATTRIBUTION.md` for the license-focused summary and
`docs/data_sources_and_attribution.md` for the detailed narrative version.
Including data here is not a claim of ownership over upstream datasets.
TS-ICL itself (used in `demo/` and `notebooks/06_tsicl_zero_shot_imputation.ipynb`)
is separately licensed by its original authors -- see Section 5 above and
`demo/README.md`.
