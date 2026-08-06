# Methods

Summary of how each part of the pipeline works, with links to the exact
code that implements it. This is a reference for readers who want the
mechanics without reading the full report (`manuscript/report/`), which
remains the authoritative narrative and the place to check for the complete
methodological argument.

## Daily targets and QC

Each calendar day is summarized from its hourly sensor readings into a
daily mean (the reconstruction target), plus median/std/min/max/quantiles,
valid-hour counts, and QA counters. A day is "eligible" (trustworthy enough
to use, either as a scoring target or a training example) only if it has at
least 18 valid hourly readings out of 24. Chlorophyll scores on
`log10(chl_mean)` (strongly right-skewed target); oxygen scores on raw
mg/L (no transform -- oxygen legitimately approaches zero under hypoxia,
where log10 is unstable). See `docs/data_dictionary.md` for the full column
reference and `src/coastal_gap_reconstruction/data_loading.py`,
`daily_target.py`.

## Artificial-gap validation

Real gaps (naturally missing days) have no withheld ground truth, so every
method comparison in this repository instead uses **artificial gaps**:
stretches of *observed, eligible* days whose true value is deliberately
hidden, then scored against after reconstruction. Chlorophyll's canonical
pool has 681 gaps at lengths 1, 3, 7, 10, 14, 21, 30, 45, and 60 days;
oxygen's has 406 primary gaps at lengths 1, 3, 7, 10, 14, 21, 30 days plus
a small exploratory-extended tier at 45, 60, 90, 120 days (never used in
headline numbers). See `docs/evidence_and_limitations.md`
for why this is the only evidence tier scored against withheld observations, and
`experiments/chlorophyll/target_and_gap_pool.py` /
`experiments/oxygen/target_and_gap_pool.py` for the exact, tested pool
construction.

## Leakage controls

Only the target column is masked for an artificial gap -- external
predictor data at the same dates stays available, mirroring a real
deployment where the sensor fails but other sources keep reporting. Any
feature derived from the target's own history becomes missing if it would
reach into the hidden gap. A method's hyperparameters are always fit
excluding the hidden days currently being scored. See
`src/coastal_gap_reconstruction/leakage.py`,
`artificial_gap_validation.py`, and `tests/test_leakage_invariance.py`,
`tests/test_leakage_dependency_window.py`.

## Model ladder

- **Model 0 (baselines)**: monthly climatology, persistence, linear
  interpolation. `src/coastal_gap_reconstruction/baseline_imputation.py`.
  Linear interpolation is a genuinely strong reference at short gap
  lengths and is not forecast-safe (it needs a post-gap observation).
- **Classical tabular models**: Ridge/ElasticNet/ExtraTrees/HistGradientBoosting
  trained on external-only feature rows (calendar, meteorology, satellite/
  reanalysis products, no target history) -- they do not require any
  target-history observation inside or around the gap, so they can
  produce a prediction wherever the external predictors are available;
  their accuracy beyond the validated gap-length support (up to 60 days
  for chlorophyll, 30 days primary for oxygen) remains unverified.
  `experiments/chlorophyll/tabular_models.py`,
  `experiments/oxygen/classical_models.py`.
- **Gap-edge residual models**: predict a correction over linear
  interpolation using observed values at both edges of a gap. Only
  admissible when both edges are observed; retrospective by construction,
  never forecast-safe. `experiments/chlorophyll/gap_edge_models.py`.
- **Probabilistic sequence models**: a Gaussian process (Matern-3/2 kernel,
  time-only, reusable across both targets,
  `src/coastal_gap_reconstruction/gaussian_process.py`) and a state-space/
  Kalman local-level model (`experiments/chlorophyll/probabilistic_models.py`).
  The Kalman fit has a known degeneracy -- its maximum-likelihood
  observation-noise estimate collapses to near-zero, making its smoothed
  output numerically equal to linear interpolation on 93% of gaps in the
  full chlorophyll pool. This is documented, not fixed, and inherited by
  the engineered hybrid pipeline's L=4-29 segment (below).
- **TS-ICL**: a pretrained, zero-shot time-series foundation model (no
  fine-tuning on this station's data), optionally conditioned on covariate
  channels. The leading method in this benchmark under artificial-gap
  validation for chlorophyll (satellite-proxy covariate) and the first
  comparator to beat interpolation for oxygen (physical-covariate arm).
  `src/coastal_gap_reconstruction/tsicl_helpers.py` is the single
  authoritative calling layer, shared unmodified by both case studies, the
  demo, and every notebook. See `docs/reproducibility.md` for environment
  and provenance details.
- **Engineered hybrid pipeline** (chlorophyll only): combines the Gaussian
  process (L=1-3), the Kalman smoother (L=4-29), and the gap-edge residual
  model (L>=30) under one deterministic, length-routed assignment rule
  ("Rule D"). `experiments/chlorophyll/engineered_hybrid.py`,
  `select_real_gap_reconstruction.py`.

**Chlorophyll's two external-tabular protocols.** Two distinct evaluation
protocols exist under similar names: a plain external-only fit (no
gap-position information) and a matched-reference fit (external features
plus 5 structural meta-features describing gap length/position, never a
hidden target value). Only the matched-reference protocol has a released
row in the benchmark tables. Both are published as separate method IDs; see
the `experiments/chlorophyll/gap_edge_models.py` and `tabular_models.py`
module docstrings for the exact feature sets.

## Oxygen adaptation

The oxygen case study (Case Study 2) reuses every generic mechanic
(masking, baseline imputation, Gaussian process, TS-ICL calling layer, run
state/manifest bookkeeping) from `src/coastal_gap_reconstruction/` and
`experiments/chlorophyll/tsicl_run_state.py`/`tsicl_run_manifest.py`
unmodified -- none of that shared code needed any oxygen-specific change.
Oxygen-specific decisions (raw-mg/L scoring, predictor admissibility,
support size, tail diagnostics) live in `experiments/oxygen/benchmark_contract.py`.
See `docs/reproducibility.md` and `notebooks/06_oxygen_case_study.ipynb` for
the full adaptation story.

## Real-gap candidate assembly

Real (naturally occurring) gaps are inventoried by pure contiguous-run
detection over the eligibility column
(`experiments/chlorophyll/real_gap_inventory.py`,
`experiments/oxygen/real_gap_inventory.py`) -- never by reading a candidate
prediction file. For chlorophyll, two independent candidate reconstruction
methods exist (`engineered_hybrid`, method-selected via Rule D; and
`tsicl_satellite_proxy`, applied uniformly); a deterministic assembly step
(`experiments/chlorophyll/assemble_real_gap_candidates.py`) joins them with
validation, but neither is presented as "the" correct reconstruction across
both methods -- see `docs/evidence_and_limitations.md`. Oxygen has an
inventory only; no reconstruction-candidate generator exists for oxygen
real gaps (`experiments/oxygen/real_gap_contract.py`).

## Running these methods yourself

Every module above is executable, not just descriptive. See
`docs/reproducibility.md` for the exact commands, expected runtime, and
which results are frozen vs. regenerable.
