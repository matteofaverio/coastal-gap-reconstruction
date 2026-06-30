"""coastal_gap_reconstruction

Reusable utilities for retrospective reconstruction (imputation) of sparse
coastal sensor time series, developed against a daily chlorophyll-a benchmark
at a Chilean coastal monitoring station and intended to generalize to other
sensors and other coastal sites.

Modules:
    data_loading: load and standardize the public daily target / feature tables.
    gap_detection: find maximal runs of eligible (non-missing) days; locate
        real (naturally occurring) gaps in a daily series.
    artificial_gap_validation: construct and apply artificial gaps for
        held-out validation of reconstruction methods.
    baseline_imputation: climatology, persistence, and linear-interpolation
        baselines (Model 0 in the model ladder).
    scoring_metrics: MAE/RMSE/bias/correlation metrics for gap reconstruction,
        with aggregation by gap length, season, and event status.
    plotting: small helpers for the figure types used in this benchmark.
    tsicl_helpers: thin convenience wrappers for invoking a zero-shot
        time-series foundation model (TS-ICL) on a gappy series with
        optional covariates.
"""

__all__ = [
    "data_loading",
    "gap_detection",
    "artificial_gap_validation",
    "baseline_imputation",
    "scoring_metrics",
    "plotting",
    "tsicl_helpers",
]
