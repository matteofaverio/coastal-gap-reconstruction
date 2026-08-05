"""Classical and engineered oxygen reconstruction comparators: Model 0
baselines, external-only tabular LOCO models, and gap-edge/GP comparators.

All predictions are on the raw `oxygen_mean_mgL` scale (mg/L) -- oxygen never
uses a log10 transform (`benchmark_contract.TARGET_TRANSFORM == "identity"`),
unlike chlorophyll. Model 0 baselines and the GP comparator reuse the
target-agnostic shared mechanics directly
(`coastal_gap_reconstruction.baseline_imputation`,
`coastal_gap_reconstruction.gaussian_process`) with no oxygen-specific code
needed; the tabular and gap-edge models are oxygen-specific because their
exact learner configurations were tuned/selected for oxygen's own frozen
release (see `oxygen_classical_models_audit.md`) and must reproduce those
exactly, not merely something structurally similar.

Exact learner configurations (ported verbatim from the private
`oxygen_tabular_sprint4_expansion.py`/`oxygen_gap_edge_sprint4_expansion.py`):
Ridge(alpha=1.0), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31,
learning_rate=0.01, l2_regularization=0.001), ExtraTreesRegressor
(n_estimators=500, max_depth=None), RandomForestRegressor(n_estimators=400,
min_samples_leaf=3) -- every learner uses `random_state=RANDOM_SEED` (42) and
a `SimpleImputer(strategy="median")` pre-step (tree ensembles tolerate NaN
natively in scikit-learn's implementation but the private pipeline imputes
uniformly across all five learners for a consistent design matrix).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

from coastal_gap_reconstruction import baseline_imputation as bi
from coastal_gap_reconstruction import gaussian_process as gp_mod

from . import benchmark_contract as bc

__all__ = [
    "RANDOM_SEED", "LEARNER_NAMES", "build_learner",
    "run_model0_gap", "run_model0_evaluation",
    "fit_predict_tabular_gap", "run_tabular_loco_evaluation",
    "run_gp_gap_edge_gap", "run_gap_edge_loco_evaluation",
]

RANDOM_SEED = 42
LEARNER_NAMES = ["ridge", "elasticnet", "hgb_high_capacity", "extratrees_high_capacity", "randomforest"]


def build_learner(name: str) -> Pipeline:
    """Build one of `LEARNER_NAMES` as a `SimpleImputer(median) -> estimator` pipeline."""
    if name == "ridge":
        model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    elif name == "elasticnet":
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_SEED)
    elif name == "hgb_high_capacity":
        model = HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.01,
            l2_regularization=0.001, random_state=RANDOM_SEED,
        )
    elif name == "extratrees_high_capacity":
        model = ExtraTreesRegressor(n_estimators=500, max_depth=None, n_jobs=-1, random_state=RANDOM_SEED)
    elif name == "randomforest":
        model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown learner {name!r}; choose from {LEARNER_NAMES}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


# ── Model 0 baselines (reuse coastal_gap_reconstruction.baseline_imputation) ──

MODEL0_METHODS = {
    "climatology": bi.monthly_climatology,
    "persistence": bi.persistence_baseline,
    "linear_interp": bi.linear_interpolation_baseline,
}


def run_model0_gap(
    method: str, target_df: pd.DataFrame, start_date: pd.Timestamp, gap_length: int,
    target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> dict[pd.Timestamp, float]:
    if method not in MODEL0_METHODS:
        raise ValueError(f"Unknown Model 0 method {method!r}; choose from {list(MODEL0_METHODS)}")
    return MODEL0_METHODS[method](target_df, start_date, gap_length, target_col, eligible_col)


def run_model0_evaluation(
    method: str, candidates: pd.DataFrame, target_df: pd.DataFrame,
    target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> pd.DataFrame:
    """Run one Model 0 method over every row of `candidates`
    (gap_id/gap_length/start_date columns required). Returns one row per
    (gap_id, date) with `pred`/`true`."""
    rows = []
    for _, row in candidates.iterrows():
        gap_id = str(row["gap_id"])
        gap_length = int(row["gap_length"])
        start = pd.Timestamp(row["start_date"])
        preds = run_model0_gap(method, target_df, start, gap_length, target_col, eligible_col)
        for d, pred in preds.items():
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            rows.append({
                "gap_id": gap_id, "date": d, "method_id": method, "gap_length": gap_length,
                "pred": pred, "true": float(true_val) if true_val == true_val else np.nan,
            })
    return pd.DataFrame(rows)


# ── External-only tabular LOCO ────────────────────────────────────────────

def _build_train_pred(
    target_df: pd.DataFrame, features_df: pd.DataFrame, hidden_dates: pd.DatetimeIndex,
    feature_cols: list[str], target_col: str, eligible_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    eligible = target_df[eligible_col].fillna(False).astype(bool)
    has_target = target_df[target_col].notna()
    train_dates = target_df.index[eligible & has_target]
    train_dates = train_dates.difference(hidden_dates)
    train_dates = train_dates.intersection(features_df.index)

    pred_dates = hidden_dates.intersection(features_df.index)

    X_train = features_df.loc[train_dates, feature_cols].to_numpy(dtype=float)
    y_train = target_df.loc[train_dates, target_col].to_numpy(dtype=float)
    X_pred = features_df.loc[pred_dates, feature_cols].to_numpy(dtype=float)
    return X_train, y_train, X_pred, pred_dates


def fit_predict_tabular_gap(
    learner_name: str, target_df: pd.DataFrame, features_df: pd.DataFrame,
    start_date: pd.Timestamp, gap_length: int, feature_cols: list[str],
    target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> dict:
    """Fit one tabular learner on all other eligible days (LOCO: the gap's own
    hidden days are excluded from training by construction) and predict one
    gap's hidden dates on the raw mg/L scale. No log transform, no back-
    transform -- unlike the chlorophyll analogue of this function."""
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    result: dict = {"pred": {}, "n_train": 0, "warning": None}

    X_train, y_train, X_pred, pred_dates = _build_train_pred(
        target_df, features_df, hidden_dates, feature_cols, target_col, eligible_col
    )
    result["n_train"] = int(len(X_train))
    if len(X_train) < 10:
        result["warning"] = f"insufficient_training_rows ({len(X_train)})"
        return result
    if len(X_pred) == 0:
        result["warning"] = "no prediction dates available in feature table"
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = build_learner(learner_name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_pred)

    for d, p in zip(pred_dates, y_pred):
        result["pred"][d] = float(p)
    return result


def run_tabular_loco_evaluation(
    learner_name: str, candidates: pd.DataFrame, target_df: pd.DataFrame, features_df: pd.DataFrame,
    feature_cols: list[str], target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_rows, warn_rows = [], []
    for _, row in candidates.iterrows():
        gap_id = str(row["gap_id"])
        gap_length = int(row["gap_length"])
        start = pd.Timestamp(row["start_date"])
        result = fit_predict_tabular_gap(
            learner_name, target_df, features_df, start, gap_length, feature_cols, target_col, eligible_col,
        )
        if result["warning"]:
            warn_rows.append({"gap_id": gap_id, "learner_name": learner_name, "warning": result["warning"]})
            continue
        for d, pred in result["pred"].items():
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            pred_rows.append({
                "gap_id": gap_id, "date": d, "learner_name": learner_name, "gap_length": gap_length,
                "pred": pred, "true": float(true_val) if true_val == true_val else np.nan,
                "n_train": result["n_train"],
            })
    predictions_df = pd.DataFrame(pred_rows)
    warnings_df = pd.DataFrame(warn_rows) if warn_rows else pd.DataFrame(
        columns=["gap_id", "learner_name", "warning"]
    )
    return predictions_df, warnings_df


# ── Gap-edge / GP comparator ──────────────────────────────────────────────
#
# The released gap-edge program evaluated three structural variants
# (direct_hindcast, residual_interp_hindcast, pre_only_forecast_safe) x up to
# four learners (three tree ensembles + GP, residual_interp_hindcast only).
# This package ports and live-validates the one genuinely reusable,
# scientifically load-bearing result -- the GP (Matern, time-only) residual
# comparator, which is the single classical/engineered arm that reaches a
# statistical tie with interpolation (oxygen_classical_models_audit.md E.2)
# -- using the real, shared, already-tested
# `coastal_gap_reconstruction.gaussian_process` module, not a placeholder.
#
# The tree-ensemble gap-edge structural variants (direct_hindcast,
# pre_only_forecast_safe, and the tree-ensemble learners under
# residual_interp_hindcast) require the private project's own multi-gap
# pooled edge-feature training design
# (`oxygen_gap_edge_sprint4_expansion.py`'s `build_tier_c_feature_table`-
# style pooled fit, analogous to chlorophyll's `gap_edge_models.py`) to
# reproduce correctly -- porting a simplified/single-gap substitute would
# silently misrepresent what those released numbers actually measure. Given
# this phase's compute/effort boundary, these remain `frozen_result_only`
# (see `benchmark_contract.METHOD_STATUSES`) rather than being faked with a
# shortcut; the frozen `oxygen_benchmark_by_length.csv` rows are the
# authoritative source. Only the GP arm below is live-executable here.

GAP_EDGE_STRUCTURES = ["residual_interp_hindcast"]


def run_gp_gap_edge_gap(
    target_df: pd.DataFrame, start_date: pd.Timestamp, gap_length: int,
    target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> dict:
    """`residual_interp_hindcast` x GP (Matern, time-only): fit GP M1
    directly on the raw mg/L scale (oxygen's `benchmark_scoring_scale` is
    identity, so no transform is applied before calling the shared,
    target-agnostic GP wrapper)."""
    result = gp_mod.run_gp_on_gap(
        target_df, start_date, gap_length, value_col=target_col, eligible_col=eligible_col,
    )
    if result is None:
        return {"pred": {}, "pred_std": {}, "warning": "gp_fit_failed_or_insufficient_context"}
    return {
        "pred": dict(zip(result["dates"], result["pred"])),
        "pred_std": dict(zip(result["dates"], result["pred_std"])),
        "warning": None,
    }


def run_gap_edge_loco_evaluation(
    candidates: pd.DataFrame, target_df: pd.DataFrame,
    target_col: str = bc.TARGET_COLUMN, eligible_col: str = bc.ELIGIBLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GP (Matern, time-only) residual-over-interpolation gap-edge
    evaluation -- the one live-executable gap-edge/GP comparator in this
    package (see module docstring)."""
    pred_rows, warn_rows = [], []
    for _, row in candidates.iterrows():
        gap_id = str(row["gap_id"])
        gap_length = int(row["gap_length"])
        start = pd.Timestamp(row["start_date"])

        gp_result = run_gp_gap_edge_gap(target_df, start, gap_length, target_col, eligible_col)
        if gp_result["warning"]:
            warn_rows.append({"gap_id": gap_id, "learner_name": "gp_matern_time_only_exploratory",
                               "warning": gp_result["warning"]})
            continue
        for d, pred in gp_result["pred"].items():
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            pred_rows.append({
                "gap_id": gap_id, "date": d, "structure": "residual_interp_hindcast",
                "learner_name": "gp_matern_time_only_exploratory",
                "gap_length": gap_length, "pred": float(pred),
                "pred_std": float(gp_result["pred_std"][d]),
                "true": float(true_val) if true_val == true_val else np.nan,
            })

    predictions_df = pd.DataFrame(pred_rows)
    warnings_df = pd.DataFrame(warn_rows) if warn_rows else pd.DataFrame(
        columns=["gap_id", "learner_name", "warning"]
    )
    return predictions_df, warnings_df
