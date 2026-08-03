"""External-only tabular reconstruction models for chlorophyll -- **Protocol A**
("plain external-only") in the private project's two-protocol split.

**Two distinct external-tabular protocols exist in the private project and are
kept apart on purpose (see `docs/methodology/model_families.md` "Two
external-tabular protocols" section):**

- **Protocol A (this module, "plain external-only")**: `ARM4_COLUMNS` only (46
  numeric predictors after dropping the categorical `sst_primary_source`), no
  gap-position information, plain LOCO with no dependency-window exclusion
  (unnecessary here since no target-history feature is used -- see
  `forbidden_target_history_columns`). Public method IDs
  `external_only_extratrees`/`external_only_hgb`. Ported from the private
  project's curated external-spatial model lineage. **This protocol is not the
  source of the frozen `ext_tabular_extratrees`/`ext_tabular_hgb` rows in
  `results_public/chlorophyll/chlorophyll_matched_support_method_metrics.csv`**
  -- it is a separate, plain-external diagnostic comparison, evaluated here
  for its own sake, not to reproduce a released number.
- **Protocol B ("matched-reference")**: `ARM4_COLUMNS` plus 5 structural
  meta-features describing gap length and within-gap position, strict
  dependency-window LOCO. Public method IDs `ext_tabular_extratrees`/
  `ext_tabular_hgb` (the frozen IDs -- kept because this protocol is what
  actually produced those released numbers; see
  `gap_edge_models.run_reference_arm_loco_evaluation`, which implements it).

Only the fitting/scoring mechanics survive from the private files here -- the
private files' reporting, figure generation, caching, and CLI wrapping
(~2000 combined lines) are dropped, not ported.

`HistGradientBoostingRegressor` is retained in both protocols as a diagnostic
comparator to `ExtraTreesRegressor`, never presented as the canonical arm on
its own.

Every model here is fit per-gap under leave-one-gap-out (LOCO): the training
set excludes the gap's own hidden days by construction (the target column is
already masked upstream in the released daily-target table) and never reads
any other gap's hidden days either, since training uses only rows where the
target is observed. No dependency-window exclusion is needed for Protocol A
specifically because it uses no target-history feature (arm4 is
target-history-free by design -- see `forbidden_target_history_columns`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from . import _config

__all__ = [
    "ARM4_COLUMNS",
    "RANDOM_SEED",
    "load_arm4_numeric_columns",
    "forbidden_target_history_columns",
    "build_extratrees",
    "build_hgb_diagnostic",
    "fit_predict_gap",
    "run_loco_evaluation",
]

RANDOM_SEED = _config.RANDOM_SEED

# arm4 / "minimal_plus_wind_relaxation": the released canonical external-only
# feature set (47 columns as published; `sst_primary_source` is categorical
# and dropped at fit time -- see `load_arm4_numeric_columns`). Exact source:
# private `curated_external_model_feature_sets_proposed.csv`, row
# `minimal_plus_wind_relaxation`.
ARM4_COLUMNS: list[str] = [
    "month", "doy_sin", "doy_cos",
    "chl_cons_log10", "chl_cons_log10_lag1", "chl_cons_log10_lag3",
    "chl_cons_log10_lag7", "chl_cons_log10_roll3", "chl_cons_log10_roll7",
    "chl_is_gapfree",
    "mur_sst_available", "sst_primary_degC", "sst_primary_source",
    "sst_primary_degC_lag1", "sst_primary_degC_lag3", "sst_primary_degC_lag7",
    "sst_primary_degC_roll3", "sst_primary_degC_roll7",
    "mur_sst_nearest_degC", "mur_sst_anom_monthly_degC",
    "chl_valid_frac_3x3", "chl_valid_frac_5x5", "chl_anom_log10_monthly",
    "plv_solar_roll3d_wm2", "plv_solar_roll7d_wm2", "plv_solar_roll14d_wm2",
    "plv_solar_roll21d_wm2",
    "plv_humid_roll7d_pct", "plv_humid_roll14d_pct",
    "plv_pressure_roll3d_hPa", "plv_pressure_roll7d_hPa", "plv_pressure_roll14d_hPa",
    "plv_upwelling_cumul3d_ms_d", "plv_upwelling_cumul7d_ms_d",
    "plv_upwelling_cumul14d_ms_d", "plv_upwelling_cumul21d_ms_d",
    "cmems_upwelling_cumul3d_ms_d", "cmems_upwelling_cumul7d_ms_d",
    "cmems_upwelling_cumul14d_ms_d", "cmems_upwelling_cumul21d_ms_d",
    "mur_front_persist_7d", "mur_front_persist_14d",
    "chl_patch_persist_7d", "chl_patch_persist_14d",
    "plv_relaxation_index_14p3r", "cmems_relaxation_index_14p3r",
    "plv_dir_persist_7d",
]

# Columns that would leak in-situ target history if ever added to arm4 (they
# are legitimate features for gap_edge_models.py's hindcast arm, which is
# explicitly a different, retrospective-only model) -- forbidden here.
FORBIDDEN_TARGET_HISTORY_COLUMNS: frozenset[str] = frozenset([
    "chl_mean", "chl_mean_lag1", "chl_mean_lag3", "chl_mean_lag7",
    "pre_last_chl", "post_first_chl", "linear_interp_chl",
])


def load_arm4_numeric_columns(features_df: pd.DataFrame) -> list[str]:
    """`ARM4_COLUMNS` filtered to columns present and numeric in `features_df`.

    Reproduces the released procedure exactly: `sst_primary_source` is a
    string column and is silently dropped here (not an error), matching the
    private project's feature-arm loader's non-numeric exclusion.
    """
    numeric: list[str] = []
    for col in ARM4_COLUMNS:
        if col not in features_df.columns:
            continue
        try:
            pd.to_numeric(features_df[col])
            numeric.append(col)
        except (ValueError, TypeError):
            continue
    return numeric


def forbidden_target_history_columns(columns: list[str]) -> list[str]:
    """Return any of `columns` that are forbidden target-history predictors.

    Callers (and tests) should assert this returns an empty list before
    fitting -- arm4 must remain external-only.
    """
    return [c for c in columns if c in FORBIDDEN_TARGET_HISTORY_COLUMNS]


def build_extratrees(n_estimators: int = 500, random_state: int = RANDOM_SEED) -> ExtraTreesRegressor:
    """The canonical external-tabular learner."""
    return ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)


def build_hgb_diagnostic(random_state: int = RANDOM_SEED) -> HistGradientBoostingRegressor:
    """Diagnostic comparator only -- never the canonical external-tabular result."""
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=0.0, max_iter=1000, early_stopping=True,
        validation_fraction=0.10, n_iter_no_change=20, random_state=random_state,
    )


def _build_train_test(
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    hidden_dates: pd.DatetimeIndex,
    feature_cols: list[str],
    target_col: str,
    eligible_col: str,
    log10_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    eligible = target_df[eligible_col].fillna(False).astype(bool)
    has_target = target_df[target_col].notna() & (target_df[target_col] > log10_floor)
    train_dates = target_df.index[eligible & has_target]
    train_dates = train_dates.difference(hidden_dates)
    train_dates = train_dates.intersection(features_df.index)

    pred_dates = hidden_dates.intersection(features_df.index)

    X_train = features_df.loc[train_dates, feature_cols].to_numpy(dtype=float)
    y_train = np.log10(target_df.loc[train_dates, target_col].to_numpy(dtype=float))
    X_pred = features_df.loc[pred_dates, feature_cols].to_numpy(dtype=float)
    return X_train, y_train, X_pred, pred_dates


def fit_predict_gap(
    model_name: str,
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    feature_cols: list[str],
    target_col: str = _config.TARGET_COL,
    eligible_col: str = _config.ELIGIBLE_COL,
    log10_floor: float = 1e-4,
) -> dict:
    """Fit one tabular model on all other eligible days and predict one gap's
    hidden dates.

    Returns a dict with `pred_log10: {date: float}`, `pred: {date: float}`
    (back-transformed to the physical unit), `n_train`, and `warning` (`None`
    on success).
    """
    if forbidden_target_history_columns(feature_cols):
        raise ValueError(
            f"feature_cols contains forbidden target-history columns: "
            f"{forbidden_target_history_columns(feature_cols)}"
        )

    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    result: dict = {"pred_log10": {}, "pred": {}, "n_train": 0, "warning": None}

    X_train, y_train, X_pred, pred_dates = _build_train_test(
        target_df, features_df, hidden_dates, feature_cols, target_col, eligible_col, log10_floor
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
        if model_name == "external_only_extratrees":
            imp = SimpleImputer(strategy="mean")
            X_train_i = imp.fit_transform(X_train)
            X_pred_i = imp.transform(X_pred)
            model = build_extratrees()
            model.fit(X_train_i, y_train)
            y_pred_log = model.predict(X_pred_i)
        elif model_name == "external_only_hgb":
            model = build_hgb_diagnostic()
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_pred)
        else:
            result["warning"] = f"unknown model_name {model_name!r}"
            return result

    y_pred = 10.0**y_pred_log
    for d, lp, p in zip(pred_dates, y_pred_log, y_pred):
        result["pred_log10"][d] = float(lp)
        result["pred"][d] = float(p)
    return result


def run_loco_evaluation(
    model_name: str,
    candidates: pd.DataFrame,
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = _config.TARGET_COL,
    eligible_col: str = _config.ELIGIBLE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `fit_predict_gap` over every row of `candidates`
    (gap_id/gap_length/start_date columns required).

    Returns `(predictions_df, warnings_df)`. `predictions_df` has one row per
    (gap_id, date) with `pred_log10`/`pred`/`true` columns.
    """
    pred_rows: list[dict] = []
    warn_rows: list[dict] = []

    for _, row in candidates.iterrows():
        gap_id = str(row["gap_id"])
        gap_length = int(row["gap_length"])
        start = pd.Timestamp(row["start_date"])

        result = fit_predict_gap(
            model_name, target_df, features_df, start, gap_length, feature_cols,
            target_col=target_col, eligible_col=eligible_col,
        )
        if result["warning"]:
            warn_rows.append({"gap_id": gap_id, "model_name": model_name, "warning": result["warning"]})
            continue

        for d, pred_log in result["pred_log10"].items():
            pred = result["pred"][d]
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            pred_rows.append({
                "gap_id": gap_id, "date": d, "model_name": model_name,
                "gap_length": gap_length, "pred_log10": pred_log, "pred": pred,
                "true": float(true_val) if true_val == true_val else np.nan,
                "n_train": result["n_train"],
            })

    predictions_df = pd.DataFrame(pred_rows)
    warnings_df = pd.DataFrame(warn_rows) if warn_rows else pd.DataFrame(
        columns=["gap_id", "model_name", "warning"]
    )
    return predictions_df, warnings_df
