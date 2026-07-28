"""Reconstruction methods used by the gap-reconstruction walkthrough notebook.

Every `run_*` function takes an `ArtificialGap` (or the real-gap equivalent) and
returns a `MethodResult`: a prediction table plus provenance (runtime, covariates
used, method name) and, where relevant, extra arrays needed for a decomposition
plot. No function here plots anything or writes to disk.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .demo_helpers import ArtificialGap, QUANTILE_LEVELS, SATELLITE_PROXY_COLUMN, SST_COLUMN, WIND_COLUMN


@dataclass
class MethodResult:
    """A single method's reconstruction output.

    Attributes:
        method: short machine-readable method name (e.g. "gaussian_process").
        prediction: DataFrame with columns date, value, and optionally q05, q95.
        runtime_s: wall-clock seconds for fit + predict.
        covariates_used: list of covariate column names used (empty if none).
        extra: method-specific auxiliary data (e.g. gap-edge decomposition
            components), never required by generic code, only by that method's
            own plotting function.
    """

    method: str
    prediction: pd.DataFrame
    runtime_s: float
    covariates_used: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def run_persistence(gap: ArtificialGap) -> MethodResult:
    """Repeat the last observed value before the gap."""
    t0 = time.time()
    context = gap.context
    last_value = context.loc[context["date"] < gap.gap_start, gap.target_column].iloc[-1]
    prediction = gap.truth[["date"]].copy()
    prediction["value"] = last_value
    return MethodResult("persistence", prediction, time.time() - t0)


def run_climatology(gap: ArtificialGap, full_record: pd.DataFrame) -> MethodResult:
    """Day-of-year mean over the full multi-year record, excluding the gap itself.

    Needs multi-year history to be meaningful, so it is computed from
    `full_record`, not from `gap.context` alone (a single-season local window
    has no other observations at the same calendar month).

    Uses a **monthly** climatology (mean over all years for the same calendar
    month, 12 bins), not a day-of-year climatology: day-of-year bins are too
    thin given ~11 years of record (each day-of-year has only a handful of
    observations), which makes a day-of-year climatology noisier than the
    smooth, low-variance reference a climatology baseline is supposed to be.
    """
    t0 = time.time()
    record = full_record.copy()
    record["log10_target"] = np.log10(record[gap.target_column])
    is_gap_in_record = (record["date"] >= gap.gap_start) & (record["date"] <= gap.gap_end)
    record.loc[is_gap_in_record, "log10_target"] = np.nan

    month = record["date"].dt.month
    climatology_by_month = record.groupby(month)["log10_target"].mean()

    prediction = gap.truth[["date"]].copy()
    prediction["value"] = 10 ** prediction["date"].dt.month.map(climatology_by_month)
    return MethodResult("climatology", prediction, time.time() - t0)


def run_linear_interpolation(gap: ArtificialGap) -> MethodResult:
    """Straight-line interpolation between the two gap edges."""
    t0 = time.time()
    interp = (
        gap.masked_series.set_index("date")[gap.target_column]
        .interpolate(method="linear", limit_area="inside")
    )
    prediction = interp.loc[gap.gap_start : gap.gap_end].reset_index()
    prediction.columns = ["date", "value"]
    return MethodResult("linear_interpolation", prediction, time.time() - t0)


def run_baselines(gap: ArtificialGap, full_record: pd.DataFrame) -> dict[str, MethodResult]:
    """Run all three simple baselines and return them keyed by method name."""
    return {
        "persistence": run_persistence(gap),
        "climatology": run_climatology(gap, full_record),
        "linear_interpolation": run_linear_interpolation(gap),
    }


def run_gaussian_process(gap: ArtificialGap) -> MethodResult:
    """Target-only Gaussian process (Matern 3/2 + white noise), fit on the local
    observed context, predicting a mean and a q05-q95 predictive interval over
    the gap.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    t0 = time.time()
    context = gap.context.copy()
    t0_date = gap.full_series["date"].min()
    context["t"] = (context["date"] - t0_date).dt.days

    X = context[["t"]].to_numpy()
    y = np.log10(context[gap.target_column].to_numpy())
    kernel = Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.05)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, normalize_y=True)
    gp.fit(X, y)

    X_gap = ((gap.truth["date"] - t0_date).dt.days).to_numpy().reshape(-1, 1)
    mean_log10, std_log10 = gp.predict(X_gap, return_std=True)

    prediction = gap.truth[["date"]].copy()
    prediction["value"] = 10 ** mean_log10
    prediction["q05"] = 10 ** (mean_log10 - 1.645 * std_log10)
    prediction["q95"] = 10 ** (mean_log10 + 1.645 * std_log10)
    return MethodResult("gaussian_process", prediction, time.time() - t0)


TABULAR_FEATURE_COLUMNS = [SATELLITE_PROXY_COLUMN, WIND_COLUMN, SST_COLUMN, "doy_sin", "doy_cos"]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    return df


def run_external_tabular(gap: ArtificialGap, full_record: pd.DataFrame) -> MethodResult:
    """Fit a HistGradientBoostingRegressor on covariates and calendar features
    only -- no target history is given to this model. Fit on the full record
    excluding the gap; HistGradientBoosting is used (not ExtraTrees) because it
    tolerates missing covariate days natively (the satellite proxy has real
    cloud-cover gaps).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    t0 = time.time()
    record = _add_calendar_features(full_record.copy())
    record["log10_target"] = np.log10(record[gap.target_column])
    train_mask = record["log10_target"].notna() & ~(
        (record["date"] >= gap.gap_start) & (record["date"] <= gap.gap_end)
    )

    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(record.loc[train_mask, TABULAR_FEATURE_COLUMNS], record.loc[train_mask, "log10_target"])

    gap_rows = record[(record["date"] >= gap.gap_start) & (record["date"] <= gap.gap_end)]
    pred_log10 = model.predict(gap_rows[TABULAR_FEATURE_COLUMNS])

    prediction = pd.DataFrame({"date": gap_rows["date"].to_numpy(), "value": 10 ** pred_log10})
    feature_table = gap_rows[["date"] + TABULAR_FEATURE_COLUMNS].reset_index(drop=True)
    return MethodResult(
        "external_tabular",
        prediction,
        time.time() - t0,
        covariates_used=TABULAR_FEATURE_COLUMNS,
        extra={"feature_table": feature_table, "n_training_rows": int(train_mask.sum())},
    )


def _interpolate_with_edge_features(record: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Given `record[target_col]` (possibly containing NaN), add columns
    `interp_log10`, `days_from_prev_obs`, `days_to_next_obs` computed from the
    NaN pattern actually present in `target_col`."""
    out = record.copy()
    out["interp_log10"] = out[target_col].interpolate(method="linear", limit_area="inside")
    observed_dates = out["date"].where(out[target_col].notna())
    out["days_from_prev_obs"] = (out["date"] - observed_dates.ffill()).dt.days
    out["days_to_next_obs"] = (observed_dates.bfill() - out["date"]).dt.days
    return out


def _build_gap_edge_training_rows(
    record: pd.DataFrame,
    gap: ArtificialGap,
    edge_feature_columns: list[str],
    rng_seed: int = 0,
    n_synthetic_gaps: int = 80,
    lengths: tuple[int, ...] = (1, 3, 7, 14),
    edge_buffer_days: int = 10,
) -> pd.DataFrame:
    """Build a residual-correction training set from synthetic held-out
    sub-gaps: repeatedly hide a short, fully-observed block elsewhere in the
    record, interpolate across it, and record the true residual at the hidden
    days. This mirrors artificial-gap validation, applied here to generate a
    training signal instead of a benchmark score -- it deliberately excludes
    the trivial "residual = 0 at every already-observed day" rows, which would
    otherwise dominate the training set and make the model collapse to
    predicting no correction at all.
    """
    rng = np.random.default_rng(rng_seed)
    true_values = record["log10_target_true"].to_numpy()
    n = len(record)
    training_frames = []
    attempts = 0
    found = 0
    while found < n_synthetic_gaps and attempts < n_synthetic_gaps * 20:
        attempts += 1
        length = int(rng.choice(lengths))
        start = int(rng.integers(edge_buffer_days, n - length - edge_buffer_days))
        block = slice(start, start + length)
        if np.isnan(true_values[block]).any():
            continue  # only mask blocks that are fully observed to begin with

        masked = record["log10_target_true"].copy()
        masked.iloc[block] = np.nan
        synthetic = _interpolate_with_edge_features(record.assign(_masked=masked), "_masked")
        rows = synthetic.iloc[block].copy()
        rows["residual"] = true_values[block] - rows["interp_log10"].to_numpy()
        if rows["residual"].isna().any() or rows[edge_feature_columns].isna().any().any():
            continue  # too close to the record edge to interpolate on both sides
        training_frames.append(rows[edge_feature_columns + ["residual"]])
        found += 1

    if not training_frames:
        raise RuntimeError("Could not build any synthetic training gaps for the gap-edge residual model.")
    return pd.concat(training_frames, ignore_index=True)


def run_gap_edge_residual(gap: ArtificialGap, full_record: pd.DataFrame) -> MethodResult:
    """Linear interpolation plus a learned residual correction, using distance
    to the nearest pre/post observation and covariates as reconstruction
    context. Trained on residuals at every other gap in the masked full record.

    Uses observations after the gap directly as reconstruction context for this
    specific gap (via `days_to_next_obs` and the post-edge interpolation
    anchor) -- this makes it retrospective, not applicable to an open-ended gap.

    Training data: residual = true - interpolated computed only at synthetic
    held-out sub-gaps carved elsewhere in the record (not at every already-
    observed day, where the residual is trivially zero by construction --
    training on those as well would swamp the genuinely informative examples
    and collapse the model to predicting ~0 everywhere).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    t0 = time.time()
    record = _add_calendar_features(full_record.copy()).sort_values("date").reset_index(drop=True)
    record["log10_target_true"] = np.log10(record[gap.target_column])
    record.loc[
        (record["date"] >= gap.gap_start) & (record["date"] <= gap.gap_end), "log10_target_true"
    ] = np.nan

    edge_feature_columns = TABULAR_FEATURE_COLUMNS + ["days_from_prev_obs", "days_to_next_obs"]
    training_rows = _build_gap_edge_training_rows(record, gap, edge_feature_columns, rng_seed=0)

    model = HistGradientBoostingRegressor(random_state=0, max_iter=150, min_samples_leaf=5)
    model.fit(training_rows[edge_feature_columns], training_rows["residual"])

    gap_rows = _interpolate_with_edge_features(record, "log10_target_true")
    gap_rows = gap_rows[(gap_rows["date"] >= gap.gap_start) & (gap_rows["date"] <= gap.gap_end)].copy()
    predicted_residual = model.predict(gap_rows[edge_feature_columns])
    corrected_log10 = gap_rows["interp_log10"].to_numpy() + predicted_residual

    prediction = pd.DataFrame({"date": gap_rows["date"].to_numpy(), "value": 10 ** corrected_log10})
    decomposition = pd.DataFrame(
        {
            "date": gap_rows["date"].to_numpy(),
            "interpolation": 10 ** gap_rows["interp_log10"].to_numpy(),
            "predicted_correction_log10": predicted_residual,
            "corrected": 10 ** corrected_log10,
        }
    )
    return MethodResult(
        "gap_edge_residual",
        prediction,
        time.time() - t0,
        covariates_used=TABULAR_FEATURE_COLUMNS,
        extra={"decomposition": decomposition, "n_training_rows": len(training_rows)},
    )


# ---------------------------------------------------------------------------
# TS-ICL
# ---------------------------------------------------------------------------


@dataclass
class TSICLStatus:
    live: bool
    device: str
    load_time_s: float
    error: str | None = None


def load_tsicl() -> tuple[object | None, TSICLStatus]:
    """Attempt to load TS-ICL live. Returns (model_or_None, status).

    Never raises: if TS-ICL cannot be loaded (package missing, checkpoint
    unavailable, incompatible environment), returns `(None, status)` with
    `status.live = False` and `status.error` set, so the caller can fall back
    to cached predictions explicitly rather than crashing.
    """
    t0 = time.time()
    try:
        import torch
        from tsicl import TSICL

        model = TSICL()
        load_time = time.time() - t0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return model, TSICLStatus(live=True, device=device, load_time_s=load_time)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure means "fall back"
        return None, TSICLStatus(live=False, device="unknown", load_time_s=time.time() - t0, error=f"{type(e).__name__}: {e}")


def _tsicl_impute(model, target_log10_masked: np.ndarray, covar_array: np.ndarray | None):
    import torch

    inputs_t = torch.from_numpy(target_log10_masked.astype(np.float32))
    covars_t = None
    if covar_array is not None:
        covars_t = torch.from_numpy(np.ascontiguousarray(covar_array[None, :, :], dtype=np.float32).copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, quantiles = model.impute(
            inputs=inputs_t,
            covars=covars_t,
            quantile_levels=QUANTILE_LEVELS,
            denormalize=True,
            replace_by_gt=False,
        )
    return mean.numpy(), quantiles.numpy()


def run_tsicl(
    model,
    gap: ArtificialGap,
    covariates: np.ndarray | None,
    covariates_used: list[str],
    method_name: str,
) -> MethodResult:
    """Run one TS-ICL configuration (target-only if `covariates` is None,
    covariate-conditioned otherwise) on the given artificial gap. `model` must
    already be loaded (see `load_tsicl`)."""
    t0 = time.time()
    target_log10 = np.log10(gap.full_series[gap.target_column].to_numpy())
    target_log10_masked = target_log10.copy()
    gap_mask = gap.is_gap.to_numpy()
    target_log10_masked[gap_mask] = np.nan

    mean, quantiles = _tsicl_impute(model, target_log10_masked, covariates)

    gap_lo = int(np.argmax(gap_mask))
    gap_hi = gap_lo + int(gap_mask.sum())
    prediction = pd.DataFrame(
        {
            "date": gap.full_series["date"].iloc[gap_lo:gap_hi].to_numpy(),
            "value": 10 ** mean[gap_lo:gap_hi],
            "q05": 10 ** quantiles[gap_lo:gap_hi, 0],
            "q95": 10 ** quantiles[gap_lo:gap_hi, 6],
        }
    )
    return MethodResult(method_name, prediction, time.time() - t0, covariates_used=covariates_used)


def run_tsicl_real_gap(model, real_gap: pd.DataFrame, target_column: str = "chl_mean") -> MethodResult:
    """Apply TS-ICL (target + satellite-proxy covariate) to a real, genuinely
    missing interval (`real_gap["in_real_gap"]` marks the missing rows). There
    is no truth to score this against."""
    t0 = time.time()
    is_gap = real_gap["in_real_gap"].to_numpy()
    target_log10 = np.log10(real_gap[target_column].to_numpy())
    target_log10_masked = target_log10.copy()
    target_log10_masked[is_gap] = np.nan

    covariates = real_gap[[SATELLITE_PROXY_COLUMN]].to_numpy(dtype=np.float32)
    mean, quantiles = _tsicl_impute(model, target_log10_masked, covariates)

    gap_lo = int(np.argmax(is_gap))
    gap_hi = gap_lo + int(is_gap.sum())
    prediction = pd.DataFrame(
        {
            "date": real_gap["date"].iloc[gap_lo:gap_hi].to_numpy(),
            "value": 10 ** mean[gap_lo:gap_hi],
            "q05": 10 ** quantiles[gap_lo:gap_hi, 0],
            "q95": 10 ** quantiles[gap_lo:gap_hi, 6],
        }
    )
    return MethodResult(
        "tsicl_satellite_proxy", prediction, time.time() - t0, covariates_used=[SATELLITE_PROXY_COLUMN]
    )


def load_cached_tsicl_predictions(data_dir, arm: str) -> pd.DataFrame:
    """Load pre-saved TS-ICL predictions for the artificial demo gap (emergency
    fallback only -- see `run_tsicl` / the notebook's live-vs-cached branch).
    `arm` is one of "target_only", "satellite_proxy", "physical_bundle"."""
    cached = pd.read_csv(data_dir / "cached_tsicl_predictions.csv", parse_dates=["date"])
    subset = cached[cached["tsicl_arm"] == arm][["date", "pred_chl", "q05", "q95"]].reset_index(drop=True)
    return subset.rename(columns={"pred_chl": "value"})


def load_cached_tsicl_real_gap_predictions(data_dir) -> pd.DataFrame:
    """Cached fallback for the real-gap TS-ICL application."""
    cached = pd.read_csv(data_dir / "cached_tsicl_predictions_real_gap.csv", parse_dates=["date"])
    return cached[["date", "pred_chl", "q05", "q95"]].rename(columns={"pred_chl": "value"})
