"""Thin convenience wrappers for invoking TS-ICL on a gappy daily series.

TS-ICL is a zero-shot, in-context-learning time-series foundation model. It
is used here as a candidate reconstruction method: the target series (with
the gap masked to NaN) is passed in directly, optionally alongside one or
more covariate channels (for example a satellite chlorophyll proxy, or a
wind/upwelling forcing variable), and the model produces a point estimate
plus quantile predictions for the missing region without any task-specific
fine-tuning.

This module does not vendor TS-ICL itself -- install it separately (see
notebooks/06_tsicl_zero_shot_imputation.ipynb) and check the authors'
license/terms before use. These helpers only show the calling convention
and a couple of non-obvious shape requirements that are easy to get wrong.

License note: TS-ICL code and pretrained weights are distributed under the
original authors' license, which is separate from this repository's
license. Review it before using TS-ICL in your own work.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

DEFAULT_QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


@dataclass
class TSICLStatus:
    """Provenance for one `load_tsicl()` attempt.

    `live=False` means the caller should fall back to cached/pre-saved
    predictions (if any) rather than crash -- see `load_tsicl`.
    """

    live: bool
    device: str
    load_time_s: float
    error: str | None = None


def load_tsicl() -> tuple[object | None, TSICLStatus]:
    """Attempt to load TS-ICL live. Returns (model_or_None, status).

    Never raises: if TS-ICL cannot be loaded (package missing, checkpoint
    unavailable, incompatible environment), returns `(None, status)` with
    `status.live = False` and `status.error` set, so the caller can fall back
    to cached predictions explicitly rather than crashing. This keeps the
    optional `tsicl`/`torch` dependency lazy -- importing this module never
    requires them, only calling `load_tsicl()` does.
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
        return None, TSICLStatus(
            live=False, device="unknown", load_time_s=time.time() - t0, error=f"{type(e).__name__}: {e}"
        )


def impute_masked_series(
    model,
    target_log10_masked: np.ndarray,
    covariate_array: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one TS-ICL `impute()` call on an already-masked, already-log10 series.

    This is the shared low-level call used by both the demo notebook and
    `notebooks/06_tsicl_zero_shot_imputation.ipynb`: it does not know about
    gap objects, dates, or units beyond "caller has already put this series
    in log10 space and masked the region to reconstruct with NaN".

    Parameters
    ----------
    model:
        A loaded TS-ICL model instance (see `load_tsicl`).
    target_log10_masked:
        1D array of length T, log10-scale, NaN at positions to reconstruct.
    covariate_array:
        Optional (T, C) covariate array aligned to target_log10_masked.
    quantile_levels:
        Quantile levels to request. Defaults to `DEFAULT_QUANTILE_LEVELS`.

    Returns
    -------
    (mean, quantiles) as numpy arrays: mean has shape (T,), quantiles has
    shape (T, len(quantile_levels)), both still in log10 space (unconverted).
    """
    import torch  # local import: torch is only needed if you actually run TS-ICL

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs_t = torch.from_numpy(np.asarray(target_log10_masked, dtype=np.float32))
        covars_t = None
        if covariate_array is not None:
            covars_t = torch.from_numpy(build_covariate_block(np.asarray(covariate_array)))
        mean, quantiles = model.impute(
            inputs=inputs_t,
            covars=covars_t,
            quantile_levels=quantile_levels,
            denormalize=True,
            replace_by_gt=False,
        )
    return mean.numpy(), quantiles.numpy()


def log10_to_physical(values_log10: np.ndarray) -> np.ndarray:
    """Convert a log10-space array (mean, quantiles, or any TS-ICL output) back
    to physical units. Kept as a named function, not an inline `10 **`, so
    call sites are explicit about which scale they are working in."""
    return 10**np.asarray(values_log10)


def build_covariate_block(covariate_array: np.ndarray) -> np.ndarray:
    """Reshape a (T, C) covariate array into the (1, T, C) batch shape TS-ICL expects.

    This is the single most common integration mistake: passing a bare 2D
    array of shape (T, C) is silently reinterpreted by TS-ICL's input
    validation as (batch=T, channel=1, time=C), which is not what you want.
    Always pass an explicit batch dimension of size 1 for a single series.

    Parameters
    ----------
    covariate_array:
        Array of shape (T, C) -- T timesteps, C covariate channels. May
        contain NaNs if a covariate is only sparsely available (see
        `allow_auto_complete` below).
    """
    if covariate_array.ndim != 2:
        raise ValueError("covariate_array must be 2D: (T, C)")
    # .copy() (not just np.ascontiguousarray, which is a no-op and can return a
    # read-only view if the input is already contiguous float32) guarantees a
    # writable array, which avoids a PyTorch UserWarning -- that warning's
    # default text embeds this file's local absolute filesystem path, which we
    # do not want leaking into any executed-notebook output.
    return np.ascontiguousarray(covariate_array[None, :, :], dtype=np.float32).copy()


def run_tsicl_imputation(
    model,
    target_series: np.ndarray,
    covariate_array: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    allow_auto_complete: bool = True,
):
    """Run a single TS-ICL imputation call on one target series.

    Parameters
    ----------
    model:
        A loaded TS-ICL model instance (see the authors' repository for
        checkpoint loading; not vendored here).
    target_series:
        1D array of length T, with NaN at the positions to be reconstructed
        (and anywhere else genuinely missing). TS-ICL treats NaN as the gap
        indicator.
    covariate_array:
        Optional (T, C) covariate array aligned to target_series. May be
        sparse (contain NaNs) -- if so, set allow_auto_complete=True so the
        model fills small covariate gaps internally rather than failing.
    quantile_levels:
        Quantile levels to request in addition to the point estimate.
        Defaults to a standard 7-quantile set spanning the 5th-95th
        percentile, useful for uncertainty bands around the reconstruction.
    allow_auto_complete:
        Passed through to the underlying model call; allows TS-ICL to
        auto-complete small gaps in the covariate channels themselves
        (distinct from gaps in the target series).

    Returns
    -------
    (mean_prediction, quantile_predictions) as returned by the underlying
    model.impute(...) call, after denormalization.

    Notes
    -----
    Evaluate TS-ICL outputs only at the artificially hidden positions used
    in validation (see notebooks/02_artificial_gap_validation.ipynb) --
    never compare against positions that were visible to the model, and
    never use predictions on real (non-validation) gaps as if they were
    validation evidence (see docs/evidence_hierarchy.md).
    """
    import torch  # local import: torch is only needed if you actually run TS-ICL

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    inputs_t = torch.from_numpy(np.asarray(target_series, dtype=np.float32))

    covars_t = None
    if covariate_array is not None:
        covars_block = build_covariate_block(np.asarray(covariate_array))
        covars_t = torch.from_numpy(covars_block)

    mean, quantiles = model.impute(
        inputs=inputs_t,
        covars=covars_t,
        quantile_levels=quantile_levels,
        denormalize=True,
        replace_by_gt=False,
        allow_auto_complete=allow_auto_complete,
    )
    return mean.numpy(), quantiles.numpy()
