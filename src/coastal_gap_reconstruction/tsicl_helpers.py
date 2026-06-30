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

import numpy as np

DEFAULT_QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


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
    return np.ascontiguousarray(covariate_array[None, :, :], dtype=np.float32)


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
