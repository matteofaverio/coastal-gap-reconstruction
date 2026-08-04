"""The authoritative reusable calling layer for invoking TS-ICL on a gappy
daily series.

TS-ICL is a zero-shot, in-context-learning time-series foundation model
("TS-ICL: A Flexible Time-Index Foundation Model for Time Series via
In-Context Learning", Le Naour, Nabil, Petralia; package `tsicl` on PyPI,
model weights on Hugging Face). It is used here as a candidate
reconstruction method: the target series (with the gap masked to NaN) is
passed in directly, optionally alongside one or more covariate channels (for
example a satellite chlorophyll proxy, or a wind/upwelling forcing
variable), and the model produces a point estimate plus quantile predictions
for the missing region without any task-specific fine-tuning.

This module does not vendor TS-ICL itself -- it is an optional dependency
(the `tsicl` extra; see `docs/methodology/tsicl_usage.md` for the exact
installation commands and environment notes) governed by its own separate
license, distinct from this repository's license: **TS-ICL Non-Commercial
License v1.0, (c) EDF SA 2026** -- review it before using TS-ICL in your own
work, in particular its restrictions on commercial/production use and
hosted-service distribution.

Every function here is the single, shared implementation used by the demo
(`demo/src/methods.py`), the notebooks, and the benchmark drivers in
`experiments/chlorophyll/` -- none of those duplicate this module's model-
calling logic; they only call it.

Two loading paths, deliberately different failure behavior:

- `load_tsicl()`: never raises. Returns `(None, status)` with
  `status.live=False` on any failure (package missing, checkpoint
  unreachable, incompatible environment) -- for interactive/demo contexts
  that should fall back to a cached result rather than crash.
- `load_tsicl_strict()`: raises `TSICLProvenanceError`/`TSICLDependencyError`
  on any failure, and additionally verifies the loaded checkpoint's revision
  and file hash against the pinned provenance below -- for reproducibility-
  focused benchmark runs, which must fail loudly rather than silently
  produce results from an unverified or different checkpoint.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# ── Pinned checkpoint/package provenance ────────────────────────────────
# Recovered by direct inspection of the private project's installed TS-ICL
# environment and its Hugging Face cache (not carried forward from an
# undocumented summary) -- see docs/methodology/tsicl_usage.md for the full
# provenance report.
CHECKPOINT_REPO_ID = "taharnbl/TS-ICL"
CHECKPOINT_REVISION = "f01f3869a735694691401cd67a5e19c17e94e220"
CHECKPOINT_FILENAME = "tsicl-v1.ckpt"
CHECKPOINT_SHA256 = "a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431"
CHECKPOINT_SIZE_BYTES = 219_150_987
EXPECTED_TSICL_PACKAGE_VERSION = "0.2.1"
LICENSE_NOTICE = "TS-ICL Non-Commercial License v1.0, (c) EDF SA 2026 -- see the authors' repository for the full text."


class TSICLError(Exception):
    """Base class for this module's explicit error types."""


class TSICLDependencyError(TSICLError):
    """`tsicl`/`torch` is not importable, or the loaded model does not
    expose the expected API surface (e.g. `max_context_length`, `.impute`).
    Raised only by `load_tsicl_strict()` -- `load_tsicl()` catches this and
    returns a status object instead."""


class TSICLProvenanceError(TSICLError):
    """The loaded checkpoint's revision, filename, or SHA-256 does not match
    the pinned provenance (`CHECKPOINT_REVISION`/`CHECKPOINT_SHA256` above),
    or the installed `tsicl` package version differs from
    `EXPECTED_TSICL_PACKAGE_VERSION`. Raised only by `load_tsicl_strict()`
    and `verify_checkpoint_provenance()` -- a reproducibility-focused run
    must never silently proceed on an unverified checkpoint."""


class TSICLInputError(TSICLError):
    """A call to `run_gap_inference`/`impute_masked_series` was given
    malformed input: covariate dates that do not align with the target
    series, a hidden target value reachable from the model input, a
    non-finite covariate/target value where one is not permitted, or an
    unrecognized context mode."""


class TSICLOutputError(TSICLError):
    """TS-ICL's own `model.impute(...)` call returned output that fails a
    basic sanity check: wrong shape, non-finite point prediction, or
    quantiles that are not monotonically non-decreasing across levels.
    Raised by `run_gap_inference` in strict/reproducibility mode; never
    silently replaced with interpolation or a cached fallback value."""


def verify_checkpoint_provenance(model) -> dict:
    """Verify a loaded `TSICL()` model instance's checkpoint against the
    pinned provenance (`CHECKPOINT_REVISION`/`CHECKPOINT_SHA256`).

    Locates the checkpoint blob via the Hugging Face Hub's local cache
    layout (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/
    <revision>/<filename>`, following `tsicl.pipeline`'s own
    `hf_hub_download` mechanism -- this module never downloads or vendors
    the checkpoint itself, only locates and hashes the file the `tsicl`
    package already fetched). Raises `TSICLProvenanceError` if the file is
    not found at the pinned revision, or if its SHA-256 does not match.

    Returns a dict of the verified provenance facts (never raises after
    building this successfully) -- useful for run-metadata provenance
    logging even outside `load_tsicl_strict()`.
    """
    cache_dir_name = f"models--{CHECKPOINT_REPO_ID.replace('/', '--')}"
    checkpoint_path = (
        Path.home() / ".cache" / "huggingface" / "hub" / cache_dir_name
        / "snapshots" / CHECKPOINT_REVISION / CHECKPOINT_FILENAME
    )
    if not checkpoint_path.exists():
        raise TSICLProvenanceError(
            f"Expected checkpoint not found at pinned revision {CHECKPOINT_REVISION!r}: "
            f"{checkpoint_path}. Either the checkpoint has not been downloaded yet (loading "
            f"TSICL() once with network access will fetch it), or a different revision is "
            f"cached -- a reproducibility run must not silently use an unverified checkpoint."
        )
    actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha256 != CHECKPOINT_SHA256:
        raise TSICLProvenanceError(
            f"Checkpoint SHA-256 mismatch at {checkpoint_path}: expected {CHECKPOINT_SHA256}, "
            f"got {actual_sha256}. Do not proceed -- this is not the pinned checkpoint."
        )
    max_context_length = getattr(model, "max_context_length", None)
    if max_context_length is None:
        raise TSICLProvenanceError(
            "Loaded model object has no `max_context_length` attribute -- does not look like "
            "a TSICL() instance from the pinned package version."
        )
    return {
        "checkpoint_repo_id": CHECKPOINT_REPO_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_sha256": actual_sha256,
        "checkpoint_path": str(checkpoint_path),
        "max_context_length": int(max_context_length),
    }


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


def load_tsicl_strict() -> tuple[object, dict]:
    """Load TS-ICL for a reproducibility-focused benchmark run. Raises
    instead of falling back.

    Unlike `load_tsicl()`, this:

    - raises `TSICLDependencyError` if `torch`/`tsicl` are not importable;
    - raises `TSICLProvenanceError` if the installed `tsicl` package version
      does not match `EXPECTED_TSICL_PACKAGE_VERSION`, or if the loaded
      checkpoint's revision/hash does not match the pinned provenance
      (`verify_checkpoint_provenance`);
    - never returns `(None, ...)` -- either a verified, loaded model and its
      provenance dict, or an exception.

    Returns `(model, provenance)` where `provenance` includes everything
    `verify_checkpoint_provenance` returns plus package/runtime versions,
    suitable for direct inclusion in a benchmark run's metadata file.
    """
    try:
        import importlib.metadata

        import torch
        from tsicl import TSICL
    except ImportError as e:
        raise TSICLDependencyError(
            f"tsicl/torch not importable: {e}. Install the optional 'tsicl' extra "
            f"(see docs/methodology/tsicl_usage.md) before running a live benchmark."
        ) from e

    try:
        installed_version = importlib.metadata.version("tsicl")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if installed_version != EXPECTED_TSICL_PACKAGE_VERSION:
        raise TSICLProvenanceError(
            f"Installed tsicl package version {installed_version!r} does not match the "
            f"pinned {EXPECTED_TSICL_PACKAGE_VERSION!r} this benchmark's provenance was "
            f"verified against. Do not proceed on an unverified package version."
        )

    model = TSICL()
    provenance = verify_checkpoint_provenance(model)
    provenance.update({
        "tsicl_package_version": installed_version,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })
    return model, provenance


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
        Array of shape (T, C) -- T timesteps, C covariate channels.
    """
    if covariate_array.ndim != 2:
        raise ValueError("covariate_array must be 2D: (T, C)")
    # .copy() (not just np.ascontiguousarray, which is a no-op and can return a
    # read-only view if the input is already contiguous float32) guarantees a
    # writable array, which avoids a PyTorch UserWarning -- that warning's
    # default text embeds this file's local absolute filesystem path, which we
    # do not want leaking into any executed-notebook output.
    return np.ascontiguousarray(covariate_array[None, :, :], dtype=np.float32).copy()


# ── Gap/context orchestration (the arm-dispatch/context-mode-slicing layer
# previously withheld from publication -- see docs/methodology/tsicl_usage.md
# for why it is now published as original wrapper code) ────────────────────

CONTEXT_MODES = ["full_series", "local_window", "edge_balanced"]


@dataclass
class GapSpec:
    """One artificial (or real) gap to reconstruct: a contiguous span of
    hidden days within a longer daily series."""

    gap_id: str
    start_date: str
    end_date: str
    length: int
    extra: dict = field(default_factory=dict)


def get_gap_row_bounds(dates: np.ndarray, gap: GapSpec) -> tuple[int, int]:
    """Return `(start_idx, end_idx)` half-open row-index bounds of `gap`
    within a sorted array of `numpy.datetime64` dates. Raises
    `TSICLInputError` if either boundary date is not present in `dates`."""
    start = np.datetime64(gap.start_date)
    end = np.datetime64(gap.end_date)
    start_matches = np.where(dates == start)[0]
    end_matches = np.where(dates == end)[0]
    if len(start_matches) == 0 or len(end_matches) == 0:
        raise TSICLInputError(
            f"gap {gap.gap_id!r}: start_date={gap.start_date} or end_date={gap.end_date} "
            f"not found in the provided date index."
        )
    return int(start_matches[0]), int(end_matches[0]) + 1


def slice_context(
    n_rows: int, gap_start_idx: int, gap_end_idx: int, mode: str, window_days: int = 730,
) -> tuple[int, int]:
    """Return `(lo, hi)` half-open row-index bounds of the visible context
    window around a gap, for one of `CONTEXT_MODES`.

    - `"full_series"`: the entire available series.
    - `"local_window"`/`"edge_balanced"`: `window_days` total, split evenly
      before/after the gap (both modes currently compute the same window;
      kept as two names because the private project's released benchmark
      distinguishes them in its arm/context bookkeeping even though the
      slicing itself coincides for this series).

    Raises `TSICLInputError` for an unrecognized mode -- never silently
    falls back to a default.
    """
    if mode == "full_series":
        return 0, n_rows
    if mode in ("local_window", "edge_balanced"):
        half = window_days // 2
        lo = max(0, gap_start_idx - half)
        hi = min(n_rows, gap_end_idx + half)
        return lo, hi
    raise TSICLInputError(f"unrecognized context mode {mode!r}; choose from {CONTEXT_MODES}")


def fold_safe_climatology(
    dates: np.ndarray, target_log10: np.ndarray, gap_mask: np.ndarray,
) -> np.ndarray:
    """Day-of-year climatology of `target_log10`, excluding `gap_mask` days.

    Returns an array the same length as `target_log10`: each position gets
    the mean of all *other*, non-hidden observations sharing its day-of-year
    -- "fold-safe" because a gap's own days never contribute to the
    climatology used to compute its own anomaly (leakage guard). Used only
    by the `target_repr="anomaly"` de-trending option in `run_gap_inference`
    -- the released benchmark's primary configuration uses `target_repr="raw"`
    (no de-trending) throughout; anomaly mode exists for the sensitivity/
    exploratory arms that used it, and is not silently substituted for raw.
    """
    doy = pd_dayofyear(dates)
    target = target_log10.copy()
    target[gap_mask] = np.nan
    clim = np.full(367, np.nan)
    for d in range(1, 367):
        vals = target[doy == d]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            clim[d] = float(np.mean(vals))
    # Fill any day-of-year bins with no observations (e.g. Feb 29 in a
    # thin record) by linear interpolation over the climatology curve itself.
    clim = _interpolate_1d(clim)
    return clim[doy]


def pd_dayofyear(dates: np.ndarray) -> np.ndarray:
    """Day-of-year (1-366) for an array of `numpy.datetime64[D]` dates,
    without requiring pandas as a hard dependency of this module."""
    years = dates.astype("datetime64[Y]")
    jan1 = years.astype("datetime64[D]")
    return (dates.astype("datetime64[D]") - jan1).astype(int) + 1


def _interpolate_1d(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN runs in a 1D array using its own non-NaN
    values (including extrapolation at the ends by nearest-value carry) --
    a tiny local helper so this module does not need pandas for the one
    `fold_safe_climatology` gap-fill it uses."""
    out = arr.copy()
    valid = ~np.isnan(out)
    if valid.sum() == 0:
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def run_gap_inference(
    model,
    dates: np.ndarray,
    target_log10: np.ndarray,
    gap: GapSpec,
    context_mode: str = "full_series",
    covariate_array: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    window_days: int = 730,
    strict: bool = True,
) -> dict:
    """Reconstruct one gap: slice context, mask the gap, call TS-ICL, and
    validate the output. The authoritative per-gap orchestration shared by
    every benchmark driver in `experiments/chlorophyll/`.

    Parameters
    ----------
    model:
        A loaded TS-ICL model instance (`load_tsicl_strict()` for a
        reproducibility run; `load_tsicl()` for a lenient/demo context).
    dates:
        Sorted 1D array of `numpy.datetime64[D]`, length T, the full series'
        date index (context is a sub-window of this).
    target_log10:
        1D array of length T, log10-scale, aligned to `dates`. Must **not**
        already have the gap masked -- this function does the masking, so
        the gap's true values are read from here only for scoring metadata,
        never passed into the model input.
    gap:
        The `GapSpec` to reconstruct.
    context_mode:
        One of `CONTEXT_MODES`.
    covariate_array:
        Optional `(T, C)` covariate array aligned to `dates`/`target_log10`.
        May contain NaN (e.g. cloud-masked satellite-proxy days) -- TS-ICL's
        own `impute()` call handles this internally, matching the
        authoritative private benchmark's behavior over the same real,
        sparsely-missing covariate columns. Shape/alignment mismatches are
        still rejected (`TSICLInputError`).
    window_days:
        Total context window width for `local_window`/`edge_balanced` modes.
    strict:
        If True (default), raise `TSICLOutputError` on a malformed model
        output (wrong shape, non-finite point prediction, non-monotonic
        quantiles) instead of returning a partial/failed result silently.
        A resumable batch driver should catch the raised error itself and
        record it as an explicit per-gap failure -- this function never
        substitutes interpolation or a cached value for a failed call.

    Returns a dict with `gap_id`, `context_mode`, `n_context`, `dates`
    (the hidden days' own dates), `pred_log10`, `quantiles_log10`,
    `true_log10` (the withheld truth, for scoring only -- never was part of
    the model input).
    """
    if context_mode not in CONTEXT_MODES:
        raise TSICLInputError(f"unrecognized context mode {context_mode!r}; choose from {CONTEXT_MODES}")
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    n_rows = len(dates)
    start_idx, end_idx = get_gap_row_bounds(dates, gap)
    lo, hi = slice_context(n_rows, start_idx, end_idx, context_mode, window_days=window_days)

    n_window = hi - lo
    max_ctx = getattr(model, "max_context_length", None)
    if max_ctx is not None and n_window > max_ctx:
        excess = n_window - max_ctx
        cut_lo = excess // 2
        cut_hi = excess - cut_lo
        lo = lo + cut_lo
        hi = hi - cut_hi

    gap_lo_local = start_idx - lo
    gap_hi_local = end_idx - lo

    series = target_log10[lo:hi].astype(np.float32).copy()
    true_gap = target_log10[start_idx:end_idx].astype(np.float32).copy()

    # The single most important leakage guard in this function: the hidden
    # gap's own values are masked to NaN in the model input, unconditionally,
    # regardless of what `target_log10` contains at those positions.
    series[gap_lo_local:gap_hi_local] = np.nan

    covars_window = None
    if covariate_array is not None:
        covars_window = np.asarray(covariate_array, dtype=np.float32)[lo:hi].copy()
        # NaN covariate values (e.g. cloud-masked satellite-proxy days) are
        # NOT rejected here -- the authoritative private benchmark ran its
        # satellite-proxy arm over the same real, sparsely-missing covariate
        # columns with 0 call failures, so TS-ICL's own `impute()` evidently
        # handles NaN covariate positions internally. Rejecting them here
        # would be *stricter* than the reproduced behavior, not a leakage
        # guard -- only genuine shape/alignment problems are checked below.
        if covars_window.shape[0] != series.shape[0]:
            raise TSICLInputError(
                f"gap {gap.gap_id!r}: covariate window length {covars_window.shape[0]} != "
                f"target window length {series.shape[0]} -- covariate dates do not align."
            )

    mean, quantiles = impute_masked_series(model, series, covars_window, quantile_levels)

    pred_log10_gap = mean[gap_lo_local:gap_hi_local]
    q_log10_gap = quantiles[gap_lo_local:gap_hi_local]

    if strict:
        if pred_log10_gap.shape != (gap.length,):
            raise TSICLOutputError(
                f"gap {gap.gap_id!r}: prediction shape {pred_log10_gap.shape} != "
                f"expected ({gap.length},)."
            )
        if q_log10_gap.shape != (gap.length, len(quantile_levels)):
            raise TSICLOutputError(
                f"gap {gap.gap_id!r}: quantile shape {q_log10_gap.shape} != "
                f"expected ({gap.length}, {len(quantile_levels)})."
            )
        if not np.isfinite(pred_log10_gap).all():
            raise TSICLOutputError(f"gap {gap.gap_id!r}: prediction contains non-finite values.")
        if q_log10_gap.size and not np.all(np.diff(q_log10_gap, axis=1) >= -1e-9):
            raise TSICLOutputError(f"gap {gap.gap_id!r}: quantiles are not monotonically ordered.")

    return {
        "gap_id": gap.gap_id,
        "context_mode": context_mode,
        "n_context": hi - lo,
        "dates": dates[start_idx:end_idx],
        "pred_log10": pred_log10_gap,
        "quantiles_log10": q_log10_gap,
        "quantile_levels": list(quantile_levels),
        "true_log10": true_gap,
    }


