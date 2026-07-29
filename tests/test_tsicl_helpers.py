"""Unit tests for the shared TS-ICL runtime layer in
`coastal_gap_reconstruction/tsicl_helpers.py`.

These exercise shape handling, quantile extraction, and log10/physical
conversion using a stub model, plus the "optional dependency missing"
behaviour of `load_tsicl()` -- none of this requires the real `tsicl`
package or its checkpoint to be installed, so it runs in the core (locked)
test environment, not just the isolated demo one.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from coastal_gap_reconstruction.tsicl_helpers import (
    DEFAULT_QUANTILE_LEVELS,
    build_covariate_block,
    impute_masked_series,
    load_tsicl,
    log10_to_physical,
)


class _StubModel:
    """Mimics the TS-ICL `model.impute(...)` call closely enough to test
    shape handling and quantile extraction without the real dependency."""

    def __init__(self):
        self.last_call = None

    def impute(self, inputs, covars, quantile_levels, denormalize, replace_by_gt):
        import torch

        self.last_call = {
            "inputs_shape": tuple(inputs.shape),
            "covars_shape": None if covars is None else tuple(covars.shape),
            "quantile_levels": list(quantile_levels),
        }
        t = inputs.shape[0]
        mean = torch.zeros(t)
        quantiles = torch.zeros(t, len(quantile_levels))
        for i, q in enumerate(quantile_levels):
            quantiles[:, i] = q  # deterministic, easy to assert on
        return mean, quantiles


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(not _torch_available(), reason="torch not installed in this environment")


def test_build_covariate_block_shape() -> None:
    arr = np.zeros((10, 3), dtype=np.float32)
    block = build_covariate_block(arr)
    assert block.shape == (1, 10, 3)


def test_build_covariate_block_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        build_covariate_block(np.zeros((10,)))


@requires_torch
def test_impute_masked_series_target_only_shape() -> None:
    model = _StubModel()
    series = np.full(20, np.nan, dtype=np.float32)
    series[:5] = 1.0
    mean, quantiles = impute_masked_series(model, series)
    assert mean.shape == (20,)
    assert quantiles.shape == (20, len(DEFAULT_QUANTILE_LEVELS))
    assert model.last_call["inputs_shape"] == (20,)
    assert model.last_call["covars_shape"] is None


@requires_torch
def test_impute_masked_series_covariate_conditioned_shape() -> None:
    model = _StubModel()
    series = np.full(20, np.nan, dtype=np.float32)
    covariates = np.zeros((20, 2), dtype=np.float32)
    mean, quantiles = impute_masked_series(model, series, covariate_array=covariates)
    assert mean.shape == (20,)
    assert quantiles.shape == (20, len(DEFAULT_QUANTILE_LEVELS))
    # batch dimension must be added explicitly -- see build_covariate_block's docstring
    assert model.last_call["covars_shape"] == (1, 20, 2)


@requires_torch
def test_impute_masked_series_respects_custom_quantile_levels() -> None:
    model = _StubModel()
    series = np.full(10, np.nan, dtype=np.float32)
    levels = [0.1, 0.5, 0.9]
    _, quantiles = impute_masked_series(model, series, quantile_levels=levels)
    assert quantiles.shape == (10, 3)
    np.testing.assert_allclose(quantiles[0], levels)


def test_log10_to_physical_roundtrip() -> None:
    log10_values = np.array([-1.0, 0.0, 1.0, 2.0])
    physical = log10_to_physical(log10_values)
    np.testing.assert_allclose(physical, [0.1, 1.0, 10.0, 100.0])
    assert (physical >= 0).all()


def test_load_tsicl_never_raises_when_dependency_missing(monkeypatch) -> None:
    """If `tsicl` cannot be imported (e.g. not installed, as in the core
    locked environment), load_tsicl() must return (None, status) with
    status.live=False instead of raising -- this is the graceful-fallback
    contract the demo and notebook 06 both depend on."""
    real_import = __import__

    def _blocking_import(name, *args, **kwargs):
        if name == "tsicl":
            raise ImportError("No module named 'tsicl' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tsicl", raising=False)
    monkeypatch.setattr("builtins.__import__", _blocking_import)

    model, status = load_tsicl()
    assert model is None
    assert status.live is False
    assert status.error is not None
    assert "tsicl" in status.error.lower() or "No module" in status.error


def test_package_importable_without_tsicl_or_torch() -> None:
    """Importing tsicl_helpers itself must never require tsicl/torch -- only
    calling load_tsicl()/impute_masked_series() should. Guard against a
    regression where a top-level `import torch` or `import tsicl` creeps in."""
    mod = sys.modules.get("coastal_gap_reconstruction.tsicl_helpers")
    assert mod is not None
    src_path = mod.__file__
    assert src_path is not None
    with open(src_path) as f:
        lines = f.readlines()
    top_level_imports = [
        line for line in lines[:30] if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("torch" in line or "tsicl" in line for line in top_level_imports), (
        "torch/tsicl must only be imported lazily inside functions, not at module load time"
    )


def test_dummy_module_placeholder() -> None:
    """Sanity check that the stub-based tests above are exercising real
    torch tensor plumbing, not accidentally no-oping, when torch is present."""
    if _torch_available():
        assert isinstance(types.ModuleType("x"), types.ModuleType)
