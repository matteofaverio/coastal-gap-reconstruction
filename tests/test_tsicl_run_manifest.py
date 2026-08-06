"""Tests for configuration-bound TS-ICL run-output directories
(`experiments.chlorophyll.tsicl_run_manifest`)."""

from __future__ import annotations

import pytest

from experiments.chlorophyll import tsicl_run_manifest as rm

_BASE_KWARGS = dict(
    driver="run_tsicl_benchmark", support="full_681", arms=["target_only"],
    context_modes=["full_series"], placebo_config={},
    target_sha256="a" * 64, features_sha256="b" * 64, gap_pool_sha256="c" * 64,
    provenance={"tsicl_package_version": "0.2.1", "torch_version": "2.9.1",
                "checkpoint_revision": "rev1", "checkpoint_sha256": "d" * 64},
    target_transform="log10", context_window_settings={"window_days": 730},
    quantile_levels=[0.5],
)


def test_identity_is_order_independent_for_arms_and_context_modes():
    id1 = rm.build_run_identity(arms=["a", "b"], context_modes=["full_series", "edge_balanced"],
                                 **{k: v for k, v in _BASE_KWARGS.items() if k not in ("arms", "context_modes")})
    id2 = rm.build_run_identity(arms=["b", "a"], context_modes=["edge_balanced", "full_series"],
                                 **{k: v for k, v in _BASE_KWARGS.items() if k not in ("arms", "context_modes")})
    assert rm.config_hash(id1) == rm.config_hash(id2)


@pytest.mark.parametrize("field,new_value", [
    ("support", "matched_449"),
    ("arms", ["target_only", "satellite_proxy"]),
    ("context_modes", ["local_window"]),
    ("target_sha256", "z" * 64),
    ("features_sha256", "z" * 64),
    ("gap_pool_sha256", "z" * 64),
    ("quantile_levels", [0.1, 0.9]),
])
def test_config_hash_changes_when_a_field_changes(field, new_value):
    id1 = rm.build_run_identity(**_BASE_KWARGS)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2[field] = new_value
    id2 = rm.build_run_identity(**kwargs2)
    assert rm.config_hash(id1) != rm.config_hash(id2)


@pytest.mark.parametrize("prov_field", ["checkpoint_revision", "checkpoint_sha256",
                                          "tsicl_package_version", "torch_version"])
def test_config_hash_changes_when_checkpoint_provenance_changes(prov_field):
    id1 = rm.build_run_identity(**_BASE_KWARGS)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2["provenance"] = dict(_BASE_KWARGS["provenance"])
    kwargs2["provenance"][prov_field] = "different"
    id2 = rm.build_run_identity(**kwargs2)
    assert rm.config_hash(id1) != rm.config_hash(id2)


def test_first_write_creates_manifest(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    h = rm.write_or_validate_manifest(tmp_path, identity)
    assert (tmp_path / "run_manifest.json").exists()
    assert h == rm.config_hash(identity)


def test_resume_with_identical_config_succeeds(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity)
    # Second invocation, freshly rebuilt identity, same inputs -- must not raise.
    identity2 = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity2)


def test_resume_with_different_support_raises(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2["support"] = "matched_449"
    identity2 = rm.build_run_identity(**kwargs2)
    with pytest.raises(rm.RunConfigMismatchError, match="different configuration"):
        rm.write_or_validate_manifest(tmp_path, identity2)


def test_resume_with_different_arm_list_raises(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2["arms"] = ["target_only", "satellite_proxy"]
    identity2 = rm.build_run_identity(**kwargs2)
    with pytest.raises(rm.RunConfigMismatchError):
        rm.write_or_validate_manifest(tmp_path, identity2)


def test_resume_with_different_checkpoint_hash_raises(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2["provenance"] = dict(_BASE_KWARGS["provenance"])
    kwargs2["provenance"]["checkpoint_sha256"] = "different_hash"
    identity2 = rm.build_run_identity(**kwargs2)
    with pytest.raises(rm.RunConfigMismatchError):
        rm.write_or_validate_manifest(tmp_path, identity2)


def test_mismatch_error_message_instructs_a_new_output_directory(tmp_path):
    identity = rm.build_run_identity(**_BASE_KWARGS)
    rm.write_or_validate_manifest(tmp_path, identity)
    kwargs2 = dict(_BASE_KWARGS)
    kwargs2["support"] = "matched_449"
    identity2 = rm.build_run_identity(**kwargs2)
    with pytest.raises(rm.RunConfigMismatchError) as excinfo:
        rm.write_or_validate_manifest(tmp_path, identity2)
    assert "new, empty --out directory" in str(excinfo.value)
