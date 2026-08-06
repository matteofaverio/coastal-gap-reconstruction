"""Configuration-bound run identity for the TS-ICL benchmark drivers.

An output directory accumulates `predictions.jsonl`/`failures.jsonl` across
possibly many resumed invocations. Before this module existed, nothing
stopped two invocations with *different* configurations (a different arm
list, a different support, a different checkpoint) from writing into the
same directory and silently mixing calls from two incompatible runs.

`build_run_identity` captures every input that changes what a call means;
`config_hash` reduces that to one stable string; `write_or_validate_manifest`
either creates `run_manifest.json` (first invocation) or requires an exact
match (every resume) -- a mismatch raises `RunConfigMismatchError` with an
explicit instruction to use a new output directory rather than silently
proceeding or silently overwriting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class RunConfigMismatchError(Exception):
    """Raised when an output directory's existing `run_manifest.json` does
    not match the configuration of the current invocation."""


# Fields that describe *what a call means* -- included in the identity hash.
# Anything not listed here (e.g. a timestamp) must never affect the hash.
_IDENTITY_FIELDS = (
    "driver", "support", "arms", "context_modes", "placebo_config",
    "target_sha256", "features_sha256", "extension_sha256", "gap_pool_sha256",
    "tsicl_package_version", "torch_version", "checkpoint_revision", "checkpoint_sha256",
    "target_transform", "context_window_settings", "quantile_levels",
)


def build_run_identity(
    driver: str,
    support: str,
    arms: list[str],
    context_modes: list[str],
    placebo_config: dict,
    target_sha256: str,
    features_sha256: str,
    gap_pool_sha256: str,
    provenance: dict,
    target_transform: str,
    context_window_settings: dict,
    quantile_levels: list[float],
    extension_sha256: str | None = None,
) -> dict:
    """Assemble the canonical run-identity dict. `arms`/`context_modes` are
    sorted before hashing so argument order never changes the identity."""
    return {
        "driver": driver,
        "support": support,
        "arms": sorted(arms),
        "context_modes": sorted(context_modes),
        "placebo_config": placebo_config,
        "target_sha256": target_sha256,
        "features_sha256": features_sha256,
        "extension_sha256": extension_sha256,
        "gap_pool_sha256": gap_pool_sha256,
        "tsicl_package_version": provenance.get("tsicl_package_version"),
        "torch_version": provenance.get("torch_version"),
        "checkpoint_revision": provenance.get("checkpoint_revision"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "target_transform": target_transform,
        "context_window_settings": context_window_settings,
        "quantile_levels": list(quantile_levels),
    }


def config_hash(identity: dict) -> str:
    canonical = json.dumps({k: identity[k] for k in _IDENTITY_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_or_validate_manifest(out_dir: Path, identity: dict) -> str:
    """Write `run_manifest.json` on first use of `out_dir`; on any later
    invocation, require the recomputed identity to hash-match exactly.

    Returns the config hash. Raises `RunConfigMismatchError` on a mismatch
    -- callers must not proceed, and must not overwrite the existing
    manifest; the error message instructs the maintainer to pick a new
    output directory.
    """
    manifest_path = out_dir / "run_manifest.json"
    new_hash = config_hash(identity)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_hash = existing.get("config_hash")
        if existing_hash != new_hash:
            raise RunConfigMismatchError(
                f"{out_dir} already holds a run with a different configuration "
                f"(existing config_hash={existing_hash}, this invocation's config_hash={new_hash}). "
                f"Resuming a benchmark output directory requires an exact configuration match -- "
                f"predictions from two different configurations must never be mixed. Use a new, "
                f"empty --out directory for this configuration instead.\n"
                f"Existing identity: {json.dumps(existing.get('identity'), indent=2, default=str)}\n"
                f"This invocation's identity: {json.dumps(identity, indent=2, default=str)}"
            )
        return new_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"config_hash": new_hash, "identity": identity}, indent=2, default=str))
    return new_hash
