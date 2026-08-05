"""Run the TS-ICL covariate-arm dissection on the chlorophyll artificial-gap
pool: every retained arm in `tsicl_covariate_registry.COVARIATE_ARMS`
(18 base arms), plus the placebo/negative-control battery
(4 transforms x 6 eligible families = 24 variants) for the full authoritative
grid -- 42 variants total, matching the private
the original covariate-dissection run-plan builder (681 gaps x 42
= 28,602 calls for the full run) exactly, not approximately.

Requires the optional `tsicl` extra. Uses `full_series` context only,
matching the released covariate-mechanism ranking's own configuration
(`chlorophyll_covariate_mechanism_summary.csv` reports one context mode per
arm, not a context-mode-by-arm grid).

Usage:
    python -m experiments.chlorophyll.run_tsicl_covariate_analysis \\
        --arms curated_physical,satellite_proxy --support full_681 \\
        --out build/chlorophyll/tsicl_covariates

Restart-safe on the same failure-safe accounting as `run_tsicl_benchmark.py`
(`tsicl_run_state.py`: state always recomputed from `predictions.jsonl`/
`failures.jsonl`, failed calls retried by default) and the same
configuration-bound output directory (`tsicl_run_manifest.py`).
Current/transport arms (`requires_extended_table=True`) load the full
265-column feature snapshot
(`coastal_gap_reconstruction.feature_tables.load_full_feature_table`)
instead of the base 126-column table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th
from coastal_gap_reconstruction.feature_tables import load_feature_table, load_full_feature_table

from . import tsicl_contract as tc
from . import tsicl_covariate_registry as reg
from . import tsicl_run_manifest as rm
from . import tsicl_run_state as rs
from .run_tsicl_benchmark import load_gap_specs, load_target_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_EXTENSION_PATH = REPO_ROOT / "data" / "shared" / "external_current_kinematic_extension.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "chlorophyll" / "tsicl_covariates"

CHECKPOINT_EVERY = 25
CONTEXT_MODE = "full_series"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _covariate_block(arm_id: str, features_df: pd.DataFrame) -> np.ndarray | None:
    spec = reg.COVARIATE_ARMS[arm_id]
    if not spec.columns:
        return None
    df = features_df
    if arm_id in ("solar_upwelling_interaction", "upwelling_cooling_interaction"):
        df = reg.build_engineered_products(df)
    return df[spec.columns].to_numpy(dtype=np.float32)


def run_analysis(
    arm_ids: list[str], support: str, include_placebos: bool,
    target_path: Path, features_path: Path, extension_path: Path, out_dir: Path,
    max_attempts: int | None = None,
) -> int:
    unknown = set(arm_ids) - set(reg.COVARIATE_ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}; choose from {sorted(reg.COVARIATE_ARMS)}")

    dates, target_log10 = load_target_series(target_path)

    needs_extended = any(reg.COVARIATE_ARMS[a].requires_extended_table for a in arm_ids)
    extension_sha256 = _sha256_file(extension_path) if needs_extended else None
    features_df = (
        load_full_feature_table(features_path, extension_path) if needs_extended
        else load_feature_table(features_path)
    )
    features_df = features_df.reindex(pd.to_datetime(dates))

    full_pool = pd.read_csv(tc.FULL_POOL_PATH, parse_dates=["start_date", "end_date"])
    if support == "matched_449":
        matched_ids = set(pd.read_csv(tc.MATCHED_SUPPORT_PATH)["gap_id"])
        pool = full_pool[full_pool["gap_id"].isin(matched_ids)].reset_index(drop=True)
        pool_path = tc.MATCHED_SUPPORT_PATH
    else:
        pool = full_pool
        pool_path = tc.FULL_POOL_PATH
    specs = load_gap_specs(pool)

    try:
        model, provenance = th.load_tsicl_strict()
    except th.TSICLError as exc:
        print(f"FATAL: could not load TS-ICL: {exc}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "RUN_STATUS").write_text(f"RUN_FAILED: {exc}\n")
        return 1
    print(f"loaded TS-ICL: {provenance}", flush=True)

    variants: list[tuple[str, str, int | None]] = []  # (arm_id, placebo_transform_or_none, seed)
    for arm_id in arm_ids:
        variants.append((arm_id, "none", None))
        if include_placebos and arm_id in reg.PLACEBO_ELIGIBLE_ARMS:
            for transform in reg.PLACEBO_TRANSFORMS:
                variants.append((arm_id, transform, 0))

    target_sha256 = _sha256_file(target_path)
    features_sha256 = _sha256_file(features_path)
    gap_pool_sha256 = _sha256_file(pool_path)

    identity = rm.build_run_identity(
        driver="run_tsicl_covariate_analysis", support=support, arms=arm_ids,
        context_modes=[CONTEXT_MODE],
        placebo_config={"include_placebos": include_placebos, "eligible_arms": sorted(reg.PLACEBO_ELIGIBLE_ARMS)},
        target_sha256=target_sha256, features_sha256=features_sha256, extension_sha256=extension_sha256,
        gap_pool_sha256=gap_pool_sha256, provenance=provenance, target_transform=tc.TARGET_TRANSFORM,
        context_window_settings={"window_days": 0, "max_context_length": tc.MAX_CONTEXT_LENGTH},
        quantile_levels=[],
    )
    try:
        rm.write_or_validate_manifest(out_dir, identity)
    except rm.RunConfigMismatchError as exc:
        print(f"FATAL: {exc}")
        return 1

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"

    expected_keys = [
        f"{gap.gap_id}|{CONTEXT_MODE}|{arm_id}|{transform}"
        for gap in specs for arm_id, transform, _seed in variants
    ]
    state = rs.load_run_state(pred_path, fail_path)
    outstanding = state.outstanding(expected_keys, max_attempts)
    if state.successful_keys:
        print(
            f"resuming: {len(state.successful_keys & set(expected_keys))} calls already "
            f"successful, {len(state.failed_keys)} previously-failed keys, "
            f"{len(outstanding)} outstanding this run",
            flush=True,
        )
    if state.duplicate_successful_keys or state.malformed_successful_records:
        print(
            f"WARNING: {len(state.duplicate_successful_keys)} duplicate successful keys, "
            f"{len(state.malformed_successful_records)} malformed successful records found in "
            f"{pred_path} -- these block RUN_COMPLETE until resolved.",
            flush=True,
        )

    f_pred = open(pred_path, "a")
    f_fail = open(fail_path, "a")

    def _json_default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(str(type(o)))

    # Precompute each variant's covariate block once (not per gap).
    blocks: dict[tuple[str, str], np.ndarray | None] = {}
    for arm_id, transform, seed in variants:
        base_block = _covariate_block(arm_id, features_df)
        if base_block is None or transform == "none":
            blocks[(arm_id, transform)] = base_block
        else:
            blocks[(arm_id, transform)] = reg.apply_placebo_transform(base_block, dates, transform, seed=seed)

    gap_by_id = {gap.gap_id: gap for gap in specs}
    t_start = time.time()
    n_done_this_run = 0
    n_fail_this_run = 0

    for key in outstanding:
        gap_id, _mode, arm_id, transform = key.split("|")
        gap = gap_by_id[gap_id]
        covar = blocks[(arm_id, transform)]
        try:
            result = th.run_gap_inference(
                model, dates, target_log10, gap, context_mode=CONTEXT_MODE,
                covariate_array=covar, strict=True,
            )
            row = {
                "key": key, "gap_id": gap.gap_id, "arm": arm_id, "placebo_transform": transform,
                "gap_length": gap.length, "n_context": result["n_context"],
                "date": [str(d) for d in result["dates"]],
                "pred_log10": result["pred_log10"], "true_log10": result["true_log10"],
            }
            f_pred.write(json.dumps(row, default=_json_default) + "\n")
        except th.TSICLError as exc:
            n_fail_this_run += 1
            f_fail.write(json.dumps({
                "key": key, "gap_id": gap.gap_id, "arm": arm_id, "placebo_transform": transform,
                "error_type": type(exc).__name__, "error": str(exc),
            }) + "\n")
            print(f"[FAILED] {key}: {exc}", flush=True)

        n_done_this_run += 1
        if n_done_this_run % CHECKPOINT_EVERY == 0:
            f_pred.flush()
            f_fail.flush()
            elapsed = time.time() - t_start
            rate = n_done_this_run / elapsed
            remaining = (len(outstanding) - n_done_this_run) / max(rate, 1e-9)
            print(
                f"progress: {n_done_this_run}/{len(outstanding)} this_run "
                f"fail_this_run={n_fail_this_run} elapsed={elapsed:.0f}s est_remaining={remaining:.0f}s",
                flush=True,
            )

    f_pred.flush()
    f_pred.close()
    f_fail.flush()
    f_fail.close()

    final_state = rs.load_run_state(pred_path, fail_path)
    run_status, detail = rs.classify_run_status(final_state, expected_keys)
    (out_dir / "RUN_STATUS").write_text(f"{run_status}: {json.dumps(detail)}\n")
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "arms": arm_ids, "support": support, "include_placebos": include_placebos,
        "n_variants": len(variants), "n_gaps": len(specs), "n_expected_calls": len(expected_keys),
        "n_done_this_invocation": n_done_this_run, "n_fail_this_invocation": n_fail_this_run,
        "run_status_detail": detail,
        "target_sha256": target_sha256, "features_sha256": features_sha256,
        "gap_pool_sha256": gap_pool_sha256,
        "provenance": provenance, "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "command_args": sys.argv[1:],
    }, indent=2))
    print(f"RUN_STATUS: {run_status} ({json.dumps(detail)})")
    return 0 if run_status == "RUN_COMPLETE" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=str, default=",".join(
        a for a, s in reg.COVARIATE_ARMS.items() if s.role == "primary" and a != "target_only"
    ))
    parser.add_argument("--support", choices=["full_681", "matched_449"], default="full_681")
    parser.add_argument("--include-placebos", action="store_true")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="Stop retrying a failed key after this many attempts. Default: unlimited retries.",
    )
    args = parser.parse_args(argv)

    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arm_ids) - set(reg.COVARIATE_ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}; choose from {sorted(reg.COVARIATE_ARMS)}")

    return run_analysis(
        arm_ids, args.support, args.include_placebos,
        args.target, args.features, args.extension, args.out,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    sys.exit(main())
