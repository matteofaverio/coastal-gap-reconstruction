"""Run the TS-ICL covariate-arm dissection on the chlorophyll artificial-gap
pool: every retained arm in `tsicl_covariate_registry.COVARIATE_ARMS`, plus
the placebo/negative-control battery for the arms it applies to.

Requires the optional `tsicl` extra. Uses `full_series` context only,
matching the released covariate-mechanism ranking's own configuration
(`chlorophyll_covariate_mechanism_summary.csv` reports one context mode per
arm, not a context-mode-by-arm grid).

Usage:
    python -m experiments.chlorophyll.run_tsicl_covariate_analysis \\
        --arms curated_physical,satellite_proxy --support full_681 \\
        --out build/chlorophyll/tsicl_covariates

Restart-safe on the same `predictions.jsonl`/`done_keys.json`/
`failures.jsonl` pattern as `run_tsicl_benchmark.py`. Current/transport arms
(`requires_extended_table=True`) load the full 265-column feature snapshot
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
from .run_tsicl_benchmark import load_gap_specs, load_target_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_EXTENSION_PATH = REPO_ROOT / "data_public" / "shared" / "external_current_kinematic_extension.csv"
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
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dates, target_log10 = load_target_series(target_path)

    needs_extended = any(reg.COVARIATE_ARMS[a].requires_extended_table for a in arm_ids)
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
        (out_dir / "RUN_STATUS").write_text(f"RUN_FAILED: {exc}\n")
        return 1
    print(f"loaded TS-ICL: {provenance}", flush=True)

    variants: list[tuple[str, str, int | None]] = []  # (arm_id, placebo_transform_or_none, seed)
    for arm_id in arm_ids:
        variants.append((arm_id, "none", None))
        if include_placebos and arm_id in reg.PLACEBO_ELIGIBLE_ARMS:
            for transform in reg.PLACEBO_TRANSFORMS:
                variants.append((arm_id, transform, 0))

    done_path = out_dir / "done_keys.json"
    done_keys: set[str] = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    if done_keys:
        print(f"resuming: {len(done_keys)} calls already done", flush=True)

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"
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

    n_total = len(specs) * len(variants)
    t_start = time.time()
    n_done_this_run = 0
    n_fail = 0

    for gap in specs:
        for arm_id, transform, _seed in variants:
            key = f"{gap.gap_id}|{CONTEXT_MODE}|{arm_id}|{transform}"
            if key in done_keys:
                continue
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
                n_fail += 1
                f_fail.write(json.dumps({
                    "key": key, "gap_id": gap.gap_id, "arm": arm_id, "placebo_transform": transform,
                    "error_type": type(exc).__name__, "error": str(exc),
                }) + "\n")
                print(f"[FAILED] {key}: {exc}", flush=True)

            done_keys.add(key)
            n_done_this_run += 1
            if n_done_this_run % CHECKPOINT_EVERY == 0:
                f_pred.flush()
                f_fail.flush()
                done_path.write_text(json.dumps(sorted(done_keys)))
                elapsed = time.time() - t_start
                rate = n_done_this_run / elapsed
                remaining = (n_total - len(done_keys)) / max(rate, 1e-9)
                print(
                    f"progress: {len(done_keys)}/{n_total} this_run={n_done_this_run} "
                    f"fail={n_fail} elapsed={elapsed:.0f}s est_remaining={remaining:.0f}s",
                    flush=True,
                )

    f_pred.flush()
    f_pred.close()
    f_fail.flush()
    f_fail.close()
    done_path.write_text(json.dumps(sorted(done_keys)))

    run_status = "RUN_COMPLETE" if len(done_keys) == n_total and n_fail == 0 else (
        "RUN_FAILED" if len(done_keys) == 0 else "RUN_PARTIAL"
    )
    (out_dir / "RUN_STATUS").write_text(
        f"{run_status}: {len(done_keys)}/{n_total} calls done, {n_fail} failure(s) total in failures.jsonl\n"
    )
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "arms": arm_ids, "support": support, "include_placebos": include_placebos,
        "n_variants": len(variants), "n_gaps": len(specs), "n_total_calls": n_total,
        "target_sha256": _sha256_file(target_path), "features_sha256": _sha256_file(features_path),
        "gap_pool_sha256": _sha256_file(pool_path),
        "provenance": provenance, "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "command_args": sys.argv[1:],
    }, indent=2))
    print(f"RUN_STATUS: {run_status} ({len(done_keys)}/{n_total}, {n_fail} failures)")
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
    args = parser.parse_args(argv)

    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arm_ids) - set(reg.COVARIATE_ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}; choose from {sorted(reg.COVARIATE_ARMS)}")

    return run_analysis(
        arm_ids, args.support, args.include_placebos,
        args.target, args.features, args.extension, args.out,
    )


if __name__ == "__main__":
    sys.exit(main())
