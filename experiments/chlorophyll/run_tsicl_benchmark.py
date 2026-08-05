"""Run the authoritative primary TS-ICL benchmark on the chlorophyll
artificial-gap pool: 681 gaps x 2 context modes x 6 primary arms
(`tsicl_contract.PRIMARY_ARM_ORDER`), `target_repr="raw"` -- resolved by
direct inspection of the private project's primary TS-ICL benchmark driver
and shared call-shaping module (see `tsicl_contract.py`'s "authoritative
primary full benchmark grid" section), not assumed.

Requires the optional `tsicl` extra (see `docs/methodology/tsicl_usage.md`).
Uses `coastal_gap_reconstruction.tsicl_helpers.load_tsicl_strict` -- fails
loudly on any provenance mismatch rather than silently proceeding.

Usage:
    python -m experiments.chlorophyll.run_tsicl_benchmark \\
        --support full_681 --arms target_only,satellite_proxy \\
        --context-modes full_series,edge_balanced \\
        --out build/chlorophyll/tsicl_benchmark

Restart-safe. All resume/failure/completion accounting is delegated to
`tsicl_run_state.py`: state is always recomputed from `predictions.jsonl`/
`failures.jsonl` (never a separate compact cache that can go stale), a
failed call is retried on every subsequent invocation by default (never
silently skipped), and `RUN_COMPLETE` requires every expected call to have a
valid successful record with zero unresolved failures. The output directory
is configuration-bound (`tsicl_run_manifest.py`): resuming with a different
support/arm-list/context-mode/checkpoint/input-file configuration raises
rather than silently mixing incompatible predictions. Never overwrites
`results_public/` by default; output lives under a gitignored `build/`
directory.

A gap's own failure (provenance/input/output error) is recorded in
`failures.jsonl` and the run continues (explicit resumable batch mode, per
`docs/methodology/tsicl_usage.md`'s publication-boundary requirements) --
never silently skipped or replaced with a cached/interpolated value.
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
from coastal_gap_reconstruction.feature_tables import load_feature_table

from . import tsicl_contract as tc
from . import tsicl_covariate_registry as reg
from . import tsicl_run_manifest as rm
from . import tsicl_run_state as rs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "chlorophyll" / "tsicl_benchmark"

CHECKPOINT_EVERY = 25


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target_series(target_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, target_log10) -- dates as datetime64[D], target
    log10-transformed with the released floor (values <= LOG10_FLOOR -> NaN,
    matching the target-only benchmark's own eligibility rule)."""
    df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    eligible = df["target_eligible_default"].fillna(False).astype(bool)
    value = df[tc.TARGET_COLUMN].where(eligible & (df[tc.TARGET_COLUMN] > tc.LOG10_FLOOR))
    target_log10 = np.log10(value).to_numpy(dtype=np.float32)
    dates = df.index.values.astype("datetime64[D]")
    return dates, target_log10


def load_gap_specs(pool: pd.DataFrame) -> list[th.GapSpec]:
    specs = []
    for _, row in pool.iterrows():
        specs.append(th.GapSpec(
            gap_id=str(row["gap_id"]), start_date=str(pd.Timestamp(row["start_date"]).date()),
            end_date=str(pd.Timestamp(row["end_date"]).date()), length=int(row["gap_length"]),
        ))
    return specs


def _load_pool(support: str) -> tuple[pd.DataFrame, Path]:
    full_pool = pd.read_csv(tc.FULL_POOL_PATH, parse_dates=["start_date", "end_date"])
    if support == "matched_449":
        matched_ids = set(pd.read_csv(tc.MATCHED_SUPPORT_PATH)["gap_id"])
        pool = full_pool[full_pool["gap_id"].isin(matched_ids)].reset_index(drop=True)
        return pool, tc.MATCHED_SUPPORT_PATH
    return full_pool, tc.FULL_POOL_PATH


def _covariate_block(arm: str, features_df: pd.DataFrame) -> np.ndarray | None:
    spec = tc.PRIMARY_ARMS[arm]
    if not spec["columns"]:
        return None
    return features_df[spec["columns"]].to_numpy(dtype=np.float32)


def run_benchmark(
    support: str, arms: list[str], context_modes: list[str],
    target_path: Path, features_path: Path, out_dir: Path,
    max_attempts: int | None = None,
) -> int:
    unknown = set(arms) - set(tc.PRIMARY_ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}; choose from {tc.PRIMARY_ARM_ORDER}")

    dates, target_log10 = load_target_series(target_path)
    features_df = load_feature_table(features_path)
    features_df = features_df.reindex(pd.to_datetime(dates))

    pool, pool_path = _load_pool(support)
    specs = load_gap_specs(pool)

    try:
        model, provenance = th.load_tsicl_strict()
    except th.TSICLError as exc:
        print(f"FATAL: could not load TS-ICL: {exc}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "RUN_STATUS").write_text(f"RUN_FAILED: {exc}\n")
        return 1
    print(f"loaded TS-ICL: {provenance}", flush=True)

    target_sha256 = _sha256_file(target_path)
    features_sha256 = _sha256_file(features_path)
    gap_pool_sha256 = _sha256_file(pool_path)

    identity = rm.build_run_identity(
        driver="run_tsicl_benchmark", support=support, arms=arms, context_modes=context_modes,
        placebo_config={}, target_sha256=target_sha256, features_sha256=features_sha256,
        gap_pool_sha256=gap_pool_sha256, provenance=provenance, target_transform=tc.TARGET_TRANSFORM,
        context_window_settings={"window_days": 730, "max_context_length": tc.MAX_CONTEXT_LENGTH},
        quantile_levels=tc.QUANTILE_LEVELS,
    )
    try:
        rm.write_or_validate_manifest(out_dir, identity)
    except rm.RunConfigMismatchError as exc:
        print(f"FATAL: {exc}")
        return 1

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"

    expected_keys = [
        f"{gap.gap_id}|{mode}|{arm}"
        for gap in specs for mode in context_modes for arm in arms
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

    covariate_blocks = {arm: _covariate_block(arm, features_df) for arm in arms}
    placebo_transforms = {arm: tc.PRIMARY_ARMS[arm]["transform"] for arm in arms}
    gap_by_id = {gap.gap_id: gap for gap in specs}

    t_start = time.time()
    n_done_this_run = 0
    n_fail_this_run = 0

    for key in outstanding:
        gap_id, mode, arm = key.split("|")
        gap = gap_by_id[gap_id]
        covar = covariate_blocks[arm]
        transform = placebo_transforms[arm]
        if covar is not None and transform != "none":
            covar = reg.apply_placebo_transform(covar, dates, transform, seed=hash(gap_id) % 10000)
        try:
            result = th.run_gap_inference(
                model, dates, target_log10, gap, context_mode=mode,
                covariate_array=covar, strict=True,
            )
            row = {
                "key": key, "gap_id": gap.gap_id, "context_mode": mode, "arm": arm,
                "gap_length": gap.length, "n_context": result["n_context"],
                "date": [str(d) for d in result["dates"]],
                "pred_log10": result["pred_log10"], "true_log10": result["true_log10"],
                "quantile_levels": result["quantile_levels"],
                "quantiles_log10": result["quantiles_log10"],
            }
            f_pred.write(json.dumps(row, default=_json_default) + "\n")
        except th.TSICLError as exc:
            n_fail_this_run += 1
            f_fail.write(json.dumps({
                "key": key, "gap_id": gap.gap_id, "context_mode": mode, "arm": arm,
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
        "support": support, "arms": arms, "context_modes": context_modes,
        "n_gaps": len(specs), "n_expected_calls": len(expected_keys),
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
    parser.add_argument("--support", choices=["full_681", "matched_449"], default="full_681")
    parser.add_argument("--arms", type=str, default=",".join(tc.PRIMARY_ARM_ORDER))
    parser.add_argument("--context-modes", type=str, default=",".join(tc.PRIMARY_CONTEXT_MODES))
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="Stop retrying a failed key after this many attempts (it remains failed and "
             "blocks RUN_COMPLETE). Default: unlimited retries (a resume always retries "
             "every unresolved failed key).",
    )
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(tc.PRIMARY_ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}; choose from {tc.PRIMARY_ARM_ORDER}")
    context_modes = [m.strip() for m in args.context_modes.split(",") if m.strip()]

    return run_benchmark(
        args.support, arms, context_modes, args.target, args.features, args.out,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    sys.exit(main())
