"""Run the target-only (and satellite-proxy) TS-ICL benchmark on the
chlorophyll artificial-gap pool.

Requires the optional `tsicl` extra (see `docs/methodology/tsicl_usage.md`).
Uses `coastal_gap_reconstruction.tsicl_helpers.load_tsicl_strict` -- fails
loudly on any provenance mismatch rather than silently proceeding.

Usage:
    python -m experiments.chlorophyll.run_tsicl_benchmark \\
        --support full_681 --arms target_only,satellite_proxy \\
        --context-modes full_series,edge_balanced \\
        --out build/chlorophyll/tsicl_benchmark

Restart-safe: writes one JSON line per (gap_id, context_mode, arm) to
`predictions.jsonl` in the output directory, plus a `done_keys.json` set of
completed keys checkpointed every `CHECKPOINT_EVERY` calls. Re-running with
the same arguments resumes from wherever it left off rather than
recomputing already-done calls. Never overwrites `results_public/` by
default; output lives under a gitignored `build/` directory.

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "chlorophyll" / "tsicl_benchmark"

TARGET_ONLY_ARMS = ["target_only", "satellite_proxy"]
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


def run_benchmark(
    support: str, arms: list[str], context_modes: list[str],
    target_path: Path, features_path: Path, out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dates, target_log10 = load_target_series(target_path)
    features_df = load_feature_table(features_path)
    features_df = features_df.reindex(pd.to_datetime(dates))

    # The matched-support manifest carries no start_date/end_date columns of
    # its own -- always load the full pool (which has them) and, for
    # matched_449, filter down to that manifest's gap_ids.
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

    satellite_proxy_block = None
    if "satellite_proxy" in arms:
        cols = reg.COVARIATE_ARMS["satellite_proxy"].columns
        satellite_proxy_block = features_df[cols].to_numpy(dtype=np.float32)

    n_total = len(specs) * len(context_modes) * len(arms)
    t_start = time.time()
    n_done_this_run = 0
    n_fail = 0

    for gap in specs:
        for mode in context_modes:
            for arm in arms:
                key = f"{gap.gap_id}|{mode}|{arm}"
                if key in done_keys:
                    continue
                covar = satellite_proxy_block if arm == "satellite_proxy" else None
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
                    n_fail += 1
                    f_fail.write(json.dumps({
                        "key": key, "gap_id": gap.gap_id, "context_mode": mode, "arm": arm,
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
        "support": support, "arms": arms, "context_modes": context_modes,
        "n_gaps": len(specs), "n_total_calls": n_total,
        "target_sha256": _sha256_file(target_path), "features_sha256": _sha256_file(features_path),
        "gap_pool_sha256": _sha256_file(pool_path),
        "provenance": provenance, "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "command_args": sys.argv[1:],
    }, indent=2))
    print(f"RUN_STATUS: {run_status} ({len(done_keys)}/{n_total}, {n_fail} failures)")
    return 0 if run_status == "RUN_COMPLETE" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", choices=["full_681", "matched_449"], default="full_681")
    parser.add_argument("--arms", type=str, default=",".join(TARGET_ONLY_ARMS))
    parser.add_argument("--context-modes", type=str, default=",".join(tc.PRIMARY_CONTEXT_MODES))
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(TARGET_ONLY_ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}; choose from {TARGET_ONLY_ARMS}")
    context_modes = [m.strip() for m in args.context_modes.split(",") if m.strip()]

    return run_benchmark(args.support, arms, context_modes, args.target, args.features, args.out)


if __name__ == "__main__":
    sys.exit(main())
