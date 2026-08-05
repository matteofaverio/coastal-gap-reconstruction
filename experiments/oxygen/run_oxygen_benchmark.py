"""Bounded oxygen benchmark driver.

Reuses the chlorophyll TS-ICL infrastructure entirely -- checkpoint loading,
provenance verification, resume-state accounting, and configuration-bound
run manifests are the exact same modules
(`experiments.chlorophyll.tsicl_run_state`,
`experiments.chlorophyll.tsicl_run_manifest`), not duplicated. This driver
only adds oxygen-specific orchestration: which gaps, which arms, which
context modes, and the deterministic stratified-subset selection the
compute-bounded live TS-ICL validation requires.

Three modes:

- `--mode frozen`: inspect the released frozen tables only, no computation.
- `--mode classical`: run Model 0 (climatology/persistence/interpolation,
  cheap and deterministic) and the GP gap-edge comparator over the full
  406-gap primary support, or a `--n-gaps` subset. Both are fast enough to
  run to completion well under the 90-minute compute cap (no per-gap model
  refitting cost beyond a single GP fit per gap).
- `--mode tsicl-bounded`: a deterministic stratified subset of at most 20
  primary gaps (several lengths represented), target_only and the best
  arm (`external_physical_plus_currents`) x both context modes by default
  -- at most 60 live TS-ICL calls, validating checkpoint loading, raw mg/L
  I/O, tensor shapes, no-hidden-truth masking, finite/ordered output, and
  repeatability, never deriving a new headline performance number from it
  (see `docs/methodology/oxygen_usage.md`).

Generated outputs go under `build/oxygen/`; `results_public/oxygen/` is
never overwritten by this driver.
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
from coastal_gap_reconstruction.data_loading import load_daily_target
from experiments.chlorophyll import tsicl_run_manifest as rm
from experiments.chlorophyll import tsicl_run_state as rs

from . import benchmark_contract as bc
from . import classical_models as cm
from . import feature_registry as fr
from . import tsicl_models as tm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "oxygen"
CHECKPOINT_EVERY = 10


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pool(support: str = "primary") -> pd.DataFrame:
    pool = pd.read_csv(bc.VALIDATION_GAPS_PATH, parse_dates=["start_date", "end_date"])
    if support == "primary":
        return pool[pool["support_role"] == bc.SUPPORT_ROLE_PRIMARY].reset_index(drop=True)
    if support == "exploratory_extended":
        return pool[pool["support_role"] == bc.SUPPORT_ROLE_EXPLORATORY].reset_index(drop=True)
    if support == "all":
        return pool
    raise ValueError(f"Unknown support {support!r}")


def select_deterministic_tsicl_subset(pool: pd.DataFrame, n_gaps: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic stratified subset of `n_gaps` primary gaps: sample
    proportionally to each length's share of the primary pool, then trim/pad
    to exactly `n_gaps`, sorted for reproducible ordering. Always includes
    every represented length category present in the pool at least once."""
    rng = np.random.default_rng(seed)
    lengths = sorted(pool["gap_length"].unique())
    per_length = max(1, n_gaps // len(lengths))
    rows = []
    for length in lengths:
        sub = pool[pool["gap_length"] == length].sort_values("gap_id")
        take = min(per_length, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        rows.append(sub.loc[sorted(idx)])
    subset = pd.concat(rows, ignore_index=True)
    if len(subset) > n_gaps:
        subset = subset.iloc[:n_gaps]
    return subset.sort_values("gap_id").reset_index(drop=True)


# ── Classical mode ─────────────────────────────────────────────────────────

def run_classical(pool: pd.DataFrame, out_dir: Path, n_gaps: int | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pool if n_gaps is None else pool.head(n_gaps)

    target_df = load_daily_target(bc.DAILY_TARGET_PATH)
    target_df.index = pd.to_datetime(target_df.index)

    t0 = time.time()
    all_preds = []
    for method in ("climatology", "persistence", "linear_interp"):
        preds = cm.run_model0_evaluation(method, candidates, target_df)
        all_preds.append(preds)
    gp_preds, gp_warnings = cm.run_gap_edge_loco_evaluation(candidates, target_df)
    gp_preds = gp_preds.rename(columns={"learner_name": "method_id"})[["gap_id", "date", "method_id", "gap_length", "pred", "true"]]
    all_preds.append(gp_preds)
    elapsed = time.time() - t0

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(out_dir / "classical_day_level.csv", index=False)
    gp_warnings.to_csv(out_dir / "classical_gp_warnings.csv", index=False)

    (out_dir / "classical_run_metadata.json").write_text(json.dumps({
        "n_gaps": len(candidates), "elapsed_s": elapsed, "methods": ["climatology", "persistence", "linear_interp",
                                                                       "gp_matern_time_only_exploratory"],
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2))
    print(f"classical run: {len(candidates)} gaps, {elapsed:.1f}s")
    return 0


# ── TS-ICL bounded mode ────────────────────────────────────────────────────

def _arm_registry_config_sha256(arms: list[str]) -> str:
    """Hash of the exact selected columns per requested arm -- changing any
    arm's column list (or which arms are requested) changes this hash."""
    config = {arm: tm.ALL_ARM_COLUMNS[arm] for arm in sorted(arms)}
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gap_ids_sha256(gap_ids) -> str:
    """Hash of the exact selected gap-ID list's canonical (sorted) serialization."""
    canonical = "|".join(sorted(gap_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_tsicl_bounded(
    pool: pd.DataFrame, arms: list[str], context_modes: list[str], n_gaps: int, out_dir: Path,
    max_calls: int = 60,
) -> int:
    # Compute the actual selected support FIRST -- the deterministic stratified
    # selection can return fewer than the requested n_gaps (e.g. capped by the
    # smallest per-length pool), so the run's identity/support label must
    # reflect what was actually selected, not what was requested.
    subset = select_deterministic_tsicl_subset(pool, n_gaps)
    actual_n_gaps = len(subset)
    n_total = actual_n_gaps * len(arms) * len(context_modes)
    if n_total > max_calls:
        print(
            f"FATAL: requested subset would need {n_total} calls, exceeding the "
            f"max_calls={max_calls} compute-boundary cap. Reduce --n-gaps/--arms/--context-modes."
        )
        return 1

    try:
        model, provenance = th.load_tsicl_strict()
    except th.TSICLError as exc:
        print(f"FATAL: could not load TS-ICL: {exc}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "RUN_STATUS").write_text(f"RUN_FAILED: {exc}\n")
        return 1
    print(f"loaded TS-ICL: {provenance}", flush=True)

    # Exact-input identity: every file actually read by this run is hashed,
    # plus the exact selected gap-ID list and the exact per-arm column
    # configuration -- a resumed run must invalidate (RunConfigMismatchError)
    # if any of these changed, not just the target table. No placeholder
    # strings (a prior version used features_sha256="n/a_multiple_arms",
    # which protected nothing).
    target_sha256 = _sha256_file(bc.DAILY_TARGET_PATH)
    base_features_sha256 = _sha256_file(bc.CHLOROPHYLL_BASE_FEATURES_PATH)
    extension_sha256 = _sha256_file(bc.CURRENT_KINEMATIC_EXTENSION_PATH)
    local_btg_sha256 = _sha256_file(bc.LOCAL_BTG_DIAGNOSTIC_FEATURES_PATH)
    gap_pool_sha256 = _sha256_file(bc.VALIDATION_GAPS_PATH)
    identity = rm.build_run_identity(
        driver="run_oxygen_benchmark_tsicl_bounded", support=f"bounded_{actual_n_gaps}_gaps", arms=arms,
        context_modes=context_modes,
        placebo_config={
            "local_btg_diagnostic_sha256": local_btg_sha256,
            "selected_gap_ids_sha256": _gap_ids_sha256(subset["gap_id"]),
            "arm_registry_config_sha256": _arm_registry_config_sha256(arms),
        },
        target_sha256=target_sha256, features_sha256=base_features_sha256,
        extension_sha256=extension_sha256, gap_pool_sha256=gap_pool_sha256,
        provenance=provenance, target_transform=bc.TARGET_TRANSFORM,
        context_window_settings={"window_days": tm.WINDOW_DAYS, "max_context_length": 4096},
        quantile_levels=th.DEFAULT_QUANTILE_LEVELS,
    )
    try:
        rm.write_or_validate_manifest(out_dir, identity)
    except rm.RunConfigMismatchError as exc:
        print(f"FATAL: {exc}")
        return 1

    dates, target_mgL = tm.load_target_series()
    date_index = pd.to_datetime(dates)
    plus_currents = fr.get_feature_arm("external_physical_plus_currents").reindex(date_index)
    local_btg = fr.get_feature_arm("local_btg_temp_pressure_diagnostic").reindex(date_index)
    features_df = plus_currents.join(
        local_btg[["btg_water_temp_daily_mean", "btg_pressure_daily_mean"]], how="left",
    )

    pred_path = out_dir / "predictions.jsonl"
    fail_path = out_dir / "failures.jsonl"
    expected_keys = [
        f"{gap_id}|{mode}|{arm}"
        for gap_id in subset["gap_id"] for mode in context_modes for arm in arms
    ]
    state = rs.load_run_state(pred_path, fail_path)
    outstanding = state.outstanding(expected_keys, max_attempts=None)
    if state.successful_keys:
        print(f"resuming: {len(state.successful_keys & set(expected_keys))} already successful, "
              f"{len(outstanding)} outstanding", flush=True)

    f_pred = open(pred_path, "a")
    f_fail = open(fail_path, "a")

    def _json_default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(str(type(o)))

    gap_by_id = {row["gap_id"]: th.GapSpec(gap_id=row["gap_id"], start_date=str(row["start_date"].date()),
                                            end_date=str(row["end_date"].date()), length=int(row["gap_length"]))
                 for _, row in subset.iterrows()}

    t_start = time.time()
    n_fail_this_run = 0
    for i, key in enumerate(outstanding):
        gap_id, mode, arm = key.split("|")
        gap = gap_by_id[gap_id]
        covar = tm.build_covariate_block(arm, features_df)
        try:
            result = tm.run_oxygen_gap_inference(model, dates, target_mgL, gap, context_mode=mode,
                                                  covariate_array=covar, strict=True)
            # Field names (pred_log10/true_log10/quantiles_log10) are inherited
            # from the shared tsicl_run_state validator's chlorophyll origin --
            # values here are raw oxygen_mean_mgL, never log10 (see
            # tsicl_models.py module docstring). Kept as-is rather than
            # duplicating/modifying the shared validator for a naming
            # preference alone.
            row = {
                "key": key, "gap_id": gap.gap_id, "context_mode": mode, "arm": arm,
                "gap_length": gap.length, "date": [str(d) for d in result["dates"]],
                "pred_log10": [float(v) for v in result["pred_log10"]],
                "true_log10": [float(v) for v in result["true_log10"]],
                "quantile_levels": result["quantile_levels"],
                "quantiles_log10": [[float(v) for v in q] for q in result["quantiles_log10"]],
            }
            f_pred.write(json.dumps(row, default=_json_default) + "\n")
        except th.TSICLError as exc:
            n_fail_this_run += 1
            f_fail.write(json.dumps({"key": key, "gap_id": gap.gap_id, "context_mode": mode, "arm": arm,
                                      "error_type": type(exc).__name__, "error": str(exc)}) + "\n")
            print(f"[FAILED] {key}: {exc}", flush=True)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            f_pred.flush()
            f_fail.flush()

    f_pred.flush()
    f_pred.close()
    f_fail.flush()
    f_fail.close()

    final_state = rs.load_run_state(pred_path, fail_path)
    run_status, detail = rs.classify_run_status(final_state, expected_keys)
    elapsed = time.time() - t_start
    (out_dir / "RUN_STATUS").write_text(f"{run_status}: {json.dumps(detail)}\n")
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "support": f"bounded_{actual_n_gaps}_gaps", "arms": arms, "context_modes": context_modes,
        "requested_max_n_gaps": n_gaps, "actual_selected_n_gaps": actual_n_gaps,
        "n_expected_calls": len(expected_keys), "elapsed_s": elapsed,
        "run_status_detail": detail, "provenance": provenance,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(), "command_args": sys.argv[1:],
    }, indent=2))
    print(f"RUN_STATUS: {run_status} ({json.dumps(detail)}), elapsed={elapsed:.1f}s")
    return 0 if run_status == "RUN_COMPLETE" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["frozen", "classical", "tsicl-bounded"], default="frozen")
    parser.add_argument("--support", choices=["primary", "exploratory_extended", "all"], default="primary")
    parser.add_argument("--n-gaps", type=int, default=None, help="Classical mode: subset size (default: full support).")
    parser.add_argument("--tsicl-n-gaps", type=int, default=15, help="TS-ICL bounded mode: stratified subset size (<=20).")
    parser.add_argument("--arms", type=str, default="target_only,external_physical_plus_currents")
    parser.add_argument("--context-modes", type=str, default="full_series,edge_balanced")
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if args.mode == "frozen":
        print("Frozen-result inspection only -- see results_public/oxygen/. No computation performed.")
        for path in (bc.BENCHMARK_BY_LENGTH_PATH, bc.PAIRED_DELTAS_VS_TSICL_PATH,
                     bc.TAIL_PERSISTENCE_METRICS_PATH, bc.TAIL_QUANTILE_BAND_METRICS_PATH):
            print(f"  {path.relative_to(REPO_ROOT)}: {len(pd.read_csv(path))} rows")
        return 0

    pool = load_pool(args.support)
    if args.mode == "classical":
        return run_classical(pool, args.out / "classical", args.n_gaps)

    if args.mode == "tsicl-bounded":
        if args.tsicl_n_gaps > 20:
            parser.error("--tsicl-n-gaps must be <= 20 (compute boundary)")
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = set(arms) - set(tm.ALL_ARM_COLUMNS)
        if unknown:
            parser.error(f"unknown arms: {sorted(unknown)}")
        context_modes = [m.strip() for m in args.context_modes.split(",") if m.strip()]
        return run_tsicl_bounded(pool, arms, context_modes, args.tsicl_n_gaps, args.out / "tsicl_bounded",
                                  max_calls=args.max_calls)

    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
