"""Run the classical/probabilistic/gap-edge/engineered-hybrid benchmark on the
449-gap matched support (`benchmark_contract.MATCHED_SUPPORT_*`).

Usage:
    python -m experiments.chlorophyll.run_classical_benchmark \\
        --target data_public/chlorophyll/chlorophyll_daily_target.csv \\
        --features data_public/chlorophyll/chlorophyll_predictor_features_curated.csv \\
        --out build/chlorophyll/classical_benchmark \\
        [--methods gp_m1,ext_tabular_extratrees,...] [--gap-lengths 1,3] [--verify]

All input/output paths are explicit CLI arguments (no implicit repo-relative
defaults baked into the driver itself, beyond argparse's own defaults, which
point at the already-public files) so this can run against a different
checkout or a build directory outside version control.

Default output directory is a gitignored `build/` path -- this driver never
overwrites the frozen `results_public/` tables by default. Progress is
checkpointed per method: a completed method's predictions are written
immediately and skipped on a re-run unless `--force` is given, so an
interrupted run can resume instead of restarting from zero.

`--verify` compares freshly generated predictions against the frozen
`results_public/chlorophyll/chlorophyll_matched_support_method_metrics.csv`
and prints a per-method exact/tolerance/not-reproduced classification instead
of running the benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import gaussian_process as gp

from . import benchmark_contract as bc
from . import engineered_hybrid as eh
from . import gap_edge_models as gem
from . import tabular_models as tm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "chlorophyll" / "classical_benchmark"

ALL_METHODS = [
    "canonical_interpolation", "gp_m1", "ext_tabular_extratrees",
    "ext_tabular_hgb", "tier_ch_deployed", "engineered_hybrid",
]


def load_inputs(target_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    features_df = pd.read_csv(features_path, parse_dates=["date"]).set_index("date").sort_index()
    return target_df, features_df


def _linear_interpolation_predictions(candidates: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    obs = gem.observed_series(target_df)
    rows = []
    for _, row in candidates.iterrows():
        start = pd.Timestamp(row["start_date"])
        L = int(row["gap_length"])
        end = start + pd.Timedelta(days=L - 1)
        pre = obs[obs.index < start]
        post = obs[obs.index > end]
        if pre.empty or post.empty:
            continue
        pre_last_date, pre_last = pre.index[-1], float(pre.iloc[-1])
        post_first_date, post_first = post.index[0], float(post.iloc[0])
        for d in pd.date_range(start, end, freq="D"):
            v, log_v = gem.compute_interp(pre_last_date, pre_last, post_first_date, post_first, d)
            true_val = target_df.loc[d, bc.TARGET_SPEC.target_col] if d in target_df.index else np.nan
            rows.append({
                "gap_id": row["gap_id"], "date": d, "gap_length": L,
                "pred_log10": log_v, "pred": v,
                "true": float(true_val) if true_val == true_val else np.nan,
            })
    return pd.DataFrame(rows)


def _gp_predictions(candidates: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    target_col = bc.TARGET_SPEC.target_col
    eligible_col = bc.TARGET_SPEC.eligible_col
    y_log = np.log10(target_df[target_col].where(target_df[target_col] > 1e-4))
    ctx_df = pd.DataFrame({
        target_col: y_log,
        eligible_col: target_df[eligible_col],
        "doy_sin": np.sin(2 * np.pi * target_df.index.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * target_df.index.dayofyear / 365.25),
    }, index=target_df.index)

    rows = []
    for _, row in candidates.iterrows():
        start = pd.Timestamp(row["start_date"])
        L = int(row["gap_length"])
        result = gp.run_gp_on_gap(
            ctx_df, start, L, value_col=target_col, eligible_col=eligible_col,
            random_state=bc.RANDOM_SEED,
        )
        if result is None:
            continue
        for d, pred_log in zip(result["dates"], result["pred"]):
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            rows.append({
                "gap_id": row["gap_id"], "date": d, "gap_length": L,
                "pred_log10": float(pred_log), "pred": float(10.0**pred_log),
                "true": float(true_val) if true_val == true_val else np.nan,
            })
    return pd.DataFrame(rows)


def run_method(
    method_id: str, candidates: pd.DataFrame, target_df: pd.DataFrame, features_df: pd.DataFrame,
) -> pd.DataFrame:
    if method_id == "canonical_interpolation":
        return _linear_interpolation_predictions(candidates, target_df)
    if method_id == "gp_m1":
        return _gp_predictions(candidates, target_df)
    if method_id in ("ext_tabular_extratrees", "ext_tabular_hgb"):
        cols = tm.load_arm4_numeric_columns(features_df)
        preds, _warns = tm.run_loco_evaluation(method_id, candidates, target_df, features_df, cols)
        return preds
    if method_id == "tier_ch_deployed":
        preds, _warns = gem.run_loco_evaluation(candidates, target_df, features_df)
        return preds
    if method_id == "engineered_hybrid":
        preds, _kalman = eh.run_engineered_hybrid(candidates, target_df, features_df)
        return preds
    raise ValueError(f"unknown method_id {method_id!r}")


def score_predictions(preds: pd.DataFrame) -> dict:
    """Day-weighted MAE/RMSE/bias on the log10 scale, matching the released
    metric definitions exactly (see `benchmark_contract.PRIMARY_METRIC`)."""
    valid = preds.dropna(subset=["pred_log10", "true"])
    if valid.empty:
        return {"n_gaps": 0, "n_rows": 0, "mae_day_weighted": np.nan, "rmse": np.nan, "bias_mean": np.nan}
    true_log = np.log10(valid["true"].clip(lower=1e-4))
    err = valid["pred_log10"] - true_log
    return {
        "n_gaps": int(valid["gap_id"].nunique()),
        "n_rows": int(len(valid)),
        "mae_day_weighted": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias_mean": float(err.mean()),
    }


def run_benchmark(
    methods: list[str], gap_lengths: list[int], target_path: Path, features_path: Path,
    out_dir: Path, force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df, features_df = load_inputs(target_path, features_path)

    pool = bc.load_matched_support_pool()
    candidates = pool[pool["gap_length"].isin(gap_lengths)].reset_index(drop=True)
    if candidates.empty:
        raise ValueError(f"no matched-support gaps for gap_lengths={gap_lengths}")

    summary_rows: list[dict] = []
    failures: list[dict] = []
    versions = {"note": "see run metadata for the current environment's package versions"}

    for method_id in methods:
        pred_path = out_dir / f"predictions_{method_id}.csv"
        if pred_path.exists() and not force:
            preds = pd.read_csv(pred_path, parse_dates=["date"])
            print(f"[skip, cached] {method_id}: {pred_path}")
        else:
            t0 = time.perf_counter()
            try:
                preds = run_method(method_id, candidates, target_df, features_df)
            except Exception as exc:  # noqa: BLE001
                failures.append({"method_id": method_id, "error": str(exc)})
                print(f"[FAILED] {method_id}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            preds.to_csv(pred_path, index=False)
            print(f"[done] {method_id}: {len(preds)} rows in {elapsed:.1f}s -> {pred_path}")

        # Explicitly record whether every requested gap produced at least one row.
        scored_gaps = set(preds["gap_id"]) if not preds.empty else set()
        missing_gaps = sorted(set(candidates["gap_id"]) - scored_gaps)
        if missing_gaps:
            failures.append({
                "method_id": method_id,
                "error": f"{len(missing_gaps)} gaps produced no prediction rows",
                "missing_gap_ids": missing_gaps,
            })

        metrics = score_predictions(preds)
        summary_rows.append({"method_id": method_id, **metrics})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "gap_lengths": gap_lengths,
        "methods": methods,
        "n_candidates": int(len(candidates)),
        "seed": bc.RANDOM_SEED,
        "versions": versions,
    }, indent=2))
    (out_dir / "COMPLETE").write_text("benchmark run complete\n")
    print(f"\nSummary written to {out_dir / 'summary_metrics.csv'}")
    if failures:
        print(f"WARNING: {len(failures)} method(s) had failures/missing gaps -- see failures.json")


def verify_against_released(out_dir: Path) -> int:
    """Compare `out_dir/summary_metrics.csv` against the frozen released
    matched-support metrics table; print a per-method classification.

    Returns the number of methods classified `not_reproduced` (0 = clean).
    """
    summary_path = out_dir / "summary_metrics.csv"
    if not summary_path.exists():
        print(f"No generated summary at {summary_path} -- run the benchmark first.")
        return 1

    generated = pd.read_csv(summary_path).set_index("method_id")
    released = pd.read_csv(bc.MATCHED_SUPPORT_METRICS_PATH)
    released = released[released["support"] == "matched_449"].set_index("method_id")

    n_not_reproduced = 0
    print(f"{'method':<26} {'released_mae':>13} {'generated_mae':>14} {'abs_diff':>10}  classification")
    for method_id in generated.index:
        if method_id not in released.index:
            print(f"{method_id:<26} {'(not in released table)':>13}")
            continue
        rel_mae = float(released.loc[method_id, "mae_day_weighted"])
        gen_mae = float(generated.loc[method_id, "mae_day_weighted"])
        diff = abs(rel_mae - gen_mae)
        if diff < 1e-9:
            cls = "bit_identical"
        elif diff < 1e-6:
            cls = "numerically_exact"
        elif diff < 5e-3:
            cls = "within_tolerance"
        else:
            cls = "not_reproduced"
            n_not_reproduced += 1
        print(f"{method_id:<26} {rel_mae:>13.6f} {gen_mae:>14.6f} {diff:>10.6f}  {cls}")
    return n_not_reproduced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    parser.add_argument(
        "--gap-lengths", type=str, default=",".join(str(x) for x in bc.MATCHED_SUPPORT_GAP_LENGTHS)
    )
    parser.add_argument("--force", action="store_true", help="Re-run methods even if cached predictions exist.")
    parser.add_argument("--verify", action="store_true", help="Compare an existing run against the frozen released table.")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}; choose from {ALL_METHODS}")
    gap_lengths = [int(x) for x in args.gap_lengths.split(",") if x.strip()]

    if args.verify:
        n_bad = verify_against_released(args.out)
        return 1 if n_bad else 0

    run_benchmark(methods, gap_lengths, args.target, args.features, args.out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
