"""The engineered hybrid reconstruction pipeline (released public name;
private project internal name: Sprint 8 "Rule D").

Ported from `sprint8_final_reconstruction.py`, the only private script that
assembles the length-routed method assignment and applies it across gap
lengths. This is **a deterministic method-assignment rule**, not a distinct
fitted model: for a given gap it dispatches to one of three already-defined
component methods based on gap length alone, then returns that component's
own prediction unmodified.

Assignment rule (`ASSIGNMENT_RULE`, "Rule D" in the private project, chosen
over the "Rule C" sensitivity variant per the private project's
`docs/status/CANONICAL_RESULTS.md` Sprint 7I-R reconciliation -- Rule D had
the lower weighted MAE under 4 of 5 weighting schemes tested there):

    L=1-3    -> Gaussian process (`gaussian_process.run_gp_on_gap`)
    L=4-29   -> Kalman local-level smoother (`probabilistic_models.run_kalman_on_gap`)
    L>=30    -> Gap-edge residual model (`gap_edge_models.run_loco_evaluation`)

**This assignment was made on evidence from the earlier 450-gap pool, not
re-derived from the 449-gap matched support used elsewhere in this package**
(per `docs/status/CANONICAL_RESULTS.md`, Sprint 7I's crossover analysis used
L=1-30 plus several intermediate lengths (2,4,5,6,12,13,21) not all of which
are part of the 5-length matched support) -- it is reproduced here exactly as
released, not re-optimized against the matched support.

**Known caveat, stated explicitly because it materially affects the L=4-29
segment**: the Kalman component's fit is degenerate on this series (see
`probabilistic_models`'s module docstring) -- its predictions in that segment
are numerically equivalent to linear interpolation in the large majority of
cases, so this hybrid's L=4-29 segment does not currently demonstrate a
"Kalman smoothing" benefit distinct from interpolation, despite the
historical rationale for the assignment. This module does not silently
repair that; `assign_method` and `reconstruct_gap` surface it via
`probabilistic_models.kalman_degeneracy_report`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import gaussian_process as gp

from . import _config
from . import gap_edge_models as gem
from . import probabilistic_models as pm
from . import tabular_models as tm

__all__ = [
    "ASSIGNMENT_RULE",
    "assign_method",
    "reconstruct_gap",
    "run_engineered_hybrid",
]

# Length-routed method assignment ("Rule D"). Boundaries are inclusive.
ASSIGNMENT_RULE: list[tuple[int, int, str]] = [
    (1, 3, "gaussian_process"),
    (4, 29, "kalman_local_level"),
    (30, 10_000, "gap_edge_residual"),
]


def assign_method(gap_length: int) -> str:
    """Return which component method `ASSIGNMENT_RULE` assigns to `gap_length`."""
    for lo, hi, method in ASSIGNMENT_RULE:
        if lo <= gap_length <= hi:
            return method
    raise ValueError(f"gap_length={gap_length} not covered by ASSIGNMENT_RULE")


def reconstruct_gap(
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    kalman_sigma_q: float,
    kalman_sigma_r: float,
    target_col: str = _config.TARGET_COL,
    eligible_col: str = _config.ELIGIBLE_COL,
    gap_edge_context_pool: pd.DataFrame | None = None,
) -> dict:
    """Reconstruct one gap using whichever component `assign_method` selects.

    `gap_edge_context_pool` supplies leave-one-gap-out training context for
    the `gap_edge_residual` branch (L>=30): the gap-edge model needs at
    least ~30 admissible training rows drawn from *other* gaps' hidden days,
    which a single gap can never supply on its own. If omitted, only this
    gap's own (empty) context is available and the gap-edge branch will
    reliably fail with "insufficient_training_rows" -- always pass the full
    candidate pool (or a representative subset of it) when reconstructing
    L>=30 gaps. `run_engineered_hybrid` does this automatically.

    `kalman_sigma_q`/`kalman_sigma_r` must come from
    `probabilistic_models.estimate_kalman_params`, fit once on the full
    observed series (not per-gap -- matching the released procedure) and
    passed in explicitly so this function never re-fits Kalman parameters
    using data from inside the gap being reconstructed.

    Returns a dict with `method`, `dates`, `pred_log10`, `pred`, and (for the
    Kalman branch only) `kalman_degeneracy`.
    """
    method = assign_method(gap_length)
    hidden = pd.date_range(start_date, periods=gap_length, freq="D")

    if method == "gaussian_process":
        y_log = np.log10(target_df[target_col].clip(lower=1e-12))
        y_log = y_log.where(target_df[target_col] > 1e-4)
        ctx_df = pd.DataFrame({
            target_col: y_log,
            eligible_col: target_df[eligible_col],
            "doy_sin": np.sin(2 * np.pi * target_df.index.dayofyear / 365.25),
            "doy_cos": np.cos(2 * np.pi * target_df.index.dayofyear / 365.25),
        }, index=target_df.index)
        result = gp.run_gp_on_gap(
            ctx_df, start_date, gap_length, value_col=target_col, eligible_col=eligible_col,
            random_state=_config.RANDOM_SEED,
        )
        if result is None:
            return {"method": method, "dates": hidden, "pred_log10": {}, "pred": {}}
        pred_log10 = dict(zip(result["dates"], result["pred"]))
        return {
            "method": method, "dates": result["dates"], "pred_log10": pred_log10,
            "pred": {d: float(10.0**v) for d, v in pred_log10.items()},
        }

    if method == "kalman_local_level":
        obs_log = np.log10(target_df[target_col].where(target_df[target_col] > 1e-4))
        result = pm.run_kalman_on_gap(
            target_df.index, obs_log.to_numpy(), start_date, gap_length, kalman_sigma_q, kalman_sigma_r,
        )
        if result is None:
            return {"method": method, "dates": hidden, "pred_log10": {}, "pred": {}}
        pred_log10 = dict(zip(result["dates"], result["pred"]))
        return {
            "method": method, "dates": result["dates"], "pred_log10": pred_log10,
            "pred": {d: float(10.0**v) for d, v in pred_log10.items()},
            "kalman_degeneracy": pm.kalman_degeneracy_report(kalman_sigma_q, kalman_sigma_r),
        }

    # gap_edge_residual: LOCO evaluation, scoring only this gap but drawing
    # admissible training rows from the full context pool (a single gap can
    # never supply its own >=30 required training rows).
    target_gap_id = f"engineered_hybrid_target_{start_date:%Y%m%d}"
    target_row = pd.DataFrame([{
        "gap_id": target_gap_id, "gap_length": gap_length,
        "start_date": start_date, "end_date": start_date + pd.Timedelta(days=gap_length - 1),
    }])
    context = target_row if gap_edge_context_pool is None else pd.concat(
        [gap_edge_context_pool, target_row], ignore_index=True
    )
    external_cols = tm.load_arm4_numeric_columns(features_df)
    preds, _ = gem.run_loco_evaluation(
        context, target_df, features_df, external_cols, score_gap_ids=[target_gap_id]
    )
    pred_log10 = dict(zip(preds["date"], preds["pred_log10"])) if not preds.empty else {}
    pred = dict(zip(preds["date"], preds["pred"])) if not preds.empty else {}
    return {"method": method, "dates": hidden, "pred_log10": pred_log10, "pred": pred}


def run_engineered_hybrid(
    candidates: pd.DataFrame,
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: str = _config.TARGET_COL,
    eligible_col: str = _config.ELIGIBLE_COL,
) -> tuple[pd.DataFrame, dict]:
    """Run the engineered hybrid over every row of `candidates`.

    Kalman parameters are estimated once, on the full observed series, before
    the per-gap loop -- matching the released procedure (a single global
    `(sigma_q, sigma_r)` pair is used for every L=4-29 gap, not refit per gap).

    Returns `(predictions_df, kalman_params)`.
    """
    obs_log = np.log10(target_df[target_col].where(target_df[target_col] > 1e-4))
    sigma_q, sigma_r = pm.estimate_kalman_params(obs_log.dropna().to_numpy())

    rows: list[dict] = []
    for _, row in candidates.iterrows():
        gap_id = str(row["gap_id"])
        gap_length = int(row["gap_length"])
        start = pd.Timestamp(row["start_date"])
        result = reconstruct_gap(
            target_df, features_df, start, gap_length, sigma_q, sigma_r,
            target_col=target_col, eligible_col=eligible_col,
            gap_edge_context_pool=candidates,
        )
        for d, pred_log in result["pred_log10"].items():
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            rows.append({
                "gap_id": gap_id, "date": d, "gap_length": gap_length,
                "assigned_method": result["method"],
                "pred_log10": pred_log, "pred": result["pred"][d],
                "true": float(true_val) if true_val == true_val else np.nan,
            })

    predictions_df = pd.DataFrame(rows)
    return predictions_df, {"sigma_q": sigma_q, "sigma_r": sigma_r}
