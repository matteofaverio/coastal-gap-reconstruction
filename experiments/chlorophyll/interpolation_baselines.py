"""The standalone linear-interpolation baseline for chlorophyll (`canonical_interpolation`).

**Two distinct linear-interpolation formulas exist in this package and must never be
conflated:**

1. **Standalone baseline** (this module; `canonical_interpolation` in
   `benchmark_contract.METHODS`): interpolates in **log10(chl_mean) space** between the
   two bracketing eligible observations, then back-transforms to physical units. This is
   an exact port of the private project's `scripts/stage_b_canonical_interpolation.py`,
   which is the source of the frozen `canonical_interpolation` row in the released
   `chlorophyll_matched_support_method_metrics.csv` (verified against the private
   per-day predictions this table was built from, not merely against the aggregate MAE).

2. **Gap-edge residual anchor** (`gap_edge_models.compute_interp`): interpolates in
   **physical (mg/m^3) space** between the same two bracketing observations, then takes
   log10 of the interpolated physical value. This is the anchor the gap-edge residual
   model (`tier_ch_deployed`) predicts a residual against -- it is an exact port of the
   private project's `features/tier_c_gap_edge.py::compute_interp`, used there and only
   there, never as a standalone baseline.

These two formulas agree only at the two bracketing observations themselves and diverge
at every interior day (log10-space interpolation is a straight line in log space, i.e. a
geometric-mean-like curve in physical space; physical-space interpolation is a straight
line in mg/m^3, i.e. a convex curve in log space). Using one formula where the other is
required silently changes the scored predictions of both methods it feeds -- this file
exists specifically so that substitution cannot happen by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _config
from . import gap_edge_models as gem

__all__ = ["standalone_log10_interpolation_predictions"]

TARGET_COL = _config.TARGET_COL
LOG10_FLOOR = 1e-4


def _log10(v: float) -> float:
    return float(np.log10(v)) if (v == v and v > LOG10_FLOOR) else np.nan


def standalone_log10_interpolation_predictions(
    candidates: pd.DataFrame, target_df: pd.DataFrame,
) -> pd.DataFrame:
    """The standalone `canonical_interpolation` baseline.

    For each candidate gap, finds the nearest eligible observation strictly
    before the gap and the nearest strictly after it, linearly interpolates
    **log10(chl_mean)** between them (not the physical value -- see module
    docstring), and back-transforms to physical units for the `pred` column.

    A gap with no bracketing observation on either side gets `NaN` predictions
    for every hidden day (kept as rows, not dropped, matching the private
    source's behavior so `n_rows` totals are comparable across methods).

    Returns one row per (gap_id, date) with `pred_log10`/`pred`/`true`/`gap_length`.
    """
    obs = gem.observed_series(target_df)
    rows: list[dict] = []
    for _, row in candidates.iterrows():
        gap_id = row["gap_id"]
        start = pd.Timestamp(row["start_date"])
        L = int(row["gap_length"])
        end = start + pd.Timedelta(days=L - 1)
        hidden = pd.date_range(start, end, freq="D")

        pre = obs[obs.index < start]
        post = obs[obs.index > end]
        bracket_ok = not pre.empty and not post.empty
        if bracket_ok:
            t0, y0_raw = pre.index[-1], float(pre.iloc[-1])
            t1, y1_raw = post.index[0], float(post.iloc[0])
            y0, y1 = _log10(y0_raw), _log10(y1_raw)
            dt = (t1 - t0).days

        for d in hidden:
            true_val = target_df.loc[d, TARGET_COL] if d in target_df.index else np.nan
            true_val = float(true_val) if true_val == true_val else np.nan

            pred_log10 = np.nan
            if bracket_ok and dt > 0:
                frac = (d - t0).days / dt
                pred_log10 = y0 + frac * (y1 - y0)
            elif bracket_ok and dt == 0:
                pred_log10 = (y0 + y1) / 2.0
            pred = float(10.0**pred_log10) if pred_log10 == pred_log10 else np.nan

            rows.append({
                "gap_id": gap_id, "date": d, "gap_length": L,
                "pred_log10": pred_log10, "pred": pred, "true": true_val,
            })
    return pd.DataFrame(rows)
