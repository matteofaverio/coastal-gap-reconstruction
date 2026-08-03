"""Gap-edge residual reconstruction for chlorophyll (the released
"Gap-edge residual model" / private "Tier C-H", `tier_ch_deployed` in
`benchmark_contract.METHODS`).

Ported from the private project's `features/tier_c_gap_edge.py` (feature
construction, published here near-verbatim -- it is already self-contained
and target/candidate-only, no private paths) and the fitting/leakage-control
core of `models/tier_c_gap_edge_eval.py` (731 lines) merged with
`models/tier_c_7c_extended_eval.py` (1117 lines, the L=10/21/45/60
extension) -- the reporting, figure, caching, and CLI code in both private
files is dropped, not ported.

**Retrospective only, not forecast-safe.** Every feature family beyond `pre`
(the `post`, `edge`, and `interp` groups) is built from post-gap in-situ
chlorophyll, so this model requires observations on *both* sides of a gap.
It is a plausible historical reconstruction method, never a deployable
forward forecaster. `META_COLS`/`PRE_COLS` alone (the "pre-only" mode) would
be forecast-safe, but that is not the arm this module fits by default --
`tier_a.py`'s external-tabular arm4 is the project's forecast-safe external
model; this module is deliberately the retrospective specialist.

**Canonical arm**: `residual_log = true_log - interpolation_log`, predicted
by an `ExtraTreesRegressor` over `external + meta + pre + post + edge +
interp` columns, then added back to the linear-interpolation baseline to
recover the final log10 prediction (`tier_c_residual_over_interp` in the
private project, the arm cited in `docs/status/CANONICAL_RESULTS.md` and
scored in `results_public/chlorophyll/
chlorophyll_matched_support_method_metrics.csv` as `tier_ch_deployed`).
Predicting the *residual* rather than the raw log value, then adding back
the (leakage-safe) interpolation baseline, is why this arm needs the
`interp` feature group even though `linear_interp_log_chl` is trivially
derivable -- it is the regression target's anchor, not a spurious feature.

**Leakage control**: leave-one-gap-out (LOCO) with dependency-window
exclusion (`admissible_train_mask`) -- a training row is only usable for a
held-out gap if its own feature-dependency window (`dep_min_hind`/
`dep_max_hind`, computed per row in `build_tier_c_feature_table`) does not
overlap the held-out gap's hidden span. This is stricter than excluding only
the held-out gap's own rows: a *different* gap's row can still leak the
held-out gap's hidden values if its pre/post windows reach into them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from . import _config
from . import tabular_models as tm

__all__ = [
    "TARGET_COL", "ELIGIBLE_COL", "LOG10_FLOOR", "SUSTAINED_MEAN_THRESHOLD",
    "META_COLS", "PRE_COLS", "POST_COLS", "EDGE_COLS", "INTERP_COLS",
    "TARGET_EVAL_COLS", "HINDCAST_ONLY_COLS",
    "observed_series", "compute_pre_features", "compute_post_features",
    "compute_edge_features", "compute_interp", "build_tier_c_feature_table",
    "build_feature_registry", "arm_feature_columns",
    "build_extratrees", "admissible_train_mask", "assemble_design",
    "run_loco_evaluation",
]

TARGET_COL = _config.TARGET_COL
ELIGIBLE_COL = _config.ELIGIBLE_COL
LOG10_FLOOR = 1e-4
SUSTAINED_MEAN_THRESHOLD = _config.SUSTAINED_MEAN_THRESHOLD
RANDOM_SEED = _config.RANDOM_SEED

SEASON_MAP = _config.SEASON_MAP

# ── Feature column groups (single source of truth for the arm registry and
# leakage tests) ──────────────────────────────────────────────────────────
META_COLS = [
    "gap_length", "day_index_within_gap", "gap_position_fraction",
    "distance_to_left_edge", "distance_to_right_edge",
]
PRE_COLS = [
    "pre_last_chl", "pre_last_log_chl",
    "pre_mean_3d", "pre_mean_7d", "pre_mean_14d",
    "pre_median_7d", "pre_std_7d",
    "pre_slope_3d", "pre_slope_7d",
    "pre_delta_1d", "days_since_last_valid",
]
POST_COLS = [
    "post_first_chl", "post_first_log_chl",
    "post_mean_3d", "post_mean_7d",
    "post_median_7d", "post_std_7d",
    "post_slope_3d", "post_slope_7d",
    "post_delta_1d", "days_to_next_valid",
]
EDGE_COLS = [
    "edge_mean_chl", "edge_log_mean", "edge_delta_chl",
    "edge_delta_log", "edge_slope_chl", "edge_slope_log",
]
INTERP_COLS = ["linear_interp_chl", "linear_interp_log_chl"]

TARGET_EVAL_COLS = [
    "true_chl", "true_log",
    "residual_log_true_minus_interp_log", "residual_raw_true_minus_interp",
]
META_BOOKKEEPING_COLS = [
    "gap_id", "date", "season", "year",
    "is_high_chl_event", "is_sustained_event",
    "dep_min_pre", "dep_max_pre", "dep_min_hind", "dep_max_hind",
    "pre_available", "post_available", "interp_available",
]
HINDCAST_ONLY_COLS = POST_COLS + EDGE_COLS + INTERP_COLS

# The canonical arm's feature groups.
CANONICAL_GROUPS = ["external", "meta", "pre", "post", "edge", "interp"]


def _log10(v: float) -> float:
    return float(np.log10(v)) if (v is not None and v == v and v > LOG10_FLOOR) else np.nan


def _slope_per_day(dates: list[pd.Timestamp], values: list[float]) -> float:
    if len(values) < 2:
        return np.nan
    x = np.array([(d - dates[0]).days for d in dates], dtype=float)
    y = np.array(values, dtype=float)
    if np.std(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def observed_series(target_df: pd.DataFrame) -> pd.Series:
    """Eligible, positive, non-NaN chl_mean indexed by date (sorted)."""
    elig = target_df[ELIGIBLE_COL].fillna(False).astype(bool)
    has = target_df[TARGET_COL].notna() & (target_df[TARGET_COL] > LOG10_FLOOR)
    return target_df.loc[elig & has, TARGET_COL].astype(float).sort_index()


def compute_pre_features(series: pd.Series, as_of: pd.Timestamp) -> tuple[dict, list]:
    """Pre-edge features as-of a date, from observations strictly before it."""
    out: dict = {k: np.nan for k in PRE_COLS}
    pre = series[series.index < as_of]
    if len(pre) == 0:
        return out, []
    src: list[pd.Timestamp] = []

    last_date = pre.index[-1]
    last_val = float(pre.iloc[-1])
    out["pre_last_chl"] = last_val
    out["pre_last_log_chl"] = _log10(last_val)
    out["days_since_last_valid"] = float((as_of - last_date).days)
    src.append(last_date)

    for w, name in [(3, "pre_mean_3d"), (7, "pre_mean_7d"), (14, "pre_mean_14d")]:
        win = pre[pre.index >= as_of - pd.Timedelta(days=w)]
        if len(win) > 0:
            out[name] = float(win.mean())
            src.extend(list(win.index))

    win7 = pre[pre.index >= as_of - pd.Timedelta(days=7)]
    if len(win7) > 0:
        out["pre_median_7d"] = float(win7.median())
        out["pre_std_7d"] = float(win7.std()) if len(win7) > 1 else 0.0
        src.extend(list(win7.index))

    for w, name in [(3, "pre_slope_3d"), (7, "pre_slope_7d")]:
        win = pre[pre.index >= as_of - pd.Timedelta(days=w)]
        if len(win) >= 2:
            out[name] = _slope_per_day(list(win.index), [float(v) for v in win.values])
            src.extend(list(win.index))

    if len(pre) >= 2:
        out["pre_delta_1d"] = float(pre.iloc[-1] - pre.iloc[-2])
        src.append(pre.index[-2])

    return out, src


def compute_post_features(series: pd.Series, gap_end: pd.Timestamp) -> tuple[dict, list]:
    """Post-edge features (hindcast only), from observations strictly after gap_end."""
    out: dict = {k: np.nan for k in POST_COLS}
    post = series[series.index > gap_end]
    if len(post) == 0:
        return out, []
    src: list[pd.Timestamp] = []

    first_date = post.index[0]
    first_val = float(post.iloc[0])
    out["post_first_chl"] = first_val
    out["post_first_log_chl"] = _log10(first_val)
    out["days_to_next_valid"] = float((first_date - gap_end).days)
    src.append(first_date)

    for w, name in [(3, "post_mean_3d"), (7, "post_mean_7d")]:
        win = post[post.index <= gap_end + pd.Timedelta(days=w)]
        if len(win) > 0:
            out[name] = float(win.mean())
            src.extend(list(win.index))

    win7 = post[post.index <= gap_end + pd.Timedelta(days=7)]
    if len(win7) > 0:
        out["post_median_7d"] = float(win7.median())
        out["post_std_7d"] = float(win7.std()) if len(win7) > 1 else 0.0
        src.extend(list(win7.index))

    for w, name in [(3, "post_slope_3d"), (7, "post_slope_7d")]:
        win = post[post.index <= gap_end + pd.Timedelta(days=w)]
        if len(win) >= 2:
            out[name] = _slope_per_day(list(win.index), [float(v) for v in win.values])
            src.extend(list(win.index))

    if len(post) >= 2:
        out["post_delta_1d"] = float(post.iloc[1] - post.iloc[0])
        src.append(post.index[1])

    return out, src


def compute_edge_features(pre_last: float, post_first: float, gap_length: int) -> dict:
    out: dict = {k: np.nan for k in EDGE_COLS}
    if pre_last != pre_last or post_first != post_first:
        return out
    pl_log = _log10(pre_last)
    pf_log = _log10(post_first)
    out["edge_mean_chl"] = 0.5 * (pre_last + post_first)
    if pl_log == pl_log and pf_log == pf_log:
        out["edge_log_mean"] = 0.5 * (pl_log + pf_log)
        out["edge_delta_log"] = pf_log - pl_log
        out["edge_slope_log"] = (pf_log - pl_log) / (gap_length + 1)
    out["edge_delta_chl"] = post_first - pre_last
    out["edge_slope_chl"] = (post_first - pre_last) / (gap_length + 1)
    return out


def compute_interp(pre_last_date, pre_last, post_first_date, post_first, d) -> tuple[float, float]:
    """Canonical linear-interpolation value for a hidden day (matches the
    Model-0 baseline exactly)."""
    if any(x is None or x != x for x in [pre_last, post_first]):
        return np.nan, np.nan
    if pre_last_date is None or post_first_date is None:
        return np.nan, np.nan
    total = (post_first_date - pre_last_date).days
    if total <= 0:
        return np.nan, np.nan
    frac = (d - pre_last_date).days / total
    v = pre_last + frac * (post_first - pre_last)
    return float(v), _log10(v)


def build_tier_c_feature_table(
    target_df: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the per-(gap, hidden-day) gap-edge feature table, registry, and
    warnings. Returns `(feature_table, registry, warnings)`."""
    obs = observed_series(target_df)
    rows: list[dict] = []
    warn_rows: list[dict] = []

    for _, g in candidates.iterrows():
        gap_id = str(g["gap_id"])
        L = int(g["gap_length"])
        start = pd.Timestamp(g["start_date"])
        end = pd.Timestamp(g["end_date"])
        hidden = pd.date_range(start, periods=L, freq="D")
        season = g.get("season", SEASON_MAP.get(start.month, "UNK"))
        year = int(g.get("year", start.year))
        is_spike = bool(g.get("is_high_chl_event", False))
        target_mean_true = float(g.get("target_mean_true", np.nan))
        is_sustained = (
            target_mean_true >= SUSTAINED_MEAN_THRESHOLD if target_mean_true == target_mean_true else False
        )

        pre_feat, pre_src = compute_pre_features(obs, start)
        post_feat, post_src = compute_post_features(obs, end)
        pre_available = pre_feat["pre_last_chl"] == pre_feat["pre_last_chl"]
        post_available = post_feat["post_first_chl"] == post_feat["post_first_chl"]

        pre_last = pre_feat["pre_last_chl"]
        post_first = post_feat["post_first_chl"]
        pre_last_date = obs[obs.index < start].index[-1] if pre_available else None
        post_first_date = obs[obs.index > end].index[0] if post_available else None
        edge_feat = compute_edge_features(pre_last, post_first, L)

        pre_src_dates = list(pre_src)
        post_src_dates = list(post_src)

        for k, d in enumerate(hidden, start=1):
            true_chl = target_df.loc[d, TARGET_COL] if d in target_df.index else np.nan
            true_chl = float(true_chl) if true_chl == true_chl else np.nan
            true_log = _log10(true_chl)

            interp_chl, interp_log = compute_interp(pre_last_date, pre_last, post_first_date, post_first, d)
            interp_available = interp_chl == interp_chl

            pre_dates_all = pre_src_dates + [d]
            hind_dates_all = pre_src_dates + post_src_dates + [d]
            dep_min_pre = min(pre_dates_all)
            dep_max_pre = max(pre_dates_all)
            dep_min_hind = min(hind_dates_all)
            dep_max_hind = max(hind_dates_all)

            row: dict = {
                "gap_id": gap_id, "date": d, "gap_length": L,
                "day_index_within_gap": k,
                "gap_position_fraction": k / (L + 1),
                "distance_to_left_edge": k,
                "distance_to_right_edge": L - k + 1,
                "season": season, "year": year,
                "is_high_chl_event": is_spike, "is_sustained_event": is_sustained,
                "true_chl": true_chl, "true_log": true_log,
                "linear_interp_chl": interp_chl, "linear_interp_log_chl": interp_log,
                "residual_log_true_minus_interp_log": (
                    true_log - interp_log if (true_log == true_log and interp_log == interp_log) else np.nan
                ),
                "residual_raw_true_minus_interp": (
                    true_chl - interp_chl if (true_chl == true_chl and interp_chl == interp_chl) else np.nan
                ),
                "pre_available": bool(pre_available),
                "post_available": bool(post_available),
                "interp_available": bool(interp_available),
                "dep_min_pre": dep_min_pre, "dep_max_pre": dep_max_pre,
                "dep_min_hind": dep_min_hind, "dep_max_hind": dep_max_hind,
            }
            row.update(pre_feat)
            row.update(post_feat)
            row.update(edge_feat)
            rows.append(row)

            if not interp_available:
                warn_rows.append({"gap_id": gap_id, "date": d.isoformat(), "warning": "linear_interp_unavailable"})
            if not pre_available:
                warn_rows.append({"gap_id": gap_id, "date": d.isoformat(), "warning": "no_pre_gap_observation"})
            if not post_available:
                warn_rows.append({"gap_id": gap_id, "date": d.isoformat(), "warning": "no_post_gap_observation"})

    feature_table = pd.DataFrame(rows)
    registry = build_feature_registry()
    warnings_df = pd.DataFrame(warn_rows) if warn_rows else pd.DataFrame(columns=["gap_id", "date", "warning"])
    return feature_table, registry, warnings_df


def build_feature_registry() -> pd.DataFrame:
    """One row per gap-edge feature: family, mode, predictor flag."""
    reg: list[dict] = []

    def add(cols, family, mode_allowed, is_predictor, note=""):
        for c in cols:
            reg.append({"feature_name": c, "family": family, "mode_allowed": mode_allowed,
                        "is_predictor": is_predictor, "note": note})

    add(META_COLS, "meta", "reference;pre_only;hindcast", True, "structural gap position; leakage-safe")
    add(PRE_COLS, "pre", "pre_only;hindcast", True, "pre-gap in-situ CHL (forecast-safe)")
    add(POST_COLS, "post", "hindcast", True, "post-gap in-situ CHL (hindcast only)")
    add(EDGE_COLS, "edge", "hindcast", True, "pre/post edge combinations (hindcast only)")
    add(INTERP_COLS, "interp", "hindcast", True, "linear interpolation value as predictor (hindcast only)")
    add(["true_chl", "true_log"], "target", "none", False, "prediction target; never a predictor")
    add(["residual_log_true_minus_interp_log", "residual_raw_true_minus_interp"], "target", "none", False,
        "residual target/eval column; never a predictor")
    add(["is_high_chl_event", "is_sustained_event"], "label", "none", False,
        "event label; eval stratifier only, never a predictor")
    return pd.DataFrame(reg)


def arm_feature_columns(external_cols: list[str], groups: list[str] = CANONICAL_GROUPS) -> list[str]:
    """Column list for the canonical (or any) arm, given the external columns
    (typically `tabular_models.load_arm4_numeric_columns(features_df)`)."""
    group_cols = {"meta": META_COLS, "pre": PRE_COLS, "post": POST_COLS,
                  "edge": EDGE_COLS, "interp": INTERP_COLS}
    cols: list[str] = []
    for g in groups:
        cols.extend(external_cols if g == "external" else group_cols[g])
    return cols


def build_extratrees(random_state: int = RANDOM_SEED) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=random_state)


def admissible_train_mask(
    gap_ids: np.ndarray, dep_min: np.ndarray, dep_max: np.ndarray,
    held_out_gid: str, a0: int, a1: int,
) -> np.ndarray:
    """Boolean mask of rows eligible to train when predicting `held_out_gid`.

    A row is admissible iff it belongs to a *different* gap AND its
    dependency window `[dep_min, dep_max]` is disjoint from the held-out
    gap's hidden span `[a0, a1]` -- this guarantees no training feature
    (edge/window/slope/interp, or the row's own label) reads a value hidden
    inside the held-out gap. `a0`/`a1`/`dep_min`/`dep_max` are int64 ns
    timestamps.
    """
    disjoint = (dep_max < a0) | (dep_min > a1)
    return (gap_ids != held_out_gid) & disjoint


def assemble_design(feature_table: pd.DataFrame, features_df: pd.DataFrame, external_cols: list[str]) -> pd.DataFrame:
    """Join external daily features onto the gap-edge table by date."""
    ext = features_df[external_cols].copy()
    return feature_table.merge(ext, left_on="date", right_index=True, how="left")


def run_loco_evaluation(
    candidates: pd.DataFrame,
    target_df: pd.DataFrame,
    features_df: pd.DataFrame,
    external_cols: list[str] | None = None,
    groups: list[str] = CANONICAL_GROUPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-gap-out evaluation of the canonical residual-over-interpolation
    gap-edge model over every row of `candidates`.

    Returns `(predictions_df, warnings_df)`. `predictions_df` has one row per
    (gap_id, date) with `pred_log10`/`pred`/`true` columns, back-transformed
    from the residual target.
    """
    if external_cols is None:
        external_cols = tm.load_arm4_numeric_columns(features_df)

    feature_table, _, _ = build_tier_c_feature_table(target_df, candidates)
    design = assemble_design(feature_table, features_df, external_cols)
    feat_cols = arm_feature_columns(external_cols, groups)

    X_all = design[feat_cols].to_numpy(dtype=float)
    gap_ids = design["gap_id"].to_numpy()
    dep_min = design["dep_min_hind"].astype("datetime64[ns]").astype("int64").to_numpy()
    dep_max = design["dep_max_hind"].astype("datetime64[ns]").astype("int64").to_numpy()
    dates = design["date"].to_numpy()
    y_all = design["residual_log_true_minus_interp_log"].to_numpy(dtype=float)
    interp_log_all = design["linear_interp_log_chl"].to_numpy(dtype=float)
    true_all = design["true_chl"].to_numpy(dtype=float)

    pred_rows: list[dict] = []
    warn_rows: list[dict] = []

    for _, gm in candidates.iterrows():
        gid = str(gm["gap_id"])
        a0 = np.datetime64(pd.Timestamp(gm["start_date"]), "ns").astype("int64")
        a1 = np.datetime64(pd.Timestamp(gm["end_date"]), "ns").astype("int64")

        test_mask = gap_ids == gid
        train_mask = admissible_train_mask(gap_ids, dep_min, dep_max, gid, a0, a1) & (~np.isnan(y_all))
        test_idx = np.where(test_mask)[0]
        n_train = int(train_mask.sum())
        if n_train < 30 or len(test_idx) == 0:
            warn_rows.append({"gap_id": gid, "warning": f"insufficient_training_rows ({n_train})"})
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = SimpleImputer(strategy="mean")
            X_tr = imp.fit_transform(X_all[train_mask])
            X_te = imp.transform(X_all[test_idx])
            model = build_extratrees()
            model.fit(X_tr, y_all[train_mask])
            resid_pred = model.predict(X_te)

        for j, ridx in enumerate(test_idx):
            il = interp_log_all[ridx]
            pred_log = float(il + resid_pred[j]) if il == il else np.nan
            pred = float(10.0**pred_log) if pred_log == pred_log else np.nan
            pred_rows.append({
                "gap_id": gid, "date": pd.Timestamp(dates[ridx]),
                "pred_log10": pred_log, "pred": pred,
                "true": float(true_all[ridx]) if true_all[ridx] == true_all[ridx] else np.nan,
                "n_train": n_train,
            })

    predictions_df = pd.DataFrame(pred_rows)
    warnings_df = pd.DataFrame(warn_rows) if warn_rows else pd.DataFrame(columns=["gap_id", "warning"])
    return predictions_df, warnings_df
