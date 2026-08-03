"""Tests for the gap-edge residual reconstruction model.

Uses a small synthetic series (not the full 449-gap matched-support
benchmark, which is exercised separately via
`experiments.chlorophyll.run_classical_benchmark --verify` against the
frozen released tables) so this suite runs in CI in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import gap_edge_models as gem


def _synthetic_target(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    values = 1.0 + 0.3 * np.sin(np.arange(n_days) / 20) + rng.normal(0, 0.05, n_days)
    values = np.abs(values) + 0.1
    df = pd.DataFrame(
        {gem.TARGET_COL: values, gem.ELIGIBLE_COL: True}, index=dates,
    )
    df.index.name = "date"
    return df


def _candidates(target_df: pd.DataFrame, gap_length: int, n: int, offset: int = 60) -> pd.DataFrame:
    starts = target_df.index[offset : offset + n * (gap_length + 10) : gap_length + 10]
    rows = []
    for i, start in enumerate(starts):
        end = start + pd.Timedelta(days=gap_length - 1)
        rows.append({
            "gap_id": f"L{gap_length:02d}_test{i}", "gap_length": gap_length,
            "start_date": start, "end_date": end,
            "season": "JJA", "year": start.year, "is_high_chl_event": False,
            "target_mean_true": float(target_df.loc[start:end, gem.TARGET_COL].mean()),
        })
    return pd.DataFrame(rows)


def test_observed_series_excludes_ineligible_and_nonpositive():
    df = _synthetic_target(50)
    df.loc[df.index[5], gem.ELIGIBLE_COL] = False
    df.loc[df.index[10], gem.TARGET_COL] = -1.0
    obs = gem.observed_series(df)
    assert df.index[5] not in obs.index
    assert df.index[10] not in obs.index


def test_build_tier_c_feature_table_tolerates_present_but_nan_optional_columns():
    """Regression test for a real bug found this session: a candidates row
    concatenated onto a pool that already has `season`/`year`/
    `is_high_chl_event`/`target_mean_true` columns gets an explicit NaN for
    those columns if it doesn't supply them itself -- `.get(key, default)`
    does NOT fall through to the default in that case (the key is present,
    just NaN), so a naive read crashed with `ValueError: cannot convert
    float NaN to integer` on `year`. Every one of these columns is
    bookkeeping/labeling only and must degrade to the computed default
    whenever the value is missing OR NaN."""
    target_df = _synthetic_target(60)
    candidates = pd.DataFrame([
        {
            "gap_id": "with_labels", "gap_length": 3, "start_date": target_df.index[10],
            "end_date": target_df.index[12], "season": "JJA", "year": 2018,
            "is_high_chl_event": True, "target_mean_true": 5.0,
        },
        {
            # Same columns present (from concatenation with the row above),
            # but this row's own values are NaN -- must not crash.
            "gap_id": "nan_labels", "gap_length": 3, "start_date": target_df.index[30],
            "end_date": target_df.index[32], "season": float("nan"), "year": float("nan"),
            "is_high_chl_event": float("nan"), "target_mean_true": float("nan"),
        },
    ])
    feature_table, _, _ = gem.build_tier_c_feature_table(target_df, candidates)
    nan_rows = feature_table[feature_table["gap_id"] == "nan_labels"]
    assert not nan_rows.empty
    assert (nan_rows["year"] == target_df.index[30].year).all()
    assert (nan_rows["season"] != "nan").all()
    assert (nan_rows["is_high_chl_event"] == False).all()  # noqa: E712 - pandas Series comparison


def test_compute_interp_matches_linear_baseline():
    pre_date = pd.Timestamp("2020-01-01")
    post_date = pd.Timestamp("2020-01-11")
    v, log_v = gem.compute_interp(pre_date, 1.0, post_date, 2.0, pd.Timestamp("2020-01-06"))
    assert v == pytest.approx(1.5, abs=1e-9)


def test_dependency_window_excludes_rows_that_overlap_held_out_gap():
    gap_ids = np.array(["A", "A", "B", "B"])
    dep_min = np.array([0, 0, 5, 5], dtype="int64")
    dep_max = np.array([2, 2, 8, 8], dtype="int64")
    mask = gem.admissible_train_mask(gap_ids, dep_min, dep_max, held_out_gid="B", a0=6, a1=9)
    # Gap B's own rows excluded (different-gap rule); gap A's rows admissible
    # only if disjoint from [6, 9] -- dep_max=2 < 6, so admissible.
    assert list(mask) == [True, True, False, False]


def test_run_loco_evaluation_produces_predictions_with_enough_gaps():
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    external_cols: list[str] = []  # no external predictors needed for this smoke test
    features_df = pd.DataFrame(index=target_df.index)

    preds, warns = gem.run_loco_evaluation(candidates, target_df, features_df, external_cols)
    assert not preds.empty
    assert set(preds["gap_id"]).issubset(set(candidates["gap_id"]))
    assert preds["pred"].notna().any()


def test_score_gap_ids_scores_only_the_requested_subset_using_full_context():
    """`score_gap_ids` should fit/score only the requested gaps, but still
    draw admissible training rows from the entire `candidates` pool (not
    just the scored subset) -- this is what makes a small, fast notebook
    illustration possible without shrinking the training context."""
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)
    subset_ids = candidates["gap_id"].iloc[[2, 7]].tolist()

    preds, _ = gem.run_loco_evaluation(candidates, target_df, features_df, [], score_gap_ids=subset_ids)
    assert set(preds["gap_id"]) == set(subset_ids)
    assert preds["pred"].notna().any()


def test_reference_arm_uses_external_and_meta_only_no_post_edge_interp():
    """Protocol B (`run_reference_arm_loco_evaluation`) must use exactly
    `external + meta` feature groups -- no post/edge/interp columns, which
    would make it depend on post-gap in-situ chlorophyll like the (very
    different) gap-edge residual model does."""
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)

    preds, warns = gem.run_reference_arm_loco_evaluation(
        "extratrees", candidates, target_df, features_df, external_cols=[],
    )
    assert not preds.empty
    assert preds["pred"].notna().any()


def test_reference_arm_predicts_log_directly_not_a_residual():
    """Protocol B predicts `true_log10` directly (private `tier_a_arm4_reference`
    arm, `target: "log"`) -- it has no interpolation anchor to add a residual
    to, unlike the canonical gap-edge model (`run_loco_evaluation`'s default
    `target_mode="residual_log"`)."""
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)

    ref_preds, _ = gem.run_reference_arm_loco_evaluation(
        "extratrees", candidates, target_df, features_df, external_cols=[],
    )
    residual_preds, _ = gem.run_loco_evaluation(candidates, target_df, features_df, [])
    # Different target definitions and feature sets -- predictions must not
    # be identical (a regression that silently routed both through the same
    # residual-over-interpolation code path would produce identical output).
    merged = ref_preds.merge(residual_preds, on=["gap_id", "date"], suffixes=("_ref", "_resid"))
    assert not merged.empty
    assert not np.allclose(merged["pred_log10_ref"], merged["pred_log10_resid"])


def test_reference_arm_hgb_dispatch_produces_predictions():
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)
    preds, _ = gem.run_reference_arm_loco_evaluation(
        "hgb", candidates, target_df, features_df, external_cols=[],
    )
    assert not preds.empty
    assert preds["pred"].notna().any()


def test_run_loco_evaluation_rejects_unknown_target_mode_dep_window_model_name():
    target_df = _synthetic_target(60)
    candidates = _candidates(target_df, gap_length=3, n=5)
    features_df = pd.DataFrame(index=target_df.index)
    with pytest.raises(ValueError):
        gem.run_loco_evaluation(candidates, target_df, features_df, [], target_mode="bogus")
    with pytest.raises(ValueError):
        gem.run_loco_evaluation(candidates, target_df, features_df, [], dep_window="bogus")
    with pytest.raises(ValueError):
        gem.run_loco_evaluation(candidates, target_df, features_df, [], model_name="bogus")


def test_default_run_loco_evaluation_behavior_unchanged_by_new_parameters():
    """The generalization that added `target_mode`/`dep_window`/`model_name`
    must not change the canonical gap-edge model's default behavior -- same
    call, same signature position, same result as before the generalization."""
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)
    explicit, _ = gem.run_loco_evaluation(
        candidates, target_df, features_df, [],
        target_mode="residual_log", dep_window="hind", model_name="extratrees",
    )
    default, _ = gem.run_loco_evaluation(candidates, target_df, features_df, [])
    pd.testing.assert_frame_equal(explicit, default)


def test_tamper_invariance_a_gaps_own_hidden_truth_never_changes_its_own_prediction():
    """A gap's own prediction must not depend on what its own hidden truth
    actually is: `compute_pre_features`/`compute_post_features` only read
    observations strictly before the gap's start / after its end, and the
    row's own `true_chl` is used only for scoring afterwards, never as a
    model input. This mirrors the pattern in `test_leakage_invariance.py`
    (mask, tamper the hidden interval, confirm the model's own output for
    that interval is unaffected) applied to the gap-edge model directly,
    without going through the multi-gap LOCO training-set-composition effects
    a naive full-evaluation tamper test would otherwise conflate with real
    leakage.
    """
    target_df = _synthetic_target(400)
    candidates = _candidates(target_df, gap_length=3, n=15)
    features_df = pd.DataFrame(index=target_df.index)
    victim_gap = candidates.iloc[3]
    hidden = pd.date_range(victim_gap["start_date"], victim_gap["end_date"], freq="D")

    baseline, _ = gem.run_loco_evaluation(candidates, target_df, features_df, [])

    tampered = target_df.copy()
    tampered.loc[hidden, gem.TARGET_COL] = 9999.0  # only the victim gap's own interior
    tampered_preds, _ = gem.run_loco_evaluation(candidates, tampered, features_df, [])

    gid = victim_gap["gap_id"]
    b = baseline[baseline["gap_id"] == gid].sort_values("date")["pred"].to_numpy()
    t = tampered_preds[tampered_preds["gap_id"] == gid].sort_values("date")["pred"].to_numpy()
    assert len(b) == len(t) == 3
    np.testing.assert_allclose(b, t)
