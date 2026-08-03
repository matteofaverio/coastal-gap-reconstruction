"""Tests for the gap-edge residual reconstruction model.

Uses a small synthetic series (not the full 449-gap benchmark, which is
covered separately by the byte-identical/tolerance regression tests in
`test_released_result_reproduction.py`) so this suite runs in CI in well
under a second.
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
