"""Enforces the generic-mechanics / target-specific boundary declared in
`coastal_gap_reconstruction/gaps.py`'s module docstring: shared code must never
emit target-specific columns, and each target's own builder must be the only
place those columns are added.
"""
from __future__ import annotations

from pathlib import Path

from coastal_gap_reconstruction.gaps import generate_candidate_gaps
from experiments.chlorophyll import _config as chl_config
from experiments.chlorophyll import target_and_gap_pool as chl_tgp
from experiments.oxygen import _config as ox_config
from experiments.oxygen import target_and_gap_pool as ox_tgp

REPO_ROOT = Path(__file__).resolve().parent.parent
CHL_TARGET = REPO_ROOT / "data" / "chlorophyll" / "chlorophyll_daily_target.csv"
OX_TARGET = REPO_ROOT / "data" / "oxygen" / "oxygen_daily_target.csv"

CHLOROPHYLL_ONLY_FIELDS = {
    "is_high_chl_event", "is_sustained_event", "is_background", "chl_90th_threshold",
}


def test_generic_candidate_gaps_emits_no_target_specific_fields() -> None:
    """`gaps.generate_candidate_gaps` (the shared, target-neutral scaffold) must
    only ever emit gap_id/gap_length/start_date/end_date/n_hidden_days/season/year
    -- never an event flag, threshold, or any other target-specific column,
    regardless of which target's data it is pointed at."""
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    candidates = generate_candidate_gaps(
        target_df,
        gap_lengths=[7],
        seed=42,
        max_per_length=5,
        eligible_col=chl_config.ELIGIBLE_COL,
    )
    expected_cols = {"gap_id", "gap_length", "start_date", "end_date", "n_hidden_days", "season", "year"}
    assert set(candidates.columns) == expected_cols
    assert CHLOROPHYLL_ONLY_FIELDS.isdisjoint(candidates.columns)


def test_chlorophyll_builder_adds_its_own_target_specific_fields() -> None:
    """The chlorophyll-specific builder, unlike the generic scaffold, must add
    its event/threshold columns -- confirming the split is real, not merely
    that the generic layer is empty by omission."""
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    checksum = chl_tgp.target_table_checksum(CHL_TARGET)
    pool = chl_tgp.build_gap_pool(target_df, checksum, gap_lengths=[7])
    assert CHLOROPHYLL_ONLY_FIELDS.issubset(pool.columns)


def test_oxygen_builder_has_no_chlorophyll_event_labels() -> None:
    """Oxygen has no event/high-value label anywhere in this project (see
    experiments/oxygen/target_and_gap_pool.py's module docstring) -- its
    released schema must never accidentally pick up a chlorophyll-shaped
    column, e.g. by copy-pasting the chlorophyll builder's column list."""
    target_df = ox_tgp.load_daily_target(OX_TARGET)
    pool = ox_tgp.build_gap_pool(target_df, gap_lengths=[7], exploratory_lengths=[])
    pool = ox_tgp.add_support_role(pool)
    assert CHLOROPHYLL_ONLY_FIELDS.isdisjoint(pool.columns)
    assert list(pool.columns) == ox_tgp.POOL_COLUMNS


def test_target_specs_declare_disjoint_target_columns() -> None:
    """Sanity check on the TargetSpec objects themselves: the two targets must
    not accidentally share a target column name, which would silently conflate
    them if a shared table ever carried both."""
    assert chl_config.TARGET_SPEC.target_col != ox_config.TARGET_SPEC.target_col
    assert chl_config.TARGET_SPEC.event_label is not None
    assert ox_config.TARGET_SPEC.event_label is None


def test_gap_lengths_are_never_defaulted_silently() -> None:
    """`gaps.generate_candidate_gaps` must require gap_lengths explicitly --
    calling it without the argument is a TypeError, not a silent fallback to
    some chlorophyll- or oxygen-shaped default."""
    import inspect

    sig = inspect.signature(generate_candidate_gaps)
    assert sig.parameters["gap_lengths"].default is inspect.Parameter.empty
