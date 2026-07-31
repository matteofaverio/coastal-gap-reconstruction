"""TargetSpec: the one place each case study declares what its target column means.

Every module that needs to know "which column is the target," "what scale is it
scored on," or "does this target have an event label" should read a `TargetSpec`
instance rather than importing scattered constants -- this is a real, used object
(see `experiments/chlorophyll/_config.py::TARGET_SPEC`,
`experiments/oxygen/_config.py::TARGET_SPEC`, and their use in
`target_and_gap_pool.py`), not decorative configuration nobody reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """Everything generic scoring/validation code needs to know about a target.

    Attributes
    ----------
    name:
        Short identifier, e.g. "chlorophyll" or "oxygen".
    target_col:
        Column holding the daily target value, in its raw/physical unit.
    eligible_col:
        Boolean column: True if the day has enough valid hourly observations to
        trust the daily value.
    date_col:
        Date column name (every released table in this repository uses "date",
        but this is not hardcoded as an assumption).
    display_unit:
        Physical unit of `target_col`, for figure/table labeling (e.g. "mg/m^3").
    benchmark_scoring_scale:
        The transform applied to `target_col` before computing the released
        benchmark metrics -- "identity" (score in the raw unit) or "log10" (score
        on log10(target_col)). Chlorophyll's released benchmark scores on log10;
        oxygen's scores on the raw mg/L scale. Getting this wrong silently
        produces numbers that look like a metric but are not comparable to the
        released tables -- see `tests/test_metrics_and_scales.py`.
    positive_only:
        True if non-positive values are physically invalid for this target (e.g.
        chlorophyll concentration) and must be excluded before a log transform,
        rather than merely treated as missing.
    event_label:
        Optional callable `(hidden_values: pd.Series, threshold: float) -> bool`
        implementing this target's high-value event definition, or `None` if the
        target has no event label (oxygen currently has none -- see
        `experiments/oxygen/target_and_gap_pool.py`'s module docstring).
    """

    name: str
    target_col: str
    eligible_col: str
    date_col: str
    display_unit: str
    benchmark_scoring_scale: str
    positive_only: bool
    event_label: Callable[[pd.Series, float], bool] | None = None

    def __post_init__(self) -> None:
        if self.benchmark_scoring_scale not in ("identity", "log10"):
            raise ValueError(
                f"benchmark_scoring_scale must be 'identity' or 'log10', "
                f"got {self.benchmark_scoring_scale!r}"
            )
