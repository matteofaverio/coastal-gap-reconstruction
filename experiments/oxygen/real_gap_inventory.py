"""Deterministic real-gap inventory detection for oxygen -- inventory only,
no reconstruction candidates (see `real_gap_contract.py`'s module docstring
for why).

Reuses the exact same contiguous-run detection and gap-class thresholds as
`experiments.chlorophyll.real_gap_inventory` (target-agnostic: gap
classification by length has no chlorophyll-specific assumption baked in,
and the released `oxygen_real_gap_inventory_by_class.csv`'s own class
boundaries -- 1-7/8-30/31-90/>90 -- are identical to chlorophyll's), applied
to the oxygen daily target's `eligible_ge_18` column instead of
`target_eligible_default`.

Only a **by-class aggregate** inventory is published for oxygen
(`data/oxygen/oxygen_real_gap_inventory_by_class.csv`) -- no
standalone per-gap CSV exists (private audit: `MISSING`). This module
detects the per-gap inventory directly from the public daily target (for
validation, written only to `build/`, never presented as an additional
authoritative public data file) and aggregates it to the same by-class
schema for comparison against the released summary.
"""

from __future__ import annotations

import pandas as pd

from experiments.chlorophyll.real_gap_inventory import gap_class, seasons_spanned

from . import real_gap_contract as rgc

__all__ = ["detect_real_gaps", "aggregate_by_class", "BY_CLASS_COLUMNS"]

BY_CLASS_COLUMNS = ["gap_class", "length_range_days", "n_gaps", "total_missing_days",
                    "median_length_days", "max_length_days"]
_LENGTH_RANGE = {"short": "1-7", "medium": "8-30", "long": "31-90", "very_long": ">90"}
_CLASS_ORDER = ["short", "medium", "long", "very_long"]


def detect_real_gaps(target_df: pd.DataFrame) -> pd.DataFrame:
    """Detect every contiguous non-eligible run in the oxygen daily target
    (indexed by date, `eligible_ge_18` boolean column). Pure detection over
    the target series only -- no candidate/prediction file is read.
    Structurally identical algorithm to the chlorophyll detector (see
    `experiments.chlorophyll.real_gap_inventory.detect_real_gaps`), not
    reimplemented separately to avoid the two silently diverging."""
    eligible = target_df[rgc.ELIGIBLE_COLUMN].fillna(False).astype(bool)
    full_range = pd.date_range(target_df.index.min(), target_df.index.max(), freq="D")
    eligible = eligible.reindex(full_range, fill_value=False)

    gaps: list[dict] = []
    in_gap = False
    gap_start = None
    for date, is_eligible in eligible.items():
        if not is_eligible:
            if not in_gap:
                in_gap = True
                gap_start = date
        else:
            if in_gap:
                gap_end = date - pd.Timedelta(days=1)
                length = (gap_end - gap_start).days + 1
                gaps.append({
                    "gap_id": f"OXREAL_{gap_start.strftime('%Y%m%d')}",
                    "start_date": gap_start.date(), "end_date": gap_end.date(),
                    "length_days": length, "gap_class": gap_class(length),
                    "seasons": seasons_spanned(gap_start, gap_end),
                })
                in_gap = False
    if in_gap:
        gap_end = full_range[-1]
        length = (gap_end - gap_start).days + 1
        gaps.append({
            "gap_id": f"OXREAL_OPEN_{gap_start.strftime('%Y%m%d')}",
            "start_date": gap_start.date(), "end_date": gap_end.date(),
            "length_days": length, "gap_class": gap_class(length),
            "seasons": seasons_spanned(gap_start, gap_end),
        })
    return pd.DataFrame(gaps)


def aggregate_by_class(inventory: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-gap inventory to the same by-class schema as the
    released `oxygen_real_gap_inventory_by_class.csv`."""
    rows = []
    for gc in _CLASS_ORDER:
        sub = inventory[inventory["gap_class"] == gc]
        rows.append({
            "gap_class": gc, "length_range_days": _LENGTH_RANGE[gc],
            "n_gaps": len(sub),
            "total_missing_days": int(sub["length_days"].sum()),
            "median_length_days": float(sub["length_days"].median()) if len(sub) else 0.0,
            "max_length_days": int(sub["length_days"].max()) if len(sub) else 0,
        })
    return pd.DataFrame(rows, columns=BY_CLASS_COLUMNS)
