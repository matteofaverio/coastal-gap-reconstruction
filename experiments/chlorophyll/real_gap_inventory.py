"""Deterministic real-gap inventory detection for chlorophyll.

Ported from the original real-gap inventory generator: identifies
every contiguous run of non-eligible days in the released daily target,
computes gap metadata (class, seasons spanned, edge availability,
admissibility, distance from the validated artificial-gap length grid), and
assigns a deterministic gap ID. This is pure detection over the daily
target series -- **it never inspects any candidate prediction file** to
decide where the real gaps are; the gap boundaries come only from
`target_eligible_default`.

Reproduces the released `data/chlorophyll/chlorophyll_real_gap_inventory.csv`
(128 rows) exactly when run against the same public daily target table.
"""

from __future__ import annotations

import pandas as pd

from . import real_gap_contract as rgc

__all__ = [
    "SEASON_MAP", "VALIDATED_GAP_LENGTHS", "gap_class", "seasons_spanned",
    "detect_real_gaps", "INVENTORY_COLUMNS",
]

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# The artificial-gap validation lengths this inventory was built against
# (matches the released chlorophyll_real_gap_inventory.csv's own
# nearest_val_lengths/extrapolation_beyond_validation values exactly --
# this is the grid that was canonical at the time that file was produced,
# not the later-expanded canonical 681-gap pool's L=1,3,7,14,30,45,60 grid;
# kept as the original grid so this module reproduces the released file
# byte-for-byte rather than a value computed against a different, later
# validation grid).
VALIDATED_GAP_LENGTHS: list[int] = [1, 3, 7, 14, 30]

INVENTORY_COLUMNS = [
    "gap_id", "start_date", "end_date", "length_days", "gap_class", "seasons",
    "year_start", "year_end", "pre_edge_available", "post_edge_available",
    "interpolation_admissible", "gap_edge_features_admissible",
    "nearest_val_lengths", "extrapolation_beyond_validation", "notes",
]


def gap_class(n: int) -> str:
    if n <= 7:
        return "short"
    if n <= 30:
        return "medium"
    if n <= 90:
        return "long"
    return "very_long"


def seasons_spanned(start: pd.Timestamp, end: pd.Timestamp) -> str:
    months = pd.date_range(start, end, freq="MS").month.tolist()
    if not months:
        months = [start.month]
    seen: list[str] = []
    for m in months:
        s = SEASON_MAP[m]
        if not seen or seen[-1] != s:
            seen.append(s)
    return "/".join(dict.fromkeys(seen))


def _nearest_val_and_extrapolation(length: int) -> tuple[str, str]:
    if length in VALIDATED_GAP_LENGTHS:
        return str(length), "no"
    below = [v for v in VALIDATED_GAP_LENGTHS if v <= length]
    above = [v for v in VALIDATED_GAP_LENGTHS if v > length]
    nearest = f"{max(below) if below else '?'}–{min(above) if above else '?'}"
    extrapolation = "yes" if length > 30 else "interpolation_within_range"
    return nearest, extrapolation


def detect_real_gaps(target_df: pd.DataFrame) -> pd.DataFrame:
    """Detect every contiguous non-eligible run in `target_df` (indexed by
    date, with an `ELIGIBLE_COLUMN` boolean column) and return the 15-column
    inventory schema (`INVENTORY_COLUMNS`), gap boundaries only -- no
    prediction file is read here.

    A gap running to the end of the series (never observed in the released
    128-gap chlorophyll inventory, but handled the same way the private
    generator handles it) gets `gap_id` prefix `REAL_OPEN_` and
    `nearest_val_lengths="open"`/`extrapolation_beyond_validation="open_ended"`.
    """
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
                prefix = "L001" if length <= 7 else "L010" if length <= 30 else "L031" if length <= 90 else "L091"
                gid = f"REAL_{prefix}_{gap_start.strftime('%Y%m%d')}"

                pre_window = eligible.loc[max(full_range[0], gap_start - pd.Timedelta(days=14)):gap_start - pd.Timedelta(days=1)]
                post_window = eligible.loc[gap_end + pd.Timedelta(days=1):min(full_range[-1], gap_end + pd.Timedelta(days=14))]
                pre_avail = bool(pre_window.any()) if len(pre_window) > 0 else False
                post_avail = bool(post_window.any()) if len(post_window) > 0 else False
                interp_admissible = pre_avail and post_avail

                nearest_val, extrapolation = _nearest_val_and_extrapolation(length)
                notes = ""
                if length > 90:
                    notes = "very long: extrapolation far beyond L=30 validation"
                if length == 256:
                    notes = "major 2020 sensor gap (2020-02-12 to 2020-10-23); PLV available"

                gaps.append({
                    "gap_id": gid, "start_date": gap_start.date(), "end_date": gap_end.date(),
                    "length_days": length, "gap_class": gap_class(length),
                    "seasons": seasons_spanned(gap_start, gap_end),
                    "year_start": int(gap_start.year), "year_end": int(gap_end.year),
                    "pre_edge_available": pre_avail, "post_edge_available": post_avail,
                    "interpolation_admissible": interp_admissible,
                    "gap_edge_features_admissible": interp_admissible,
                    "nearest_val_lengths": nearest_val,
                    "extrapolation_beyond_validation": extrapolation,
                    "notes": notes,
                })
                in_gap = False

    if in_gap:
        gap_end = full_range[-1]
        length = (gap_end - gap_start).days + 1
        gaps.append({
            "gap_id": f"REAL_OPEN_{gap_start.strftime('%Y%m%d')}",
            "start_date": gap_start.date(), "end_date": gap_end.date(),
            "length_days": length, "gap_class": gap_class(length),
            "seasons": seasons_spanned(gap_start, gap_end),
            "year_start": int(gap_start.year), "year_end": int(gap_end.year),
            "pre_edge_available": True, "post_edge_available": False,
            "interpolation_admissible": False, "gap_edge_features_admissible": False,
            "nearest_val_lengths": "open", "extrapolation_beyond_validation": "open_ended",
            "notes": "gap extends to end of available data",
        })

    return pd.DataFrame(gaps, columns=INVENTORY_COLUMNS)
