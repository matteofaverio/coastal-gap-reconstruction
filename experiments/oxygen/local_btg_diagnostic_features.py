"""Local BTG water-temperature/pressure diagnostic feature construction for oxygen.

Builds the daily aggregation behind `data_public/oxygen/oxygen_local_btg_diagnostic_features.csv`
-- the single feature source used by the `local_btg_temp_pressure_diagnostic` arm
in the oxygen benchmark (a diagnostic-only arm, never the primary comparator; see
the module-level "Diagnostic status" note below).

This module builds only this one small table, not the full oxygen feature-arm
construction (external_physical_core/plus_currents/all_available, which need the
shared 265-column external table and are not yet published -- see
`experiments/oxygen/_config.py`'s module docstring for the current scope
boundary).

Ported from the private project's `oxygen_features.py::build_local_btg_daily`,
verified byte-identical against the private output (see
`tests/test_local_btg_diagnostic_features.py`).

Diagnostic status
------------------
Water temperature (BTGTA) and pressure (BTGPA) are measured on the *same
physical station/buoy* as the oxygen sensor itself (BTGOXD2). Their availability
may therefore covary with oxygen sensor outages in a way that is hard to
distinguish from genuine predictive signal -- a day with no oxygen reading is
more likely to also have no BTGTA/BTGPA reading, for reasons unrelated to
oxygen's true physical relationship with temperature/pressure. For this reason,
the private project restricts this feature arm to diagnostic/appendix use only,
never the primary external-only benchmark (`oxygen_benchmark_by_length.csv` and
every other released oxygen results table exclude this arm). Treat any result
using these columns as diagnostic, not as evidence for the core benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_COL = "date"
TIMESTAMP_COL = "timestamp"
VALUE_COL = "prom"


def build_local_btg_daily(merged_hourly_path: str | Path, var_name: str) -> pd.DataFrame:
    """Aggregate an hourly BTG merged file (BTGTA or BTGPA) to a daily mean +
    valid-hour count + availability flag.

    Does not apply oxygen's own target-eligibility rule to the predictor itself --
    a predictor's own daily aggregation convention need not match the target's
    eligibility rule (this mirrors the private project's explicit O3.1/O3.2
    design choice, not an oversight).

    Returns a DataFrame indexed by date with three columns:
    `{var_name}_daily_mean`, `{var_name}_valid_hours`, `{var_name}_available`
    (True iff valid_hours > 0, i.e. any observation that day).
    """
    df = pd.read_csv(merged_hourly_path, parse_dates=[TIMESTAMP_COL])
    df[DATE_COL] = df[TIMESTAMP_COL].dt.normalize()
    grouped = df.groupby(DATE_COL)[VALUE_COL]
    daily_mean = grouped.mean()
    valid_hours = grouped.apply(lambda s: s.notna().sum())

    out = pd.DataFrame({
        f"{var_name}_daily_mean": daily_mean,
        f"{var_name}_valid_hours": valid_hours,
    })
    out[f"{var_name}_available"] = out[f"{var_name}_valid_hours"] > 0
    return out.sort_index()


def build_oxygen_local_btg_diagnostic_features(
    btg_ta_merged_path: str | Path,
    btg_pa_merged_path: str | Path,
) -> pd.DataFrame:
    """Build the full local-BTG diagnostic feature table: water temperature
    (BTGTA) joined with pressure (BTGPA), outer join on date."""
    ta_daily = build_local_btg_daily(btg_ta_merged_path, "btg_water_temp")
    pa_daily = build_local_btg_daily(btg_pa_merged_path, "btg_pressure")
    return ta_daily.join(pa_daily, how="outer").sort_index()
