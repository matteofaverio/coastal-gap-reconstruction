"""Data loading, gap construction, and export helpers for the gap-reconstruction
walkthrough notebook (`demo/gap_reconstruction_walkthrough.ipynb`).

No plotting lives here (see `plotting.py`) and no reconstruction method lives here
(see `methods.py`). Every function takes explicit inputs and returns explicit
outputs; nothing is read from or written to module-level mutable state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]  # demo/
DATA_DIR = DEMO_ROOT / "data"
OUTPUTS_DIR = DEMO_ROOT / "outputs"

TARGET_COLUMN = "chl_mean"
SATELLITE_PROXY_COLUMN = "chl_satellite_proxy_log10"
WIND_COLUMN = "wind_spd_ms"
SST_COLUMN = "sst_primary_degC"

QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def load_demo_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the local demonstration window: in-situ chlorophyll target plus three
    covariates (satellite chlorophyll proxy, wind speed, sea-surface temperature).

    Returns a DataFrame sorted by date with columns:
    date, chl_mean, chl_satellite_proxy_log10, wind_spd_ms, sst_primary_degC,
    target_eligible_default, coverage_fraction.
    """
    df = pd.read_csv(data_dir / "chlorophyll_demo_series.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_full_record(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the full multi-year chlorophyll record (same columns as
    `load_demo_data`), used to fit methods that need more history than the local
    demonstration window contains (climatology, the tabular models)."""
    df = pd.read_csv(data_dir / "chlorophyll_full_record_for_tabular_fit.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_real_gap_example(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the window around one real, naturally-occurring chlorophyll gap
    (2015-07-01 to 2015-07-14). There is no withheld truth for this interval."""
    df = pd.read_csv(data_dir / "chlorophyll_real_gap_example.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def data_summary(data: pd.DataFrame) -> dict:
    """Compact summary of a loaded data window: date range, resolution, target
    column, available covariates, and missing-value counts. No printing --
    the caller decides how to display this."""
    covariate_cols = [c for c in (SATELLITE_PROXY_COLUMN, WIND_COLUMN, SST_COLUMN) if c in data.columns]
    return {
        "start_date": data["date"].min().date().isoformat(),
        "end_date": data["date"].max().date().isoformat(),
        "n_days": len(data),
        "resolution": "daily",
        "target_column": TARGET_COLUMN,
        "covariate_columns": covariate_cols,
        "missing_target_days": int(data[TARGET_COLUMN].isna().sum()),
        **{f"missing_{c}_days": int(data[c].isna().sum()) for c in covariate_cols},
    }


@dataclass
class ArtificialGap:
    """An artificial gap carved out of an otherwise-observed interval.

    Attributes:
        full_series: the complete, unmodified input data (target column intact).
        masked_series: the same data with `target_column` set to NaN inside
            [gap_start, gap_end] -- this is what every reconstruction method sees.
        truth: date + true target value for the hidden interval, kept aside only
            for scoring after reconstruction.
        gap_start, gap_end: inclusive interval bounds.
        target_column: name of the column that was masked.
        is_gap: boolean mask over `full_series`, True inside the gap.
    """

    full_series: pd.DataFrame
    masked_series: pd.DataFrame
    truth: pd.DataFrame
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    target_column: str
    is_gap: pd.Series

    @property
    def gap_length_days(self) -> int:
        return int((self.gap_end - self.gap_start).days) + 1

    @property
    def context(self) -> pd.DataFrame:
        """Observed rows outside the gap (target not NaN)."""
        return self.masked_series.dropna(subset=[self.target_column]).reset_index(drop=True)

    def completeness(self, column: str) -> float:
        """Fraction of non-missing values for `column` within the gap interval,
        as observed in the original (unmasked) data -- used to check how much
        covariate information remains available while the target is hidden."""
        window = self.full_series.loc[self.is_gap, column]
        if len(window) == 0:
            return float("nan")
        return float(window.notna().mean())


def create_artificial_gap(
    data: pd.DataFrame,
    start: str,
    end: str,
    target_column: str = TARGET_COLUMN,
) -> ArtificialGap:
    """Hide `target_column` between `start` and `end` (inclusive), keeping the
    true values aside as `truth`. Every covariate column is left untouched --
    only the target is masked.

    Raises if the interval is not fully observed in `data` (an artificial gap
    must start from real, known values to be usable for validation) or if there
    are duplicate dates.
    """
    if data["date"].duplicated().any():
        raise ValueError("Input data has duplicate dates -- fix before creating a gap.")

    gap_start = pd.Timestamp(start)
    gap_end = pd.Timestamp(end)
    is_gap = (data["date"] >= gap_start) & (data["date"] <= gap_end)

    truth = (
        data.loc[is_gap, ["date", target_column]]
        .rename(columns={target_column: "truth"})
        .reset_index(drop=True)
    )
    if truth["truth"].isna().any():
        raise ValueError(
            f"Interval {start}..{end} is not fully observed in the input data -- "
            "pick a different interval for an artificial gap."
        )

    masked = data.copy()
    masked.loc[is_gap, target_column] = np.nan

    return ArtificialGap(
        full_series=data,
        masked_series=masked,
        truth=truth,
        gap_start=gap_start,
        gap_end=gap_end,
        target_column=target_column,
        is_gap=is_gap,
    )


def mean_absolute_error(prediction: pd.DataFrame, truth: pd.DataFrame, value_col: str = "value") -> float:
    """MAE between a prediction table (columns: date, `value_col`) and a truth
    table (columns: date, truth), joined on date."""
    merged = truth.merge(prediction[["date", value_col]], on="date", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping dates between prediction and truth.")
    return float((merged[value_col] - merged["truth"]).abs().mean())


def build_export_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble the operational results table from a list of row dicts (see
    `methods.py` callers for the expected keys). Pure function: does not write
    to disk."""
    expected_columns = [
        "date",
        "original_target",
        "observed_or_missing",
        "method",
        "reconstructed_median",
        "q05",
        "q95",
        "artificial_or_real_gap",
        "validation_status",
        "covariates_used",
    ]
    df = pd.DataFrame(rows)
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"build_export_table: missing expected column(s) {missing}")
    return df[expected_columns]


def export_csv(df: pd.DataFrame, filename: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    """Write `df` to `outputs_dir/filename` and return the path written.
    The only disk-writing function in this module; call it explicitly, not from
    inside a plotting or computation helper."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / filename
    df.to_csv(path, index=False)
    return path
