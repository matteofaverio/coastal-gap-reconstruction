"""Generates gap_reconstruction_walkthrough.ipynb from plain Python/Markdown cell
strings (easier to author and diff than hand-written notebook JSON). Not needed
to run the demo -- only to regenerate the notebook file. Run from demo/:

    python3 _build_notebook.py
"""
import json

cells = []


def md(text, tags=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    if tags:
        cell["metadata"]["tags"] = tags
    cells.append(cell)


def code(text, tags=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }
    cells.append(cell)


# ===========================================================================
# 0. Title and workflow
# ===========================================================================

md("""\
# Reconstructing a gap step by step

- One coastal chlorophyll record, in-situ daily mean.
- One observed 14-day interval, temporarily hidden.
- Several reconstruction methods applied to the same gap.
- Withheld truth used only for evaluation, at the end.

**data → hide observed interval → reconstruct → compare → apply to a real gap**
""")

md("""\
Setup cells below (imports, path handling) are mechanical -- collapse them in
JupyterLab if you like. Scientific steps start at Section 1.
""", tags=["hide-input"])

code("""\
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
from src import demo_helpers as dh
from src import methods as mth
from src import plotting as pl

RNG_SEED = 0
np.random.seed(RNG_SEED)

GAP_START, GAP_END = "2017-04-21", "2017-05-04"
""", tags=["hide-input"])

# ===========================================================================
# 1. Inspect the available data
# ===========================================================================

md("""\
## 1. Inspect the available data

Which of these four series is the local sensor, and which come from external
products?
""")

code("""\
data = dh.load_demo_data()
fig, axes = pl.plot_available_data(data, GAP_START, GAP_END)
plt.show()
""")

code("""\
dh.data_summary(data)
""")

md("""\
Panel A is the only series measured at this station. Panels B-D are
external/satellite and reanalysis products, included as candidate covariates.
The shaded interval is the demonstration window used for the rest of this
notebook.
""")

# ===========================================================================
# 2. Select an observed interval
# ===========================================================================

md("""\
## 2. Select an observed interval

The 14-day interval below is fully observed right now -- that is what makes
it usable for validation: the true values exist, so a reconstruction can
later be scored against them.
""")

code("""\
fig, ax = pl.plot_selected_interval(data, GAP_START, GAP_END)
plt.show()
""")

code("""\
window = data[(data["date"] >= GAP_START) & (data["date"] <= GAP_END)]
{
    "start_date": GAP_START,
    "end_date": GAP_END,
    "length_days": len(window),
    "target_completeness": float(window["chl_mean"].notna().mean()),
    "satellite_proxy_completeness": float(window["chl_satellite_proxy_log10"].notna().mean()),
    "wind_completeness": float(window["wind_spd_ms"].notna().mean()),
    "sst_completeness": float(window["sst_primary_degC"].notna().mean()),
}
""")

# ===========================================================================
# 3. Create the artificial gap
# ===========================================================================

md("""\
## 3. Create the artificial gap

The values above still exist in memory (kept as `truth`, for scoring only).
Every method below sees only the masked series in panel B.

*This interval was selected automatically from observed 14-day candidates to
provide a clear demonstration of the workflow; it is an illustrative example,
not the full benchmark.*
""")

code("""\
gap = dh.create_artificial_gap(data, start=GAP_START, end=GAP_END)
fig, axes = pl.plot_gap_creation(gap)
plt.show()
""")

md("""\
The target is hidden; the satellite proxy (and wind, SST -- not shown here)
remain fully available inside the gap. Every method from this point on
receives the exact same masked series.
""")

# ===========================================================================
# 4. Simple baselines
# ===========================================================================

md("""\
## 4. Simple baselines

Persistence, climatology, and linear interpolation: no covariates, no fitting
beyond a formula.
""")

code("""\
full_record = dh.load_full_record()
baseline_results = mth.run_baselines(gap, full_record)
fig, axes = pl.plot_baseline_reconstructions(gap, baseline_results)
plt.show()
""")

md("""\
Simple methods provide the reference that more complex models must improve
upon.
""")

# ===========================================================================
# 5. Gaussian process
# ===========================================================================

md("""\
## 5. Gaussian process

target history → temporal covariance model → predictive distribution
""")

code("""\
gp_result = mth.run_gaussian_process(gap)
fig, ax = pl.plot_gp_reconstruction(gap, gp_result)
plt.show()
""")

md("""\
The GP is fit on the local observed context only (no covariates) and returns
a predictive mean plus a q05-q95 interval -- a genuine predictive
distribution, not a fixed-parameter confidence interval. It is fit fresh for
every gap (Matern 3/2 kernel, white-noise term); see `src/methods.py:run_gaussian_process`
for the exact kernel.
""")

# ===========================================================================
# 6. External tabular model
# ===========================================================================

md("""\
## 6. External tabular model

What does a model that never sees the target's own recent history receive
instead?
""")

code("""\
tabular_result_preview = mth.run_external_tabular(gap, full_record)
fig, axes = pl.plot_tabular_inputs(gap, tabular_result_preview.extra["feature_table"])
plt.show()
""")

md("""\
daily covariates → calendar terms, lags, rolling summaries → one feature
vector per day → HistGradientBoosting → chlorophyll estimate
""")

code("""\
tabular_result_preview.extra["feature_table"].head()
""")

code("""\
tabular_result = tabular_result_preview  # already fit above
fig, ax = pl.plot_tabular_reconstruction(gap, tabular_result)
plt.show()
""")

md("""\
This model is fitted locally on covariates and calendar terms only -- it
never receives the target's recent trajectory as a sequence input, unlike
every other method in this notebook.
""")

# ===========================================================================
# 7. Gap-edge residual correction
# ===========================================================================

md("""\
## 7. Gap-edge residual correction

corrected reconstruction = interpolation + predicted correction
""")

code("""\
edge_result = mth.run_gap_edge_residual(gap, full_record)
fig, axes = pl.plot_gap_edge_decomposition(gap, edge_result)
plt.show()
""")

md("""\
This formulation uses both sides of a closed historical gap -- the correction
model is trained on residuals at other gaps, using distance to the nearest
pre/post observation as a feature. It is not directly applicable to an
open-ended gap with no post-gap observation yet.
""")

# ===========================================================================
# 8. TS-ICL live, zero-shot
# ===========================================================================

md("""\
## 8. TS-ICL, live, zero-shot

TS-ICL (https://github.com/EDF-Lab/ts-icl) is a pretrained time-series
foundation model, called here with no training or fine-tuning on this data.
""")

code("""\
tsicl_model, tsicl_status = mth.load_tsicl()
if tsicl_status.live:
    print(f"TS-ICL loaded live -- device: {tsicl_status.device}, load time: {tsicl_status.load_time_s:.2f}s")
else:
    print(f"TS-ICL not available live ({tsicl_status.error}) -- using cached fallback predictions.")
""")

md("""\
Three configurations of the same pretrained model, run live below:

1. **target-only** -- no covariates;
2. **target + satellite chlorophyll proxy**;
3. **target + wind and SST** (physical bundle).

No local fitting or fine-tuning happens in any of the three.
""")

code("""\
fig, axes = pl.plot_tsicl_inputs(gap)
plt.show()
""")

code("""\
if tsicl_status.live:
    tsicl_results = {
        "tsicl_target_only": mth.run_tsicl(tsicl_model, gap, None, [], "tsicl_target_only"),
        "tsicl_satellite_proxy": mth.run_tsicl(
            tsicl_model, gap,
            gap.full_series[["chl_satellite_proxy_log10"]].to_numpy(dtype=np.float32),
            ["chl_satellite_proxy_log10"], "tsicl_satellite_proxy",
        ),
        "tsicl_physical_bundle": mth.run_tsicl(
            tsicl_model, gap,
            gap.full_series[["wind_spd_ms", "sst_primary_degC"]].to_numpy(dtype=np.float32),
            ["wind_spd_ms", "sst_primary_degC"], "tsicl_physical_bundle",
        ),
    }
else:
    tsicl_results = {
        arm: mth.MethodResult(f"tsicl_{arm}", mth.load_cached_tsicl_predictions(dh.DATA_DIR, arm), runtime_s=float("nan"))
        for arm in ["target_only", "satellite_proxy", "physical_bundle"]
    }
    tsicl_results = {f"tsicl_{k}": v for k, v in tsicl_results.items()}

fig, axes = pl.plot_tsicl_reconstructions(gap, tsicl_results)
plt.show()
""")

md("""\
Target-only is a complete, valid configuration on its own -- covariates are
optional, not a requirement. Adding a covariate channel changes what
information reaches the same pretrained model; it does not retrain it.
""")

# ===========================================================================
# 9. Compare all methods
# ===========================================================================

md("""\
## 9. Compare all methods

Six methods, identical axes, one illustrative gap.
""")

code("""\
all_results = {
    **baseline_results,
    "gaussian_process": gp_result,
    "external_tabular": tabular_result,
    "gap_edge_residual": edge_result,
    **tsicl_results,
}
selected = ["linear_interpolation", "gaussian_process", "external_tabular",
            "gap_edge_residual", "tsicl_target_only", "tsicl_satellite_proxy"]
fig, axes = pl.plot_method_comparison(gap, all_results, selected)
plt.show()
""")

code("""\
mae_by_method = {name: dh.mean_absolute_error(r.prediction, gap.truth) for name, r in all_results.items()}
runtime_by_method = {name: r.runtime_s for name, r in all_results.items()}
fig, ax = pl.plot_mae_bars(mae_by_method, runtime_by_method)
plt.show()
""")

code("""\
pd.DataFrame({"runtime_s": runtime_by_method}).sort_values("runtime_s")
""")

md("""\
This ranking is one illustrative gap, not the full benchmark -- the
repository's actual validation scores many gaps across lengths and seasons
(see the root `README.md`). Do not read this single ranking as final.
""")

# ===========================================================================
# 10. Apply the method to a real gap
# ===========================================================================

md("""\
## 10. Apply the method to a real gap

A genuinely missing interval: no withheld truth exists here.
""")

code("""\
real_gap = dh.load_real_gap_example()
if tsicl_status.live:
    real_result = mth.run_tsicl_real_gap(tsicl_model, real_gap)
else:
    real_result = mth.MethodResult(
        "tsicl_satellite_proxy", mth.load_cached_tsicl_real_gap_predictions(dh.DATA_DIR), runtime_s=float("nan")
    )
fig, axes = pl.plot_real_gap_reconstruction(real_gap, real_result)
plt.show()
""")

code("""\
{
    "start_date": "2015-07-01",
    "end_date": "2015-07-14",
    "gap_length_days": 14,
    "method": "tsicl_satellite_proxy",
    "covariates_used": real_result.covariates_used,
    "validation_status": "candidate_not_validation_evidence",
}
""")

# ===========================================================================
# 11. Export and reuse
# ===========================================================================

md("""\
## 11. Export and reuse
""")

code("""\
export_rows = []
for name, result in all_results.items():
    for _, row in result.prediction.iterrows():
        truth_row = gap.truth.loc[gap.truth["date"] == row["date"], "truth"]
        export_rows.append({
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "original_target": float(truth_row.iloc[0]) if len(truth_row) else None,
            "observed_or_missing": "artificially_hidden",
            "method": name,
            "reconstructed_median": row["value"],
            "q05": row.get("q05"),
            "q95": row.get("q95"),
            "artificial_or_real_gap": "artificial",
            "validation_status": "single_illustrative_gap",
            "covariates_used": ",".join(result.covariates_used) if result.covariates_used else "none",
        })
for _, row in real_result.prediction.iterrows():
    export_rows.append({
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "original_target": None,
        "observed_or_missing": "really_missing",
        "method": "tsicl_satellite_proxy",
        "reconstructed_median": row["value"],
        "q05": row.get("q05"),
        "q95": row.get("q95"),
        "artificial_or_real_gap": "real",
        "validation_status": "candidate_not_validation_evidence",
        "covariates_used": "chl_satellite_proxy_log10",
    })

export_df = dh.build_export_table(export_rows)
export_path = dh.export_csv(export_df, "demo_reconstruction_results.csv")
print(f"Exported {len(export_df)} rows to {export_path.relative_to(dh.DEMO_ROOT)}")
export_df.head()
""")

md("""\
## Use your own series

Required columns: `date`, `target`. Optional: one or more time-aligned
covariate columns.

```python
my_data = pd.read_csv("my_series.csv", parse_dates=["date"])
target_col = "my_target_column"
covariate_cols = ["my_covariate_1", "my_covariate_2"]  # optional, can be empty

gap = dh.create_artificial_gap(my_data, start="2020-01-01", end="2020-01-14",
                                target_column=target_col)
```

See `demo/README.md` for the full adaptation checklist (units, transform
scale, minimum context, TS-ICL covariate shape).
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "TS-ICL (environments/tsicl)", "language": "python", "name": "tsicl-env"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("gap_reconstruction_walkthrough.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)

print(f"Wrote gap_reconstruction_walkthrough.ipynb with {len(cells)} cells")
