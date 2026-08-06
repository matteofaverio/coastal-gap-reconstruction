# Notebooks

Six notebooks covering the full pipeline for both case studies, in reading
order. `tests/test_notebooks_smoke.py` executes every notebook end to end
on every CI run.

| Notebook | Purpose | Executes locally | Uses cached public results |
|---|---|---|---|
| [`01_data_and_gap_audit.ipynb`](01_data_and_gap_audit.ipynb) | Chlorophyll target coverage/missingness audit, then the artificial-gap validation pool that makes method comparison possible | yes | no |
| [`02_artificial_gap_validation.ipynb`](02_artificial_gap_validation.ipynb) | Runs the validation protocol for real: the three Model 0 baselines, scored against the pool | yes | no |
| [`03_classical_and_probabilistic_models.ipynb`](03_classical_and_probabilistic_models.ipynb) | External-predictor tabular models and gap-edge residual/probabilistic models | yes (small samples; full runs in `docs/reproducibility.md`) | partial |
| [`04_tsicl_and_covariates.ipynb`](04_tsicl_and_covariates.ipynb) | The real TS-ICL calling API, then the full cross-method benchmark comparison and covariate-mechanism ranking | yes (TS-ICL cells need `environments/tsicl/`, degrade gracefully otherwise) | partial |
| [`05_real_gap_candidates.ipynb`](05_real_gap_candidates.ipynb) | Real-gap inventory, candidate-method routing, and assembly, applied to the 128 naturally-occurring chlorophyll gaps | yes | yes |
| [`06_oxygen_case_study.ipynb`](06_oxygen_case_study.ipynb) | Case Study 2: adapting the workflow to oxygen, and the released oxygen benchmark result | yes | yes |

For the full, always-executable, visual, multi-configuration comparison
(target-only vs. +satellite chlorophyll vs. +wind/SST, on a real 14-day
gap, plotted against withheld ground truth, live TS-ICL run included), see
[`../demo/gap_reconstruction_walkthrough.ipynb`](../demo/gap_reconstruction_walkthrough.ipynb)
and `bash demo/run_demo.sh`.

## Frozen vs. newly-computed

Every notebook is explicit about which values it computes fresh (from the
public data tables) and which it loads from an already-released `results/`
table -- see `docs/evidence_and_limitations.md` for what evidential weight
each carries. None of these six notebooks runs a complete expensive
benchmark; see `docs/reproducibility.md` for the standard and optional
full-grid reproduction commands.

## Setup

```bash
uv sync --extra notebooks --extra test
uv run jupyter lab notebooks/01_data_and_gap_audit.ipynb
```

Notebooks 04 and the demo need the separate `environments/tsicl/`
environment for live TS-ICL inference (`docs/reproducibility.md`); every
other notebook needs only the core locked environment.
