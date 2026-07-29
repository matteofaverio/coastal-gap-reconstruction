# Notebooks

Numbered, runnable notebooks covering the full pipeline for both case
studies. `tests/test_notebooks_smoke.py` executes every notebook marked
"executable" below on every CI run.

## Core reading path

1. [`../demo/gap_reconstruction_walkthrough.ipynb`](../demo/gap_reconstruction_walkthrough.ipynb) — visual, live, includes a real TS-ICL run.
2. `01_target_and_gap_audit.ipynb` — target construction and missingness audit.
3. `02_artificial_gap_validation.ipynb` — validation protocol and gap pool.
4. `07_benchmark_comparison_and_diagnostics.ipynb` — full method comparison.
5. `10_oxygen_case_study.ipynb` — transfer to a second sensor.
6. `09_adapting_the_workflow_to_a_new_sensor.ipynb` — checklist to extend to a third.

## Full index

| Notebook | Purpose | Executes locally | Uses cached public results | Audience | Reading path role |
|---|---|---|---|---|---|
| `01_target_and_gap_audit.ipynb` | Coverage/missingness audit for the daily chlorophyll target | yes | no | anyone | core |
| `02_artificial_gap_validation.ipynb` | Validation protocol and artificial-gap pool | yes | no | anyone | core |
| `03_baselines.ipynb` | Climatology/persistence/interpolation, scored | yes | no | anyone | depth |
| `04_engineered_tabular_models.ipynb` | External-predictor tabular models | yes | no | ML-focused | depth |
| `05_gap_edge_residual_models.ipynb` | Gap-edge residual correction models | partial* | yes | ML-focused | depth |
| `06_tsicl_zero_shot_imputation.ipynb` | Real TS-ICL API usage template | requires TS-ICL env | no | ML-focused | depth |
| `07_benchmark_comparison_and_diagnostics.ipynb` | Full cross-method benchmark | yes | no | anyone | core |
| `08_real_gap_candidate_reconstructions.ipynb` | Applying validated methods to real gaps | yes | no | anyone | depth |
| `09_adapting_the_workflow_to_a_new_sensor.ipynb` | Markdown checklist for a new sensor/target | n/a (markdown) | n/a | maintainers | core |
| `10_oxygen_case_study.ipynb` | Case Study 2: oxygen, worked result of notebook 9 | yes | no | anyone | core |

\* Notebook 05 loads and visualizes the released public benchmark tables and
figure; the full gap-edge residual model training pipeline itself is not
republished (see the notebook's opening cell for why). It explains and
visualizes the model family without reproducing the private training run.

Notebook 06 is a template for the real `TSICL()` API used elsewhere in this
repository — see `demo/gap_reconstruction_walkthrough.ipynb` for the full
executable, visual, multi-configuration comparison, and `demo/README.md` for
the tested installation route.
