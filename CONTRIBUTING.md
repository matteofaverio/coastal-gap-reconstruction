# Contributing

## Environment setup

```bash
git clone https://github.com/matteofaverio/coastal-gap-reconstruction
cd coastal-gap-reconstruction
uv sync --extra notebooks --extra test
```

This resolves and installs the exact versions recorded in `uv.lock` (the same
environment CI uses). Prefix commands with `uv run` (e.g. `uv run pytest`,
`uv run jupyter lab ...`), or `source .venv/bin/activate` once and run them
directly. If `uv` is unavailable, `python3 -m venv .venv && source
.venv/bin/activate && pip install -e ".[notebooks,test]"` works but is not
version-pinned and may resolve different dependency versions than CI.

This does not install TS-ICL/torch. For the live TS-ICL demo environment,
see `demo/README.md` (`bash demo/run_demo.sh`, a separate, isolated,
informally-pinned venv -- not covered by `uv.lock`, see `demo/README.md` for
why).

## Tests

```bash
uv run pytest tests/
```

`tests/test_notebooks_smoke.py` executes every notebook end to end;
`tests/test_demo_smoke.py` executes the demo notebook (falls back to cached
TS-ICL predictions if `tsicl` is not installed).

## Notebook execution

Open any notebook with `jupyter lab notebooks/<name>.ipynb` after the
install above. See `notebooks/README.md` for the recommended reading order
and per-notebook execution status.

## Formatting/linting

```bash
ruff check .
```

Configuration lives in `pyproject.toml` under `[tool.ruff]`. Keep changes
mechanical (import order, obvious lint fixes) — do not run a full
reformatter across files with no other changes in the same commit.

## LaTeX compilation

Each document under `manuscript/` compiles independently with
[Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd manuscript/report && tectonic main.tex
cd manuscript/presentation && tectonic main.tex
cd manuscript/presentation_colleagues_es && tectonic main.tex
cd manuscript/poster && tectonic main.tex
```

## Adding a new target/sensor case

Follow `notebooks/09_adapting_the_workflow_to_a_new_sensor.ipynb` as a
checklist, and `notebooks/10_oxygen_case_study.ipynb` as a worked example.
The pipeline functions in `src/coastal_gap_reconstruction/` take
`target_col`/`eligible_col` arguments rather than hardcoding chlorophyll's
column names — do not inherit chlorophyll-specific thresholds (event
definition, gap-length range, eligibility rule) by default. Each new case
should set these from its own missingness structure -- see
`docs/methodology/target_and_gap_construction.md` for the questions to
answer for chlorophyll and oxygen, as a template for a new target.

## Data licensing

Data, results, and figures under `data_public/`, `results_public/`, and
`figures/` are **not** MIT licensed — see `DATA_LICENSE_AND_ATTRIBUTION.md`.
Do not add new data files without confirming their license and required
attribution first.

## What not to commit

- Private/absolute local paths or references to non-public handoff
  documents.
- Raw/large data files — only the curated `data_public/`/`results_public/`
  tables belong in this repository.
- Model checkpoints (TS-ICL's checkpoint is downloaded and cached locally by
  `demo/run_demo.sh`, never committed).
- Generated LaTeX auxiliary files (`.aux`, `.log`, `.synctex.gz`, etc. — see
  `.gitignore`) or local virtual environments.

## Pull requests

Keep PRs scoped to one concern (a notebook fix, a doc correction, a
dependency bump). Include the command you ran to verify the change (test
output, notebook execution, or a compiled PDF) in the PR description.
