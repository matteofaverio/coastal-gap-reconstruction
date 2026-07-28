"""Smoke tests for the public notebooks.

These do NOT check numeric correctness -- only that each notebook executes
top to bottom without raising an exception, against the packages declared
in `pyproject.toml`'s `notebooks` extra (`pip install -e ".[notebooks]"`)
plus `nbconvert`/`ipykernel`.

Notebook 06 (`06_tsicl_zero_shot_imputation.ipynb`) is included here
because, as shipped, its actual `tsicl` calls are commented out (it is a
usage template) -- so it executes cleanly without installing `tsicl`
itself. The live TS-ICL run lives in `demo/`, exercised separately by
`test_demo_smoke.py`.

Run with:

    pytest tests/test_notebooks_smoke.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# All numbered public notebooks are expected to execute cleanly against the
# public data in this repository (see README.md section 10 for the
# per-notebook executability notes this list is meant to match).
NOTEBOOKS = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

TIMEOUT_SECONDS = 180


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_cleanly(notebook_path: Path, tmp_path: Path) -> None:
    """Execute a notebook end to end via nbconvert; fail on any raised error."""
    output_dir = tmp_path / notebook_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}",
        "--output-dir",
        str(output_dir),
        str(notebook_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=notebook_path.parent,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS + 30,
    )
    assert result.returncode == 0, (
        f"{notebook_path.name} failed to execute cleanly.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
