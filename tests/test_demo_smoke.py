"""Smoke test for the live demo (`demo/`).

By default this only does a fast, static check: if the demo has already
been run (`demo/gap_reconstruction_walkthrough_executed.ipynb` exists,
e.g. because `bash demo/run_demo.sh` was run manually), verify it contains
no error-type cell outputs and that the expected output files exist.

To actually (re-)run the full demo -- creating `.venv_tsicl_demo/`,
installing `tsicl`/`torch`, downloading the TS-ICL checkpoint on first use,
and executing the notebook -- set `RUN_DEMO_SMOKE=1`. This is opt-in
because it needs internet access on first run and takes noticeably longer
than the rest of this test suite (see demo/README.md for tested timings).

    RUN_DEMO_SMOKE=1 pytest tests/test_demo_smoke.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"
EXECUTED_NOTEBOOK = DEMO_DIR / "gap_reconstruction_walkthrough_executed.ipynb"
RUN_SCRIPT = DEMO_DIR / "run_demo.sh"

EXPECTED_OUTPUT_FILES = [
    DEMO_DIR / "outputs" / "reconstruction_figure.png",
    DEMO_DIR / "outputs" / "demo_reconstruction_results.csv",
    DEMO_DIR / "outputs" / "runtime_summary.json",
]


def _assert_no_error_outputs(notebook_path: Path) -> None:
    nb = json.loads(notebook_path.read_text())
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"Executed demo notebook contains an error output in a "
                f"{cell.get('cell_type')} cell: "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


def test_run_demo_script_exists_and_is_executable() -> None:
    assert RUN_SCRIPT.exists(), "demo/run_demo.sh is missing"
    assert os.access(RUN_SCRIPT, os.X_OK), "demo/run_demo.sh is not executable (chmod +x)"


@pytest.mark.skipif(
    not EXECUTED_NOTEBOOK.exists(),
    reason=(
        "demo has not been run yet in this checkout -- run "
        "`bash demo/run_demo.sh` first, or set RUN_DEMO_SMOKE=1 to run it "
        "as part of this test"
    ),
)
def test_existing_demo_run_has_no_errors() -> None:
    _assert_no_error_outputs(EXECUTED_NOTEBOOK)
    for f in EXPECTED_OUTPUT_FILES:
        assert f.exists(), f"expected demo output missing: {f}"


@pytest.mark.skipif(
    os.environ.get("RUN_DEMO_SMOKE") != "1",
    reason="set RUN_DEMO_SMOKE=1 to actually (re-)run the full demo (installs a venv, needs internet on first run)",
)
def test_full_demo_run_succeeds() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"demo/run_demo.sh failed.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert EXECUTED_NOTEBOOK.exists()
    _assert_no_error_outputs(EXECUTED_NOTEBOOK)
    for f in EXPECTED_OUTPUT_FILES:
        assert f.exists(), f"expected demo output missing: {f}"
