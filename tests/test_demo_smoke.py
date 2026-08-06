"""Smoke test for the live demo (`demo/`).

By default this only does fast, static checks and, if the demo has already
been run (`demo/gap_reconstruction_walkthrough_executed.ipynb` exists, e.g.
because `bash demo/run_demo.sh` was run manually), that it contains no
error-type cell outputs and that TS-ICL ran live (not the cached fallback).

To actually (re-)run the full demo -- setting up `environments/tsicl/`,
downloading the TS-ICL checkpoint on first use, and executing the notebook
-- set `RUN_DEMO_SMOKE=1`. This is opt-in
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
    DEMO_DIR / "outputs" / "demo_reconstruction_results.csv",
]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_no_error_outputs(nb: dict) -> None:
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"Executed demo notebook contains an error output in a "
                f"{cell.get('cell_type')} cell: "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


def _tsicl_live_status(nb: dict) -> bool | None:
    """Look at the printed status from the TS-ICL loading cell. Returns True
    if it reports live, False if it reports cached fallback, None if not
    found (e.g. the notebook hasn't reached that cell)."""
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if "TS-ICL loaded live" in text:
                return True
            if "TS-ICL not available live" in text:
                return False
    return None


# ---------------------------------------------------------------------------
# Static source checks -- run unconditionally, no execution required.
# ---------------------------------------------------------------------------


def test_run_demo_script_exists_and_is_executable() -> None:
    assert RUN_SCRIPT.exists(), "demo/run_demo.sh is missing"
    assert os.access(RUN_SCRIPT, os.X_OK), "demo/run_demo.sh is not executable (chmod +x)"


# ---------------------------------------------------------------------------
# Checks against an already-executed notebook (fast path: skipped if absent).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not EXECUTED_NOTEBOOK.exists(),
    reason=(
        "demo has not been run yet in this checkout -- run "
        "`bash demo/run_demo.sh` first, or set RUN_DEMO_SMOKE=1 to run it "
        "as part of this test"
    ),
)
class TestExistingDemoRun:
    @pytest.fixture(scope="class")
    def executed_nb(self) -> dict:
        return _load_notebook(EXECUTED_NOTEBOOK)

    def test_no_errors(self, executed_nb: dict) -> None:
        _assert_no_error_outputs(executed_nb)

    def test_expected_output_files_exist(self) -> None:
        for f in EXPECTED_OUTPUT_FILES:
            assert f.exists(), f"expected demo output missing: {f}"

    def test_tsicl_status_reported(self, executed_nb: dict) -> None:
        status = _tsicl_live_status(executed_nb)
        assert status is not None, "Could not find a TS-ICL live/cached status message in the executed notebook"

    def test_tsicl_ran_live_not_cached(self, executed_nb: dict) -> None:
        """In a normal verification run (internet available, standard
        environment) TS-ICL must run live -- the cached fallback exists only
        for presentation-day contingency, not routine use."""
        status = _tsicl_live_status(executed_nb)
        assert status is True, (
            "TS-ICL used the cached fallback in this run. That is a valid "
            "contingency path, but a normal verification run is expected to "
            "have live internet/environment access and should exercise the "
            "live path -- investigate before treating this run as a pass."
        )

    def test_real_gap_section_has_no_score(self, executed_nb: dict) -> None:
        """Section 10 (real gap) must never compute or display a MAE -- real
        gaps have no withheld ground truth, so any score there would be
        fabricated."""
        for cell in executed_nb["cells"]:
            src = "".join(cell.get("source", []))
            if "real_gap" in src and "load_real_gap_example" in src:
                assert "mean_absolute_error" not in src


# ---------------------------------------------------------------------------
# Opt-in full run.
# ---------------------------------------------------------------------------


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
    nb = _load_notebook(EXECUTED_NOTEBOOK)
    _assert_no_error_outputs(nb)
    for f in EXPECTED_OUTPUT_FILES:
        assert f.exists(), f"expected demo output missing: {f}"
