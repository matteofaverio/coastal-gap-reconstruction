"""Smoke test for the live demo (`demo/`).

By default this only does fast, static checks: source-code hygiene (no
`plt.savefig`, no external figure directory) and, if the demo has already
been run (`demo/gap_reconstruction_walkthrough_executed.ipynb` exists, e.g.
because `bash demo/run_demo.sh` was run manually), that it contains no
error-type cell outputs, that every plotting cell produced an inline image,
and that TS-ICL ran live (not the cached fallback).

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

# Sections that must each produce at least one inline figure (Section 1 is the
# only exception with two plotting cells; every other numbered section has
# exactly one). Counted as a minimum, not an exact match, so adding an extra
# explanatory plot later doesn't break this test.
MIN_EXPECTED_INLINE_FIGURES = 12

EXPECTED_OUTPUT_FILES = [
    DEMO_DIR / "outputs" / "demo_reconstruction_results.csv",
]

# These are the only files a fresh demo run is allowed to produce under
# demo/outputs/ -- anything else (in particular any image file) would mean a
# plotting function started saving to disk again.
ALLOWED_OUTPUT_FILENAMES = {
    "demo_reconstruction_results.csv",
    # One-off diagnostic audit from demo/search_demo_gap.py (not produced by a
    # normal demo run, but committed alongside it -- see that script's
    # docstring for the selection methodology).
    "demo_gap_selection_audit.csv",
}


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


def _count_inline_images(nb: dict) -> int:
    count = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                count += 1
    return count


def _notebook_code_source(nb: dict) -> str:
    return "".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


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


def _find_calls(source: str, method_name: str) -> list[int]:
    """Return line numbers of actual `<something>.<method_name>(...)` calls in
    `source`, parsed via `ast` so docstrings/comments mentioning the method
    name (e.g. explaining that it must NOT be called) don't trigger a false
    positive."""
    import ast

    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
        ):
            hits.append(node.lineno)
    return hits


def test_no_savefig_anywhere_in_demo_code() -> None:
    """Figures must stay inline-only: no plotting function or notebook cell
    may write an image file to disk."""
    offenders = {}
    for path in (DEMO_DIR / "src").glob("*.py"):
        hits = _find_calls(path.read_text(), "savefig")
        if hits:
            offenders[str(path)] = hits

    nb = _load_notebook(DEMO_DIR / "gap_reconstruction_walkthrough.ipynb")
    code_source = _notebook_code_source(nb)
    hits = _find_calls(code_source, "savefig")
    if hits:
        offenders["gap_reconstruction_walkthrough.ipynb"] = hits

    assert not offenders, f"Found actual .savefig(...) call(s) in: {offenders}"


def test_no_external_figures_directory() -> None:
    assert not (DEMO_DIR / "outputs" / "figures").exists(), (
        "demo/outputs/figures/ must not exist -- figures are inline-only in the notebook"
    )


def test_plotting_functions_do_not_call_show_or_savefig() -> None:
    """`src/plotting.py` functions must return (fig, axes) without displaying
    or saving -- that responsibility belongs to the notebook cell."""
    plotting_source = (DEMO_DIR / "src" / "plotting.py").read_text()
    assert not _find_calls(plotting_source, "show")
    assert not _find_calls(plotting_source, "savefig")


def test_notebook_calls_show_after_every_plotting_function() -> None:
    nb = _load_notebook(DEMO_DIR / "gap_reconstruction_walkthrough.ipynb")
    plot_calls = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "pl.plot_" in src:
            plot_calls += 1
            assert "plt.show()" in src, f"Cell calling a pl.plot_* function without plt.show():\n{src}"
    assert plot_calls >= MIN_EXPECTED_INLINE_FIGURES


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

    def test_no_unexpected_files_under_outputs(self) -> None:
        outputs_dir = DEMO_DIR / "outputs"
        actual = {p.name for p in outputs_dir.iterdir() if p.is_file()}
        unexpected = actual - ALLOWED_OUTPUT_FILENAMES
        assert not unexpected, f"Unexpected file(s) under demo/outputs/: {unexpected}"

    def test_inline_figures_present(self, executed_nb: dict) -> None:
        n_images = _count_inline_images(executed_nb)
        assert n_images >= MIN_EXPECTED_INLINE_FIGURES, (
            f"Expected at least {MIN_EXPECTED_INLINE_FIGURES} inline figures, found {n_images}"
        )

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
        """Section 10 (real gap) must never compute or display a MAE."""
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
