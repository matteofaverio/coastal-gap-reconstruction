"""The oxygen real-gap publication contract.

**Deliberately much thinner than `experiments.chlorophyll.real_gap_contract`.**
The private publication audit
(`reports/advanced_code_publication_audit/AUTHORITATIVE_IMPLEMENTATIONS.csv`
row 22) classified the oxygen real-gap candidate generator as `MISSING` --
no dedicated script producing a by-class real-gap summary was located, and
the per-gap oxygen real-gap inventory such a summary would aggregate from
was also not found as a tracked file. This module therefore covers **real-
gap inventory only** -- oxygen has no published reconstruction-candidate
artifact of any kind, and this module must not be extended to imply one
exists. If a genuine oxygen real-gap candidate generator is later located
or built, it gets its own contract entry then, not a placeholder here.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PUBLIC_DIR = REPO_ROOT / "data" / "oxygen"
DAILY_TARGET_PATH = DATA_PUBLIC_DIR / "oxygen_daily_target.csv"
REAL_GAP_INVENTORY_BY_CLASS_PATH = DATA_PUBLIC_DIR / "oxygen_real_gap_inventory_by_class.csv"

TARGET_COLUMN = "oxygen_mean_mgL"
ELIGIBLE_COLUMN = "eligible_ge_18"
DATE_COLUMN = "date"

# No reconstruction-candidate artifact exists for oxygen real gaps -- this
# constant is the single source of truth other modules/tests should check
# rather than each re-deriving "oxygen has no candidates" independently.
OXYGEN_REAL_GAP_RECONSTRUCTION_CANDIDATES_EXIST = False
OXYGEN_REAL_GAP_STATUS = (
    "inventory_only -- no authoritative reconstruction-candidate generator or output exists "
    "for oxygen real gaps (private audit classification: MISSING). This package covers "
    "artificial-gap model evaluation (experiments.oxygen.benchmark_contract) and real-gap "
    "inventory/classification only."
)

__all__ = [
    "DATA_PUBLIC_DIR", "DAILY_TARGET_PATH", "REAL_GAP_INVENTORY_BY_CLASS_PATH",
    "TARGET_COLUMN", "ELIGIBLE_COLUMN", "DATE_COLUMN",
    "OXYGEN_REAL_GAP_RECONSTRUCTION_CANDIDATES_EXIST", "OXYGEN_REAL_GAP_STATUS",
]
