"""Make the repository root importable so tests and notebooks can `import experiments...`
alongside the pip-installed `coastal_gap_reconstruction` package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
