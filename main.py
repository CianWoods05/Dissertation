"""
main.py
-------
Top-level command-line entry point for the Rugby Union ML dissertation project.

This is a thin wrapper around ``src.experiments`` that lives at the project
root so users (and CI) can run::

    python main.py --task all

without worrying about ``-m src.experiments`` and module path details.

Examples
--------
    # Reproduce everything in the dissertation (sections 4–6)
    python main.py --task all --seed 42

    # Just rerun the cross-season evaluation (Section 4.5)
    python main.py --task cross-season

    # Classification comparison into a custom output folder
    python main.py --task classify --output-dir results/_tmp
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable whether main.py is run from the project root or not.
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments import main as _experiments_main  # noqa: E402


def main(argv=None) -> int:
    return _experiments_main(argv)


if __name__ == "__main__":
    sys.exit(main())
