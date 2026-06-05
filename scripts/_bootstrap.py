"""Make ``localguard`` importable when running scripts directly.

Every script imports this first so that ``python scripts/foo.py`` works without
requiring ``pip install -e .`` or setting PYTHONPATH manually.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
