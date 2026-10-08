"""Compatibility import for the canonical :mod:`simvla.skills` registry.

Isaac Sim entry points historically import ``skills`` as a top-level module from this directory.
The declarations now live only in ``src/simvla/skills.py``. Re-exporting every non-dunder name
keeps private compatibility helpers such as ``_authored`` available to existing authoring code
while ensuring CPU tools and simulator processes use the same implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simvla import skills as _canonical  # noqa: E402

globals().update(
    {name: getattr(_canonical, name) for name in dir(_canonical) if not name.startswith("__")}
)
