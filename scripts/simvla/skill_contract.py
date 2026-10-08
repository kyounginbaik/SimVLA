"""Compatibility import for the canonical :mod:`simvla.skill_contract` module.

Simulator scripts historically import ``skill_contract`` as a top-level module. Keep that
interface while making ``src/simvla`` the single source of truth for the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from simvla.skill_contract import *  # noqa: F401,F403,E402
