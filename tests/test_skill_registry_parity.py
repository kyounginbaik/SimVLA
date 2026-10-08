"""The simulator compatibility modules must expose the packaged skill registry."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_simulator_imports_use_canonical_registry():
    if not (ROOT / "scripts/simvla/skills.py").is_file():
        pytest.skip("portable sdist intentionally excludes simulator compatibility modules")
    script = """
import skill_contract
import skills
from simvla import skill_contract as canonical_contract
from simvla import skills as canonical_skills

assert skill_contract.REGISTRY is canonical_contract.REGISTRY
assert skills.REGISTRY is canonical_skills.REGISTRY
assert skills.register_planner is canonical_skills.register_planner
assert set(skill_contract.REGISTRY) == set(canonical_contract.REGISTRY)
print(len(skill_contract.REGISTRY))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts/simvla"), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0
