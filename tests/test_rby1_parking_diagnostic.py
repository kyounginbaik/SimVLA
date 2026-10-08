import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parking", ROOT / "scripts/tools/make_rby1_parking_diagnostic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_offset_moves_base_left_without_changing_any_manipulation_step():
    goal = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    before = json.dumps(goal)
    result = module.with_lateral_parking_offset(goal)
    assert json.dumps(goal) == before
    assert result["goals"][0][1:] == goal["goals"][0][1:]
    new = result["goals"][0][0]["goal"]
    old = goal["goals"][0][0]["goal"]
    assert new[0] == pytest.approx(old[0]-.15)
    assert new[1:] == pytest.approx(old[1:])
    assert result["run_config"] == goal["run_config"]
