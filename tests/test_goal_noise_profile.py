"""Target augmentation can be disabled explicitly without changing old defaults."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/simvla"))
from collector_profile import goal_noise_standard_deviations

NAMES = ("SIMVLA_GOAL_NAV_XY_STD_M", "SIMVLA_GOAL_NAV_YAW_STD_RAD",
         "SIMVLA_GOAL_ARM_XYZ_STD_M")


def test_defaults_preserve_historical_noise():
    assert goal_noise_standard_deviations({}) == (.03, .1, .01)


def test_nominal_targets_and_independent_overrides():
    assert goal_noise_standard_deviations(dict.fromkeys(NAMES, "0")) == (0., 0., 0.)
    assert goal_noise_standard_deviations({NAMES[0]: ".02"}) == (.02, .1, .01)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-.01", "4", "", "invalid", None])
def test_invalid_noise_fails_loudly(name, value):
    with pytest.raises(ValueError, match=name):
        goal_noise_standard_deviations({name: value})


def test_collector_uses_explicit_noise_profile_and_evidence_records_it():
    tree = ast.parse((ROOT / "scripts/simvla/simvla_gen.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == "goal_noise_standard_deviations")
    assert [n.id for n in assignment.targets[0].elts] == [
        "noise_xy_std", "noise_yaw_std", "noise_xyz_std"]
    source = (ROOT / "scripts/tools/release_evidence.py").read_text()
    assert all(f'"{name}"' in source for name in NAMES)
