import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/simvla"))
from nav_tuning import profile_for


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_final_yaw_override_is_explicit_and_does_not_change_bearing_gate(robot):
    baseline = profile_for(robot, {})
    adjusted = profile_for(robot, {"SIMVLA_NAV_FINAL_YAW": ".02"})
    assert adjusted.goal_final_yaw == .02
    assert adjusted.goal_reached_yaw == baseline.goal_reached_yaw
    assert profile_for(robot, {}) == baseline


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "4"])
def test_invalid_final_yaw_fails_closed(value):
    with pytest.raises(ValueError):
        profile_for("anubis", {"SIMVLA_NAV_FINAL_YAW": value})


def test_collector_uses_final_yaw_for_parking_not_initial_bearing():
    source = (Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py").read_text()
    assert 'aligned3 = torch.abs(yaw_err3) < goal_final_yaw' in source
    assert 'aligned0 = torch.abs(yaw_err0) < goal_reached_yaw' in source
    assert '_nav.goal_final_yaw if "SIMVLA_NAV_FINAL_YAW" in os.environ' in source


def test_turn_dynamics_overrides_preserve_parking_accuracy():
    before = profile_for("aiworker", {})
    after = profile_for("aiworker", {"SIMVLA_NAV_YAW_RATE_MAX": ".12",
                                     "SIMVLA_NAV_YAW_SLEW": ".006"})
    assert after.yaw_rate_max == .12 and after.yaw_slew == .006
    assert after.goal_final_yaw == before.goal_final_yaw
    assert after.goal_reached_distance == before.goal_reached_distance


@pytest.mark.parametrize("name", ["SIMVLA_NAV_YAW_RATE_MAX", "SIMVLA_NAV_YAW_SLEW"])
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-.1", "1.1"])
def test_invalid_turn_dynamics_rejected(name, value):
    with pytest.raises(ValueError):
        profile_for("aiworker", {name: value})
