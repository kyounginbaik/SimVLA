import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import math
import pytest


def _settling():
    path = Path(__file__).parents[1] / "scripts/simvla/replay_targets.py"
    spec = importlib.util.spec_from_file_location("replay_targets_settling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay_settle_steps


@pytest.mark.parametrize("value", ["-1", "201", "1.0", "true", "", " 1", "１", None, 1])
def test_terminal_settling_rejects_invalid_budget(value):
    with pytest.raises(ValueError):
        _settling()(value, joint_replay=True)


def test_terminal_settling_requires_motor_control_and_defaults_off():
    assert _settling()("0", joint_replay=False) == 0
    assert _settling()("200", joint_replay=True) == 200
    with pytest.raises(ValueError, match="motor targets"):
        _settling()("1", joint_replay=False)


def _tracking():
    path = Path(__file__).parents[1] / "scripts/simvla/replay_targets.py"
    spec = importlib.util.spec_from_file_location("replay_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.base_tracking_correction


def _non_gripper_ids():
    path = Path(__file__).parents[1] / "scripts/simvla/replay_targets.py"
    spec = importlib.util.spec_from_file_location("replay_targets_hybrid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.non_gripper_joint_ids


def test_hybrid_replay_preserves_gripper_controller_ownership():
    assert _non_gripper_ids()(8, [[2, 4], [5, 7]]) == [0, 1, 3, 6]
    assert _non_gripper_ids()(8, [slice(2, 4), [6]]) == [0, 1, 4, 5, 7]


def test_hybrid_replay_keeps_full_width_action_manager_targets():
    source = (Path(__file__).parents[1] / "scripts/simvla/simvla_replay.py").read_text()
    assert "joint_targets = joint_targets[:, replay_joint_ids]" not in source
    assert "joint_ids=replay_joint_ids" in source


@pytest.mark.parametrize("groups", [[], [[]], [list(range(8))], [[-1]], [[8]], [[True]]])
def test_hybrid_replay_rejects_ambiguous_joint_ownership(groups):
    with pytest.raises(ValueError):
        _non_gripper_ids()(8, groups)


def test_base_tracking_zero_and_small_errors():
    correction = _tracking()
    pose = [1, 2, 1, 0, 0, 0]
    assert correction(pose, pose) == [0, 0, 0]
    np.testing.assert_allclose(correction([1.01, 1.995, 1, 0, 0, 0], pose), [.02, -.01, 0])


def test_base_tracking_limits_norm_and_wraps_yaw():
    correction = _tracking()
    def pose(yaw): return [0, 0, math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    np.testing.assert_allclose(correction(pose(-math.pi+.01), pose(math.pi-.01)), [0, 0, .04])
    result = correction([10, 10, 0, 0, 0, 1], pose(0))
    assert math.hypot(*result[:2]) == pytest.approx(.05)
    assert abs(result[2]) == .1


@pytest.mark.parametrize("pose", [[0]*6, [0]*5, [float("nan"), 0, 1, 0, 0, 0]])
def test_base_tracking_rejects_invalid_reference(pose):
    with pytest.raises(ValueError):
        _tracking()(pose, [0, 0, 1, 0, 0, 0])


def test_replay_holds_only_the_uncontrolled_lift_at_its_configured_home():
    path = Path(__file__).parents[1] / "scripts/simvla/replay_targets.py"
    spec = importlib.util.spec_from_file_location("replay_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=np.array([[0., -.3, 1.]])),
                            set_joint_position_target=lambda values, **kwargs: calls.append((values, kwargs)))
    module.hold_uncontrolled_lift(robot, None)
    assert not calls
    module.hold_uncontrolled_lift(robot, [1])
    np.testing.assert_array_equal(calls[0][0], [[-.3]])
    assert calls[0][1] == {"joint_ids": [1]}
