import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/simvla"))
from collector_profile import track_loaded_home_reset


@pytest.mark.parametrize("holding,step,home,nav,expected", [
    (True, 20, True, False, True), (True, 20, False, True, True),
    (True, 20, False, False, False), (False, 20, True, False, False),
    (False, 20, False, True, False), (True, 0, True, False, False),
    (True, 1, False, True, False),
])
def test_loaded_home_motor_ownership_lifetime(holding, step, home, nav, expected):
    from collector_profile import loaded_home_hold_allowed
    assert loaded_home_hold_allowed(holding, step, is_home_reset=home, is_navigation=nav) is expected


def test_loaded_home_targets_survive_control_steps_but_are_pruned():
    import ast
    tree = ast.parse((Path(__file__).parents[1] / "scripts/simvla/simvla_gen.py").read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    for loop in loops:
        assert not any(isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "_loaded_home_motor_overrides" for t in n.targets)
                       for n in ast.walk(loop))
    assert "del _loaded_home_motor_overrides[i]" in ast.unparse(tree)


@pytest.mark.parametrize("robot,skill,holding,expected", [
    ("aiworker", "arm.reset", True, True),
    ("aiworker", "arm.reset", False, False),
    ("aiworker", "arm.grasp", True, False),
    ("rby1", "arm.reset", True, False),
    ("anubis", "arm.reset", True, False),
])
def test_only_loaded_aiworker_home_is_gated(robot, skill, holding, expected):
    assert track_loaded_home_reset(robot, skill, holding, "1") is expected
    assert not track_loaded_home_reset(robot, skill, holding, "0")


def test_invalid_setting_is_not_silently_enabled():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        track_loaded_home_reset("aiworker", "arm.reset", True, "yes")


def test_loaded_home_excludes_locked_lift_and_preserves_name_mapping():
    from collector_profile import loaded_home_arm_indices
    arm = [f"arm_l_joint{i}" for i in range(1, 8)]
    robot_ids, plan_columns = loaded_home_arm_indices(["lift_joint"] + arm[::-1],
                                                      ["head_joint1"] + arm)
    assert robot_ids == list(range(1, 8))
    assert plan_columns == list(range(7, 0, -1))


def test_loaded_home_rejects_incomplete_arm_plan():
    from collector_profile import loaded_home_arm_indices
    import pytest
    with pytest.raises(ValueError, match="exactly once"):
        loaded_home_arm_indices(["arm_l_joint1"], ["arm_l_joint1"])


def test_home_pose_survives_reuse_of_curobo_fk_buffers():
    import ast
    import copy

    class Buffer:
        def __init__(self, values):
            self.values = values

        def squeeze(self, axis):
            return self

        def clone(self):
            return copy.deepcopy(self)

    source = Path(__file__).parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(source.read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call)
                   and "_home_fk_l[" in ast.unparse(n.value)]
    assert len(assignments) == 2
    buffers = [Buffer([1., 2., 3.]), Buffer([1., 0., 0., 0.])]
    namespace = {"_home_fk_l": buffers}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec"), namespace)
    buffers[0].values[:] = [9., 9., 9.]
    buffers[1].values[:] = [0., 0., 0., 1.]
    assert namespace["goal_ee_pose_b_l"].values == [1., 2., 3.]
    assert namespace["goal_ee_quat_b_l"].values == [1., 0., 0., 0.]
