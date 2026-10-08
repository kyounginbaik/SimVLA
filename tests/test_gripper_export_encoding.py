import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def test_action_export_uses_binary_command_not_signed_motor_target():
    tree = ast.parse((ROOT / "source/isaaclab/isaaclab/managers/action_manager.py").read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "gripper_real" for t in n.targets))
    for motor in (-.04, .04, .65, 1.1):
        namespace = {"torch": np, "term": SimpleNamespace(raw_actions=np.array([[1.], [-1.]]),
                                                          processed_actions=np.full((2, 1), motor))}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "action_export", "exec"), namespace)
        np.testing.assert_allclose(namespace["gripper_real"], [[-1.6], [.1]])


@pytest.mark.parametrize("robot,names,positions", [
    ("rby1", ["gripper_finger_l1", "gripper_finger_r1"], [[-.04, 0.], [0., -.04]]),
    ("aiworker", ["gripper_l_joint1", "gripper_r_joint1"], [[0., 1.1002], [1.1002, 0.]]),
])
def test_recovery_exact_commands_and_measured_observation(robot, names, positions):
    spec = importlib.util.spec_from_file_location("recovery", ROOT / "scripts/tools/recover_gripper_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays = {"action__eef_pos": np.zeros((2, 20)), "obs__eef_pose": np.zeros((2, 20)),
              "joint_angles": np.array(positions)}
    commands = np.ones((3, 17))
    commands[1, 13] = -1
    commands[2, 12] = -1
    result = module.correct_arrays(arrays, commands, names, robot)
    np.testing.assert_allclose(result["action__eef_pos"][:, [9, 19]], [[-1.6, .1], [.1, -1.6]])
    np.testing.assert_allclose(result["obs__eef_pose"][:, 18:20], [[-1.6, .1], [.1, -1.6]])
    assert not arrays["action__eef_pos"].any()
    with pytest.raises(ValueError, match="shape"):
        module.correct_arrays(arrays, commands[1:], names, robot)
