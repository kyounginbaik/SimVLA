import json

import pytest

from simvla.goal_validate import GoalValidationError, validate_goal


def write_goal(tmp_path, action="A_l", pose=None, params=None):
    path = tmp_path / "goal.json"
    path.write_text(json.dumps({"version": 2, "goals": [[{
        "skill": "arm.pose", "action": action,
        "params": {} if params is None else params,
        "goal": [0., 0., 1., 1., 0., 0., 0.] if pose is None else pose,
    }]]}))
    return path


@pytest.mark.parametrize("action", ["A_l", "A_r", "A_b"])
def test_executor_pose_is_valid_without_authorable_dummy_skill(tmp_path, action):
    from simvla.skill_contract import REGISTRY
    pose = [0., 0., 1., 1., 0., 0., 0.] * (2 if action == "A_b" else 1)
    assert validate_goal(write_goal(tmp_path, action=action, pose=pose))["valid"]
    assert "arm.pose" not in REGISTRY


@pytest.mark.parametrize("kwargs", [
    {"action": "N_s"}, {"pose": [0.] * 6}, {"pose": [0.] * 7},
    {"pose": [0., 0., 1., 2., 0., 0., 0.]},
    {"pose": [float("nan"), 0., 1., 1., 0., 0., 0.]},
    {"pose": [True, 0., 1., 1., 0., 0., 0.]},
    {"params": {"unsupported": 1}}, {"params": []},
])
def test_invalid_executor_pose_is_rejected(tmp_path, kwargs):
    with pytest.raises(GoalValidationError, match="arm.pose"):
        validate_goal(write_goal(tmp_path, **kwargs))
