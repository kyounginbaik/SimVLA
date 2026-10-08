import json

import pytest

from simvla.goal_validate import GoalValidationError, validate_goal


def test_goal_preflight_rejects_active_grasp_index_that_points_at_a_release(tmp_path):
    data = _v2_fixture()
    data["goals"][0][4]["params"]["grasp"] = False
    data["goals"][0][4]["goal"] = False
    data["run_config"]["sub_grasp_idx_l"] = 4
    data["run_config"]["sub_good_goal_count_l"] = 1
    path = tmp_path / "stale-index.json"
    path.write_text(json.dumps(data))

    with pytest.raises(GoalValidationError, match="not a closing G_l step"):
        validate_goal(path)


def test_goal_preflight_accepts_current_object_close_checkpoint(tmp_path):
    data = _v2_fixture()
    data["run_config"]["sub_good_goal_count_l"] = 1
    path = tmp_path / "valid-index.json"
    path.write_text(json.dumps(data))

    assert validate_goal(path)["valid"] is True


def _v2_fixture():
    close = {"skill": "gripper.set", "action": "G_l", "params": {"grasp": True}, "goal": True}
    return {
        "version": 2,
        "goals": [[close.copy() for _ in range(5)]],
        "run_config": {
            "sub_grasp_idx_r": 999, "sub_good_goal_count_r": 0,
            "sub_grasp_idx_l": 2, "sub_good_goal_count_l": 0,
        },
    }
