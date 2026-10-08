import copy
import json

import pytest

from select_goal_grasp import select_candidate, side_approach_indices


def _goal():
    return {
        "version": 2,
        "goals": [[
            {"action": "A_l", "goal": [[float(i), 0, 0, 1, 0, 0, 0] for i in range(3)]},
            {"action": "G_l", "goal": True},
        ]],
    }


def test_select_candidate_keeps_task_and_reduces_only_selected_arm_options():
    source = _goal()
    selected = select_candidate(source, "left", 2)
    assert selected["goals"][0][0]["goal"] == [[2.0, 0, 0, 1, 0, 0, 0]]
    assert selected["goals"][0][1] == source["goals"][0][1]
    assert source == _goal()


@pytest.mark.parametrize(
    ("arm", "index", "message"),
    [
        ("both", 0, "arm must"),
        ("left", -1, "nonnegative"),
        ("left", 3, "out of range"),
        ("left", 1, "finite"),
    ],
)
def test_select_candidate_rejects_invalid_requests(arm, index, message):
    goal = _goal()
    if index == 1:
        goal["goals"][0][0]["goal"][1][0] = float("nan")
    with pytest.raises(ValueError, match=message):
        select_candidate(goal, arm, index)


def test_select_candidate_requires_one_arm_step():
    goal = copy.deepcopy(_goal())
    goal["goals"][0].append(copy.deepcopy(goal["goals"][0][0]))
    with pytest.raises(ValueError, match="exactly one A_l"):
        select_candidate(goal, "left", 0)


def test_select_legacy_anubis_bank_preserves_other_steps_and_input():
    from pathlib import Path
    source = json.loads((Path(__file__).resolve().parents[2]
                        / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    before = copy.deepcopy(source)
    selected = select_candidate(source, "right", 10)
    assert selected["goals"][0][1][1] == [before["goals"][0][1][1][10]]
    assert selected["goals"][0][2:] == before["goals"][0][2:]
    selected["goals"][0][1][1][0][0] += 1
    assert source == before


def test_side_approach_rejects_downward_bodex_pose_and_keeps_horizontal_pose():
    down = [0, 0, 0, 1, 0, 0, 0]
    side = [0, 0, 0, 0.70710678, 0, 0.70710678, 0]
    assert side_approach_indices([down, side]) == [1]


def test_public_aiworker_goal_has_authored_side_approach_candidates():
    from pathlib import Path

    goal = json.loads((Path(__file__).resolve().parents[2] / "examples" / "goals"
                       / "Isaac-Kitchen-v813a-00.json").read_text())
    poses = next(step["goal"] for step in goal["goals"][0]
                 if step.get("action") == "A_l" and step.get("skill") == "arm.grasp")
    assert side_approach_indices(poses) == [1, 2, 5, 8, 9]
