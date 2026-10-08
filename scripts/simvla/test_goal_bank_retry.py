import json
from types import SimpleNamespace

from goal_bank_retry import retire_failed_partial_goal


def manager():
    goal = [["A_r", [1., 2., 3.]]]
    key = json.dumps(goal, sort_keys=True, separators=(",", ":"))
    return SimpleNamespace(stage="collect_full", env_current={0: goal},
                           right_good_goals=[goal], left_good_goals=[],
                           right_keys={key}, left_keys=set(), full_keys=set())


def test_repeated_physical_failure_retires_only_partial_bank():
    m = manager()
    assert retire_failed_partial_goal(m, [0], "retry_termination") == []
    assert m.right_good_goals
    assert retire_failed_partial_goal(m, [0], "retry_termination") == [(0, "right", 2)]
    assert not m.right_good_goals and not m.right_keys
    assert m.env_current[0]  # original script remains available; no destructive mutation


def test_planner_and_timeout_failures_do_not_retire_grasps():
    m = manager()
    for _ in range(4):
        for reason in ("timeout", "plan_fail_r", "reset"):
            assert retire_failed_partial_goal(m, [0], reason) == []
    assert m.right_good_goals


def test_full_success_and_unbanked_candidates_are_protected():
    m = manager()
    m.full_keys.update(m.right_keys)
    for _ in range(4):
        assert retire_failed_partial_goal(m, [0], "retry_termination") == []
    assert m.right_good_goals
    m.full_keys.clear()
    m.env_current[0] = [["A_r", [4., 5., 6.]]]
    assert retire_failed_partial_goal(m, [0], "retry_termination", failure_limit=1) == []
    assert m.right_good_goals


def test_final_bank_is_not_pruned_after_random_failure():
    m = manager()
    m.stage = "final_only"
    assert retire_failed_partial_goal(m, [0], "retry_termination", failure_limit=1) == []
    assert m.right_good_goals


def test_unreachable_partial_recipe_does_not_lock_bulk_collection_forever():
    m = manager()
    reason = "plan_fail_r at step 1 (arm.grasp), status=IK_FAIL"
    for _ in range(7):
        assert retire_failed_partial_goal(m, [0], reason) == []
    assert retire_failed_partial_goal(m, [0], reason) == [(0, "right", 8)]
    assert not m.right_good_goals
    assert m.env_current[0]


def test_physical_and_planning_failure_budgets_are_separate():
    m = manager()
    for _ in range(7):
        assert retire_failed_partial_goal(m, [0], "plan_fail_r") == []
    assert retire_failed_partial_goal(m, [0], "fail_but_done") == []
    assert m.right_good_goals
