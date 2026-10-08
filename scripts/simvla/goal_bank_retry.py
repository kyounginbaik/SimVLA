"""Do not indefinitely reuse a partial-success goal that fails the full task."""

import json


def retire_failed_partial_goal(manager, env_ids, reason, *, failure_limit=2, planning_failure_limit=8):
    """Retire only matching partial-bank entries after repeated physical failures.

    A passed lift is not a passed delivery. Never remove a fully successful goal.
    Isolated planner failures are not evidence of a bad grasp. Repeated inability
    to execute the same *whole recipe*, however, must not lock collection forever.
    Original candidates remain available for fresh sampling and perturbation.
    """
    if failure_limit < 1 or planning_failure_limit < 1:
        raise ValueError("failure_limit must be positive")
    physical = reason in {
        "retry_termination", "post_lift_object_not_retained", "fail_but_done",
    }
    planning = reason.startswith(("plan_fail_r", "plan_fail_l", "A_r not reached", "motion_fail_l"))
    if manager.stage != "collect_full" or not (physical or planning):
        return []
    limit = failure_limit if physical else planning_failure_limit
    counts = getattr(manager, "_partial_goal_failures", {})
    manager._partial_goal_failures = counts
    key_of = lambda goal: json.dumps(goal, sort_keys=True, separators=(",", ":"))
    retired = []
    for env_id in env_ids:
        goal = manager.env_current.get(env_id)
        if goal is None:
            continue
        key = key_of(goal)
        if key in manager.full_keys or not (key in manager.right_keys or key in manager.left_keys):
            continue
        count_key = ("physical" if physical else "planning", key)
        counts[count_key] = counts.get(count_key, 0) + 1
        if counts[count_key] < limit:
            continue
        for side in ("right", "left"):
            keys = getattr(manager, f"{side}_keys")
            bank = getattr(manager, f"{side}_good_goals")
            if key in keys:
                bank[:] = [entry for entry in bank if key_of(entry) != key]
                keys.discard(key)
                retired.append((env_id, side, counts[count_key]))
    return retired
