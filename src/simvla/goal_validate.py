"""Portable structural validation for generated simulator goal files."""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import skills as _skills  # Populate the portable registry.
from .skill_contract import REGISTRY, Bool, Choice, Float, PrimPath


class GoalValidationError(ValueError):
    """A generated goal file cannot be consumed safely."""


LEGACY_ACTIONS = {"N", "N_s", "A_l", "A_r", "A_b", "G_l", "G_r"}


def validate_goal(path: str | Path) -> dict:
    """Validate goal-file structure without importing Isaac Sim.

    Version 2 files are checked against the skill registry. Unversioned legacy
    files can only be checked structurally because their skill identity is
    encoded in historical numeric payloads; regenerate them to get full checks.
    """
    source = Path(path)
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise GoalValidationError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GoalValidationError(f"{source}: goal file must contain a JSON object")
    goals = data.get("goals")
    if not (isinstance(goals, list) and len(goals) == 1 and isinstance(goals[0], list)):
        raise GoalValidationError(f"{source}: 'goals' must be a one-item list containing the steps")
    if not goals[0]:
        raise GoalValidationError(f"{source}: goal sequence is empty")

    version = data.get("version", 1)
    if version not in (1, 2):
        raise GoalValidationError(f"{source}: unsupported goal version {version!r}")
    if version == 2:
        for index, step in enumerate(goals[0]):
            if not isinstance(step, dict):
                raise GoalValidationError(f"{source}: step {index} must be an object in version 2")
            skill_id = step.get("skill")
            # Explicit diagnostic waypoints use the executor's raw-pose branch.
            # It is not an authorable skill, so do not add a dummy declaration to
            # the shared registry merely to accept an already planned goal.
            if skill_id == "arm.pose":
                action = step.get("action")
                width = {"A_l": 7, "A_r": 7, "A_b": 14}.get(action)
                pose = step.get("goal")
                if (width is None or not isinstance(pose, list) or len(pose) != width
                        or not isinstance(step.get("params", {}), dict)
                        or step.get("params", {})
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in pose)):
                    raise GoalValidationError(
                        f"{source}: step {index} arm.pose requires an arm action, "
                        "empty params and finite authored pose values"
                    )
                for offset in range(0, width, 7):
                    norm = math.sqrt(sum(v * v for v in pose[offset + 3:offset + 7]))
                    if not math.isclose(norm, 1.0, abs_tol=1e-4):
                        raise GoalValidationError(
                            f"{source}: step {index} arm.pose quaternion must be unit length"
                        )
                continue
            spec = REGISTRY.get(skill_id)
            if spec is None:
                raise GoalValidationError(f"{source}: step {index} names unknown skill {skill_id!r}")
            action = step.get("action")
            if action not in spec.actions:
                raise GoalValidationError(
                    f"{source}: step {index} uses action {action!r} for {skill_id!r}; "
                    f"expected one of {list(spec.actions)}"
                )
            if not isinstance(step.get("params", {}), dict):
                raise GoalValidationError(f"{source}: step {index} 'params' must be an object")
            params = step.get("params", {})
            declared = {param.name: param for param in spec.params}
            unknown = sorted(set(params) - set(declared))
            if unknown:
                raise GoalValidationError(
                    f"{source}: step {index} has unknown params for {skill_id!r}: {unknown}"
                )
            for name, param in declared.items():
                if isinstance(param, PrimPath) and name not in params:
                    raise GoalValidationError(
                        f"{source}: step {index} is missing required prim path {name!r}"
                    )
                if name not in params:
                    continue
                value = params[name]
                valid = (
                    isinstance(param, PrimPath) and isinstance(value, str) and value.startswith("/")
                    or isinstance(param, Choice) and value in param.options
                    or isinstance(param, Float) and isinstance(value, (int, float)) and not isinstance(value, bool)
                    or isinstance(param, Bool) and isinstance(value, bool)
                )
                if not valid:
                    raise GoalValidationError(
                        f"{source}: step {index} has invalid value {value!r} for {name!r}"
                    )
            if "goal" not in step:
                raise GoalValidationError(f"{source}: step {index} has no 'goal' field")
            if spec.is_runtime != (step["goal"] is None):
                expected = "null" if spec.is_runtime else "an authored value"
                raise GoalValidationError(
                    f"{source}: step {index} goal for {skill_id!r} must be {expected}"
                )
        run_config = data.get("run_config")
        if isinstance(run_config, dict):
            for arm, action in (("r", "G_r"), ("l", "G_l")):
                index = run_config.get(f"sub_grasp_idx_{arm}")
                count = run_config.get(f"sub_good_goal_count_{arm}", 0)
                if (not isinstance(index, int) or isinstance(index, bool)
                        or not isinstance(count, int) or isinstance(count, bool)):
                    raise GoalValidationError(
                        f"{source}: run_config grasp index/count for arm {arm!r} must be integers"
                    )
                if count <= 0:
                    continue
                if index < 0 or index >= len(goals[0]):
                    raise GoalValidationError(
                        f"{source}: active {arm.upper()} grasp check index {index} is outside "
                        f"the {len(goals[0])}-step goal"
                    )
                step = goals[0][index]
                if (step.get("skill") != "gripper.set" or step.get("action") != action
                        or step.get("params", {}).get("grasp") is not True):
                    raise GoalValidationError(
                        f"{source}: active {arm.upper()} grasp check index {index} points to "
                        f"{step.get('skill')!r}/{step.get('action')!r}, not a closing {action} "
                        "step; regenerate the goal with the current task emitter"
                    )
        detail = "skill and action schemas checked"
    else:
        for index, step in enumerate(goals[0]):
            if not (isinstance(step, list) and len(step) == 2):
                raise GoalValidationError(
                    f"{source}: legacy step {index} must be [action, payload]"
                )
            if step[0] not in LEGACY_ACTIONS:
                raise GoalValidationError(
                    f"{source}: legacy step {index} uses unknown action {step[0]!r}"
                )
        detail = "structure checked; regenerate as version 2 for skill-level validation"

    return {"path": str(source), "version": version, "steps": len(goals[0]),
            "valid": True, "detail": detail}
