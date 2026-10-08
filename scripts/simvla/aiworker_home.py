"""Keep the AI Worker cuRobo home aligned with the live Isaac reset pose.

Pure Python so the invariant can be tested without importing Isaac Sim or cuRobo.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def sync_planner_home(robot_cfg: dict[str, Any], start: Mapping[str, float]) -> None:
    """Rewrite cuRobo's retract and locked lift from an Isaac joint state.

    AI Worker's lift is part of the URDF chain but locked during planning. A stale lock value
    changes the arm-base height in cuRobo, while a stale retract makes warmup validate a pose the
    simulator never uses. Both failures surface later as unreachable grasps or invalid starts.
    """
    kin = robot_cfg["kinematics"]
    cspace = kin["cspace"]
    names = list(cspace["joint_names"])
    missing = [name for name in names if name not in start]
    if missing:
        raise ValueError(f"AI Worker reset state is missing cuRobo joints: {missing}")
    if "lift_joint" not in names or "lift_joint" not in kin.get("lock_joints", {}):
        raise ValueError("AI Worker cuRobo config must list and lock lift_joint")

    retract = [float(start[name]) for name in names]
    if len(retract) != len(cspace.get("null_space_weight", [])):
        raise ValueError("AI Worker cuRobo cspace and null-space weights differ in length")
    if len(retract) != len(cspace.get("cspace_distance_weight", [])):
        raise ValueError("AI Worker cuRobo cspace and distance weights differ in length")

    cspace["retract_config"] = retract
    kin["lock_joints"]["lift_joint"] = float(start["lift_joint"])
