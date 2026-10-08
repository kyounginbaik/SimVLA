# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot joint-name lists for the LeRobot feature schema.

Each entry in :data:`ROBOT_SCHEMAS` declares the column order of two arrays
that the SimVLA writer stages per timestep:

- ``joint_angles``: the observation feature ``joint_angles`` (no base joints).
- ``action_joint``: the action feature ``action.joint`` (full joint vector).

The order MUST match the order produced by the live articulation
(``data.joint_pos`` / the action manager's ``joint_pos`` action), not the
declaration order in the robot ``ArticulationCfg``. Isaac Sim reorders
joints to match the USD; capture the order from a real run before adding a
new robot here.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class _RobotSchema(TypedDict):
    joint_angles: List[str]
    action_joint: List[str]


ANUBIS_JOINTS: _RobotSchema = {
    "joint_angles": [
        "arm1_base_link_joint", "arm2_base_link_joint",
        "link11_joint", "link21_joint",
        "link12_joint", "link22_joint",
        "link13_joint", "link23_joint",
        "link14_joint", "link24_joint",
        "link15_joint", "link25_joint",
        "gripper1_joint", "gripper2_joint",
    ],
    "action_joint": [
        "base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint",
        "arm1_base_link_joint", "arm2_base_link_joint",
        "link11_joint", "link21_joint",
        "link12_joint", "link22_joint",
        "link13_joint", "link23_joint",
        "link14_joint", "link24_joint",
        "link15_joint", "link25_joint",
        "gripper1R_joint", "gripper1_joint",
        "gripper2R_joint", "gripper2_joint",
    ],
}

# RBY1 — order captured from data.joint_names in a live --robot rby1 run.
# action_joint = the full 29-name list verbatim.
# joint_angles = action_joint with these 5 names removed:
#   base_prismatic_x_joint, base_prismatic_y_joint, base_revolute_z_joint,
#   gripper_finger_l2, gripper_finger_r2.
# That matches what kitchen/mdp/observations.py:joint_angles actually emits
# (drops [:, :3] then masks the two secondary gripper fingers).
RBY1_JOINTS: _RobotSchema = {
    "joint_angles": [
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        "head_0",
        "left_arm_0", "right_arm_0",
        "head_1",
        "left_arm_1", "right_arm_1",
        "left_arm_2", "right_arm_2",
        "left_arm_3", "right_arm_3",
        "left_arm_4", "right_arm_4",
        "left_arm_5", "right_arm_5",
        "left_arm_6", "right_arm_6",
        "gripper_finger_l1",
        "gripper_finger_r1",
    ],
    "action_joint": [
        "base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint",
        "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
        "head_0",
        "left_arm_0", "right_arm_0",
        "head_1",
        "left_arm_1", "right_arm_1",
        "left_arm_2", "right_arm_2",
        "left_arm_3", "right_arm_3",
        "left_arm_4", "right_arm_4",
        "left_arm_5", "right_arm_5",
        "left_arm_6", "right_arm_6",
        "gripper_finger_l1", "gripper_finger_l2",
        "gripper_finger_r1", "gripper_finger_r2",
    ],
}

# AI WORKER (FFW_SG2) -- 28 joints.
#
# ORDER CONFIRMED AGAINST A LIVE RUN, 2026-08-27. It was written as a breadth-first walk of the
# USD's articulation tree, which is how PhysX usually assigns DOF indices, and marked provisional
# because a wrong ORDER mis-labels the columns of an exported dataset WITHOUT RAISING -- the
# lengths are right either way, so the writer constructs and the video records regardless, and
# nothing downstream would ever catch it.
#
# probe_aiworker_base_stall.py dumped robot.joint_names from the running sim and it matches this
# action_joint list position for position, all 28:
#
#   base_prismatic_x_joint, base_prismatic_y_joint, base_revolute_z_joint, lift_joint,
#   arm_l_joint1, arm_r_joint1, head_joint1, arm_l_joint2, arm_r_joint2, head_joint2,
#   arm_l_joint3, arm_r_joint3, arm_l_joint4, arm_r_joint4, arm_l_joint5, arm_r_joint5,
#   arm_l_joint6, arm_r_joint6, arm_l_joint7, arm_r_joint7, gripper_l_joint1, gripper_l_joint3,
#   gripper_r_joint1, gripper_r_joint3, gripper_l_joint2, gripper_l_joint4, gripper_r_joint2,
#   gripper_r_joint4
#
# joint_angles is a NAME-KEYED projection rather than a slice, so its ordering of the gripper
# entries differs from the articulation's and that is fine.
#
# joint_angles = action_joint minus the 3 base joints and the two mirror fingers
# (gripper_l_joint3, gripper_r_joint3), matching what kitchen/mdp/observations.py:joint_angles
# emits for this robot.
AIWORKER_JOINTS: _RobotSchema = {
    "joint_angles": [
        "lift_joint", "arm_l_joint1", "arm_r_joint1",
        "head_joint1", "arm_l_joint2", "arm_r_joint2",
        "head_joint2", "arm_l_joint3", "arm_r_joint3",
        "arm_l_joint4", "arm_r_joint4", "arm_l_joint5",
        "arm_r_joint5", "arm_l_joint6", "arm_r_joint6",
        "arm_l_joint7", "arm_r_joint7", "gripper_l_joint1",
        "gripper_r_joint1", "gripper_l_joint2", "gripper_l_joint4",
        "gripper_r_joint2", "gripper_r_joint4",
    ],
    "action_joint": [
        "base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint",
        "lift_joint", "arm_l_joint1", "arm_r_joint1",
        "head_joint1", "arm_l_joint2", "arm_r_joint2",
        "head_joint2", "arm_l_joint3", "arm_r_joint3",
        "arm_l_joint4", "arm_r_joint4", "arm_l_joint5",
        "arm_r_joint5", "arm_l_joint6", "arm_r_joint6",
        "arm_l_joint7", "arm_r_joint7", "gripper_l_joint1",
        "gripper_l_joint3", "gripper_r_joint1", "gripper_r_joint3",
        "gripper_l_joint2", "gripper_l_joint4", "gripper_r_joint2",
        "gripper_r_joint4",
    ],
}

ROBOT_SCHEMAS: Dict[str, _RobotSchema] = {
    "anubis": ANUBIS_JOINTS,
    "rby1": RBY1_JOINTS,
    "aiworker": AIWORKER_JOINTS,
}


def get_robot_schema(robot: str) -> _RobotSchema:
    """Return the joint-name lists for ``robot``.

    Raises:
        ValueError: if ``robot`` is not configured in :data:`ROBOT_SCHEMAS`.
            The error message names the robot and points at this module so
            the fix is one edit away.
    """
    if robot not in ROBOT_SCHEMAS:
        raise ValueError(
            f"Unknown robot {robot!r}. Configured robots: {sorted(ROBOT_SCHEMAS)}. "
            f"Add an entry in robot_schemas.py to support this robot."
        )
    return ROBOT_SCHEMAS[robot]
