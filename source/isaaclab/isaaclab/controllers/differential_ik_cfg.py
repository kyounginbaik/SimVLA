# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .differential_ik import DifferentialIKController


@configclass
class DifferentialIKControllerCfg:
    """Configuration for differential inverse kinematics controller."""

    class_type: type = DifferentialIKController
    """The associated controller class."""

    command_type: Literal["position", "pose"] = MISSING
    """Type of task-space command to control the articulation's body.

    If "position", then the controller only controls the position of the articulation's body.
    Otherwise, the controller controls the pose of the articulation's body.
    """

    use_relative_mode: bool = False
    """Whether to use relative mode for the controller. Defaults to False.

    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """

    ik_method: Literal["pinv", "svd", "trans", "dls"] = MISSING
    """Method for computing inverse of Jacobian."""

    ik_params: dict[str, float] | None = None
    """Parameters for the inverse-kinematics method. Defaults to None, in which case the default
    parameters for the method are used.

    - Moore-Penrose pseudo-inverse ("pinv"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - Adaptive Singular Value Decomposition ("svd"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
        - "min_singular_value": Single values less than this are suppressed to zero (default: 1e-5).
    - Jacobian transpose ("trans"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
    - Damped Moore-Penrose pseudo-inverse ("dls"):
        - "lambda_val": Damping coefficient (default: 0.01).
    """

    delta_joint_deadzone: float = 1e-2
    """Per-joint magnitude below which a commanded joint delta is discarded, in radians.

    This fork zeroes small joint deltas to keep the arm from chattering on numerical noise. The
    threshold is per-joint, so it scales badly with redundancy: a differential-IK solution spreads
    one Cartesian step across every joint, and on a 7-DOF arm each share can fall below the
    threshold while their sum is a real motion. Measured on RB-Y1, all 7 joints of a 0.0137 rad
    command were zeroed and the arm was issued nothing, while the desired pose it was chasing kept
    advancing -- the arm ended its trajectories a median 0.071 m short.

    Defaults to 1e-2, the value this fork has always used. Set to 0.0 to disable.
    """

    init_joint_pos: list[float] = None

    def __post_init__(self):
        # check valid input
        if self.command_type not in ["position", "pose"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.command_type}.")
        if self.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.ik_method}.")
        # default parameters for different inverse kinematics approaches.
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 0.01},
        }
        # update parameters for IK-method if not provided
        ik_params = default_ik_params[self.ik_method].copy()
        if self.ik_params is not None:
            ik_params.update(self.ik_params)
        self.ik_params = ik_params
