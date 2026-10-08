# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The dead zone in DifferentialIKController.compute(), on CPU.

This fork zeroes any commanded joint delta below a threshold. On a 7-DOF arm a differential-IK
solution spreads one small Cartesian step across every joint, so every share can land under a
per-joint threshold and the entire command is discarded -- measured on RB-Y1: 7 of 7 joints zeroed,
a 0.0137 rad command issued as 0.0, while the desired pose kept advancing. These tests pin the
threshold's default, so the Anubis chain that depends on it cannot change by accident, and pin that
a robot can switch it off.

Unlike test_differential_ik.py this needs no Isaac Sim: the dead zone is tensor arithmetic inside
compute(), and isaaclab.controllers.differential_ik imports without a sim app.
"""

import torch
import unittest

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

NUM_JOINTS = 7


def _controller(deadzone=None):
    kw = {} if deadzone is None else {"delta_joint_deadzone": deadzone}
    cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls", **kw
    )
    return DifferentialIKController(cfg, num_envs=1, device="cpu")


def _step(ctrl, position_error, num_joints=NUM_JOINTS):
    """One compute() through a Jacobian that splits the error evenly across every joint.

    That even split is the whole point: it is what a redundant arm's IK does to a small Cartesian
    step, and it is what drives each joint's share under a per-joint threshold.
    """
    ee_pos = torch.zeros(1, 3)
    ee_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    jacobian = torch.zeros(1, 6, num_joints)
    jacobian[0, 0, :] = 1.0  # every joint contributes equally to +x
    joint_pos = torch.zeros(1, num_joints)
    # Absolute mode: the command IS the desired position, so the error is position_error in +x.
    ctrl.set_command(torch.tensor([[position_error, 0.0, 0.0]]), ee_pos=ee_pos, ee_quat=ee_quat)
    return ctrl.compute(ee_pos, ee_quat, jacobian, joint_pos) - joint_pos


class TestDeadzone(unittest.TestCase):
    def test_default_threshold_is_the_forks_historical_one_hundredth(self):
        # Anubis's whole working chain runs on this value; it must not move silently.
        cfg = DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls"
        )
        self.assertEqual(cfg.delta_joint_deadzone, 1e-2)

    def test_a_seven_way_split_is_discarded_whole_at_the_default(self):
        dq = _step(_controller(), 0.005)
        self.assertLess(dq.abs().max().item(), 1e-2, "fixture must produce sub-threshold shares")
        self.assertEqual(dq.abs().sum().item(), 0.0, "the default must discard the whole command")

    def test_the_same_command_survives_with_the_deadzone_off(self):
        dq = _step(_controller(deadzone=0.0), 0.005)
        self.assertGreater(dq.abs().sum().item(), 0.0)

    def test_a_supra_threshold_joint_is_kept_at_the_default(self):
        dq = _step(_controller(), 0.5)
        self.assertGreater(dq.abs().max().item(), 1e-2)
        self.assertGreater(dq.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
