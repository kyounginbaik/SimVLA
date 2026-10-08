# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np
from scipy.spatial.transform import Rotation


@torch.no_grad()
def quat_wxyz_to_R_scipy(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    quat_wxyz: (B,4) or (4,) torch tensor in (w,x,y,z), any device
    returns:   (B,3,3) torch tensor on same device/dtype (computed via CPU SciPy)
    """
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz.unsqueeze(0)

    q = F.normalize(quat_wxyz, dim=-1)

    # Guard against a degenerate (0,0,0,0) or non-finite quaternion — an object knocked into an
    # invalid physics state in ONE env would otherwise make scipy's from_quat raise and kill the
    # whole batch. Replace any bad quat with the identity (wxyz = 1,0,0,0); valid quats are unchanged.
    _n = quat_wxyz.norm(dim=-1)
    _bad = (_n < 1e-6) | ~torch.isfinite(_n)
    if bool(_bad.any()):
        _ident = torch.zeros_like(q)
        _ident[..., 0] = 1.0
        q = torch.where(_bad.unsqueeze(-1), _ident, q)

    q_np = q.detach().cpu().numpy().astype(np.float64)  # (B,4) wxyz
    q_xyzw = np.concatenate([q_np[:, 1:4], q_np[:, 0:1]], axis=-1)  # (B,4) xyzw

    Rm = Rotation.from_quat(q_xyzw).as_matrix()  # (B,3,3)
    return torch.from_numpy(Rm).to(device=quat_wxyz.device, dtype=quat_wxyz.dtype)


def mobile_base(env: ManagerBasedRLEnv) -> torch.Tensor:
    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    pos = env.scene.articulations["robot"].data.body_pos_w[:,base_link_idx, :2] - env.scene.env_origins[:, :2]
    quat = env.scene.articulations["robot"].data.body_quat_w[:, base_link_idx]
    return torch.cat((pos, quat), dim=1)


def ee_6d_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    observation = []
    
    # Robot base position   
    robot_data = env.scene.articulations["robot"].data
    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    root_pos_w = robot_data.body_pos_w[:, base_link_idx].clone()
    root_quat_w = robot_data.body_quat_w[:, base_link_idx].clone()

    # 1. ee_link2 (left arm)
    l_eef_idx = env.scene.articulations["robot"].find_bodies("ee_link2")[0][0]
    r_eef_idx = env.scene.articulations["robot"].find_bodies("ee_link1")[0][0]

    ee_L_pos_w = env.scene.articulations["robot"].data.body_pos_w[:, l_eef_idx]
    ee_L_quat_w = env.scene.articulations["robot"].data.body_quat_w[:, l_eef_idx]
    
    # 2. ee_link1 (right arm)
    ee_R_pos_w = env.scene.articulations["robot"].data.body_pos_w[:,r_eef_idx]
    ee_R_quat_w = env.scene.articulations["robot"].data.body_quat_w[:, r_eef_idx]
        
    ee_l_pose_b, ee_l_quat_b = math_utils.subtract_frame_transforms( root_pos_w, root_quat_w, ee_L_pos_w, ee_L_quat_w)
    ee_r_pose_b, ee_r_quat_b = math_utils.subtract_frame_transforms( root_pos_w, root_quat_w, ee_R_pos_w, ee_R_quat_w)

    # --- LEFT ARM ---
    quat = ee_l_quat_b  # (B,4) wxyz
    R = quat_wxyz_to_R_scipy(quat)  # (B,3,3)

    delta_local = torch.tensor([0.0, 0.0, -0.10956], device=R.device, dtype=R.dtype).view(1, 3, 1)
    delta_world = (R @ delta_local).squeeze(-1)  # (B,3)
    ee_l_pose_b = ee_l_pose_b + delta_world
    
    ee_l_pose_b[:,0] += 0.095
    ee_l_pose_b[:,2] += -0.823356

    B = R.shape[0]
    rot6d_l = R[:, :, :2].reshape(B, 6)

    observation.append(ee_l_pose_b)
    observation.append(rot6d_l)

    # --- RIGHT ARM ---
    quat = ee_r_quat_b  # (B,4) wxyz
    R = quat_wxyz_to_R_scipy(quat)  # (B,3,3)

    delta_local = torch.tensor([0.0, 0.0, -0.10956], device=R.device, dtype=R.dtype).view(1, 3, 1)
    delta_world = (R @ delta_local).squeeze(-1)  # (B,3)
    ee_r_pose_b = ee_r_pose_b + delta_world
    ee_r_pose_b[:,0] += 0.095
    ee_r_pose_b[:,2] += -0.823356

    B = R.shape[0]
    rot6d_r = R[:, :, :2].reshape(B, 6)

    observation.append(ee_r_pose_b)
    observation.append(rot6d_r)

    # Gripper.
    #
    # THE DATASET CONVENTION IS: fully OPEN reads -1.6, fully CLOSED reads +0.1. Anubis's jaws
    # are prismatic over [0, 0.04] m where 0.04 is open, which is what `0.1 - 1.7*q/0.04` maps.
    #
    # The negative indices below are Anubis's layout: its last four DOFs are gripper1R_joint,
    # gripper1_joint, gripper2R_joint, gripper2_joint, so -1 is the LEFT hand and -3 the RIGHT.
    # THAT ORDER IS NOT UNIVERSAL. AI Worker's last four are gripper_l_joint2, gripper_l_joint4,
    # gripper_r_joint2, gripper_r_joint4 -- so -1 is its RIGHT hand and -3 its LEFT, transposed
    # -- and its joints are REVOLUTE with 0 = OPEN, closing near 1.1 rad, so the same expression
    # also inverts the polarity and lands about 26x outside the convention's range. None of that
    # raises; it just writes wrong numbers into obs__eef_pose for every frame of every episode.
    #
    # So resolve BY NAME where the robot offers a name, and keep the index path for the two
    # robots whose datasets were recorded with it.
    art = env.scene.articulations["robot"]
    #: robot -> (left drive joint, right drive joint, joint value at FULLY CLOSED)
    _GRIPPER_READ = {
        "gripper_l_joint1": ("gripper_l_joint1", "gripper_r_joint1", 0., 1.1002),
        "gripper_finger_l1": ("gripper_finger_l1", "gripper_finger_r1", -.04, 0.),
    }
    _named = next((v for k, v in _GRIPPER_READ.items() if k in art.joint_names), None)
    if _named is not None:
        l_name, r_name, q_open, q_closed = _named
        li = art.find_joints(l_name)[0][0]
        ri = art.find_joints(r_name)[0][0]
        # q runs 0 (open) -> q_closed (shut), the opposite sense to Anubis's stroke, so the map
        # is -1.6 + 1.7*(q/q_closed) rather than 0.1 - 1.7*(q/0.04). Both send open to -1.6 and
        # closed to +0.1.
        finger_joint_L1 = robot_data.joint_pos[:, li].clone().unsqueeze(1)
        finger_joint_R1 = robot_data.joint_pos[:, ri].clone().unsqueeze(1)
        observation.append(-1.6 + 1.7 * (finger_joint_L1 - q_open) / (q_closed - q_open))
        observation.append(-1.6 + 1.7 * (finger_joint_R1 - q_open) / (q_closed - q_open))
    else:
        finger_joint_L1 = robot_data.joint_pos[:, -1].clone().unsqueeze(1)
        finger_joint_R1 = robot_data.joint_pos[:, -3].clone().unsqueeze(1)
        observation.append(0.1 - 1.7 * finger_joint_L1 /0.04)
        observation.append(0.1 - 1.7 * finger_joint_R1 /0.04)
    return torch.cat(observation, dim=-1)

def base_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    # compute angular velocity (yaw rate)
    base_vel = env.scene.articulations["robot"].data.joint_vel[:,:3]
    #base_vel[:,1] *= -1
    return base_vel

# Joint position
_BODY_NAMES_PRINTED = False

def joint_angles(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _BODY_NAMES_PRINTED
    if not _BODY_NAMES_PRINTED:
        bodies = env.scene.articulations["robot"].data.body_names
        finger_bodies = [b for b in bodies if "finger" in b.lower() or "gripper" in b.lower()]
        print(f"[BODY-NAMES-CAPTURE] count={len(bodies)} finger_bodies={finger_bodies}", flush=True)
        print(f"[BODY-NAMES-CAPTURE] all={list(bodies)}", flush=True)
        _BODY_NAMES_PRINTED = True
    q = env.scene.articulations["robot"].data.joint_pos[:, 3:]
    if q.shape[-1] == 26: # rby1
        gripper1R_idx = env.scene.articulations["robot"].find_joints("gripper_finger_r2")[0][0] - 3
        gripper2R_idx = env.scene.articulations["robot"].find_joints("gripper_finger_l2")[0][0] - 3
    elif q.shape[-1] == 25: # aiworker (FFW_SG2)
        # 28 joints less the 3 base ones. Without this branch the count falls through to the
        # Anubis case and find_joints("gripper1R_joint") raises on a robot that has no such
        # joint -- every step, from the first observation.
        #
        # WHICH TWO ARE DROPPED. Each RH-P12-RN hand has four joints: _joint1/_joint3 are the
        # two proximal fingers and _joint2/_joint4 their distal links. _joint3 mirrors _joint1
        # across the jaw, so its angle carries no information the other does not -- the same
        # relationship Anubis's gripper1R_joint has to gripper1_joint and RB-Y1's
        # gripper_finger_r2 to r1. robot_schemas.AIWORKER_JOINTS omits exactly these two.
        gripper1R_idx = env.scene.articulations["robot"].find_joints("gripper_r_joint3")[0][0] - 3
        gripper2R_idx = env.scene.articulations["robot"].find_joints("gripper_l_joint3")[0][0] - 3
    else: # anubis:
        gripper1R_idx = env.scene.articulations["robot"].find_joints("gripper1R_joint")[0][0] - 3
        gripper2R_idx = env.scene.articulations["robot"].find_joints("gripper2R_joint")[0][0] - 3

    n = q.shape[1]
    idx = torch.arange(n, device=q.device)
# keep everything except indices n-2 and n-4
    mask = (idx != gripper1R_idx) & (idx != gripper2R_idx)
    q_new = q[:, mask]   # shape: (num_envs, N-2)
    return q_new 

# Force
# EEF distance to all 
