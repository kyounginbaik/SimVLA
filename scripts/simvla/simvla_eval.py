"""
Assumptions:
  - Isaac Lab task exposes observations under obs_dict["policy"].
  - The keys in obs_dict["policy"] match the policy input feature names,
    e.g. "observation.state", "observation.images.front",
    "observation.images.wrist_left", "observation.images.wrist_right".
  - Action dimension is 23 (matching your dataset's "action" feature).
"""

import argparse
import os
from isaaclab.app import AppLauncher

# === simvla path resolution (auto-added) ===
import os as _simvla_os
from pathlib import Path as _SimvlaPath
def _simvla_find_repo_root():
    env = _simvla_os.environ.get("SIMVLA_REPO_ROOT")
    if env:
        return env
    for p in _SimvlaPath(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return str(p)
    raise RuntimeError("Cannot find repo root; set SIMVLA_REPO_ROOT env var")
SIMVLA_REPO_ROOT = _simvla_find_repo_root()
# === end simvla path resolution ===

import simvla_paths
from collector_profile import apply_ik_joint_deadzone


# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser(
    description="Evaluate Openpi for Isaac Lab environment."
)

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Name of the Isaac Lab task (e.g., Isaac-Kitchen-v1103-00_sub0).",
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Name of the Isaac Lab task (e.g., Isaac-Kitchen-v1103-00).",
)
parser.add_argument(
    "--policy_dir",
    type=str,
    default=None,
    help=(
        "Unused by this script (the policy is served over websocket by "
        "--host_ip). Kept for logging/back-compat. "
        "(e.g., checkpoints/pi05_simvla/simvla/20000)"
    ),
)
parser.add_argument(
    "--horizon",
    type=int,
    default=400,
    help="Step horizon per rollout.",
)
parser.add_argument("--step_hz", type=int, default=20, help="Environment stepping rate in Hz.")
parser.add_argument("--action_chunk", type=int, default=50, help="Action chunk size.")
parser.add_argument(
    "--tile_size",
    type=int,
    default=20,
    help="Max envs per tiled overview video. More envs -> split into multiple tiled videos.",
)
parser.add_argument(
    "--tile_downscale",
    type=int,
    default=2,
    help="Downscale factor for each tile in the overview video (memory/size).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=20,
    help="Number of envs.",
)
parser.add_argument(
    "--num_seeds",
    type=int,
    default=1,
    help="Number of random seeds to evaluate.",
)
parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=None,
    help="Specific seeds to use (overrides --num_seeds).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/tmp/act_policy_evaluation_results",
    help="Directory to write results to.",
)
parser.add_argument(
    "--log_file",
    type=str,
    default="results",
    help="Base name of output file.",
)
parser.add_argument(
    "--OOD",
    type=str,
    default="True",
    help="True for OOD, False for ID",
)
parser.add_argument(
    "--enable_pinocchio",
    default=False,
    action="store_true",
    help="Enable Pinocchio (needed by some controllers/retargeters).",
)
parser.add_argument(
    "--model",
    type=str,
    default="pi0",
    help="Type of model.",
)
parser.add_argument(
    "--task_language",
    type=str,
    default="task",
    help="Task in language.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default=None,
    help="LeRobot data repo_id",
)
parser.add_argument(
    "--host_ip",
    type=str,
    default="localhost",
    help="Insert the spinning server ip.",
)
# --- Open-loop demo replay -------------------------------------------------------------
# Drive the env from a RECORDED episode's actions instead of the policy's, through the exact
# same conversion path (apply_action_postprocess -> target-to-target deltas -> gripper
# binarisation -> base pass-through). The demo is ground truth: if the robot does not retrace it,
# the defect is in how eval applies actions, not in the checkpoint. No policy server is contacted.
parser.add_argument(
    "--replay_kitchen",
    type=int,
    default=None,
    help="Replay a recorded demo instead of querying the policy. Value is the dataset's "
         "kitchen_num (e.g. 1 for Isaac-SceneSmith-001).",
)
parser.add_argument(
    "--replay_episode",
    type=int,
    default=0,
    help="Which episode of that kitchen to replay (index among its episodes).",
)
parser.add_argument(
    "--replay_repo",
    type=str,
    default="lerobot_data/pushchair_all",
    help="LeRobot dataset root to replay from.",
)
parser.add_argument(
    "--start_kitchen",
    type=int,
    default=None,
    help="Keep the POLICY in control but start it from a recorded episode's exact initial pose "
         "(that kitchen_num, episode --replay_episode). Separates 'the policy cannot do the task' "
         "from 'the goal file's OOD start box puts it somewhere the demos never started'.",
)
# Pass AppLauncher args (e.g., --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Launch Omniverse app early
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------- Imports after app -----------------------
import copy
import gymnasium as gym
import os
import pathlib
import random
import torch
import torch._dynamo
torch._dynamo.config.disable = True

import numpy as np
import ipdb
from scipy.spatial.transform import Rotation as R
from isaaclab_tasks.utils import parse_env_cfg
#import robomimic.utils.torch_utils as TorchUtils  # for device util
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import lerobot.datasets.lerobot_dataset as lerobot_dataset

from openpi_client import image_tools
from openpi_client import websocket_client_policy
import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import time
import json
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from dataclasses import dataclass, field
import abc
import builtins
import logging
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, TypeVar

from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn
from typing_extensions import Unpack

from isaaclab.utils.math import quat_from_matrix, quat_mul, euler_xyz_from_quat, quat_inv

class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# ----------------------- Helpers -----------------------

def build_policy_observation(
    obs_dict: dict,
    policy: ACTPolicy,
    device: torch.device,
) -> dict:
    """
    Convert Isaac Lab observation (obs_dict["policy"]) into the dict of
    tensors expected by LeRobot's ACTPolicy.select_action.

    We:
      - Look at policy.config.input_features.keys() to know which feature
        names the policy expects.
      - For features whose name contains "image" or "images", we interpret
        them as RGB images and:
            HWC uint8/float -> CHW float in [0,1], add batch dim.
      - For others, we treat them as vector features (e.g. "observation.state")
        and:
            (D,) -> (1, D) float32 on the correct device.

    IMPORTANT:
      This assumes that obs_dict["policy"] already has keys like
      "observation.state", "observation.images.front", ...
      If your Isaac Lab observations use different names, replace the
      src[...] access below with your own mapping.
    """
    src = obs_dict["policy"]
    ee_6d = src["ee_6D_pos"].to(device).float()
    base_vel = src["base_vel"].to(device).float()

    observation_state = torch.cat([ee_6d, base_vel], dim=-1)

    # This key name should match what your LeRobot dataset / policy expects
    src["observation.state"] = observation_state

    policy_obs = {}

    for name in policy.config.input_features.keys():
        if name not in src:
            continue

        value = src[name]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)

        value = value.to(device=device)

        if ("image" in name) or ("images" in name):
            # Image feature
            # Expect HWC in 0..255 or 0..1 -> CHW in 0..1
            if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                # HWC -> CHW
                value = value.permute(2, 0, 1)
            value = value.to(torch.float32)
            # If coming in as 0..255, normalize to 0..1
            if value.max() > 1.1:
                value = value / 255.0
            # Add batch dim: (C,H,W) -> (1,C,H,W)
            if value.ndim == 3:
                value = value.unsqueeze(0)
        else:
            # Non-image feature, e.g. "observation.state"
            value = value.to(torch.float32)
            # Add batch dim if it's a single vector
            if value.ndim == 1:
                value = value.unsqueeze(0)

        policy_obs[name] = value
    return policy_obs


import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

from isaaclab.utils.math import (
    quat_mul,
    quat_inv,
    euler_xyz_from_quat,
)

# --------------------------------------------------
# Basic helpers
# --------------------------------------------------
def tensor_to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def _to_hwc_uint8(t):
    """Single image tensor/array -> HWC uint8 RGB."""
    x = tensor_to_numpy(t)

    # CHW -> HWC
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.moveaxis(x, 0, -1)

    if x.shape[-1] == 4:
        x = x[..., :3]

    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = x.clip(0, 255).astype(np.uint8)
    return x


class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# --------------------------------------------------
# Rotation helpers (fully batched torch)
# --------------------------------------------------
def rot6d_to_R(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rot6d: (..., 6)
    returns: (..., 3, 3)
    """
    orig_shape = rot6d.shape[:-1]
    x = rot6d.reshape(-1, 6)

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = F.normalize(a1, dim=-1, eps=eps)
    a2_ortho = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    Rm = torch.stack([b1, b2, b3], dim=-1)  # (B,3,3), columns
    return Rm.reshape(*orig_shape, 3, 3)


def matrix_to_quat_wxyz(Rm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Rm: (..., 3, 3)
    returns: (..., 4) in wxyz
    """
    m = Rm.reshape(-1, 3, 3)
    B = m.shape[0]

    q = torch.empty((B, 4), device=m.device, dtype=m.dtype)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    cond = trace > 0.0

    # case 1: trace > 0
    if cond.any():
        t = torch.sqrt(trace[cond] + 1.0 + eps) * 2.0
        q[cond, 0] = 0.25 * t
        q[cond, 1] = (m[cond, 2, 1] - m[cond, 1, 2]) / t
        q[cond, 2] = (m[cond, 0, 2] - m[cond, 2, 0]) / t
        q[cond, 3] = (m[cond, 1, 0] - m[cond, 0, 1]) / t

    # case 2/3/4
    not_cond = ~cond
    if not_cond.any():
        mn = m[not_cond]
        diag = torch.stack([mn[:, 0, 0], mn[:, 1, 1], mn[:, 2, 2]], dim=1)
        idx = torch.argmax(diag, dim=1)

        # m00 max
        mask0 = idx == 0
        if mask0.any():
            mm = mn[mask0]
            t = torch.sqrt(1.0 + mm[:, 0, 0] - mm[:, 1, 1] - mm[:, 2, 2] + eps) * 2.0
            q_sub = torch.empty((mask0.sum(), 4), device=m.device, dtype=m.dtype)
            q_sub[:, 0] = (mm[:, 2, 1] - mm[:, 1, 2]) / t
            q_sub[:, 1] = 0.25 * t
            q_sub[:, 2] = (mm[:, 0, 1] + mm[:, 1, 0]) / t
            q_sub[:, 3] = (mm[:, 0, 2] + mm[:, 2, 0]) / t
            q[not_cond.nonzero(as_tuple=True)[0][mask0]] = q_sub

        # m11 max
        mask1 = idx == 1
        if mask1.any():
            mm = mn[mask1]
            t = torch.sqrt(1.0 + mm[:, 1, 1] - mm[:, 0, 0] - mm[:, 2, 2] + eps) * 2.0
            q_sub = torch.empty((mask1.sum(), 4), device=m.device, dtype=m.dtype)
            q_sub[:, 0] = (mm[:, 0, 2] - mm[:, 2, 0]) / t
            q_sub[:, 1] = (mm[:, 0, 1] + mm[:, 1, 0]) / t
            q_sub[:, 2] = 0.25 * t
            q_sub[:, 3] = (mm[:, 1, 2] + mm[:, 2, 1]) / t
            q[not_cond.nonzero(as_tuple=True)[0][mask1]] = q_sub

        # m22 max
        mask2 = idx == 2
        if mask2.any():
            mm = mn[mask2]
            t = torch.sqrt(1.0 + mm[:, 2, 2] - mm[:, 0, 0] - mm[:, 1, 1] + eps) * 2.0
            q_sub = torch.empty((mask2.sum(), 4), device=m.device, dtype=m.dtype)
            q_sub[:, 0] = (mm[:, 1, 0] - mm[:, 0, 1]) / t
            q_sub[:, 1] = (mm[:, 0, 2] + mm[:, 2, 0]) / t
            q_sub[:, 2] = (mm[:, 1, 2] + mm[:, 2, 1]) / t
            q_sub[:, 3] = 0.25 * t
            q[not_cond.nonzero(as_tuple=True)[0][mask2]] = q_sub

    q = F.normalize(q, dim=-1, eps=eps)

    # sign convention: w >= 0
    mask = q[:, 0] < 0
    q[mask] = -q[mask]

    return q.reshape(*Rm.shape[:-2], 4)


def sixd_to_quat_wxyz(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rot6d: (..., 6)
    returns: (..., 4) in wxyz
    """
    Rm = rot6d_to_R(rot6d, eps=eps)
    return matrix_to_quat_wxyz(Rm, eps=eps)


def quat_delta_axis_angle(q_t: torch.Tensor, q_tp1: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    q_t, q_tp1: (..., 4) wxyz
    returns: (..., 3) axis-angle vector
    """
    q_t = F.normalize(q_t, dim=-1, eps=eps)
    q_tp1 = F.normalize(q_tp1, dim=-1, eps=eps)

    q_delta = quat_mul(q_tp1, quat_inv(q_t))

    mask = q_delta[..., 0] < 0
    q_delta[mask] = -q_delta[mask]

    w = q_delta[..., 0].clamp(-1.0, 1.0)
    xyz = q_delta[..., 1:]

    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp_min(eps))
    axis = xyz / sin_half.unsqueeze(-1)

    rot_vec = axis * angle.unsqueeze(-1)
    rot_vec = torch.nan_to_num(rot_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return rot_vec


# --------------------------------------------------
# OpenPI helpers
# --------------------------------------------------
# The two 6-D rotation blocks of the raw sim state (before reorder_state_for_policy moves the
# gripper): one per arm, at dims 3:9 and 12:18 (each arm is pos(3) + rot6d(6)).
_RAW_ROT6D_BLOCKS = (3, 12)
# Interleaved (a0,b0,a1,b1,a2,b2) -> column-concatenated (a0,a1,a2,b0,b1,b2).
_ROT6D_INTERLEAVED_TO_CONCAT = (0, 2, 4, 1, 3, 5)


def rot6d_sim_to_dataset(state: torch.Tensor) -> torch.Tensor:
    """Re-flatten each arm's 6-D rotation from the SIM's layout into the DATASET's.

    Both encode the same thing — the first two columns of the EE rotation matrix — but they
    flatten it differently, and the policy was trained on the dataset's:

      sim      `mdp.ee_6d_pos`: `R[:, :, :2].reshape(B, 6)` is row-major -> a0,b0,a1,b1,a2,b2
      dataset  recorded state / the real robot's quat_to_6d: [R[:,0], R[:,1]] -> a0,a1,a2,b0,b1,b2

    MEASURED, not inferred (2026-08-01): at the reset pose the sim sends
    (-0.5178, -0.1185, -0.8517, -0.0217, 0.0804, -0.9927) where every recorded episode of BOTH
    the kitchen (`type1`, task1) and SceneSmith (`pushchair_all`) datasets starts at
    (-0.5193, -0.8508, 0.0806, -0.1189, -0.0215, -0.9927). Same two column vectors, interleaved
    vs concatenated — the position dims match to 4 decimals, so it is the flattening, not the pose.

    Un-permuted, the policy is handed an orientation it never saw in training while the position
    dims look perfectly normal, so nothing errors and the rollout just fails quietly. Note the
    ACTION path already uses the dataset convention (`rot6d_to_R` reads x[:,0:3] / x[:,3:6] as the
    two columns), so only the observation was inconsistent.

    Set SIMVLA_ROT6D_SIM_ORDER=1 to send the raw sim order instead (the pre-fix behavior).
    """
    if os.environ.get("SIMVLA_ROT6D_SIM_ORDER") == "1":
        return state
    out = state.clone()
    for start in _RAW_ROT6D_BLOCKS:
        block = state[:, start:start + 6]
        out[:, start:start + 6] = block[:, _ROT6D_INTERLEAVED_TO_CONCAT]
    return out


def reorder_state_for_policy(state: torch.Tensor) -> torch.Tensor:
    """
    state: (N, D)
    move original index 18 -> new index 9
    """
    order = list(range(state.shape[1]))
    moved = order.pop(18)
    order.insert(9, moved)
    return state[:, order]


def normalize_single_env_action_chunk(raw_actions, expected_act_dim=23) -> np.ndarray:
    """
    Accept a single-env infer output and normalize to (chunk, act_dim).
    """
    arr = np.asarray(raw_actions)

    # Possible shapes:
    # (chunk, act_dim)
    # (1, chunk, act_dim)
    # (chunk, 1, act_dim)
    # maybe even (act_dim,) if chunk=1
    if arr.ndim == 1:
        if arr.shape[0] != expected_act_dim:
            raise ValueError(f"1D action has wrong dim: {arr.shape}")
        arr = arr[None, :]  # (1, act_dim)

    elif arr.ndim == 2:
        # expected
        if arr.shape[-1] != expected_act_dim:
            raise ValueError(f"2D action has wrong last dim: {arr.shape}")

    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[-1] == expected_act_dim:
            arr = arr[0]  # (chunk, act_dim)
        elif arr.shape[1] == 1 and arr.shape[-1] == expected_act_dim:
            arr = arr[:, 0, :]
        else:
            raise ValueError(f"Cannot normalize 3D single-env action shape: {arr.shape}")
    else:
        raise ValueError(f"Unexpected single-env action shape: {arr.shape}")

    return np.array(arr, copy=True)


def infer_action_chunk_per_env(
    client,
    front_rgb: torch.Tensor,
    wrist_left_rgb: torch.Tensor,
    wrist_right_rgb: torch.Tensor,
    state: torch.Tensor,
    prompt: str,
    num_envs: int,
) -> np.ndarray:
    """
    Returns action_chunk of shape (chunk, N, act_dim)
    """
    per_env_chunks = []

    for env_i in range(num_envs):
        # _to_hwc_uint8 FIRST, so the policy sees exactly the HWC-uint8 RGB the rollout videos
        # show. Previously the raw sensor tensor went straight out: resize_with_pad early-returns
        # when the size already matches (it always does — the cameras are 240x320, the model's
        # input size), so nothing normalised it. openpi's SimVLAInputs._parse_image then treats
        # ANY floating-point image as [0,1] and multiplies by 255 — so a float image in [0,255]
        # (what IsaacLab hands back for some camera configurations) wraps around to noise and the
        # policy is effectively blind, with no error anywhere. Converting up front makes the input
        # dtype-independent; for a uint8 sensor tensor this is a no-op.
        front_img_i = image_tools.resize_with_pad(_to_hwc_uint8(front_rgb[env_i]), 240, 320)
        wrist_left_img_i = image_tools.resize_with_pad(_to_hwc_uint8(wrist_left_rgb[env_i]), 240, 320)
        wrist_right_img_i = image_tools.resize_with_pad(_to_hwc_uint8(wrist_right_rgb[env_i]), 240, 320)

        obs_i = {
            "observation/images/front": tensor_to_numpy(front_img_i),
            "observation/images/wrist_left": tensor_to_numpy(wrist_left_img_i),
            "observation/images/wrist_right": tensor_to_numpy(wrist_right_img_i),
            "observation/state": tensor_to_numpy(state[env_i]),
            "prompt": prompt,
            "task": prompt,
        }

        raw = client.infer(obs_i)["actions"]
        chunk_i = normalize_single_env_action_chunk(raw, expected_act_dim=23)  # (chunk, act_dim)
        per_env_chunks.append(chunk_i)

    # [(chunk, act_dim)] * N -> (chunk, N, act_dim)
    chunk_lens = [x.shape[0] for x in per_env_chunks]
    if len(set(chunk_lens)) != 1:
        raise ValueError(f"Per-env chunk lengths differ: {chunk_lens}")

    action_chunk = np.stack(per_env_chunks, axis=1)
    return action_chunk


def apply_action_postprocess(action_chunk: np.ndarray, device: torch.device) -> np.ndarray:
    """
    action_chunk: (chunk, N, act_dim)
    returns same shape
    """
    action_chunk = np.array(action_chunk, copy=True)

    # fixed offsets
    action_chunk[:, :, 0] += -0.095
    action_chunk[:, :, 2] += 0.823356
    action_chunk[:, :, 10] += -0.095
    action_chunk[:, :, 12] += 0.823356

    chunk_len, N, act_dim = action_chunk.shape
    flat = torch.as_tensor(
        action_chunk.reshape(chunk_len * N, act_dim),
        device=device,
        dtype=torch.float32,
    )

    R_l = rot6d_to_R(flat[:, 3:9])      # (B,3,3)
    R_r = rot6d_to_R(flat[:, 13:19])    # (B,3,3)

    delta_local = torch.tensor([0.0, 0.0, 0.10956], device=device, dtype=torch.float32).view(1, 3, 1)
    delta_world_l = (R_l @ delta_local).squeeze(-1)
    delta_world_r = (R_r @ delta_local).squeeze(-1)

    flat[:, 0:3] += delta_world_l
    flat[:, 10:13] += delta_world_r

    out = flat.detach().cpu().numpy().reshape(chunk_len, N, act_dim)
    return out


# --------------------------------------------------
# Success detection (task1: mug0 in sink + both arms home)
# --------------------------------------------------
# Self-contained physical success check for eval. Mirrors the geometry of
# mdp.task1 (source/.../kitchen/mdp/terminations.py) — the env's registered
# `success` DoneTerm — but WITHOUT its `variable.env_goal_indices == last_subtask`
# gate, which is data-gen state-machine state the eval never advances (and which,
# together with a `sink_pos`/`sink` param mismatch, is why the term could not be
# used as-is). Returns the two physical components so partial success is visible.
# home_r/home_l are the base-frame home EE positions from terminations.py.
_TASK1_HOME_R = (0.2257, -0.0988, 1.0351)
_TASK1_HOME_L = (0.2203, 0.1080, 1.0342)


def task1_success_components(env, device, radius: float = 0.12):
    """Returns (close_sink, close_home, success) per-env bool tensors, shape (N,)."""
    from isaaclab.utils.math import quat_rotate_inverse

    mug = env.scene["mug0"]
    obj_pos = mug.data.body_pos_w.squeeze(1)  # (N,3) world

    sink = env.scene["sink_cabinet"]
    bpw = sink.data.body_pos_w  # (N, num_bodies, 3): [0]=corpus, [1..]=doors
    sink_pos = bpw[:, 0, :].clone()
    # basin center: shift toward the door depending on cabinet orientation (as in task1)
    if torch.allclose(bpw[:, 2, 0], bpw[:, 1, 0], rtol=1e-5, atol=1e-6):
        sink_pos[:, 0] = (sink_pos[:, 0] + bpw[:, 1, 0]) / 2
    elif torch.allclose(bpw[:, 2, 1], bpw[:, 1, 1], rtol=1e-5, atol=1e-6):
        sink_pos[:, 1] = (sink_pos[:, 1] + bpw[:, 1, 1]) / 2
    sink_pos[:, 2] = 0.88
    close_sink = torch.linalg.norm(obj_pos - sink_pos, dim=-1) <= radius

    robot = env.scene.articulations["robot"]
    ri = robot.find_bodies("ee_link1")[0][0]
    li = robot.find_bodies("ee_link2")[0][0]
    bi = robot.find_bodies("base_link")[0][0]
    base_pos = robot.data.body_pos_w[:, bi]
    base_quat = robot.data.body_quat_w[:, bi]
    eef_r_base = quat_rotate_inverse(base_quat, robot.data.body_pos_w[:, ri] - base_pos)
    eef_l_base = quat_rotate_inverse(base_quat, robot.data.body_pos_w[:, li] - base_pos)
    home_r = torch.tensor(_TASK1_HOME_R, device=device)
    home_l = torch.tensor(_TASK1_HOME_L, device=device)
    close_home = (torch.linalg.norm(eef_r_base - home_r, dim=-1) < radius) & (
        torch.linalg.norm(eef_l_base - home_l, dim=-1) < radius
    )

    return close_sink, close_home, (close_sink & close_home)


def task3_success_components(env, device, hold_radius: float = 0.2, home_radius: float = 0.12,
                             lift_z: float = 0.90):
    """task3 (mdp.task3): hold bottle0 in the right hand AND mug0 in the left hand, with the
    right arm returned home. Returns (held_both, close_home, success), each (N,) bool.

    IMPORTANT: an object is only "held" if it is BOTH within hold_radius of its gripper AND
    LIFTED off the counter (world z > lift_z). The counter rest height is ~0.82 m in these
    kitchens, so lift_z=0.90 requires a real ~8 cm lift. Proximity alone is a false-positive
    trap — an object sitting on the counter can be within 0.2 m of a passing gripper without
    being grasped (this is what made a non-grasping policy score 4/20 before). home_radius is
    relaxed vs the term's tight 0.02 m so eval isn't gated on millimetre homing."""
    from isaaclab.utils.math import quat_rotate_inverse

    robot = env.scene.articulations["robot"]
    ri = robot.find_bodies("ee_link1")[0][0]
    li = robot.find_bodies("ee_link2")[0][0]
    bi = robot.find_bodies("base_link")[0][0]
    eef_r_w = robot.data.body_pos_w[:, ri]
    eef_l_w = robot.data.body_pos_w[:, li]
    base_pos = robot.data.body_pos_w[:, bi]
    base_quat = robot.data.body_quat_w[:, bi]

    eef_r_base = quat_rotate_inverse(base_quat, eef_r_w - base_pos)
    home_r = torch.tensor(_TASK1_HOME_R, device=device)
    close_home = torch.linalg.norm(eef_r_base - home_r, dim=-1) < home_radius

    bottle_pos = env.scene.rigid_objects["bottle0"].data.body_pos_w.squeeze(1)
    mug_pos = env.scene.rigid_objects["mug0"].data.body_pos_w.squeeze(1)
    # held = near the gripper AND lifted off the counter (a real grasp, not mere proximity)
    bottle_held = (torch.linalg.norm(bottle_pos - eef_r_w, dim=-1) < hold_radius) & (bottle_pos[:, 2] > lift_z)
    mug_held = (torch.linalg.norm(mug_pos - eef_l_w, dim=-1) < hold_radius) & (mug_pos[:, 2] > lift_z)
    held_both = bottle_held & mug_held

    return held_both, close_home, (held_both & close_home)


# Dispatch success detection by the env's registered success-term function name, so the same
# eval scores whatever task the kitchen defines. Returns (comp1, comp2, success, (l1, l2)).
def _bowl_actually_in_drawer(env, device, drawer_r: float = 0.28):
    """Strict 'bowl in drawer': the drawer is OPEN, the bowl is at the drawer's x-y location
    (not merely at drawer height somewhere in the scene), and at a plausible drawer height.
    Fixes the false positives of the bare height-band check (mdp.task2's own loose definition)."""
    bowl = env.scene["bowl0"].data.body_pos_w.squeeze(1)  # (N,3)
    bc = env.scene["base_cabinet"]
    ji = bc.find_joints("corpus_to_drawer_0_0")[0][0]
    drawer_open = bc.data.joint_pos[:, ji] > 0.08
    try:
        dbi = bc.find_bodies("drawer_0_0")[0][0]
        drawer_xy = bc.data.body_pos_w[:, dbi, :2]
        near_drawer = torch.linalg.norm(bowl[:, :2] - drawer_xy, dim=-1) < drawer_r
    except Exception:
        near_drawer = torch.zeros(bowl.shape[0], dtype=torch.bool, device=device)
    height_ok = (bowl[:, 2] > 0.5) & (bowl[:, 2] < 0.85)
    return drawer_open & near_drawer & height_ok


def task2_success_components(env, device, home_radius: float = 0.12):
    """task2: bowl0 actually placed in the (open) drawer AND both arms returned home.
    Returns (in_drawer, close_home, success)."""
    from isaaclab.utils.math import quat_rotate_inverse

    in_drawer = _bowl_actually_in_drawer(env, device)

    robot = env.scene.articulations["robot"]
    ri = robot.find_bodies("ee_link1")[0][0]
    li = robot.find_bodies("ee_link2")[0][0]
    bi = robot.find_bodies("base_link")[0][0]
    base_pos = robot.data.body_pos_w[:, bi]
    base_quat = robot.data.body_quat_w[:, bi]
    eef_r = quat_rotate_inverse(base_quat, robot.data.body_pos_w[:, ri] - base_pos)
    eef_l = quat_rotate_inverse(base_quat, robot.data.body_pos_w[:, li] - base_pos)
    home_r = torch.tensor(_TASK1_HOME_R, device=device)
    home_l = torch.tensor(_TASK1_HOME_L, device=device)
    close_home = (torch.linalg.norm(eef_r - home_r, dim=-1) < home_radius) & (
        torch.linalg.norm(eef_l - home_l, dim=-1) < home_radius
    )
    return in_drawer, close_home, (in_drawer & close_home)


def _chair_entity(env):
    """The chair rigid object this scene's pushchair term targets.

    The SceneSmith pushchair scenes do not agree on a name: scene_001/005/015/016/098/200/203/207
    call it `chair`, scene_014 calls it `chair_0` (it has a second stool, `chair_1`). Both terms
    take the name via SceneEntityCfg, so resolve it from the scene rather than hardcoding one.
    """
    for name in ("chair", "chair_0"):
        if name in env.scene.rigid_objects:
            return env.scene[name]
    raise KeyError(
        "no chair rigid object in this scene (looked for 'chair', 'chair_0') — the pushchair "
        f"scorer cannot apply. Scene has: {sorted(env.scene.rigid_objects)}"
    )


def load_demo_episode(repo: str, kitchen: int, which: int):
    """(actions (T,23) float32, initial_pose (6,)) for one recorded episode of `kitchen`.

    Actions are read exactly as stored — absolute, i.e. the same space the policy emits (openpi's
    DeltaActions/AbsoluteActions pair lives inside the model pipeline, not in the dataset).
    """
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{repo}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {repo}/data")
    eps = {}
    for f in files:
        t = pq.read_table(f, columns=["kitchen_num", "episode_index", "action",
                                      "observation.state", "initial_pose"])
        kn = [v[0] if isinstance(v, list) else v for v in t.column("kitchen_num").to_pylist()]
        ei = [v[0] if isinstance(v, list) else v for v in t.column("episode_index").to_pylist()]
        ac = t.column("action").to_pylist()
        st = t.column("observation.state").to_pylist()
        ip = t.column("initial_pose").to_pylist()
        for k, e, a, s, p in zip(kn, ei, ac, st, ip):
            if k == kitchen:
                slot = eps.setdefault(e, {"actions": [], "states": [], "initial_pose": p})
                slot["actions"].append(a)
                slot["states"].append(s)
    if not eps:
        raise ValueError(f"no episodes with kitchen_num={kitchen} in {repo}")
    keys = sorted(eps)
    key = keys[which % len(keys)]
    actions = np.asarray(eps[key]["actions"], dtype=np.float32)
    states = np.asarray(eps[key]["states"], dtype=np.float32)
    print(f"[replay] kitchen={kitchen} episode_index={key} ({which} of {len(keys)} available) "
          f"actions={actions.shape} initial_pose={[round(v, 4) for v in eps[key]['initial_pose']]}")
    return actions, eps[key]["initial_pose"], states


def _never_terminate(env, **kwargs):
    """A termination term that never fires."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def neutralize_degenerate_retry(env) -> bool:
    """Disable the env's `retry` term if it is already true for EVERY env in the reset state.

    Such a term is not detecting anything — and the cost is far worse than a missed detection:
    `ManagerBasedRLEnv.step()` does `reset_buf = termination_manager.compute()` and immediately
    `_reset_idx(reset_env_ids)`, so a term that is always true RESETS EVERY ENV ON EVERY STEP.
    The rollout then never accumulates any state: the object is teleported back to its initial
    pose each step, the arms snap back to home, and the episode is a 700-step sequence of frame-0
    restarts that still records video and still reports a success rate. Nothing errors.

    `mdp.OOB_chair` is exactly that: `chair_z > -0.03` is true of any chair standing on the floor,
    so SceneSmith 001/005/015/016/200/203/207 were resetting continuously. It shows up in the
    diagnostics as an object that never moves by even a millimetre (`gap_closed = 0.000`) while
    the camera thrashes. scene_014's `OOB_chair_2` (`z > 2`) is correctly false at rest and is
    left alone — this only ever disarms a term that cannot be doing its job.

    Call AFTER env.reset(), so the scene is in its reset state, and BEFORE the rollout.
    Returns True if the term was disabled.
    """
    import copy as _copy

    try:
        cfg = env.termination_manager.get_term_cfg("retry")
    except ValueError:
        return False  # no retry term in this env; nothing to do
    try:
        mask = torch.as_tensor(cfg.func(env, **cfg.params))
    except Exception as exc:  # a term that cannot even be evaluated is not a working detector
        print(f"[retry] WARNING: could not evaluate the retry term ({exc!r}); leaving it in place.")
        return False
    if not bool(mask.all()):
        return False

    new_cfg = _copy.copy(cfg)
    new_cfg.func = _never_terminate
    new_cfg.params = {}
    env.termination_manager.set_term_cfg("retry", new_cfg)
    print(
        f"[retry] WARNING: the env's retry term (mdp."
        f"{getattr(cfg.func, '__name__', '?')}) is TRUE for all {env.num_envs} envs in the reset "
        f"state, so it cannot be an out-of-bounds detector. DISABLED for this run — left active it "
        f"would make Isaac Lab reset every env on every step, and the rollout would be a sequence "
        f"of frame-0 restarts (object never moves, arms always home) that still reports a success "
        f"rate. Fix the term's threshold to restore OOB handling for this scene."
    )
    return True


def chair_desk_distance(env):
    """(N,) distance from the chair to the desk — the quantity `chair_at_desk` thresholds."""
    desk_pos = env.scene.extras["desk"].get_world_poses()[0]
    chair_pos = _chair_entity(env).data.body_pos_w.squeeze(1)
    return torch.linalg.norm(desk_pos[:, :3] - chair_pos[:, :3], dim=-1)


def pushchair_success_components(env, device, chair_radius: float, home_radius: float = 0.12):
    """pushchair / pushchair_2 (mdp.terminations_other): the chair ends up tucked at the desk and
    the right arm returns home. Returns (chair_at_desk, close_home, success), each (N,) bool.

    Mirrors the geometry of the env's own term minus its
    `variable.env_goal_indices == last_subtask` gate — data-generation state-machine state the
    eval never advances (the same reason task1/2/3 have hand-written scorers here).

    `chair_radius` is the term's own: 0.75 m for mdp.pushchair, 0.52 m for mdp.pushchair_2.
    `home_radius` is deliberately RELAXED from the term's 0.05 m / 0.02 m to the 0.12 m the other
    scorers use, so the metric is not gated on millimetre-accurate homing (same choice, and same
    reasoning, as task3_success_components).

    The desk is an AssetBaseCfg, so it lives in `scene.extras` (an XFormPrim), not in
    `scene.rigid_objects` — hence get_world_poses() rather than `.data.body_pos_w`.
    """
    from isaaclab.utils.math import quat_rotate_inverse

    desk_pos = env.scene.extras["desk"].get_world_poses()[0]  # (N,3) world
    chair_pos = _chair_entity(env).data.body_pos_w.squeeze(1)  # (N,3) world
    chair_at_desk = torch.linalg.norm(desk_pos[:, :3] - chair_pos[:, :3], dim=-1) < chair_radius

    robot = env.scene.articulations["robot"]
    ri = robot.find_bodies("ee_link1")[0][0]
    bi = robot.find_bodies("base_link")[0][0]
    base_pos = robot.data.body_pos_w[:, bi]
    base_quat = robot.data.body_quat_w[:, bi]
    eef_r_base = quat_rotate_inverse(base_quat, robot.data.body_pos_w[:, ri] - base_pos)
    home_r = torch.tensor(_TASK1_HOME_R, device=device)
    # Both terms check the RIGHT arm only (their `close_home` reads distance_r alone).
    close_home = torch.linalg.norm(eef_r_base - home_r, dim=-1) < home_radius

    return chair_at_desk, close_home, (chair_at_desk & close_home)


def pushchair_1_success_components(env, device):
    """mdp.pushchair — chair within 0.75 m of the desk."""
    return pushchair_success_components(env, device, chair_radius=0.75)


def pushchair_2_success_components(env, device):
    """mdp.pushchair_2 — chair within 0.52 m of the desk.

    The term ANDs the same chair-to-desk distance twice (`close_chair` and `close_chair_1` both
    read obj_cfg, i.e. chair_0 — chair_1 is never actually measured), so one distance check
    reproduces it exactly. Mirrored as-written: this is the condition scene_014's demos were
    generated against.
    """
    return pushchair_success_components(env, device, chair_radius=0.52)


_SUCCESS_DISPATCH = {
    "task1": (task1_success_components, ("placed", "home")),
    "task2": (task2_success_components, ("in_drawer", "home")),
    "task3": (task3_success_components, ("held", "home")),
    "pushchair": (pushchair_1_success_components, ("chair_at_desk", "home")),
    "pushchair_2": (pushchair_2_success_components, ("chair_at_desk", "home")),
}


def _term_task_name(success_term):
    """`success_term`'s function name, or "" if it has none.

    "" deliberately matches neither a _SUCCESS_DISPATCH key nor "composed" — a term whose func
    lacks __name__ must not silently route to task1_success_components (the mug-in-sink scorer).
    Defaulting to "task1" here would have quietly reintroduced, for this one edge case, exactly
    the silent-fallback failure mode this task exists to remove.
    """
    return getattr(getattr(success_term, "func", None), "__name__", "")


def _success_spec(success_term):
    """The composed spec `success_term` carries, or None for a legacy named term (task1/2/3) — or
    for a `composed` term whose params dict has no usable 'spec' (malformed; see
    _scorability_problem, which reports that case distinctly from an unrecognised task name).

    Takes the DoneTerm object itself, NOT `env.cfg.terminations.success`: main() captures this
    term into a local variable and then sets `env_cfg.terminations.success = None` (so Isaac Lab's
    own termination check doesn't fire; eval controls episode length itself) before `gym.make`
    builds `env` from that same cfg object. By the time rollout() runs, `env.cfg.terminations.
    success` is already None — reading it from there would silently disable the composed path for
    every composed task, not just fall back for the legacy ones.
    """
    if _term_task_name(success_term) != "composed":
        return None
    return (success_term.params or {}).get("spec")


def _scorability_problem(success_term):
    """None if `success_term` can be scored; otherwise a diagnostic string naming the actual
    term, what IS scorable, and the remedy.

    Shared by the pre-flight check in main() (fail fast, before env construction / policy connect
    / reset — all slow) and the raise inside success_components (the runtime backstop for any path
    that reaches scoring without going through pre-flight first).
    """
    task_name = _term_task_name(success_term)
    if task_name in _SUCCESS_DISPATCH:
        return None
    if task_name != "composed":
        return (
            f"the env's success term is func={task_name or '<no __name__>'!r}, which eval has no "
            f"scorer for. Scorable today: {', '.join(sorted(_SUCCESS_DISPATCH))} (hand-written "
            f"legacy checks), or any term with func=mdp.composed carrying a usable spec. This "
            f"kitchen almost certainly predates composable success conditions — older generated "
            f"kitchens declare `func=mdp.<name>` for a bare terminations.py function named sink "
            f"(~1700 of them), mug2sink, in_pot, pot, ramen, or "
            f"task1_molmospace, none of which eval can score. Remedy: regenerate this kitchen's "
            f"env config via task_emit so its success term carries func=mdp.composed."
        )
    spec = _success_spec(success_term)
    if spec is None:
        return (
            "the env's success term is func=mdp.composed, but its params dict has no usable "
            "'spec' — that is a malformed composed term (params missing, empty, or lacking a "
            "'spec' key), not an unrecognised task name. Remedy: regenerate this kitchen's env "
            "config via task_emit; a correctly emitted composed term always carries "
            "params={'spec': <condition>}."
        )
    from predicate_contract import physical_only, unregistered_predicates
    unknown = unregistered_predicates(spec)
    if unknown:
        # physical_only drops an unregistered leaf exactly as it drops a script-phase one, so
        # without this eval would score the condition MINUS that leaf and report the loss as
        # "script-phase leaves" — while the data generator's compile_spec raises KeyError for the
        # same spec. A number produced from a condition the generator would not even run is worse
        # than no number.
        return (
            f"the env's composed success condition names predicate(s) "
            f"{', '.join(repr(p) for p in unknown)}, which the predicate registry does not "
            f"declare. Nothing can evaluate them, and dropping them would score a DIFFERENT "
            f"condition than the one this kitchen's demos were generated against. This spec was "
            f"emitted by a task_emit that knew a predicate this build does not (renamed or "
            f"removed since). Remedy: restore the predicate in predicate_contract.py, or "
            f"regenerate this kitchen's env config from its template."
        )
    if physical_only(spec) is None:
        return (
            "the env's composed success condition has no physical component — every leaf is "
            "script-phase only (e.g. last_subtask), which validate_spec is supposed to refuse at "
            "authoring, so this spec should not have been emitted. Nothing here can be scored; "
            "regenerate this kitchen."
        )
    return None


def _compile_physical(spec):
    """One-time compile of a composed spec's PHYSICAL portion (script-phase leaves dropped).

    physical_only deep-copies its input and compile_spec rebuilds a fresh tree of closures on
    every call; paying that once per rollout is fine, paying it once per simulation step is not
    (mdp.composed keeps its own _COMPILED cache for the identical reason — see
    kitchen/mdp/composed.py). Call this ONCE per rollout, before the step loop: the spec cannot
    change mid-rollout.

    Returns (physical_spec, {label: compiled_leaf_fn}, compiled_whole_fn, dropped_predicate_ids).
    `dropped_predicate_ids` is what the full spec had that the physical projection does not — so
    whenever it is non-empty eval is scoring a DIFFERENT condition than the data generator's, and
    not in a fixed direction: under `all` (everything the composer can author) dropping a leaf can
    only make it easier, under `any` only harder, and under `not` it flips whichever way the
    enclosing operator would have gone. Raises if nothing physical remains, or if the spec names a
    predicate this build has no declaration for; _scorability_problem is meant to catch both before
    rollout() is ever entered, so reaching these raises means something bypassed pre-flight.
    """
    from collections import Counter
    from predicate_contract import leaves, physical_only, unregistered_predicates
    from predicates_math import compile_spec

    unknown = unregistered_predicates(spec)
    if unknown:
        raise ValueError(
            f"this env's success condition names predicate(s) "
            f"{', '.join(repr(p) for p in unknown)}, which the predicate registry does not "
            f"declare. physical_only would DROP them, scoring a different condition than the one "
            f"the demos were generated against — while the generator's compile_spec raises for "
            f"this same spec. Restore the predicate, or regenerate this kitchen's env config."
        )
    physical = physical_only(spec)
    if physical is None:
        raise ValueError(
            "this env's success condition has no physical component — it is script-phase only, "
            "which validate_spec refuses at authoring. Nothing here can be scored."
        )
    per_leaf_fns = {}
    for pid, params in leaves(physical):
        label = f"{pid}:{params.get('role', params.get('arm', ''))}".rstrip(":")
        per_leaf_fns[label] = compile_spec({pid: params})
    whole_fn = compile_spec(physical)
    dropped = sorted((Counter(pid for pid, _ in leaves(spec))
                       - Counter(pid for pid, _ in leaves(physical))).elements())
    return physical, per_leaf_fns, whole_fn, dropped


def composed_success_components(env, physical, per_leaf_fns, whole_fn):
    """Evaluate an ALREADY-COMPILED composed condition (from _compile_physical) against the env's
    CURRENT state.

    The whole point of the composed form: eval does not keep a second definition of success that
    can drift from the one the data generator used. `physical` already has script-phase leaves
    (e.g. last_subtask, which reads variable.env_goal_indices — data-generation state machine
    state eval never advances) dropped, by the _compile_physical call that produced these
    closures.
    """
    from isaaclab_tasks.manager_based.kitchen.mdp.composed import build_context

    ctx = build_context(env, physical, with_script_state=False)
    per_leaf = {label: fn(ctx) for label, fn in per_leaf_fns.items()}
    success = whole_fn(ctx)
    return per_leaf, success


def success_components(env, device, task_name, success_term, compiled=None):
    """(comp1, comp2, success, labels) — the shape rollout()'s per-step check consumes.

    `compiled`, if given, is the (physical, per_leaf_fns, whole_fn) triple _compile_physical
    already produced ONCE for this rollout (see rollout()'s setup) — passing it in avoids
    recompiling the spec on every simulation step. If omitted, this call compiles it itself
    (self-sufficient but slower — not the hot path; rollout()'s own per-step call always passes
    the precompiled triple).
    """
    if compiled is None:
        spec = _success_spec(success_term)
        if spec is not None:
            physical, per_leaf_fns, whole_fn, _dropped = _compile_physical(spec)
            compiled = (physical, per_leaf_fns, whole_fn)
    if compiled is not None:
        physical, per_leaf_fns, whole_fn = compiled
        per_leaf, ok = composed_success_components(env, physical, per_leaf_fns, whole_fn)
        labels = list(per_leaf)
        # The two most informative components, so the existing two-slot readout stays meaningful.
        c1 = per_leaf[labels[0]] if labels else ok
        c2 = per_leaf[labels[1]] if len(labels) > 1 else ok
        return c1, c2, ok, (labels[0] if labels else "cond",
                            labels[1] if len(labels) > 1 else "cond")
    entry = _SUCCESS_DISPATCH.get(task_name)
    if entry is None:
        # Backstop for anything that reaches this point without going through main()'s pre-flight
        # _scorability_problem check first — kept deliberately, not deleted: a silent wrong number
        # is exactly what this plan exists to eliminate, so an unscoreable term must still raise
        # even if pre-flight was somehow bypassed.
        raise ValueError(_scorability_problem(success_term) or
                          f"the env's success term is {task_name!r}, which eval has no scorer for.")
    fn, labels = entry
    c1, c2, ok = fn(env, device)
    return c1, c2, ok, labels


# Objects whose initial height we snapshot (for the "lifted" part of the grasp check).
_TASK_OBJECTS = {"task1": ["mug0"], "task2": ["bowl0"], "task3": ["bottle0", "mug0"]}


def _objects_of_interest(env, task_name, success_term):
    """Objects whose start height is snapshotted for the lifted check."""
    spec = _success_spec(success_term)
    if spec is None:
        return _TASK_OBJECTS.get(task_name, [])
    from predicate_contract import leaves
    return sorted({params["role"] for _, params in leaves(spec)
                   if isinstance(params.get("role"), str)
                   and params["role"] in env.scene.rigid_objects})


def subtask_milestones(env, device, task_name, obj_z0, drawer_ever_open=None,
                       grasp_r: float = 0.15, move_r: float = 1.0, lift: float = 0.03):
    """Per-step boolean for each *subtask* milestone of a full-task rollout, so the composing
    paper subtasks can be scored from one rollout. Returns an ordered dict {label: (N,) bool}.
    Milestones are geometric: move-to (base within move_r of the object, x-y), grasp (object
    within grasp_r of the correct EE AND lifted > `lift` above its start height), place-in-sink,
    open/close drawer (prismatic joint), place-in-drawer (object at drawer height)."""
    if task_name == "composed":
        # Milestones are a per-task hand-written decomposition; a composed task has none yet. The
        # success number itself still comes from the env's own condition.
        return {}
    robot = env.scene.articulations["robot"]

    def base_xy():
        bi = robot.find_bodies("base_link")[0][0]
        return robot.data.body_pos_w[:, bi, :2]

    def ee(side):
        idx = robot.find_bodies("ee_link1" if side == "r" else "ee_link2")[0][0]
        return robot.data.body_pos_w[:, idx]

    def objp(name):
        return env.scene[name].data.body_pos_w.squeeze(1)

    def moved_to(name):
        return torch.linalg.norm(objp(name)[:, :2] - base_xy(), dim=-1) < move_r

    def grasped(name, side):
        o = objp(name)
        return (torch.linalg.norm(o - ee(side), dim=-1) < grasp_r) & (o[:, 2] > obj_z0[name] + lift)

    out = {}
    if task_name == "task1":
        out["T5_move_to_mug"] = moved_to("mug0")
        out["T8_grasp_mug"] = grasped("mug0", "r")
        cs, _, _ = task1_success_components(env, device)
        out["T11_place_mug_sink"] = cs
    elif task_name == "task3":
        out["T7_move_to_bottle"] = moved_to("bottle0")
        out["T10_grasp_bottle"] = grasped("bottle0", "r")
        out["T8b_grasp_mug"] = grasped("mug0", "l")
        hb, _, _ = task3_success_components(env, device)
        out["T27_pour_hold"] = hb
    elif task_name == "task2":
        out["T6_move_to_bowl"] = moved_to("bowl0")
        out["T9_grasp_bowl"] = grasped("bowl0", "r")
        bc = env.scene["base_cabinet"]
        ji = bc.find_joints("corpus_to_drawer_0_0")[0][0]
        dj = bc.data.joint_pos[:, ji]
        out["T18_open_drawer"] = dj > 0.10
        out["T19_place_bowl_drawer"] = _bowl_actually_in_drawer(env, device)
        if drawer_ever_open is not None:
            out["T21_close_drawer"] = drawer_ever_open & (dj < 0.05)
        out["_drawer_pos"] = dj  # consumed by the accumulator, not a milestone
    return out


# --------------------------------------------------
# Tiled overview video (all envs in a grid, green=success / red=fail overlay)
# --------------------------------------------------
def write_tiled_videos(
    frames_np,          # (T, N, H, W, C) uint8
    success_step,       # (N,) int: first frame each env succeeds, -1 = never
    out_dir: Path,
    prefix: str,
    fps: int = 20,
    tile_size: int = 20,
    downscale: int = 2,
    border: int = 3,
    border_color: int = 210,   # gray gridlines
    alpha_green: float = 0.5,  # success tint opacity (stronger/greener)
    alpha_red: float = 0.28,   # fail tint opacity
):
    """Write one or more tiled overview videos, time-aware:
      - a strong GREEN box appears on an env's tile at the frame it first succeeds, and stays;
      - an env that never succeeds is tinted RED from the start of the clip;
      - frames before success (for eventual successes) are left untinted.
    Envs are split into groups of <= tile_size, one video per group:
    <prefix>_tile{g}.mp4 (or <prefix>.mp4 for a single group)."""
    import math

    frames_np = np.asarray(frames_np)
    T, N, H, W, C = frames_np.shape
    success_step = [int(s) for s in success_step]
    if downscale > 1:
        frames_np = frames_np[:, :, ::downscale, ::downscale, :]  # spatial only; time index preserved
        _, _, H, W, C = frames_np.shape

    GREEN = np.array([0, 255, 0], dtype=np.float32)
    RED = np.array([220, 0, 0], dtype=np.float32)

    def _tint(block, color, a):
        return np.clip(block.astype(np.float32) * (1.0 - a) + color * a, 0, 255).astype(np.uint8)

    n_groups = math.ceil(N / tile_size)
    written = []
    for g in range(n_groups):
        idx = list(range(g * tile_size, min((g + 1) * tile_size, N)))
        n = len(idx)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        Hb, Wb = H + border, W + border  # one border strip on top/left of each cell
        grid = np.full((T, rows * Hb + border, cols * Wb + border, C), border_color, dtype=np.uint8)
        for cell, e in enumerate(idx):
            r, c = divmod(cell, cols)
            tile = frames_np[:, e].copy()  # (T,H,W,C) uint8
            ss = success_step[e]
            if ss is None or ss < 0:
                tile = _tint(tile, RED, alpha_red)              # never succeeded -> red from start
            elif ss < T:
                tile[ss:] = _tint(tile[ss:], GREEN, alpha_green)  # strong green from success frame onward
            y0 = r * Hb + border
            x0 = c * Wb + border
            grid[:, y0:y0 + H, x0:x0 + W, :] = tile
        name = f"{prefix}.mp4" if n_groups == 1 else f"{prefix}_tile{g}.mp4"
        path = out_dir / name
        imageio.mimwrite(path, list(grid), fps=fps, codec="libx264")
        n_succ = sum(1 for e in idx if success_step[e] is not None and success_step[e] >= 0)
        written.append((str(path), n, n_succ))
        print(f"[tiled] wrote {path}  ({n} envs, {n_succ} success, grid {rows}x{cols})")
    return written


# --------------------------------------------------
# Main rollout
# --------------------------------------------------
video_dir = Path(f"videos_eval/{args_cli.model}/{args_cli.task_language[:20]}/{args_cli.OOD}")
video_dir.mkdir(parents=True, exist_ok=True)

front_frames = []
wrist_left_frames = []
wrist_right_frames = []


def rollout(
    client,
    env,
    success_term,
    horizon: int,
    device: torch.device,
    data,
    data_subtask,
    ee_link1_idx,
    ee_link2_idx,
    action_chunk_size,
    prompt,
    num_envs,
    replay_actions=None,
):
    global front_frames, wrist_left_frames, wrist_right_frames

    obs_dict, _ = env.reset()

    env.action_manager.get_term("armL_action")._ik_controller.reset()
    env.action_manager.get_term("armR_action")._ik_controller.reset()

    rate_limiter = RateLimiter(args_cli.step_hz)

    # previous per-env targets
    prev_l_xyz = None   # (N, 3)
    prev_r_xyz = None   # (N, 3)
    prev_l_quat = None  # (N, 4)
    prev_r_quat = None  # (N, 4)

    action_chunk = None  # (chunk, N, 23)
    action_idx = 0

    # per-env success accumulators: an env counts as success if the task's physical
    # condition holds at ANY step. The check + component labels are dispatched by the env's
    # registered success-term name (task1: mug-in-sink+home; task3: hold bottle&mug+home).
    _task_name = _term_task_name(success_term)
    # Composed spec compiled ONCE per rollout (not once per simulation step — see
    # _compile_physical's docstring). main()'s pre-flight check already established this term is
    # scorable, so _spec is None here only for a legacy task1/2/3 term.
    _spec = _success_spec(success_term)
    if _spec is not None:
        _physical, _per_leaf_fns, _whole_fn, _dropped_predicates = _compile_physical(_spec)
        _compiled_success = (_physical, _per_leaf_fns, _whole_fn)
    else:
        _compiled_success = None
        _dropped_predicates = []
    ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ever_comp1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ever_comp2 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # first timestep each env achieves success (-1 = never). Drives the overlay timing:
    # green box appears from this frame onward; envs that stay -1 are tinted red.
    success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    # subtask-milestone tracking: snapshot each manipulated object's start height (for the
    # "lifted" grasp check) and accumulate which milestones each env ever reaches.
    obj_z0 = {}
    for _o in _objects_of_interest(env, _task_name, success_term):
        try:
            obj_z0[_o] = env.scene[_o].data.body_pos_w.squeeze(1)[:, 2].clone()
        except Exception:
            obj_z0[_o] = torch.zeros(num_envs, device=device)
    ever_milestone = {}          # label -> (N,) bool accumulator
    drawer_ever_open = torch.zeros(num_envs, dtype=torch.bool, device=device)

    _prev_ee_r_w = None  # diagnostic: previous actual right-EE world pos
    min_chair_desk = None    # chair tasks: closest the chair ever gets to the desk, per env
    start_chair_desk = None  # ... and where it started, so progress is readable
    # Commanded vs achieved base velocity. When a rollout ends with the robot parked against the
    # object and going nowhere, these separate "the policy stopped asking for motion" (cmd ~ 0)
    # from "it asked and the base could not deliver" (cmd >> achieved — a stalled velocity joint).
    base_cmd_hist = []
    base_ach_hist = []
    base_disp_hist = []  # actual world-frame xy displacement of base_link per step
    base_yaw_hist = []   # base yaw at the start of each step
    replay_state_err = []  # replay only: |sim state - demo state| per step, env 0

    front_frames = []
    wrist_left_frames = []
    wrist_right_frames = []

    identity_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).view(1, 4)

    # A retry (out-of-bounds) term that is ALREADY true for every env in the untouched reset
    # state is not detecting anything — it is mis-specified. mdp.OOB_chair is the live example:
    # it fires on `chair_z > -0.03`, which is true of any chair standing on the floor, so it
    # holds from step 0 in every scene that declares it (SceneSmith 001/005/015/016/200/203/207).
    # Left alone it silently destroys the rollout rather than failing: the retry branch below
    # zeroes prev_* every step, and the SIMVLA_SEAM_FIX path then forces a zero arm delta for
    # every retrying env, so the arms never move and the policy is scored on a robot it was
    # never allowed to drive. Detect that at the start of the rollout and ignore the term for the
    # rest of the run, loudly. This cannot mask a real OOB detector: one is false in the reset state
    # (mdp.OOB_chair_2, which scene_014 uses, checks `z > 2` and is correctly false at rest).
    #
    # Checked over steps 0 AND 1, not step 0 alone: env.reset() zeroes the termination manager's
    # buffers and get_term() returns that buffer, so at step_i == 0 every term still reads False
    # no matter what it would compute — the term's reset-state value first becomes visible after
    # the first env.step(). One step in, with the arm delta still zero, the scene has not
    # meaningfully moved, so an all-envs-true there is the same evidence of mis-specification.
    retry_degenerate = False

    for step_i in range(horizon):
        retry_mask = env.termination_manager.get_term("retry").clone()
        if not retry_degenerate and step_i <= 1 and bool(retry_mask.all()):
            retry_degenerate = True
            _retry_fn = getattr(getattr(env.cfg.terminations, "retry", None), "func", None)
            print(
                f"[retry] WARNING: the env's retry term "
                f"(mdp.{getattr(_retry_fn, '__name__', '?')}) is TRUE for all {num_envs} envs, so "
                f"it cannot be an out-of-bounds detector. Ignoring it for the rest of this "
                f"rollout, so it does not zero the arm deltas on every step. NOTE this only "
                f"disarms eval's OWN use of the mask — the env's automatic reset-on-termination "
                f"is upstream of here, and main() calls neutralize_degenerate_retry() before "
                f"stepping to stop that. Reaching this message means that call was bypassed."
            )
        if retry_degenerate:
            retry_mask = torch.zeros_like(retry_mask)
        retry_idx = torch.where(retry_mask)[0]

        if torch.any(retry_mask):
            env.action_manager.get_term("armL_action")._ik_controller.reset(retry_mask)
            env.action_manager.get_term("armR_action")._ik_controller.reset(retry_mask)
            print(f"Reset envs for OBB: {retry_idx.tolist()}")

            if prev_l_xyz is not None:
                prev_l_xyz[retry_mask] = 0.0
                prev_r_xyz[retry_mask] = 0.0
                prev_l_quat[retry_mask] = identity_quat.expand(retry_mask.sum(), -1)
                prev_r_quat[retry_mask] = identity_quat.expand(retry_mask.sum(), -1)

        # --------------------------------------------------
        # batched sensors
        # --------------------------------------------------
        front_rgb = env.scene.sensors["front"].data.output["rgb"]              # (N,...)
        wrist_left_rgb = env.scene.sensors["wrist_left"].data.output["rgb"]    # (N,...)
        wrist_right_rgb = env.scene.sensors["wrist_right"].data.output["rgb"]  # (N,...)

        # save all env videos
        front_frames.append(np.stack([_to_hwc_uint8(front_rgb[i]) for i in range(num_envs)], axis=0))
        wrist_left_frames.append(np.stack([_to_hwc_uint8(wrist_left_rgb[i]) for i in range(num_envs)], axis=0))
        wrist_right_frames.append(np.stack([_to_hwc_uint8(wrist_right_rgb[i]) for i in range(num_envs)], axis=0))

        # --------------------------------------------------
        # infer action chunk
        # --------------------------------------------------
        if step_i % action_chunk_size == 0:
            state = torch.cat(
                (obs_dict["policy"]["ee_6D_pos"], obs_dict["policy"]["base_vel"]),
                dim=1,
            ).to(device=device, dtype=torch.float32)  # (N,D)

            # Rotation blocks first (they are addressed in the RAW layout), then the gripper move.
            state = rot6d_sim_to_dataset(state)
            state = reorder_state_for_policy(state)

            if step_i == 0:
                # What the camera actually hands over, once per rollout: dtype and range decide
                # whether openpi rescales the image behind our back (see infer_action_chunk_per_env).
                _fr = front_rgb[0]
                print(f"[image] front sensor: shape={tuple(_fr.shape)} dtype={_fr.dtype} "
                      f"min={float(_fr.min()):.3f} max={float(_fr.max()):.3f}")
                # Spawn heading. Every recorded episode of every training kitchen starts at
                # yaw ~ 0 (measured: -1.3 to +1.3 deg), because nothing randomises it — each
                # scene's robot_init_pos declares yaw (0.0, 0.0) and eval overrides only x/y.
                # If this ever prints something else, the reset is rotating the robot into a
                # heading the policy has never started from.
                _bi = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
                _, _, _yaw0 = euler_xyz_from_quat(
                    env.scene.articulations["robot"].data.body_quat_w[:, _bi]
                )
                _yaw0 = torch.rad2deg((_yaw0 + math.pi) % (2 * math.pi) - math.pi)
                print(f"[yaw] base yaw at reset (deg): "
                      f"{[round(v, 2) for v in _yaw0.tolist()]}  (training starts: ~0 +/- 1.3)")
                # Where the robot ACTUALLY spawned, env-local. reset_root_state_uniform writes
                # default_root_state + env_origin + sampled_offset to the ROOT, and this robot's
                # root is a fixed link ('world') with the base riding on prismatic joints — so
                # this is the check that the requested start pose took effect at all.
                _bp = (env.scene.articulations["robot"].data.body_pos_w[:, _bi, :2]
                       - env.scene.env_origins[:, :2])
                print(f"[pos] base_link xy at reset (env-local): "
                      f"{[[round(v, 3) for v in e] for e in _bp.tolist()]}")
                # Where the task objects actually are, so the robot's travel direction can be
                # checked against the direction it would need to go.
                try:
                    _ch = (_chair_entity(env).data.body_pos_w.squeeze(1)[:, :2]
                           - env.scene.env_origins[:, :2])
                    _dk = (env.scene.extras["desk"].get_world_poses()[0][:, :2]
                           - env.scene.env_origins[:, :2])
                    print(f"[pos] chair xy={[round(v, 3) for v in _ch[0].tolist()]}  "
                          f"desk xy={[round(v, 3) for v in _dk[0].tolist()]}  "
                          f"robot->chair vector="
                          f"{[round(v, 3) for v in (_ch[0] - _bp[0]).tolist()]}")
                except Exception as _exc:
                    print(f"[pos] chair/desk lookup failed: {_exc!r}")
                # The exact 23-vector the policy is asked to act on, once per rollout. Cheap, and
                # the only way to tell a policy that fails the task from a policy being fed a
                # state outside the distribution it was trained on (compare against the training
                # set's per-dim q01/q99 in the checkpoint's norm_stats.json). Silent state-layout
                # drift between the sim observation and the recorded dataset is invisible
                # otherwise — the rollout just quietly does nothing useful.
                print(f"[state] env0 t=0 as sent to policy: "
                      f"{np.round(tensor_to_numpy(state[0]), 4).tolist()}")

            if replay_actions is not None:
                # Open-loop replay: the next `action_chunk_size` recorded actions, identical for
                # every env. Same shape and same downstream handling as a policy chunk, so the
                # conversion path under test is bit-for-bit the one the policy runs through.
                _seg = replay_actions[step_i:step_i + action_chunk_size]
                if len(_seg) == 0:
                    _seg = replay_actions[-1:]           # demo exhausted: hold the last action
                if len(_seg) < action_chunk_size:
                    _seg = np.concatenate([_seg, np.repeat(_seg[-1:], action_chunk_size - len(_seg), 0)])
                action_chunk = np.repeat(_seg[:, None, :], num_envs, axis=1)  # (chunk, N, 23)
            else:
                action_chunk = infer_action_chunk_per_env(
                    client=client,
                    front_rgb=front_rgb,
                    wrist_left_rgb=wrist_left_rgb,
                    wrist_right_rgb=wrist_right_rgb,
                    state=state,
                    prompt=prompt,
                    num_envs=num_envs,
                )  # (chunk, N, 23)

            # In absolute-pose mode the policy's raw output IS the base-frame EE target
            # (like the real robot's send_action); the world-lift offsets in
            # apply_action_postprocess must NOT be applied.
            if os.environ.get("SIMVLA_ABS_POSE") != "1":
                action_chunk = apply_action_postprocess(action_chunk, device=device)
            action_idx = 0

            # SIMVLA_SEAM_FIX: the new chunk is re-anchored to the current state, so its first
            # pose ~ the current EE. Differencing new_chunk[0] against the previous chunk's stale
            # lookahead target (old_chunk[49]) injects the accumulated open-loop lag as one big
            # step -> the periodic boundary jerk. Resetting prev to None makes this boundary step
            # emit a ZERO arm delta (like the very first step); within-chunk differencing is
            # untouched. Kills only the seam.
            if os.environ.get("SIMVLA_SEAM_FIX") == "1":
                prev_l_xyz = prev_r_xyz = prev_l_quat = prev_r_quat = None
            print(step_i)
            print("new action_chunk shape:", action_chunk.shape)

        # current step action
        if action_chunk is None:
            raise RuntimeError("action_chunk is None before first infer.")

        if action_idx >= action_chunk.shape[0]:
            raise RuntimeError(
                f"action_idx {action_idx} out of range for action_chunk with len {action_chunk.shape[0]}"
            )

        action = torch.as_tensor(
            action_chunk[action_idx],
            device=device,
            dtype=torch.float32,
        )  # (N,23)
        action_idx += 1

        # --------------------------------------------------
        # parse arm pose targets
        # --------------------------------------------------
        l_xyz = action[:, 0:3]
        l_rot6d = action[:, 3:9]
        l_gripper = action[:, 9:10]

        r_xyz = action[:, 10:13]
        r_rot6d = action[:, 13:19]
        r_gripper = action[:, 19:20]

        delta_pose_base = action[:, 20:23]

        l_quat = sixd_to_quat_wxyz(l_rot6d)   # (N,4)
        r_quat = sixd_to_quat_wxyz(r_rot6d)   # (N,4)

        if os.environ.get("SIMVLA_DELTA_VS_EE") == "1":
            # Closed-loop: command = absolute policy target - ACTUAL current EE pose.
            # The env arm term is DifferentialInverseKinematicsActionCfg(use_relative_mode=True),
            # so it applies the delta on top of the real EE. Differencing consecutive policy
            # TARGETS instead (the branch below) injects the accumulated open-loop tracking lag
            # as one big step at every re-inference boundary -> the periodic post-grasp jerk.
            # ee_6D_pos layout (mdp.ee_6d_pos): L_pos(0:3) L_rot6d(3:9) R_pos(9:12) R_rot6d(12:18)
            # in env-local world frame; apply_action_postprocess already lifts the target into
            # that same frame (base offset) and adds the 0.10956 m tool offset, so we add the
            # same tool offset to the actual EE and compare fingertip-to-fingertip.
            ee = obs_dict["policy"]["ee_6D_pos"].to(device=device, dtype=torch.float32)
            aL_R = rot6d_to_R(ee[:, 3:9])
            aR_R = rot6d_to_R(ee[:, 12:18])
            _tool = torch.tensor([0.0, 0.0, 0.10956], device=device, dtype=torch.float32).view(1, 3, 1)
            aL_tip = ee[:, 0:3] + (aL_R @ _tool).squeeze(-1)
            aR_tip = ee[:, 9:12] + (aR_R @ _tool).squeeze(-1)
            aL_quat = matrix_to_quat_wxyz(aL_R)
            aR_quat = matrix_to_quat_wxyz(aR_R)
            d_l_xyz = l_xyz - aL_tip
            d_r_xyz = r_xyz - aR_tip
            d_l_rot = quat_delta_axis_angle(aL_quat, l_quat)
            d_r_rot = quat_delta_axis_angle(aR_quat, r_quat)
        elif prev_l_xyz is None:
            d_l_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            d_r_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            d_l_rot = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            d_r_rot = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        else:
            d_l_xyz = l_xyz - prev_l_xyz
            d_r_xyz = r_xyz - prev_r_xyz
            d_l_rot = quat_delta_axis_angle(prev_l_quat, l_quat)
            d_r_rot = quat_delta_axis_angle(prev_r_quat, r_quat)

        prev_l_xyz = l_xyz.clone()
        prev_r_xyz = r_xyz.clone()
        prev_l_quat = l_quat.clone()
        prev_r_quat = r_quat.clone()

        # SIMVLA_SEAM_FIX (retry path): the OOB-retry handler above zeros prev for reset envs,
        # which would make this step's delta = target - 0 ~ 1 m (a huge jerk). Same root cause
        # as the boundary seam. Force a zero arm delta for just-reset envs this step; prev is
        # updated to the current target above, so the next step differences normally.
        if os.environ.get("SIMVLA_SEAM_FIX") == "1" and torch.any(retry_mask):
            for _d in (d_l_xyz, d_r_xyz, d_l_rot, d_r_rot):
                _d[retry_mask] = 0.0

        # gripper binarize
        _l_grip_raw = l_gripper.clone()
        _r_grip_raw = r_gripper.clone()
        l_gripper = torch.where(
            l_gripper < -0.8,
            torch.ones_like(l_gripper),
            -torch.ones_like(l_gripper),
        )
        r_gripper = torch.where(
            r_gripper < -0.8,
            torch.ones_like(r_gripper),
            -torch.ones_like(r_gripper),
        )

        # --- DIAGNOSTIC (SIMVLA_GRIP_LOG): dump per-step gripper + ACTUAL executed right-EE
        # motion, no behavior change. exec_dr = per-step displacement of the real right EE
        # in world frame -> the ground-truth jerk signal (spikes at chunk boundaries = bug).
        _grip_log_path = os.environ.get("SIMVLA_GRIP_LOG")
        if _grip_log_path:
            _ee_r_w = env.scene.articulations["robot"].data.body_pos_w[:, ee_link1_idx].clone()
            if _prev_ee_r_w is None:
                _exec = torch.zeros(num_envs, device=device)
            else:
                _exec = (_ee_r_w - _prev_ee_r_w).norm(dim=-1)
            _prev_ee_r_w = _ee_r_w
            with open(_grip_log_path, "a") as _gf:
                for _e in range(num_envs):
                    _gf.write(
                        f"{step_i},{_e},"
                        f"{_r_grip_raw[_e,0].item():.4f},{r_gripper[_e,0].item():.1f},"
                        f"{_l_grip_raw[_e,0].item():.4f},{l_gripper[_e,0].item():.1f},"
                        f"{d_r_xyz[_e].norm().item():.5f},{_exec[_e].item():.5f}\n"
                    )

        # --------------------------------------------------
        # base local -> world
        # --------------------------------------------------
        # The base command is BODY-frame and must be rotated into the world frame here, because
        # the base action term drives [base_prismatic_x, base_prismatic_y, base_revolute_z], whose
        # prismatic axes sit before the revolute in the chain and are therefore WORLD-fixed.
        #
        # PROVEN by open-loop replay of a recorded demo through this exact path (2026-08-02,
        # kitchen 1, episode 0, robot pinned to the demo's own start pose). The chair sits at
        # +1.07 m in y from that start:
        #     rotation removed : net travel (+1.135, -0.021) -> chair untouched, success 0/2
        #     rotation applied : net travel (+0.22,  +1.03)  -> chair pushed 0.725 m, success 2/2
        # Only the rotated form reproduces the demo, and it reproduces it exactly.
        #
        # Do NOT be fooled by two checks that look like evidence against this and are not:
        #   * comparing the dataset's recorded base ACTION against its recorded base_vel STATE —
        #     both are stored in the same body-frame convention, so they agree with no rotation
        #     and the comparison says nothing about which frame either is in;
        #   * comparing the command against the base's actual world displacement in sim — by the
        #     time it is compared the rotation has already been applied, so it agrees in either
        #     mode. (The [base] frame check below now uses the RAW pre-rotation command precisely
        #     so it cannot repeat that mistake.)
        #
        # SIMVLA_BASE_NO_YAW_ROTATE=1 disables the rotation (measured to break the task).
        if os.environ.get("SIMVLA_BASE_NO_YAW_ROTATE") == "1":
            delta_pose_base_world = delta_pose_base
        else:
            vx_local = delta_pose_base[:, 0]
            vy_local = delta_pose_base[:, 1]
            omega = delta_pose_base[:, 2]

            base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
            _, _, yaw = euler_xyz_from_quat(
                env.scene.articulations["robot"].data.body_quat_w[:, base_link_idx]
            )  # (N,)

            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            vx_world = cos_yaw * vx_local - sin_yaw * vy_local
            vy_world = sin_yaw * vx_local + cos_yaw * vy_local

            delta_pose_base_world = torch.stack([vx_world, vy_world, omega], dim=1)  # (N,3)

        # --------------------------------------------------
        # final action to env
        # --------------------------------------------------
        if os.environ.get("SIMVLA_ABS_POSE") == "1":
            # Absolute base-frame EE pose per arm: [pos(3), quat_wxyz(4)] = 7 dims,
            # matching the real robot's send_action. l_xyz/r_xyz are the raw (un-postprocessed)
            # base-frame targets; l_quat/r_quat are wxyz from the policy's 6D rotation.
            actions = torch.cat(
                [
                    l_xyz,                 # (N,3) left pos
                    l_quat,                # (N,4) left quat wxyz
                    r_xyz,                 # (N,3) right pos
                    r_quat,                # (N,4) right quat wxyz
                    l_gripper,             # (N,1)
                    r_gripper,             # (N,1)
                    delta_pose_base_world, # (N,3)
                ],
                dim=1,
            )  # (N,19)
        else:
            actions = torch.cat(
                [
                    d_l_xyz,               # (N,3)
                    d_l_rot,               # (N,3)
                    d_r_xyz,               # (N,3)
                    d_r_rot,               # (N,3)
                    l_gripper,             # (N,1)
                    r_gripper,             # (N,1)
                    delta_pose_base_world, # (N,3)
                ],
                dim=1,
            )  # (N,17)

        env_ids = torch.arange(num_envs, device=device)

        # RAW, pre-rotation command — comparing the post-rotation value against the resulting
        # world displacement is tautological and previously produced a confident wrong answer.
        base_cmd_hist.append(delta_pose_base.detach().clone())
        _bi_diag = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
        _base_xy_before = env.scene.articulations["robot"].data.body_pos_w[:, _bi_diag, :2].clone()
        _, _, _yaw_before = euler_xyz_from_quat(
            env.scene.articulations["robot"].data.body_quat_w[:, _bi_diag]
        )

        obs_tuple = env.step(actions, env_ids)
        obs_dict = obs_tuple[0]

        base_ach_hist.append(obs_dict["policy"]["base_vel"].detach().clone().float())

        # Replay only: how far has the robot's ACTUAL state drifted from the demo's recorded
        # state at the same timestep? The demo is ground truth, so a growing error localises the
        # defect to a specific block of the state (arm position, arm rotation, or base).
        _rs = globals().get("REPLAY_STATES")
        if replay_actions is not None and _rs is not None and step_i < len(_rs):
            _st_now = reorder_state_for_policy(rot6d_sim_to_dataset(torch.cat(
                (obs_dict["policy"]["ee_6D_pos"], obs_dict["policy"]["base_vel"]), dim=1
            ).to(device=device, dtype=torch.float32)))
            _demo_st = torch.as_tensor(_rs[step_i], device=device, dtype=torch.float32)
            replay_state_err.append((_st_now[0] - _demo_st).abs().detach().clone())
        # Which frame is the base command actually in? Compare the commanded (vx, vy) against the
        # base's ACTUAL world displacement this step, both as-is and rotated by the base yaw.
        # Whichever agrees is the truth, independent of any reasoning about the joint chain.
        _base_xy_after = env.scene.articulations["robot"].data.body_pos_w[:, _bi_diag, :2]
        base_disp_hist.append((_base_xy_after - _base_xy_before).detach().clone())
        base_yaw_hist.append(_yaw_before.detach().clone())

        # per-env success check on the freshly stepped state (task-aware)
        _c1, _c2, _succ, _labels = success_components(
            env, device, _task_name, success_term, _compiled_success
        )
        _newly = _succ & (success_step < 0)
        success_step[_newly] = step_i
        ever_success |= _succ
        ever_comp1 |= _c1
        ever_comp2 |= _c2

        # For the chair tasks, track how close the chair ever gets to the desk. A bare
        # `chair_at_desk=False` cannot distinguish "the policy pushed it most of the way",
        # "the policy never touched it", and "the success radius is unreachable in this scene"
        # — the per-episode minimum separates all three, and it is one norm per step.
        if _task_name.startswith("pushchair"):
            _d = chair_desk_distance(env)
            min_chair_desk = _d if min_chair_desk is None else torch.minimum(min_chair_desk, _d)
            if step_i == 0:
                start_chair_desk = _d.clone()

        # subtask milestones (paper subtasks composing this full task)
        try:
            _ms = subtask_milestones(env, device, _task_name, obj_z0, drawer_ever_open)
            _dj = _ms.pop("_drawer_pos", None)
            if _dj is not None:
                drawer_ever_open |= _dj > 0.10
            for _k, _v in _ms.items():
                ever_milestone[_k] = ever_milestone.get(_k, torch.zeros(num_envs, dtype=torch.bool, device=device)) | _v
        except Exception as _e:
            if step_i == 0:
                print(f"[subtask] detection skipped: {_e}")

        if rate_limiter:
            rate_limiter.sleep(env)

    if _dropped_predicates:
        # Once per run, next to the rate it qualifies: eval scores physical_only(spec), so a
        # success like all[obj_z, eef_home, last_subtask] is checked here as all[obj_z, eef_home]
        # — NOT the condition the data generator applied. Which way that moves the number depends
        # on the operator the leaf sat under, so the message must not claim a direction: under
        # `all` a dropped leaf can only make success easier, under `any` only harder, and under
        # `not` it flips whichever the enclosing operator would have given.
        print(
            f"[success] composed condition scored WITHOUT script-phase leaves (dropped: "
            f"{', '.join(_dropped_predicates)}) — this is not the condition the data generator "
            f"applied, so this rate is not directly comparable to the generator's. Dropping a "
            f"leaf makes success EASIER under 'all', HARDER under 'any', and reverses either of "
            f"those under 'not'."
        )
    print(
        f"[success] task={_task_name} per-env success={ever_success.tolist()}  "
        f"({_labels[0]}={ever_comp1.tolist()}, {_labels[1]}={ever_comp2.tolist()})  "
        f"rate={ever_success.float().mean().item():.3f}"
    )
    if min_chair_desk is not None:
        # Success needs this below the term's radius (0.75 m for pushchair, 0.52 m for
        # pushchair_2). Printing start -> closest makes a 0.000 rate readable: whether the chair
        # moved toward the desk at all, and how much of the gap the policy actually closed.
        _closed = (start_chair_desk - min_chair_desk)
        print(
            f"[chair] desk distance per env: start={[round(v, 3) for v in start_chair_desk.tolist()]}  "
            f"closest={[round(v, 3) for v in min_chair_desk.tolist()]}  "
            f"gap_closed={[round(v, 3) for v in _closed.tolist()]}"
        )
    if base_cmd_hist:
        _cmd = torch.stack(base_cmd_hist)   # (T,N,3) commanded  [vx, vy, wz]
        _ach = torch.stack(base_ach_hist)   # (T,N,3) achieved joint velocity
        _tail = slice(int(0.75 * _cmd.shape[0]), None)   # the end of the episode = the stall
        for _name, _sl in (("whole episode", slice(None)), ("last quarter", _tail)):
            _c = _cmd[_sl].abs().mean(dim=(0, 1))
            _a = _ach[_sl].abs().mean(dim=(0, 1))
            print(f"[base] {_name:13s} mean|commanded| vx,vy,wz = "
                  f"{[round(v, 4) for v in _c.tolist()]}   mean|achieved| = "
                  f"{[round(v, 4) for v in _a.tolist()]}")
        # Frame check: cosine between the commanded xy and the base's ACTUAL world displacement,
        # as-is (world-frame command) vs rotated by the base yaw (body-frame command). ~+1 marks
        # the true convention; if the yaw-rotated variant wins, the pass-through is wrong and
        # SIMVLA_BASE_YAW_ROTATE=1 is the correct default.
        _d = torch.stack(base_disp_hist)                 # (T,N,2) actual world displacement
        _y = torch.stack(base_yaw_hist)                  # (T,N)
        _cxy = _cmd[..., :2]
        _cos = torch.cos(_y); _sin = torch.sin(_y)
        _rot = torch.stack([_cos * _cxy[..., 0] - _sin * _cxy[..., 1],
                            _sin * _cxy[..., 0] + _cos * _cxy[..., 1]], dim=-1)
        _moving = _d.norm(dim=-1) > 1e-4                 # ignore steps where nothing moved
        def _agree(a):
            num = (a * _d).sum(-1)
            den = a.norm(dim=-1) * _d.norm(dim=-1) + 1e-9
            c = (num / den)[_moving]
            return float(c.mean()) if c.numel() else float("nan")
        # Net travel. In replay mode this is directly comparable to the demo's own net travel
        # (integrate its recorded base velocities), which says where the robot SHOULD have ended
        # up. A mismatch localises the divergence to the base path rather than the arms.
        _net = _d.sum(dim=0)                              # (N,2) total world displacement
        print(f"[base] net world travel per env (dx,dy): "
              f"{[[round(v, 3) for v in e] for e in _net.tolist()]}")
        _cmd_net = (_cmd[..., :2] * (1.0 / max(int(args_cli.step_hz), 1))).sum(dim=0)
        print(f"[base] net travel implied by the commands (dx,dy): "
              f"{[[round(v, 3) for v in e] for e in _cmd_net.tolist()]}")
        print(f"[base] frame check over {int(_moving.sum())} moving steps: "
              f"cos(commanded_as_is, actual_world_disp)={_agree(_cxy):+.3f}   "
              f"cos(yaw_rotated_command, actual_world_disp)={_agree(_rot):+.3f}   "
              f"(the one near +1 is the true frame; mean|yaw|={float(_y.abs().mean()):.2f} rad)")
    if replay_state_err:
        _e = torch.stack(replay_state_err)   # (T,23)
        _blocks = {"armA_pos": slice(0, 3), "armA_rot": slice(3, 9), "armA_grip": slice(9, 10),
                   "armB_pos": slice(10, 13), "armB_rot": slice(13, 19), "armB_grip": slice(19, 20),
                   "base_vel": slice(20, 23)}
        print("[replay] |sim state - demo state|, mean over first/last quarter of the episode:")
        _q = max(1, _e.shape[0] // 4)
        for _name, _sl in _blocks.items():
            print(f"[replay]   {_name:10s} first={_e[:_q, _sl].mean().item():.4f}  "
                  f"last={_e[-_q:, _sl].mean().item():.4f}  max={_e[:, _sl].max().item():.4f}")
    if ever_milestone:
        print("[subtask] per-milestone success rate (fraction of envs reaching it):")
        for _k, _acc in ever_milestone.items():
            print(f"[subtask]   {_k:24s} {_acc.float().mean().item():.3f}   {_acc.tolist()}")

    # SIMVLA_NO_VIDEO: fast mode for multi-kitchen sweeps — skip all video encoding
    # (the slow part), just return the success vector.
    if os.environ.get("SIMVLA_NO_VIDEO") == "1":
        print("[video] skipped (SIMVLA_NO_VIDEO=1)")
        return ever_success

    # --------------------------------------------------
    # save all env videos
    # --------------------------------------------------
    fps = 20

    front_frames_np = np.stack(front_frames, axis=0)          # (T,N,H,W,C)
    wrist_left_frames_np = np.stack(wrist_left_frames, axis=0)
    wrist_right_frames_np = np.stack(wrist_right_frames, axis=0)

    print("front_frames all shape:", front_frames_np.shape)
    print("wrist_left_frames all shape:", wrist_left_frames_np.shape)
    print("wrist_right_frames all shape:", wrist_right_frames_np.shape)

    for env_i in range(num_envs):
        imageio.mimwrite(
            video_dir / f"test_front_{env_i}.mp4",
            front_frames_np[:, env_i],
            fps=fps,
            codec="libx264",
        )
        imageio.mimwrite(
            video_dir / f"wrist_left_env{env_i}.mp4",
            wrist_left_frames_np[:, env_i],
            fps=fps,
            codec="libx264",
        )
        imageio.mimwrite(
            video_dir / f"wrist_right_env{env_i}.mp4",
            wrist_right_frames_np[:, env_i],
            fps=fps,
            codec="libx264",
        )

    # tiled overview of the front camera: all envs in a grid, green=success / red=fail
    try:
        write_tiled_videos(
            front_frames_np,
            success_step.tolist(),
            video_dir,
            prefix="tiled_front",
            fps=fps,
            tile_size=args_cli.tile_size,
            downscale=args_cli.tile_downscale,
        )
    except Exception as _e:
        print(f"[tiled] skipped (error building tiled video): {_e}")

    return ever_success  # (N,) bool: task completed at any step, per env



def subtask_mobile(base_pose, goal_pose, pos_tol=0.03, yaw_tol_deg=10):
    """
    base_pose: (N, 3) tensor of [x, y, yaw]
    goal_pose: (N, 3) tensor or (3,) tensor/list of [x, y, yaw]
    pos_tol:   position tolerance in meters (default 0.03 m = 3 cm)
    yaw_tol_deg: yaw tolerance in degrees (default 10°)

    Returns:
        success: (N,) bool tensor – True where pose is close enough.
    """
    if not isinstance(base_pose, torch.Tensor):
        base_pose = torch.as_tensor(base_pose, dtype=torch.float32)
    device, dtype = base_pose.device, base_pose.dtype

    if not isinstance(goal_pose, torch.Tensor):
        goal_pose = torch.as_tensor(goal_pose, dtype=dtype, device=device)
    else:
        goal_pose = goal_pose.to(device=device, dtype=dtype)

    if goal_pose.ndim == 1:
        goal_pose = goal_pose.unsqueeze(0).expand_as(base_pose)

    pos_err = torch.norm(base_pose[..., :2] - goal_pose[..., :2], dim=-1)

    yaw_err = base_pose[..., 2] - goal_pose[..., 2]
    yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
    yaw_tol = yaw_tol_deg * math.pi / 180.0

    print(pos_err, yaw_err)
    success = (pos_err <= pos_tol) & (torch.abs(yaw_err) <= yaw_tol)
    print(pos_err <= pos_tol)
    print(torch.abs(yaw_err) <= yaw_tol)
    print(success)
    return success


def subtask_manipulation(eef_curr, eef_goal, pos_tol=0.02, rot_tol_deg=10.0):
    """
    eef_curr: (N, 7) tensor/array of [x, y, z, w, x, y, z]
    eef_goal: (7,) or (N, 7) tensor/array of [x, y, z, w, x, y, z]
    pos_tol:  position tolerance in meters (default 0.02 m = 2 cm)
    rot_tol_deg: rotation tolerance in degrees (default 10°)

    Returns:
        success: (N,) bool tensor – True where pose is close enough.
    """
    if not isinstance(eef_curr, torch.Tensor):
        eef_curr = torch.as_tensor(eef_curr, dtype=torch.float32)
    device, dtype = eef_curr.device, eef_curr.dtype

    if not isinstance(eef_goal, torch.Tensor):
        eef_goal = torch.as_tensor(eef_goal, dtype=dtype, device=device)
    else:
        eef_goal = eef_goal.to(device=device, dtype=dtype)

    if eef_goal.ndim == 1:
        eef_goal = eef_goal.unsqueeze(0).expand_as(eef_curr)

    curr_pos = eef_curr[..., :3]
    curr_quat = eef_curr[..., 3:]  # (N, 4) wxyz
    goal_pos = eef_goal[..., :3]
    goal_quat = eef_goal[..., 3:]  # (N, 4) wxyz

    pos_err = torch.norm(curr_pos - goal_pos, dim=-1)

    curr_quat = curr_quat / torch.norm(curr_quat, dim=-1, keepdim=True)
    goal_quat = goal_quat / torch.norm(goal_quat, dim=-1, keepdim=True)

    dot = torch.sum(curr_quat * goal_quat, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    dot = torch.abs(dot)

    rot_err = 2.0 * torch.arccos(dot)
    print(pos_err, rot_err)
    rot_tol = rot_tol_deg * math.pi / 180.0

    success = (pos_err <= pos_tol) & (rot_err <= rot_tol)
    return success


# ----------------------- Evaluation -----------------------
def evaluate_policy_on_seed(
    client,
    env: gym.Env,
    device: torch.device,
    success_term,
    num_envs: int,
    horizon: int,
    seed: int,
    output_file: str,
    data,
    data_subtask,
    ee_link1_idx,
    ee_link2_idx,
    repo_id,
    action_chunk,
    prompt,
    replay_actions=None,
) -> float:
    """Evaluate policy for one seed (no eval settings loop)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if hasattr(env, "seed"):
        env.seed(seed)

    success = rollout(
        client, env, success_term, horizon, device,
     data, data_subtask, ee_link1_idx, ee_link2_idx, action_chunk, prompt, num_envs,
     replay_actions=replay_actions
    )
    # `success` is a per-env (N,) bool tensor: task completed at any step.
    success = torch.as_tensor(success).float()
    success_rate = float(success.mean().item())
    return success_rate


def main() -> None:
    # -------------- Isaac Lab env config --------------
    num_envs=args_cli.num_envs

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
#        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.observations.policy.concatenate_terms = False

    # SIMVLA_ABS_POSE: command absolute EE poses (base frame) like the real robot
    # (send_action), instead of frame-to-frame deltas. Switches the arm IK terms from
    # relative to absolute mode: each arm command becomes [pos(3), quat_wxyz(4)] = 7 dims.
    if os.environ.get("SIMVLA_ABS_POSE") == "1":
        env_cfg.actions.armL_action.controller.use_relative_mode = False
        env_cfg.actions.armR_action.controller.use_relative_mode = False
        print("[SIMVLA_ABS_POSE] arm IK set to ABSOLUTE pose mode (7-dim per arm)")

    # Terminations & eval mode
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Fail fast: this kitchen's success term may be unscorable — an older generated kitchen whose
    # func name predates composable conditions (mdp.sink, mdp.mug2sink, mdp.in_pot, mdp.pot,
    # mdp.pushchair, mdp.pushchair_2, mdp.ramen, mdp.task1_molmospace, ...), or a malformed
    # composed term. Find out BEFORE building the env / connecting to the policy / resetting — all
    # slow — so a bad kitchen fails in seconds, not after minutes of setup. success_components'
    # own raise (inside rollout()) is the backstop for anything that reaches scoring without going
    # through this check first; it is intentionally NOT removed.
    _problem = _scorability_problem(success_term)
    if _problem is not None:
        raise SystemExit(f"[simvla_eval] refusing to run — {_problem}")

    env_cfg.eval_mode = True

    file_path, file_sub_path = (str(p) for p in simvla_paths.goal_files(args_cli.task))

    with open(file_sub_path, 'r') as file:
        data_subtask = json.load(file)

    with open(file_path, 'r') as file:
        data = json.load(file)

        if args_cli.OOD == "False":
            from datasets import load_dataset
            repo_id = args_cli.repo_id
            if not repo_id:
                hf_user = os.environ.get("HF_USER")
                if not hf_user:
                    raise ValueError("ID evaluation requires --repo_id or HF_USER")
                repo_id = f"{hf_user}/{args_cli.data}"
            print(f"Loading dataset: {repo_id}")
            ds = load_dataset(repo_id, split="train")
            first_idxs = np.nonzero(ds["is_first"])[0]
            idx = random.choice(first_idxs)
            init = ds[idx]

            # Initial base pose
            init_base = init["initial_pose"]
            x, y, quat = init_base[0], init_base[1], init_base[2:]
            r, p, yaw = euler_xyz_from_quat(torch.tensor(quat, device='cuda:0').unsqueeze(0))
            env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (x, x + 0.0001)
            env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (y, y + 0.0001)

        elif args_cli.OOD == "True":
            # 1. Init pos
            init_pos = random.choice(data["initial_pos_ranges"])
            env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (init_pos[0][1], init_pos[0][2])
            env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (init_pos[1][1], init_pos[1][2])
            # 2. Init rot (TODO)

#    env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (0.5, 0.51)
#    env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (-1.2, -1.19)

    # Open-loop replay: start exactly where the recorded episode started, so the demo's actions
    # are being applied from the demo's own initial condition. Only x/y are pinned — the recorded
    # initial yaw is ~0 for every episode, which is already the scene default (see [yaw] readout).
    replay_actions = None
    if args_cli.replay_kitchen is not None:
        replay_actions, _replay_init, _replay_states = load_demo_episode(
            args_cli.replay_repo, args_cli.replay_kitchen, args_cli.replay_episode
        )
        globals()["REPLAY_STATES"] = _replay_states
        _rx, _ry = float(_replay_init[0]), float(_replay_init[1])
        env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (_rx, _rx + 0.0001)
        env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (_ry, _ry + 0.0001)
        print(f"[replay] pinning start pose to the demo's: x={_rx:.4f} y={_ry:.4f}")
    elif args_cli.start_kitchen is not None:
        _, _start_init, _ = load_demo_episode(
            args_cli.replay_repo, args_cli.start_kitchen, args_cli.replay_episode
        )
        _sx, _sy = float(_start_init[0]), float(_start_init[1])
        env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (_sx, _sx + 0.0001)
        env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (_sy, _sy + 0.0001)
        print(f"[start] policy in control, but starting from the demo's pose: x={_sx:.4f} y={_sy:.4f}")

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[ik] per-joint DLS deadzone set to {apply_ik_joint_deadzone(env):g} rad", flush=True)
    env.reset()

    # Must come after reset (needs the reset state) and before any stepping: a retry term that is
    # true at reset would otherwise reset every env on every step for the whole rollout.
    neutralize_degenerate_retry(env)

    ee_link1_idx = env.scene.articulations['robot'].find_bodies("ee_link1")[0][0]
    ee_link2_idx = env.scene.articulations['robot'].find_bodies("ee_link2")[0][0]

    # Device
    if args_cli.device is not None:
        device = torch.device(args_cli.device)
    else:
        print("ji")
#        device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Policy client — skipped entirely in replay mode (WebsocketClientPolicy retries forever, so
    # constructing it without a server would hang before the replay ever started).
    if replay_actions is None:
        client = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host_ip, port=8000)
        print("Connected to Openpi server.")
    else:
        client = None
        print("[replay] open-loop demo replay — no policy server involved.")

    # Seeds
    if args_cli.seeds is None:
        seeds = random.sample(range(0, 10000), args_cli.num_seeds)
    else:
        seeds = args_cli.seeds

    # Logs
    os.makedirs(args_cli.log_dir, exist_ok=True)

    # Per-seed evaluation (single run, no settings)
    for seed in seeds:
        output_path = os.path.join(args_cli.log_dir, f"{args_cli.log_file}_seed_{seed}")
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"Task: {args_cli.task}\n")
            f.write(f"Seed: {seed}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Evaluation (single run, no settings)\n")
            f.write("=" * 80 + "\n\n")

        print("Evaluation (single run, no settings)")
        print("=" * 80)

        sr = evaluate_policy_on_seed(
            client=client,
            env=env,
            device=device,
            success_term=success_term,
            num_envs=num_envs,
            horizon=args_cli.horizon,
            seed=seed,
            output_file=output_path,
            data=data,
            data_subtask=data_subtask,
            ee_link1_idx=ee_link1_idx,
            ee_link2_idx=ee_link2_idx,
            repo_id=args_cli.repo_id,
            action_chunk=args_cli.action_chunk,
            prompt=args_cli.task_language,
            replay_actions=replay_actions
        )

        with open(output_path, "a") as f:
            f.write("\nSummary:\n")
            f.write(f"success_rate: {sr}\n")

        env.reset()

    env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback
        print(traceback.format_exc(), flush=True)
        exit_code = 1
    finally:
        from workflow_status import record_status
        record_status(exit_code)
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        simulation_app.close()

    raise SystemExit(exit_code)
