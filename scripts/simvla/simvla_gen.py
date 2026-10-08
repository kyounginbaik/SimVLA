"""
SimVLA: SimAction generation
"""

import argparse
import importlib.util
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

# Runtime-context plumbing (see scripts/simvla/skill_runtime.py).
# Pure-Python, no Isaac/USD imports, safe to import here before AppLauncher.
import sys as _simvla_sys
_simvla_sys.path.insert(0, f"{SIMVLA_REPO_ROOT}/scripts/simvla")
from skill_runtime import RuntimeContext  # noqa: E402
from workflow_status import close_with_status  # noqa: E402
from hdf5_compat import write_success_if_missing  # noqa: E402
from recording_state import ensure_initial_snapshot  # noqa: E402
from collector_profile import (  # noqa: E402
    augmentation_seeds,
    cartesian_fallback_allowed,
    anchor_relative_ik_target,
    arm_grasp_geometries,
    apply_ik_joint_deadzone,
    choose_pregrasp_back_sign,
    configure_physics_substeps,
    goal_noise_standard_deviations,
    grasp_geometry,
    gripper_open_target_reached,
    jaw_gap_shows_object_contact,
    jaw_contact_midpoint,
    measured_locked_joints,
    post_lift_object_retained,
    postrelease_retreat_target,
    postrelease_final_home_step,
    left_postgrasp_lift_enabled,
    mug_grasp_position_tolerance,
    mug_grasp_target_height,
    mug_jaw_contact_ready,
    mug_pad_contacts_loaded,
    mug_pre_lift_grasp_ready,
    advance_pregrasp_stage,
    pinch_axis_xy_correction,
    pinch_center_z_correction,
    planar_hold_command,
    recorded_episode_env_ids,
    track_loaded_home_reset,
    loaded_home_arm_indices,
    loaded_home_hold_allowed,
)

# `parse_export_groups` is --export_groups' `type=` (below), and it is ALSO what run_config_merge
# applies to the same field coming out of a goal file. One definition, imported, because two would
# drift — and the drift only surfaces in finalize_lerobot, i.e. after the run has recorded its
# dataset. It lives over there because run_config_merge is stdlib-only and importable; this file is
# not (it boots Isaac), so the arrow cannot point the other way. Imported HERE, before the parser is
# built, for the same reason as skill_runtime: neither pulls in Isaac.
from run_config_merge import (  # noqa: E402
    merge_run_config, parse_export_groups, validate_grasp_check_steps,
)
from task_runconfig import GRASP_CHECK_DISABLED  # noqa: E402
from select_goal_grasp import side_approach_indices  # noqa: E402

#argparse arguments
parser = argparse.ArgumentParser(description="Motion Planning Anubis in IsaacLab")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."   
)
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v01-01", help="Name of the task.")
parser.add_argument("--task_language", type=str, default="Put bottle to the sink.", help="Name of the task in high level language.")
parser.add_argument("--task_type", type=str, default="LocoManipulation", help="Type of the task. Among Navigation, Manipulation, NavManipulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--robot", choices=("anubis", "aiworker", "rby1"), default="anubis",
    help="Robot used for demonstration collection.",
)
parser.add_argument("--record", type=bool, default=False, help="Whether to record the simulation.")
parser.add_argument(
    "--dump_frames", type=str, default="",
    help="Directory to write env 0's RGB frames to, as PNGs. Off unless set. The scene already "
         "carries front/wrist cameras but nothing ever writes them, so a run that misbehaves can "
         "only be read through printed counters. Encode with e.g. "
         "`ffmpeg -framerate 30 -i frame_%%06d.png out.mp4`.",  # %% : argparse %-formats help
)
parser.add_argument(
    "--dump_camera", type=str, default="front",
    help="Which camera --dump_frames records. `front` is bolted to the robot's base_link and is a "
         "200-degree fisheye, so the robot never moves in shot and the kitchen sweeps past it -- "
         "fine for reading what the policy sees, actively misleading for judging where the base "
         "went. Use `scene_cam` (world-fixed pinhole) for anything a human has to judge.",
)
parser.add_argument(
    "--dump_every", type=int, default=10,
    help="Write one frame every N sim steps when --dump_frames is set.",
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/anubis/Isaac_Kitchen_v1_1.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=20, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed Python, NumPy, Torch, and the Isaac Lab environment (default: 0).")
parser.add_argument(
    "--action_chunk_size",
    type=int,
    default=50,
    help=(
        "VLA policy action-chunk size"
    ),
)
parser.add_argument(
    "--num_demos", type=int, default=128, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=20,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--pour_pot",
    action="store_true",
    help="Toggle to fix the initial robot pose."
)
parser.add_argument(
    "--robot_fix_init",
    action="store_true",
    help="Toggle to fix the initial robot pose."
)
parser.add_argument(
    "--obj_fix_init",
    action="store_true",
    help="Toggle to fix the initial robot pose or not.",
)
parser.add_argument(
    "--save_each_subtask",
    action="store_true",
    help="Toggle to save each subtask in lerobot format."
)
parser.add_argument(
    "--skip_finalize",
    action="store_true",
    help="Only save raw demos to 000000/000001/... and skip building LeRobot datasets.",
)
parser.add_argument(
    "--run_id", 
    type=str, 
    default=None,
    help="Fixed run id (e.g. 20260106_053328). If not set, use current time."
)
parser.add_argument(
    "--output_root",
    type=str,
    default=None,
    help="Root dir for datasets. Default: ./datasets/<robot>"
)
parser.add_argument(
    "--obj_name",
    type=str,
    default="mug0",
    help="Which object is this task about",
)
parser.add_argument(
    "--obj_name_l",
    type=str,
    default="none",
    help="Which object is this task about",
)
parser.add_argument(
    "--sub_grasp_idx_r",
    type=int,
    default=4,
    help="Goal index at which to run RIGHT-arm sub-grasp check (once per env)."
)
parser.add_argument(
    "--sub_grasp_idx_l",
    type=int,
    default=9,
    help="Goal index at which to run LEFT-arm sub-grasp check (once per env)."
)
parser.add_argument("--sub_good_goal_count_r", type=int, default=7,
    help="Collect this many RIGHT-subgrasp-good goals, then sample only from them.")
parser.add_argument("--sub_good_goal_count_l", type=int, default=7,
    help="Collect this many LEFT-subgrasp-good goals, then sample only from them.")

parser.add_argument("--sub_good_goals_r_in", type=str, default=None)
parser.add_argument("--sub_good_goals_r_out", type=str, default=None)
parser.add_argument("--sub_good_goals_l_in", type=str, default=None)
parser.add_argument("--sub_good_goals_l_out", type=str, default=None)
parser.add_argument(
    "--good_goal_count",
    type=int,
    default=3,
    help="Collect this many successful (good) goal scripts, then sample only from them."
)
parser.add_argument(
    "--good_goals_in",
    type=str,
    default=None,
    help="Path to a saved good-goals JSON. If provided, skip filtering and sample only from it."
)
parser.add_argument(
    "--good_goals_out",
    type=str,
    default=None,
    help="Where to save good-goals JSON whenever a new good goal is found. Default: <output_dir>/good_goals_<task>.json"
)
parser.add_argument(
    "--export_groups",
    type=parse_export_groups,
    default=None,
    help="Optional. Export grouped sub-idx sets. Example: '0,1,2;3,4' -> [[0,1,2],[3,4]]. "
         "Use ';' between groups, ',' inside group.",
)
parser.add_argument(
    "--target_idx",
    type=int,
    default=1,
    help="Which idx will get noise from randomization of obj."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Validate the selected external robot USD before Kit acquires a GPU or spends time compiling.
simvla_paths.validate_robot_model_file(args_cli.robot)

# Refuse a missing repository-specific cuRobo helper before Kit grabs a GPU or spends minutes
# compiling shaders. `simvla doctor` performs the same source-level check without importing CUDA.
_curobo_spec = importlib.util.find_spec("curobo")
_curobo_helper = (
    _SimvlaPath(next(iter(_curobo_spec.submodule_search_locations))) / "util/usd_helper.py"
    if _curobo_spec and _curobo_spec.submodule_search_locations else None
)
if not _curobo_helper or not _curobo_helper.is_file() or \
        "get_obstacles_from_stage_simvla" not in _curobo_helper.read_text():
    raise RuntimeError(
        "The installed cuRobo is missing SimVLA's USD obstacle-frame compatibility patch. "
        "From the repository root, apply third_party/curobo.simvla.patch to "
        "third_party/curobo and reinstall it as documented in env/README.md."
    )

#: THE COMMAND LINE, AS THE USER TYPED IT. Snapshotted here because the merge in main() has to know
#: which flags were EXPLICITLY PASSED, and a parsed Namespace cannot say: `--target_idx 1` and no
#: flag at all are both `args_cli.target_idx == 1`. And snapshotted BEFORE AppLauncher, which edits
#: sys.argv in place — it appends its own kit/livestream args and filters them out again
#: (app_launcher.py:760, 789-794), so sys.argv down in main() is no longer the list this process
#: started with.
CLI_ARGV = list(_simvla_sys.argv)

app_launcher_args = vars(args_cli)
# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Import basic packages
import torch, queue, threading
from pathlib import Path
import gymnasium as gym
import gc
# Import Omniverse logger
import omni.log
import numpy as np
import json
# Import for reset
# Import for record Demo
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;8.9"
os.environ["TORCH_EXTENSIONS_DIR"] = f"/scratch/{os.environ['USER']}/torch_extensions"
print("ARCH=", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("EXT =", os.environ.get("TORCH_EXTENSIONS_DIR"))
import time
import itertools
import concurrent.futures
import random
import copy
from datetime import datetime
import multiprocessing as mp
import contextlib
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
    InitialStateRecorderCfg,
)
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.simvla import variable

# THIS IMPORT IS LOAD-BEARING. Importing `skills` is what runs the @skill decorators, i.e. what
# puts anything in the REGISTRY at all — and the REGISTRY is what SKILL_ID() (and therefore this
# file's whole dispatch) is derived from. Without it every skill lookup below raises KeyError,
# which is exactly what happened while the skills lived in simvla_data_generator.py, a module this
# process cannot import (pxr, omni.usd, tkinter, PIL). skills.py imports none of that.
import skills  # noqa: F401,E402  (imported for its registration side effects)
import nav_tuning  # noqa: E402  (per-robot nav constants)
from planner_settings import bounded_float_env, curobo_search_budget  # noqa: E402
from skill_contract import REGISTRY  # noqa: E402
from executor_dispatch import PADDING_SKILL_ID, load_script  # noqa: E402


class RawActionsRecorder(RecorderTerm):
    """Record the actual action vector for HDF5 replay alongside LeRobot features."""

    def record_pre_step(self):
        return "raw_actions", self._env.action_manager.action


@configclass
class RawActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = RawActionsRecorder


#: JOG BUDGET AND STEP SIZE -- SHARED BY BOTH ARMS.
#:
#: These live at module scope, not inside main(), because the two arms' reach
#: blocks sit under SEPARATE guards: the right arm's under
#: `if executing_indices.numel() > 0:` and the left's under
#: `if executing_indices_l.numel() > 0:`. Assigning them inside main() makes
#: them function-local for the WHOLE of main, so a left-arm-only task -- every
#: handle task in this repo uses A_l -- hits
#:   UnboundLocalError: local variable 'REACH_RETRIES' referenced before assignment
#: on the first left reach that misses its gate, because the right arm never ran
#: the assignment. Measured: that is exactly how run 2111617 died.
#:
#: 300 steps of 0.0025 m is 0.75 m of authority. The reach at the counter needs
#: 2-97 of them, but the retreat home is a 0.4-0.5 m motion and the jog is now
#: what performs it, so the budget has to cover that too. It only runs while the
#: error is outside the gate, so a generous ceiling costs nothing on the short
#: corrections. 0.0025 m at the 20 Hz env rate is 0.05 m/s, the pace the plan
#: tracker already drives these arms at.
REACH_RETRIES = int(os.environ.get("SIMVLA_REACH_RETRIES", "300"))
if not 1 <= REACH_RETRIES <= 10_000:
    raise ValueError("SIMVLA_REACH_RETRIES must be between 1 and 10000")
REACH_STEP_M = 0.0025
REACH_STEP_RAD = 0.01
#: ``arm.reset`` is a contract, not an ordinary carry waypoint.  A closed gripper used to route
#: the left reset through the deliberately disabled 1 m / 180 degree carry gate, so navigation
#: could start while the hand was still visibly extended.  These bounds keep the Cartesian jog
#: active until the reset actually reaches its saved joint home's FK pose.
RESET_POS_TOL_M = 0.02
RESET_ROT_TOL_DEG = bounded_float_env("SIMVLA_RESET_ROT_TOL_DEG", 10.0, 0.0, 180.0)
ALLOW_CARTESIAN_FALLBACK = cartesian_fallback_allowed()

def resolve_skill(skill_id: str, ctx, envs, eef_idx=None, **params):
    """Run a runtime-resolved skill's resolve() for a batch of envs. One table, one call shape.

    This replaces `RUNTIME_RESOLVERS[SkillSentinel.PAUSE](...)` — a lookup keyed by the magic float
    the step's payload happened to contain. The key is now the skill's name, which is written down
    rather than guessed at.
    """
    return REGISTRY[skill_id].cls().resolve(ctx, envs, params, eef_idx)
# Import for cuRobo
from isaacsim.core.utils.types import ArticulationAction
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.util.usd_helper import UsdHelper
import curobo.util.usd_helper as _curobo_usd_helper
from collision_mesh import install_polygon_mesh_reader
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.geom.types import WorldConfig, Mesh, Sphere
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from isaacsim.core.api.objects import cuboid
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.utils.datasets import HDF5DatasetFileHandler, TwoPhaseEpisodeWriter
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, quat_mul

import imageio.v2 as imageio
# TODO Do I need RateLimiter?
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

def world2base(env, ee_pos_w, ee_quat_w, env_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a single world-frame pose to a base-frame pose for a specific environment.
    """
    robot_data = env.scene.articulations["robot"].data
    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    
    root_pos_w = robot_data.body_pos_w[env_idx, base_link_idx]
    root_quat_w = robot_data.body_quat_w[env_idx, base_link_idx]
    root_pos_w = root_pos_w.unsqueeze(0)
    root_quat_w = root_quat_w.unsqueeze(0)
    ee_pos_w = ee_pos_w.unsqueeze(0)
    ee_quat_w = ee_quat_w.unsqueeze(0)

    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    return ee_pose_b.squeeze(0), ee_quat_b.squeeze(0)

def base2world(env, ee_pos_b, ee_quat_b, env_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    robot_data = env.scene.articulations["robot"].data
    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]

    root_pos_w = robot_data.body_pos_w[env_idx, base_link_idx].unsqueeze(0)
    root_quat_w = robot_data.body_quat_w[env_idx, base_link_idx].unsqueeze(0)

    ee_pos_b = ee_pos_b.unsqueeze(0)
    ee_quat_b = ee_quat_b.unsqueeze(0)

    ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_b, ee_quat_b
    )
    return ee_pos_w.squeeze(0), ee_quat_w.squeeze(0)

def compute_eef_deltas_batched(motion_gen, plans, cmd_indices, joint_names, eef_link_name):
    """
    Computes end-effector delta poses for a batch of arms executing plans.

    If cmd index exceeds the last valid index of a plan, it holds the last state
    so the resulting delta becomes zero for that arm.
    """
    batch_size = len(plans)

    q_t_list, q_tp1_list = [], []

    # THE FK VECTOR MUST BE cuRobo'S ACTIVE JOINTS, IN cuRobo'S ORDER.
    #
    # This selected them by dropping any name containing "gripper", which is a guess about which
    # joints cuRobo locked. For Anubis the guess is exactly right -- its lock_joints are
    # gripper1_joint and gripper1R_joint, so the filter leaves 6 names against 6 active DOF -- and
    # that is why it has never been wrong before. AI Worker locks `lift_joint`, which contains no
    # "gripper", so the filter left EIGHT names against SEVEN active DOF and handed
    # kinematics.forward a vector one element too long: lift_joint's 0.0 read as arm_r_joint1,
    # arm_r_joint1 as arm_r_joint2, and so on down the arm.
    #
    # The plan itself was fine -- result.success was True and cuRobo reached the goal in its own
    # model. Only these DELTAS were computed from the shifted vector, and the arm is driven by
    # them, so it went somewhere unrelated: measured 0.69-0.91 m from the goal with 45-92 degrees
    # of rotation error, every env, every episode.
    #
    # motion_gen.kinematics.joint_names IS the authority: it is the active set, already in the
    # order forward() expects, so no filtering heuristic is needed at all.
    arm_joint_names = list(motion_gen.kinematics.joint_names)
    _guessed = [name for name in joint_names if "gripper" not in name]
    if _guessed != arm_joint_names and not getattr(compute_eef_deltas_batched, "_warned", False):
        compute_eef_deltas_batched._warned = True
        print(f"[fk] EEF-delta joint vector: using cuRobo's active joints {arm_joint_names} "
              f"instead of the name-filtered guess {_guessed}", flush=True)

    for i in range(batch_size):
        plan = plans[i]
        raw_idx = int(cmd_indices[i].item())

        plan_len = len(plan)
        if plan_len <= 0:
            raise ValueError("Encountered empty plan in compute_eef_deltas_batched.")

        # Clamp current index to valid range
        idx = min(raw_idx, plan_len - 1)

        ordered_plan = plan.get_ordered_joint_state(arm_joint_names)

        # current state
        q_t_state = ordered_plan[idx]
        q_t_list.append(q_t_state.position)

        # next state
        if idx + 1 < plan_len:
            q_tp1_state = ordered_plan[idx + 1]
            q_tp1_list.append(q_tp1_state.position)
        else:
            # already at final step -> hold last state
            q_tp1_list.append(q_t_state.position)

    q_t = torch.stack(q_t_list)
    q_tp1 = torch.stack(q_tp1_list)

    kinematics = motion_gen.kinematics

    pose_t_tuple = kinematics.forward(q_t, link_name=eef_link_name)
    position_t = pose_t_tuple[0].clone()
    quaternion_t = pose_t_tuple[1].clone()

    pose_tp1_tuple = kinematics.forward(q_tp1, link_name=eef_link_name)
    position_tp1 = pose_tp1_tuple[0]
    quaternion_tp1 = pose_tp1_tuple[1]

    delta_pos = position_tp1 - position_t

    r_t = R.from_quat(quaternion_t.detach().cpu().numpy()[:, [1, 2, 3, 0]])
    r_tp1 = R.from_quat(quaternion_tp1.detach().cpu().numpy()[:, [1, 2, 3, 0]])
    delta_r = r_tp1 * r_t.inv()

    delta_rotvec = torch.tensor(delta_r.as_rotvec(), dtype=torch.float32, device=delta_pos.device)

    return torch.cat((delta_pos, delta_rotvec), dim=-1)


def wrap_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def closest_cardinal_yaw(yaw):
    """
    yaw: (...,) radian tensor in [-pi, pi]
    return:
        closest_angle_rad: (...,)
        closest_angle_deg: (...,)
        index: (...,)  # 0:0°, 1:90°, 2:-90°, 3:180°
    """
    device = yaw.device
    targets = torch.tensor(
        [0.0,  torch.pi/2, -torch.pi/2,  torch.pi],
        device=device
    )  # (4,)

    diff = wrap_to_pi(yaw.unsqueeze(-1) - targets)
    dist = torch.abs(diff)   # (..., 4)
    index = torch.argmin(dist, dim=-1)   # (...)
    closest_angle_rad = targets[index]
    closest_angle_deg = closest_angle_rad * 180 / torch.pi

    return closest_angle_rad, closest_angle_deg, index

def pre_process_actions(
    env,
    delta_pose_L: torch.Tensor,
    gripper_command_L: bool,
    delta_pose_R: torch.Tensor,
    gripper_command_R: bool,
    delta_pose_base: torch.Tensor,
    robot
) -> torch.Tensor:
    """Pre-process actions for the environment."""
    batch_size = delta_pose_L.shape[0]

    gripper_command_L = torch.as_tensor(gripper_command_L, device=delta_pose_L.device,
                                       dtype=torch.bool).reshape(-1, 1)
    gripper_command_R = torch.as_tensor(gripper_command_R, device=delta_pose_R.device,
                                       dtype=torch.bool).reshape(-1, 1)

    if gripper_command_L.shape[0] == 1:
        gripper_command_L = gripper_command_L.expand(batch_size, 1)
    if gripper_command_R.shape[0] == 1:
        gripper_command_R = gripper_command_R.expand(batch_size, 1)

    gripper_vel_L = torch.where(gripper_command_L, -1.0, 1.0)
    gripper_vel_R = torch.where(gripper_command_R, -1.0, 1.0)

    # Delta base for local robot frame
    # local velocities in base frame
    vx_local =  delta_pose_base[:, 0]
    vy_local = -1 * delta_pose_base[:, 1]
    omega    = delta_pose_base[:, 2]   # scalar yaw rate (same in base/world for planar z)

    robot = env.scene.articulations["robot"]

    base_link_idx = robot.find_bodies("base_link")[0][0]
    r, p, yaw = euler_xyz_from_quat(robot.data.body_quat_w[:, base_link_idx])

    # rotate [vx, vy] from base frame -> world frame using yaw
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    vx_world = cos_yaw * vx_local - sin_yaw * vy_local
    vy_world = sin_yaw * vx_local + cos_yaw * vy_local

    # pack back into (num_envs, 3)
    delta_pose_base_world = torch.stack([ vx_world, vy_world, omega], dim=1)
    action = torch.cat(
        [
            delta_pose_L,
            delta_pose_R,
            gripper_vel_L,
            gripper_vel_R,
            delta_pose_base_world,   # <-- use world-frame base motion
        ],
        dim=1,
    )

    return torch.cat([action], dim=1)


from pxr import Gf, UsdGeom

def pose_to_gf_matrix_tensor(position: torch.Tensor, quaternion: torch.Tensor) -> Gf.Matrix4d:
    """
    position: torch.Tensor of shape (3,)
    quaternion: torch.Tensor of shape (4,) in w, x, y, z order
    """
    pos = position.detach().cpu().numpy().astype(np.float64).reshape(3)
    q_wxyz = quaternion.detach().cpu().numpy().astype(np.float64).reshape(4)

    # (w, x, y, z) -> (x, y, z, w) for SciPy
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

    # normalize (recommended)
    n = np.linalg.norm(q_xyzw)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    q_xyzw /= n

    # rotation matrix via SciPy
    rot = R.from_quat(q_xyzw).as_matrix().astype(np.float64)  # (3,3)

    # build 4x4
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3,  3] = pos

    # numpy -> Gf.Matrix4d (row-major)
    return Gf.Matrix4d(
        tuple(map(float, mat[0])),
        tuple(map(float, mat[1])),
        tuple(map(float, mat[2])),
        tuple(map(float, mat[3])),
    )

import numpy as np, math
from copy import deepcopy

def quat_normalize(q, eps=1e-12):
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

def quat_mul_wxyz_np(q1, q2):
    # q1,q2: (...,4) in (w,x,y,z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def axis_angle_to_quat_wxyz(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle_rad
    s = math.sin(half)
    return np.array([math.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

def _noisy_pose_once(base_pos, base_q, rng, pos_sigma, rot_max_deg, yaw_only):
    # --- pos_sigma is a *radius* (meters): max ||dpos|| <= pos_sigma ---
    if pos_sigma <= 0:
        dpos = np.zeros(3, dtype=np.float64)
    else:
        # random direction
        v = rng.normal(0.0, 1.0, size=(3,))
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n = 1.0
        v = v / n
        # uniform inside 3D ball: r ~ U(0,1)^(1/3)
        r = float(pos_sigma) * (rng.random() ** (1.0 / 3.0))
        dpos = v * r

    new_pos = base_pos + dpos

    ang = rng.uniform(-math.radians(rot_max_deg), math.radians(rot_max_deg))
    if yaw_only:
        dq = axis_angle_to_quat_wxyz([0, 1, 0], ang)
    else:
        axis = rng.normal(size=(3,))
        axis /= (np.linalg.norm(axis) + 1e-12)
        dq = axis_angle_to_quat_wxyz(axis, ang)

    new_q = quat_mul_wxyz_np(dq, base_q)  # delta ⊗ base
    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)
    return new_pos, new_q

_frame_state = {"n": 0, "written": 0, "dir": None}


def _dump_frame(env):
    """Write env 0's `front` camera to a PNG, every --dump_every sim steps.

    The scene carries front/wrist cameras but nothing writes them anywhere, so a run that
    misbehaves can only be read through printed counters — which say a grasp failed, never why.
    Opt-in and side-effect free: without --dump_frames this function is never called.
    """
    n = _frame_state["n"]
    _frame_state["n"] = n + 1
    if n % max(1, args_cli.dump_every):
        return

    if _frame_state["dir"] is None:
        _frame_state["dir"] = args_cli.dump_frames
        os.makedirs(_frame_state["dir"], exist_ok=True)

    # Say why, loudly, once. A silent `except: return` here would make a broken dump look
    # identical to a scene with no camera — the same failure this codebase is being cured of.
    cam = args_cli.dump_camera
    try:
        rgb = env.scene[cam].data.output["rgb"]          # (num_envs, H, W, 3), uint8
    except Exception as exc:
        if not _frame_state.get("warned"):
            _frame_state["warned"] = True
            print(f"[dump_frames] cannot read the {cam!r} camera: {type(exc).__name__}: {exc}. "
                  f"A kitchen emitted before scene_cam existed will not have it -- re-emit, or "
                  f"pass --dump_camera front.", flush=True)
        return
    if rgb is None or rgb.shape[0] == 0:
        if not _frame_state.get("warned"):
            _frame_state["warned"] = True
            print(f"[dump_frames] {cam!r} camera produced no frames (rgb={rgb!r})", flush=True)
        return

    # WHICH ENV. This was hard-coded to 0, and the only successful grasp of the campaign happened
    # in env 9 -- so it was never filmed. SIMVLA_DUMP_ENV selects; 0 keeps the old behaviour.
    #
    # SIMVLA_DUMP_ALL_ENVS FILMS EVERY ENV INSTEAD, one subdirectory each. You cannot know which
    # env will grasp until it has, and by then the frames are gone -- pinning the camera to one
    # env means filming a coin flip and losing it most times. Two real grasps of this campaign
    # (lifted_by +0.1915 and +0.1898) went unfilmed for exactly this reason. The caller encodes
    # whichever env the log reports as the winner and deletes the rest, so the cost is disk held
    # for the length of one run, not a rerun hoping the same env repeats.
    if os.environ.get("SIMVLA_DUMP_ALL_ENVS"):
        idx = _frame_state["written"]
        _frame_state["written"] = idx + 1
        arr = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
        for _e in range(arr.shape[0]):
            _d = os.path.join(_frame_state["dir"], f"env{_e:02d}")
            os.makedirs(_d, exist_ok=True)
            imageio.imwrite(os.path.join(_d, f"frame_{idx:06d}.png"), arr[_e])
        return
    _de = int(os.environ.get("SIMVLA_DUMP_ENV", "0"))
    _de = max(0, min(_de, rgb.shape[0] - 1))
    frame = rgb[_de, ..., :3].detach().cpu().numpy().astype(np.uint8)
    idx = _frame_state["written"]
    _frame_state["written"] = idx + 1
    imageio.imwrite(os.path.join(_frame_state["dir"], f"frame_{idx:06d}.png"), frame)
    # Optional synchronized review view, from the same simulated timestep as the
    # requested policy camera. Single selected environment only.
    _secondary = os.environ.get("SIMVLA_REVIEW_SECONDARY_CAMERA", "")
    if _secondary and _secondary != cam:
        try:
            _rgb2 = env.scene[_secondary].data.output["rgb"]
            _dir2 = os.path.join(_frame_state["dir"], f"secondary-{_secondary}")
            os.makedirs(_dir2, exist_ok=True)
            imageio.imwrite(os.path.join(_dir2, f"frame_{idx:06d}.png"),
                            _rgb2[_de, ..., :3].detach().cpu().numpy().astype(np.uint8))
        except Exception as _secondary_exc:
            if not _frame_state.get("secondary_warned"):
                _frame_state["secondary_warned"] = True
                print(f"[dump_frames] secondary {_secondary!r} unavailable: "
                      f"{_secondary_exc}", flush=True)


def add_camera_noise_to_scene_cfg_once(env_cfg,
                                       cams=("front", "wrist_left", "wrist_right"),
                                       pos_sigma=0.05,
                                       rot_max_deg=2.0,
                                       yaw_only=False,
                                       seed=None):
    rng = np.random.default_rng(seed)
    cfg = deepcopy(env_cfg)

    if isinstance(cams, str):
        cams = (cams,)

    for name in cams:
        if not hasattr(cfg.scene, name):
            continue
        cam = getattr(cfg.scene, name)
        if not hasattr(cam, "offset"):
            continue

        base_pos = np.array(cam.offset.pos, dtype=np.float64)
        base_q   = np.array(cam.offset.rot, dtype=np.float64)  # wxyz

        new_pos, new_q = _noisy_pose_once(base_pos, base_q, rng, pos_sigma, rot_max_deg, yaw_only)
        cam.offset.pos = tuple(new_pos.tolist())
        cam.offset.rot = tuple(new_q.tolist())

    return cfg

def add_light_noise_to_scene_cfg_once(
    env_cfg,
    intensity_base=6000.0,
    intensity_stop_range=1.5,      # log2 range for intensity multiplier
    kelvin_choices=(2700, 3200, 4000, 5000, 6500, 8000, 9000),
    kelvin_uniform=None,           # e.g. (4000, 9000) if you prefer uniform
    rgb_jitter_sigma=0.03,
    seed=None,
):
    rng = np.random.default_rng(seed)
    cfg = deepcopy(env_cfg)

    if not hasattr(cfg.scene, "light") or not hasattr(cfg.scene.light, "spawn"):
        return cfg

    light = cfg.scene.light.spawn

    # ---- intensity: log-uniform via stops (recommended) ----
    # m = 2^U(-r, +r)
    m = 2.0 ** rng.uniform(-intensity_stop_range, intensity_stop_range)
    light.intensity = float(intensity_base * m)
    # ---- color temperature ----
    if kelvin_uniform is not None:
        k_lo, k_hi = kelvin_uniform
        light.enable_color_temperature = True
        light.color_temperature = float(rng.uniform(k_lo, k_hi))
    else:
        light.enable_color_temperature = True
        light.color_temperature = float(rng.choice(kelvin_choices))

    # ---- small RGB tint around white (optional but useful) ----
    # keep it subtle; Kelvin already changes white point
    jitter = rng.normal(0.0, rgb_jitter_sigma, size=(3,))
    col = np.clip(1.0 + jitter, 0.7, 1.3)
    light.color = (float(col[0]), float(col[1]), float(col[2]))

    return cfg


def _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, env_ids: torch.Tensor, reason: str = "reset"):
    """
    env_ids: 1D Long tensor of GLOBAL env indices (e.g., tensor([0,5,9], device=device))
    """
    if env_ids is None or env_ids.numel() == 0:
        return
    if env_ids.dtype != torch.long:
        env_ids = env_ids.to(dtype=torch.long)

    # 1) reset simulator envs
    env._reset_idx(env_ids)

    # 2) book-keeping / goal buffer
    goal_mgr.on_failure(env_ids, reason=reason)
    resample_goals_for_envs(env_ids, target_idx)

    # 3) reset per-env script pointers / plans
    env_goal_indices[env_ids] = 0
    env_cmd_indices_r[env_ids] = 0
    env_cmd_indices_l[env_ids] = 0

    # IMPORTANT: clear stored plans for these envs
    for i in env_ids.tolist():
        env_cmd_plans_r[i] = None
        env_cmd_plans_l[i] = None
        ik_goals_r[i] = None
        ik_goals_l[i] = None

    # 4) reset your flags/states
    grasp_checked_r[env_ids] = False
    grasp_checked_l[env_ids] = False
    nav_phase[env_ids] = 0

    env_cmd_pause_r[env_ids] = 0
    env_cmd_pause_trig_r[env_ids] = False
    env_cmd_pause_l[env_ids] = 0
    env_cmd_pause_trig_l[env_ids] = False

    drawer[env_ids] = 0
    pot_first[env_ids] = 0
    first_reset[env_ids] = 0
    env_sub_saved_r[env_ids] = False
    env_sub_saved_l[env_ids] = False
    new_gripper_commands_L[env_ids] = False
    new_gripper_commands_R[env_ids] = False
    timestep[env_ids] = 0

    # 5) reset IK controllers
    # (your code uses env_ids directly, keep consistent)
    env.action_manager.get_term("armL_action")._ik_controller.reset(env_ids)
    env.action_manager.get_term("armR_action")._ik_controller.reset(env_ids)

    print(f"[Reset] env={env_ids.tolist()} reason={reason}")

home_r = torch.tensor([ 0.2257, -0.0988,  1.0351], device='cuda:0')
home_l = torch.tensor([ 0.2203,  0.1080,  1.0342], device='cuda:0')
home_r_quat = torch.tensor([ 0.6010, -0.6257,  0.3885, -0.3106], device='cuda:0')
home_l_quat = torch.tensor([-0.3093,  0.3876, -0.6270,  0.6008], device='cuda:0')

#: THE TWO JAW BODIES OF ONE HAND, per robot. The G_r/G_l steps below decide the gripper has
#: finished closing by watching the distance between these two bodies stop changing, so a name
#: the articulation does not carry is a hard ValueError out of find_bodies, mid-episode.
#:
#: Resolved by ASKING THE ARTICULATION rather than by robot name, which is how
#: kitchen/mdp/composed.py does it -- there is no --robot value to consult at the call sites and
#: the name is not the authority anyway, the body list is. Anubis's spelling stays first, so
#: nothing about an Anubis run changes.
#:
#: THIS IS THE THIRD COPY of this table (simvla_video.jaw_bodies and composed._FINGER_CANDIDATES
#: are the others) and none of the three can import the others: composed.py lives in
#: isaaclab_tasks, and simvla_video/simvla_gen each boot Isaac at module scope. A robot added to
#: one and not the others fails in whichever runs first.
_JAW_CANDIDATES = {
    "right": (("gripper1R", "gripper1L"), ("ee_finger_r1", "ee_finger_r2"),
              ("gripper_r_rh_p12_rn_r2", "gripper_r_rh_p12_rn_l2")),
    "left": (("gripper2R", "gripper2L"), ("ee_finger_l1", "ee_finger_l2"),
             ("gripper_l_rh_p12_rn_r2", "gripper_l_rh_p12_rn_l2")),
}


def jaw_body_indices(robot, arm: str):
    """(index_of_jaw_a, index_of_jaw_b) for `arm` on whichever robot this articulation is."""
    names = set(robot.body_names)
    for pair in _JAW_CANDIDATES[arm]:
        if names.issuperset(pair):
            return robot.find_bodies(pair[0])[0][0], robot.find_bodies(pair[1])[0][0]
    raise ValueError(
        f"no known {arm}-hand jaw bodies on this robot; tried {_JAW_CANDIDATES[arm]}, "
        f"articulation has {sorted(names)}"
    )


def main():
    if not hasattr(UsdHelper, "get_obstacles_from_stage_simvla"):
        raise RuntimeError(
            "The installed cuRobo is missing SimVLA's USD obstacle-frame compatibility patch. "
            "From the repository root, apply third_party/curobo.simvla.patch to "
            "third_party/curobo and reinstall it as documented in env/README.md."
        )
    # Collection choices include Python, NumPy, Torch, and Isaac Lab RNGs. Seed all of them
    # before scene/config randomization so the same CLI command starts from the same sampled
    # initial conditions. GPU physics remains numerically approximate across hardware/driver
    # versions; this pins the experiment's random inputs, not bitwise simulation output.
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    print(f"[seed] {args_cli.seed}", flush=True)

    # HOW LONG AN EPISODE MAY RUN, read ONCE and stated in the log. It was the literal 1500 in
    # `timeout_mask = timestep > 1500` below, and a script that outgrew it did not say so: every env
    # just reset with `reason=timeout`, which reads exactly like a task that failed. Job 2108770
    # cost a round to that -- its ten-leg bimanual push-chair script reached step 5 of 10 at t=1500
    # with every completed leg perfect.
    #
    # THE DEFAULT IS UNCHANGED at 1500, so every task that already runs is byte-identical. It is
    # also TIGHT, which is the evidence for making it settable rather than for raising it: the
    # six-leg CHASSIS push-chair script finishes at 1434 of 1500 -- 66 steps, about 3 seconds, a 4%
    # margin -- and that is the shortest script still in use.
    #
    # FIRST, before parse_env_cfg and before the scene is built, so a mistyped value costs seconds
    # rather than the minutes Omniverse takes to come up. episode_budget refuses a non-integer or
    # non-positive value instead of falling back, for the reason spawn_select gives about a pinned
    # spawn band: a silent fallback looks identical in the log while producing a different run.
    import episode_budget
    _episode_steps_raw = os.environ.get(episode_budget.ENV_VAR, "")
    episode_steps = episode_budget.episode_steps(_episode_steps_raw)
    print(episode_budget.describe(episode_steps, _episode_steps_raw), flush=True)

    # Bound diagnostic/collection runs even when no episode succeeds. The Slurm collector
    # exports this value; without reading it, an unsuccessful run only stops at walltime and
    # never reaches recorder/video finalization.
    _max_frames_raw = os.environ.get("SIMVLA_MAXFRAMES", "").strip()
    try:
        max_frames = int(_max_frames_raw) if _max_frames_raw else None
    except ValueError as exc:
        raise ValueError("SIMVLA_MAXFRAMES must be a positive integer") from exc
    if max_frames is not None and max_frames <= 0:
        raise ValueError("SIMVLA_MAXFRAMES must be a positive integer")
    print(f"[run] max_frames={max_frames if max_frames is not None else 'unbounded'}", flush=True)

    # Rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz)
    # Save Dataset
    timestamp = args_cli.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args_cli.output_root or f"./datasets/{args_cli.robot}"
    output_dir = os.path.join(root, timestamp, args_cli.task)
    output_file_name = f"{args_cli.task}.hdf5"

    output_dir = os.path.join(root, timestamp, args_cli.task)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file_name)
#   output_dir = os.path.dirname(args_cli.dataset_file)
#   output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    from sink_collision import configure_sink_collisions
    configure_sink_collisions(env_cfg)
    configure_physics_substeps(env_cfg)
    env_cfg.seed = args_cli.seed
    simvla_paths.validate_robot_asset(args_cli.robot, env_cfg.scene.robot.spawn.usd_path)
    env_cfg.env_name = args_cli.task

    # TODO: What is this concatenate_terms?
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    # LeRobot consumes the absolute action features already provided by the base config. HDF5
    # replay needs the raw vector passed to env.step plus the randomized initial state.
    env_cfg.recorders.record_initial_state = InitialStateRecorderCfg()
    env_cfg.recorders.record_pre_step_raw_actions = RawActionsRecorderCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    # SUCCEEDED-ONLY IS THE RIGHT DEFAULT -- failed episodes are not demonstrations, and training
    # on them teaches the policy to fail. But it makes the dataset hostage to the LAST step of the
    # chain: a run can grasp the bottle, lift it, drive it to the table and still write a 96-byte
    # empty HDF5 because the place refused to plan. Every file this campaign produced was that.
    #
    # SIMVLA_EXPORT_MODE=both writes successes and failures to SEPARATE files, so the camera data
    # exists and is labelled for what it is. Use it to inspect or to build a dataset when
    # successes are still rare; never mistake the failure file for demonstrations.
    _em = os.environ.get("SIMVLA_EXPORT_MODE", "succeeded").strip().lower()
    if _em == "both":
        env_cfg.recorders.dataset_export_mode = (
            DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES)
        print("[record] SIMVLA_EXPORT_MODE=both -- successes AND failures are being written, to "
              "separate files. The failure file is NOT demonstration data.", flush=True)
    elif _em not in ("succeeded", ""):
        raise SystemExit(f"SIMVLA_EXPORT_MODE={_em!r} is not one of: succeeded, both")
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    from pathlib import Path

    writer = TwoPhaseEpisodeWriter(staging_dir=output_dir, robot=args_cli.robot,
                                   record_initial_step=True)

    # For reset
    device = args_cli.device
    num_envs = args_cli.num_envs
    env_cmd_pause_r = torch.zeros(num_envs, dtype=torch.int32, device=device)   
    env_cmd_pause_trig_r = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_cmd_pause_l = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_cmd_pause_trig_l = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_cmd_track_stall_l = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_cmd_track_stall_r = torch.zeros(num_envs, dtype=torch.int32, device=device)
    _direct_base_anchors = {}

    # Sparse goal & initial pos and rot
    all_goals = []
    file_path, reloadable_file_path = simvla_paths.goal_files(args_cli.task)

    # The steps: decoded through goal_format (v2 native; v1 through the quarantined legacy reader)
    # and given a skill id each. `script[i].skill_id` is what every dispatch below compares against
    # — no branch reads a payload float to decide what a step means any more.
    #
    # The <task>.reloadable.json twin is NOT read. It only ever carried the per-step sentence, which
    # a v2 step carries itself (`ScriptStep.language`), and the code that read it built `sub_task_l`
    # and then never used it. v2 does not write a twin at all.
    script, goal_meta, skill_ids = load_script(file_path)
    from sink_collision import validate_sink_goal_environment
    validate_sink_goal_environment(goal_meta)
    step_skill_ids = [st.skill_id for st in script]
    print(f"[Goal] {file_path.name}: {len(script)} steps -> "
          f"{[st.skill for st in script]}")

    # A v2 goal file carries its own task config: obj_name, the grasp-check indices and the count
    # each one is allowed to ask for, target_idx, export_groups, task_language, task_type. They are
    # facts about the TASK, the file implies every one of them, and until now a human transcribed
    # them into flags by counting steps in the goal script. A flag the user actually TYPED still
    # wins — read from CLI_ARGV, because a parsed default is indistinguishable from a typed value.
    #
    # A v1 file has no run_config block, `_from_file` is empty, and nothing here changes: the flags
    # supply the values exactly as they do today. That is every goal file that has ever been run.
    #
    # Every field is coerced with the flag's own `type=`, from run_config_merge.FIELD_TYPES — the
    # file states the flag's TEXT ("4,5,6,7;8,9,10;11") and the rest of this program reads the list of
    # lists argparse makes of it. That map is the MERGE's, not this call site's: it was passed in from
    # here, and a call site is not where a fact about a field belongs — the next caller would not know
    # to pass it, and nothing would say so until finalize_lerobot, with the dataset already recorded.
    _from_file = merge_run_config(args_cli, goal_meta, CLI_ARGV)
    validate_grasp_check_steps(args_cli, script)
    if _from_file:
        print(f"[run config] taken from the goal file: {', '.join(_from_file)}", flush=True)

    _grasp_override = os.environ.get("SIMVLA_GRASP_GEOMETRY")
    _grasp_geometry_r, _grasp_geometry_l = arm_grasp_geometries(
        args_cli.robot, args_cli.obj_name, args_cli.obj_name_l, _grasp_override)
    _cylindrical_mug_grasp_r = _grasp_geometry_r == "cylindrical_mug"
    _cylindrical_mug_grasp_l = _grasp_geometry_l == "cylindrical_mug"
    _aiworker_grasp_route = os.environ.get("SIMVLA_AIWORKER_GRASP_ROUTE", "direct").strip().lower()
    if _aiworker_grasp_route not in {"direct", "staged"}:
        raise ValueError("SIMVLA_AIWORKER_GRASP_ROUTE must be direct or staged")
    _direct_replan_limit_l = int(os.environ.get("SIMVLA_DIRECT_REPLAN_LIMIT", "2"))
    if not 0 <= _direct_replan_limit_l <= 5:
        raise ValueError("SIMVLA_DIRECT_REPLAN_LIMIT must be between 0 and 5")
    _direct_track_timeout_l = int(os.environ.get("SIMVLA_DIRECT_TRACK_TIMEOUT", "50"))
    if not 1 <= _direct_track_timeout_l <= 500:
        raise ValueError("SIMVLA_DIRECT_TRACK_TIMEOUT must be between 1 and 500")
    _direct_track_pos_tol_l = bounded_float_env("SIMVLA_DIRECT_TRACK_POS_TOL", 0.03, 0.001, 0.03)
    _direct_track_rot_tol_l = bounded_float_env("SIMVLA_DIRECT_TRACK_ROT_TOL_DEG", 12.0, 0.5, 12.0)
    _segmented_grasp_standoff_l = bounded_float_env(
        "SIMVLA_AIWORKER_SEGMENTED_GRASP_STANDOFF_M", 0.0, 0.0, 0.20)
    if _segmented_grasp_standoff_l and args_cli.robot != "aiworker":
        raise ValueError("Segmented BoDex grasp is supported only for AI Worker")
    _loaded_home_track_pos_tol = bounded_float_env("SIMVLA_LOADED_HOME_TRACK_POS_TOL", .015, .001, .02)
    _loaded_home_track_rot_tol = bounded_float_env("SIMVLA_LOADED_HOME_TRACK_ROT_TOL_DEG", 6., .5, 10.)
    _track_loaded_home_enabled = track_loaded_home_reset(args_cli.robot, "arm.reset", True)
    _loaded_home_joint_targets = track_loaded_home_reset(
        args_cli.robot, "arm.reset", True,
        value=os.environ.get("SIMVLA_LOADED_HOME_JOINT_TARGETS", "0"))
    _payload_collision_l = os.environ.get("SIMVLA_AIWORKER_PAYLOAD_COLLISION", "0")
    if _payload_collision_l not in ("0", "1"):
        raise ValueError("SIMVLA_AIWORKER_PAYLOAD_COLLISION must be 0 or 1")
    _payload_collision_l = _payload_collision_l == "1"
    if _payload_collision_l and args_cli.robot != "aiworker":
        raise ValueError("Payload collision experiment currently supports only AI Worker")
    if _loaded_home_joint_targets and not _track_loaded_home_enabled:
        raise ValueError("Loaded-home joint targets require SIMVLA_TRACK_LOADED_HOME=1")
    _right_track_pos_tol = bounded_float_env("SIMVLA_RIGHT_TRACK_POS_TOL", 0.01, 0.001, 0.03)
    _right_track_rot_tol = bounded_float_env("SIMVLA_RIGHT_TRACK_ROT_TOL_DEG", 5.0, 0.5, 12.0)
    # The planner and final approach must use the same explicit standoff.
    # A hard-coded 18 cm in the jog used to retreat from a planned 8 cm
    # standoff, and even inserted a retreat when direct approach was requested.
    _right_pregrasp_standoff = bounded_float_env("SIMVLA_PREGRASP_STANDOFF", 0.0, 0.0, 0.3)
    _right_pregrasp_above = bounded_float_env("SIMVLA_PREGRASP_ABOVE", 0.0, 0.0, 0.3)
    _postrelease_retreat_m = bounded_float_env("SIMVLA_POSTRELEASE_RETREAT_M", 0.0, 0.0, 0.3)
    _final_home_raw = os.environ.get("SIMVLA_POSTRELEASE_FINAL_HOME", "0")
    if _final_home_raw not in ("0", "1"):
        raise ValueError("SIMVLA_POSTRELEASE_FINAL_HOME must be 0 or 1")
    _postrelease_final_home = _final_home_raw == "1"
    if _postrelease_final_home and args_cli.robot != "aiworker":
        raise ValueError("SIMVLA_POSTRELEASE_FINAL_HOME is currently validated only for AI Worker")
    _final_home_joint_raw = os.environ.get("SIMVLA_AIWORKER_FINAL_HOME_JOINT_TARGETS", "0")
    if _final_home_joint_raw not in ("0", "1"):
        raise ValueError("SIMVLA_AIWORKER_FINAL_HOME_JOINT_TARGETS must be 0 or 1")
    _final_home_joint_targets = _final_home_joint_raw == "1"
    if _final_home_joint_targets and not (_postrelease_final_home and args_cli.robot == "aiworker"):
        raise ValueError("Final-home joint targets require AI Worker postrelease final home")
    if _final_home_joint_targets:
        _track_loaded_home_enabled = True
        _loaded_home_joint_targets = True
    _postrelease_retreat_lift_m = bounded_float_env(
        "SIMVLA_POSTRELEASE_RETREAT_LIFT_M", 0.05, -0.05, 0.15)
    _direct_bodex_grasp_l = (args_cli.robot == "aiworker"
                             and _cylindrical_mug_grasp_l
                             and _aiworker_grasp_route == "direct")
    print(f"[grasp] geometry_r={_grasp_geometry_r} object_r={args_cli.obj_name}; "
          f"geometry_l={_grasp_geometry_l} object_l={args_cli.obj_name_l} "
          f"robot={args_cli.robot} aiworker_route={_aiworker_grasp_route}", flush=True)

    # After the merge, never before it: these two used to be read off args_cli twenty lines before
    # the goal file was even opened, and would now print the flags rather than the run. (Nothing
    # else reads them — the checks themselves read args_cli.sub_grasp_idx_r/_l directly, below.)
    SUB_GRASP_IDX_R = args_cli.sub_grasp_idx_r
    SUB_GRASP_IDX_L = args_cli.sub_grasp_idx_l
    print(f"[SubGrasp] Right check idx={SUB_GRASP_IDX_R} | Left check idx={SUB_GRASP_IDX_L}")

    # `goal_meta` is the goal file minus its steps — kitchen_type, initial_pos_ranges,
    # initial_rot_yaw_range, island_bound — which is every field the code below reads off `data`.
    # The block used to be `with open(file_path) as file: data = json.load(file)` and its body runs
    # to ~line 1530; dedenting 770 lines is a separate change, so the context manager stays and only
    # what it binds has changed. Nothing can reach the raw [action, payload] pairs any more.
    with contextlib.nullcontext():
        data = goal_meta
        # 1. Init pos
        # ONE band is drawn for the WHOLE run and every env then spawns inside it, so on a kitchen
        # with more than one band the draw makes the run bimodal rather than the spawn varied.
        # More than one band is exactly what FURNITURE produces: the free-space search in
        # simvla_data_generator._generate_env_config treats every /world/<name> prim as an
        # obstacle, so a table splits the floor in two. Kitchen 1216 (a table with two chairs
        # pulled out) emits one band beside the table and one 1.3 m away past it; a run that draws
        # the wrong one starts every robot on the far side of the furniture it has to push.
        # SIMVLA_INIT_POS_IDX pins the choice. Unset, this is the SAME random.choice over the same
        # list that was here before -- same call, same rng, same distribution -- which is why
        # spawn_select takes `rng=random` rather than owning one.
        import spawn_select
        _init_idx, init_pos = spawn_select.choose_init_pos(data["initial_pos_ranges"], rng=random)
        print(spawn_select.describe(_init_idx, init_pos), flush=True)
        # 2. Init rot
        init_rot = data["initial_rot_yaw_range"]
        if args_cli.robot_fix_init:
            print("Robot is small randomization.")
            env_cfg.events.robot_init_pos.params["pose_range"]["x"] = ((init_pos[0][1] + init_pos[0][2])/2, (init_pos[0][1] + init_pos[0][2])/2 + 0.03)
            env_cfg.events.robot_init_pos.params["pose_range"]["y"] = ((init_pos[1][1] + init_pos[1][2])/2, (init_pos[1][1] + init_pos[1][2])/2 + 0.03)
#           env_cfg.events.robot_init_pos.params["pose_range"]["yaw"] = ((init_rot[0][1] + init_rot[0][2])/2, (init_rot[0][1] + init_rot[0][2])/2 + 0.1)

        else:
            print("Robot initial state is randomized.")
            env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (init_pos[0][1], init_pos[0][2])
            env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (init_pos[1][1], init_pos[1][2])
#           env_cfg.events.robot_init_pos.params["pose_range"]["yaw"] = (init_rot[0][1]*57.2958, init_rot[0][2]*57.2958)
        has_ramen = hasattr(env_cfg.scene, 'ramen0') and env_cfg.scene.ramen0 is not None
        has_sweet_potato = hasattr(env_cfg.scene, 'sweet_potato0') and env_cfg.scene.sweet_potato0 is not None
        
        if args_cli.obj_name == "pot0" or has_ramen or args_cli.pour_pot or has_sweet_potato:
            obj_init = {
                "Isaac-Kitchen-v549-00":{"x_range":(1.3,1.34), "y_range":(-1.7,-1.5), "z_range":(0.868,0.869),"direction":"N", "range_direction":"W"},
                "Isaac-Kitchen-v388-01":{"x_range":(1.3,1.34), "y_range":(-1.7,-1.5), "z_range":(0.868,0.869),"direction":"N", "range_direction":"N"},
                "Isaac-Kitchen-v384-08":{"x_range":(1.3,1.34), "y_range":(-1.7,-1.5), "z_range":(0.868,0.869),"direction":"N", "range_direction":"N"},
                "Isaac-Kitchen-v485-07":{"x_range":(1.3,1.34), "y_range":(-1.7,-1.5), "z_range":(0.868,0.869),"direction":"N", "range_direction":"N"},
            }
        if args_cli.obj_fix_init:
            print("Object is fixed.")
        else:
            if args_cli.obj_name == "pot0":
                env_cfg.events.obj_init_pos.params["pose_range"]["x"] = obj_init[args_cli.task]["x_range"]
                env_cfg.events.obj_init_pos.params["pose_range"]["y"] = obj_init[args_cli.task]["y_range"]
                env_cfg.events.obj_init_pos.params["pose_range"]["z"] = obj_init[args_cli.task]["z_range"]


            else:
                print("Object position is randomized.")
                env_cfg.events.obj_init_pos.params["pose_range"]["x"] = (-0.05,0.05)
                env_cfg.events.obj_init_pos.params["pose_range"]["y"] = (-0.05,0.05)
            

        kitchen_type = data["kitchen_type"]
        if kitchen_type == "island":
            island_min = [data["island_bound"][0],data["island_bound"][2]]
            island_max = [data["island_bound"][1],data["island_bound"][3]]

        # 3. Goal
        # The candidates only ever vary a step's POSE (arm.grasp offers interchangeable grasps).
        # The step's action and skill are fixed by the goal file, so `step_skill_ids[i]` is the
        # skill of step i in every candidate — which is what lets the tensor builders below index
        # it by position.
        all_goals_cartesian = [[]]

        for st in script:
            name, spec = st.action, st.spec
            # Each step may have 1 option or multiple options
            if isinstance(spec, bool):
                options = [spec]
            elif isinstance(spec, list) and any(isinstance(x, list) for x in spec):
                # list-of-lists => multiple pose options
                options = spec
            else:
                # single pose list (e.g., [x,y,z,qx,qy,qz,qw]) or other scalar-like
                options = [spec]

            if (_direct_bodex_grasp_l and st.skill == "arm.grasp" and name == "A_l"):
                allowed = side_approach_indices(options)
                if not allowed:
                    raise ValueError(
                        "AI Worker direct mug grasp needs a side-approach BoDex pose "
                        "(tool +Z horizontal component >= 0.65); use an authored side "
                        "candidate or SIMVLA_AIWORKER_GRASP_ROUTE=staged"
                    )
                print(f"[Goal] AI Worker side-approach BoDex candidates: "
                      f"{allowed}/{len(options)}", flush=True)
                options = [options[index] for index in allowed]

            # Expand cartesian product
            all_goals_cartesian = [
                combo + [(name, opt)]
                for combo in all_goals_cartesian
                for opt in options
            ]

        print(f"[Goal] all_goals_cartesian size = {len(all_goals_cartesian)}")

        def _goal_key(raw_goal) -> str:
            return json.dumps(raw_goal, sort_keys=True, separators=(",", ":"))

        def _pretty_goal(raw_goal) -> str:
            try:
                return " -> ".join([step[0] for step in raw_goal])
            except Exception:
                return str(raw_goal)

        class ThreeStageGoalBufferManager:
            """
            Pose-Bank based 3-stage manager (Design A)

            Stage R: collect RIGHT A_r poses into right_pose_bank until target_right
            Stage L: collect LEFT  A_l poses into left_pose_bank  until target_left
            Stage F: sample base scripts from all candidates, then REPLACE A_r/A_l poses from banks,
                     and collect full-good scripts until target_full
            Stage Final: sample only from full-good scripts
            """
            def __init__(
                self,
                all_candidates,
                target_right: int,
                target_left: int,
                target_full: int,
                right_in: str | None, right_out: str | None,
                left_in: str | None, left_out: str | None,
                full_in: str | None, full_out: str | None,
                replace_first_only: bool = True,
            ):
                self.target_right = int(target_right)
                self.target_left  = int(target_left)
                self.target_full  = int(target_full)
                self.replace_first_only = bool(replace_first_only)

                self.right_out = right_out
                self.left_out  = left_out
                self.full_out  = full_out

                self.candidates = copy.deepcopy(list(all_candidates))

                # Pose banks
                self.right_pose_bank = {"A_r": []}
                self.left_pose_bank  = {"A_l": []}

                # Full good scripts (after replacement)
                self.full_good_goals = []
                self.full_keys = set()

                # Current per-env sampled base goal (raw, list[(name,opt)])
                self.env_current = {}  # env -> base raw goal

                # Current per-env "save goal" (list[(name, pose_list)]) (after replacement + stage noise)
                self.env_current_noised = {}  # env -> jsonable executed/save goal

                # load from disk if provided
                if right_in:
                    self._load_pose_bank(right_in, side="r")
                if left_in:
                    self._load_pose_bank(left_in, side="l")
                if full_in:
                    self._load_full(full_in)

                self._update_stage()
                print(self._status_str())

            def _goal_key(self, raw_goal) -> str:
                return json.dumps(raw_goal, sort_keys=True, separators=(",", ":"))

            def _pretty_goal(self, raw_goal) -> str:
                try:
                    return " -> ".join([step[0] for step in raw_goal])
                except Exception:
                    return str(raw_goal)

            def _save(self, out_path: str | None, payload: dict):
                if not out_path:
                    return
                tmp = out_path + ".tmp"
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(tmp, "w") as f:
                        json.dump(payload, f, indent=2)
                    os.replace(tmp, out_path)
                except Exception as e:
                    print(f"[Goal3Stage] save failed {out_path}: {e}")

            def _save_right(self):
                self._save(self.right_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_right": self.target_right,
                    "right_pose_bank": self.right_pose_bank,
                })

            def _save_left(self):
                self._save(self.left_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_left": self.target_left,
                    "left_pose_bank": self.left_pose_bank,
                })

            def _save_full(self):
                self._save(self.full_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_full": self.target_full,
                    "full_good_goals": self.full_good_goals,
                })

            def _status_str(self) -> str:
                return (f"[Goal3Stage] stage={self.stage} | "
                        f"A(cand)={len(self.candidates)} | "
                        f"Rpose={len(self.right_pose_bank['A_r'])}/{self.target_right} | "
                        f"Lpose={len(self.left_pose_bank['A_l'])}/{self.target_left} | "
                        f"F={len(self.full_good_goals)}/{self.target_full}")

            def _update_stage(self):
                if self.target_full > 0 and len(self.full_good_goals) >= self.target_full:
                    self.stage = "final_only"; return
                if self.target_right > 0 and len(self.right_pose_bank["A_r"]) < self.target_right:
                    self.stage = "collect_right"; return
                if self.target_left > 0 and len(self.left_pose_bank["A_l"]) < self.target_left:
                    self.stage = "collect_left"; return
                self.stage = "collect_full"

            # ---------- loading helpers ----------
            def _load_pose_bank(self, path: str, side: str):
                try:
                    with open(path, "r") as f:
                        payload = json.load(f)
                    if not isinstance(payload, dict):
                        return
                    if side == "r":
                        bank = payload.get("right_pose_bank") or {}
                        poses = bank.get("A_r") or []
                        # ensure list-of-list length 7
                        for p in poses:
                            if isinstance(p, list) and len(p) == 7:
                                self.right_pose_bank["A_r"].append(p)
                    else:
                        bank = payload.get("left_pose_bank") or {}
                        poses = bank.get("A_l") or []
                        for p in poses:
                            if isinstance(p, list) and len(p) == 7:
                                self.left_pose_bank["A_l"].append(p)
                except Exception as e:
                    print(f"[Goal3Stage] load pose bank failed {path}: {e}")

            def _load_full(self, path: str):
                try:
                    with open(path, "r") as f:
                        payload = json.load(f)
                    gg = None
                    if isinstance(payload, list):
                        gg = payload
                    elif isinstance(payload, dict):
                        gg = payload.get("full_good_goals") or payload.get("goals") or payload.get("good_goals")
                    if not gg:
                        return
                    for g in gg:
                        k = self._goal_key(g)
                        if k not in self.full_keys:
                            self.full_good_goals.append(g)
                            self.full_keys.add(k)
                except Exception as e:
                    print(f"[Goal3Stage] load full failed {path}: {e}")

            # ---------- env bookkeeping ----------
            def set_env_noised_goal(self, env_i: int, processed_goal):
                """
                processed_goal: list[(name, pose_tensor_or_list)]
                store as jsonable list[(name, pose_list)]
                """
                goal_jsonable = []
                for name, pose_t in processed_goal:
                    pose_list = pose_t.detach().cpu().tolist() if torch.is_tensor(pose_t) else list(pose_t)
                    goal_jsonable.append((name, pose_list))
                self.env_current_noised[env_i] = goal_jsonable

            # ---------- core sampling ----------
            def _replace_pose_in_goal(self, raw_goal):
                """
                raw_goal: list[(name, pose_list)]
                returns: list[(name, pose_list)] with A_r/A_l replaced if banks have entries
                """
                out = []
                replaced_r = False
                replaced_l = False

                for name, pose in raw_goal:
                    if name == "A_r" and self.right_pose_bank["A_r"] and (not self.replace_first_only or not replaced_r):
                        pose = random.choice(self.right_pose_bank["A_r"])
                        replaced_r = True
                    elif name == "A_l" and self.left_pose_bank["A_l"] and (not self.replace_first_only or not replaced_l):
                        pose = random.choice(self.left_pose_bank["A_l"])
                        replaced_l = True
                    out.append((name, pose))
                return out

            def sample_for_env(self, env_i: int):
                if self.stage == "final_only" and self.full_good_goals:
                    g = random.choice(self.full_good_goals)
                    self.env_current[env_i] = g
                    return g

                if self.candidates:
                    base = self.candidates.pop(random.randrange(len(self.candidates)))
                else:
                    base = random.choice(all_goals_cartesian)

                if self.stage == "collect_right":
                    self.env_current[env_i] = base
                    return base

                if self.stage == "collect_left":
                    g = []
                    replaced_r = False
                    for name, pose in base:
                        if name == "A_r" and self.right_pose_bank["A_r"] and not replaced_r:
                            pose = random.choice(self.right_pose_bank["A_r"])
                            replaced_r = True
                        g.append((name, pose))
                    self.env_current[env_i] = g
                    return g

                g = []
                replaced_r = False
                replaced_l = False
                for name, pose in base:
                    if name == "A_r" and self.right_pose_bank["A_r"] and not replaced_r:
                        pose = random.choice(self.right_pose_bank["A_r"])
                        replaced_r = True
                    elif name == "A_l" and self.left_pose_bank["A_l"] and not replaced_l:
                        pose = random.choice(self.left_pose_bank["A_l"])
                        replaced_l = True
                    g.append((name, pose))

                self.env_current[env_i] = g
                return g
            def _extract_pose_by_name(self, env_i: int, name: str):
                """
                Try to extract pose list for given step name from env_current_noised[env_i]
                (preferred) else from env_current[env_i].
                Returns pose_list (len==7) or None
                """
                g = self.env_current_noised.get(env_i, None) or self.env_current.get(env_i, None)
                if g is None:
                    return None
                for step_name, pose in g:
                    if step_name == name:
                        # pose could be list already
                        pose_list = pose.detach().cpu().tolist() if torch.is_tensor(pose) else list(pose)
                        if isinstance(pose_list, list) and len(pose_list) == 7:
                            return pose_list
                return None

            def on_sub_success(self, env_indices: torch.Tensor, arm: str):
                if env_indices is None or env_indices.numel() == 0:
                    return
                env_list = env_indices.view(-1).detach().cpu().tolist()

                added_any = False

                if arm == "r":
                    for env_i in env_list:
                        pose = self._extract_pose_by_name(env_i, "A_r")
                        if pose is None:
                            continue
                        # dedup by json key
                        k = json.dumps(pose, separators=(",", ":"), sort_keys=True)
                        # (simple dedup set stored as python set on list itself)
                        # We'll keep a hidden set for speed:
                        if not hasattr(self, "_right_pose_keys"):
                            self._right_pose_keys = set(json.dumps(p, separators=(",", ":"), sort_keys=True) for p in self.right_pose_bank["A_r"])
                        if k in self._right_pose_keys:
                            continue
                        self.right_pose_bank["A_r"].append(pose)
                        self._right_pose_keys.add(k)
                        added_any = True
                        print(f"[Goal3Stage] +RPOSE env={env_i} | A_r")

                    if added_any:
                        self._save_right()

                elif arm == "l":
                    for env_i in env_list:
                        pose = self._extract_pose_by_name(env_i, "A_l")
                        if pose is None:
                            continue
                        k = json.dumps(pose, separators=(",", ":"), sort_keys=True)
                        if not hasattr(self, "_left_pose_keys"):
                            self._left_pose_keys = set(json.dumps(p, separators=(",", ":"), sort_keys=True) for p in self.left_pose_bank["A_l"])
                        if k in self._left_pose_keys:
                            continue
                        self.left_pose_bank["A_l"].append(pose)
                        self._left_pose_keys.add(k)
                        added_any = True
                        print(f"[Goal3Stage] +LPOSE env={env_i} | A_l")

                    if added_any:
                        self._save_left()

                prev = self.stage
                self._update_stage()
                if self.stage != prev:
                    print(self._status_str())

            # ---------- full success: store FULL script (after replacement) ----------
            def on_full_success(self, env_indices: torch.Tensor):
                if env_indices is None or env_indices.numel() == 0:
                    return
                env_list = env_indices.view(-1).detach().cpu().tolist()

                added_any = False
                for env_i in env_list:
                    g = self.env_current_noised.get(env_i, None) or self.env_current.get(env_i, None)
                    if g is None:
                        continue
                    k = self._goal_key(g)
                    if k not in self.full_keys:
                        self.full_good_goals.append(g)
                        self.full_keys.add(k)
                        added_any = True
                        print(f"[Goal3Stage] +FULL env={env_i} | {self._pretty_goal(g)}")

                if added_any:
                    self._save_full()

                prev = self.stage
                self._update_stage()
                if self.stage != prev:
                    print(self._status_str())

            def on_failure(self, env_indices: torch.Tensor, reason: str = "failure"):
                # keep hook for compatibility
                return


            """
            Stage R: collect right-subgrasp-good goals (size Nr) from candidates
            Stage L: collect left-subgrasp-good goals (size Nl) sampling only from right-good
            Stage F: collect full-good goals (size Nf) sampling only from left-good
            Stage Final: sample only from full-good
            """
            def __init__(
                self,
                all_candidates,
                target_right: int,
                target_left: int,
                target_full: int,
                right_in: str | None, right_out: str | None,
                left_in: str | None, left_out: str | None,
                full_in: str | None, full_out: str | None,
            ):
                self.target_right = int(target_right)
                self.target_left  = int(target_left)
                self.target_full  = int(target_full)

                self.right_out = right_out
                self.left_out  = left_out
                self.full_out  = full_out

                self.candidates = copy.deepcopy(list(all_candidates))

                self.right_good_goals = []
                self.right_keys = set()

                self.left_good_goals = []
                self.left_keys = set()

                self.full_good_goals = []
                self.full_keys = set()

                self.env_current = {}         # env -> raw goal (list[(name,opt)])
                self.env_current_noised = {}  # env -> jsonable executed goal

                if right_in:
                    self._load_any(right_in, self.right_good_goals, self.right_keys, key_name="right_good_goals")
                if left_in:
                    self._load_any(left_in, self.left_good_goals, self.left_keys, key_name="left_good_goals")
                if full_in:
                    self._load_any(full_in, self.full_good_goals, self.full_keys, key_name="full_good_goals")

                self._update_stage()
                print(self._status_str())

            def _goal_key(self, raw_goal) -> str:
                return json.dumps(raw_goal, sort_keys=True, separators=(",", ":"))

            def _pretty_goal(self, raw_goal) -> str:
                try:
                    return " -> ".join([step[0] for step in raw_goal])
                except Exception:
                    return str(raw_goal)

            def _load_any(self, path: str, dst_list: list, dst_keys: set, key_name: str):
                try:
                    with open(path, "r") as f:
                        payload = json.load(f)
                    gg = None
                    if isinstance(payload, list):
                        gg = payload
                    elif isinstance(payload, dict):
                        gg = payload.get(key_name) or payload.get("goals") or payload.get("good_goals")
                    if not gg:
                        return
                    for g in gg:
                        k = self._goal_key(g)
                        if k not in dst_keys:
                            dst_list.append(g); dst_keys.add(k)
                except Exception as e:
                    print(f"[Goal3Stage] load failed {path}: {e}")

            def _save(self, out_path: str | None, payload: dict):
                if not out_path:
                    return
                tmp = out_path + ".tmp"
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(tmp, "w") as f:
                        json.dump(payload, f, indent=2)
                    os.replace(tmp, out_path)
                except Exception as e:
                    print(f"[Goal3Stage] save failed {out_path}: {e}")

            def _save_right(self):
                self._save(self.right_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_right": self.target_right,
                    "right_good_goals": self.right_good_goals,
                })

            def _save_left(self):
                self._save(self.left_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_left": self.target_left,
                    "left_good_goals": self.left_good_goals,
                })

            def _save_full(self):
                self._save(self.full_out, {
                    "task": args_cli.task,
                    "saved_at": datetime.now().isoformat(),
                    "target_full": self.target_full,
                    "full_good_goals": self.full_good_goals,
                })

            def _status_str(self) -> str:
                return (f"[Goal3Stage] stage={self.stage} | "
                        f"A(cand)={len(self.candidates)} | "
                        f"R={len(self.right_good_goals)}/{self.target_right} | "
                        f"L={len(self.left_good_goals)}/{self.target_left} | "
                        f"F={len(self.full_good_goals)}/{self.target_full}")

            def _update_stage(self):
                if self.target_full > 0 and len(self.full_good_goals) >= self.target_full:
                    self.stage = "final_only"; return
                if self.target_right > 0 and len(self.right_good_goals) < self.target_right:
                    self.stage = "collect_right"; return
                if self.target_left > 0 and len(self.left_good_goals) < self.target_left:
                    self.stage = "collect_left"; return
                self.stage = "collect_full"

            def set_env_noised_goal(self, env_i: int, processed_goal):
                goal_jsonable = []
                for name, pose_t in processed_goal:
                    pose_list = pose_t.detach().cpu().tolist() if torch.is_tensor(pose_t) else list(pose_t)
                    goal_jsonable.append((name, pose_list))
                self.env_current_noised[env_i] = goal_jsonable

            def sample_for_env(self, env_i: int):
                if self.stage == "final_only" and self.full_good_goals:
                    g = random.choice(self.full_good_goals)
                    self.env_current[env_i] = g
                    return g

                if self.stage == "collect_right":
                    if self.candidates:
                        g = self.candidates.pop(random.randrange(len(self.candidates)))
                    else:
                        g = random.choice(self.right_good_goals) if self.right_good_goals else random.choice(all_goals_cartesian)
                    self.env_current[env_i] = g
                    return g

                if self.stage == "collect_left":
                    # IMPORTANT: only sample from right-good (otherwise left-good won’t be subset)
                    if self.right_good_goals:
                        g = random.choice(self.right_good_goals)
                    else:
                        g = random.choice(all_goals_cartesian)
                    self.env_current[env_i] = g
                    return g

                # collect_full
                if self.left_good_goals:
                    g = random.choice(self.left_good_goals)
                elif self.right_good_goals:
                    g = random.choice(self.right_good_goals)
                elif self.candidates:
                    g = self.candidates.pop(random.randrange(len(self.candidates)))
                else:
                    g = random.choice(all_goals_cartesian)
                self.env_current[env_i] = g
                return g

            def on_failure(self, env_indices: torch.Tensor, reason: str = "failure"):
                from goal_bank_retry import retire_failed_partial_goal
                retired = retire_failed_partial_goal(
                    self, env_indices.view(-1).detach().cpu().tolist(), reason)
                if retired:
                    for env_id, side, failures in retired:
                        print(f"[Goal3Stage] retired {side} partial goal after {failures} "
                              f"repeated recipe failures (env={env_id}, reason={reason})", flush=True)
                    self._save_right()
                    self._save_left()
                    self._update_stage()
                    print(self._status_str(), flush=True)

            def _add_goal(self, env_i: int, dst_list: list, dst_keys: set, tag: str) -> bool:
                g = self.env_current_noised.get(env_i, None) or self.env_current.get(env_i, None)
                if g is None:
                    return False
                k = self._goal_key(g)
                if k in dst_keys:
                    return False
                dst_list.append(g); dst_keys.add(k)
                print(f"[Goal3Stage] +{tag} env={env_i} | {self._pretty_goal(g)}")
                return True

            def on_sub_success(self, env_indices: torch.Tensor, arm: str):
                if env_indices is None or env_indices.numel() == 0:
                    return
                env_list = env_indices.view(-1).detach().cpu().tolist()

                added_any = False

                if arm == "r":
                    # Only meaningful during/after collect_right, but harmless otherwise
                    for env_i in env_list:
                        added_any |= self._add_goal(env_i, self.right_good_goals, self.right_keys, "RIGHT")
                    if added_any:
                        self._save_right()

                elif arm == "l":
                    # left-good should be filtered later, but we still dedup normally
                    for env_i in env_list:
                        added_any |= self._add_goal(env_i, self.left_good_goals, self.left_keys, "LEFT")
                    if added_any:
                        self._save_left()

                prev = self.stage
                self._update_stage()
                if self.stage != prev:
                    print(self._status_str())

            def on_full_success(self, env_indices: torch.Tensor):
                if env_indices is None or env_indices.numel() == 0:
                    return
                env_list = env_indices.view(-1).detach().cpu().tolist()

                added_any = False
                for env_i in env_list:
                    g = self.env_current.get(env_i, None)
                    if g is None:
                        continue
                    k = self._goal_key(g)
                    if k not in self.full_keys:
                        self.full_good_goals.append(g)
                        self.full_keys.add(k)
                        added_any = True
                        print(f"[Goal3Stage] +FULL env={env_i} | {self._pretty_goal(g)}")

                if added_any:
                    self._save_full()

                prev = self.stage
                self._update_stage()
                if self.stage != prev:
                    print(self._status_str())

        right_out = args_cli.sub_good_goals_r_out or os.path.join(output_dir, f"right_pose_bank_{args_cli.task}.json")
        left_out  = args_cli.sub_good_goals_l_out or os.path.join(output_dir, f"left_pose_bank_{args_cli.task}.json")
        full_out  = args_cli.good_goals_out       or os.path.join(output_dir, f"good_goals_{args_cli.task}.json")

        goal_mgr = ThreeStageGoalBufferManager(
            all_candidates=all_goals_cartesian,
            target_right=args_cli.sub_good_goal_count_r,
            target_left=args_cli.sub_good_goal_count_l,
            target_full=args_cli.good_goal_count,
            right_in=args_cli.sub_good_goals_r_in, right_out=right_out,
            left_in=args_cli.sub_good_goals_l_in,  left_out=left_out,
            full_in=args_cli.good_goals_in,        full_out=full_out,
        )

        # --- helper: resample goals + noise for selected envs and rebuild tensors ---
        def resample_goals_for_envs(env_indices: torch.Tensor, target_idx):
            nonlocal all_goals, task_ids_tensor, skill_ids_tensor, payloads_tensor, resolved_tensor

            if env_indices is None or env_indices.numel() == 0:
                return

            env_list = env_indices.view(-1).detach().cpu().tolist()
            noise_idx = 0

            for env_i in env_list:
                goal = goal_mgr.sample_for_env(env_i)

                # (A) RAW goal (no noise) -> what you want to SAVE
                processed_goal_raw = []
                # (B) EXEC goal (with noise) -> what the env actually EXECUTES
                processed_goal_save = []
                processed_goal_exec = []

                apply_noise = (goal_mgr.stage == "collect_right") or (goal_mgr.stage == "collect_left")

                for step_i, (name, pose) in enumerate(goal):
                    # raw tensor (no noise)
                    pose_raw = torch.tensor(pose, dtype=torch.float32, device=device)
                    pose_save = pose_raw.clone()
                    
                    if apply_noise and (name == "N_s"):
                        pose_save[0:2] += torch.randn(2, device=device) * noise_xy_std
                        pose_save[2] += torch.randn((), device=device) * noise_yaw_std
                    elif apply_noise and (name == "A_b"):
                        pose_save[0:3] += torch.randn(3, device=device) * noise_xyz_std
                        pose_save[7:10] += torch.randn(3, device=device) * noise_xyz_std
                    elif apply_noise and ((name == "A_r") or (name == "A_l")):
                        pose_save[0:3] += torch.randn(3, device=device) * noise_xyz_std

                    pose_exec = pose_save.clone()
                    if (step_i == target_idx) and (variable.rand_samples_simvla is not None):
                        noise = variable.rand_samples_simvla[noise_idx]  # (6,)
                        pose_exec[:6] += noise

                    processed_goal_save.append((name, pose_save))
                    processed_goal_exec.append((name, pose_exec))

                # ✅ Save RAW (unnoised) version for sub-grasp / success buffers
                goal_mgr.set_env_noised_goal(env_i, processed_goal_save)

                # ✅ Execute NOISED version in the env
                all_goals[env_i] = processed_goal_exec

                noise_idx += 1

                # --- rebuild task_ids, skill_ids & payloads using the EXECUTED (noised) goal ---
                task_ids_list_env = []
                skill_ids_list_env = []
                payloads_list_env = []

                if len(processed_goal_exec) != len(step_skill_ids):
                    raise RuntimeError(
                        f"env {env_i}: sampled goal has {len(processed_goal_exec)} steps but the "
                        f"goal file declares {len(step_skill_ids)}. The skill ids are indexed by "
                        f"step position, so a length mismatch would dispatch the wrong skill."
                    )

                for step_j, (task, data_t) in enumerate(processed_goal_exec):
                    task_ids_list_env.append(TASK_IDS[task])
                    skill_ids_list_env.append(step_skill_ids[step_j])

                    payload = torch.zeros(max_payload_size, device=device)
                    if task in ["N", "A_l", "A_r", "A_b", "N_s"]:
                        if data_t.ndim == 1:
                            payload[: data_t.numel()] = data_t
                        else:
                            idx_rand = torch.randint(data_t.size(0), (1,), device=device)
                            rand_data = data_t[idx_rand]
                            payload[: rand_data.numel()] = rand_data.squeeze(0)
                    elif task in ["G_l", "G_r"]:
                        payload[0] = 1.0 if data_t.item() else -1.0

                    payloads_list_env.append(payload)

                num_goals_env = len(processed_goal_exec)
                padding_needed = max_sequence_length - num_goals_env
                if padding_needed > 0:
                    task_ids_list_env.extend([PADDING_TASK_ID] * padding_needed)
                    skill_ids_list_env.extend([PADDING_SKILL_ID] * padding_needed)
                    payloads_list_env.extend([torch.zeros(max_payload_size, device=device)] * padding_needed)

                task_ids_tensor[env_i] = torch.tensor(task_ids_list_env, device=device, dtype=torch.long)
                skill_ids_tensor[env_i] = torch.tensor(skill_ids_list_env, device=device, dtype=torch.long)
                payloads_tensor[env_i] = torch.stack(payloads_list_env)
                # Rebuilding the payload puts the runtime steps back into their unresolved state.
                # In v1 that was implicit: resolving a step overwrote the sentinel it was recognised
                # by, so it could not be resolved twice, and rebuilding the payload put the sentinel
                # back. Dispatching on the skill id has no such side effect, so the latch is
                # explicit — and it is REQUIRED, not decorative: nav.open_articulation resolves to a
                # pose 0.25 m behind wherever the base is NOW, so re-resolving it every tick would
                # move the goal with the robot and it would reverse forever.
                resolved_tensor[env_i] = False

        # noise
        # There is deliberately no rotation-noise knob here. `rot_noise_std = 0.0` sat on the next
        # line, defined and never read, one edit away from being the most expensive typo in the
        # repo: v1 matched its arm sentinels in the quaternion slots [3:7], and rotation noise lands
        # exactly there, so turning it on would have silently demoted every runtime-resolved arm
        # step to a literal IK target. Dispatch is on skill ids now, so the trap is disarmed — but
        # nothing wants the variable, so nothing should offer it.
        noise_xy_std, noise_yaw_std, noise_xyz_std = goal_noise_standard_deviations()
        print(f"[goal-noise] navigation XY={noise_xy_std} m, yaw={noise_yaw_std} rad, "
              f"arm XYZ={noise_xyz_std} m (grasp-search stages only)", flush=True)

        all_goals = []
        for i in range(num_envs):
            goal = goal_mgr.sample_for_env(i)
            processed_goal = []
            for name, pose in goal:
                pose_t = torch.tensor(pose, dtype=torch.float32, device=device)  # NO NOISE here
                processed_goal.append((name, pose_t))
            all_goals.append(processed_goal)
            goal_mgr.set_env_noised_goal(i, processed_goal)  # store as executed (initially unnoised)

        kitchen_type = data["kitchen_type"]
        if kitchen_type == "island":
            island_min = [data["island_bound"][0],data["island_bound"][2]]
            island_max = [data["island_bound"][1],data["island_bound"][3]]

        if args_cli.task_type == "Navigation":
            from isaaclab.simvla import terminations
            env_cfg.terminations.success.func = terminations.navigation 
            if script[0].action == "N_s":
                goal_pos = torch.tensor(script[0].spec, device=device)
            else:
                raise ValueError(
                    f"Navigation task expected first goal to be an 'N_s' "
                    f"(nav-to-spot) skill; got {script[0].action!r}"
                )

            env_cfg.terminations.success.params = {"goal_pos":goal_pos}
    # Domain Randomization
    # Camera
    camera_seed, light_seed = augmentation_seeds(args_cli.seed)
    print(f"[augmentation-seeds] camera={camera_seed} light={light_seed}", flush=True)
    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.05, rot_max_deg=10.0, yaw_only=True, seed=camera_seed, cams = "front")
    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.01, rot_max_deg=5.0, yaw_only=False, seed=camera_seed, cams = "wrist_right")
    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.01, rot_max_deg=5.0, yaw_only=False, seed=camera_seed, cams = "wrist_left")
    # Light
    env_cfg = add_light_noise_to_scene_cfg_once(
       env_cfg,
       intensity_base=15000.0,
       intensity_stop_range=1.5,  # intensity *= 2^{U(-1.5, 1.5)}
       kelvin_choices=(2700, 3200, 4000, 5000, 6500, 8000, 9000),
       rgb_jitter_sigma=0.08,
       seed=light_seed,
    )
    # To opimize env
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    _ik_deadzone = apply_ik_joint_deadzone(env)
    print(f"[ik] per-joint DLS deadzone set to {_ik_deadzone:g} rad", flush=True)
    # Reset environment
    env.reset()
    # Success demo
    current_recorded_demo_count = 0

    base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
    
    # define goal types with integer ids for tensor operations
    TASK_IDS = {"N": 0, "A_l": 1, "A_r": 2, "G_l": 3, "G_r": 4, "N_s": 5, "A_b": 6}
    N_ID, AL_ID, AR_ID, GL_ID, GR_ID, NS_ID, AB_ID = TASK_IDS["N"], TASK_IDS["A_l"], TASK_IDS["A_r"], TASK_IDS["G_l"], TASK_IDS["G_r"], TASK_IDS["N_s"], TASK_IDS["A_b"]
    
    # --- Find the longest goal sequence to determine padding ---
    max_sequence_length = max(len(goals) for goals in all_goals)
    
    variable.max_sequence_length = max_sequence_length

    max_payload_size = 14
    all_task_ids_list = []
    all_skill_ids_list = []
    all_payloads_list = []

    # Special ID for padded steps, so we can ignore them
    PADDING_TASK_ID = -1

    for env_idx in range(num_envs):
        goals = all_goals[env_idx]
        task_ids_list_env = []
        skill_ids_list_env = []
        payloads_list_env = []

        for step_j, (task, data) in enumerate(goals):
            task_ids_list_env.append(TASK_IDS[task])
            skill_ids_list_env.append(step_skill_ids[step_j])

            payload = torch.zeros(max_payload_size, device=device)
            if task in ["N", "N_s", "A_l", "A_r", "A_b"]:
                if len(data.size()) == 1:
                    payload[:data.numel()] = data
                else:
                    idx = torch.randint(data.size(0), (1,))
                    rand_data = data[idx]
                    payload[:data.numel()] = rand_data
            elif task in ["G_l", "G_r"]:
                payload[0] = 1.0 if data.item() else -1.0
            payloads_list_env.append(payload)

        # --- Pad the current environment's script to the max length ---
        num_goals = len(goals)
        padding_needed = max_sequence_length - num_goals

        task_ids_list_env.extend([PADDING_TASK_ID] * padding_needed)
        # PADDING_SKILL_ID is -1 and the real ids are 0..n-1, so a padded row equals no skill and
        # every `skill_ids_tensor == SKILL_ID[...]` mask below is False on it. (It cannot be reached
        # anyway: its task id is -1 too, so no action block claims it.)
        skill_ids_list_env.extend([PADDING_SKILL_ID] * padding_needed)
        payloads_list_env.extend([torch.zeros(max_payload_size, device=device)] * padding_needed)

        all_task_ids_list.append(torch.tensor(task_ids_list_env, device=device, dtype=torch.long))
        all_skill_ids_list.append(torch.tensor(skill_ids_list_env, device=device, dtype=torch.long))
        all_payloads_list.append(torch.stack(payloads_list_env))

    task_ids_tensor = torch.stack(all_task_ids_list)
    # Which SKILL produced each step. This is the executor's dispatch: every branch below asks this
    # tensor, not the payload floats. Shape: (num_envs, max_sequence_length).
    skill_ids_tensor = torch.stack(all_skill_ids_list)
    # Shape: (num_envs, max_sequence_length, max_payload_size)
    payloads_tensor = torch.stack(all_payloads_list)    # The final tensors that define the "script" for all environments

    # Has this (env, step) cell's runtime goal already been resolved? See resample_goals_for_envs:
    # v1 got this for free because resolving a step destroyed the sentinel it was recognised by.
    resolved_tensor = torch.zeros(num_envs, max_sequence_length, dtype=torch.bool, device=device)

    # The skill ids this file's dispatch compares against. Looked up once: `skill_ids` is keyed by
    # name, and a typo'd name must fail here, at startup, not silently never match.
    SID_PULL = skill_ids["nav.open_articulation"]
    SID_PUSH = skill_ids["nav.close_articulation"]
    SID_TO_POT = skill_ids["nav.to_pot"]
    SID_TO_RANGE = skill_ids["nav.to_range"]
    SID_BOWL_PLACE = skill_ids["arm.bowl_place"]
    SID_BOTTLE_TO_POSITION = skill_ids["arm.bottle_to_position"]   # v1's MOVE_LEFT  (A_r)
    SID_MUG_TO_POSITION = skill_ids["arm.mug_to_position"]         # v1's MOVE_RIGHT (A_l)
    SID_BOTTLE_POUR = skill_ids["arm.bottle_pour"]                 # v1's BOTTLE_TILT
    SID_MOVE_FRONT = skill_ids["arm.move_front"]
    SID_GRASP_TARGET = skill_ids["arm.grasp_target"]
    SID_PAUSE = skill_ids["arm.pause"]
    SID_RESET = skill_ids["arm.reset"]
    SID_PLACE = skill_ids["arm.place"]
    SID_POSE = skill_ids["arm.pose"]
    SID_GRASP = skill_ids["arm.grasp"]
    SID_HANDLE_PREGRASP = skill_ids["arm.handle_pregrasp"]
    SID_HANDLE_GRASP = skill_ids["arm.handle_grasp"]
    SID_BAR_HANDLE_PREGRASP = skill_ids["arm.bar_handle_pregrasp"]
    SID_BAR_HANDLE_GRASP = skill_ids["arm.bar_handle_grasp"]
    SID_POT = skill_ids["arm.pot"]
    SID_ARC = skill_ids["nav.open_door_arc"]

    # --- Initial resample of goals for all envs after first reset ---
    init_envs = torch.arange(num_envs, device=device)
    resample_goals_for_envs(init_envs, args_cli.target_idx)


    # --- Per-environment state variables ---
    env_goal_indices = torch.zeros(num_envs, device=device, dtype=torch.long)
    env_cmd_plans_r = [None] * num_envs
    env_cmd_plans_l = [None] * num_envs
    env_cmd_indices_r = torch.zeros(num_envs, device=device, dtype=torch.long)
    env_cmd_indices_l = torch.zeros(num_envs, device=device, dtype=torch.long)
    env_r_gripper_dist_prev = torch.full((num_envs,), 10.0, device=device)
    env_l_gripper_dist_prev = torch.full((num_envs,), 10.0, device=device)

    # HOW LONG THE JAWS MUST BE STILL BEFORE THE STEP COUNTS AS DONE.
    #
    # The gripper step used to finish the instant the jaw separation stopped changing by more than
    # 1 mm in ONE tick. A gripper closing on a handle is not moving smoothly -- it can be
    # momentarily still while the controller ramps, while a contact settles, or between solver
    # substeps -- so a single quiet tick ends the step with the hand not yet holding anything. The
    # next step then runs against an open or half-closed gripper.
    #
    # Observed on the open-the-fridge task, and spotted by watching the video rather than the
    # counters: the base began its arc before the jaws had closed on the door handle, so there was
    # nothing gripping the bar when the pull started and the door never moved.
    #
    # Requiring the stillness to PERSIST costs half a second on a step that is otherwise
    # instantaneous, and makes "the gripper has finished closing" mean what it says.
    GRIPPER_SETTLE_TICKS = int(os.environ.get("SIMVLA_GRIPPER_SETTLE", "10"))
    GRIPPER_CLOSE_MAX_TICKS = int(os.environ.get("SIMVLA_GRIPPER_CLOSE_MAX_TICKS", "80"))
    _open_target_tol_raw = os.environ.get("SIMVLA_GRIPPER_OPEN_TARGET_TOL_M")
    _open_target_tol = (float(_open_target_tol_raw)
                        if _open_target_tol_raw is not None else None)
    if (_open_target_tol is not None
            and (not math.isfinite(_open_target_tol)
                 or not 0 < _open_target_tol <= 0.01)):
        raise ValueError("SIMVLA_GRIPPER_OPEN_TARGET_TOL_M must be in (0, 0.01] m")
    if GRIPPER_SETTLE_TICKS < 1 or GRIPPER_CLOSE_MAX_TICKS < GRIPPER_SETTLE_TICKS:
        raise ValueError("gripper settle ticks must be positive and no greater than close max ticks")
    print(f"[gripper] close_max_ticks={GRIPPER_CLOSE_MAX_TICKS} "
          f"settle_ticks={GRIPPER_SETTLE_TICKS} "
          f"aiworker_speed_rad_s={os.environ.get('SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S', '1.0')}",
          flush=True)

    # ---------------------------------------------------------------------------------------
    # POST-GRASP LIFT, AND WHY IT IS NOT A cuRobo PLAN.
    #
    # The instant the jaws arrest on the object the hand is, by construction, touching it,
    # millimetres over the surface it stood on -- and that is the state the NEXT plan has to
    # start from. cuRobo refuses it. Measured on this task before the lift existed here: the
    # reach worked (10 successes, 29 re-plans, zero reach resets) and every episode still died,
    # 21 times, as plan_fail_r -- the step after the grasp, planning from a hand on the bottle.
    #
    # So the lift is done HERE, as a straight-up Cartesian jog through the same relative-IK
    # channel the plan waypoints drive (pose_R is zero for an env sitting in a G step, so the
    # row is free), and only then does the G step finish. Two things come of it: the object
    # actually rises, and the next plan starts from a hand held ABOVE the counter, which cuRobo
    # accepts.
    #
    # This was already in simvla_video.py and NOT here, while the job scripts set the variable
    # for both -- so the runs looked configured for it and silently were not. On Anubis the same
    # fix is the difference between 0 and 34 recorded demonstrations.
    #
    # SIMVLA_POSTGRASP_LIFT is the height in metres; unset/0 = off, byte-identical behaviour.
    # 0.0025 m/step at the 20 Hz env rate is 0.05 m/s commanded, the pace the plan tracker
    # already drives this arm at.
    _lift_m = float(os.environ.get("SIMVLA_POSTGRASP_LIFT", "0") or 0)
    _lift_left = left_postgrasp_lift_enabled(os.environ.get("SIMVLA_POSTGRASP_LIFT_LEFT"))
    if _lift_left and _lift_m <= 0:
        raise ValueError("SIMVLA_POSTGRASP_LIFT_LEFT=1 requires SIMVLA_POSTGRASP_LIFT > 0")
    _LIFT_STEP_M = 0.0025
    _lift_steps = max(1, round(_lift_m / _LIFT_STEP_M)) if _lift_m > 0.0 else 0
    #: Per-env countdown of lift steps remaining; 0 = not lifting. Armed after the jaws settle
    #: under a CLOSE command, then run to completion even if object contact moves the jaws.
    #: Explicitly cleared when an episode's timestep restarts at one.
    env_lift_r = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_lift_l = torch.zeros(num_envs, dtype=torch.int32, device=device)
    #: The object's height at the moment each close arms its lift, so the lift log can report how
    #: far the OBJECT actually rose rather than only how far the hand did.
    _obj_z0 = {}
    _obj_z0_l = {}
    #: base_link xy at the moment each arm goal was captured, for the drift check above.
    _base_at_plan = {}
    #: pad-to-object miss and pad height AT THE MOMENT THE JAWS CLOSE, which is what decides the
    #: grasp -- as opposed to the same quantities after the post-grasp lift has run.
    _miss0, _padz0 = {}, {}
    #: distance from the authored grasp GOAL to the object, in world, at close.
    _goal_miss = {}
    #: the object's tilt from upright, in degrees, at the moment the jaws close.
    _tilt = {}
    #: the object's tilt BEFORE the grasp reach starts, to locate when it goes over.
    _tilt_at_plan = {}
    #: last reported tilt for the per-step trace, so it only prints on change.
    _tilt_trace_prev = [0.0]
    #: (position, quaternion) of each arm goal in WORLD, so it survives base motion.
    _goal_w_r = {}
    #: Left-arm counterpart; world-frame goals are needed when its mobile base moves during approach.
    _goal_w_l = {}
    if _lift_steps:
        print(f"[lift] post-grasp lift {_lift_m} m = {_lift_steps} steps of {_LIFT_STEP_M} m "
              f"(right; left={'enabled' if _lift_left else 'disabled'})",
              flush=True)
    env_r_gripper_still = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    env_l_gripper_still = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    env_r_gripper_ticks = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    env_l_gripper_ticks = torch.zeros((num_envs,), dtype=torch.int32, device=device)
    ik_goals_r = [None] * num_envs 
    ik_goals_l = [None] * num_envs 
    robot = env.scene.articulations["robot"]
    r_eef_idx = robot.find_bodies("ee_link1")[0][0]
    l_eef_idx = robot.find_bodies("ee_link2")[0][0]

    print("Setup for parallel execution complete.")
    # Hyperparameter (Navigation) -- PER ROBOT, from nav_tuning, not hard-coded.
    #
    # These five were literals here, and the four numbers among them were Anubis's. nav_tuning
    # exists precisely because robots differ in mass, base drive and footprint, and RB-Y1 and AI
    # Worker each have a tuned profile -- but only simvla_video ever read them, so every
    # simvla_gen run drove every robot at Anubis's settings.
    #
    # BEHAVIOUR-PRESERVING FOR ANUBIS: PROFILES["anubis"] is (0.02, 0.1, 0.15, 0.1, 0.75), which
    # is byte-identical to the literals this replaces.
    _nav = nav_tuning.profile_for(args_cli.robot)
    goal_reached_distance = _nav.goal_reached_distance
    goal_reached_yaw = _nav.goal_reached_yaw
    # Preserve the collector's historical final tolerance unless explicitly
    # overridden. The profile's separate final-yaw field was previously ignored.
    goal_final_yaw = (_nav.goal_final_yaw if "SIMVLA_NAV_FINAL_YAW" in os.environ
                      else goal_reached_yaw)
    default_speed = _nav.default_speed
    min_speed = _nav.min_speed
    slowdown_radius = _nav.slowdown_radius
    print(f"[nav] profile for {args_cli.robot!r}: dist={goal_reached_distance} "
          f"yaw={goal_reached_yaw} final_yaw={goal_final_yaw} speed={default_speed} min={min_speed} "
          f"slowdown={slowdown_radius}", flush=True)
    # Hyperparameter (cuRobo)
    num_targets = 0
    n_obstacle_cuboids = 100
    n_obstacle_mesh = 8000
    target_pose = None
    tensor_args = TensorDeviceType(device="cuda:0")
    # cuRobo's ``success`` is an endpoint-pose gate, not just an optimizer status. The old 10 m /
    # 10 radian-scale thresholds made a home-pose trajectory count as successful for a grasp
    # waypoint (public AI Worker trace: 0.697 m and 0.786 rotation-metric residual). Keep the
    # planner gate tighter than the collector's physical grasp gate so bad trajectories enter
    # the bounded Cartesian recovery path instead of being executed as successful plans.
    rotation_threshold = 0.05
    position_threshold = 0.02
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    # cuRobo retries and seed count dominate end-to-end collection time on public kitchen
    # tasks (dense USD mesh collision checks). Keep the tuned production defaults, but expose
    # validated per-run overrides so a one-demo release smoke can complete in an ordinary Slurm
    # allocation instead of spending its whole wall time in planner retries.
    max_attempts, num_trajopt_seeds, num_graph_seeds, enable_finetune_trajopt = (
        curobo_search_budget()
    )
    print(f"[planner] attempts={max_attempts} trajopt_seeds={num_trajopt_seeds} "
          f"graph_seeds={num_graph_seeds} finetune={enable_finetune_trajopt}", flush=True)
    print(f"[reset] Cartesian home gate: {RESET_POS_TOL_M:.3f} m / "
          f"{RESET_ROT_TOL_DEG:.1f} deg", flush=True)
    interpolation_dt = 0.03
    cmd_plan = None
    cmd_plan_l = None
    # Vendored simvla curobo robot configs (anubis/aiworker arm .yml) live in
    # the repo, not the curobo submodule. Resolved against the repo root so any
    # clone works regardless of where it was checked out.
    robot_cfg_path = str(simvla_paths.assets_dir() / "curobo/robot")
    dummy = torch.tensor([1e10, 1e10, 0], device='cuda:0')
    # Gripper
    gripper_command_L = False
    gripper_command_R = False

    if args_cli.robot == "anubis":
        right_arm = "anubis_right_arm.yml"
        left_arm = "anubis_left_arm.yml"
    elif args_cli.robot == "aiworker":
        robot_cfg_path = str(simvla_paths.repo_root() / "configs/curobo/robot")
        right_arm = "aiworker_right_arm.yml"
        left_arm = "aiworker_left_arm.yml"
    elif args_cli.robot == "rby1":
        # The project-tuned planner configs are small source files in this repository. The
        # Rainbow Robotics model remains external under SIMVLA_RBY1M_DIR.
        robot_cfg_path = str(simvla_paths.repo_root() / "configs/curobo/robot")
        right_arm = "rby1_right_arm.yml"
        left_arm = "rby1_left_arm.yml"
    r_robot_cfg = load_yaml(join_path(robot_cfg_path, right_arm))["robot_cfg"]
    l_robot_cfg = load_yaml(join_path(robot_cfg_path, left_arm))["robot_cfg"]
    # The vendored configs store repo-relative urdf/usd/asset paths; absolutize
    # them against the repo root before handing the config to curobo.
    for _cfg in (r_robot_cfg, l_robot_cfg):
        _kin = _cfg["kinematics"]
        # collision_spheres IS IN THIS LIST and was not, which is why the AI Worker run died with
        # FileNotFoundError on
        #   third_party/curobo/src/curobo/content/configs/robot/source/isaaclab_assets/...
        # -- curobo had joined the repo-relative path onto its OWN content root. simvla_video.py
        # already handles it; this copy did not.
        #
        # curobo accepts collision_spheres either INLINE (a dict, as the anubis configs write it)
        # or as a PATH to a second yaml (as the rby1 and aiworker ones do). Only the string form
        # is a path, hence the isinstance guard -- without it the dict form would be mangled into
        # a path join.
        for _key in ("urdf_path", "usd_path", "usd_robot_root", "asset_root_path",
                     "collision_spheres"):
            _val = _kin.get(_key)
            if not _val or not isinstance(_val, str):
                continue
            _val = os.path.expandvars(_val)
            if "$" in _val:
                raise SystemExit(f"{_key} contains an unset environment variable: {_val}")
            if os.path.isabs(_val):
                _kin[_key] = _val
            elif _key == "collision_spheres" and Path(_val).parts[:2] == ("configs", "curobo"):
                _kin[_key] = str(simvla_paths.repo_root() / _val)
            else:
                _kin[_key] = str(simvla_paths.robot_asset_path(_val))
    r_j_names = r_robot_cfg["kinematics"]["cspace"]["joint_names"]
    l_j_names = l_robot_cfg['kinematics']['cspace']['joint_names']
    r_j_index = env.scene.articulations['robot'].find_joints(r_j_names)[0]
    l_j_index = env.scene.articulations['robot'].find_joints(l_j_names)[0]
    aiworker_lift_ids = None
    aiworker_lift_target = None
    if args_cli.robot == "aiworker":
        # The lift is locked in cuRobo, so both its lock value and the retract vector must match
        # the Isaac reset state. This also keeps an explicit SIMVLA_AIWORKER_LIFT_M=0 override
        # valid without maintaining a second set of planner YAMLs.
        from aiworker_home import sync_planner_home
        for _cfg, _names, _indices in (
            (r_robot_cfg, r_j_names, r_j_index),
            (l_robot_cfg, l_j_names, l_j_index),
        ):
            _start = robot.data.default_joint_pos[0, _indices].detach().cpu().tolist()
            sync_planner_home(_cfg, dict(zip(_names, _start)))
            # The loaded column settles above its position target. Lock the planner
            # to the measured lift, otherwise every FK wrist is about 13 mm too low.
            _lift_slot = _names.index("lift_joint")
            _cfg["kinematics"]["lock_joints"]["lift_joint"] = float(
                robot.data.joint_pos[0, _indices[_lift_slot]])
        aiworker_lift_ids = robot.find_joints(["lift_joint"])[0]
        aiworker_lift_target = robot.data.default_joint_pos[:, aiworker_lift_ids].clone()
    # First, get the world configuration from the USD stage
    usd_helper = UsdHelper()
    install_polygon_mesh_reader(_curobo_usd_helper)
    usd_helper.load_stage(env.sim.stage)
    root_pos = robot.data.body_pos_w[:,base_link_idx]
    root_quat = robot.data.body_quat_w[:,base_link_idx]


    _world_mode = os.environ.get(
        "SIMVLA_CUROBO_WORLD", "live" if _direct_bodex_grasp_l else "dummy")
    if _world_mode not in {"live", "dummy"}:
        raise ValueError("SIMVLA_CUROBO_WORLD must be live or dummy")
    env_0_pos = env.scene.env_origins[0]
    if _world_mode == "live":
        # Cache static kitchen poses in environment coordinates, then transform into
        # the current base frame before each plan (the base moves between skills).
        r_T_w = pose_to_gf_matrix_tensor(
            env_0_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
    else:
        r_T_w = pose_to_gf_matrix_tensor(dummy, root_quat[0,:])
        print("[WORLD] dummy mode: kitchen collision checking is disabled", flush=True)
    _world_ignore = ["/Robot"]
    _target_collision_mode = os.environ.get("SIMVLA_CUROBO_TARGET_COLLISION", "exclude")
    if _target_collision_mode not in {"exclude", "approach"}:
        raise ValueError("SIMVLA_CUROBO_TARGET_COLLISION must be exclude or approach")
    if _target_collision_mode == "approach" and _world_mode != "live":
        raise ValueError("Target approach collision checking requires SIMVLA_CUROBO_WORLD=live")
    for _world_obj in (args_cli.obj_name, args_cli.obj_name_l):
        if _world_obj and _world_obj != "none" and _target_collision_mode == "exclude":
            _world_ignore.append(f"/Kitchen/{_world_obj}")
    world_cfg = usd_helper.get_obstacles_from_stage_simvla(
        only_paths=["env_0"], ignore_substring=_world_ignore, r_T_w=r_T_w,
    )
    # The mesh checker directly loads meshes/cuboids only. Convert USD spheres,
    # capsules and cylinders first so they are included in both collision and pose caches.
    world_cfg = world_cfg.get_collision_check_world()
    _target_obstacles = {"r": [], "l": []}
    if _target_collision_mode == "approach":
        for _side, _target in (("r", args_cli.obj_name), ("l", args_cli.obj_name_l)):
            if _target and _target != "none":
                _prefix = f"/World/envs/env_0/Kitchen/{_target}"
                _target_obstacles[_side] = [o.name for o in world_cfg.objects
                    if o.name == _prefix or o.name.startswith(_prefix + "/")]
                if not _target_obstacles[_side]:
                    raise RuntimeError(f"No collision geometry for grasp target {_prefix}")
    print(f"[WORLD] mode={_world_mode} obstacles={len(world_cfg.objects)}", flush=True)
    if _payload_collision_l:
        if _world_mode != "live" or _target_collision_mode != "approach":
            raise ValueError("Payload collision requires live world and approach target collision")
        from carried_collision import PAYLOAD_LINK, box_sphere_cover, configure_payload_proxy
        configure_payload_proxy(l_robot_cfg)
    print(f"Initializing Motion Generator for {args_cli.robot} right arm...")
    motion_gen_config_r = MotionGenConfig.load_from_robot_config(
        r_robot_cfg,
        world_cfg,
        tensor_args,
        rotation_threshold = rotation_threshold,
        position_threshold = position_threshold,
        collision_checker_type = CollisionCheckerType.MESH,
        num_trajopt_seeds = num_trajopt_seeds,
        num_graph_seeds = num_graph_seeds,
        interpolation_dt = interpolation_dt,
        collision_cache = {"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt = optimize_dt,
        trajopt_dt = trajopt_dt,
        trajopt_tsteps = trajopt_tsteps,
        trim_steps = trim_steps,
        collision_activation_distance=1e-3,
        maximum_trajectory_dt = 1,
    )
    motion_gen_r = MotionGen(motion_gen_config_r)
    if _world_mode == "live":
        motion_gen_r.clear_world_cache()
    # warmup_js_trajopt=True, BECAUSE arm.reset FALLS BACK TO plan_single_js.
    #
    # cuRobo captures a CUDA graph for each solver during warmup, and a solver that was not
    # warmed up cannot be called later under inference_mode: it dies with
    #
    #   RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    #
    # from deep inside newton_base, which names nothing recognisable. Measured: with this False,
    # the first joint-space fallback killed the run outright. Warmup costs a little more time and
    # buys a solver that actually works when it is needed.
    motion_gen_r.warmup(enable_graph=True, warmup_js_trajopt=True)
    if _world_mode == "live":
        motion_gen_r.update_world(world_cfg)
    
    print(f"Initializing Motion Generator for {args_cli.robot} left arm...")
    motion_gen_config_l = MotionGenConfig.load_from_robot_config(
        l_robot_cfg,
        world_cfg,
        tensor_args,
        rotation_threshold = rotation_threshold,
        position_threshold = position_threshold,
        collision_checker_type = CollisionCheckerType.MESH,
        num_trajopt_seeds = num_trajopt_seeds,
        num_graph_seeds = num_graph_seeds,
        interpolation_dt = interpolation_dt,
        collision_cache = {"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt = optimize_dt,
        trajopt_tsteps = trajopt_tsteps,
        trim_steps = trim_steps,
        collision_activation_distance=1e-3,
        maximum_trajectory_dt = 1,
    )
    motion_gen_l = MotionGen(motion_gen_config_l)
    if _world_mode == "live":
        motion_gen_l.clear_world_cache()
    motion_gen_l.warmup(enable_graph=True, warmup_js_trajopt=True)   # see the right arm's note
    if _world_mode == "live":
        motion_gen_l.update_world(world_cfg)

    # Ported from the research collector's live-world path. Update existing cache
    # pose tensors so CUDA graph references remain valid and meshes are not reloaded.
    _world_reposers = {}
    if _world_mode == "live":
        _local_obstacles = {o.name: o.pose for o in world_cfg.objects}
        # Fabric/PhysX updates need not be written back to USD. Bind rigid-object
        # meshes to their physics roots, not stale authored scene transforms.
        _rigid_world_roots = {}
        _root_cache = UsdGeom.XformCache()
        for _asset_name, _asset in env.scene.rigid_objects.items():
            _root_path = _asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            _root_path = _root_path.replace("env_.*", "env_0")
            _prim = env.sim.stage.GetPrimAtPath(_root_path)
            if not _prim.IsValid():
                raise RuntimeError(f"Cannot bind live collision root {_root_path}")
            _matrix = _root_cache.GetLocalToWorldTransform(_prim)
            _translation = _matrix.ExtractTranslation()
            _rotation = _matrix.ExtractRotationQuat()
            _root_p = torch.tensor(list(_translation), device=device, dtype=torch.float32) - env_0_pos
            _root_q = torch.tensor([_rotation.GetReal(), *list(_rotation.GetImaginary())],
                                   device=device, dtype=torch.float32)
            _rigid_world_roots[_root_path] = (_asset_name, _root_p, _root_q)
        for _side, _mg in (("r", motion_gen_r), ("l", motion_gen_l)):
            _wc = _mg.world_coll_checker
            _groups = []
            for _attr, _names_attr in (("_mesh_tensor_list", "_env_mesh_names"),
                                        ("_cube_tensor_list", "_env_obbs_names")):
                _cache = getattr(_wc, _attr, None)
                _names = getattr(_wc, _names_attr, None)
                if _cache is None or _names is None:
                    continue
                _rows, _poses = [], []
                for _row, _name in enumerate(_names[0]):
                    if _name in _local_obstacles:
                        _rows.append(_row)
                        _poses.append(_local_obstacles[_name])
                if _rows:
                    _p = torch.tensor(_poses, dtype=torch.float32, device=device)
                    _bindings = []
                    _mapped_names = [_names[0][_row] for _row in _rows]
                    for _root_path, (_asset_name, _root_p, _root_q) in _rigid_world_roots.items():
                        _slots = [j for j, name in enumerate(_mapped_names)
                                  if name == _root_path or name.startswith(_root_path + "/")]
                        if not _slots:
                            continue
                        _slots_t = torch.tensor(_slots, device=device)
                        _rel_p, _rel_q = math_utils.subtract_frame_transforms(
                            _root_p.expand(len(_slots), 3), _root_q.expand(len(_slots), 4),
                            _p[_slots_t, :3], _p[_slots_t, 3:7])
                        _bindings.append((_slots_t, _asset_name, _rel_p, _rel_q))
                    _groups.append((_cache[1], torch.tensor(_rows, device=device),
                                    _p[:, :3].contiguous(), _p[:, 3:7].contiguous(), _bindings))
            _count = sum(int(g[1].numel()) for g in _groups)
            if _count == 0 or _count != len(_local_obstacles):
                raise RuntimeError(f"Live world {_side}: mapped {_count}/"
                                   f"{len(_local_obstacles)} obstacles; refusing stale geometry")
            _world_reposers[_side] = _groups
            print(f"[WORLD] {_side}: live transforms for {_count} obstacles", flush=True)

    if _payload_collision_l:
        # The stage's authored bounds and root transform share a frame. Bind the
        # proxy to the physics root now; Fabric can leave USD transforms stale later.
        _payload_asset = env.scene.rigid_objects[args_cli.obj_name_l]
        _payload_path = _payload_asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        _payload_path = _payload_path.replace("env_.*", "env_0")
        _bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                     UsdGeom.Tokens.proxy]).ComputeWorldBound(
                                         env.sim.stage.GetPrimAtPath(_payload_path)).ComputeAlignedBox()
        _proxy = torch.tensor(box_sphere_cover(list(_bounds.GetMin()), list(_bounds.GetMax())),
                              dtype=torch.float32, device=device)
        _, _authored_p, _authored_q = _rigid_world_roots[_payload_path]
        _payload_local_centers = math_utils.quat_rotate_inverse(
            _authored_q.expand(8, 4), _proxy[:, :3] - env_0_pos - _authored_p)
        _payload_radii = _proxy[:, 3:].clone()

    def _repose_world_to_base(side, env_idx, grasp_reach=False):
        # Updating locked joints rebuilds cuRobo's kinematics and copies its
        # configured link_spheres, replacing any dynamic payload attachment.
        # Refresh FIRST, then attach/detach the live mug for this planning call.
        _mg = motion_gen_l if side == "l" else motion_gen_r
        _cfg = l_robot_cfg if side == "l" else r_robot_cfg
        _locks = dict(_cfg["kinematics"].get("lock_joints") or {})
        _updated_locks = measured_locked_joints(
            _locks, robot.joint_names, robot.data.joint_pos[env_idx].detach().cpu().tolist())
        if _updated_locks != _locks:
            _mg.update_locked_joints(_updated_locks, _cfg)
        if _payload_collision_l and side == "l":
            if bool(new_gripper_commands_L[env_idx]) and not grasp_reach:
                _obj = _payload_asset.data
                _centers_w = math_utils.quat_apply(_obj.root_quat_w[env_idx].expand(8, 4),
                                                  _payload_local_centers) + _obj.root_pos_w[env_idx]
                _centers_ee = math_utils.quat_rotate_inverse(
                    robot.data.body_quat_w[env_idx, l_eef_idx].expand(8, 4),
                    _centers_w - robot.data.body_pos_w[env_idx, l_eef_idx])
                _attached_spheres = torch.cat((_centers_ee, _payload_radii), dim=1)
                motion_gen_l.attach_spheres_to_robot(
                    sphere_tensor=_attached_spheres, link_name=PAYLOAD_LINK)
                _live_spheres = motion_gen_l.kinematics.kinematics_config.get_link_spheres(PAYLOAD_LINK)
                if (_live_spheres.shape != _attached_spheres.shape
                        or not torch.allclose(_live_spheres, _attached_spheres, atol=1e-6, rtol=0)):
                    raise RuntimeError("Carried-object collision spheres were not retained in planning kinematics")
                print(f"[payload] env{env_idx}: verified eight carried-object spheres after locked-joint refresh; physics remains free", flush=True)
            else:
                motion_gen_l.detach_spheres_from_robot(link_name=PAYLOAD_LINK)
        if _target_collision_mode == "approach":
            # Check the OPEN-hand reach against the target too: excluding it for
            # the whole trajectory lets a finger sweep through it before closing.
            # Disable it for subsequent carrying/retreat plans, where contact is
            # intentional. The optional payload proxy above checks carried geometry
            # against the world without welding the object in the simulator.
            _mg_target = motion_gen_l if side == "l" else motion_gen_r
            _own_targets = set(_target_obstacles[side])
            for _name in set(_target_obstacles["l"] + _target_obstacles["r"]):
                _mg_target.world_coll_checker.enable_obstacle(
                    _name, enable=bool(grasp_reach and _name in _own_targets))
        if side not in _world_reposers:
            return
        _rd = env.scene.articulations["robot"].data
        _bp = (_rd.body_pos_w[env_idx, base_link_idx]
               - env.scene.env_origins[env_idx]).view(1, 3)
        _bq = _rd.body_quat_w[env_idx, base_link_idx].view(1, 4)
        for _cache, _rows, _op, _oq, _bindings in _world_reposers[side]:
            for _slots, _asset_name, _rel_p, _rel_q in _bindings:
                _asset_data = env.scene.rigid_objects[_asset_name].data
                _live_p = _asset_data.root_pos_w[env_idx] - env.scene.env_origins[env_idx]
                _live_q = _asset_data.root_quat_w[env_idx]
                _op[_slots], _oq[_slots] = math_utils.combine_frame_transforms(
                    _live_p.expand(len(_slots), 3), _live_q.expand(len(_slots), 4),
                    _rel_p, _rel_q)
            _n = _op.shape[0]
            _pb, _qb = math_utils.subtract_frame_transforms(
                _bp.expand(_n, 3), _bq.expand(_n, 4), _op, _oq)
            _cache[0, _rows, :7] = Pose(position=_pb, quaternion=_qb).inverse().get_pose_vector()
    # A planner failure can still execute a bounded Cartesian recovery from a hold plan, so the
    # FK delta path's active joint order must exist even when no cuRobo trajectory was returned.
    common_j_names_l = [
        name for name in l_j_names if name in motion_gen_l.kinematics.joint_names
    ]

    # --- Create a reusable plan configuration ---
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,
    )
    #: The same config with the finetune stage off, for one retry when -- and only when -- that
    #: stage is what failed. FINETUNE_TRAJOPT_FAIL means cuRobo DID find a trajectory and then
    #: could not polish it, so the coarse one is usable; refusing to move at all is the worse
    #: outcome. On the AI Worker retreat this was 8 of 10 failures, every episode, and my notes
    #: record the same signature on Anubis's post-grasp plans.
    plan_config_no_finetune = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=max_attempts,
        enable_finetune_trajopt=False,
        time_dilation_factor=0.5,
    )
    #: For the arm.reset retreat, which is the hardest motion in the task: from an extended,
    #: lifted, carrying pose back to C2, a tightly folded pose beside the torso.
    #:
    #: enable_graph=True IS THE POINT. Trajopt seeds straight-line initialisations, which is why
    #: it fails on a long joint-space motion between two very different configurations --
    #: measured as FINETUNE_TRAJOPT_FAIL, then TRAJOPT_FAIL once the finetune stage was removed,
    #: then TRAJOPT_FAIL again in joint space. cuRobo's graph planner exists for exactly this and
    #: is already warmed up (warmup is called with enable_graph=True); plan_config simply never
    #: switched it on, so it has been sitting there unused the whole time.
    plan_config_graph = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=1,
        max_attempts=max_attempts,
        enable_finetune_trajopt=False,
        time_dilation_factor=0.5,
    )
    print(f"cuRobo Motion Generators are ready for {args_cli.robot}.")

    new_gripper_commands_R = torch.zeros(num_envs, dtype=torch.bool, device=device)
    new_gripper_commands_L = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Per-env nav state
    nav_phase = torch.zeros(num_envs, dtype=torch.int8, device=device)  # 0,1,2,3
    #: Number of corrective right-arm reference steps at the current goal. The jog advances a
    #: persistent relative-IK target while the controller uses measured pose feedback to track it.
    arm_r_retry = torch.zeros(num_envs, dtype=torch.long, device=device)
    #: Freeze the planner's pregrasp side so its recovery jog uses the same wrist-relative side.
    arm_r_pregrasp_back_sign = torch.zeros(num_envs, dtype=torch.int8, device=device)
    #: The same, for the LEFT arm, which until now had no jog to count. See the left-arm reach
    #: block: it accepted anything within a metre and never corrected, and every handle task in
    #: this repo grasps with this arm.
    arm_l_retry = torch.zeros(num_envs, dtype=torch.long, device=device)
    #: 0=move to a distant clear point, 1=orient there, 2=move to standoff, 3=approach.
    arm_l_pregrasp_stage = torch.zeros(num_envs, dtype=torch.int8, device=device)
    #: Freeze which side of the mug the left wrist approaches from for the whole jog.
    arm_l_pregrasp_back_sign = torch.zeros(num_envs, dtype=torch.int8, device=device)
    locked_bearing = torch.zeros(num_envs, dtype=torch.float32, device=device)  # radians
    # LAST COMMANDED YAW RATE, so the next one can be slew-limited against it. Both rotation
    # phases used sign(err) * 0.463 -- full speed or nothing -- so the command STEPPED from 0 to
    # 0.463 at the start of every turn and back to 0 at the end. Those acceleration steps ring
    # the AI Worker's lift column, and the head camera 1.43 m up shows it as shake.
    prev_yaw_cmd = torch.zeros(num_envs, dtype=torch.float32, device=device)
    _YAW_W_MAX = _nav.yaw_rate_max
    _yaw_kp = _nav.yaw_kp
    _yaw_slew = _nav.yaw_slew
    _yaw_min_speed = bounded_float_env("SIMVLA_NAV_MIN_YAW_SPEED", 0.0, 0.0, 0.15)
    if _yaw_min_speed > _YAW_W_MAX:
        raise ValueError("minimum yaw speed exceeds maximum yaw rate")
    print(f"[nav] yaw rate min={_yaw_min_speed:g} max={_YAW_W_MAX:g} rad/s "
          f"slew={_yaw_slew:g} rad/s per step", flush=True)

    def _shaped_yaw(err, rows):
        """Proportional + slew-limited yaw rate. Defaults reproduce sign(err)*W_MAX exactly."""
        w = torch.clamp(_yaw_kp * err, -_YAW_W_MAX, _YAW_W_MAX)
        if _yaw_min_speed > 0:
            # A velocity-drive command can fall below static friction before a
            # tight parking gate is met. Keep a bounded minimum moving rate;
            # the phase gate still commands zero once aligned, and slew limiting
            # below still applies. No angular acceptance threshold is changed.
            w = torch.sign(err) * torch.clamp(w.abs(), min=_yaw_min_speed)
        prev = prev_yaw_cmd[rows]
        w = prev + torch.clamp(w - prev, -_yaw_slew, _yaw_slew)
        prev_yaw_cmd[rows] = w
        return w

    # LAST COMMANDED LINEAR SPEED, the translation twin of prev_yaw_cmd: the drive used to step
    # 0 -> default_speed in one control step at the rotate->translate hand-off. Defaults never
    # bind; a profile with a finite lin_slew ramps in over ~default_speed/lin_slew steps.
    prev_lin_cmd = torch.zeros(num_envs, dtype=torch.float32, device=device)
    _lin_slew = _nav.lin_slew

    def _shaped_lin(target, rows):
        """Slew-limited linear speed magnitude. Defaults reproduce the old step exactly."""
        prev = prev_lin_cmd[rows]
        v = prev + torch.clamp(target - prev, -_lin_slew, _lin_slew)
        prev_lin_cmd[rows] = v
        return v

    reset_r = torch.zeros(num_envs, 7, dtype= torch.float32, device=device)
    #: The right arm's JOINT configuration at the same moment reset_r records its pose.
    #: arm.reset means "go back to where you started", which IS a joint configuration; expressing
    #: it only as a Cartesian pose throws that away and adds an IK failure mode for nothing. C2 is
    #: a tightly folded pose near the torso and cuRobo would not plan to it from a lifted,
    #: extended, carrying pose -- FINETUNE_TRAJOPT_FAIL, then no solution at all once the
    #: finetune stage was taken out. A joint-space goal asks a strictly easier question.
    reset_r_js = env.scene.articulations["robot"].data.joint_pos[:, r_j_index].clone()
    # A pause and a place may both re-bank reset_r. Preserve the true episode-start
    # home separately so the final, open-hand reset can return home after carrying
    # without attempting to fold a loaded arm through the counter.
    episode_home_r_base = torch.zeros(num_envs, 7, dtype=torch.float32, device=device)
    episode_home_r_js = reset_r_js.clone()
    if _postrelease_final_home:
        from isaaclab_tasks.manager_based.kitchen.mdp.homes import home_for
        _home_r_xyz, _home_l_xyz = home_for(env.scene.articulations["robot"].joint_names)
        _home_r_xyz = torch.tensor(_home_r_xyz, dtype=torch.float32, device=device)
        _home_l_xyz = torch.tensor(_home_l_xyz, dtype=torch.float32, device=device)
    reset_l = torch.zeros(num_envs, 7, dtype= torch.float32, device=device)
    #: The left arm's JOINT configuration, for the same reason reset_r_js exists -- and the left
    #: arm needs it MORE, because every handle task in this repo grasps with A_l. Measured on the
    #: dishwasher (job 2113323): planning arm.reset back to a Cartesian home pose failed 69 times
    #: in 112 episodes, and 3 times in 3 on the episodes that had actually opened the door. A plan
    #: failure resets the episode before success is evaluated, so those demos were discarded.
    reset_l_js = env.scene.articulations["robot"].data.joint_pos[:, l_j_index].clone()
    episode_home_l_base = torch.zeros(num_envs, 7, dtype=torch.float32, device=device)
    episode_home_l_js = reset_l_js.clone()
    # Physical contact checks must measure displacement caused by this approach,
    # not the mug's normal settling/randomization since its authored spawn pose.
    arm_l_mug_approach_start_xy = {}
    arm_l_segment_pause_sample = {}
    arm_l_final_home_motor_envs = set()
    # TODO: timestep for each env and if timestep is bigger than 900 reset
    timestep = torch.zeros(num_envs, dtype=torch.int32, device=device)

    video_dir = Path(f"videos")
    video_dir.mkdir(parents=True, exist_ok=True)


    def _to_hwc_uint8(t):
        """IsaacLab tensor -> HWC uint8 RGB."""
        x = t.detach().cpu().numpy()
        # If CHW, move channels to last
        if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.moveaxis(x, 0, -1)
        # Drop alpha if present
        if x.shape[-1] == 4:
            x = x[..., :3]
        # [0,1] -> [0,255]
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        return x
            
    video_dir = Path(f"videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # For sub grasp detection
    eef_r_idx = robot.find_bodies("ee_link1")[0]
    eef_l_idx = robot.find_bodies("ee_link2")[0]
    grasp_checked_r = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    grasp_checked_l = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    # 0 = not validated, 1 = banked, 2 = waiting for an enabled post-grasp lift check.
    # _reset_envs clears these on every episode/reset path.
    env_sub_saved_r = torch.zeros(num_envs, dtype=torch.uint8, device=device)
    env_sub_saved_l = torch.zeros(num_envs, dtype=torch.uint8, device=device)
    
    first_reset = torch.zeros((num_envs, 1), device=device)
    drawer = torch.zeros((num_envs, 1), device=device)
    pot_first = torch.zeros((num_envs,1),device=device)
    front_frames = [[] for _ in range(num_envs)]
    # ------------------------------------------------------------------ chair -> table distance
    # SIMVLA_CHAIRTRACE: per env and per chair, the chair->table XY distance at reset, its running
    # minimum AND MAXIMUM, and where it ended, plus one summary line per chair at the end. Unset,
    # nothing below runs and the file behaves exactly as it did.
    #
    # WHY. Nothing in the GENERATION path measures the chair. simvla_eval.py has a [chair]
    # diagnostic and this file had none, so a push-chair run that scores 0/32 cannot distinguish
    # "the chair never moved" from "the chair moved but not far enough" -- the two need opposite
    # fixes (a nav/contact bug versus a push distance or success radius that is too tight), and
    # that distinction is what diagnosed the earlier pushchair eval work.
    #
    # AND FROM A RUNNING MINIMUM ALONE THERE IS A THIRD CASE IT CANNOT SEE. A chair shoved AWAY
    # from the table never lowers that minimum, so `start - closest` is 0.000 for it -- byte
    # identical to a chair nothing ever touched, which is the very distinction this exists to make.
    # That is not hypothetical here: the nav is 3-phase bang-bang (rotate on the spot, then drive
    # STRAIGHT), the derived spawn band sits between the two chairs, and the straight line to
    # chair_0's approach pose passes closer to the chair centre than the chair's own half-extent.
    # Clipping it drives it outward. So the maximum is tracked beside the minimum, `final` records
    # where the chair actually ENDED (a chair nudged in and then knocked back out is not a partial
    # success), and `net = start - final` is signed so an outward push is a negative number a grep
    # can find.
    #
    # AND THE CHAIR NUMBERS ALONE STILL CANNOT SAY WHY. "moved 0.000" is the same line for a robot
    # that drove through the chair as for one that never came near it, so the base is traced beside
    # them: its XY, its distance to each chair, its total path, and THE SCRIPT STEP IT IS ON. That
    # last one is the single most diagnostic field in here -- a step index stuck at 0 means the base
    # never reached the first approach pose, and nothing about the chairs can tell you that.
    # Measured on kitchen 1216: spawn band 0 is x[0.98,1.52] y[-4.07,-3.62], chair_0's AABB is
    # x[-0.230,+0.246] y[-4.315,-3.840], and the straight line from the band centre to chair_0's
    # approach pose at (-0.546,-4.563) passes y=-4.249 at x=0.246 -- inside that box. The nav is
    # rotate-then-drive-straight with no obstacle avoidance and the room shell's walls are
    # use_collision_geometry=False, so there is nothing to stop the base shoving the chair outward.
    #
    # AND THE BASE'S OWN XY STILL CANNOT SAY WHY IT STOPPED. A base parked short of its goal looks
    # the same whether it is still rotating to face it (phase 0 adds no XY path at all, so `path`
    # cannot tell a spin from a stall), driving on a stale locked bearing, ramped down to nothing,
    # or physically blocked. So each periodic line also carries the phase, the goal it is driving
    # at, and COMMANDED versus ACHIEVED base motion -- commanded non-zero with achieved ~0 is
    # blocked, commanded ~0 is the phase machine having given up. That pair is the same [base]
    # diagnostic that settled the earlier pushchair eval work.
    #
    # THE PHASES, from the N_s block this task's four steps all take:
    #   0  rotate on the spot to face the goal, re-locking locked_bearing every tick
    #   1  drive straight, vx only, until dist <= 0.2
    #   2  close in on (vx, vy) until dist <= goal_reached_distance (nav_tuning: 0.02 for anubis)
    #   3  rotate to the goal's own yaw, then the step is finished
    #
    # IT READS WHAT THE SUCCESS TERM READS. mdp.composed.build_context takes a rigid role from
    # env.scene.rigid_objects[name].data.body_pos_w.squeeze(1) -- the rigid-body ROOT -- and an
    # articulation role from body 0; predicates_math._pos/_anchor_pos then compare exactly those.
    # Three different notions of "centre" are already in play in this pipeline (the planner's bbox
    # centre, this rigid-body root, the recipe's scene-graph transform), so a trace that measured a
    # different one would mislead the very diagnosis it exists to support. Hence _ct_xy below is
    # build_context's lookup, in the same order (rigid object first, articulation second).
    #
    # Value: "1", or any value with no ':', auto-detects every rigid object whose name starts with
    # "chair" and measures to "table". "chair_0,chair_1:table" names them explicitly. Scene-entity
    # names, not prim paths -- that is what composed resolves against.
    _ct_raw = os.environ.get("SIMVLA_CHAIRTRACE", "").strip()
    _ct_names, _ct_target = [], "table"
    #: Per chair, all (num_envs,) and all about the CURRENT episode except _ct_best/_ct_worst,
    #: which are that env's largest approach and largest push-away over every episode it has run.
    _ct_start, _ct_min, _ct_max, _ct_last, _ct_best, _ct_worst = {}, {}, {}, {}, {}, {}
    #: TIP OR JAM. Everything above this line measures the chair in PLAN VIEW, and a chair that
    #: stopped moving reads identically whether it toppled over or wedged against something -- the
    #: XY distance simply stops changing. Those need opposite fixes (a lower push height versus a
    #: geometry or contact problem), which is exactly the distinction the min/max pair was added to
    #: make one level up, and it is missing here.
    #:
    #: Two numbers settle it and neither is inferable from XY. `tiltmax` is the largest angle the
    #: chair's own +z has made with the world's, over the episode; `zmin` is the lowest its origin
    #: has been. A TOPPLE drives tilt toward 90 degrees and drops z by most of a chair height. A
    #: JAM leaves both flat -- tilt near 0, z unchanged -- while the XY distance freezes just the
    #: same. Running extremes rather than instantaneous values, because a chair that tipped and
    #: then settled back would otherwise show nothing by the time the line is printed.
    _ct_tiltmax, _ct_zmin = {}, {}
    _ct_frames, _ct_eps = 0, None
    #: The base: body index (resolved once, only when armed), its previous XY, and the path length
    #: it has walked. Reset teleports the base to its spawn, and that jump is not travel -- see the
    #: `just_reset` guard in the tick.
    _ct_base_i, _ct_bprev, _ct_bpath = None, None, None
    #: Commanded base motion, summed over the frames since the last periodic line, so the line can
    #: report a MEAN rather than one noisy sample. _ct_mark is the path length at that same moment,
    #: which turns into the achieved speed over the identical window -- the two are only comparable
    #: because they cover the same frames.
    _ct_cmd_lin, _ct_cmd_ang, _ct_cmd_n, _ct_mark = None, None, 0, None
    #: Frames between the in-run progress lines. The reset lines and the end-of-run summary are not
    #: throttled; only this periodic one is, because it fires every frame otherwise. It carries the
    #: maximum as well as the minimum BECAUSE a `timeout` kill never reaches the summary -- SIGTERM
    #: does not run python's shutdown path -- so these lines are the only surviving record of a run
    #: that was cut off, and a push-away has to be visible in them.
    _CT_EVERY = 100
    #: Metres below which a difference is settling noise rather than motion, used only to decide
    #: which words the summary uses. A rigid body dropped onto the floor at reset jitters well
    #: under this; a push is two orders of magnitude above it.
    _CT_EPS_M = 0.005

    def _ct_xy(name):
        """`name`'s world XY, from the SAME source mdp.composed.build_context uses."""
        if name in env.scene.rigid_objects:
            return env.scene.rigid_objects[name].data.body_pos_w.squeeze(1)[:, :2]
        return env.scene.articulations[name].data.body_pos_w[:, 0, :2]

    def _ct_tilt_deg_and_z(name):
        """(tilt of `name`'s +z from vertical in degrees, `name`'s world z), per env.

        The tilt is read straight off the body quaternion: for (w, x, y, z) the third column of the
        rotation is the body's +z in world, and its k component is 1 - 2(x^2 + y^2). Taking the
        angle from that one term needs no matrix and no scipy, and it is exactly acos of the dot
        product with the world's +z.

        Same source as _ct_xy, deliberately: a tilt measured off a different body than the distance
        would let the two disagree about which chair they are describing.
        """
        if name in env.scene.rigid_objects:
            _q = env.scene.rigid_objects[name].data.body_quat_w.squeeze(1)
            _z = env.scene.rigid_objects[name].data.body_pos_w.squeeze(1)[:, 2]
        else:
            _q = env.scene.articulations[name].data.body_quat_w[:, 0]
            _z = env.scene.articulations[name].data.body_pos_w[:, 0, 2]
        _up = 1.0 - 2.0 * (_q[:, 1] ** 2 + _q[:, 2] ** 2)
        return torch.rad2deg(torch.acos(_up.clamp(-1.0, 1.0))), _z

    def _ct_yaw_deg(name):
        """`name`'s world yaw in degrees, per env.

        WHERE THE CHAIR ACTUALLY FACES AT RUN TIME. It matches the authored USD prim: chair_0's
        /world/chair_0 carries xformOp:orient (0.9314151, 0, 0, 0.36395872) = +42.687 deg and the
        runtime reads +42.69; chair_1's is identity and the runtime reads 0.00. env_cfg_emit copies
        that prim transform into init_state.rot verbatim (w,x,y,z both ends) and IsaacLab writes it
        as an absolute root pose at reset, so authored and runtime cannot drift apart. Nothing is
        applied on top of anything.

        DO NOT COMPARE THIS AGAINST THE PUSH HEADING. This docstring used to say "compare this
        against the authored -47.31 / -90.00", and someone did: those two numbers are the chair->
        TABLE BEARINGS (push_prim_base_pose's returned base heading), not authored chair yaws, and
        the difference is a constant +90.00 on both chairs. That was read as "both chairs are a
        quarter turn from where the planner believes" and cost a ruling and a GPU round. The +90 is
        -angle(facing_direction(chair mesh)): face_chair_toward aims the chair's measured FRONT at
        the table, this asset's front is its local -Y (its backrest sits at local +Y, offset
        +0.1204 m), so pose_yaw = bearing + 90 BY CONSTRUCTION. Two chairs of one uid show it twice.
        Measured on the shipped USD, each chair's backrest points 132.687 / 90.000 deg while the base
        stands off along 132.689 / 90.002 -- square-on to within 0.002 deg. See
        test_chair_orientation.py.

        WHAT THIS FIELD IS ACTUALLY FOR: watching the yaw MOVE. A chair the approach grazes or the
        retreat hooks turns, and nothing else in the trace says so -- tilt cannot, since a chair can
        face anywhere while standing perfectly upright. `_ct_front_offset_deg` beside it is the
        quantity that is supposed to be constant.
        """
        if name in env.scene.rigid_objects:
            _q = env.scene.rigid_objects[name].data.body_quat_w.squeeze(1)
        else:
            _q = env.scene.articulations[name].data.body_quat_w[:, 0]
        # yaw of (w,x,y,z): atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        return torch.rad2deg(torch.atan2(
            2.0 * (_q[:, 0] * _q[:, 3] + _q[:, 1] * _q[:, 2]),
            1.0 - 2.0 * (_q[:, 2] ** 2 + _q[:, 3] ** 2)))

    def _ct_bearing_deg(name):
        """The bearing from `name` to the trace's target, in degrees, per env.

        THE OTHER HALF OF THE COMPARISON, printed beside the yaw so the two can never again be read
        from different places and assumed to be the same kind of number. This is the push heading:
        push_prim_base_pose returns exactly atan2(target - prim) as the base yaw, and the base stands
        off on the opposite ray.

        Frame-free by construction -- both XYs come from _ct_xy, so a per-env origin cancels in the
        difference and the bearing is the same in world and kitchen frames.
        """
        _d = _ct_xy(_ct_target) - _ct_xy(name)
        return torch.rad2deg(torch.atan2(_d[:, 1], _d[:, 0]))

    def _ct_front_offset_deg(name):
        """yaw - bearing, wrapped to (-180, 180]: THE NUMBER THAT IS SUPPOSED TO BE CONSTANT.

        It is -angle(facing_direction(mesh)) for the chair's own asset -- +90.00 for the two chairs
        of kitchen 1218, whose backrest is at local +Y. A defect looks like this number DIFFERING
        between two chairs of the same uid, or CHANGING within an episode (the chair got turned).
        A steady +90 on every chair of one asset is the asset's front axis and is not news.
        """
        _d = _ct_yaw_deg(name) - _ct_bearing_deg(name)
        return (_d + 180.0) % 360.0 - 180.0

    def _ct_base_xy():
        """The base's XY in the KITCHEN frame -- world minus this env's origin.

        THE SAME READ THE NAV ITSELF DOES (`body_pos_w[:, base_link_idx, :2] - env_origins[:, :2]`),
        and the same frame the goal file's N_s poses are written in, so a printed base XY can be
        compared with the approach pose in the goal file by eye and the comparison means something.
        """
        _r = env.scene.articulations["robot"]
        return _r.data.body_pos_w[:, _ct_base_i, :2] - env.scene.env_origins[:, :2]

    def _ct_base_yaw():
        """The base's yaw, by the SAME formula the nav uses on the same body quaternion."""
        _q = env.scene.articulations["robot"].data.body_quat_w[:, _ct_base_i]
        _w, _x, _y, _z = _q.unbind(-1)
        return torch.atan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z))

    def _ct_wrap(a):
        """An angle wrapped to [-pi, pi], as the nav's own wrap_pi does it."""
        return (a + torch.pi) % (2 * torch.pi) - torch.pi

    def _ct_sgn(v):
        """`v` as a signed 3-dp string. The `+ 0.0` matters: -0.0 formats as "-0.000", and a
        spurious minus here is noise in exactly the grep the sign exists for."""
        return f"{float(v) + 0.0:+.3f}"

    if _ct_raw:
        _ct_lhs, _, _ct_rhs = _ct_raw.partition(":")
        _ct_target = _ct_rhs.strip() or "table"
        _ct_names = [n.strip() for n in _ct_lhs.split(",") if n.strip() and n.strip() != "1"]
        if not _ct_names:
            _ct_names = sorted(n for n in env.scene.rigid_objects if n.startswith("chair"))
        _ct_have = sorted(set(env.scene.rigid_objects) | set(env.scene.articulations))
        # Loud, not silent: an operator who asked for the trace and got nothing back would read the
        # empty log as "the chair never moved", which is one of the answers this exists to tell
        # apart. Refuse before the run instead of lying about it afterwards.
        _ct_missing = [n for n in _ct_names + [_ct_target] if n not in _ct_have]
        if not _ct_names or _ct_missing:
            raise SystemExit(
                f"SIMVLA_CHAIRTRACE={_ct_raw!r}: "
                + (f"no scene entity named {_ct_missing}" if _ct_missing
                   else "no rigid object whose name starts with 'chair'")
                + f". This scene has {_ct_have}."
            )
        _ct_base_i = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
        print(f"[chair] tracing {_ct_names} -> {_ct_target} (rigid-body roots, XY, the same "
              f"positions mdp.composed reads) + the base, in the kitchen frame", flush=True)

    def _ct_verdict(start, closest, farthest, final):
        """What happened to ONE chair in ONE env, in words -- or "" for a plain approach.

        Ordered most-alarming-first, and every branch is a case a bare `start - closest` reports
        as the same number as the case above it.
        """
        net = start - final
        out = farthest - start          # how far past its start it was ever driven
        if net < -_CT_EPS_M:
            return "PUSHED AWAY"        # it ended FARTHER from the table than it began
        if out > _CT_EPS_M:
            return "WENT OUT AND CAME BACK"
        if net > _CT_EPS_M and final > closest + _CT_EPS_M:
            return "APPROACHED THEN FELL BACK"
        if (start - closest) <= _CT_EPS_M and out <= _CT_EPS_M:
            return "NEVER MOVED"
        return ""

    def _chairtrace_tick(just_reset):
        """One frame. `just_reset` is a bool mask of the envs whose episode restarted this frame."""
        nonlocal _ct_frames, _ct_eps, _ct_bprev, _ct_bpath
        nonlocal _ct_cmd_lin, _ct_cmd_ang, _ct_cmd_n, _ct_mark
        _ct_frames += 1
        _tgt = _ct_xy(_ct_target)
        _org = env.scene.env_origins[:, :2]
        _b = _ct_base_xy()
        if _ct_bprev is None:
            _ct_bprev = _b.clone()
            _ct_bpath = torch.zeros(num_envs, device=_b.device)
        else:
            # A reset TELEPORTS the base to its spawn band. Counting that jump would add metres of
            # "path" the robot never drove, and on a storming run it would dominate the number.
            _hop = torch.linalg.norm(_b - _ct_bprev, dim=-1)
            _ct_bpath = _ct_bpath + torch.where(just_reset, torch.zeros_like(_hop), _hop)
            _ct_bprev = _b.clone()
        if _ct_mark is None:
            _ct_mark = _ct_bpath.clone()
            _ct_cmd_lin = torch.zeros(num_envs, device=_b.device)
            _ct_cmd_ang = torch.zeros(num_envs, device=_b.device)
        # THE COMMAND AS IT WAS LAST APPLIED. delta_pose_base is rebound to zeros a few lines below
        # this call, so at tick time it still holds the row pre_process_actions handed to env.step()
        # on the PREVIOUS frame -- which is the command whose effect the achieved speed above has
        # just measured. It does not exist at all on the very first tick, before the loop has built
        # one; observation only, so a missing sample is skipped rather than faked.
        try:
            _cmd = delta_pose_base
        except NameError:
            _cmd = None
        if _cmd is not None:
            _ct_cmd_lin = _ct_cmd_lin + torch.linalg.norm(_cmd[:, :2], dim=-1)
            _ct_cmd_ang = _ct_cmd_ang + _cmd[:, 2]
            _ct_cmd_n += 1
        _now = {}
        for _n in _ct_names:
            _d = torch.linalg.norm(_ct_xy(_n) - _tgt, dim=-1)
            _now[_n] = _d
            _tilt, _zz = _ct_tilt_deg_and_z(_n)
            if _ct_start.get(_n) is None:
                _ct_start[_n] = _d.clone()
                _ct_min[_n] = _d.clone()
                _ct_max[_n] = _d.clone()
                _ct_last[_n] = _d.clone()
                _ct_best[_n] = torch.zeros_like(_d)
                _ct_worst[_n] = torch.zeros_like(_d)
                _ct_tiltmax[_n] = _tilt.clone()
                _ct_zmin[_n] = _zz.clone()
                continue
            # Fold the episode that is ending into the all-time best AND worst BEFORE the reset
            # overwrites its start. Applying the max to every env (not just the resetting ones) is
            # deliberate: for a running env `start - min` is its approach so far and `max - start`
            # its push-away so far, and a running maximum of each is the same number, so neither
            # needs reset gating to stay correct.
            _ct_best[_n] = torch.maximum(_ct_best[_n], _ct_start[_n] - _ct_min[_n])
            _ct_worst[_n] = torch.maximum(_ct_worst[_n], _ct_max[_n] - _ct_start[_n])
            _ct_start[_n] = torch.where(just_reset, _d, _ct_start[_n])
            _ct_min[_n] = torch.where(just_reset, _d, torch.minimum(_ct_min[_n], _d))
            _ct_max[_n] = torch.where(just_reset, _d, torch.maximum(_ct_max[_n], _d))
            _ct_last[_n] = _d.clone()
            # RESET-GATED, unlike best/worst above. Those are all-time extremes on purpose; these
            # two describe ONE episode, because "did this push topple the chair" is a question about
            # the episode in front of you. A chair left on its side by an earlier episode would
            # otherwise pin tiltmax at 90 for the rest of the run and say nothing about any of them.
            _ct_tiltmax[_n] = torch.where(just_reset, _tilt, torch.maximum(_ct_tiltmax[_n], _tilt))
            _ct_zmin[_n] = torch.where(just_reset, _zz, torch.minimum(_ct_zmin[_n], _zz))
        if _ct_eps is None:
            _ct_eps = torch.ones(num_envs, dtype=torch.int32, device=_tgt.device)
        elif bool(just_reset.any()):
            _ct_eps = _ct_eps + just_reset.to(torch.int32)
        if bool(just_reset.any()):
            _ct_which = torch.nonzero(just_reset).flatten().tolist()
            print(f"[chair] t={_ct_frames} reset env={_ct_which} start "
                  + "  ".join(f"{_n}=" + ",".join(f"{float(_now[_n][_e]):.3f}" for _e in _ct_which)
                              for _n in _ct_names)
                  + " | base " + " ".join(f"{_e}:({float(_b[_e][0]):.3f},{float(_b[_e][1]):.3f})"
                                          for _e in _ct_which), flush=True)
        if _ct_frames % _CT_EVERY == 0:
            for _n in _ct_names:
                _ap = torch.maximum(_ct_best[_n], _ct_start[_n] - _ct_min[_n])
                _aw = torch.maximum(_ct_worst[_n], _ct_max[_n] - _ct_start[_n])
                _net = _ct_start[_n] - _ct_last[_n]
                _bi, _wi = int(torch.argmax(_ap)), int(torch.argmax(_aw))
                print(f"[chair] t={_ct_frames} {_n}: closest {float(_ct_min[_n].min()):.3f} "
                      f"farthest {float(_ct_max[_n].max()):.3f} | best approach "
                      f"{_ct_sgn(_ap[_bi])} (env {_bi}) worst push-away "
                      f"{_ct_sgn(-_aw[_wi])} (env {_wi}) | mean net {_ct_sgn(_net.mean())} "
                      f"| envs ending outward {int((_net < -_CT_EPS_M).sum())}"
                      f" | tilt max {float(_ct_tiltmax[_n].max()):.1f}deg"
                      f" (env {int(torch.argmax(_ct_tiltmax[_n]))})"
                      f" z min {float(_ct_zmin[_n].min()):.3f}"
                      # BOTH ANGLES, TOGETHER, WITH THE DELTA NAMED. Printing the yaw alone once
                      # cost a ruling: a reader compared it against the push heading quoted in a
                      # docstring, found a constant 90 deg, and concluded every chair was a quarter
                      # turn from where the planner believed. They are different quantities and the
                      # delta is the chair asset's own front axis. Side by side with the offset
                      # labelled, that misreading is not available. See _ct_front_offset_deg.
                      f" yaw {float(_ct_yaw_deg(_n)[0]):.2f}deg"
                      f" (push heading {float(_ct_bearing_deg(_n)[0]):.2f}deg,"
                      f" front-axis offset {float(_ct_front_offset_deg(_n)[0]):.2f}deg)",
                      flush=True)
            # PER ENV, because the per-chair aggregate cannot separate "drove through the chair"
            # from "never got near it". The step index is the one to read first: env_goal_indices
            # stuck at 0 means the base never reached the first approach pose at all, and no chair
            # number says that.
            _loc = {_n: _ct_xy(_n) - _org for _n in _ct_names}
            _yaw = _ct_base_yaw()
            # Achieved speed over the SAME window the commanded means cover. env.step_dt is the
            # env's own step size; 1/20 is this pipeline's rate everywhere and is only the fallback.
            _dt = float(getattr(env, "step_dt", 0.0) or 0.05)
            _win = max(_ct_cmd_n, 1)
            _ach = (_ct_bpath - _ct_mark) / (_win * _dt)
            _nsteps = payloads_tensor.shape[1]
            for _e in range(num_envs):
                # The goal the CURRENT step is driving at, read out of the same payloads_tensor row
                # the nav reads, so the log is self-contained. Clamped: an env that has run off the
                # end of its script has env_goal_indices == len(script), which would index out of
                # range -- that env is finished, not navigating, and is marked so.
                _j = int(env_goal_indices[_e])
                _done = _j >= _nsteps
                _j = min(_j, _nsteps - 1)
                _isnav = int(task_ids_tensor[_e, _j]) == NS_ID and not _done
                if _isnav:
                    _g = payloads_tensor[_e, _j, :3]
                    _dg = float(torch.linalg.norm(_g[:2] - _b[_e]))
                    _gs = (f"goal({float(_g[0]):.3f},{float(_g[1]):.3f},{float(_g[2]):+.3f}) "
                           f"d={_dg:.3f} yawerr={_ct_sgn(_ct_wrap(_g[2] - _yaw[_e]))}")
                else:
                    _gs = "goal(script done)" if _done else "goal(not an N_s step)"
                print(f"[chair] t={_ct_frames} env {_e}: base"
                      f"({float(_b[_e][0]):.3f},{float(_b[_e][1]):.3f}) yaw {_ct_sgn(_yaw[_e])} "
                      f"step {int(env_goal_indices[_e])} phase {int(nav_phase[_e])} "
                      f"path {float(_ct_bpath[_e]):.2f}m {_gs} "
                      f"cmd lin={float(_ct_cmd_lin[_e]) / _win:.4f} "
                      f"ang={_ct_sgn(_ct_cmd_ang[_e] / _win)} ach={float(_ach[_e]):.4f}m/s "
                      + " ".join(f"d({_n})={float(torch.linalg.norm(_loc[_n][_e] - _b[_e])):.3f}"
                                 for _n in _ct_names), flush=True)
            _ct_mark = _ct_bpath.clone()
            _ct_cmd_lin = torch.zeros_like(_ct_cmd_lin)
            _ct_cmd_ang = torch.zeros_like(_ct_cmd_ang)
            _ct_cmd_n = 0

    def _chairtrace_summary():
        if not _ct_names or _ct_start.get(_ct_names[0]) is None:
            return
        for _n in _ct_names:
            _ap = torch.maximum(_ct_best[_n], _ct_start[_n] - _ct_min[_n])
            _aw = torch.maximum(_ct_worst[_n], _ct_max[_n] - _ct_start[_n])
            _net = _ct_start[_n] - _ct_last[_n]
            for _e in range(num_envs):
                # start/closest/farthest/final are THIS env's CURRENT episode. `net` is signed on
                # purpose: negative means the chair ENDED farther out than it started, which a
                # running minimum reports as a flat 0.000 and cannot be told from untouched.
                _s = float(_ct_start[_n][_e]); _c = float(_ct_min[_n][_e])
                _f = float(_ct_max[_n][_e]); _fi = float(_ct_last[_n][_e])
                _v = _ct_verdict(_s, _c, _f, _fi)
                _ep = int(_ct_eps[_e])
                print(f"[chair] {_n} env {_e}: start {_s:.3f} closest {_c:.3f} farthest {_f:.3f} "
                      f"final {_fi:.3f} net {_ct_sgn(_s - _fi)}"
                      + (f" [ep {_ep}: best approach {_ct_sgn(_ap[_e])}, worst push-away "
                         f"{_ct_sgn(-_aw[_e])}]" if _ep > 1 else "")
                      + (f" {_v}" if _v else ""), flush=True)
            _bi, _wi = int(torch.argmax(_ap)), int(torch.argmax(_aw))
            print(f"[chair] {_n} over {num_envs} env(s) / {_ct_frames} frames: best approach "
                  f"{_ct_sgn(_ap[_bi])} (env {_bi}), worst push-away {_ct_sgn(-_aw[_wi])} "
                  f"(env {_wi}), closest ever {float(_ct_min[_n].min()):.3f}, farthest ever "
                  f"{float(_ct_max[_n].max()):.3f}, envs ending outward "
                  f"{int((_net < -_CT_EPS_M).sum())}/{num_envs}", flush=True)
        if _ct_bpath is not None:
            _b = _ct_base_xy()
            for _e in range(num_envs):
                # `path` excludes the teleport at every reset, so it is metres actually driven. A
                # large path with a step index of 0 is a base that drove and never arrived; a path
                # near zero is one that never moved at all.
                print(f"[chair] base env {_e}: path {float(_ct_bpath[_e]):.2f} m, final xy "
                      f"({float(_b[_e][0]):.3f},{float(_b[_e][1]):.3f}), step "
                      f"{int(env_goal_indices[_e])}, episodes {int(_ct_eps[_e])}", flush=True)

    initial_objects_by_env = {}
    global_frames = 0
    _loaded_home_motor_overrides = {}
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            if max_frames is not None and global_frames >= max_frames:
                print(f"[run] reached SIMVLA_MAXFRAMES={max_frames}; finalizing", flush=True)
                break
            global_frames += 1
            # --- Initialize action tensors for all environments ---
            robot = env.scene.articulations["robot"]
            timestep+=1
            if _postrelease_final_home:
                for _home_env in torch.where(timestep == 1)[0].tolist():
                    _home_pos_b, _home_quat_b = world2base(
                        env, robot.data.body_pos_w[_home_env, r_eef_idx],
                        robot.data.body_quat_w[_home_env, r_eef_idx], _home_env)
                    episode_home_r_base[_home_env, :3] = _home_r_xyz
                    episode_home_r_base[_home_env, 3:7] = _home_quat_b
                    episode_home_r_js[_home_env] = robot.data.joint_pos[_home_env, r_j_index]
                    _home_pos_l_b, _home_quat_l_b = world2base(
                        env, robot.data.body_pos_w[_home_env, l_eef_idx],
                        robot.data.body_quat_w[_home_env, l_eef_idx], _home_env)
                    episode_home_l_base[_home_env, :3] = _home_l_xyz
                    episode_home_l_base[_home_env, 3:7] = _home_quat_l_b
                    episode_home_l_js[_home_env] = robot.data.joint_pos[_home_env, l_j_index]
                    print(f"[home-snapshot] env{_home_env} measured={_home_pos_b.tolist()} "
                          f"target={_home_r_xyz.tolist()} left_measured={_home_pos_l_b.tolist()} "
                          f"left_target={_home_l_xyz.tolist()}", flush=True)
            # Reset the lift state explicitly at the start of each episode. A moving jaw during
            # an already-armed lift is contact, not evidence that the episode was reset.
            env_lift_r[timestep == 1] = 0
            for _new_env in torch.where(timestep == 1)[0].tolist():
                arm_l_mug_approach_start_xy.pop(_new_env, None)
                arm_l_final_home_motor_envs.discard(_new_env)
            env_lift_l[timestep == 1] = 0
            env_r_gripper_ticks[timestep == 1] = 0
            env_l_gripper_ticks[timestep == 1] = 0

            if _ct_names:
                # timestep is zeroed by _reset_envs and incremented once per frame just above, so
                # ==1 is "this env has taken no step since its reset" -- true on the first frame of
                # the run and on the first frame of every episode after it.
                _chairtrace_tick(timestep == 1)
            pose_L = torch.zeros((num_envs, 6), device=device)
            pose_R = torch.zeros((num_envs, 6), device=device)
            delta_pose_base = torch.zeros((num_envs, 3), device=device)
            # --- Create a mask for environments that are still active (not finished all goals) ---
            variable.env_goal_indices = env_goal_indices
            active_envs_mask = torch.logical_not(env_goal_indices >= max_sequence_length)

            # Retry
            # 1. Failed but finished
            fail_done_idx = torch.where(~active_envs_mask)[0]
            if fail_done_idx.numel() == 0:
                pass
            else:
                # A finished command list is not a successful physical task. Preserve the
                # terminal scene state before reset so a failed drawer/sink predicate can be
                # diagnosed without weakening it or mistaking goal exhaustion for a demo.
                for _e in fail_done_idx.tolist():
                    _diag = [f"[fail_done] env{_e} robot={args_cli.robot}"]
                    for _name in (args_cli.obj_name, args_cli.obj_name_l):
                        if _name != "none" and _name in env.scene.rigid_objects:
                            _p = (env.scene.rigid_objects[_name].data.body_pos_w[_e, 0]
                                  - env.scene.env_origins[_e])
                            _diag.append(f"{_name}_xyz=({float(_p[0]):+.4f},"
                                         f"{float(_p[1]):+.4f},{float(_p[2]):+.4f})")
                    if "base_cabinet" in env.scene.articulations:
                        _drawer_q = float(env.scene.articulations["base_cabinet"]
                                          .data.joint_pos[_e, 0])
                        _diag.append(f"drawer_joint={_drawer_q:+.4f}")
                    print(" ".join(_diag), flush=True)
                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, fail_done_idx, reason="fail_but_done") 
            # 2. Time out
            # `episode_steps`, not a literal: read once at the top of main() from
            # SIMVLA_EPISODE_STEPS and printed there, so a run's own log says what horizon its
            # timeouts were measured against. Default 1500, unchanged.
            timeout_mask = timestep > episode_steps
            timeout_idx = torch.where(timeout_mask)[0] 

            if torch.any(timeout_mask):
                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, timeout_idx, reason="timeout") 

            retry_mask = env.termination_manager.get_term("retry").clone()
            retry_idx = torch.where(retry_mask)[0] 
            if torch.any(retry_mask):
                # `retry` is task-defined. It may mean a dropped object or fallen robot, and is
                # not necessarily a navigation out-of-bounds condition.
                try:
                    from retry_diagnostics import describe_retry_union
                    _retry_spec = env_cfg.terminations.retry.params["spec"]
                    _object_z = {}
                    for _leaf in _retry_spec.get("any", []):
                        _params = _leaf.get("obj_z") if isinstance(_leaf, dict) else None
                        if _params:
                            _role = _params["role"]
                            _object_z[_role] = (
                                env.scene.rigid_objects[_role].data.body_pos_w[:, 0, 2]
                                .detach().cpu().tolist()
                            )
                    _robot = env.scene.articulations["robot"]
                    _base_idx = _robot.find_bodies("base_link")[0][0]
                    _base_z = _robot.data.body_pos_w[:, _base_idx, 2].detach().cpu().tolist()
                    for _line in describe_retry_union(
                        _retry_spec, retry_idx.detach().cpu().tolist(),
                        object_z=_object_z, base_z=_base_z,
                    ):
                        print(f"[retry] {_line}", flush=True)
                except Exception as _exc:
                    print(f"[retry] diagnostic unavailable: {type(_exc).__name__}: {_exc}",
                          flush=True)
                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, retry_idx, reason="retry_termination")
            # Filter global indices to get only active environments
            finished_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            post_lift_failed_envs = set()
            env_indices = torch.arange(num_envs, device=device)
            active_env_indices = env_indices[active_envs_mask]
            current_task_ids = task_ids_tensor[active_env_indices, env_goal_indices[active_envs_mask]]
            # --- Create boolean masks for each task type on the ACTIVE environments ---
            nav_mask = current_task_ids == N_ID
            arm_l_mask = current_task_ids == AL_ID
            arm_r_mask = current_task_ids == AR_ID
            arm_b_mask = current_task_ids == AB_ID
            grip_l_mask = current_task_ids == GL_ID
            grip_r_mask = current_task_ids == GR_ID
            nav_s_mask = current_task_ids == NS_ID

            def _mug_pad_sensor_names(arm):
                """Sensor pair for this embodiment/arm, or None without mug-pad instrumentation."""
                if args_cli.robot == "rby1":
                    return (f"touch_mug_{arm}_pad1", f"touch_mug_{arm}_pad2")
                if args_cli.robot == "aiworker" and arm == "l":
                    return ("touch_mug_l_r2", "touch_mug_l_l2")
                return None

            def _mug_pad_contact_forces(env_id, arm, *, proximal=False):
                """Return actual two-pad loads, None when this task has no mug-pad sensors."""
                sensors = getattr(env.scene, "sensors", {})
                names = _mug_pad_sensor_names(arm)
                if proximal:
                    names = (f"touch_mug_{arm}_r1", f"touch_mug_{arm}_l1") if args_cli.robot == "aiworker" else None
                if names is None:
                    return None
                if not any(sensors.get(name) is not None for name in names):
                    # This embodiment/task path requires pad measurements for mug acceptance.
                    # Missing instrumentation must not silently downgrade to root-relative
                    # geometry and thereby authorize a grasp we could not physically verify.
                    return (0.0, 0.0)
                values = []
                for _name in names:
                    try:
                        _sensor = sensors.get(_name)
                        if _sensor is None:
                            values.append(0.0)
                            continue
                        _forces = getattr(_sensor.data, "force_matrix_w", None)
                        if _forces is None:
                            values.append(0.0)
                            continue
                        _vectors = _forces[env_id].reshape(-1, 3)
                        _peak = torch.linalg.vector_norm(_vectors, dim=-1).max()
                        values.append(float(_peak) if bool(torch.isfinite(_peak)) else 0.0)
                    except Exception:  # unavailable measurements must not authorize a grasp
                        values.append(0.0)
                return tuple(values)

            def _mug_finger_contact_forces(env_id, arm):
                """Measure both links of each adaptive finger, or the parallel pad pair."""
                distal = _mug_pad_contact_forces(env_id, arm)
                proximal = _mug_pad_contact_forces(env_id, arm, proximal=True)
                if distal is None or proximal is None:
                    return distal
                return tuple(a + b for a, b in zip(distal, proximal))

            def _mug_pad_contact_diag(env_id, arm):
                """Format measured normal-contact magnitudes for collector telemetry."""
                values = _mug_finger_contact_forces(env_id, arm)
                if values is None:
                    return "not-configured"
                names = _mug_pad_sensor_names(arm)
                suffix = "+proximal" if args_cli.robot == "aiworker" else ""
                return f"{names[0]}{suffix}={values[0]:.3f}N, {names[1]}{suffix}={values[1]:.3f}N"

            def _trace_lift(arm, env_ids, remaining):
                """Opt-in measured contact telemetry; never changes acceptance/control."""
                if os.environ.get("SIMVLA_TRACE_LIFT", "0") != "1":
                    return
                robot = env.scene.articulations["robot"]
                object_name = args_cli.obj_name if arm == "r" else args_cli.obj_name_l
                obj = env.scene.rigid_objects[object_name]
                bodies = list(jaw_body_indices(robot, "right" if arm == "r" else "left"))
                joints = [i for i, name in enumerate(robot.joint_names)
                          if "gripper" in name and (f"_{arm}_" in name or f"_{arm}1" in name
                                                    or f"_{arm}2" in name)]
                for env_id, ticks in zip(env_ids.tolist(), remaining.tolist()):
                    if ticks % 10:
                        continue
                    record = {
                        "arm": arm, "env_id": env_id, "remaining_ticks": ticks,
                        "object": object_name,
                        "object_mass_kg": obj.root_physx_view.get_masses()[env_id].tolist(),
                        "object_material": obj.root_physx_view.get_material_properties()[env_id].tolist(),
                        "object_pos_w": obj.data.body_pos_w[env_id, 0].tolist(),
                        "jaw_body_names": [robot.body_names[i] for i in bodies],
                        "jaw_pos_w": robot.data.body_pos_w[env_id, bodies].tolist(),
                        "jaw_quat_wxyz": robot.data.body_quat_w[env_id, bodies].tolist(),
                        "joint_names": [robot.joint_names[i] for i in joints],
                        "joint_pos": robot.data.joint_pos[env_id, joints].tolist(),
                        "joint_targets": robot.data.joint_pos_target[env_id, joints].tolist(),
                        "pad_load_n": _mug_pad_contact_forces(env_id, arm),
                        "proximal_load_n": _mug_pad_contact_forces(env_id, arm, proximal=True),
                    }
                    print("[lift-trace] " + json.dumps(record, allow_nan=False), flush=True)

            def _verify_subgrasp(arm, sub_envs):
                """Check a settled close before the close handler begins any optional lift."""
                if sub_envs.numel() == 0:
                    return
                is_right = arm == "r"
                eef_idx = eef_r_idx if is_right else eef_l_idx
                obj_name = args_cli.obj_name if is_right else args_cli.obj_name_l
                checked = grasp_checked_r if is_right else grasp_checked_l
                saved = env_sub_saved_r if is_right else env_sub_saved_l
                eef_pos = env.scene.articulations["robot"].data.body_pos_w[sub_envs, eef_idx]
                obj_pos = env.scene.rigid_objects[obj_name].data.body_pos_w[sub_envs].clone().squeeze(1)
                distance = torch.linalg.norm(obj_pos - eef_pos, dim=-1)
                robot = env.scene.articulations["robot"]
                jaw_r_idx, jaw_l_idx = jaw_body_indices(robot, "right" if is_right else "left")
                jaw_mid = jaw_contact_midpoint(robot, sub_envs, jaw_r_idx, jaw_l_idx)
                jaw_delta_xy = jaw_mid[:, :2] - obj_pos[:, :2]
                jaw_axis_xy = (
                    robot.data.body_pos_w[sub_envs, jaw_r_idx, :2]
                    - robot.data.body_pos_w[sub_envs, jaw_l_idx, :2])
                jaw_xy = torch.linalg.norm(jaw_mid[:, :2] - obj_pos[:, :2], dim=-1)
                jaw_z = jaw_mid[:, 2] - obj_pos[:, 2]
                jaw_gap = torch.linalg.norm(
                    robot.data.body_pos_w[sub_envs, jaw_r_idx]
                    - robot.data.body_pos_w[sub_envs, jaw_l_idx], dim=-1,
                )
                # A hand near an object with its jaws collapsed shut is not a grasp. The
                # kitchen-813 Anubis bowl trace was within 11 cm of the EEF but had only 7.8 mm
                # between jaw-body centers before lift; it closed to 0.3 mm during lift and the
                # bowl fell. Requiring a small positive loaded gap prevents that false-positive
                # subgoal from entering the reusable reachability bank.
                if obj_name.startswith("mug"):
                    # Apply one testable predicate so configured pad sensors are an actual gate,
                    # not merely diagnostic text. A geometry-only near-grasp is not proof of
                    # contact when the embodiment exposes per-pad force measurements.
                    _pad_sensor_names = _mug_pad_sensor_names(arm)
                    _has_pad_sensors = (_pad_sensor_names is not None and any(
                        getattr(env.scene, "sensors", {}).get(_name) is not None
                        for _name in _pad_sensor_names))
                    fail_mask = torch.as_tensor(
                        [not mug_pre_lift_grasp_ready(
                            float(distance[_i]), float(jaw_gap[_i]), float(jaw_xy[_i]),
                            float(jaw_z[_i]),
                            _mug_finger_contact_forces(int(_env), arm) if _has_pad_sensors else None,
                        ) for _i, _env in enumerate(sub_envs)],
                        dtype=torch.bool, device=jaw_xy.device)
                else:
                    fail_mask = distance > 0.15
                    fail_mask |= torch.as_tensor(
                        [not jaw_gap_shows_object_contact(float(_g)) for _g in jaw_gap],
                        dtype=torch.bool, device=jaw_gap.device)
                failed = sub_envs[fail_mask]
                succeeded = sub_envs[~fail_mask]
                side = "right" if is_right else "left"
                for _env_id, _distance in zip(failed.tolist(), distance[fail_mask].tolist()):
                    _i = (sub_envs == _env_id).nonzero(as_tuple=False)[0, 0]
                    _contact = (_mug_pad_contact_diag(_env_id, arm)
                                if obj_name.startswith("mug") else "not-configured")
                    print(
                        f"[sub_grasp] env{_env_id} {side}-hand pre-lift grasp check failed: "
                        f"object-to-eef={_distance:.4f}m (limit 0.1500m) "
                        f"jaw_xy={jaw_xy[_i].item():.4f}m jaw_dz={jaw_z[_i].item():+.4f}m "
                        f"jaw_minus_obj_xy=({jaw_delta_xy[_i, 0].item():+.4f},"
                        f"{jaw_delta_xy[_i, 1].item():+.4f})m "
                        f"jaw_axis_xy=({jaw_axis_xy[_i, 0].item():+.4f},"
                        f"{jaw_axis_xy[_i, 1].item():+.4f})m "
                        f"jaw_mid_w=({jaw_mid[_i, 0].item():+.4f},"
                        f"{jaw_mid[_i, 1].item():+.4f},{jaw_mid[_i, 2].item():+.4f}) "
                        f"obj_w=({obj_pos[_i, 0].item():+.4f},"
                        f"{obj_pos[_i, 1].item():+.4f},{obj_pos[_i, 2].item():+.4f}) "
                        f"jaw_gap={jaw_gap[_i].item():.4f}m "
                        f"obj_z={obj_pos[_i, 2].item():.4f}m "
                        f"pad_contact=[{_contact}]", flush=True,
                    )
                for _env_id, _distance in zip(succeeded.tolist(), distance[~fail_mask].tolist()):
                    _i = (sub_envs == _env_id).nonzero(as_tuple=False)[0, 0]
                    _contact = (_mug_pad_contact_diag(_env_id, arm)
                                if obj_name.startswith("mug") else "not-configured")
                    print(
                        f"[sub_grasp] env{_env_id} {side}-hand pre-lift grasp check passed: "
                        f"object-to-eef={_distance:.4f}m (limit 0.1500m) "
                        f"jaw_xy={jaw_xy[_i].item():.4f}m jaw_dz={jaw_z[_i].item():+.4f}m "
                        f"jaw_minus_obj_xy=({jaw_delta_xy[_i, 0].item():+.4f},"
                        f"{jaw_delta_xy[_i, 1].item():+.4f})m "
                        f"jaw_axis_xy=({jaw_axis_xy[_i, 0].item():+.4f},"
                        f"{jaw_axis_xy[_i, 1].item():+.4f})m "
                        f"jaw_mid_w=({jaw_mid[_i, 0].item():+.4f},"
                        f"{jaw_mid[_i, 1].item():+.4f},{jaw_mid[_i, 2].item():+.4f}) "
                        f"obj_w=({obj_pos[_i, 0].item():+.4f},"
                        f"{obj_pos[_i, 1].item():+.4f},{obj_pos[_i, 2].item():+.4f}) "
                        f"jaw_gap={jaw_gap[_i].item():.4f}m "
                        f"obj_z={obj_pos[_i, 2].item():.4f}m "
                        f"pad_contact=[{_contact}]", flush=True,
                    )
                checked[succeeded] = True
                newly_succeeded = succeeded[saved[succeeded] == 0]
                if failed.numel() > 0:
                    _reset_envs(
                        env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,
                        env_goal_indices, env_cmd_indices_r, env_cmd_indices_l,
                        grasp_checked_r, grasp_checked_l, nav_phase,
                        env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r,
                        drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l,
                        new_gripper_commands_L, new_gripper_commands_R, timestep,
                        env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l,
                        failed, reason=f"sub_grasp_fail_{arm}",
                    )
                if newly_succeeded.numel() > 0:
                    lift_is_enabled = _lift_steps > 0 and (is_right or _lift_left)
                    if lift_is_enabled:
                        # The pose bank is used to build later full scripts. A pre-lift pose check
                        # cannot establish that the fingers actually carried the object; defer
                        # promotion until the commanded lift has moved the object and the jaws
                        # remain separated. State 2 is reset with the episode by _reset_envs.
                        saved[newly_succeeded] = 2
                    else:
                        goal_mgr.on_sub_success(newly_succeeded, arm=arm)
                        saved[newly_succeeded] = 1


            # =================== Mobile Base Motion ("N") ===================
            def _wrap_pi_t(a):
                return (a + torch.pi) % (2 * torch.pi) - torch.pi

            if torch.any(nav_mask):
                nav_rows = active_env_indices[nav_mask]
                nav_cols = env_goal_indices[nav_rows]
                nav_skill = skill_ids_tensor[nav_rows, nav_cols]
                # `~resolved` reproduces v1's one-shot behaviour: the sentinel dispatch fired once
                # because resolving the step overwrote the very slots the sentinel lived in. Without
                # the latch, nav.open_articulation would recompute "0.25 m behind the base" from the
                # base's NEW pose on every tick and the robot would reverse forever.
                nav_unresolved = ~resolved_tensor[nav_rows, nav_cols]
                pull_mask = (nav_skill == SID_PULL) & nav_unresolved
                push_mask = (nav_skill == SID_PUSH) & nav_unresolved
                # Build the runtime context once; resolvers read from it.
                _ctx_nav = RuntimeContext(
                    robot=env.scene.articulations["robot"],
                    env_origins=env.scene.env_origins,
                    r_eef_idx=r_eef_idx,
                    l_eef_idx=l_eef_idx,
                    base_link_idx=base_link_idx,
                    has_ramen=has_ramen,
                    has_sweet_potato=has_sweet_potato,
                    rigid_objects=env.scene.rigid_objects,
                )
                if torch.any(pull_mask):
                    pull_idx = nav_rows[pull_mask]
                    pull_j = env_goal_indices[pull_idx]
                    # back_off_m rides in slot 0 of a runtime step's (otherwise all-zero) payload.
                    # Read it BEFORE the resolver overwrites slots 0..2 with the answer. This is an
                    # argument to a named skill, not a marker: nothing sniffs it to decide anything.
                    back_off_m = payloads_tensor[pull_idx, pull_j, 0].clone()
                    src, yaw = resolve_skill("nav.open_articulation", _ctx_nav, pull_idx,
                                             back_off_m=back_off_m)
                    payloads_tensor[pull_idx, pull_j, :2] = src
                    payloads_tensor[pull_idx, pull_j,  2] = yaw
                    resolved_tensor[pull_idx, pull_j] = True
                if torch.any(push_mask):
                    push_idx = nav_rows[push_mask]
                    push_j = env_goal_indices[push_idx]
                    back_off_m = payloads_tensor[push_idx, push_j, 0].clone()
                    src, yaw = resolve_skill("nav.close_articulation", _ctx_nav, push_idx,
                                             back_off_m=back_off_m)
                    payloads_tensor[push_idx, push_j, :2] = src
                    payloads_tensor[push_idx, push_j,  2] = yaw
                    resolved_tensor[push_idx, push_j] = True
                arc_mask = (nav_skill == SID_ARC) & nav_unresolved
                if torch.any(arc_mask):
                    arc_idx = nav_rows[arc_mask]
                    arc_j = env_goal_indices[arc_idx]
                    # CLONE ALL FIVE BEFORE RESOLVING. The answer goes back into slots 0..2 --
                    # the same slots these params live in (executor_dispatch.encode_payload) --
                    # so reading any of them afterwards would read the answer instead of the
                    # argument. Identical ordering to back_off_m above; spelled out because
                    # there are five of them and only the first is protected by habit.
                    radius_m = payloads_tensor[arc_idx, arc_j, 0].clone()
                    sweep_deg = payloads_tensor[arc_idx, arc_j, 1].clone()
                    hinge_sign = payloads_tensor[arc_idx, arc_j, 2].clone()
                    arc_back = payloads_tensor[arc_idx, arc_j, 3].clone()
                    arc_diag = payloads_tensor[arc_idx, arc_j, 4].clone()
                    src, yaw = resolve_skill("nav.open_door_arc", _ctx_nav, arc_idx,
                                             radius_m=radius_m, sweep_deg=sweep_deg,
                                             hinge_sign=hinge_sign, back_off_m=arc_back,
                                             diag_deg=arc_diag)
                    payloads_tensor[arc_idx, arc_j, :2] = src
                    payloads_tensor[arc_idx, arc_j,  2] = yaw
                    resolved_tensor[arc_idx, arc_j] = True
                    try:
                        import door_open_trace
                        door_open_trace.publish_goal(arc_idx, src, yaw)
                    except Exception:
                        pass



                # Get the global indices of environments that need a nav plan
                nav_indices_global = active_env_indices[nav_mask]

                # Get robot's absolute WORLD position and orientation
                root_pos_w = env.scene.articulations['robot'].data.body_pos_w[nav_indices_global, base_link_idx, :2]
                w, x, y, z = env.scene.articulations['robot'].data.body_quat_w[nav_indices_global, base_link_idx].unbind(-1)
                current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

                nav_env_origins = env.scene.env_origins[nav_indices_global]
                base_pos_e = root_pos_w - nav_env_origins[:, :2]

                # Get the environment-relative goals
                nav_goal_step_indices = env_goal_indices[nav_indices_global]
                nav_goals = payloads_tensor[nav_indices_global, nav_goal_step_indices]
                goal_pos_e, goal_yaw = nav_goals[:, :2], nav_goals[:, 2]
                # Now the subtraction is correct because both positions are in the same environment frame
                delta_pos = goal_pos_e - base_pos_e
                distance = torch.linalg.norm(delta_pos, dim=1)
                moving_mask = distance > (goal_reached_distance+0.03)
                #rotating_mask = ~moving_mask
                done_mask = ~moving_mask
                if torch.any(moving_mask):
                    m_delta_pos = delta_pos[moving_mask]
                    m_current_yaw = current_yaw[moving_mask]
                    m_distance = distance[moving_mask]
                    cos_yaw, sin_yaw = torch.cos(m_current_yaw), torch.sin(m_current_yaw)
                    vx_local = m_delta_pos[:, 0] * cos_yaw + m_delta_pos[:, 1] * sin_yaw
                    vy_local = -m_delta_pos[:, 0] * sin_yaw + m_delta_pos[:, 1] * cos_yaw
                     
                    speed_scale = torch.where(m_distance > (slowdown_radius-0.7), default_speed, min_speed + (default_speed - min_speed) * (m_distance / (slowdown_radius-0.7)))
                    
                    nav_actions = torch.zeros(torch.sum(moving_mask), 3, device=device)
                    nav_actions[:, 0] =  speed_scale * vx_local * 3
                    nav_actions[:, 1] = 0#speed_scale * vy_local * 2

                    # HOLONOMIC MOTION, FOR THE ARC SKILL ONLY.
                    #
                    # The two lines above drive the base along its own FORWARD axis and nothing
                    # else: the lateral term is commented out and yaw is never commanded at all.
                    # For nav.open_articulation that is correct and sufficient -- a drawer is
                    # opened by reversing in a straight line.
                    #
                    # nav.open_door_arc is the opposite case. Its target is a rotation of the whole
                    # robot about the door's hinge, so the required motion is mostly LATERAL and
                    # carries a yaw change of the same angle. Dropping both leaves only the
                    # backward component, which is why the door stopped after exactly one step's
                    # worth of rotation whatever the step size (10.8 deg at an 11.25 deg step,
                    # 27.5 deg at 30). The user, watching it: "the base did not move much and just
                    # went behind it should move like in at fist move back but later move in a quad
                    # way as the door joint is rotating joint".
                    #
                    # The base is holonomic -- base_prismatic_x_joint, base_prismatic_y_joint and
                    # base_revolute_z_joint are all actuated -- so strafing and turning are
                    # available, they were simply never commanded. Enabled ONLY for the arc skill,
                    # so every existing task keeps the exact behaviour it has today.
                    _arc_rows = (skill_ids_tensor[nav_indices_global, nav_goal_step_indices]
                                 == SID_ARC)[moving_mask]
                    if torch.any(_arc_rows):
                        # SIGN: vy_local here is the standard body-frame rotation (+y = LEFT,
                        # matching skills.py's right_hat convention) -- but pre_process_actions
                        # negates column 1 before rotating body->world (its own vy_local = -1 *
                        # delta_pose_base[:, 1], see this file's pre_process_actions above), so
                        # whatever gets written to nav_actions[:, 1] must be PRE-negated to cancel
                        # that. Phase 2 of this same nav state machine (a few hundred lines up,
                        # `nav_actions2[:, 1] *= -1`) and simvla_video.py's phase 1/2 both do this;
                        # this line was the one instance that didn't, which inverted the arc
                        # skill's lateral retreat (commanded RIGHT executed as LEFT, GPU job
                        # 2106993). Do not remove this negation without re-deriving the whole
                        # chain -- see base-lateral-sign-report.md.
                        nav_actions[_arc_rows, 1] = -(speed_scale * vy_local * 3)[_arc_rows]
                        _yaw_err = _wrap_pi_t(goal_yaw[moving_mask] - m_current_yaw)
                        nav_actions[_arc_rows, 2] = (2.0 * _yaw_err)[_arc_rows]

                    delta_pose_base[nav_indices_global[moving_mask]] = nav_actions
                if torch.any(done_mask):
                    delta_pose_base[nav_indices_global[done_mask]] = 0.0
                    finished_mask[nav_indices_global[done_mask]] = True

            # =================== Mobile Base straight Motion ("N_s") ===================
            def wrap_pi(a):
                return (a + torch.pi) % (2 * torch.pi) - torch.pi
            if torch.any(nav_s_mask):
                idx = active_env_indices[nav_s_mask]
                # pose
                root_pos_w = env.scene.articulations['robot'].data.body_pos_w[idx, base_link_idx, :2].clone()
                w, x, y, z = env.scene.articulations['robot'].data.body_quat_w[idx, base_link_idx].unbind(-1)
                yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

                # env frame
                origins = env.scene.env_origins[idx].clone()
                base_pos_e = root_pos_w - origins[:, :2]

                # goals (env frame)
                goal_step = env_goal_indices[idx]                 # long, in-range
                goals = payloads_tensor[idx, goal_step]           # [N,3]
                goal_pos_e, goal_yaw = goals[:, :2], goals[:, 2]

                # These two used to be `goal_yaw <= -1000` / `>= 1000` — a marker parked in the yaw
                # slot. No latch: `goals` is an advanced-index COPY of payloads_tensor, so the
                # branches below never destroyed the marker and re-fired every tick. They still do.
                ns_skill = skill_ids_tensor[idx, goal_step]

                # Go to range
                range_N_s_mask = ns_skill == SID_TO_RANGE
                range_N_s_idx = torch.where(range_N_s_mask)[0]
                if torch.any(range_N_s_mask):
                    if obj_init[args_cli.task]["range_direction"] == "N":
                        goal_pos_e[range_N_s_idx] = (env.scene.articulations["range"].data.body_pos_w[range_N_s_idx, -1] - env.scene.env_origins[range_N_s_idx] - torch.tensor([0.4, 0.1, 0.0], device=range_N_s_idx.device).unsqueeze(0))[:,:2] 
                        goal_yaw[range_N_s_idx] = torch.tensor([0.0], device=device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["range_direction"] == "E":
                        goal_pos_e[range_N_s_idx] = (env.scene.articulations["range"].data.body_pos_w[range_N_s_idx, -1] - env.scene.env_origins[range_N_s_idx] - torch.tensor([-0.1, -0.4, 0.0], device=range_N_s_idx.device).unsqueeze(0))[:,:2] 
                        goal_yaw[range_N_s_idx] = torch.tensor([-1.5708], device=device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["range_direction"] == "S":
                        goal_pos_e[range_N_s_idx] = (env.scene.articulations["range"].data.body_pos_w[range_N_s_idx, -1] - env.scene.env_origins[range_N_s_idx] - torch.tensor([-0.4, -0.1, 0.0], device=range_N_s_idx.device).unsqueeze(0))[:,:2] 
                        goal_yaw[range_N_s_idx] = torch.tensor([3.141592], device=device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["range_direction"] == "W":
                        goal_pos_e[range_N_s_idx] = (env.scene.articulations["range"].data.body_pos_w[range_N_s_idx, -1] - env.scene.env_origins[range_N_s_idx] - torch.tensor([0.2, 0.4, 0.0], device=range_N_s_idx.device).unsqueeze(0))[:,:2] 
                        goal_yaw[range_N_s_idx] = torch.tensor([1.5708], device=device).unsqueeze(0) 
                
                # Go to pot
                # -0.5 direction
                pot_N_s_mask = ns_skill == SID_TO_POT
                pot_N_s_idx = torch.where(pot_N_s_mask)[0]
                if torch.any(pot_N_s_mask):
                    if obj_init[args_cli.task]["direction"] == "N":
                        goal_pos_e[pot_N_s_idx] = (env.scene.rigid_objects["pot0"].data.body_pos_w[pot_N_s_idx].squeeze(1) - env.scene.env_origins[pot_N_s_idx] - torch.tensor([0.5, 0.0, 0.0], device=pot_N_s_idx.device).unsqueeze(0))[:,:2]
                        goal_yaw[pot_N_s_idx] = torch.tensor([0.0], device=device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["direction"] == "E":
                        goal_pos_e[pot_N_s_idx] = (env.scene.rigid_objects["pot0"].data.body_pos_w[pot_N_s_idx].squeeze(1) - env.scene.env_origins[pot_N_s_idx] - torch.tensor([0.0, -0.5, 0.0], device=pot_N_s_idx.device).unsqueeze(0))[:,:2]
                        goal_yaw[pot_N_s_idx] = torch.tensor([-1.5708], device=pot_N_s_idx.device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["direction"] == "S":
                        goal_pos_e[pot_N_s_idx] = (env.scene.rigid_objects["pot0"].data.body_pos_w[pot_N_s_idx].squeeze(1) - env.scene.env_origins[pot_N_s_idx] - torch.tensor([-0.5, 0.0, 0.0], device=pot_N_s_idx.device).unsqueeze(0))[:,:2]
                        goal_yaw[pot_N_s_idx] = torch.tensor([3.141592], device=pot_N_s_idx.device).unsqueeze(0) 
                    elif obj_init[args_cli.task]["direction"] == "W":
                        goal_pos_e[pot_N_s_idx] = (env.scene.rigid_objects["pot0"].data.body_pos_w[pot_N_s_idx].squeeze(1) - env.scene.env_origins[pot_N_s_idx] - torch.tensor([0.0, 0.5, 0.0], device=pot_N_s_idx.device).unsqueeze(0))[:,:2]
                        goal_yaw[pot_N_s_idx] = torch.tensor([1.5708], device=pot_N_s_idx.device).unsqueeze(0) 

                # distances & current bearing-to-goal (for phase 0 init and stopping)
                delta_pos = goal_pos_e - base_pos_e               # env/world frame
                #print(torch.norm(delta_pos, dim=1))
                dist = torch.linalg.norm(delta_pos, dim=1)
                bearing_now = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

                # ---- Phase 0: rotate to face goal line (lock bearing) ----
                m0 = nav_phase[idx] == 0
                if torch.any(m0):
                    idx0 = idx[m0]

                    # lock bearing (you can cache this once externally if you prefer)
                    locked_bearing[idx0] = bearing_now[m0]
                    yaw_err0 = wrap_pi(locked_bearing[idx0] - yaw[m0])

                    # pure rotation, ramped at both ends (see _shaped_yaw)
                    ang_cmd = _shaped_yaw(yaw_err0, idx0)
                    delta_pose_base[idx0, 0:2] = 0.0
                    delta_pose_base[idx0, 2] = ang_cmd

                    # advance when aligned
                    aligned0 = torch.abs(yaw_err0) < goal_reached_yaw
                    if torch.any(aligned0):
                        rows = idx0[aligned0]
                        nav_phase[rows] = 1
                        delta_pose_base[rows] = 0.0   # optional: kill residual spin
                        prev_yaw_cmd[rows] = 0.0

                # ---- Phase 1: approach until dist <= 0.2 ----
                # Profiles may steer continuously or preserve the legacy straight body-x drive.
                m1 = nav_phase[idx] == 1
                if torch.any(m1):
                    idx1 = idx[m1]
                    dist1 = dist[m1]

                    # speed ramp (same as your logic), slew-limited so the drive starts
                    # steadily instead of stepping to full speed in one frame
                    speed_scale = torch.where(
                        dist1 > slowdown_radius,
                        torch.full_like(dist1, default_speed),
                        min_speed + (default_speed - min_speed) * (dist1 / slowdown_radius),
                    )
                    speed_scale = _shaped_lin(speed_scale, idx1)
                    nav_actions = torch.zeros(idx1.shape[0], 3, device=delta_pose_base.device)

                    if _nav.closed_loop_approach:
                        # Keep steering toward the goal while crossing the room. The initial
                        # bearing is only exact at phase entry; contact, base slip, and yaw
                        # tracking error otherwise turn this long segment into a blind line that
                        # can enter furniture before phase 2 gets a chance to correct it.
                        dxy1 = delta_pos[m1]
                        cy1, sy1 = torch.cos(yaw[m1]), torch.sin(yaw[m1])
                        bx1 = dxy1[:, 0] * cy1 + dxy1[:, 1] * sy1
                        by1 = -dxy1[:, 0] * sy1 + dxy1[:, 1] * cy1
                        d1 = torch.stack([bx1, by1], dim=-1)
                        n1 = torch.linalg.norm(d1, dim=-1, keepdim=True).clamp(min=1e-6)
                        nav_actions[:, :2] = (d1 / n1) * speed_scale.unsqueeze(-1)
                        # pre_process_actions negates body y before rotating it into world.
                        nav_actions[:, 1] *= -1
                    else:
                        nav_actions[:, 0] = speed_scale
                        nav_actions[:, 1] = 0.0
                    nav_actions[:, 2] = 0.0
                    delta_pose_base[idx1] = nav_actions

                    # transition to phase 2 when close enough to the goal line
                    near_line = dist1 <= 0.2
                    if torch.any(near_line):
                        rows = idx1[near_line]
                        nav_phase[rows] = 2
                        delta_pose_base[rows] = 0.0

                # ---- Phase 2: move to goal position using (vx, vy) until dist <= goal_reached_distance ----
                m2 = nav_phase[idx] == 2
                if torch.any(m2):
                    idx2 = idx[m2]
                    dist2 = dist[m2]

                    # rotate world/env delta into body frame: d_body = R(-yaw) * delta_pos
                    dxy = delta_pos[m2]                                   # [K,2] env/world
                    cy, sy = torch.cos(yaw[m2]), torch.sin(yaw[m2])
                    # [vx, vy] = R^T * d = [[ cy,  sy],
                    #                        [-sy,  cy]] @ [dx, dy]
                    vx = dxy[:, 0] * cy + dxy[:, 1] * sy
                    vy = -dxy[:, 0] * sy + dxy[:, 1] * cy
                    d_body = torch.stack([vx, vy], dim=-1)                # [K,2]

                    # normalized direction with slowdown
                    d_norm = torch.linalg.norm(d_body, dim=-1, keepdim=True).clamp(min=1e-6)
                    dir_body = d_body / d_norm

                    speed_scale2 = torch.where(
                        dist2 > slowdown_radius,
                        torch.full_like(dist2, default_speed),
                        min_speed + (default_speed - min_speed) * (dist2 / slowdown_radius),
                    )
                    speed_scale2 = _shaped_lin(speed_scale2, idx2).unsqueeze(-1)

                    nav_actions2 = torch.zeros(idx2.shape[0], 3, device=delta_pose_base.device)
                    # again, keep the sign convention consistent with your sim
                    nav_actions2[:, :2] = dir_body * speed_scale2
                    nav_actions2[:,1] *= -1
                    nav_actions2[:, 2] = 0.0
                    delta_pose_base[idx2] = nav_actions2

                    # transition to phase 3 when we have reached the goal position
                    at_goal_xy = dist2 <= goal_reached_distance
                    if torch.any(at_goal_xy):
                        rows = idx2[at_goal_xy]
                        nav_phase[rows] = 3
                        delta_pose_base[rows] = 0.0
                        prev_lin_cmd[rows] = 0.0      # next drive ramps in from rest

                # ---- Phase 3: rotate to goal yaw ----
                m3 = nav_phase[idx] == 3
                if torch.any(m3):
                    idx3 = idx[m3]
                    yaw_err3 = wrap_pi(goal_yaw[m3] - yaw[m3])

                    rot_actions = torch.zeros(idx3.shape[0], 3, device=delta_pose_base.device)
                    rot_actions[:, 2] = _shaped_yaw(yaw_err3, idx3)
                    delta_pose_base[idx3] = rot_actions

                    aligned3 = torch.abs(yaw_err3) < goal_final_yaw
                    if torch.any(aligned3):
                        rows = idx3[aligned3]
                        delta_pose_base[rows] = 0.0
                        prev_yaw_cmd[rows] = 0.0
                        finished_mask[rows] = True
                        nav_phase[rows] = 0  # or keep at 3 / set to 0 for next waypoint
                if global_frames % 200 == 0:
                    for _row, _env in enumerate(idx.tolist()):
                        print("[nav-state] " + json.dumps({
                            "frame": global_frames, "env_id": _env,
                            "step": int(env_goal_indices[_env]),
                            "phase": int(nav_phase[_env]),
                            "base_xy": base_pos_e[_row].tolist(),
                            "base_yaw": float(yaw[_row]),
                            "goal_xy": goal_pos_e[_row].tolist(),
                            "goal_yaw": float(goal_yaw[_row]),
                            "distance_m": float(dist[_row]),
                            "command": delta_pose_base[_env].tolist(),
                        }), flush=True)
            # =================== Right Arm Motion ("A_r") ===================
            if torch.any(arm_r_mask):
                arm_r_rows = active_env_indices[arm_r_mask]
                arm_r_cols = env_goal_indices[arm_r_rows]
                skill_r = skill_ids_tensor[arm_r_rows, arm_r_cols]
                # One-shot, exactly as the sentinels were: each resolver below writes its answer
                # into payload slots [:3] and [3:7], which is where v1's sentinel lived, so the
                # isclose stopped matching after the first tick. See resample_goals_for_envs.
                arm_r_unresolved = ~resolved_tensor[arm_r_rows, arm_r_cols]
                bowl_place_mask = (skill_r == SID_BOWL_PLACE) & arm_r_unresolved
                move_left_mask = (skill_r == SID_BOTTLE_TO_POSITION) & arm_r_unresolved
                bottle_tilt_mask = (skill_r == SID_BOTTLE_POUR) & arm_r_unresolved
                pause_mask = (skill_r == SID_PAUSE) & arm_r_unresolved

                # Build context once for all right-arm resolvers below.
                # All resolver bodies live in skills.py, on the skill class that owns them.
                _ctx_r = RuntimeContext(
                    robot=env.scene.articulations["robot"],
                    env_origins=env.scene.env_origins,
                    r_eef_idx=r_eef_idx,
                    l_eef_idx=l_eef_idx,
                    base_link_idx=base_link_idx,
                    has_ramen=has_ramen,
                    has_sweet_potato=has_sweet_potato,
                    rigid_objects=env.scene.rigid_objects,
                )

                if torch.any(pause_mask):
                    pause_idx = arm_r_rows[pause_mask]
                    pos_e, quat = resolve_skill("arm.pause", _ctx_r, pause_idx, r_eef_idx)
                    j = env_goal_indices[pause_idx]
                    payloads_tensor[pause_idx, j, :3] = pos_e
                    payloads_tensor[pause_idx, j, 3:7] = quat
                    resolved_tensor[pause_idx, j] = True

                if torch.any(bottle_tilt_mask):
                    bottle_tilt_idx = arm_r_rows[bottle_tilt_mask]
                    pos_e, q_out = resolve_skill("arm.bottle_pour", _ctx_r, bottle_tilt_idx, r_eef_idx)
                    j = env_goal_indices[bottle_tilt_idx]
                    payloads_tensor[bottle_tilt_idx, j, :3] = pos_e
                    payloads_tensor[bottle_tilt_idx, j, 3:7] = q_out
                    resolved_tensor[bottle_tilt_idx, j] = True

                if torch.any(move_left_mask):
                    move_left_idx = arm_r_rows[move_left_mask]
                    pos_e, q_out = resolve_skill("arm.bottle_to_position", _ctx_r, move_left_idx, r_eef_idx)
                    j = env_goal_indices[move_left_idx]
                    payloads_tensor[move_left_idx, j, :3] = pos_e
                    payloads_tensor[move_left_idx, j, 3:7] = q_out
                    resolved_tensor[move_left_idx, j] = True

                arm_r_indices = active_env_indices[arm_r_mask]

                if torch.any(bowl_place_mask):
                    bowl_place_idx = arm_r_rows[bowl_place_mask]
                    j = env_goal_indices[bowl_place_idx]
                    # The place params ride slots 0..4 of this runtime step's otherwise-zero row
                    # (executor_dispatch.encode_payload). Read them BEFORE the resolver overwrites
                    # the row with its answer -- same order as nav.open_articulation's back_off_m.
                    #
                    # Passing them is the whole point: resolve_skill(**params) with none supplied
                    # silently used the skill's own constants, so every authored place height was
                    # discarded and three A/B'd variants scored identically over ~280 episodes.
                    _pp = payloads_tensor[bowl_place_idx, j, :5].clone()
                    pos_e, q_out = resolve_skill("arm.bowl_place", _ctx_r, bowl_place_idx,
                                                 r_eef_idx, forward_m=_pp[:, 0],
                                                 down_m=_pp[:, 1], min_eef_z=_pp[:, 2],
                                                 roll_deg=_pp[:, 3], lateral_m=_pp[:, 4])
                    payloads_tensor[bowl_place_idx, j, :3] = pos_e
                    payloads_tensor[bowl_place_idx, j, 3:7] = q_out
                    resolved_tensor[bowl_place_idx, j] = True
                arm_r_indices = active_env_indices[arm_r_mask]
                needs_plan_indices = [i.item() for i in arm_r_indices if env_cmd_plans_r[i.item()] is None]

                if needs_plan_indices:
                    needs_plan_indices_temp = torch.as_tensor(needs_plan_indices, device=reset_r.device)
                    goal_step_indices = env_goal_indices[needs_plan_indices]
                    arm_r_goals = payloads_tensor[needs_plan_indices, goal_step_indices]
                    # No latch on these two, and none is needed: `arm_r_goals` is an advanced-index
                    # COPY, so v1's 999.0 flags were never overwritten in payloads_tensor and these
                    # branches re-fired on every replan. They still do.
                    #
                    # arm.place is NOT runtime-resolved. Only its quaternion is (from the live eef,
                    # below); slots [:3] are the authored, bbox-derived IK target and stay.
                    skill_r_plan = skill_ids_tensor[needs_plan_indices, goal_step_indices]
                    reset_mask = (skill_r_plan == SID_RESET)
                    reset_env = torch.nonzero(reset_mask, as_tuple=True)[0].tolist()
                    reset_env = torch.as_tensor(reset_env, device=reset_r.device)
                    _postrelease_reset_envs = set()
                    _final_home_reset_envs = set()
                    place_mask = (skill_r_plan == SID_PLACE)
                    place_env = torch.nonzero(place_mask, as_tuple=True)[0].tolist()
                    place_env = torch.as_tensor(place_env, device=reset_r.device)
                    not_reset_and_not_place = ~(reset_mask | place_mask)
                    grasp_env = torch.nonzero(not_reset_and_not_place, as_tuple=True)[0].tolist()
                    grasp_env = torch.as_tensor(grasp_env, device=reset_r.device)

                    # 1. Grasp, for reset
                    if torch.any(not_reset_and_not_place):

                        idx = needs_plan_indices_temp[not_reset_and_not_place]  
                        sel = first_reset[idx].view(-1)                         

                        m3 = (sel == 2)

                        idx3 = idx[m3]
                        
                        robot_data = env.scene.articulations["robot"].data

                        # A RE-PLAN IS NOT A NEW GRASP STEP, and both of the things below think
                        # in grasp steps. reset_r records where the arm should RETURN to after
                        # grasping, captured from the live eef at plan time, and first_reset
                        # counts how many grasp steps this episode has planned. A retry re-enters
                        # this branch, so without excluding it the return pose would be
                        # re-recorded from PARTWAY THROUGH the reach -- the arm would afterwards
                        # retreat to somewhere over the counter rather than to where it started --
                        # and the counter would run ahead of the step it is supposed to track.
                        #
                        # Envs not retrying are untouched, so with arm_r_retry all zero this is
                        # exactly the previous behaviour.
                        _ge = needs_plan_indices_temp[grasp_env]
                        _fresh = _ge[arm_r_retry[_ge] == 0] if _ge.numel() else _ge
                        if idx3.numel() > 0:
                            print("skip reset mark.")
                        elif _fresh.numel():
                            reset_r[_fresh, :3] = robot_data.body_pos_w[_fresh, r_eef_idx]
                            reset_r[_fresh, 3:7] = robot_data.body_quat_w[_fresh, r_eef_idx]
                            reset_r_js[_fresh] = robot_data.joint_pos[_fresh][:, r_j_index]
                        _inc = needs_plan_indices_temp[not_reset_and_not_place]
                        if _inc.numel():
                            first_reset[_inc[arm_r_retry[_inc] == 0]] += 1
                    # 2. Reset
                    if torch.any(reset_mask):
                        arm_r_goals[reset_env,:7] = reset_r[needs_plan_indices_temp[reset_env]]
                        if _postrelease_retreat_m > 0:
                            for _row in reset_env.tolist():
                                _e = int(needs_plan_indices_temp[_row])
                                _step = int(goal_step_indices[_row])
                                if (_step == 0
                                        or int(skill_ids_tensor[_e, _step - 1])
                                        != skill_ids["gripper.set"]
                                        or float(payloads_tensor[_e, _step - 1, 0]) > 0):
                                    continue
                                _object = env.scene.rigid_objects[args_cli.obj_name]
                                _xyz = postrelease_retreat_target(
                                    robot.data.body_pos_w[_e, r_eef_idx].tolist(),
                                    _object.data.body_pos_w[_e, 0].tolist(),
                                    robot.data.body_pos_w[_e, base_link_idx].tolist(),
                                    _postrelease_retreat_m, _postrelease_retreat_lift_m)
                                arm_r_goals[_row, :3] = torch.as_tensor(
                                    _xyz, dtype=arm_r_goals.dtype, device=device)
                                arm_r_goals[_row, 3:7] = robot.data.body_quat_w[_e, r_eef_idx]
                                _postrelease_reset_envs.add(_e)
                                print(f"[release-retreat] env{_e} open right hand to "
                                      f"{tuple(round(v, 4) for v in _xyz)}", flush=True)
                        if _postrelease_final_home:
                            for _row in reset_env.tolist():
                                _e = int(needs_plan_indices_temp[_row])
                                _step = int(goal_step_indices[_row])
                                if not postrelease_final_home_step(
                                        _step, skill_ids_tensor[_e], payloads_tensor[_e],
                                        reset_id=SID_RESET,
                                        gripper_id=skill_ids["gripper.set"]):
                                    continue
                                _home_pos_w, _home_quat_w = base2world(
                                    env, episode_home_r_base[_e, :3],
                                    episode_home_r_base[_e, 3:7], _e)
                                arm_r_goals[_row, :3] = _home_pos_w
                                arm_r_goals[_row, 3:7] = _home_quat_w
                                _final_home_reset_envs.add(_e)
                                print(f"[release-home] env{_e} returning to episode home "
                                      f"at {tuple(round(float(v), 4) for v in _home_pos_w)}",
                                      flush=True)
                        arm_r_goals[reset_env, :3] -= env.scene.env_origins[needs_plan_indices_temp[reset_env], :3]
                        if has_ramen or has_sweet_potato:
                            arm_r_goals[reset_env, 3:7] = robot.data.body_quat_w[needs_plan_indices_temp[reset_env], r_eef_idx]

                    # 3. Place
                    if torch.any(place_mask):
                        arm_r_goals[place_env, 3:7] = env.scene["robot"].data.body_quat_w[needs_plan_indices_temp[place_env], r_eef_idx]
                        if args_cli.pour_pot:
                            arm_r_goals[place_env, 0:3] = env.scene.rigid_objects["pot0"].data.body_pos_w[needs_plan_indices_temp[place_env], 0:3].squeeze(1).clone()
                            if has_ramen:
                                arm_r_goals[place_env, 2] += 0.2 
                                arm_r_goals[place_env, 1] -= 0.05
                            else:
# for pour water in pot
                                arm_r_goals[place_env, 2] += 0.2 
                                arm_r_goals[place_env, 1] -= 0.15
                                arm_r_goals[place_env, 0] -= 0.1

                        else:
                            reset_r[needs_plan_indices_temp[place_env], :3] = env.scene["robot"].data.body_pos_w[needs_plan_indices_temp[place_env], r_eef_idx]
                            reset_r[needs_plan_indices_temp[place_env], 3:7] = env.scene["robot"].data.body_quat_w[needs_plan_indices_temp[place_env], r_eef_idx]

                    for i, env_idx in enumerate(needs_plan_indices):
                        _repose_world_to_base("r", env_idx, int(skill_r_plan[i]) == SID_GRASP)
                        # Get current joint state data for the specific environment
                        sim_data = env.scene.articulations['robot'].data
                        # CLAMP THE START STATE INTO ITS LIMITS. The arm is positioned by
                        # differential IK -- the plan tracker, the reach jog, the post-grasp lift
                        # -- and none of those respect cuRobo's joint limits, so a joint can end
                        # a hair outside one. cuRobo then refuses the whole plan with
                        # INVALID_START_STATE_JOINT_LIMITS and the episode is discarded over a
                        # fraction of a milliradian. Measured on the AI Worker retreat: 2 of 10
                        # failures. Clamping inside by 1e-4 costs nothing when the state is
                        # already legal, which is the usual case.
                        _jl = sim_data.joint_limits[env_idx, r_j_index]
                        joint_pos = torch.clamp(sim_data.joint_pos[env_idx, r_j_index],
                                                _jl[:, 0] + 1e-4, _jl[:, 1] - 1e-4).tolist()
                        joint_vel = sim_data.joint_vel[env_idx, r_j_index].tolist()

                        # The measured start is passed via JointState below. Do not
                        # replace cuRobo's tensor retract configuration with a list:
                        # a later locked-joint refresh copies into that tensor.
                        
                        # Convert goal to world frame and then to base frame
                        goal_env = arm_r_goals[i]
                        env_origin_pos = env.scene.env_origins[env_idx]
                        goal_pos_w = goal_env[:3] + env_origin_pos
                        goal_quat_w = goal_env[3:7]
                        # FOLLOW THE OBJECT. The grasp pose is authored at EMIT time against the
                        # object's spawn pose, and the object does not stay there: measured, the
                        # bottle settles to obj_z 0.9818 against the 0.952 it is spawned at, and
                        # the authored goal ends a median 0.033 m from it -- the bottle's entire
                        # radius. The arm then reaches that goal accurately and the pads close
                        # 0.055 m away, outside the object.
                        #
                        # Translating the goal by however far the object has actually moved keeps
                        # the authored ORIENTATION and candidate choice -- which are correct; the
                        # pads straddle the bottle at the authored pose, checked numerically --
                        # and corrects only the part that went stale.
                        if int(skill_r_plan[i]) == SID_GRASP and _cylindrical_mug_grasp_r:
                            # PLAN TO A STANDOFF, NEVER TO THE BOTTLE.
                            #
                            # tilt_at_plan is 0.0 degrees in every env and obj_tilt is 90.0 at
                            # close: the bottle is upright when the grasp is planned and flat by
                            # the time the jaws arrive. The approach knocks it over, and cuRobo
                            # cannot avoid it -- the collision world is captured once at startup
                            # against env 0's base pose, so it holds every obstacle at 1e10 m.
                            # Restoring the real transform would not help either: after the robot
                            # navigates, that captured frame is stale, so the obstacles would be
                            # in the wrong place rather than merely absent.
                            #
                            # No world model is needed if the planner is never asked to go there.
                            # It plans to a point backed off along the hand's own approach axis,
                            # and the axial jog -- which already exists and already approaches
                            # along that axis -- covers the last 0.09 m. The jog still aims at the
                            # true grasp pose, so nothing downstream changes.
                            try:
                                _og = env.scene.rigid_objects[args_cli.obj_name]
                                _now = _og.data.body_pos_w[env_idx, 0]
                                _spawn = (_og.data.default_root_state[env_idx, :3]
                                          + env.scene.env_origins[env_idx])
                                _shift = _now - _spawn
                                goal_pos_w = goal_pos_w + _shift
                                # Centre the physical jaw gap, not a robot-specific hard-coded
                                # wrist offset. Measure wrist-to-jaw in the current articulation,
                                # express it in the wrist frame, then rotate it into the authored
                                # grasp orientation. This supports each robot's actual gripper
                                # geometry and avoids treating the AI Worker offset as universal.
                                _jr_i, _jl_i = jaw_body_indices(robot, "right")
                                _jaw_mid = jaw_contact_midpoint(robot, env_idx, _jr_i, _jl_i)
                                _eef_now = robot.data.body_pos_w[env_idx, r_eef_idx]
                                _eef_q_now = R.from_quat(
                                    robot.data.body_quat_w[env_idx, r_eef_idx].cpu().numpy()[
                                        [1, 2, 3, 0]])
                                _jaw_offset_local = _eef_q_now.inv().apply(
                                    (_jaw_mid - _eef_now).cpu().numpy())
                                _pin = torch.as_tensor(
                                    R.from_quat(goal_quat_w.cpu().numpy()[[1, 2, 3, 0]]).apply(
                                        _jaw_offset_local),
                                    dtype=goal_pos_w.dtype, device=goal_pos_w.device)
                                goal_pos_w = goal_pos_w - _pin
                                # AND THEN PUT THE GAP ON THE AXIS, rather than trusting that the
                                # pinch offset alone got it there.
                                #
                                # The comment above says the correction hands the whole 0.025 m
                                # budget back to the jog. Measured over 147 closes where the
                                # bottle was STILL UPRIGHT -- so the goal cannot be blamed on the
                                # object having moved -- GOAL_TO_OBJ has a median of 0.0420 m and
                                # only 27 of those 147 are inside the 0.025 m budget at all. The
                                # arm then converges faithfully on that point and the pads land a
                                # median 0.0953 m from the bottle.
                                #
                                # bottle0 is a cylinder of revolution, so the pad midpoint MUST
                                # sit on its axis; any authored candidate that does not is wrong
                                # for this object regardless of how it was synthesised. Snap the
                                # goal in xy so the gap lands on the live axis, place the gap at
                                # half-height on the mug, and retain the authored orientation.
                                _pad_xy = (goal_pos_w + _pin)[:2]
                                _axis_fix = _now[:2] - _pad_xy
                                goal_pos_w = torch.cat(
                                    [goal_pos_w[:2] + _axis_fix, goal_pos_w[2:]])
                                # Stay strictly inside the physical jaw-contact window. Half
                                # height (0.040 m) is its excluded lower boundary.
                                _jaw_height = mug_grasp_target_height(
                                    float(os.environ.get("SIMVLA_MUG_HEIGHT", "0.08")),
                                    float(os.environ["SIMVLA_MUG_GRASP_HEIGHT"])
                                    if "SIMVLA_MUG_GRASP_HEIGHT" in os.environ else None)
                                goal_pos_w[2] = _now[2] + _jaw_height - _pin[2]
                                if int(env_idx) == 0:
                                    print(f"[track] env0 grasp goal shifted by "
                                          f"({float(_shift[0]):+.4f},{float(_shift[1]):+.4f},"
                                          f"{float(_shift[2]):+.4f}) to follow the object; "
                                          f"gap snapped to the axis by "
                                          f"{float(torch.linalg.norm(_axis_fix)):.4f} m",
                                          flush=True)
                            except Exception as _exc:                   # noqa: BLE001
                                print(f"[track] could not follow the object: "
                                      f"{type(_exc).__name__}: {_exc}", flush=True)
                        # PLAN TO A POINT ABOVE THE STANDOFF, NOT TO THE GRASP POSE.
                        #
                        # cuRobo plans with every obstacle pushed to 1e10 m -- it does not know
                        # the bottle is there -- so asking it for the grasp pose directly lets it
                        # sweep the OPEN jaws in through the bottle on the way. The widened trace
                        # caught exactly that: nearest=gripper_r_rh_p12_rn_l2 at 0.114-0.129 m
                        # while the tilt ran 4.1 -> 89.6 degrees at step=1, and it begins in
                        # phase=PLAN(retry=0) -- before the jog has taken over. The hand's own
                        # finger is what knocks the bottle down, during cuRobo's trajectory.
                        #
                        # Above the STANDOFF rather than above the grasp: from directly over the
                        # bottle the jog's first move is back along the tool axis, and that
                        # descending diagonal clips the bottle's top (its rim is at 1.132 against
                        # a grasp height near 1.065). Over the standoff, the descent is a clean
                        # vertical 0.18 m clear of the bottle, and the existing axial jog does
                        # the last stretch along the tool axis, which is the one direction the
                        # open jaws can travel without striking it.
                        #
                        # _goal_w_r below still stores the TRUE grasp pose, and the reach check
                        # re-resolves ik_goals_r from it every step, so the jog still finishes on
                        # the bottle. Only what cuRobo is asked for changes.
                        _plan_pos_w = goal_pos_w
                        if (int(skill_r_plan[i]) == SID_GRASP
                                and (_right_pregrasp_standoff > 0.0 or _right_pregrasp_above > 0.0)):
                            # ON BY DEFAULT, but only because the standoff can now give up.
                            #
                            # This backoff stops cuRobo sweeping the open jaws through the bottle
                            # on its way to the grasp pose -- it plans with every obstacle at
                            # 1e10 m and does not know the bottle is there. When first added it
                            # took grasps to ZERO across four variations, and the trace said why:
                            # the jog aims at the standoff while its lateral error exceeds
                            # GRASP_LATERAL_TOL, so an arm that cannot quite reach that pose aims
                            # there forever -- fingers straddling the bottle, eef_to_obj pinned at
                            # 0.147 for 70+ retries, tilt climbing 6 to 36 degrees under the
                            # sustained push.
                            #
                            # _STANDOFF_JOG_CAP fixed that, and I disabled the backoff one commit
                            # later without ever testing the two together. Measured side by side
                            # on four jobs each, same build otherwise:
                            #
                            #     with backoff:  miss median 0.0194, 8/15 inside the 0.025 m jaw
                            #                    budget, 53% toppled, grasps 6.7% of closes
                            #     without:       miss median 0.0359, 20/67 inside, 78% toppled,
                            #                    grasps 3.0%
                            #
                            # Better on every per-close statistic. SIMVLA_PREGRASP_STANDOFF=0
                            # turns it off again.
                            _PREGRASP_STANDOFF_M = _right_pregrasp_standoff
                            # THE VERTICAL LIFT IS OFF BY DEFAULT, because it strands the arm.
                            #
                            # Planning 0.12 m ABOVE the standoff measured worse, not better: the
                            # jog aims at the standoff while its lateral error exceeds 0.012 m,
                            # and `lateral` is the component perpendicular to the tool axis --
                            # which a purely vertical offset is entirely made of. The hand
                            # arrives above the standoff and keeps aiming at it instead of the
                            # bottle. Measured: envs sitting at 281 of 300 jog steps still 0.17 m
                            # short, which is the 0.18 m standoff almost exactly, and 0 grasps in
                            # 31 closes against 1 clean grasp in the fleet without it.
                            #
                            # The horizontal backoff is what stops cuRobo sweeping the open jaws
                            # through the bottle, and it is kept. Arriving AT the standoff leaves
                            # the lateral error near zero, so the jog switches to the true grasp
                            # pose immediately and comes in along the tool axis.
                            _PREGRASP_ABOVE_M = _right_pregrasp_above
                            # WHICH SIDE, COMPUTED -- the same correction the jog needed.
                            #
                            # This hardcoded -Z. Where a candidate's tool frame points the other
                            # way that puts the PLAN TARGET beyond the bottle, so cuRobo drives
                            # the hand past the bottle to reach it and crosses it on the way --
                            # strictly worse than planning straight at the grasp pose, which is
                            # what this backoff replaced. Grasps went from one per fleet to none
                            # when the backoff landed, and this is the reason.
                            #
                            # The planner and jog must choose the same side. Use the measured
                            # wrist pose: base_link can be below an elevated grasp and select the
                            # opposite side of a vertical approach axis.
                            _axw = torch.as_tensor(
                                R.from_quat(goal_quat_w.cpu().numpy()[[1, 2, 3, 0]]).apply(
                                    np.array([0.0, 0.0, 1.0])),
                                dtype=goal_pos_w.dtype, device=goal_pos_w.device)
                            _back_sign = choose_pregrasp_back_sign(
                                goal_pos_w.detach().cpu().tolist(),
                                _axw.detach().cpu().tolist(),
                                _eef_now.detach().cpu().tolist(),
                                _PREGRASP_STANDOFF_M,
                            )
                            arm_r_pregrasp_back_sign[env_idx] = _back_sign
                            _bk = _axw * _back_sign
                            _up = torch.zeros_like(goal_pos_w)
                            _up[2] = _PREGRASP_ABOVE_M
                            _plan_pos_w = goal_pos_w + _bk * _PREGRASP_STANDOFF_M + _up
                        goal_ee_pose_b, goal_ee_quat_b = world2base(env, _plan_pos_w, goal_quat_w, env_idx)
                        cu_js = JointState(
                            position=tensor_args.to_device(joint_pos),
                            # Optimized timing assumes a stationary start; otherwise its
                            # finite-difference stencil extrapolates the first waypoint.
                            velocity=tensor_args.to_device(joint_vel) * (0.0 if optimize_dt else 1.0),
                            acceleration=tensor_args.to_device(joint_vel) * 0.0,
                            jerk=tensor_args.to_device(joint_vel) * 0.0,
                            joint_names=r_j_names,
                        )
                        cu_js = cu_js.get_ordered_joint_state(motion_gen_r.kinematics.joint_names)
                        ik_goal = Pose(position=goal_ee_pose_b, quaternion=goal_ee_quat_b)
                        ik_goals_r[env_idx] = ik_goal
                        # THE BASE POSE THIS GOAL IS EXPRESSED AGAINST. ik_goals_r is in the BASE
                        # frame and is captured once, here. This robot's base is a velocity drive
                        # with no position hold, so if it drifts afterwards the same base-frame
                        # goal names a DIFFERENT world point -- while pos_err, measured in that
                        # same drifting frame, keeps reporting convergence. Recording it lets the
                        # close log say whether that happened instead of leaving it inferred.
                        _base_at_plan[env_idx] = robot.data.body_pos_w[env_idx, base_link_idx, :2].clone()
                        # THE OBJECT'S TILT BEFORE THE ARM MOVES. obj_tilt reads 90 degrees at
                        # close in most envs and 0-7 in the ones that succeed, so the bottle is
                        # going over at some point -- but the close reading cannot say WHEN. If
                        # it is upright here and flat at close, the approach knocks it, and the
                        # cause is cuRobo planning with every obstacle pushed to 1e10 m: it does
                        # not know the bottle is there. Anubis survives that because it grasps a
                        # squat mug; a 0.064 x 0.182 bottle does not.
                        try:
                            _ot = env.scene.rigid_objects[args_cli.obj_name]
                            _q = _ot.data.body_quat_w[env_idx, 0]
                            _upz = 1.0 - 2.0 * (float(_q[1]) ** 2 + float(_q[2]) ** 2)
                            _tilt_at_plan[env_idx] = math.degrees(
                                math.acos(max(-1.0, min(1.0, _upz))))
                        except Exception:                               # noqa: BLE001
                            pass
                        # KEEP THE GOAL IN WORLD TOO, so it can be re-resolved against the LIVE
                        # base instead of trusting a base-frame snapshot. Measured: the base
                        # drifts 0.018-0.096 m between planning a grasp and closing on it, and
                        # the pads end 0.048-0.267 m from the bottle while pos_err -- computed in
                        # that same drifting frame -- reports 0.011 m. The arm converges
                        # faithfully on a point that has moved.
                        # THE TRUE GRASP POSE, not the standoff. The jog re-resolves from this
                        # every step, so storing the standoff here would leave the hand parked
                        # 0.09 m short of the bottle and never closing the last stretch.
                        _goal_w_r[env_idx] = (goal_pos_w.clone(), goal_quat_w.clone())
                        _plan_cfg_r = plan_config
                        _coarse_cfg_r = plan_config_no_finetune
                        _offset_r = bounded_float_env("SIMVLA_RIGHT_APPROACH_OFFSET_M", 0.0, -0.30, 0.30)
                        if int(skill_r_plan[i]) == SID_GRASP and abs(_offset_r) > 1e-6:
                            from curobo.rollout.cost.pose_cost import PoseCostMetric
                            _metric_r = PoseCostMetric.create_grasp_approach_metric(
                                offset_position=_offset_r, linear_axis=2, tstep_fraction=0.8,
                                tensor_args=tensor_args)
                            _metric_r.project_to_goal_frame = True
                            _plan_cfg_r = plan_config.clone()
                            _plan_cfg_r.pose_cost_metric = _metric_r
                            _coarse_cfg_r = plan_config_no_finetune.clone()
                            _coarse_cfg_r.pose_cost_metric = _metric_r.clone()
                        result = motion_gen_r.plan_single(cu_js.unsqueeze(0), ik_goals_r[env_idx], _plan_cfg_r)
                        if (not result.success.item() and int(skill_r_plan[i]) == SID_RESET
                                and env_idx not in _postrelease_reset_envs):
                            # JOINT-SPACE FALLBACK FOR arm.reset ONLY. The goal is literally the
                            # configuration this arm started in, which is recorded alongside the
                            # pose, so ask for it directly instead of routing a folded torso-side
                            # pose back through IK from a lifted carrying pose.
                            # ALL FOUR FIELDS, not just position. cuRobo copies the goal into a
                            # preallocated buffer (rollout_base.copy_) and reads velocity,
                            # acceleration and jerk off it unconditionally; a JointState built
                            # with position alone leaves those None and the copy dies with
                            # AttributeError: 'NoneType' object has no attribute 'shape'.
                            # cu_js above passes zeros for exactly this reason. Separate tensors
                            # rather than one shared zero tensor, so nothing aliases.
                            _home_js = (episode_home_r_js[env_idx]
                                        if env_idx in _final_home_reset_envs
                                        else reset_r_js[env_idx])
                            _jsp = tensor_args.to_device(_home_js.tolist())
                            _gjs = JointState(
                                position=_jsp,
                                velocity=_jsp * 0.0,
                                acceleration=_jsp * 0.0,
                                jerk=_jsp * 0.0,
                                joint_names=r_j_names,
                            ).get_ordered_joint_state(motion_gen_r.kinematics.joint_names)
                            result = motion_gen_r.plan_single_js(cu_js.unsqueeze(0),
                                                                 _gjs.unsqueeze(0),
                                                                 plan_config_graph)
                            if result.success.item():
                                print(f"[plan] Env {env_idx}: arm.reset planned in joint space",
                                      flush=True)
                        if (not result.success.item()
                                and "FINETUNE_TRAJOPT" in str(getattr(result, "status", ""))):
                            # Only this status, and only once: the coarse trajectory exists and
                            # is usable, so take it rather than discard a grasp that succeeded.
                            result = motion_gen_r.plan_single(cu_js.unsqueeze(0),
                                                              ik_goals_r[env_idx],
                                                              _coarse_cfg_r)
                            if result.success.item():
                                print(f"[plan] Env {env_idx}: finetune trajopt failed; planned "
                                      f"without it", flush=True)
                        if result.success.item():
                            cmd_plan = result.get_interpolated_plan()
                            cmd_plan = motion_gen_r.get_full_js(cmd_plan)
                            # The joint names in the plan can differ from the robot's joint names
                            plan_j_names = cmd_plan.joint_names
                            # Filter r_j_names to only include joints present in the plan
                            common_j_names = [name for name in r_j_names if name in plan_j_names]
                            env_cmd_plans_r[env_idx] = cmd_plan.get_ordered_joint_state(common_j_names)
                            env_cmd_indices_r[env_idx] = 0
                            env_cmd_track_stall_r[env_idx] = 0
                            _start_pos_r, _start_quat_r = world2base(
                                env, robot.data.body_pos_w[env_idx, r_eef_idx],
                                robot.data.body_quat_w[env_idx, r_eef_idx], env_idx)
                            if int(skill_r_plan[i]) == SID_GRASP:
                                _active_start_r = cu_js.get_ordered_joint_state(
                                    motion_gen_r.kinematics.joint_names).position.unsqueeze(0)
                                _fk_start_r = motion_gen_r.kinematics.forward(
                                    _active_start_r, link_name="ee_link1")[0].squeeze().clone()
                                print(f"[plan_r] Env {env_idx}: start FK/Isaac error="
                                      f"{float(torch.linalg.norm(_fk_start_r - _start_pos_r.squeeze())):.6f}m "
                                      f"base_pos_w={robot.data.body_pos_w[env_idx, base_link_idx].tolist()} "
                                      f"base_quat_wxyz={robot.data.body_quat_w[env_idx, base_link_idx].tolist()} "
                                      f"locked_joints={r_robot_cfg['kinematics'].get('lock_joints', {})}", flush=True)
                            anchor_relative_ik_target(
                                env.action_manager.get_term("armR_action"), env_idx,
                                _start_pos_r.squeeze(), _start_quat_r.squeeze())
                            if int(skill_r_plan[i]) == SID_GRASP and env_idx not in _direct_base_anchors:
                                _park_pos = robot.data.body_pos_w[env_idx, base_link_idx]
                                _park_yaw = euler_xyz_from_quat(
                                    robot.data.body_quat_w[env_idx:env_idx + 1, base_link_idx])[2][0]
                                _direct_base_anchors[env_idx] = (
                                    float(_park_pos[0]), float(_park_pos[1]), float(_park_yaw))
                        elif ALLOW_CARTESIAN_FALLBACK and int(skill_r_plan[i]) in (SID_RESET, SID_PLACE):
                            # JOG INSTEAD OF SKIPPING, BECAUSE SUCCESS REQUIRES BOTH OF THESE.
                            #
                            # Skipping unblocked the episode and simultaneously guaranteed it
                            # could never be scored: the task's success spec is all of
                            # obj_near_prim, obj_z, last_subtask AND eef_home (right arm within
                            # 0.18 m of home). An arm that never folds back fails the last one,
                            # so the bottle can be placed perfectly and the run still records
                            # nothing.
                            #
                            # cuRobo will not plan this motion by any route (FINETUNE_TRAJOPT_FAIL,
                            # TRAJOPT_FAIL, TRAJOPT_FAIL in joint space, TRAJOPT_FAIL with the
                            # graph planner), but the Cartesian jog does not need a planner and
                            # already closes the reach. Hand it a two-sample HOLD plan at the
                            # current joint state: the executor sees a plan that finishes
                            # immediately, measures the error against ik_goals_r -- which is the
                            # goal for this step -- and jogs the rest of the way.
                            #
                            # arm.place NEEDS THIS TOO, and its failure is not trajopt but
                            # INVALID_START_STATE_SELF_COLLISION: cuRobo judges the carrying pose
                            # -- arm folded in, bottle held -- to be in self-collision and
                            # refuses before it plans anything. That is a refusal to START, so no
                            # amount of planner tuning reaches it, and every episode that got the
                            # bottle off the counter and drove it to the table died there. It is
                            # the last step between a working grasp and a recorded demonstration.
                            #
                            # The place target is about 0.47 m away, which is 188 jog steps of
                            # 0.0025 m against a 300-step budget, and the path from the carrying
                            # pose to over the table is free space.
                            _p = (cu_js.position if cu_js.position.dim() == 1
                                  else cu_js.position.squeeze(0))
                            _pp = torch.stack([_p, _p])
                            env_cmd_plans_r[env_idx] = JointState(
                                position=_pp, velocity=_pp * 0.0,
                                acceleration=_pp * 0.0, jerk=_pp * 0.0,
                                joint_names=list(motion_gen_r.kinematics.joint_names),
                            )
                            env_cmd_indices_r[env_idx] = 0
                            env_cmd_pause_trig_r[env_idx] = True   # no mid-plan pause on a hold
                            env_cmd_pause_r[env_idx] = 0
                            arm_r_retry[env_idx] = 0
                            _un = "arm.place" if int(skill_r_plan[i]) == SID_PLACE else "arm.reset"
                            print(f"[home] Env {env_idx}: {_un} unplannable "
                                  f"({getattr(result, 'status', None)}); jogging instead",
                                  flush=True)
                        else:
                            idx = torch.tensor([env_idx], device=device)
                            # SAY WHICH STEP, AND TO WHERE. "plan_fail_r" alone does not
                            # distinguish an unreachable place target from a grasp that cuRobo
                            # cannot leave, and the two want completely different fixes. The step
                            # index and skill id are already to hand here.
                            _si = int(env_goal_indices[env_idx])
                            _sk = int(skill_ids_tensor[env_idx, _si])
                            _skname = next((k for k, v in skill_ids.items() if v == _sk), _sk)
                            _gp = goal_ee_pose_b.squeeze()
                            # cuRobo'S OWN REASON, AND WHERE THE HAND WAS. "plan_fail_r" does not
                            # distinguish a start state colliding with the world from a trajopt
                            # that would not converge, and those want opposite fixes -- the first
                            # is the post-grasp lift, the second is time or seeds. result.status
                            # carries it and was being discarded. The eef height says whether the
                            # lift that is supposed to clear the counter actually ran.
                            _st = getattr(result, "status", None)
                            _ez = float(robot.data.body_pos_w[env_idx, r_eef_idx, 2]
                                        - env.scene.env_origins[env_idx, 2])
                            try:
                                from planner_diagnostics import failure_snapshot
                                print("[planner-start] " + failure_snapshot(
                                    motion_gen_r.kinematics, cu_js, arm="r", step=_si,
                                    status=_st), flush=True)
                            except Exception as diagnostic_error:
                                print(f"[planner-start] diagnostic unavailable: {diagnostic_error}", flush=True)
                            _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,  env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, idx, reason=(
                                f"plan_fail_r at step {_si} ({_skname}), status={_st!r}, "
                                f"eef_z={_ez:.4f}, "
                                f"goal_base=({float(_gp[0]):+.4f},{float(_gp[1]):+.4f},"
                                f"{float(_gp[2]):+.4f})"))
                paused_mask = env_cmd_pause_r[arm_r_indices] > 0
                if torch.any(paused_mask):
                    paused_envs = arm_r_indices[paused_mask]
                    env_cmd_pause_r[paused_envs] -= 1  # hold pose_R, do not advance cmd idx while paused
                has_plan_mask = torch.tensor(
                    [env_cmd_plans_r[i.item()] is not None for i in arm_r_indices],
                    device=device
                )

                not_paused_mask = (env_cmd_pause_r[arm_r_indices] == 0)
                can_run_mask = has_plan_mask & not_paused_mask

                if torch.any(can_run_mask):
                    executing_indices = arm_r_indices[can_run_mask]                  # env ids that can execute now
                    indices_to_exec = env_cmd_indices_r[executing_indices]           # per-env command pointer

                    plans_to_exec_r = [env_cmd_plans_r[i.item()] for i in executing_indices]
                    plan_lengths = torch.tensor([len(p) for p in plans_to_exec_r], device=device, dtype=torch.long)

                    pause_at = torch.clamp((plan_lengths.float() * 0.5).floor().long(), min=1)

                    already_trig = env_cmd_pause_trig_r[executing_indices]
                    should_pause = (~already_trig) & (indices_to_exec >= pause_at)

                    if torch.any(should_pause):
                        pause_envs = executing_indices[should_pause]
                        env_cmd_pause_r[pause_envs] = args_cli.action_chunk_size + 1
                        env_cmd_pause_trig_r[pause_envs] = True

                        # remove paused envs from THIS timestep’s execution (so they hold pose_R immediately)
                        keep = ~should_pause
                        executing_indices = executing_indices[keep]
                        indices_to_exec = indices_to_exec[keep]
                        plans_to_exec_r = [p for p, k in zip(plans_to_exec_r, keep.tolist()) if k]
                        plan_lengths = plan_lengths[keep]

                    if executing_indices.numel() > 0:
                        delta_poses = compute_eef_deltas_batched(
                            motion_gen=motion_gen_r,
                            plans=plans_to_exec_r,
                            cmd_indices=indices_to_exec,
                            joint_names=common_j_names,
                            eef_link_name="ee_link1",
                        )

                        pose_R[executing_indices] = delta_poses
                        _advance_r = torch.ones(len(executing_indices), dtype=torch.bool, device=device)
                        for _row_r, _env_r in enumerate(executing_indices.tolist()):
                            if int(skill_ids_tensor[_env_r, int(env_goal_indices[_env_r])]) != SID_GRASP:
                                continue
                            if int(arm_r_retry[_env_r]) > 0:
                                continue  # A finished plan is already in its bounded reach gate.
                            _controller_r = env.action_manager.get_term("armR_action")._ik_controller
                            _actual_p_r, _actual_q_r = world2base(
                                env, robot.data.body_pos_w[_env_r, r_eef_idx],
                                robot.data.body_quat_w[_env_r, r_eef_idx], _env_r)
                            _pos_error_r = float(torch.linalg.norm(
                                _actual_p_r.squeeze() - _controller_r.ee_pos_des[_env_r]))
                            _qa_r = R.from_quat(_actual_q_r.squeeze().cpu().numpy()[[1, 2, 3, 0]])
                            _qt_r = R.from_quat(_controller_r.ee_quat_des[_env_r].cpu().numpy()[[1, 2, 3, 0]])
                            _rot_error_r = math.degrees(float(np.linalg.norm((_qt_r.inv() * _qa_r).as_rotvec())))
                            if _pos_error_r > _right_track_pos_tol or _rot_error_r > _right_track_rot_tol:
                                _advance_r[_row_r] = False
                                pose_R[_env_r] = 0.0
                                env_cmd_track_stall_r[_env_r] += 1
                                if int(env_cmd_track_stall_r[_env_r]) in (1, 20, 40):
                                    print(f"[track_r] Env {_env_r}: waiting at sample "
                                          f"{int(indices_to_exec[_row_r])}/{len(plans_to_exec_r[_row_r])}; "
                                          f"error={_pos_error_r:.4f}m/{_rot_error_r:.1f}deg", flush=True)
                                if int(env_cmd_track_stall_r[_env_r]) >= _direct_track_timeout_l:
                                    _active_names_r = list(motion_gen_r.kinematics.joint_names)
                                    _active_ids_r = [robot.joint_names.index(name) for name in _active_names_r]
                                    _planned_q_r = plans_to_exec_r[_row_r].get_ordered_joint_state(
                                        _active_names_r)[int(indices_to_exec[_row_r])].position
                                    print("[track-stall-joints] " + json.dumps({
                                        "arm": "r", "env_id": _env_r,
                                        "joint_names": _active_names_r,
                                        "actual": robot.data.joint_pos[_env_r, _active_ids_r].tolist(),
                                        "planned": _planned_q_r.tolist(),
                                        "limits": robot.data.soft_joint_pos_limits[_env_r, _active_ids_r].tolist(),
                                    }), flush=True)
                                    env_cmd_indices_r[_env_r] = len(plans_to_exec_r[_row_r])
                                    print(f"[track_r] Env {_env_r}: waypoint timeout; "
                                          "rejecting the stalled trajectory", flush=True)
                            else:
                                env_cmd_track_stall_r[_env_r] = 0
                        env_cmd_indices_r[executing_indices[_advance_r]] += 1

                        finished_exec_mask = (env_cmd_indices_r[executing_indices] >= plan_lengths)

                        # Check if the current EEF is far from the goal
                        # THE GATE HAS TO BE TIGHTER THAN THE OBJECT IS WIDE.
                        #
                        # 0.05 m is fine for setting something down and hopeless for picking
                        # something up: the bottle is 0.064 m across, so a hand that stops 0.047 m
                        # from the goal -- which is what every [OK] line reported, right at the
                        # gate -- closes its jaws beside it. Measured consequence: the arm lifted
                        # 0.25-0.35 m and the bottle rose by 0.0000, in every env of every run,
                        # while the whole sequence otherwise ran correctly.
                        #
                        # The jog closes the residual geometrically and has 300 steps of 0.0025 m
                        # to spend, so a tight gate costs a few extra steps and nothing else. It
                        # is tightened only where it matters -- reaching TO grasp -- because a
                        # place genuinely does not need millimetres.
                        rot_threshold_deg = 10
                        #: Correction STEPS allowed at one goal before the episode is written
                        #: off, and the size of one step. 0.0025 m at the 20 Hz env rate is
                        #: 0.05 m/s, the pace the plan tracker already drives this arm at, so
                        #: 120 steps closes 0.30 m -- far more than the 0.07-0.16 m residuals
                        #: measured. Bounded, so an unreachable goal still fails in finite time.
                        # 300 steps of 0.0025 m is 0.75 m of authority. The reach at the counter
                        # needs 2-97 of them, but the retreat home is a 0.4-0.5 m motion and the
                        # jog is now what performs it, so the budget has to cover that too. It
                        # only runs while the error is outside the gate, so a generous ceiling
                        # costs nothing on the short corrections.
                        # REACH_RETRIES / REACH_STEP_M / REACH_STEP_RAD are MODULE-scope (see
                        # top of file). They must NOT be assigned here: an assignment anywhere in
                        # main() rebinds them as function-locals for all of main, which is what
                        # made the left arm's jog raise UnboundLocalError.
                        #: How square the hand must be to the grasp before the jaws may shut.
                        #:
                        #: The jog ALREADY corrects orientation -- pose_R[3:6] drives the rotation
                        #: vector toward the goal at REACH_STEP_RAD per step -- but its loop
                        #: condition only ever looked at pos_err, so the moment position converged
                        #: the jog stopped and whatever rotation error remained was simply kept.
                        #: rot_threshold_deg was assigned and never compared; the equivalent
                        #: checks at the both-arms site are commented out.
                        #:
                        #: THE ARITHMETIC. The fingers sit about 0.09 m from ee_link1, so an
                        #: orientation error theta displaces the pad midpoint by 0.09*theta. The
                        #: reaches that "succeed" report pos_err 0.012 m and rot_err up to 11.9
                        #: deg, and 0.09 * 11.9 deg = 0.0187 m. Add the position tolerance and
                        #: that is 0.031 m -- which is MISS_AT_CLOSE's median of 0.033 almost
                        #: exactly, against a 0.025 m jaw budget. The missing constraint is here.
                        #:
                        #: 5 degrees costs 0.0079 m of pad swing, leaving room inside the budget.
                        #: OVERRIDABLE, because the budget these two share is a property of the
                        #: HANDLE, not of the robot. The pad midpoint's total error is
                        #: pos_tol + 0.09 * rot_tol, and what it has to fit inside is
                        #: (jaw half-opening 0.040) - (half the handle's thickness along the jaw
                        #: axis):
                        #:
                        #:   cabinet door bar   0.024 m thick -> budget 0.028 -> 0.020 fits
                        #:   dishwasher bar     0.050 m thick -> budget 0.015 -> 0.020 DOES NOT
                        #:
                        #: Measured on kitchen 1221 across four runs: the hand arrived 0.012 m from
                        #: the authored pose -- in tolerance -- and the jaws closed to 7e-06 m on a
                        #: 0.050 m bar, every episode. Nothing was wrong with the pose; the gate was
                        #: simply looser than that handle can afford.
                        #:
                        #: Tightening globally would change every task that works today, so it is an
                        #: env var and the default is what has always been used. The jog has 300
                        #: steps of 0.0025 m to spend, so a tighter gate costs steps, not feasibility.
                        GRASP_ROT_TOL_DEG = float(
                            _simvla_os.environ.get("SIMVLA_GRASP_ROT_TOL_DEG", "5.0"))
                        #: Position tolerance when the step being executed is a GRASP.
                        #:
                        #: 0.05 m is fine for setting something down and hopeless for picking
                        #: something up: this bottle is 0.064 m across, so a hand that stops
                        #: 0.047 m from the goal -- which is what every [OK] line reported, sitting
                        #: right at the old gate -- closes its jaws beside it. Measured: the arm
                        #: lifted 0.25-0.35 m and the bottle rose by 0.0000, in every env of every
                        #: run, while the rest of the sequence ran correctly.
                        #:
                        #: 0.012 m is under a fifth of the bottle's width. The jog closes the
                        #: residual geometrically with 300 steps of 0.0025 m to spend, so a tight
                        #: gate costs a few extra steps and nothing else.
                        GRASP_POS_TOL = float(
                            _simvla_os.environ.get("SIMVLA_GRASP_POS_TOL", "0.012"))
                        #: Pre-grasp standoff along the hand's approach axis, and how close the
                        #: LATERAL error must be before the hand is allowed to close in. Without
                        #: this the jog crosses the object instead of approaching it.
                        # Reuse the planned distance. Object/finger clearance is task-specific;
                        # do not insert the historical bottle's 18 cm retreat for every mug.
                        GRASP_STANDOFF_M = _right_pregrasp_standoff if _cylindrical_mug_grasp_r else 0.0
                        # Physical pre-close validation permits 25 mm jaw-to-axis error. Align
                        # within 20 mm before switching from the standoff to the final approach,
                        # leaving 5 mm of margin while the live mug pose moves slightly.
                        GRASP_LATERAL_TOL = 0.020
                        

                        for j, env_idx in enumerate(executing_indices[finished_exec_mask].tolist()):
                            # PER ENV, because it depends on which skill this env is executing:
                            # each one is at its own step. Reaching TO GRASP must be tighter than
                            # the object is wide; everything else keeps the old tolerance.
                            _si_now = int(env_goal_indices[env_idx])
                            _sk_now = int(skill_ids_tensor[env_idx, _si_now])
                            # A REACH IS A GRASP IF THE NEXT STEP CLOSES THE GRIPPER.
                            #
                            # The skill-id test alone was not firing: pos_err at [OK] came back
                            # with a median of 0.0484 and a max of 0.0497, which is the 0.05
                            # tolerance, not the 0.012 one -- so the jog was stopping 5 cm out and
                            # the pads landed ~10 cm from the bottle. Whatever the id mismatch is,
                            # "the very next step shuts the jaws" is a property of the task that
                            # cannot drift, so it decides this instead.
                            _next_is_close = False
                            try:
                                if _si_now + 1 < skill_ids_tensor.shape[1]:
                                    _nxt = int(skill_ids_tensor[env_idx, _si_now + 1])
                                    _next_is_close = (_nxt == skill_ids["gripper.set"]
                                                      and bool(payloads_tensor[env_idx,
                                                                               _si_now + 1, 0] > 0))
                            except Exception:                           # noqa: BLE001
                                pass
                            _is_grasp_reach = (_sk_now == SID_GRASP) or _next_is_close
                            pos_threshold = GRASP_POS_TOL if _is_grasp_reach else 0.05
                            pos_threshold = mug_grasp_position_tolerance(
                                args_cli.obj_name, pos_threshold, _next_is_close)
                            # A GRASP IS READY WHEN THE BOTTLE IS BETWEEN THE JAWS, not when
                            # ee_link1 has reached a commanded depth.
                            #
                            # pos_err is a proxy, and it fails in exactly the case that matters:
                            # the trace shows the fingers straddling the bottle -- l2 and r2
                            # alternating as the nearest body -- while eef_to_obj stays pinned at
                            # 0.147 for 70+ jog retries because the hand is blocked, and the tilt
                            # climbs 6 to 36 degrees under the sustained push. The hand was in a
                            # grasping position the whole time and the code would not let it
                            # close, so it kept driving and pushed the bottle over instead.
                            #
                            # So test the thing itself: the pad midpoint within the jaw's lateral
                            # budget of the bottle's axis, and at a height that is ON the bottle.
                            # Jaws half-span 0.057 and the bottle radius is 0.032, so 0.020 keeps
                            # a 5 mm margin inside the 0.025 m budget. The pinned public mug is
                            # 0.086 m tall; the gate stops before its rim so the pads contact
                            # its cylindrical body rather than sliding off during lift.
                            _accept_now = False
                            _mug_shift_during_approach_r = 0.0
                            _lat_miss = float("nan")
                            _pad_h = float("nan")
                            if _is_grasp_reach and _cylindrical_mug_grasp_r:
                                try:
                                    _gr_i, _gl_i = jaw_body_indices(robot, "right")
                                    _pmw = jaw_contact_midpoint(robot, env_idx, _gr_i, _gl_i)
                                    _mug_obj_r = env.scene.rigid_objects[args_cli.obj_name]
                                    _obw = _mug_obj_r.data.body_pos_w[env_idx, 0]
                                    _spawn_r = (_mug_obj_r.data.default_root_state[env_idx, :3]
                                                + env.scene.env_origins[env_idx])
                                    _mug_shift_during_approach_r = float(torch.linalg.norm(
                                        _obw[:2] - _spawn_r[:2]))
                                    _lat_miss = float(torch.linalg.norm(_pmw[:2] - _obw[:2]))
                                    _pad_h = float(_pmw[2] - _obw[2])
                                    if (_mug_shift_during_approach_r <= 0.03
                                            and mug_jaw_contact_ready(_lat_miss, _pad_h)):
                                        if int(env_idx) == 0:
                                            print(f"[close] Env {env_idx}: jaws are on the bottle "
                                                  f"(lateral {_lat_miss:.4f} m, pad {_pad_h:.4f} m "
                                                  f"up); closing without waiting for the commanded "
                                                  f"depth", flush=True)
                                        pos_threshold = 9.9      # accept this reach now
                                        _accept_now = True
                                except Exception:                           # noqa: BLE001
                                    pass
                            # RE-RESOLVE THE GOAL AGAINST THE BASE AS IT IS NOW. ik_goals_r was
                            # captured in the base frame at plan time; the base has moved since,
                            # so that snapshot names a different world point and every error
                            # measured against it is measured in the wrong frame.
                            _gw = _goal_w_r.get(env_idx)
                            # KEEP THE PHYSICAL GAP ON THE AXIS AS THE MUG MOVES, not only at plan time.
                            #
                            # The plan-time snap puts the pad midpoint exactly on the bottle's
                            # axis, and then the bottle drifts: even the successful grasp of this
                            # fleet closed with GOAL_TO_OBJ=0.0168, and the fleet's median miss is
                            # 0.0234 against a 0.020 m acceptance. That residual IS the drift --
                            # the goal is a world snapshot and the object is not.
                            #
                            # Only while it is still standing. Past 15 degrees it is on its way
                            # over, its axis is no longer where a grasp authored for an upright
                            # bottle wants to be, and chasing it would drive the hand after a
                            # falling object and push it further.
                            if _is_grasp_reach and _cylindrical_mug_grasp_r and _gw is not None:
                                try:
                                    _obr = env.scene.rigid_objects[args_cli.obj_name]
                                    _oqr = _obr.data.body_quat_w[env_idx, 0]
                                    _uz = 1.0 - 2.0 * (float(_oqr[1]) ** 2 + float(_oqr[2]) ** 2)
                                    if math.degrees(math.acos(max(-1.0, min(1.0, _uz)))) < 15.0:
                                        _gpw, _gqw = _gw
                                        _jr_live, _jl_live = jaw_body_indices(robot, "right")
                                        _jaw_mid_live = jaw_contact_midpoint(robot, env_idx, _jr_live, _jl_live)
                                        _eef_pos_live = robot.data.body_pos_w[
                                            env_idx, r_eef_idx]
                                        _eef_q_live = R.from_quat(
                                            robot.data.body_quat_w[
                                                env_idx, r_eef_idx].cpu().numpy()[[1, 2, 3, 0]])
                                        _jaw_offset_live = _eef_q_live.inv().apply(
                                            (_jaw_mid_live - _eef_pos_live).cpu().numpy())
                                        _pw = torch.as_tensor(
                                            R.from_quat(_gqw.cpu().numpy()[[1, 2, 3, 0]]).apply(
                                                _jaw_offset_live),
                                            dtype=_gpw.dtype, device=_gpw.device)
                                        _fix = (_obr.data.body_pos_w[env_idx, 0][:2]
                                                - (_gpw + _pw)[:2])
                                        # The open fingers may move relative to the wrist after
                                        # planning. XY is already re-centred above; opt in to
                                        # tracking the same measured offset in Z as well. This
                                        # keeps the physical pads at the mug's mid-body height
                                        # instead of jogging forever at a wrist goal that is
                                        # position-perfect but above/below the contact window.
                                        if os.environ.get("SIMVLA_MUG_LIVE_Z_TRACK", "0") == "1":
                                            _height = mug_grasp_target_height(
                                                float(os.environ.get("SIMVLA_MUG_HEIGHT", "0.08")),
                                                float(os.environ["SIMVLA_MUG_GRASP_HEIGHT"])
                                                if "SIMVLA_MUG_GRASP_HEIGHT" in os.environ else None)
                                            _z = (_obr.data.body_pos_w[env_idx, 0, 2]
                                                  + _height - _pw[2])
                                            _gpw = torch.cat([_gpw[:2] + _fix, _z.unsqueeze(0)])
                                        else:
                                            _gpw = torch.cat([_gpw[:2] + _fix, _gpw[2:]])
                                        _goal_w_r[env_idx] = (_gpw, _gqw)
                                        _gw = (_gpw, _gqw)
                                except Exception:                           # noqa: BLE001
                                    pass
                            if _gw is not None:
                                _gp_b, _gq_b = world2base(env, _gw[0], _gw[1], env_idx)
                                ik_goals_r[env_idx] = Pose(position=_gp_b, quaternion=_gq_b)
                            curr_eef_pos, curr_eef_quat = world2base(env, env.scene.articulations['robot'].data.body_pos_w[env_idx,r_eef_idx], env.scene.articulations['robot'].data.body_quat_w[env_idx,r_eef_idx],env_idx)
                            pos_err = torch.norm(curr_eef_pos - ik_goals_r[env_idx].position)
                            r_eef = R.from_quat(curr_eef_quat.cpu().numpy()[[1, 2, 3, 0]])
                            r_goal = R.from_quat(ik_goals_r[env_idx].quaternion.squeeze(0).cpu().numpy()[[1, 2, 3, 0]])
                            rot_err_deg = np.linalg.norm((r_goal.inv() * r_eef).as_rotvec()) * 180 / np.pi
                            # Rotation matters at a grasp: a hand that is position-aligned but
                            # 180 degrees flipped must not be marked reached and close its jaws.
                            _rot_out = (_is_grasp_reach and not _accept_now
                                        and float(rot_err_deg) > GRASP_ROT_TOL_DEG)
                            # A wrist can enter its 4 cm tolerance while the open jaws are
                            # still beside or above the mug. For a calibrated cylindrical
                            # grasp, physical jaw placement is required before closing.
                            _mug_jaws_not_ready_r = (
                                _is_grasp_reach and _cylindrical_mug_grasp_r
                                and not _accept_now)
                            _tracking_failed_r = bool(
                                _is_grasp_reach and
                                env_cmd_track_stall_r[env_idx] >= _direct_track_timeout_l)
                            if _tracking_failed_r:
                                # Do not turn an aborted mid-trajectory plan into an
                                # unchecked Cartesian jog through its remaining path.
                                arm_r_retry[env_idx] = REACH_RETRIES
                            if _mug_shift_during_approach_r > 0.03:
                                print(f"[grasp_r] Env {env_idx}: mug moved "
                                      f"{_mug_shift_during_approach_r:.4f}m during approach; "
                                      "rejecting this candidate", flush=True)
                                arm_r_retry[env_idx] = REACH_RETRIES
                            if ((pos_err > pos_threshold or _rot_out or _mug_jaws_not_ready_r or _tracking_failed_r)
                                    and arm_r_retry[env_idx] < REACH_RETRIES):
                                # CLOSE THE LOOP WITHOUT THE PLANNER. Relative actions advance an
                                # IK reference; the differential controller tracks that reference,
                                # but actuators can lag. The jog anchors at the measured pose once,
                                # then advances the persistent reference toward the goal.
                                #
                                # The first version of this re-planned with cuRobo from the
                                # arm's current state. That does converge -- measured
                                # 0.0635 -> 0.0320, 0.0655 -> 0.0282, 0.0835 -> 0.0433 m -- but
                                # the new start state is a hand 7-15 cm from the bottle and
                                # inches over the counter, and cuRobo refuses to plan from it:
                                # "plan_fail_r at step 1 (arm.grasp)", which resets the episode
                                # and throws away a reach that had all but succeeded.
                                #
                                # So the correction is a straight Cartesian jog through the same
                                # relative-IK channel the plan waypoints drive -- the mechanism
                                # the post-grasp lift already uses for the same reason. The
                                # residual is small and the path is free space just above the
                                # counter, so no planner is needed and none can refuse it.
                                #
                                # Rewinding the command pointer by one leaves the plan holding
                                # its last waypoint (delta zero) and makes this check fire again
                                # next step, so each env step contributes exactly one correction
                                # step and the budget below is in steps, not re-plans.
                                arm_r_retry[env_idx] += 1
                                _plen = len(env_cmd_plans_r[env_idx])
                                _aim = ik_goals_r[env_idx].position.squeeze()
                                # THE STANDOFF HAS TO TIME OUT, OR IT DEADLOCKS.
                                #
                                # The jog aims at the pre-grasp standoff for as long as its
                                # lateral error exceeds 0.012 m -- but if the arm cannot actually
                                # achieve that pose, the error never drops, and it aims at the
                                # standoff forever. Measured: envs at 241 and 261 of 300 jog
                                # steps still 0.18-0.20 m short, which is the 0.18 m standoff
                                # almost exactly, while their neighbours converged to 0.012 m in
                                # 103-209 steps. Half the fleet parked and never closed.
                                #
                                # 80 steps is 0.20 m of commanded travel, more than the 0.18 m
                                # standoff itself, so an arm that is going to get there has got
                                # there. After that, aim at the bottle directly: a non-axial
                                # approach is the behaviour that existed before the standoff and
                                # it did produce grasps, whereas a parked arm produces nothing.
                                _STANDOFF_JOG_CAP = 80
                                if (_sk_now == SID_GRASP and _cylindrical_mug_grasp_r
                                        and arm_r_retry[env_idx] > _STANDOFF_JOG_CAP):
                                    if int(env_idx) == 0 and arm_r_retry[env_idx] == _STANDOFF_JOG_CAP + 1:
                                        print(f"[close] Env {env_idx}: standoff not reached in "
                                              f"{_STANDOFF_JOG_CAP} jog steps; aiming at the "
                                              f"bottle directly", flush=True)
                                elif _sk_now == SID_GRASP and _cylindrical_mug_grasp_r:
                                    # APPROACH ALONG THE TOOL AXIS, NOT STRAIGHT AT THE GOAL.
                                    #
                                    # The plan's shortfall is consistently lateral -- measured
                                    # x arriving at 1.00 of command while y reached 0.79 -- so a
                                    # straight line from where the plan stopped to the grasp pose
                                    # travels ACROSS the bottle and sweeps a pad through it. That
                                    # is what the logs show: the jaws shut to 0.0102 m on a
                                    # 0.064 m bottle (nothing between them) while the bottle
                                    # itself had been knocked up to 0.03 m out of place before
                                    # they closed.
                                    #
                                    # So aim first at a pre-grasp point backed off along the
                                    # hand's own approach axis, and only close in once the
                                    # lateral error is small. The pads measure at ee_link1's -Z
                                    # (scripts/tools/measure_aiworker_pinch.py), so the hand
                                    # extends along -Z and backing off is +Z in the goal frame.
                                    _gq = ik_goals_r[env_idx].quaternion.squeeze(0)
                                    _gR = R.from_quat(_gq.cpu().numpy()[[1, 2, 3, 0]])
                                    # -Z, NOT +Z. The pre-grasp point must sit BETWEEN the robot
                                    # and the object; with +Z it landed beyond the bottle
                                    # (measured: 0.5365 m from the parked base against the goal's
                                    # 0.4602 m), so the jog drove the hand PAST the object to
                                    # reach it and swept the left finger through it on the way.
                                    # The trace caught exactly that -- nearest=l2 closing from
                                    # 0.103 m to 0.078 m while the tilt ran 2.6 to 85.6 degrees,
                                    # all of it in phase=JOG.
                                    #
                                    # The earlier +Z came from the pads measuring at
                                    # ee_link1 + (0, 0, -0.0235). That offset is real but far too
                                    # small to establish an approach direction -- the fingers
                                    # reach roughly 0.09 m further -- and it is not what decides
                                    # which side the standoff belongs on. Which side is nearer
                                    # the robot is.
                                    # PICK THE SIDE, DO NOT ASSUME IT.
                                    #
                                    # This hardcoded -Z while the comment above says the side
                                    # nearer the robot is what decides. Those agree only while
                                    # every grasp candidate's tool frame happens to point the
                                    # same way, and this corpus spans many kitchens and
                                    # rotations. Where they disagree the standoff lands on the
                                    # FAR side of the bottle, so the axial approach starts beyond
                                    # it and comes through it -- which topples it exactly as the
                                    # unbacked-off plan did, and leaves the close looking like a
                                    # near miss because the pads do arrive near the axis.
                                    #
                                    # These are base-frame positions. Choose the candidate nearest
                                    # the measured wrist so the open-finger alignment begins on
                                    # the side the arm is actually approaching from.
                                    _ax = torch.as_tensor(
                                        _gR.apply(np.array([0.0, 0.0, 1.0])),
                                        dtype=_aim.dtype, device=_aim.device)
                                    _current_pre = curr_eef_pos.squeeze()
                                    if int(arm_r_pregrasp_back_sign[env_idx]) == 0:
                                        arm_r_pregrasp_back_sign[env_idx] = (
                                            choose_pregrasp_back_sign(
                                                _aim.detach().cpu().tolist(),
                                                _ax.detach().cpu().tolist(),
                                                _current_pre.detach().cpu().tolist(),
                                                GRASP_STANDOFF_M,
                                            )
                                        )
                                    _back = _ax * int(arm_r_pregrasp_back_sign[env_idx])
                                    _pre = _aim + _back * GRASP_STANDOFF_M
                                    _lat = float(torch.linalg.norm(
                                        (curr_eef_pos.squeeze() - _pre)
                                        - _back * torch.dot(curr_eef_pos.squeeze() - _pre, _back)))
                                    if _lat > GRASP_LATERAL_TOL:
                                        _aim = _pre
                                _ik_term_r = env.action_manager.get_term("armR_action")
                                if int(arm_r_retry[env_idx]) == 1:
                                    # Start one reference trajectory at the measured pose. On later
                                    # frames, keep advancing that reference even if the physical
                                    # arm lags; rebasing every frame made a 2.5 mm command disappear
                                    # into actuator lag and repeatedly recomputed rotation from a
                                    # pose that had barely moved.
                                    anchor_relative_ik_target(
                                        _ik_term_r, env_idx, curr_eef_pos.squeeze(),
                                        curr_eef_quat.squeeze())
                                _reference_pos_r = _ik_term_r._ik_controller.ee_pos_des[env_idx]
                                _d = _aim - _reference_pos_r
                                _n = float(torch.linalg.norm(_d))
                                if _n > 1e-9:
                                    pose_R[env_idx, :3] = _d * min(1.0, REACH_STEP_M / _n)
                                _reference_quat_r = _ik_term_r._ik_controller.ee_quat_des[env_idx]
                                _reference_rot_r = R.from_quat(
                                    _reference_quat_r.detach().cpu().numpy()[[1, 2, 3, 0]])
                                _rv = torch.as_tensor((r_goal * _reference_rot_r.inv()).as_rotvec(),
                                                      dtype=pose_R.dtype, device=pose_R.device)
                                _rn = float(torch.linalg.norm(_rv))
                                if _rn > 1e-9:
                                    pose_R[env_idx, 3:6] = _rv * min(1.0, REACH_STEP_RAD / _rn)
                                env_cmd_indices_r[env_idx] = max(0, _plen - 1)
                                if int(arm_r_retry[env_idx]) % 20 == 1:
                                    print(f"[close] Env {env_idx}: A_r short by "
                                          f"{float(pos_err):.4f} m ({rot_err_deg:.1f} deg); "
                                          f"jogging in "
                                          f"({int(arm_r_retry[env_idx])}/{REACH_RETRIES} steps)",
                                          flush=True)
                                    print(f"[close-target] arm=r env={env_idx} "
                                          f"aim={_aim.tolist()} reference={_reference_pos_r.tolist()} "
                                          f"actual={curr_eef_pos.squeeze().tolist()}", flush=True)
                            elif pos_err > pos_threshold or _rot_out or _mug_jaws_not_ready_r or _tracking_failed_r:
                                idx = torch.tensor([env_idx], device=device)
                                arm_r_retry[env_idx] = 0
                                # SAY BY HOW MUCH, AND TO WHERE. "not reached to goal" alone
                                # cannot separate a few centimetres of gravity sag from a
                                # constant frame offset, and those want opposite fixes: the
                                # first is actuator gains, the second is a wrong tool frame or
                                # a cuRobo URDF whose ee link disagrees with the USD's. Fifteen
                                # identical failures were logged before anyone could tell which.
                                _g = ik_goals_r[env_idx].position.squeeze()
                                _a = curr_eef_pos.squeeze()
                                _d = (_a - _g).tolist()
                                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, idx, reason=(
                                    f"A_r not reached to goal. pos_err={float(pos_err):.4f} m "
                                    f"(limit {pos_threshold}), rot_err={rot_err_deg:.1f} deg "
                                    f"(grasp limit {GRASP_ROT_TOL_DEG if _is_grasp_reach else 'n/a'}), "
                                    f"goal_base=({float(_g[0]):+.4f},{float(_g[1]):+.4f},{float(_g[2]):+.4f}) "
                                    f"eef_base=({float(_a[0]):+.4f},{float(_a[1]):+.4f},{float(_a[2]):+.4f}) "
                                    f"delta=({_d[0]:+.4f},{_d[1]:+.4f},{_d[2]:+.4f}) "
                                    f"jaw_xy={_lat_miss:.4f}m jaw_dz={_pad_h:+.4f}m "
                                    f"mug_shift={_mug_shift_during_approach_r:.4f}m "
                                    f"jaw_ready={_accept_now} track_failed={_tracking_failed_r}"))
                            else:
                                print(f"[OK] Env {env_idx}: EEF reached goal within threshold "
                                      f"(tol={pos_threshold:.4f}"
                                      f"{' GRASP' if _is_grasp_reach else ''}, "
                                      f"pos_err={float(pos_err):.4f} m, {rot_err_deg:.1f} deg, "
                                      f"{int(arm_r_retry[env_idx])} jog step(s)).", flush=True)
                                arm_r_retry[env_idx] = 0
                                # THIS ENV, NOT THE WHOLE BATCH.
                                #
                                # It read finished_mask[executing_indices[finished_exec_mask]],
                                # which marks EVERY env whose plan finished this tick -- from
                                # inside a loop over those envs, deciding one at a time whether
                                # each actually reached its goal. So the first env to converge
                                # dragged all the others past the reach with it, and their
                                # grippers closed wherever they happened to be.
                                #
                                # Measured: 13 closes against 1 [OK] in a single run, and a
                                # bimodal MISS_AT_CLOSE -- min 0.0086 for the env that converged,
                                # median 0.1145 for the ones carried along. Both grasps of the
                                # campaign came from envs that converged on their own.
                                finished_mask[env_idx] = True

            # =================== Left Arm Motion ("A_l") ===================
            if torch.any(arm_l_mask):
                arm_l_rows = active_env_indices[arm_l_mask]
                arm_l_cols = env_goal_indices[arm_l_rows]
                skill_l = skill_ids_tensor[arm_l_rows, arm_l_cols]
                arm_l_unresolved = ~resolved_tensor[arm_l_rows, arm_l_cols]
                bowl_place_mask_l = (skill_l == SID_BOWL_PLACE) & arm_l_unresolved
                move_right_mask = (skill_l == SID_MUG_TO_POSITION) & arm_l_unresolved
                move_front_mask = (skill_l == SID_MOVE_FRONT) & arm_l_unresolved
                grasp_target_mask = (skill_l == SID_GRASP_TARGET) & arm_l_unresolved
                pause_mask_l = (skill_l == SID_PAUSE) & arm_l_unresolved

                # Build context once for all left-arm resolvers below.
                _ctx_l = RuntimeContext(
                    robot=env.scene.articulations["robot"],
                    env_origins=env.scene.env_origins,
                    r_eef_idx=r_eef_idx,
                    l_eef_idx=l_eef_idx,
                    base_link_idx=base_link_idx,
                    has_ramen=has_ramen,
                    has_sweet_potato=has_sweet_potato,
                    rigid_objects=env.scene.rigid_objects,
                )

                if torch.any(pause_mask_l):
                    pause_idx_l = arm_l_rows[pause_mask_l]
                    pos_e, quat = resolve_skill(
                        "arm.pause", _ctx_l, pause_idx_l, l_eef_idx)
                    j = env_goal_indices[pause_idx_l]
                    payloads_tensor[pause_idx_l, j, :3] = pos_e
                    payloads_tensor[pause_idx_l, j, 3:7] = quat
                    resolved_tensor[pause_idx_l, j] = True

                if torch.any(bowl_place_mask_l):
                    bowl_place_idx_l = arm_l_rows[bowl_place_mask_l]
                    j = env_goal_indices[bowl_place_idx_l]
                    # arm.bowl_place arguments are encoded in slots 0..4 before the runtime
                    # resolver replaces the row with its Cartesian pose. Mirror the right-arm
                    # path exactly, but resolve against the left end-effector body.
                    _pp_l = payloads_tensor[bowl_place_idx_l, j, :5].clone()
                    pos_e, q_out = resolve_skill("arm.bowl_place", _ctx_l, bowl_place_idx_l,
                        l_eef_idx,
                        forward_m=_pp_l[:, 0], down_m=_pp_l[:, 1],
                        min_eef_z=_pp_l[:, 2], roll_deg=_pp_l[:, 3], lateral_m=_pp_l[:, 4],
                    )
                    payloads_tensor[bowl_place_idx_l, j, :3] = pos_e
                    payloads_tensor[bowl_place_idx_l, j, 3:7] = q_out
                    resolved_tensor[bowl_place_idx_l, j] = True

                if torch.any(move_right_mask):
                    move_right_idx = arm_l_rows[move_right_mask]
                    pos_e, quat = resolve_skill("arm.mug_to_position", _ctx_l, move_right_idx, l_eef_idx)
                    j = env_goal_indices[move_right_idx]
                    payloads_tensor[move_right_idx, j, :3] = pos_e
                    payloads_tensor[move_right_idx, j, 3:7] = quat
                    resolved_tensor[move_right_idx, j] = True

                if torch.any(move_front_mask):
                    move_front_idx = arm_l_rows[move_front_mask]
                    # arm.move_front / arm.grasp_target are v1-only: skills.py has the resolver, but
                    # no @skill emits either, so there is no class to hang them on. They keep a
                    # legacy-only dispatch id (executor_dispatch.LEGACY_ONLY_SKILLS) — 1 step each
                    # on disk — and are called directly rather than through the registry.
                    pos_e, quat = skills.resolve_move_front(_ctx_l, move_front_idx, l_eef_idx)
                    j = env_goal_indices[move_front_idx]
                    payloads_tensor[move_front_idx, j, :3] = pos_e
                    payloads_tensor[move_front_idx, j, 3:7] = quat
                    resolved_tensor[move_front_idx, j] = True

                if torch.any(grasp_target_mask):
                    grasp_target_idx = arm_l_rows[grasp_target_mask]
                    pos_e, quats = skills.resolve_grasp_target(_ctx_l, grasp_target_idx, l_eef_idx)
                    j = env_goal_indices[grasp_target_idx]
                    payloads_tensor[grasp_target_idx, j, :3] = pos_e
                    payloads_tensor[grasp_target_idx, j, 3:7] = quats
                    resolved_tensor[grasp_target_idx, j] = True

                arm_l_indices = active_env_indices[arm_l_mask]
                needs_plan_indices_l = [i.item() for i in arm_l_indices if env_cmd_plans_l[i.item()] is None]
                
                if needs_plan_indices_l:
                    needs_plan_indices_temp_l = torch.as_tensor(needs_plan_indices_l, device=reset_l.device)
                    goal_step_indices_l = env_goal_indices[needs_plan_indices_l]
                    arm_l_goals = payloads_tensor[needs_plan_indices_l, goal_step_indices_l]
                    skill_l_plan = skill_ids_tensor[needs_plan_indices_l, goal_step_indices_l]
                    reset_mask_l = (skill_l_plan == SID_RESET)
                    reset_env_l = torch.nonzero(reset_mask_l, as_tuple=True)[0].tolist()
                    reset_env_l = torch.as_tensor(reset_env_l, device=reset_l.device)
                    _postrelease_reset_envs_l = set()
                    _final_home_reset_envs_l = set()
                    place_mask_l = (skill_l_plan == SID_PLACE)
                    place_env_l = torch.nonzero(place_mask_l, as_tuple=True)[0].tolist()
                    place_env_l = torch.as_tensor(place_env_l, device=reset_l.device)
                    not_reset_and_not_place_l = ~(reset_mask_l | place_mask_l)
                    grasp_env_l = torch.nonzero(not_reset_and_not_place_l, as_tuple=True)[0].tolist()
                    grasp_env_l = torch.as_tensor(grasp_env_l, device=reset_l.device)
                    # 1. Grasp, for reset
                    if torch.any(not_reset_and_not_place_l):
                        reset_l[needs_plan_indices_temp_l[grasp_env_l], :3] = env.scene["robot"].data.body_pos_w[needs_plan_indices_temp_l[grasp_env_l], l_eef_idx]
                        reset_l[needs_plan_indices_temp_l[grasp_env_l], 3:7] = env.scene["robot"].data.body_quat_w[needs_plan_indices_temp_l[grasp_env_l], l_eef_idx]
                        # THE JOINT CONFIGURATION AT THE SAME MOMENT, so arm.reset can ask cuRobo
                        # the strictly easier joint-space question when the Cartesian one fails.
                        # reset_l_js is the episode's original home snapshot.  Do not overwrite it
                        # at later arm skills (notably arm.bowl_place), or the final arm.reset
                        # returns to the pre-place pose instead of home.
                    # 2. Reset
                    if torch.any(reset_mask_l):
                        arm_l_goals[reset_env_l, :7] = reset_l[needs_plan_indices_temp_l[reset_env_l]]
                        if _postrelease_retreat_m > 0:
                            for _row in reset_env_l.tolist():
                                _e = int(needs_plan_indices_temp_l[_row])
                                _step = int(goal_step_indices_l[_row])
                                if (_step == 0
                                        or int(skill_ids_tensor[_e, _step - 1])
                                        != skill_ids["gripper.set"]
                                        or float(payloads_tensor[_e, _step - 1, 0]) > 0):
                                    continue
                                _object = env.scene.rigid_objects[args_cli.obj_name_l]
                                _xyz = postrelease_retreat_target(
                                    env.scene.articulations["robot"].data.body_pos_w[_e, l_eef_idx].tolist(),
                                    _object.data.body_pos_w[_e, 0].tolist(),
                                    env.scene.articulations["robot"].data.body_pos_w[_e, base_link_idx].tolist(),
                                    _postrelease_retreat_m, _postrelease_retreat_lift_m)
                                arm_l_goals[_row, :3] = torch.as_tensor(
                                    _xyz, dtype=arm_l_goals.dtype, device=device)
                                arm_l_goals[_row, 3:7] = env.scene.articulations["robot"].data.body_quat_w[_e, l_eef_idx]
                                _postrelease_reset_envs_l.add(_e)
                                print(f"[release-retreat] env{_e} open left hand to "
                                      f"{tuple(round(v, 4) for v in _xyz)}", flush=True)
                        if _postrelease_final_home:
                            for _row in reset_env_l.tolist():
                                _e = int(needs_plan_indices_temp_l[_row])
                                _step = int(goal_step_indices_l[_row])
                                if not postrelease_final_home_step(
                                        _step, skill_ids_tensor[_e], payloads_tensor[_e],
                                        reset_id=SID_RESET,
                                        gripper_id=skill_ids["gripper.set"]):
                                    continue
                                _home_pos_w, _home_quat_w = base2world(
                                    env, episode_home_l_base[_e, :3],
                                    episode_home_l_base[_e, 3:7], _e)
                                arm_l_goals[_row, :3] = _home_pos_w
                                arm_l_goals[_row, 3:7] = _home_quat_w
                                _final_home_reset_envs_l.add(_e)
                                if _final_home_joint_targets:
                                    arm_l_final_home_motor_envs.add(_e)
                                print(f"[release-home] env{_e} returning left arm to episode home "
                                      f"at {tuple(round(float(v), 4) for v in _home_pos_w)}",
                                      flush=True)
                        arm_l_goals[reset_env_l, :3] -= env.scene.env_origins[needs_plan_indices_temp_l[reset_env_l], :3]

                    # 3. Place
                    if torch.any(place_mask_l):
                        arm_l_goals[place_env_l, 3:7] = env.scene["robot"].data.body_quat_w[needs_plan_indices_temp_l[place_env_l], l_eef_idx]
                        reset_l[needs_plan_indices_temp_l[place_env_l], :3] = env.scene["robot"].data.body_pos_w[needs_plan_indices_temp_l[place_env_l], l_eef_idx]
                        reset_l[needs_plan_indices_temp_l[place_env_l], 3:7] = env.scene["robot"].data.body_quat_w[needs_plan_indices_temp_l[place_env_l], l_eef_idx]

                    for i, env_idx in enumerate(needs_plan_indices_l):
                        _repose_world_to_base("l", env_idx, int(skill_l_plan[i]) == SID_GRASP)
                        # Get current joint state data for the specific environment
                        sim_data = env.scene.articulations['robot'].data
                        # Match the right-arm planner path: differential IK can leave one live
                        # joint a few floating-point ulps outside cuRobo's limits, which makes
                        # plan_single_js reject arm.reset with INVALID_START_STATE_JOINT_LIMITS.
                        # Clamp only the planner snapshot, not the simulated articulation state.
                        _jl_l = sim_data.joint_limits[env_idx, l_j_index]
                        joint_pos_l = torch.clamp(
                            sim_data.joint_pos[env_idx, l_j_index],
                            _jl_l[:, 0] + 1e-4,
                            _jl_l[:, 1] - 1e-4,
                        ).tolist()
                        joint_vel_l = sim_data.joint_vel[env_idx, l_j_index].tolist()

                        # Preserve cuRobo's tensor-valued retract configuration;
                        # measured joints belong in this plan's JointState.
                        
                        # Convert goal to world frame and then to base frame
                        goal_env_l = arm_l_goals[i]
                        env_origin_pos_l = env.scene.env_origins[env_idx]
                        goal_pos_w_l = goal_env_l[:3] + env_origin_pos_l
                        goal_quat_w_l = goal_env_l[3:7]
                        _eef_quat_l = None
                        _plan_pos_w_l = goal_pos_w_l
                        if int(skill_l_plan[i]) == SID_GRASP and _cylindrical_mug_grasp_l:
                            # The authored target is the wrist frame, not the physical jaw gap.
                            # On this hand the gap missed the mug axis by 49.6 mm and sat 59.3 mm
                            # above its centre at close; it nudged the mug without lifting it.
                            # Re-express the live wrist-to-jaw-midpoint transform in the authored
                            # target orientation, then center the gap on the live mug axis/height.
                            try:
                                _obj_l = env.scene.rigid_objects[args_cli.obj_name_l]
                                _obj_pos_l = _obj_l.data.body_pos_w[env_idx, 0]
                                _obj_spawn_l = (
                                    _obj_l.data.default_root_state[env_idx, :3]
                                    + env.scene.env_origins[env_idx]
                                )
                                if env_idx not in arm_l_mug_approach_start_xy:
                                    arm_l_mug_approach_start_xy[env_idx] = _obj_pos_l[:2].clone()
                                    print(f"[grasp-start_l] env{env_idx} live_xy="
                                          f"{_obj_pos_l[:2].tolist()} spawn_xy="
                                          f"{_obj_spawn_l[:2].tolist()} spawn_drift="
                                          f"{float(torch.linalg.norm(_obj_pos_l[:2] - _obj_spawn_l[:2])):.4f}m",
                                          flush=True)
                                goal_pos_w_l = goal_pos_w_l + (_obj_pos_l - _obj_spawn_l)
                                _robot_l = env.scene.articulations["robot"]
                                _jaw_l_r, _jaw_l_l = jaw_body_indices(_robot_l, "left")
                                _jaw_mid_l = jaw_contact_midpoint(_robot_l, env_idx, _jaw_l_r, _jaw_l_l)
                                _eef_pos_l = _robot_l.data.body_pos_w[env_idx, l_eef_idx]
                                _eef_quat_l = _robot_l.data.body_quat_w[env_idx, l_eef_idx]
                                _q_now_l = R.from_quat(
                                    _eef_quat_l.cpu().numpy()[[1, 2, 3, 0]])
                                _jaw_offset_local_l = _q_now_l.inv().apply(
                                    (_jaw_mid_l - _eef_pos_l).cpu().numpy())
                                if args_cli.robot == "aiworker":
                                    # Optional distal-pad contact point, still inside the
                                    # measured 40 mm contact surface. Physical gates below
                                    # continue to measure the real pad centre and pad forces.
                                    _jaw_offset_local_l[2] += bounded_float_env(
                                        "SIMVLA_AIWORKER_CONTACT_DEPTH_M", 0.0, 0.0, 0.02)
                                _q_goal_l = R.from_quat(
                                    goal_quat_w_l.cpu().numpy()[[1, 2, 3, 0]])
                                _jaw_pin_l = torch.as_tensor(
                                    _q_goal_l.apply(_jaw_offset_local_l),
                                    dtype=goal_pos_w_l.dtype, device=goal_pos_w_l.device)
                                goal_pos_w_l = goal_pos_w_l - _jaw_pin_l
                                _axis_fix_l = torch.as_tensor(
                                    pinch_axis_xy_correction(
                                        goal_pos_w_l.tolist(), _jaw_pin_l.tolist(),
                                        _obj_pos_l.tolist()),
                                    dtype=goal_pos_w_l.dtype, device=goal_pos_w_l.device)
                                goal_pos_w_l = torch.cat([
                                    goal_pos_w_l[:2] + _axis_fix_l, goal_pos_w_l[2:]])
                                # Keep the physical jaw midpoint above the mug's surface-level
                                # root. Direct side grasps use a higher contact point for counter
                                # clearance without rotating the authored BoDex orientation;
                                # staged mode retains the half-height calibration.
                                _mug_height_l = float(os.environ.get("SIMVLA_MUG_HEIGHT", "0.08"))
                                _jaw_height_l = mug_grasp_target_height(
                                    _mug_height_l,
                                    float(os.environ["SIMVLA_MUG_GRASP_HEIGHT"])
                                    if "SIMVLA_MUG_GRASP_HEIGHT" in os.environ else
                                    (0.065 if _direct_bodex_grasp_l else None))
                                _center_z_fix_l = pinch_center_z_correction(
                                    goal_pos_w_l.tolist(), _jaw_pin_l.tolist(),
                                    _obj_pos_l.tolist(), _jaw_height_l)
                                goal_pos_w_l[2] += _center_z_fix_l
                                _goal_w_l[env_idx] = (goal_pos_w_l.clone(), goal_quat_w_l.clone())
                                _axis_l = torch.as_tensor(
                                    _q_goal_l.apply(np.array([0.0, 0.0, 1.0])),
                                    dtype=goal_pos_w_l.dtype, device=goal_pos_w_l.device)
                                if _direct_bodex_grasp_l:
                                    # cuRobo plans to the BoDex grasp itself. The executor's
                                    # midpoint pause resumes this same collision-checked plan.
                                    _plan_pos_w_l = goal_pos_w_l
                                else:
                                    _standoff_l = float(os.environ.get(
                                        "SIMVLA_PREGRASP_STANDOFF", "0.18"))
                                    _orient_standoff_l = max(
                                        _standoff_l,
                                        float(os.environ.get(
                                            "SIMVLA_PREGRASP_ORIENT_STANDOFF", "0.35")),
                                    )
                                    _back_sign_l = choose_pregrasp_back_sign(
                                        goal_pos_w_l.detach().cpu().tolist(),
                                        _axis_l.detach().cpu().tolist(),
                                        _eef_pos_l.detach().cpu().tolist(),
                                        _standoff_l,
                                    )
                                    arm_l_pregrasp_back_sign[env_idx] = _back_sign_l
                                    _back_l = _axis_l * _back_sign_l
                                    _plan_pos_w_l = goal_pos_w_l + _back_l * _orient_standoff_l
                                if int(env_idx) == 0:
                                    print(f"[track_l] env0 corrected mug grasp; "
                                          f"route={'direct' if _direct_bodex_grasp_l else 'staged'}, "
                                          f"plan_target={_plan_pos_w_l.tolist()}, "
                                          f"axis={_axis_l.tolist()}, "
                                          f"axis snap={float(torch.linalg.norm(_axis_fix_l)):.4f}m, "
                                          f"center z={_center_z_fix_l:+.4f}m",
                                          flush=True)
                            except Exception as _exc:  # noqa: BLE001
                                print(f"[track_l] could not correct mug grasp: "
                                      f"{type(_exc).__name__}: {_exc}", flush=True)
                        # In direct mode the target is the corrected BoDex grasp; staged mode
                        # retains the older distant orientation waypoint for comparison.
                        _plan_quat_w_l = goal_quat_w_l
                        goal_ee_pose_b_l, goal_ee_quat_b_l = world2base(
                            env, _plan_pos_w_l, _plan_quat_w_l, env_idx)
                        cu_js_l = JointState(
                            position=tensor_args.to_device(joint_pos_l),
                            # optimize_dt assumes a stationary start. With live nonzero
                            # velocity, the CENTRAL stencil's first sample extrapolates into
                            # the past (0.405 rad / 9.8 cm in job 2395130).
                            velocity=tensor_args.to_device(joint_vel_l) * (0.0 if optimize_dt else 1.0),
                            acceleration=tensor_args.to_device(joint_vel_l) * 0.0,
                            jerk=tensor_args.to_device(joint_vel_l) * 0.0,
                            joint_names=l_j_names,
                        )
                        cu_js_l = cu_js_l.get_ordered_joint_state(motion_gen_l.kinematics.joint_names)
                        _is_reset_l = int(skill_l_plan[i]) == SID_RESET
                        if _is_reset_l and env_idx not in _postrelease_reset_envs_l:
                            # A saved world pose stops being "home" after the mobile base moves.
                            # Derive the reset pose from the original joint home in the planner's
                            # current base frame, which remains valid before and after navigation.
                            _home_js_l = (episode_home_l_js[env_idx]
                                          if env_idx in _final_home_reset_envs_l
                                          else reset_l_js[env_idx])
                            _jsp_l = tensor_args.to_device(_home_js_l.tolist())
                            _gjs_l = JointState(
                                position=_jsp_l,
                                velocity=_jsp_l * 0.0,
                                acceleration=_jsp_l * 0.0,
                                jerk=_jsp_l * 0.0,
                                joint_names=l_j_names,
                            ).get_ordered_joint_state(motion_gen_l.kinematics.joint_names)
                            _home_fk_l = motion_gen_l.kinematics.forward(
                                _gjs_l.position.unsqueeze(0), link_name="ee_link2")
                            # cuRobo reuses FK output buffers. Preserve the home goal
                            # across planning and per-sample FK calls during playback.
                            goal_ee_pose_b_l = _home_fk_l[0].squeeze(0).clone()
                            goal_ee_quat_b_l = _home_fk_l[1].squeeze(0).clone()
                        ik_goal_l = Pose(position=goal_ee_pose_b_l, quaternion=goal_ee_quat_b_l)
                        ik_goals_l[env_idx] = ik_goal_l
                        _grasp_plan_config_l = plan_config
                        _grasp_coarse_config_l = plan_config_no_finetune
                        _segmented_cmd_plan_l = None
                        if _direct_bodex_grasp_l and int(skill_l_plan[i]) == SID_GRASP:
                            _approach_offset_l = bounded_float_env(
                                "SIMVLA_DIRECT_APPROACH_OFFSET_M", 0.0, -0.30, 0.30)
                            if abs(_approach_offset_l) > 1e-6:
                                from curobo.rollout.cost.pose_cost import PoseCostMetric
                                _metric_l = PoseCostMetric.create_grasp_approach_metric(
                                    offset_position=_approach_offset_l, linear_axis=2,
                                    tstep_fraction=0.8, tensor_args=tensor_args)
                                # Set explicitly: this cuRobo version's factory accepts but
                                # does not forward project_to_goal_frame.
                                _metric_l.project_to_goal_frame = True
                                _grasp_plan_config_l = plan_config.clone()
                                _grasp_plan_config_l.pose_cost_metric = _metric_l
                                _grasp_coarse_config_l = plan_config_no_finetune.clone()
                                _grasp_coarse_config_l.pose_cost_metric = _metric_l.clone()
                                print(f"[plan_l] gripper-axis approach cost: "
                                      f"offset={_approach_offset_l:+.3f}m, final 20% of one plan",
                                      flush=True)
                        # Reset is authored as a joint home. Plan to that joint state directly so
                        # cuRobo follows the intended folded branch; execution still uses the
                        # collector's Cartesian delta controller, and the strict FK-pose gate below
                        # prevents navigation until the visible hand has reached home.
                        if _is_reset_l and env_idx not in _postrelease_reset_envs_l:
                            result_l = motion_gen_l.plan_single_js(cu_js_l.unsqueeze(0),
                                                                   _gjs_l.unsqueeze(0),
                                                                   plan_config_graph)
                            if result_l.success.item():
                                print(f"[plan] Env {env_idx}: arm.reset planned in joint space (L)",
                                      flush=True)
                        elif (_segmented_grasp_standoff_l and _direct_bodex_grasp_l
                              and int(skill_l_plan[i]) == SID_GRASP):
                            # A single long free-space plan can sweep the outer left finger
                            # across the mug. Plan to a collision-checked pose behind the
                            # BoDex target, then plan the short final entry with the same
                            # authored orientation. Playback pauses exactly at the join.
                            _q_pre_l = R.from_quat(
                                goal_ee_quat_b_l.cpu().numpy()[[1, 2, 3, 0]])
                            _back_b_l = torch.as_tensor(
                                _q_pre_l.apply([0.0, 0.0, -_segmented_grasp_standoff_l]),
                                dtype=goal_ee_pose_b_l.dtype, device=goal_ee_pose_b_l.device)
                            _pre_goal_l = Pose(
                                position=goal_ee_pose_b_l + _back_b_l,
                                quaternion=goal_ee_quat_b_l)
                            _pre_result_l = motion_gen_l.plan_single(
                                cu_js_l.unsqueeze(0), _pre_goal_l, plan_config)
                            if not _pre_result_l.success.item():
                                result_l = _pre_result_l
                            else:
                                _pre_full_l = motion_gen_l.get_full_js(
                                    _pre_result_l.get_interpolated_plan())
                                _pre_names_l = list(_pre_full_l.joint_names)
                                _pre_positions_l = _pre_full_l.position.clone()
                                _active_pre_l = _pre_full_l.get_ordered_joint_state(
                                    motion_gen_l.kinematics.joint_names).position[-1].clone()
                                _pre_start_l = JointState(
                                    position=_active_pre_l,
                                    velocity=_active_pre_l * 0.0,
                                    acceleration=_active_pre_l * 0.0,
                                    jerk=_active_pre_l * 0.0,
                                    joint_names=list(motion_gen_l.kinematics.joint_names))
                                result_l = motion_gen_l.plan_single(
                                    _pre_start_l.unsqueeze(0), ik_goals_l[env_idx],
                                    _grasp_plan_config_l)
                                if result_l.success.item():
                                    _final_full_l = motion_gen_l.get_full_js(
                                        result_l.get_interpolated_plan())
                                    _final_positions_l = _final_full_l.get_ordered_joint_state(
                                        _pre_names_l).position.clone()
                                    _joined_positions_l = torch.cat(
                                        (_pre_positions_l, _final_positions_l[1:]), dim=0)
                                    _segmented_cmd_plan_l = JointState(
                                        position=_joined_positions_l,
                                        velocity=_joined_positions_l * 0.0,
                                        acceleration=_joined_positions_l * 0.0,
                                        jerk=_joined_positions_l * 0.0,
                                        joint_names=_pre_names_l)
                                    arm_l_segment_pause_sample[env_idx] = len(_pre_positions_l) - 1
                                    print(f"[plan_l] env{env_idx} segmented BoDex grasp: "
                                          f"pregrasp={len(_pre_positions_l)} samples, "
                                          f"final={len(_final_positions_l)} samples, "
                                          f"standoff={_segmented_grasp_standoff_l:.3f}m",
                                          flush=True)
                        else:
                            result_l = motion_gen_l.plan_single(
                                cu_js_l.unsqueeze(0), ik_goals_l[env_idx], _grasp_plan_config_l)
                        if (not result_l.success.item() and not _is_reset_l
                                and "FINETUNE_TRAJOPT" in str(getattr(result_l, "status", ""))):
                            # cuRobo found a trajectory and could not polish it; the coarse one is
                            # usable, and refusing to move at all is the worse outcome.
                            result_l = motion_gen_l.plan_single(cu_js_l.unsqueeze(0),
                                                                ik_goals_l[env_idx],
                                                                _grasp_coarse_config_l)
                            if result_l.success.item():
                                print(f"[plan] Env {env_idx}: finetune trajopt failed; planned "
                                      f"without it (L)", flush=True)
                        if result_l.success.item():
                            if _segmented_cmd_plan_l is None:
                                arm_l_segment_pause_sample.pop(env_idx, None)
                                cmd_plan_l = result_l.get_interpolated_plan()
                                cmd_plan_l = motion_gen_l.get_full_js(cmd_plan_l)
                            else:
                                cmd_plan_l = _segmented_cmd_plan_l
                            if (_cylindrical_mug_grasp_l
                                    and int(skill_l_plan[i]) == SID_GRASP
                                    and int(env_idx) == 0):
                                # Record whether cuRobo's supposedly distant-orientation plan
                                # actually ends at the requested pose. The subsequent Cartesian
                                # reach trace cannot tell planner execution from local-IK drift.
                                _plan_ordered_l = cmd_plan_l.get_ordered_joint_state(
                                    motion_gen_l.kinematics.joint_names)
                                _plan_end_fk_l = motion_gen_l.kinematics.forward(
                                    _plan_ordered_l.position[-1].unsqueeze(0),
                                    link_name="ee_link2",
                                )
                                _plan_end_fk_l = tuple(v.clone() if torch.is_tensor(v) else v
                                                       for v in _plan_end_fk_l)
                                _plan_start_fk_l = motion_gen_l.kinematics.forward(
                                    _plan_ordered_l.position[0].unsqueeze(0),
                                    link_name="ee_link2",
                                )
                                _plan_start_fk_l = tuple(v.clone() if torch.is_tensor(v) else v
                                                         for v in _plan_start_fk_l)
                                _current_plan_q_l = cu_js_l.get_ordered_joint_state(
                                    motion_gen_l.kinematics.joint_names).position
                                _current_fk_l = motion_gen_l.kinematics.forward(
                                    _current_plan_q_l.unsqueeze(0), link_name="ee_link2")
                                _physical_start_l, _ = world2base(
                                    env, env.scene.articulations["robot"].data.body_pos_w[
                                        env_idx, l_eef_idx],
                                    env.scene.articulations["robot"].data.body_quat_w[
                                        env_idx, l_eef_idx], env_idx)
                                _plan_end_pos_l = _plan_end_fk_l[0].squeeze(0)
                                _plan_end_quat_l = _plan_end_fk_l[1].squeeze(0)
                                _plan_target_l = ik_goals_l[env_idx]
                                _plan_end_pos_err_l = float(torch.linalg.norm(
                                    _plan_end_pos_l - _plan_target_l.position.squeeze(0)))
                                _plan_end_rot_l = R.from_quat(
                                    _plan_end_quat_l.detach().cpu().numpy()[[1, 2, 3, 0]])
                                _plan_target_rot_l = R.from_quat(
                                    _plan_target_l.quaternion.squeeze(0).detach().cpu().numpy()[
                                        [1, 2, 3, 0]])
                                _plan_end_rot_err_l = float(np.linalg.norm(
                                    (_plan_target_rot_l.inv() * _plan_end_rot_l).as_rotvec())
                                    * 180.0 / np.pi)
                                _solver_pos_err_l = result_l.position_error
                                _solver_rot_err_l = result_l.rotation_error
                                if torch.is_tensor(_solver_pos_err_l):
                                    _solver_pos_err_l = float(
                                        _solver_pos_err_l.reshape(-1)[0].detach().cpu())
                                if torch.is_tensor(_solver_rot_err_l):
                                    _solver_rot_err_l = float(
                                        _solver_rot_err_l.reshape(-1)[0].detach().cpu())
                                print(f"[plan_l] Env 0 cylindrical mug target: "
                                      f"waypoint={_plan_target_l.position.squeeze(0).tolist()} "
                                      f"samples={len(cmd_plan_l)} "
                                      f"status={result_l.status} "
                                      f"solver_error={_solver_pos_err_l}/{_solver_rot_err_l} "
                                      f"fk_end={_plan_end_pos_l.tolist()} "
                                      f"FK residual={_plan_end_pos_err_l:.4f}m/"
                                      f"{_plan_end_rot_err_l:.1f}deg", flush=True)
                                print(f"[plan_l] Env 0 start calibration: "
                                      f"joint_delta_max={float(torch.max(torch.abs(_plan_ordered_l.position[0] - _current_plan_q_l))):.4f}rad "
                                      f"fk_current={_current_fk_l[0].squeeze().tolist()} "
                                      f"fk_plan_start={_plan_start_fk_l[0].squeeze().tolist()} "
                                      f"isaac_start={_physical_start_l.squeeze().tolist()}",
                                      flush=True)
                                print(f"[plan_l] start joints current={_current_plan_q_l.tolist()} "
                                      f"interpolated={_plan_ordered_l.position[0].tolist()} "
                                      f"optimized={result_l.optimized_plan.position[0].tolist()}",
                                      flush=True)
                            # The joint names in the plan can differ from the robot's joint names
                            plan_j_names_l = cmd_plan_l.joint_names
                            # Filter r_j_names to only include joints present in the plan
                            common_j_names_l = [name for name in l_j_names if name in plan_j_names_l]
                            env_cmd_plans_l[env_idx] = cmd_plan_l.get_ordered_joint_state(common_j_names_l)
                            env_cmd_indices_l[env_idx] = 0
                            env_cmd_track_stall_l[env_idx] = 0
                            if _direct_bodex_grasp_l and int(skill_l_plan[i]) == SID_GRASP:
                                _anchor_robot_l = env.scene.articulations["robot"]
                                _anchor_pos_l, _anchor_quat_l = world2base(
                                    env, _anchor_robot_l.data.body_pos_w[env_idx, l_eef_idx],
                                    _anchor_robot_l.data.body_quat_w[env_idx, l_eef_idx], env_idx)
                                anchor_relative_ik_target(
                                    env.action_manager.get_term("armL_action"), env_idx,
                                    _anchor_pos_l, _anchor_quat_l)
                                if env_idx not in _direct_base_anchors:
                                    _bp_hold = _anchor_robot_l.data.body_pos_w[env_idx, base_link_idx]
                                    _yaw_hold = euler_xyz_from_quat(
                                        _anchor_robot_l.data.body_quat_w[
                                            env_idx:env_idx + 1, base_link_idx])[2][0]
                                    _direct_base_anchors[env_idx] = (
                                        float(_bp_hold[0]), float(_bp_hold[1]), float(_yaw_hold))
                        elif ALLOW_CARTESIAN_FALLBACK and (int(skill_l_plan[i]) in (
                                SID_RESET, SID_PLACE, SID_POSE,
                                SID_HANDLE_PREGRASP, SID_HANDLE_GRASP,
                                SID_BAR_HANDLE_PREGRASP, SID_BAR_HANDLE_GRASP)
                              or (_cylindrical_mug_grasp_l
                                  and not _direct_bodex_grasp_l
                                  and int(skill_l_plan[i]) == SID_GRASP)):
                            # HOLD-PLAN FALLBACK. For reset/place/authored pose this lets the established
                            # bounded Cartesian recovery finish a goal cuRobo cannot plan. For an
                            # explicitly selected cylindrical-mug grasp, strict endpoint gates
                            # can reject cuRobo's approximate plan; the staged jog then tries the
                            # same physically gated reach without pretending the planner succeeded.
                            #
                            # A reset/place can still miss without the Cartesian stage: measured
                            # on the dishwasher with only the planner retries (job 2114497), 42 of
                            # 71 episodes still ended at the withdraw, and the joint-space retry
                            # never once succeeded.
                            #
                            # A two-sample HOLD plan at the current joint state: the executor sees a
                            # plan that finishes immediately, then the grasp/reset/place recovery
                            # below drives the hand against its bounded retry and physical-contact
                            # gates. Keep this available to other explicitly selected cylindrical
                            # mug grasps rather than special-casing one robot model.
                            _pl = (cu_js_l.position if cu_js_l.position.dim() == 1
                                   else cu_js_l.position.squeeze(0))
                            _ppl = torch.stack([_pl, _pl])
                            env_cmd_plans_l[env_idx] = JointState(
                                position=_ppl, velocity=_ppl * 0.0,
                                acceleration=_ppl * 0.0, jerk=_ppl * 0.0,
                                joint_names=list(motion_gen_l.kinematics.joint_names),
                            )
                            env_cmd_indices_l[env_idx] = 0
                            env_cmd_pause_trig_l[env_idx] = True   # no mid-plan pause on a hold
                            env_cmd_pause_l[env_idx] = 0
                            arm_l_retry[env_idx] = 0
                            arm_l_pregrasp_stage[env_idx] = 0
                            arm_l_pregrasp_back_sign[env_idx] = 0
                            if int(skill_l_plan[i]) == SID_PLACE:
                                _unl = "arm.place"
                            elif int(skill_l_plan[i]) == SID_RESET:
                                _unl = "arm.reset"
                            elif int(skill_l_plan[i]) == SID_POSE:
                                _unl = "arm.pose"
                            elif int(skill_l_plan[i]) == SID_HANDLE_PREGRASP:
                                _unl = "arm.handle_pregrasp"
                            elif int(skill_l_plan[i]) == SID_HANDLE_GRASP:
                                _unl = "arm.handle_grasp"
                            elif int(skill_l_plan[i]) == SID_BAR_HANDLE_PREGRASP:
                                _unl = "arm.bar_handle_pregrasp"
                            elif int(skill_l_plan[i]) == SID_BAR_HANDLE_GRASP:
                                _unl = "arm.bar_handle_grasp"
                            else:
                                _unl = "cylindrical mug grasp"
                            print(f"[home] Env {env_idx}: {_unl} unplannable "
                                  f"({getattr(result_l, 'status', None)}); jogging instead (L)",
                                  flush=True)
                        else:
                            idx = torch.tensor([env_idx], device=device)
                            # SAY WHICH STEP AND WHICH SKILL. "plan_fail_l" alone cannot tell an
                            # unreachable grasp from a reset cuRobo will not leave, and the two want
                            # different fixes. Both are already to hand here.
                            _sil = int(env_goal_indices[env_idx])
                            _skl = int(skill_ids_tensor[env_idx, _sil])
                            _nml = next((n for n, v in skill_ids.items() if v == _skl), _skl)
                            print(f"[plan_fail_l] Env {env_idx}: step {_sil} ({_nml}) "
                                  f"status={getattr(result_l, 'status', None)}", flush=True)
                            try:
                                from planner_diagnostics import failure_snapshot
                                print("[planner-start] " + failure_snapshot(
                                    motion_gen_l.kinematics, cu_js_l, arm="l", step=_sil,
                                    status=getattr(result_l, "status", None)), flush=True)
                            except Exception as diagnostic_error:
                                print(f"[planner-start] diagnostic unavailable: {diagnostic_error}", flush=True)
                            _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,  env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, idx, reason="plan_fail_l") 
                paused_mask_l = env_cmd_pause_l[arm_l_indices] > 0
                if torch.any(paused_mask_l):
                    paused_envs_l = arm_l_indices[paused_mask_l]
                    env_cmd_pause_l[paused_envs_l] -= 1
                    if _direct_bodex_grasp_l:
                        for _resume_env_l in paused_envs_l[
                                env_cmd_pause_l[paused_envs_l] == 0].tolist():
                            if int(skill_ids_tensor[
                                    _resume_env_l, int(env_goal_indices[_resume_env_l])]) == SID_GRASP:
                                print(f"[resume_l] Env {_resume_env_l}: continuing cuRobo "
                                      f"at sample {int(env_cmd_indices_l[_resume_env_l])}",
                                      flush=True)

# 2) normal executing mask (has plan) AND not paused
                has_plan_mask_l = torch.tensor(
                    [env_cmd_plans_l[i.item()] is not None for i in arm_l_indices],
                    device=device
                )
                not_paused_mask_l = (env_cmd_pause_l[arm_l_indices] == 0)
                can_run_mask_l = has_plan_mask_l & not_paused_mask_l

                if torch.any(can_run_mask_l):
                    executing_indices_l = arm_l_indices[can_run_mask_l]

                    # Get plans and current cmd indices
                    plans_to_exec_l = [env_cmd_plans_l[i.item()] for i in executing_indices_l]
                    indices_to_exec_l = env_cmd_indices_l[executing_indices_l]

                    # plan lengths (needed for midpoint pause + completion)
                    plan_lengths_l = torch.tensor([len(p) for p in plans_to_exec_l], device=device, dtype=torch.long)

                    # 3) trigger pause at 50% ONCE, then resume the same plan
                    pause_at_l = torch.clamp((plan_lengths_l.float() * 0.5).floor().long(), min=1)
                    for _pause_row_l, _pause_env_l in enumerate(executing_indices_l.tolist()):
                        _pause_skill_l = int(skill_ids_tensor[
                            _pause_env_l, int(env_goal_indices[_pause_env_l])])
                        if (_pause_skill_l == SID_GRASP
                                and _pause_env_l in arm_l_segment_pause_sample):
                            pause_at_l[_pause_row_l] = arm_l_segment_pause_sample[_pause_env_l]
                    already_trig_l = env_cmd_pause_trig_l[executing_indices_l]
                    should_pause_l = (~already_trig_l) & (indices_to_exec_l >= pause_at_l)

                    if torch.any(should_pause_l):
                        pause_envs_l = executing_indices_l[should_pause_l]
                        env_cmd_pause_l[pause_envs_l] = args_cli.action_chunk_size + 1
                        env_cmd_pause_trig_l[pause_envs_l] = True
                        for _pause_env_l in pause_envs_l.tolist():
                            if (_direct_bodex_grasp_l and int(skill_ids_tensor[
                                    _pause_env_l, int(env_goal_indices[_pause_env_l])]) == SID_GRASP):
                                print(f"[pause_l] Env {_pause_env_l}: BoDex/cuRobo grasp "
                                      f"at sample {int(env_cmd_indices_l[_pause_env_l])}/"
                                      f"{len(env_cmd_plans_l[_pause_env_l])}; "
                                      f"holding {int(env_cmd_pause_l[_pause_env_l])} steps, "
                                      "then resuming", flush=True)

                        # remove paused envs from this timestep's execution (so they stop immediately)
                        keep_l = ~should_pause_l
                        executing_indices_l = executing_indices_l[keep_l]
                        indices_to_exec_l = indices_to_exec_l[keep_l]
                        plans_to_exec_l = [p for p, k in zip(plans_to_exec_l, keep_l.tolist()) if k]
                        plan_lengths_l = plan_lengths_l[keep_l]

                    # 4) execute the remaining envs
                    if executing_indices_l.numel() > 0:
                        delta_poses_l = compute_eef_deltas_batched(
                            motion_gen=motion_gen_l,
                            plans=plans_to_exec_l,
                            cmd_indices=indices_to_exec_l,
                            joint_names=common_j_names_l,
                            eef_link_name="ee_link2",
                        )

                        # Update pose_L
                        pose_L[executing_indices_l] = delta_poses_l

                        # Direct AI Worker grasps replay each cuRobo segment only after the
                        # measured wrist has caught up to that segment's starting pose. The
                        # relative-IK target otherwise races ahead of the physical arm: an
                        # FK-exact 177-sample plan landed 20 cm short in job 2395004.
                        _advance_l = torch.ones(
                            len(executing_indices_l), dtype=torch.bool, device=device)
                        if _direct_bodex_grasp_l:
                            _robot_track_l = env.scene.articulations["robot"]
                            for _row_l, _env_l in enumerate(executing_indices_l.tolist()):
                                _track_skill_l = int(skill_ids_tensor[
                                    _env_l, int(env_goal_indices[_env_l])])
                                _track_loaded_home_l = (_track_loaded_home_enabled
                                    and _track_skill_l == SID_RESET
                                    and (bool(new_gripper_commands_L[_env_l])
                                         or _env_l in arm_l_final_home_motor_envs))
                                if _track_skill_l != SID_GRASP and not _track_loaded_home_l:
                                    continue
                                _plan_l = plans_to_exec_l[_row_l]
                                _sample_l = min(int(indices_to_exec_l[_row_l]), len(_plan_l) - 1)
                                if _track_loaded_home_l and _loaded_home_joint_targets:
                                    _motor_ids_l, _plan_columns_l = loaded_home_arm_indices(
                                        _plan_l.joint_names, _robot_track_l.joint_names)
                                    _motor_target_l = _plan_l.position[_sample_l, _plan_columns_l]
                                    _joint_error_l = float(torch.max(torch.abs(
                                        _robot_track_l.data.joint_pos[_env_l, _motor_ids_l] - _motor_target_l)))
                                    _actual_home_pos, _actual_home_quat = world2base(
                                        env, _robot_track_l.data.body_pos_w[_env_l, l_eef_idx],
                                        _robot_track_l.data.body_quat_w[_env_l, l_eef_idx], _env_l)
                                    anchor_relative_ik_target(env.action_manager.get_term("armL_action"),
                                                              _env_l, _actual_home_pos, _actual_home_quat)
                                    _advance_l[_row_l] = _joint_error_l <= .035
                                    # Keep the recorded Cartesian command consistent with the
                                    # same next sample commanded through the joint motors.
                                    # A zero Cartesian stream during a moving reset would be
                                    # invalid action data, even if joint-target replay passed.
                                    _motor_sample_l = min(_sample_l + int(_advance_l[_row_l]), len(_plan_l) - 1)
                                    _loaded_home_motor_overrides[_env_l] = (
                                        _motor_ids_l, _plan_l.position[_motor_sample_l, _plan_columns_l].clone())
                                    if not bool(_advance_l[_row_l]):
                                        pose_L[_env_l] = 0.0
                                        env_cmd_track_stall_l[_env_l] += 1
                                        if int(env_cmd_track_stall_l[_env_l]) % 20 == 0:
                                            print(f"[home-joints] env{_env_l} sample={_sample_l}/{len(_plan_l)} "
                                                  f"max_error_rad={_joint_error_l:.5f} "
                                                  f"names={[_robot_track_l.joint_names[j] for j in _motor_ids_l]} "
                                                  f"actual={_robot_track_l.data.joint_pos[_env_l, _motor_ids_l].tolist()} "
                                                  f"target={_motor_target_l.tolist()} "
                                                  f"motor_target={_robot_track_l.data.joint_pos_target[_env_l, _motor_ids_l].tolist()} "
                                                  f"velocity={_robot_track_l.data.joint_vel[_env_l, _motor_ids_l].tolist()} "
                                                  f"feedforward={_robot_track_l.data.joint_effort_target[_env_l, _motor_ids_l].tolist()} "
                                                  f"effort={_robot_track_l.data.applied_torque[_env_l, _motor_ids_l].tolist()}",
                                                  flush=True)
                                        if int(env_cmd_track_stall_l[_env_l]) >= _direct_track_timeout_l:
                                            env_cmd_indices_l[_env_l] = len(_plan_l)
                                    else:
                                        env_cmd_track_stall_l[_env_l] = 0
                                    continue
                                # Playback accumulates plan deltas from the measured anchor.
                                # Compare against that same reference: raw FK differs by the
                                # live lift deflection (13 mm in job 2395164).
                                _track_controller_l = env.action_manager.get_term(
                                    "armL_action")._ik_controller
                                _fk_pos_l = _track_controller_l.ee_pos_des[_env_l]
                                _fk_quat_l = _track_controller_l.ee_quat_des[_env_l]
                                _actual_pos_l, _actual_quat_l = world2base(
                                    env, _robot_track_l.data.body_pos_w[_env_l, l_eef_idx],
                                    _robot_track_l.data.body_quat_w[_env_l, l_eef_idx], _env_l)
                                _track_pos_err_l = float(torch.linalg.norm(
                                    _actual_pos_l.squeeze() - _fk_pos_l.squeeze()))
                                _track_actual_r_l = R.from_quat(
                                    _actual_quat_l.squeeze().detach().cpu().numpy()[
                                        [1, 2, 3, 0]])
                                _track_plan_r_l = R.from_quat(
                                    _fk_quat_l.squeeze().detach().cpu().numpy()[
                                        [1, 2, 3, 0]])
                                _track_rot_err_l = math.degrees(float(np.linalg.norm(
                                    (_track_plan_r_l.inv() * _track_actual_r_l).as_rotvec())))
                                _pos_track_tol_l = (_loaded_home_track_pos_tol if _track_loaded_home_l
                                                    else _direct_track_pos_tol_l)
                                _rot_track_tol_l = (_loaded_home_track_rot_tol if _track_loaded_home_l
                                                    else _direct_track_rot_tol_l)
                                if (_track_pos_err_l > _pos_track_tol_l
                                        or _track_rot_err_l > _rot_track_tol_l):
                                    _advance_l[_row_l] = False
                                    pose_L[_env_l] = 0.0  # hold the last commanded IK target
                                    env_cmd_track_stall_l[_env_l] += 1
                                    if int(env_cmd_track_stall_l[_env_l]) == 1 or (
                                            int(env_cmd_track_stall_l[_env_l]) % 20 == 0):
                                        print(f"[track_l] Env {_env_l}: waiting at cuRobo sample "
                                              f"{_sample_l}/{len(_plan_l)}; measured error "
                                              f"{_track_pos_err_l:.4f}m/"
                                              f"{_track_rot_err_l:.1f}deg; "
                                              f"stall={int(env_cmd_track_stall_l[_env_l])}",
                                              flush=True)
                                    if int(env_cmd_track_stall_l[_env_l]) >= _direct_track_timeout_l:
                                        print(f"[track_l] Env {_env_l}: waypoint timeout; "
                                              "rejecting/replanning tracked arm motion", flush=True)
                                        env_cmd_indices_l[_env_l] = len(_plan_l)
                                else:
                                    env_cmd_track_stall_l[_env_l] = 0
                        env_cmd_indices_l[executing_indices_l[_advance_l]] += 1
                        finished_exec_mask_l = (env_cmd_indices_l[executing_indices_l] >= plan_lengths_l)
                        # Check if the current EEF is far from the goal
                        #
                        # THE LEFT ARM HAD NO GRASP GATE AT ALL, and that is what this is fixing.
                        #
                        # pos_threshold was 1.0 -- one METRE -- and rot_threshold_deg was assigned
                        # and never compared, the same defect the right arm's block documents and
                        # has since fixed. So a left-arm reach that landed anywhere within a metre,
                        # at any orientation, was accepted and the next step shut the jaws on
                        # whatever happened to be there.
                        #
                        # It matters because EVERY handle task in this repo uses the left arm: the
                        # drawer templates author A_l, the refrigerator's hinge derives Left, and
                        # the right arm's cuRobo config cannot plan a handle grasp at all. So no
                        # handle grasp has ever been gated.
                        #
                        # Measured on the dishwasher (kitchen 1221, jobs 2111308-2111564): the hand
                        # arrived on the bar -- hand_l_to_door 0.736-0.757 m against 0.749 for a
                        # hand exactly on it -- and the jaws closed to 7e-06 m on a 0.050 m bar,
                        # every episode, because "on the bar" to within a metre is not on the bar.
                        # The cabinet door survived the same gate only because its handle is a
                        # 0.168 m tall bar, forgiving of centimetres.
                        #
                        # This applies the SAME tolerance the right arm uses when the next step
                        # closes the gripper. There is still no jog here (that machinery is
                        # right-arm-only), so a miss RESETS the episode instead of proceeding --
                        # a retry rather than a wasted grasp.
                        _next_is_close_l = False
                        rot_threshold_deg = 10
                        for j, env_idx in enumerate(executing_indices_l[finished_exec_mask_l].tolist()):
                            _si_l = int(env_goal_indices[env_idx])
                            _sk_now_l = int(skill_ids_tensor[env_idx, _si_l])
                            try:
                                if _si_l + 1 < skill_ids_tensor.shape[1]:
                                    _nxt_l = int(skill_ids_tensor[env_idx, _si_l + 1])
                                    _next_is_close_l = (_nxt_l == skill_ids["gripper.set"]
                                                        and bool(payloads_tensor[env_idx,
                                                                                 _si_l + 1, 0] > 0))
                            except Exception:                           # noqa: BLE001
                                _next_is_close_l = False
                            # THREE REGIMES, not two. Reaching TO grasp needs millimetres; a free
                            # move needs nothing; and CARRYING something already gripped needs
                            # enough to stay on the arc it is following.
                            #
                            # The third one is why the dishwasher arc failed. Its waypoints are a
                            # circle about the door's hinge, so a correct motion holds
                            # hand_to_door CONSTANT at 0.7508 m. Ungated, step 4 drifted it to
                            # 0.7729 -- ~0.02 m RADIALLY OUTWARD, away from the hinge, which is
                            # precisely the direction that strips a friction-held bar out of the
                            # pads. Measured (job 2111851, env 7): jaw 0.0500 m on the bar at the
                            # end of the close, 0.0071 m by the end of a 5 degree arc step, door
                            # never moved. The pose was authored correctly; nothing corrected the
                            # ~0.03 m the planner leaves behind, because the gate was 1.0 m and
                            # the jog only runs when the gate is missed.
                            #
                            # 0.008 + 0.09*3deg = 0.0127 m of pad-midpoint error, inside the
                            # 0.015 m budget a 0.050 m bar allows (see grasp-gate-is-per-handle).
                            # Looser than the grasp gate on purpose: a carry step is a longer
                            # motion and need only stay on the arc, not land on a millimetre.
                            _holding_l = bool(new_gripper_commands_L[env_idx])
                            if _sk_now_l == SID_RESET:
                                pos_threshold = RESET_POS_TOL_M
                                rot_tol_l = RESET_ROT_TOL_DEG
                            elif _sk_now_l in (SID_POSE, SID_HANDLE_PREGRASP,
                                               SID_BAR_HANDLE_PREGRASP):
                                # Authored arm poses are dataset waypoints, so accepting the old
                                # 1 m free-motion gate would silently bank a failed plan as success.
                                # Use the same bounded Cartesian retry for a HOLD-PLAN recovery as
                                # for other physical reaches.
                                pos_threshold = float(_simvla_os.environ.get(
                                    "SIMVLA_POSE_POS_TOL", "0.05"))
                                rot_tol_l = float(_simvla_os.environ.get(
                                    "SIMVLA_POSE_ROT_TOL_DEG", "10.0"))
                            elif _next_is_close_l:
                                pos_threshold = float(_simvla_os.environ.get(
                                    "SIMVLA_GRASP_POS_TOL", "0.012"))
                                pos_threshold = mug_grasp_position_tolerance(
                                    args_cli.obj_name_l, pos_threshold, True)
                                rot_tol_l = float(_simvla_os.environ.get(
                                    "SIMVLA_GRASP_ROT_TOL_DEG", "5.0"))
                            elif _holding_l:
                                # OFF BY DEFAULT, opt in per task. Gating a carry sounded right --
                                # nothing corrects the planner's residual while the hand holds
                                # something, and that residual is partly RADIAL, the direction that
                                # strips a bar out of the pads. But the only measurement of it says
                                # it costs more than it buys: on the dishwasher it could not be
                                # satisfied once the door pulled the hand off its goal, and it
                                # discarded every episode that opened the door (7 of 7, job
                                # 2112357). Defaulting it ON would impose that on the cabinet and
                                # drawer tasks, which work today with no carry gate at all and have
                                # never been measured against one.
                                pos_threshold = float(_simvla_os.environ.get(
                                    "SIMVLA_CARRY_POS_TOL", "1.0"))
                                rot_tol_l = float(_simvla_os.environ.get(
                                    "SIMVLA_CARRY_ROT_TOL_DEG", "180.0"))
                            else:
                                pos_threshold, rot_tol_l = 1.0, 180.0
                            if (_sk_now_l == SID_GRASP and _next_is_close_l
                                    and _cylindrical_mug_grasp_l
                                    and env_idx in _goal_w_l):
                                # The base can drift while the plan runs and an upright mug can
                                # move on contact. Re-resolve the true grasp against the live base
                                # and keep the physical jaw midpoint over the mug axis.
                                try:
                                    _obj_live_l = env.scene.rigid_objects[args_cli.obj_name_l]
                                    _obj_pos_live_l = _obj_live_l.data.body_pos_w[env_idx, 0]
                                    _obj_q_live_l = _obj_live_l.data.body_quat_w[env_idx, 0]
                                    _up_z_live_l = 1.0 - 2.0 * (
                                        float(_obj_q_live_l[1]) ** 2 + float(_obj_q_live_l[2]) ** 2)
                                    _tilt_live_l = math.degrees(math.acos(
                                        max(-1.0, min(1.0, _up_z_live_l))))
                                    _gp_live_l, _gq_live_l = _goal_w_l[env_idx]
                                    if _tilt_live_l < 15.0:
                                        _jr_l, _jl_l = jaw_body_indices(
                                            env.scene.articulations["robot"], "left")
                                        _jm_l = jaw_contact_midpoint(
                                            env.scene.articulations["robot"], env_idx, _jr_l, _jl_l)
                                        _ep_l = env.scene.articulations["robot"].data.body_pos_w[
                                            env_idx, l_eef_idx]
                                        _eq_l = env.scene.articulations["robot"].data.body_quat_w[
                                            env_idx, l_eef_idx]
                                        _local_pin_l = R.from_quat(
                                            _eq_l.cpu().numpy()[[1, 2, 3, 0]]).inv().apply(
                                                (_jm_l - _ep_l).cpu().numpy())
                                        if args_cli.robot == "aiworker":
                                            _local_pin_l[2] += bounded_float_env(
                                                "SIMVLA_AIWORKER_CONTACT_DEPTH_M", 0.0, 0.0, 0.02)
                                        _pin_live_l = torch.as_tensor(
                                            R.from_quat(_gq_live_l.cpu().numpy()[[1, 2, 3, 0]])
                                            .apply(_local_pin_l),
                                            dtype=_gp_live_l.dtype, device=_gp_live_l.device)
                                        _gp_live_l = torch.cat([
                                            _obj_pos_live_l[:2] - _pin_live_l[:2],
                                            _gp_live_l[2:]])
                                        _goal_w_l[env_idx] = (_gp_live_l, _gq_live_l)
                                    _gp_base_l, _gq_base_l = world2base(
                                        env, _gp_live_l, _gq_live_l, env_idx)
                                    ik_goals_l[env_idx] = Pose(
                                        position=_gp_base_l, quaternion=_gq_base_l)
                                except Exception as _exc:  # noqa: BLE001
                                    print(f"[track_l] could not update live mug grasp: "
                                          f"{type(_exc).__name__}: {_exc}", flush=True)
                            curr_eef_pos, curr_eef_quat = world2base(env, env.scene.articulations['robot'].data.body_pos_w[env_idx,l_eef_idx], env.scene.articulations['robot'].data.body_quat_w[env_idx,l_eef_idx],env_idx)
                            pos_err = torch.norm(curr_eef_pos - ik_goals_l[env_idx].position)
                            l_eef = R.from_quat(curr_eef_quat.cpu().numpy()[[1, 2, 3, 0]])
                            l_goal = R.from_quat(ik_goals_l[env_idx].quaternion.squeeze(0).cpu().numpy()[[1, 2, 3, 0]])
                            rot_err_deg = np.linalg.norm((l_goal.inv() * l_eef).as_rotvec()) * 180 / np.pi
                            _physical_mug_contact_l = False
                            _mug_shift_during_approach_l = 0.0
                            if (_sk_now_l == SID_GRASP and _next_is_close_l
                                    and _cylindrical_mug_grasp_l):
                                try:
                                    _obj_contact_l = env.scene.rigid_objects[args_cli.obj_name_l]
                                    _obj_pos_contact_l = _obj_contact_l.data.body_pos_w[env_idx, 0]
                                    _obj_spawn_contact_l = (
                                        _obj_contact_l.data.default_root_state[env_idx, :3]
                                        + env.scene.env_origins[env_idx])
                                    _approach_start_l = arm_l_mug_approach_start_xy.get(env_idx)
                                    if _approach_start_l is None:
                                        _approach_start_l = _obj_spawn_contact_l[:2]
                                    _mug_shift_during_approach_l = float(torch.linalg.norm(
                                        _obj_pos_contact_l[:2] - _approach_start_l))
                                    if int(env_idx) == 0:
                                        print(f"[grasp-shift_l] env0 approach="
                                              f"{_mug_shift_during_approach_l:.4f}m spawn="
                                              f"{float(torch.linalg.norm(_obj_pos_contact_l[:2] - _obj_spawn_contact_l[:2])):.4f}m",
                                              flush=True)
                                    _jr_contact_l, _jl_contact_l = jaw_body_indices(
                                        env.scene.articulations["robot"], "left")
                                    _robot_contact_l = env.scene.articulations["robot"]
                                    _jaw_mid_contact_l = jaw_contact_midpoint(
                                        _robot_contact_l, env_idx, _jr_contact_l, _jl_contact_l)
                                    _jaw_xy_contact_l = float(torch.linalg.norm(
                                        _jaw_mid_contact_l[:2] - _obj_pos_contact_l[:2]))
                                    _jaw_delta_contact_l = (
                                        _jaw_mid_contact_l[:2] - _obj_pos_contact_l[:2]
                                    ).detach().cpu().tolist()
                                    _jaw_z_contact_l = float(
                                        _jaw_mid_contact_l[2] - _obj_pos_contact_l[2])
                                    _physical_mug_contact_l = (
                                        float(rot_err_deg) <= 5.0
                                        and mug_jaw_contact_ready(
                                            _jaw_xy_contact_l, _jaw_z_contact_l))
                                    if _physical_mug_contact_l and int(env_idx) == 0:
                                        print(f"[close_l] Env {env_idx}: jaws are on the mug "
                                              f"(lateral {_jaw_xy_contact_l:.4f} m, pad "
                                              f"{_jaw_z_contact_l:.4f} m up, "
                                              f"jaw_minus_obj_xy=({_jaw_delta_contact_l[0]:+.4f},"
                                              f"{_jaw_delta_contact_l[1]:+.4f}) m, "
                                              f"jaw_mid_w=({_jaw_mid_contact_l[0].item():+.4f},"
                                              f"{_jaw_mid_contact_l[1].item():+.4f},"
                                              f"{_jaw_mid_contact_l[2].item():+.4f}), "
                                              f"obj_w=({_obj_pos_contact_l[0].item():+.4f},"
                                              f"{_obj_pos_contact_l[1].item():+.4f},"
                                              f"{_obj_pos_contact_l[2].item():+.4f})); closing without "
                                              f"waiting for the wrist target", flush=True)
                                except Exception as _exc:  # noqa: BLE001
                                    print(f"[close_l] mug jaw check unavailable: "
                                          f"{type(_exc).__name__}: {_exc}", flush=True)
                            _mug_jaws_not_ready_l = (
                                _sk_now_l == SID_GRASP and _next_is_close_l
                                and _cylindrical_mug_grasp_l
                                and not _physical_mug_contact_l)
                            _miss_l = (pos_err > pos_threshold
                                       or float(rot_err_deg) > rot_tol_l
                                       or _mug_jaws_not_ready_l)
                            if (_sk_now_l == SID_RESET and bool(new_gripper_commands_L[env_idx])
                                    and int(env_cmd_track_stall_l[env_idx]) >= _direct_track_timeout_l):
                                # A timed-out waypoint is not a finished trajectory. Reject this
                                # attempt instead of rewinding to its last sample and jogging forever.
                                _miss_l = True
                                arm_l_retry[env_idx] = REACH_RETRIES
                            if _mug_shift_during_approach_l > 0.03:
                                print(f"[grasp_l] Env {env_idx}: mug moved "
                                      f"{_mug_shift_during_approach_l:.4f}m during approach; "
                                      "rejecting this candidate", flush=True)
                                _miss_l = True
                                arm_l_retry[env_idx] = REACH_RETRIES
                            if _physical_mug_contact_l and _mug_shift_during_approach_l <= 0.03:
                                _miss_l = False
                            if (_direct_bodex_grasp_l and _sk_now_l == SID_GRASP
                                    and _next_is_close_l and _miss_l):
                                # A cuRobo FK-perfect endpoint can be missed by the relative-IK
                                # tracker. Replan from the *measured* joint state, with a strict
                                # budget; never replace this with an unplanned vertical jog.
                                if (arm_l_retry[env_idx] < _direct_replan_limit_l
                                        and _mug_shift_during_approach_l <= 0.03):
                                    arm_l_retry[env_idx] += 1
                                    print(f"[replan_l] Env {env_idx}: direct BoDex grasp "
                                          f"tracking miss {float(pos_err):.4f}m/"
                                          f"{float(rot_err_deg):.1f}deg; cuRobo replan "
                                          f"{int(arm_l_retry[env_idx])}/"
                                          f"{_direct_replan_limit_l} from measured joints",
                                          flush=True)
                                    env_cmd_plans_l[env_idx] = None
                                    env_cmd_indices_l[env_idx] = 0
                                    env_cmd_pause_l[env_idx] = 0
                                    env_cmd_pause_trig_l[env_idx] = False
                                    continue
                                arm_l_retry[env_idx] = REACH_RETRIES
                            if _miss_l and arm_l_retry[env_idx] < REACH_RETRIES:
                                # CLOSE THE LOOP WITHOUT THE PLANNER -- the right arm's jog,
                                # ported. Measured need: the left arm's raw plan lands 0.031 m and
                                # 4.5 deg from the goal, which is 0.038 m of pad swing against a
                                # 0.015 m budget on a dishwasher bar. No gate can fix that; only a
                                # correction can.
                                #
                                # Rewinding the command pointer by one leaves the plan holding
                                # its last waypoint (delta zero) and makes this check fire again
                                # next step. The correction advances a persistent IK reference,
                                # rather than replacing it with the measured pose each frame.
                                #
                                # Deliberately WITHOUT the right arm's standoff staging: that
                                # exists because a bottle is free to be knocked over, and a handle
                                # is bolted to a door. A straight Cartesian jog over a few
                                # centimetres of free space is all this needs.
                                arm_l_retry[env_idx] += 1
                                _plen_l = len(env_cmd_plans_l[env_idx])
                                _jog_target_l = ik_goals_l[env_idx].position.squeeze()
                                _pregrasp_lateral_l = None
                                if (_sk_now_l == SID_GRASP and _next_is_close_l
                                        and _cylindrical_mug_grasp_l
                                        and env_idx in _goal_w_l):
                                    # Keep the open fingers clear while aligning laterally; only
                                    # then close in along the tool axis. This is the left-hand
                                    # counterpart of the proven AI Worker right-hand mug approach.
                                    _gp_world_l, _gq_world_l = _goal_w_l[env_idx]
                                    _axis_world_l = torch.as_tensor(
                                        R.from_quat(_gq_world_l.cpu().numpy()[[1, 2, 3, 0]])
                                        .apply(np.array([0.0, 0.0, 1.0])),
                                        dtype=_gp_world_l.dtype, device=_gp_world_l.device)
                                    _current_eef_world_l = env.scene.articulations[
                                        "robot"].data.body_pos_w[env_idx, l_eef_idx]
                                    _standoff_l = float(os.environ.get(
                                        "SIMVLA_PREGRASP_STANDOFF", "0.18"))
                                    _orient_standoff_l = max(
                                        _standoff_l,
                                        float(os.environ.get(
                                            "SIMVLA_PREGRASP_ORIENT_STANDOFF", "0.35")),
                                    )
                                    if int(arm_l_pregrasp_back_sign[env_idx]) == 0:
                                        arm_l_pregrasp_back_sign[env_idx] = (
                                            choose_pregrasp_back_sign(
                                                _gp_world_l.detach().cpu().tolist(),
                                                _axis_world_l.detach().cpu().tolist(),
                                                _current_eef_world_l.detach().cpu().tolist(),
                                                _standoff_l,
                                            )
                                        )
                                    _back_world_l = (
                                        _axis_world_l
                                        * int(arm_l_pregrasp_back_sign[env_idx])
                                    )
                                    _pre_world_l = _gp_world_l + _back_world_l * _standoff_l
                                    _pre_base_l, _ = world2base(
                                        env, _pre_world_l, _gq_world_l, env_idx)
                                    _orient_pre_world_l = (
                                        _gp_world_l + _back_world_l * _orient_standoff_l)
                                    _orient_pre_base_l, _ = world2base(
                                        env, _orient_pre_world_l, _gq_world_l, env_idx)
                                    _back_base_l = _pre_base_l - ik_goals_l[
                                        env_idx].position.squeeze()
                                    _back_norm_l = torch.linalg.norm(_back_base_l)
                                    if float(_back_norm_l) > 1e-9:
                                        _back_base_l = _back_base_l / _back_norm_l
                                        _to_pre_l = _pre_base_l - curr_eef_pos.squeeze()
                                        _lateral_l = torch.linalg.norm(
                                            _to_pre_l - _back_base_l
                                            * torch.dot(_to_pre_l, _back_base_l))
                                        _pregrasp_lateral_l = float(_lateral_l)
                                        # Move well clear before rotating the open fingers, then
                                        # advance to the near standoff before the final axial
                                        # approach. Rotation at the 0.18 m standoff still swept
                                        # the hand into the mug when initial orientation error
                                        # exceeded 45 degrees.
                                        _orient_error_l = float(torch.linalg.norm(
                                            curr_eef_pos.squeeze() - _orient_pre_base_l))
                                        _pre_error_l = float(torch.linalg.norm(
                                            curr_eef_pos.squeeze() - _pre_base_l))
                                        _stage_rotation_tolerance_l = float(os.environ.get(
                                            "SIMVLA_PREGRASP_ROT_TOL_DEG", "20.0"))
                                        arm_l_pregrasp_stage[env_idx] = advance_pregrasp_stage(
                                            int(arm_l_pregrasp_stage[env_idx]),
                                            _orient_error_l, float(rot_err_deg), _pre_error_l,
                                            rotation_tolerance_deg=(
                                                _stage_rotation_tolerance_l),
                                            final_rotation_tolerance_deg=float(rot_tol_l),
                                        )
                                        if int(arm_l_pregrasp_stage[env_idx]) < 2:
                                            _jog_target_l = _orient_pre_base_l
                                        elif int(arm_l_pregrasp_stage[env_idx]) == 2:
                                            _jog_target_l = _pre_base_l
                                _ik_term_l = env.action_manager.get_term("armL_action")
                                if int(arm_l_retry[env_idx]) == 1:
                                    # Preserve a moving reference after the one-time measured-pose
                                    # anchor. Re-anchoring on every control frame prevents a
                                    # lagging actuator from ever accumulating the requested motion.
                                    anchor_relative_ik_target(
                                        _ik_term_l, env_idx, curr_eef_pos.squeeze(),
                                        curr_eef_quat.squeeze())
                                _reference_pos_l = _ik_term_l._ik_controller.ee_pos_des[env_idx]
                                _d_l = _jog_target_l - _reference_pos_l
                                _n_l = float(torch.linalg.norm(_d_l))
                                if _n_l > 1e-9:
                                    pose_L[env_idx, :3] = _d_l * min(1.0, REACH_STEP_M / _n_l)
                                if (_pregrasp_lateral_l is not None
                                        and int(arm_l_pregrasp_stage[env_idx]) == 0):
                                    # Translate to the clear standoff without rotating the open
                                    # fingers beside the mug. Other reaches (especially drawer
                                    # handles) never advance this mug-only stage counter, so
                                    # applying it to them would suppress rotation forever.
                                    pose_L[env_idx, 3:6] = 0.0
                                else:
                                    _reference_quat_l = _ik_term_l._ik_controller.ee_quat_des[env_idx]
                                    _reference_rot_l = R.from_quat(
                                        _reference_quat_l.detach().cpu().numpy()[[1, 2, 3, 0]])
                                    _rv_l = torch.as_tensor(
                                        (l_goal * _reference_rot_l.inv()).as_rotvec(),
                                        dtype=pose_L.dtype, device=pose_L.device)
                                    _rn_l = float(torch.linalg.norm(_rv_l))
                                    if _rn_l > 1e-9:
                                        pose_L[env_idx, 3:6] = _rv_l * min(
                                            1.0, REACH_STEP_RAD / _rn_l)
                                env_cmd_indices_l[env_idx] = max(0, _plen_l - 1)
                                if int(arm_l_retry[env_idx]) % 40 == 1:
                                    _stage_diag_l = (
                                        "" if _pregrasp_lateral_l is None else
                                        f", pregrasp_lateral={_pregrasp_lateral_l:.4f}m, "
                                        f"stage={int(arm_l_pregrasp_stage[env_idx])}")
                                    if (_sk_now_l == SID_GRASP and _next_is_close_l
                                            and _cylindrical_mug_grasp_l
                                            and args_cli.obj_name_l != "none"):
                                        # Log the live contact geometry at coarse intervals. The
                                        # terminal jaw error alone cannot distinguish a bad tool
                                        # frame from a mug being nudged/tipped during approach.
                                        _probe_obj_l = env.scene.rigid_objects[
                                            args_cli.obj_name_l]
                                        _probe_pos_l = _probe_obj_l.data.body_pos_w[
                                            env_idx, 0]
                                        _probe_spawn_l = (
                                            _probe_obj_l.data.default_root_state[
                                                env_idx, :3]
                                            + env.scene.env_origins[env_idx]
                                        )
                                        _probe_q_l = _probe_obj_l.data.body_quat_w[
                                            env_idx, 0]
                                        _probe_up_z_l = 1.0 - 2.0 * (
                                            float(_probe_q_l[1]) ** 2
                                            + float(_probe_q_l[2]) ** 2
                                        )
                                        _probe_tilt_l = math.degrees(math.acos(
                                            max(-1.0, min(1.0, _probe_up_z_l))))
                                        _probe_jr_l, _probe_jl = jaw_body_indices(
                                            env.scene.articulations["robot"], "left")
                                        _probe_jaw_l = jaw_contact_midpoint(
                                            env.scene.articulations["robot"], env_idx, _probe_jr_l, _probe_jl)
                                        _probe_shift_l = float(torch.linalg.norm(
                                            _probe_pos_l[:2] - _probe_spawn_l[:2]))
                                        _stage_diag_l += (
                                            f", mug_shift={_probe_shift_l:.4f}m"
                                            f", mug_tilt={_probe_tilt_l:.1f}deg"
                                            f", jaw_xy={float(torch.linalg.norm(_probe_jaw_l[:2] - _probe_pos_l[:2])):.4f}m"
                                            f", jaw_dz={float(_probe_jaw_l[2] - _probe_pos_l[2]):+.4f}m"
                                        )
                                    print(f"[close_l] Env {env_idx}: A_l short by "
                                          f"{float(pos_err):.4f} m ({rot_err_deg:.1f} deg); "
                                          f"jogging in ({int(arm_l_retry[env_idx])}/"
                                          f"{REACH_RETRIES} steps{_stage_diag_l})", flush=True)
                            elif _miss_l:
                                if _next_is_close_l:
                                    _grasp_diag_l = ""
                                    if (_cylindrical_mug_grasp_l
                                            and args_cli.obj_name_l != "none"):
                                        _obj_diag_l = env.scene.rigid_objects[
                                            args_cli.obj_name_l].data.body_pos_w[env_idx, 0]
                                        _jr_diag_l, _jl_diag_l = jaw_body_indices(
                                            env.scene.articulations["robot"], "left")
                                        _jaw_diag_l = jaw_contact_midpoint(
                                            env.scene.articulations["robot"], env_idx, _jr_diag_l, _jl_diag_l)
                                        _grasp_diag_l = (
                                            f" eef_base={curr_eef_pos.squeeze().tolist()}"
                                            f" goal_base={ik_goals_l[env_idx].position.squeeze().tolist()}"
                                            f" eef_quat_wxyz={curr_eef_quat.squeeze().tolist()}"
                                            f" goal_quat_wxyz={ik_goals_l[env_idx].quaternion.squeeze().tolist()}"
                                            f" jaw_xy={float(torch.linalg.norm(_jaw_diag_l[:2] - _obj_diag_l[:2])):.4f}m"
                                            f" jaw_dz={float(_jaw_diag_l[2] - _obj_diag_l[2]):+.4f}m"
                                            f" jaw_minus_obj_xy={(_jaw_diag_l[:2] - _obj_diag_l[:2]).tolist()}"
                                            f" obj_xy={_obj_diag_l[:2].tolist()}")
                                    _route_diag_l = (
                                        "bounded cuRobo execution"
                                        if _direct_bodex_grasp_l else
                                        f"{int(arm_l_retry[env_idx])} jog steps")
                                    print(f"[grasp_l] Env {env_idx}: reach missed the gate after "
                                          f"{_route_diag_l} -- "
                                          f"pos_err={float(pos_err):.4f} m (limit {pos_threshold}), "
                                          f"rot_err={float(rot_err_deg):.1f} deg (limit {rot_tol_l})"
                                          f"{_grasp_diag_l}",
                                          flush=True)
                                else:
                                    print(
                                        f"[reach_l] Env {env_idx}: target missed after "
                                        f"{int(arm_l_retry[env_idx])} jog steps; "
                                        f"eef_quat_wxyz={curr_eef_quat.squeeze().tolist()} "
                                        f"goal_quat_wxyz={ik_goals_l[env_idx].quaternion.squeeze().tolist()} "
                                        f"pos_err={float(pos_err):.4f}m "
                                        f"rot_err={float(rot_err_deg):.1f}deg",
                                        flush=True,
                                    )
                                # Clear the budget with the episode. Left high, the next episode on
                                # this env starts already at REACH_RETRIES and can never jog at all
                                # -- a counter that only ever counts up silently disables itself.
                                arm_l_retry[env_idx] = 0
                                arm_l_pregrasp_stage[env_idx] = 0
                                arm_l_pregrasp_back_sign[env_idx] = 0
                                idx = torch.tensor([env_idx], device=device)
                                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, idx, reason="motion_fail_l") 
                            else:
                                # This env, not the whole batch -- the same fault as the right
                                # arm's.
                                if int(arm_l_retry[env_idx]) > 0:
                                    print(f"[OK_l] Env {env_idx}: EEF reached goal "
                                          f"(tol={pos_threshold:.4f}, pos_err={float(pos_err):.4f} m, "
                                          f"{rot_err_deg:.1f} deg, {int(arm_l_retry[env_idx])} jog "
                                          f"step(s))", flush=True)
                                arm_l_retry[env_idx] = 0
                                arm_l_pregrasp_stage[env_idx] = 0
                                arm_l_pregrasp_back_sign[env_idx] = 0
                                finished_mask[env_idx] = True
            
            # =================== Both Arms Motion ("A_b") ===================
            if torch.any(arm_b_mask):
                arm_b_indices = active_env_indices[arm_b_mask]

                needs_plan_indices_b = [
                    i.item()
                    for i in arm_b_indices
                    if (env_cmd_plans_r[i.item()] is None or env_cmd_plans_l[i.item()] is None)
                ]

                if needs_plan_indices_b:
                    needs_plan_indices_temp_b = torch.as_tensor(
                        needs_plan_indices_b, device=device, dtype=torch.long
                    )
                    goal_step_indices_b = env_goal_indices[needs_plan_indices_temp_b]
                    arm_b_goals = payloads_tensor[needs_plan_indices_temp_b, goal_step_indices_b].clone()   # (B, 14)

                    # A_b payload layout:
                    #   [:7]   = left arm goal
                    #   [7:14] = right arm goal
                    arm_l_goals_b = arm_b_goals[:, :7].clone()
                    arm_r_goals_b = arm_b_goals[:, 7:14].clone()

                    # A_b drives BOTH arms with ONE skill, so the two halves get the same branch.
                    # v1 flagged each half independently (slots 0/3 for the left, 7/10 for the
                    # right) and could in principle have disagreed; in the corpus it never does —
                    # all 25 A_b steps are (reset, reset), (place, place) or (pose, pose), and
                    # executor_dispatch.normalise_skill rejects a step whose halves disagree rather
                    # than guessing which arm wins.
                    skill_b = skill_ids_tensor[needs_plan_indices_temp_b, goal_step_indices_b]

                    reset_mask_l_b = (skill_b == SID_RESET)
                    place_mask_l_b = (skill_b == SID_PLACE)
                    normal_mask_l_b = ~(reset_mask_l_b | place_mask_l_b)

                    reset_mask_r_b = reset_mask_l_b
                    place_mask_r_b = place_mask_l_b
                    normal_mask_r_b = normal_mask_l_b

                    robot_data = env.scene.articulations["robot"].data

                    # Save current pose as reset target for NORMAL goals
                    if torch.any(normal_mask_l_b):
                        idx_l_norm = needs_plan_indices_temp_b[normal_mask_l_b]
                        sel = first_reset[idx_l_norm].view(-1)
                        m_skip = (sel == 1)
                        idx_store = idx_l_norm[~m_skip]
                        if idx_store.numel() > 0:
                            reset_l[idx_store, :3] = robot_data.body_pos_w[idx_store, l_eef_idx]
                            reset_l[idx_store, 3:7] = robot_data.body_quat_w[idx_store, l_eef_idx]
                        if args_cli.obj_name == "pot0":
                            reset_l[idx_l_norm, 3:7] = robot_data.body_quat_w[idx_l_norm, l_eef_idx]

                    if torch.any(normal_mask_r_b):
                        idx_r_norm = needs_plan_indices_temp_b[normal_mask_r_b]
                        sel = first_reset[idx_r_norm].view(-1)
                        m_skip = (sel == 1)
                        idx_store = idx_r_norm[~m_skip]
                        if idx_store.numel() > 0:
                            reset_r[idx_store, :3] = robot_data.body_pos_w[idx_store, r_eef_idx]
                            reset_r[idx_store, 3:7] = robot_data.body_quat_w[idx_store, r_eef_idx]
                        if args_cli.obj_name == "pot0":
                            reset_r[idx_r_norm, 3:7] = robot_data.body_quat_w[idx_r_norm, r_eef_idx]
                        first_reset[idx_r_norm] += 1

                    # Apply RESET mode
                    if torch.any(reset_mask_l_b):
                        idx_l_reset = needs_plan_indices_temp_b[reset_mask_l_b]
                        arm_l_goals_b[reset_mask_l_b] = reset_l[idx_l_reset]
                        arm_l_goals_b[reset_mask_l_b, :3], _ = base2world(env, home_l.unsqueeze(0).repeat(idx_l_reset.shape[0],1), home_l_quat.unsqueeze(0).repeat(idx_l_reset.shape[0],1), idx_l_reset)
                        arm_l_goals_b[reset_mask_l_b, :3] -= env.scene.env_origins[idx_l_reset, :3]

                    if torch.any(reset_mask_r_b):
                        idx_r_reset = needs_plan_indices_temp_b[reset_mask_r_b]
                        arm_r_goals_b[reset_mask_r_b] = reset_r[idx_r_reset]
                        arm_r_goals_b[reset_mask_r_b, :3], _ = base2world(env, home_r.unsqueeze(0).repeat(idx_r_reset.shape[0],1), home_r_quat.unsqueeze(0).repeat(idx_r_reset.shape[0],1), idx_r_reset)
                        arm_r_goals_b[reset_mask_r_b, :3] -= env.scene.env_origins[idx_r_reset, :3]
                    
                    # Apply Pot mode
                    # Was `arm_l_goals_b[:, 2] >= 1000` — a marker in the left arm's z slot, read
                    # after the reset overwrite so that a reset step could never also be a pot step.
                    # arm.pot and arm.reset are different skills, so that is now true by
                    # construction rather than by ordering.
                    pot_mask_b = (skill_b == SID_POT)
                    if torch.any(pot_mask_b):
                        idx_pot = needs_plan_indices_temp_b[pot_mask_b]
                        pot_first_mask = pot_first[idx_pot, 0] == 0
                        pot_second_mask = pot_first[idx_pot, 0] == 1
                        if torch.any(pot_first_mask):
                            idx_pot_first = idx_pot[pot_first_mask]
                            # Build batch-relative mask for pot_first envs
                            pot_first_batch = pot_mask_b.clone()
                            pot_first_batch[pot_mask_b] = pot_first_mask
                            if obj_init[args_cli.task]["direction"] == "N":
                                arm_l_goals_b[pot_first_batch, :3] = env.scene.rigid_objects["pot0"].data.body_pos_w[idx_pot_first].squeeze(1) - env.scene.env_origins[idx_pot_first, :3] + torch.tensor([-0.1 , 0.25, 0.15], device=device)
                                arm_l_goals_b[pot_first_batch, 3:7] = torch.tensor([-0.12886, -0.51292, 0.48674, -0.69527], device=device)
                                arm_r_goals_b[pot_first_batch, :3] = env.scene.rigid_objects["pot0"].data.body_pos_w[idx_pot_first].squeeze(1) - env.scene.env_origins[idx_pot_first, :3] + torch.tensor([-0.1, -0.25, 0.15], device=device)
                                arm_r_goals_b[pot_first_batch, 3:7] = torch.tensor([0.70323,-0.45452, 0.54168, 0.07391], device=device)
                        if torch.any(pot_second_mask):
                            idx_pot_second = idx_pot[pot_second_mask]
                            # Build batch-relative mask for pot_second envs
                            pot_second_batch = pot_mask_b.clone()
                            pot_second_batch[pot_mask_b] = pot_second_mask
                            if obj_init[args_cli.task]["direction"] == "N":
                                arm_l_goals_b[pot_second_batch, :3] = env.scene.rigid_objects["pot0"].data.body_pos_w[idx_pot_second].squeeze(1) - env.scene.env_origins[idx_pot_second, :3] + torch.tensor([-0.03 , 0.16, 0.1], device=device)
                                arm_l_goals_b[pot_second_batch, 3:7] = torch.tensor([-0.12886, -0.51292, 0.48674, -0.69527], device=device)
                                arm_r_goals_b[pot_second_batch, :3] = env.scene.rigid_objects["pot0"].data.body_pos_w[idx_pot_second].squeeze(1) - env.scene.env_origins[idx_pot_second, :3] + torch.tensor([-0.03, -0.16, 0.1], device=device)
                                arm_r_goals_b[pot_second_batch, 3:7] = torch.tensor([0.70323,-0.45452, 0.54168, 0.07391], device=device)
                        pot_first[idx_pot] += 1

                    
                    # Apply PLACE mode
                    if torch.any(place_mask_l_b):
                        idx_l_place = needs_plan_indices_temp_b[place_mask_l_b]
                        arm_l_goals_b[place_mask_l_b, 3:7] = robot_data.body_quat_w[idx_l_place, l_eef_idx]
                        if args_cli.obj_name == "pot0":
                            offset = 0
                            if obj_init[args_cli.task]["range_direction"] == "N":
                                offset = torch.tensor([0.3, 0.0, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "E":
                                offset = torch.tensor([0.0, -0.3, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "S":
                                offset = torch.tensor([-0.3, 0.0, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "W":
                                offset = torch.tensor([0.0, 0.3, -0.15], device=device) 
                            arm_l_goals_b[place_mask_l_b, 0:3] = robot_data.body_pos_w[idx_l_place, l_eef_idx] - env.scene.env_origins[idx_l_place] + offset
                        
                        reset_l[idx_l_place, :3] = robot_data.body_pos_w[idx_l_place, l_eef_idx]
                        reset_l[idx_l_place, 3:7] = robot_data.body_quat_w[idx_l_place, l_eef_idx]

                    if torch.any(place_mask_r_b):
                        idx_r_place = needs_plan_indices_temp_b[place_mask_r_b]
                        arm_r_goals_b[place_mask_r_b, 3:7] = robot_data.body_quat_w[idx_r_place, r_eef_idx]
                        if args_cli.obj_name == "pot0":
                            offset = 0
                            if obj_init[args_cli.task]["range_direction"] == "N":
                                offset = torch.tensor([0.3, 0.0, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "E":
                                offset = torch.tensor([0.0, -0.3, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "S":
                                offset = torch.tensor([-0.3, 0.0, -0.15], device=device) 
                            elif obj_init[args_cli.task]["range_direction"] == "W":
                                offset = torch.tensor([0.0, 0.3, -0.15], device=device) 
                            arm_r_goals_b[place_mask_r_b, 0:3] = robot_data.body_pos_w[idx_r_place, r_eef_idx] - env.scene.env_origins[idx_r_place] + offset
                        
                        reset_r[idx_r_place, :3] = robot_data.body_pos_w[idx_r_place, r_eef_idx]
                        reset_r[idx_r_place, 3:7] = robot_data.body_quat_w[idx_r_place, r_eef_idx]

                    for i, env_idx in enumerate(needs_plan_indices_b):
                        _repose_world_to_base("l", env_idx)
                        _repose_world_to_base("r", env_idx)
                        sim_data = env.scene.articulations["robot"].data
                        env_origin_pos = env.scene.env_origins[env_idx]

                        # ---------------- RIGHT ARM PLAN ----------------
                        joint_pos_r = sim_data.joint_pos[env_idx, r_j_index].tolist()
                        joint_vel_r = sim_data.joint_vel[env_idx, r_j_index].tolist()

                        # Keep the model's retract tensor intact (see single-arm path).

                        goal_env_r = arm_r_goals_b[i]
                        goal_pos_w_r = goal_env_r[:3] + env_origin_pos
                        goal_quat_w_r = goal_env_r[3:7]

                        goal_ee_pose_b_r, goal_ee_quat_b_r = world2base(
                            env, goal_pos_w_r, goal_quat_w_r, env_idx
                        )

                        cu_js_r = JointState(
                            position=tensor_args.to_device(joint_pos_r),
                            velocity=tensor_args.to_device(joint_vel_r),
                            acceleration=tensor_args.to_device(joint_vel_r) * 0.0,
                            jerk=tensor_args.to_device(joint_vel_r) * 0.0,
                            joint_names=r_j_names,
                        )
                        cu_js_r = cu_js_r.get_ordered_joint_state(motion_gen_r.kinematics.joint_names)

                        ik_goal_r = Pose(position=goal_ee_pose_b_r, quaternion=goal_ee_quat_b_r)

                        result_r = motion_gen_r.plan_single(
                            cu_js_r.unsqueeze(0), ik_goal_r, plan_config
                        )

                        # ---------------- LEFT ARM PLAN ----------------
                        joint_pos_l = sim_data.joint_pos[env_idx, l_j_index].tolist()
                        joint_vel_l = sim_data.joint_vel[env_idx, l_j_index].tolist()

                        # Keep the model's retract tensor intact (see single-arm path).

                        goal_env_l = arm_l_goals_b[i]
                        goal_pos_w_l = goal_env_l[:3] + env_origin_pos
                        goal_quat_w_l = goal_env_l[3:7]

                        goal_ee_pose_b_l, goal_ee_quat_b_l = world2base(
                            env, goal_pos_w_l, goal_quat_w_l, env_idx
                        )

                        cu_js_l = JointState(
                            position=tensor_args.to_device(joint_pos_l),
                            velocity=tensor_args.to_device(joint_vel_l),
                            acceleration=tensor_args.to_device(joint_vel_l) * 0.0,
                            jerk=tensor_args.to_device(joint_vel_l) * 0.0,
                            joint_names=l_j_names,
                        )
                        cu_js_l = cu_js_l.get_ordered_joint_state(motion_gen_l.kinematics.joint_names)

                        ik_goal_l = Pose(position=goal_ee_pose_b_l, quaternion=goal_ee_quat_b_l)

                        result_l = motion_gen_l.plan_single(
                            cu_js_l.unsqueeze(0), ik_goal_l, plan_config
                        )

                        # ---------------- HANDLE PLAN RESULT ----------------
                        if result_r.success.item() and result_l.success.item():
                            cmd_plan_r = result_r.get_interpolated_plan()
                            cmd_plan_r = motion_gen_r.get_full_js(cmd_plan_r)

                            cmd_plan_l = result_l.get_interpolated_plan()
                            cmd_plan_l = motion_gen_l.get_full_js(cmd_plan_l)

                            plan_j_names_r = cmd_plan_r.joint_names
                            common_j_names_r_b = [name for name in r_j_names if name in plan_j_names_r]
                            env_cmd_plans_r[env_idx] = cmd_plan_r.get_ordered_joint_state(common_j_names_r_b)
                            if env_cmd_plans_r[env_idx].shape[0] > 130:
                                env_cmd_plans_r[env_idx] = env_cmd_plans_r[env_idx][:130]

                            plan_j_names_l = cmd_plan_l.joint_names
                            common_j_names_l_b = [name for name in l_j_names if name in plan_j_names_l]
                            env_cmd_plans_l[env_idx] = cmd_plan_l.get_ordered_joint_state(common_j_names_l_b)
                            if env_cmd_plans_l[env_idx].shape[0] > 130:
                                env_cmd_plans_l[env_idx] = env_cmd_plans_l[env_idx][:130]

                            env_cmd_indices_r[env_idx] = 0
                            env_cmd_indices_l[env_idx] = 0

                            ik_goals_r[env_idx] = ik_goal_r
                            ik_goals_l[env_idx] = ik_goal_l

                        else:
                            idx = torch.tensor([env_idx], device=device)
                            _reset_envs(
                                env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,
                                env_goal_indices, env_cmd_indices_r, env_cmd_indices_l,
                                grasp_checked_r, grasp_checked_l, nav_phase,
                                env_cmd_pause_l, env_cmd_pause_trig_l,
                                env_cmd_pause_r, env_cmd_pause_trig_r,
                                drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l,
                                new_gripper_commands_L, new_gripper_commands_R,
                                timestep, env_cmd_plans_r, env_cmd_plans_l,
                                ik_goals_r, ik_goals_l,
                                idx, reason="plan_fail_b"
                            )

                paused_mask_r_b = env_cmd_pause_r[arm_b_indices] > 0
                if torch.any(paused_mask_r_b):
                    paused_envs_r_b = arm_b_indices[paused_mask_r_b]
                    env_cmd_pause_r[paused_envs_r_b] -= 1

                paused_mask_l_b = env_cmd_pause_l[arm_b_indices] > 0
                if torch.any(paused_mask_l_b):
                    paused_envs_l_b = arm_b_indices[paused_mask_l_b]
                    env_cmd_pause_l[paused_envs_l_b] -= 1

                has_plan_mask_r_b = torch.tensor(
                    [env_cmd_plans_r[i.item()] is not None for i in arm_b_indices],
                    device=device
                )
                has_plan_mask_l_b = torch.tensor(
                    [env_cmd_plans_l[i.item()] is not None for i in arm_b_indices],
                    device=device
                )

                not_paused_mask_r_b = (env_cmd_pause_r[arm_b_indices] == 0)
                not_paused_mask_l_b = (env_cmd_pause_l[arm_b_indices] == 0)

                can_run_mask_b = (
                    has_plan_mask_r_b
                    & has_plan_mask_l_b
                    & not_paused_mask_r_b
                    & not_paused_mask_l_b
                )

                if torch.any(can_run_mask_b):
                    executing_indices_b = arm_b_indices[can_run_mask_b]

                    plans_to_exec_r_b = [env_cmd_plans_r[i.item()] for i in executing_indices_b]
                    plans_to_exec_l_b = [env_cmd_plans_l[i.item()] for i in executing_indices_b]

                    indices_to_exec_r_b = env_cmd_indices_r[executing_indices_b]
                    indices_to_exec_l_b = env_cmd_indices_l[executing_indices_b]

                    plan_lengths_r_b = torch.tensor(
                        [len(p) for p in plans_to_exec_r_b],
                        device=device,
                        dtype=torch.long,
                    )
                    plan_lengths_l_b = torch.tensor(
                        [len(p) for p in plans_to_exec_l_b],
                        device=device,
                        dtype=torch.long,
                    )

                    pause_at_r_b = torch.clamp((plan_lengths_r_b.float() * 0.5).floor().long(), min=1)
                    pause_at_l_b = torch.clamp((plan_lengths_l_b.float() * 0.5).floor().long(), min=1)

                    already_trig_r_b = env_cmd_pause_trig_r[executing_indices_b]
                    already_trig_l_b = env_cmd_pause_trig_l[executing_indices_b]

                    should_pause_r_b = (~already_trig_r_b) & (indices_to_exec_r_b >= pause_at_r_b)
                    should_pause_l_b = (~already_trig_l_b) & (indices_to_exec_l_b >= pause_at_l_b)

                    should_pause_b = should_pause_r_b | should_pause_l_b

                    if torch.any(should_pause_b):
                        pause_envs_b = executing_indices_b[should_pause_b]
                        env_cmd_pause_r[pause_envs_b] = args_cli.action_chunk_size + 1
                        env_cmd_pause_l[pause_envs_b] = args_cli.action_chunk_size + 1
                        env_cmd_pause_trig_r[pause_envs_b] = True
                        env_cmd_pause_trig_l[pause_envs_b] = True

                        keep_b = ~should_pause_b
                        executing_indices_b = executing_indices_b[keep_b]
                        indices_to_exec_r_b = indices_to_exec_r_b[keep_b]
                        indices_to_exec_l_b = indices_to_exec_l_b[keep_b]
                        plans_to_exec_r_b = [p for p, k in zip(plans_to_exec_r_b, keep_b.tolist()) if k]
                        plans_to_exec_l_b = [p for p, k in zip(plans_to_exec_l_b, keep_b.tolist()) if k]
                        plan_lengths_r_b = plan_lengths_r_b[keep_b]
                        plan_lengths_l_b = plan_lengths_l_b[keep_b]

                    if executing_indices_b.numel() > 0:
                        delta_poses_r_b = compute_eef_deltas_batched(
                            motion_gen=motion_gen_r,
                            plans=plans_to_exec_r_b,
                            cmd_indices=indices_to_exec_r_b,
                            joint_names=r_j_names,
                            eef_link_name="ee_link1",
                        )

                        delta_poses_l_b = compute_eef_deltas_batched(
                            motion_gen=motion_gen_l,
                            plans=plans_to_exec_l_b,
                            cmd_indices=indices_to_exec_l_b,
                            joint_names=l_j_names,
                            eef_link_name="ee_link2",
                        )

                        pose_R[executing_indices_b] = delta_poses_r_b
                        pose_L[executing_indices_b] = delta_poses_l_b

                        done_r_before = env_cmd_indices_r[executing_indices_b] >= (plan_lengths_r_b - 1)
                        done_l_before = env_cmd_indices_l[executing_indices_b] >= (plan_lengths_l_b - 1)

                        if torch.any(~done_r_before):
                            env_cmd_indices_r[executing_indices_b[~done_r_before]] += 1
                        if torch.any(~done_l_before):
                            env_cmd_indices_l[executing_indices_b[~done_l_before]] += 1

                        done_r_after = env_cmd_indices_r[executing_indices_b] >= (plan_lengths_r_b - 1)
                        done_l_after = env_cmd_indices_l[executing_indices_b] >= (plan_lengths_l_b - 1)

                        finished_exec_mask_b = done_r_after & done_l_after

                        pos_threshold_r = 0.12
                        pos_threshold_l = 0.12
                        rot_threshold_deg = 10.0

                        for env_idx in executing_indices_b[finished_exec_mask_b].tolist():
                            curr_eef_pos_r, curr_eef_quat_r = world2base(
                                env,
                                env.scene.articulations["robot"].data.body_pos_w[env_idx, r_eef_idx],
                                env.scene.articulations["robot"].data.body_quat_w[env_idx, r_eef_idx],
                                env_idx,
                            )
                            pos_err_r = torch.norm(curr_eef_pos_r - ik_goals_r[env_idx].position)
                            r_eef = R.from_quat(curr_eef_quat_r.cpu().numpy()[[1, 2, 3, 0]])
                            r_goal = R.from_quat(
                                ik_goals_r[env_idx].quaternion.squeeze(0).cpu().numpy()[[1, 2, 3, 0]]
                            )
                            rot_err_deg_r = np.linalg.norm((r_goal.inv() * r_eef).as_rotvec()) * 180.0 / np.pi

                            curr_eef_pos_l, curr_eef_quat_l = world2base(
                                env,
                                env.scene.articulations["robot"].data.body_pos_w[env_idx, l_eef_idx],
                                env.scene.articulations["robot"].data.body_quat_w[env_idx, l_eef_idx],
                                env_idx,
                            )
                            pos_err_l = torch.norm(curr_eef_pos_l - ik_goals_l[env_idx].position)
                            l_eef = R.from_quat(curr_eef_quat_l.cpu().numpy()[[1, 2, 3, 0]])
                            l_goal = R.from_quat(
                                ik_goals_l[env_idx].quaternion.squeeze(0).cpu().numpy()[[1, 2, 3, 0]]
                            )
                            rot_err_deg_l = np.linalg.norm((l_goal.inv() * l_eef).as_rotvec()) * 180.0 / np.pi

                            if (
                                (pos_err_r > pos_threshold_r)
                                or (pos_err_l > pos_threshold_l)
                                # or (rot_err_deg_r > rot_threshold_deg)
                                # or (rot_err_deg_l > rot_threshold_deg)
                            ):
                                idx = torch.tensor([env_idx], device=device)
                                _reset_envs(
                                    env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,
                                    env_goal_indices, env_cmd_indices_r, env_cmd_indices_l,
                                    grasp_checked_r, grasp_checked_l, nav_phase,
                                    env_cmd_pause_l, env_cmd_pause_trig_l,
                                    env_cmd_pause_r, env_cmd_pause_trig_r,
                                    drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l,
                                    new_gripper_commands_L, new_gripper_commands_R,
                                    timestep, env_cmd_plans_r, env_cmd_plans_l,
                                    ik_goals_r, ik_goals_l,
                                    idx, reason="motion_fail_b"
                                )
                            else:
                                print(f"[OK] Env {env_idx}: BOTH EEFs reached goal within threshold.")
                                finished_mask[env_idx] = True


            # =================== Gripper Motion ("G_r" & "G_l") ===================
            if torch.any(grip_r_mask):
                grip_r_env_indices = active_env_indices[grip_r_mask]
                grip_r_goal_step_indices = env_goal_indices[grip_r_env_indices]
                grip_r_goals = payloads_tensor[grip_r_env_indices, grip_r_goal_step_indices]
                
                current_commands = (grip_r_goals[:, 0] > 0)
                new_gripper_commands_R[grip_r_env_indices] = current_commands
                robot = env.scene.articulations["robot"]
                r_gripperR_idx, r_gripperL_idx = jaw_body_indices(robot, "right")

                r_gripper_pos = robot.data.body_pos_w[grip_r_env_indices][:, [r_gripperR_idx, r_gripperL_idx]]
                r_dist_curr = torch.linalg.norm(r_gripper_pos[:, 0] - r_gripper_pos[:, 1], dim=1)

                is_stopped = (env_r_gripper_dist_prev[grip_r_env_indices] - r_dist_curr).abs() < 0.001
                # Count CONSECUTIVE still ticks; one is not enough (see GRIPPER_SETTLE_TICKS).
                env_r_gripper_still[grip_r_env_indices] = torch.where(
                    is_stopped,
                    env_r_gripper_still[grip_r_env_indices] + 1,
                    torch.zeros_like(env_r_gripper_still[grip_r_env_indices]),
                )
                env_r_gripper_ticks[grip_r_env_indices] += 1
                _forced_r = current_commands & (
                    env_r_gripper_ticks[grip_r_env_indices] >= GRIPPER_CLOSE_MAX_TICKS)
                for _e in grip_r_env_indices[
                    current_commands & (env_r_gripper_ticks[grip_r_env_indices] == GRIPPER_CLOSE_MAX_TICKS)
                ].tolist():
                    print(f"[gripper] env{_e} right close still moving after "
                          f"{GRIPPER_CLOSE_MAX_TICKS} ticks; advancing close step", flush=True)
                settled = (env_r_gripper_still[grip_r_env_indices] >= GRIPPER_SETTLE_TICKS) | _forced_r
                if _open_target_tol is not None:
                    _grip_action_r = env.action_manager.get_term("gripperR_action")
                    _open_reached_r = gripper_open_target_reached(
                        robot.data.joint_pos[grip_r_env_indices][:, _grip_action_r._joint_ids],
                        _grip_action_r._open_command, _open_target_tol)
                    settled = settled & (current_commands | _open_reached_r)
                if args_cli.sub_grasp_idx_r != GRASP_CHECK_DISABLED:
                    _check_r = (
                        current_commands & settled
                        & (grip_r_goal_step_indices == args_cli.sub_grasp_idx_r)
                        & (~grasp_checked_r[grip_r_env_indices])
                    )
                    _verify_subgrasp("r", grip_r_env_indices[_check_r])
                if _lift_steps > 0:
                    # An OPEN command finishes exactly as before. A CLOSE finishes only once the
                    # lift countdown has run dry, so the jog happens while the step is still
                    # current and pose_R is still being consumed.
                    _lr = env_lift_r[grip_r_env_indices]
                    _check_ready_r = (
                        (args_cli.sub_grasp_idx_r == GRASP_CHECK_DISABLED)
                        | grasp_checked_r[grip_r_env_indices]
                    )
                    _arming = current_commands & settled & (_lr == 0) & _check_ready_r
                    for _e in grip_r_env_indices[_arming].tolist():
                        try:
                            _obj = env.scene.rigid_objects[args_cli.obj_name]
                            _obj_z0[_e] = float(_obj.data.body_pos_w[_e, 0, 2]
                                                - env.scene.env_origins[_e, 2])
                            # AND THE PAD-TO-OBJECT MISS AT THE MOMENT THE JAWS CLOSE, which is
                            # the only moment that decides the grasp. The existing reading is
                            # taken at close-FINISH, after the 0.20 m lift, and the differential
                            # IK drifts in xy while executing a commanded pure-z jog -- so it
                            # describes where the hand ended, not where it grasped. Measuring the
                            # wrong instant is the same error that made eef_z look like proof the
                            # lift worked.
                            _pm0 = jaw_contact_midpoint(robot, _e, r_gripperR_idx, r_gripperL_idx)
                            _op0 = _obj.data.body_pos_w[_e, 0]
                            _miss0[_e] = float(torch.linalg.norm(_pm0[:2] - _op0[:2]))
                            _padz0[_e] = float(_pm0[2] - env.scene.env_origins[_e, 2])
                            # THE GOAL'S OWN WORLD POSITION vs THE BOTTLE'S. The reach reports
                            # converging to 0.005-0.026 m of the goal while the pads land a
                            # median 0.069 m from the bottle. Both cannot be true. Either the
                            # goal is not at the bottle -- it is authored at emit time and the
                            # bottle sits 0.03 m higher at runtime than the facts file says, so
                            # staleness is plausible -- or the pads are not where ee_link1 says
                            # they are. This distinguishes them outright.
                            _gwr = _goal_w_r.get(_e)
                            if _gwr is not None:
                                _goal_miss[_e] = float(torch.linalg.norm(
                                    _gwr[0].squeeze()[:2] - _op0[:2]))
                            # IS THE BOTTLE STILL UPRIGHT? obj_z reads 0.9818 against a 0.952
                            # counter, and 0.952 + the bottle's 0.032 radius is 0.984 -- exactly
                            # where a bottle lying on its SIDE would sit. If it topples at spawn
                            # then the grasp, authored for an upright bottle, is aimed at a pose
                            # the object no longer occupies, which is what GOAL_TO_OBJ=0.115 says.
                            _oq = _obj.data.body_quat_w[_e, 0]
                            _w, _x, _y, _z2 = (float(_oq[0]), float(_oq[1]),
                                               float(_oq[2]), float(_oq[3]))
                            # the object's own +Z expressed in world; tilt is its angle from up
                            _upz = 1.0 - 2.0 * (_x * _x + _y * _y)
                            _tilt[_e] = math.degrees(math.acos(max(-1.0, min(1.0, _upz))))
                        except Exception:                               # noqa: BLE001
                            pass
                    _lr[_arming] = _lift_steps + 1
                    _lifting = _lr > 0
                    pose_R[grip_r_env_indices[_lifting], 2] = _LIFT_STEP_M
                    _lr[_lifting] -= 1
                    env_lift_r[grip_r_env_indices] = _lr
                    _trace_lift("r", grip_r_env_indices[_lifting], _lr[_lifting])
                    _done = (settled & ~current_commands) | (current_commands & _lifting & (_lr == 0))
                    # Report the object's height as the close completes: the hand rising does
                    # not prove that it carried the object, and the next plan depends on that.
                    for _e in grip_r_env_indices[_done & current_commands].tolist():
                        _z = float(robot.data.body_pos_w[_e, r_eef_idx, 2]
                                   - env.scene.env_origins[_e, 2])
                        # THE OBJECT'S HEIGHT TOO, WHICH IS THE ONLY THING THAT SETTLES WHETHER
                        # THE HAND CLOSED ON THE BOTTLE OR BESIDE IT. eef_z rising proves the ARM
                        # lifted; it says nothing about what it lifted. The success predicate has
                        # been reading obj_z=F and obj_near_prim=F in every sample of every run,
                        # which is exactly what a bottle still standing on the counter looks like.
                        _lifted_by = float("nan")
                        try:
                            _ob = env.scene.rigid_objects[args_cli.obj_name]
                            _oz = float(_ob.data.body_pos_w[_e, 0, 2]
                                        - env.scene.env_origins[_e, 2])
                            _lifted_by = _oz - _obj_z0.get(_e, _oz)
                            _og = f"obj_z={_oz:.4f} lifted_by={_lifted_by:+.4f}"
                        except Exception as _exc:                       # noqa: BLE001
                            _og = f"obj_z=unreadable ({type(_exc).__name__})"
                        # AND THE JAW GAP, which is the difference between "the hand was in the
                        # wrong place" and "the hand was right and the jaws shut through thin
                        # air". The bottle is 0.064 m across and the jaws span 0.1147 m open, so
                        # a gap settling near 0.064 means the pads are ON it and something else
                        # loses it; a gap near 0.000 means they met with nothing in between.
                        _gap = float(r_dist_curr[(grip_r_env_indices == _e).nonzero()[0, 0]]) \
                            if (grip_r_env_indices == _e).any() else float("nan")
                        # WHERE THE PADS ARE RELATIVE TO THE BOTTLE, which is the measurement
                        # every previous guess stood in for. jaw_gap says the jaws met with
                        # nothing between them; it does NOT say whether the hand was at the
                        # object. This does:
                        #   miss ~0.02 m -> the hand is there and the object was displaced
                        #   miss ~0.15 m -> the hand is not at the object at all, and the reach
                        #                   'succeeding' means it reached the WRONG POSE
                        try:
                            _pm = jaw_contact_midpoint(robot, _e, r_gripperR_idx, r_gripperL_idx)
                            _ob2 = env.scene.rigid_objects[args_cli.obj_name]
                            _op = _ob2.data.body_pos_w[_e, 0]
                            _miss = float(torch.linalg.norm(_pm[:2] - _op[:2]))
                            _pmz = float(_pm[2] - env.scene.env_origins[_e, 2])
                            _b0 = _base_at_plan.get(_e)
                            _bn = robot.data.body_pos_w[_e, base_link_idx, :2]
                            _drift = (float(torch.linalg.norm(_bn - _b0))
                                      if _b0 is not None else float("nan"))
                            _pd = (f"pad_mid=({float(_pm[0]):+.3f},{float(_pm[1]):+.3f},"
                                   f"{_pmz:+.3f}) obj_xy=({float(_op[0]):+.3f},"
                                   f"{float(_op[1]):+.3f}) MISS={_miss:.4f} "
                                   f"MISS_AT_CLOSE={_miss0.get(_e, float('nan')):.4f} "
                                   f"padz_at_close={_padz0.get(_e, float('nan')):.4f} "
                                   f"GOAL_TO_OBJ={_goal_miss.get(_e, float('nan')):.4f} "
                                   f"tilt_at_plan={_tilt_at_plan.get(_e, float('nan')):.1f}deg "
                                   f"obj_tilt={_tilt.get(_e, float('nan')):.1f}deg "
                                   f"base_drift={_drift:.4f}")
                        except Exception as _exc:                       # noqa: BLE001
                            _pd = f"pad_mid=unreadable ({type(_exc).__name__})"
                        print(f"[lift] env{_e} close finished at eef_z={_z:.4f} {_og} "
                              f"jaw_gap={_gap:.4f} {_pd} (lift budget {_lift_steps} steps)",
                              flush=True)
                        if int(env_sub_saved_r[_e]) == 2:
                            _pad_forces_r = _mug_finger_contact_forces(_e, "r")
                            _retained = post_lift_object_retained(
                                _lifted_by, _gap, pad_forces_n=_pad_forces_r)
                            if _retained:
                                goal_mgr.on_sub_success(
                                    torch.tensor([_e], dtype=torch.long, device=device), arm="r")
                                env_sub_saved_r[_e] = 1
                                print(f"[sub_grasp] env{_e} right-hand pose banked after "
                                      f"object lift { _lifted_by:+.4f}m and jaw gap "
                                      f"{_gap:.4f}m", flush=True)
                            else:
                                env_sub_saved_r[_e] = 0
                                post_lift_failed_envs.add(_e)
                                print(f"[sub_grasp] env{_e} right-hand pre-lift candidate "
                                      f"rejected after lift: object rise {_lifted_by:+.4f}m, "
                                      f"jaw gap {_gap:.4f}m, "
                                      f"pad_contact=[{_mug_pad_contact_diag(_e, 'r')}]",
                                      flush=True)
                else:
                    _done = settled
                finished_mask[grip_r_env_indices[_done]] = True
                # Clear for the next gripper step, so a later one starts its count from zero.
                env_r_gripper_still[grip_r_env_indices[_done]] = 0
                env_r_gripper_ticks[grip_r_env_indices[_done]] = 0

                env_r_gripper_dist_prev[grip_r_env_indices] = r_dist_curr
            
            if torch.any(grip_l_mask):
                grip_l_env_indices = active_env_indices[grip_l_mask]
                grip_l_goal_step_indices = env_goal_indices[grip_l_env_indices]
                grip_l_goals = payloads_tensor[grip_l_env_indices, grip_l_goal_step_indices]

                current_commands = (grip_l_goals[:, 0] > 0)
                new_gripper_commands_L[grip_l_env_indices] = current_commands
                robot = env.scene.articulations["robot"]
                l_gripperR_idx, l_gripperL_idx = jaw_body_indices(robot, "left")

                l_gripper_pos = robot.data.body_pos_w[grip_l_env_indices][:, [l_gripperR_idx, l_gripperL_idx]]
                l_dist_curr = torch.linalg.norm(l_gripper_pos[:, 0] - l_gripper_pos[:, 1], dim=1)

                is_stopped = (env_l_gripper_dist_prev[grip_l_env_indices] - l_dist_curr).abs() < 0.001
                # Count CONSECUTIVE still ticks; one is not enough (see GRIPPER_SETTLE_TICKS).
                env_l_gripper_still[grip_l_env_indices] = torch.where(
                    is_stopped,
                    env_l_gripper_still[grip_l_env_indices] + 1,
                    torch.zeros_like(env_l_gripper_still[grip_l_env_indices]),
                )
                env_l_gripper_ticks[grip_l_env_indices] += 1
                _forced_l = current_commands & (
                    env_l_gripper_ticks[grip_l_env_indices] >= GRIPPER_CLOSE_MAX_TICKS)
                for _e in grip_l_env_indices[
                    current_commands & (env_l_gripper_ticks[grip_l_env_indices] == GRIPPER_CLOSE_MAX_TICKS)
                ].tolist():
                    print(f"[gripper] env{_e} left close still moving after "
                          f"{GRIPPER_CLOSE_MAX_TICKS} ticks; continuing", flush=True)
                settled = (env_l_gripper_still[grip_l_env_indices] >= GRIPPER_SETTLE_TICKS) | _forced_l
                if _open_target_tol is not None:
                    _grip_action_l = env.action_manager.get_term("gripperL_action")
                    _open_reached_l = gripper_open_target_reached(
                        robot.data.joint_pos[grip_l_env_indices][:, _grip_action_l._joint_ids],
                        _grip_action_l._open_command, _open_target_tol)
                    settled = settled & (current_commands | _open_reached_l)
                if args_cli.sub_grasp_idx_l != GRASP_CHECK_DISABLED:
                    _check_l = (
                        current_commands & settled
                        & (grip_l_goal_step_indices == args_cli.sub_grasp_idx_l)
                        & (~grasp_checked_l[grip_l_env_indices])
                    )
                    _verify_subgrasp("l", grip_l_env_indices[_check_l])
                if _lift_steps > 0 and _lift_left:
                    _ll = env_lift_l[grip_l_env_indices]
                    _check_ready_l = (
                        (args_cli.sub_grasp_idx_l == GRASP_CHECK_DISABLED)
                        | grasp_checked_l[grip_l_env_indices]
                    )
                    _arming_l = current_commands & settled & (_ll == 0) & _check_ready_l
                    for _e in grip_l_env_indices[_arming_l].tolist():
                        try:
                            _obj_z0_l[_e] = float(
                                env.scene.rigid_objects[args_cli.obj_name_l]
                                .data.body_pos_w[_e, 0, 2] - env.scene.env_origins[_e, 2])
                        except (KeyError, AttributeError, IndexError):
                            _obj_z0_l[_e] = float("nan")
                    _ll[_arming_l] = _lift_steps + 1
                    _lifting_l = _ll > 0
                    pose_L[grip_l_env_indices[_lifting_l], 2] = _LIFT_STEP_M
                    _ll[_lifting_l] -= 1
                    env_lift_l[grip_l_env_indices] = _ll
                    _trace_lift("l", grip_l_env_indices[_lifting_l], _ll[_lifting_l])
                    _done_l = (settled & ~current_commands) | (current_commands & _lifting_l & (_ll == 0))
                    for _e in grip_l_env_indices[_done_l & current_commands].tolist():
                        _eef_z = float(robot.data.body_pos_w[_e, l_eef_idx, 2]
                                       - env.scene.env_origins[_e, 2])
                        _obj_name_l = args_cli.obj_name_l
                        _lifted_by_l = float("nan")
                        try:
                            _obj_z = float(env.scene.rigid_objects[_obj_name_l].data.body_pos_w[_e, 0, 2]
                                           - env.scene.env_origins[_e, 2])
                            _lifted_by_l = _obj_z - _obj_z0_l.get(_e, _obj_z)
                            _obj_detail = (f"{_obj_name_l}_z={_obj_z:.4f} "
                                           f"lifted_by={_lifted_by_l:+.4f}")
                        except (KeyError, AttributeError):
                            _obj_detail = f"{_obj_name_l}_z=unavailable"
                        _gap_l = float(l_dist_curr[(grip_l_env_indices == _e).nonzero()[0, 0]]) \
                            if (grip_l_env_indices == _e).any() else float("nan")
                        _contact_l = (_mug_pad_contact_diag(_e, "l")
                                      if args_cli.obj_name_l.startswith("mug") else "not-configured")
                        print(f"[lift] env{_e} left close finished at eef_z={_eef_z:.4f} "
                              f"{_obj_detail} jaw_gap={_gap_l:.4f} "
                              f"pad_contact=[{_contact_l}] "
                              f"(lift budget {_lift_steps} steps)", flush=True)
                        if int(env_sub_saved_l[_e]) == 2:
                            _pad_forces_l = _mug_finger_contact_forces(_e, "l")
                            _retained_l = post_lift_object_retained(
                                _lifted_by_l, _gap_l, pad_forces_n=_pad_forces_l)
                            if _retained_l:
                                goal_mgr.on_sub_success(
                                    torch.tensor([_e], dtype=torch.long, device=device), arm="l")
                                env_sub_saved_l[_e] = 1
                                print(f"[sub_grasp] env{_e} left-hand pose banked after "
                                      f"object lift {_lifted_by_l:+.4f}m and jaw gap "
                                      f"{_gap_l:.4f}m", flush=True)
                            else:
                                env_sub_saved_l[_e] = 0
                                post_lift_failed_envs.add(_e)
                                print(f"[sub_grasp] env{_e} left-hand pre-lift candidate "
                                      f"rejected after lift: object rise {_lifted_by_l:+.4f}m, "
                                      f"jaw gap {_gap_l:.4f}m, "
                                      f"pad_contact=[{_mug_pad_contact_diag(_e, 'l')}]", flush=True)
                else:
                    _done_l = settled
                finished_mask[grip_l_env_indices[_done_l]] = True
                # Clear for the next gripper step, so a later one starts its count from zero.
                env_l_gripper_still[grip_l_env_indices[_done_l]] = 0
                env_l_gripper_ticks[grip_l_env_indices[_done_l]] = 0

                env_l_gripper_dist_prev[grip_l_env_indices] = l_dist_curr
            
            # A failed physical lift is a failed attempt, not a completed close
            # followed by an empty-handed reset/navigation/place sequence.
            if post_lift_failed_envs:
                failed_lift_ids = torch.tensor(sorted(post_lift_failed_envs), device=device,
                                               dtype=torch.long)
                _reset_envs(
                    env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx,
                    env_goal_indices, env_cmd_indices_r, env_cmd_indices_l,
                    grasp_checked_r, grasp_checked_l, nav_phase,
                    env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r,
                    drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l,
                    new_gripper_commands_L, new_gripper_commands_R, timestep,
                    env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l,
                    failed_lift_ids, reason="post_lift_object_not_retained")
                finished_mask[failed_lift_ids] = False
                pose_L[failed_lift_ids] = 0
                pose_R[failed_lift_ids] = 0
                delta_pose_base[failed_lift_ids] = 0
                env_lift_r[failed_lift_ids] = 0
                env_lift_l[failed_lift_ids] = 0

            # env step
            # Hold the parked base through grasp/close/reset. Zero velocity alone left
            # about 7 degrees of yaw drift during job 2395228, moving the target 9 cm
            # in base coordinates after cuRobo had already planned the reach.
            for _hold_env, _hold_anchor in list(_direct_base_anchors.items()):
                _hold_step = int(env_goal_indices[_hold_env])
                if (int(timestep[_hold_env]) <= 1 or _hold_step >= max_sequence_length
                        or int(task_ids_tensor[_hold_env, _hold_step]) in (N_ID, NS_ID)):
                    del _direct_base_anchors[_hold_env]
                    continue
                _hold_pos = robot.data.body_pos_w[_hold_env, base_link_idx]
                _hold_yaw = euler_xyz_from_quat(
                    robot.data.body_quat_w[_hold_env:_hold_env + 1, base_link_idx])[2][0]
                delta_pose_base[_hold_env] = torch.tensor(planar_hold_command(
                    _hold_anchor, (float(_hold_pos[0]), float(_hold_pos[1]), float(_hold_yaw))),
                    dtype=delta_pose_base.dtype, device=device)
            if aiworker_lift_ids is not None:
                # No ActionManager term owns this joint. Isaac initializes an uncovered joint's
                # position target at zero, which would pull the new -0.30 m reset home back to the
                # top while cuRobo plans with the column locked below. Reassert the reset target
                # after every reset and before every physics step.
                robot.set_joint_position_target(
                    aiworker_lift_target, joint_ids=aiworker_lift_ids)
            actions = pre_process_actions(env,
                pose_L, new_gripper_commands_L,
                pose_R, new_gripper_commands_R,
                delta_pose_base,
                args_cli.robot
            )
            if _loaded_home_joint_targets:
                # Apply after ordinary IK terms on every substep. Gripper, base,
                # lift and right-arm controllers retain ownership of their joints.
                # Do not hand a loaded arm back to redundant Cartesian IK during
                # its midpoint pause or the following base turn. Release ownership
                # on reset, opening, or the next arm skill, never across episodes.
                _motor_envs = []
                for i in list(_loaded_home_motor_overrides):
                    _step = int(env_goal_indices[i])
                    _valid_step = 0 <= _step < max_sequence_length
                    _is_home = _valid_step and int(skill_ids_tensor[i, _step]) == SID_RESET
                    _is_nav = _valid_step and int(task_ids_tensor[i, _step]) in (N_ID, NS_ID)
                    _final_open_home = (i in arm_l_final_home_motor_envs
                                        and _is_home and int(timestep[i]) > 1)
                    if not (_final_open_home or loaded_home_hold_allowed(
                            bool(new_gripper_commands_L[i]), int(timestep[i]),
                            is_home_reset=_is_home, is_navigation=_is_nav)):
                        del _loaded_home_motor_overrides[i]
                        continue
                    _motor_envs.append(i)
                    if _is_nav or int(env_cmd_pause_l[i]) > 0:
                        _held_pos, _held_quat = world2base(
                            env, robot.data.body_pos_w[i, l_eef_idx],
                            robot.data.body_quat_w[i, l_eef_idx], i)
                        anchor_relative_ik_target(env.action_manager.get_term("armL_action"),
                                                  i, _held_pos, _held_quat)
                if _motor_envs:
                    _motor_targets = robot.data.joint_pos.clone()
                    _motor_ids = _loaded_home_motor_overrides[_motor_envs[0]][0]
                    for _motor_env in _motor_envs:
                        _ids, _target = _loaded_home_motor_overrides[_motor_env]
                        if _ids != _motor_ids:
                            raise RuntimeError("Inconsistent loaded-home joint ordering")
                        _motor_targets[_motor_env, _motor_ids] = _target
                    env.action_manager.set_joint_position_replay_targets(
                        _motor_targets, joint_ids=_motor_ids, env_ids=_motor_envs)
                else:
                    env.action_manager.set_joint_position_replay_targets(None)
            # A manual retry can reset physics midway through this iteration.
            # Capture at the actual first-step boundary, after ALL retry branches,
            # not at the next loop's script timestep==1 (one action too late).
            _initial_env_ids = torch.where(env.episode_length_buf == 0)[0].tolist()
            ensure_initial_snapshot(env.recorder_manager, _initial_env_ids)
            for _env_id in _initial_env_ids:
                _initial_objects = {}
                for _name, _object in env.scene.rigid_objects.items():
                    _pos = _object.data.body_pos_w[_env_id, 0] - env.scene.env_origins[_env_id]
                    _quat = _object.data.body_quat_w[_env_id, 0]
                    _initial_objects[_name] = torch.cat((_pos, _quat)).tolist()
                initial_objects_by_env[_env_id] = _initial_objects
            obv = env.step(actions, env_goal_indices.clone())
            # Follow the object beyond the lift gate: a retained lift alone does not
            # establish retention during arm retraction, navigation, or placement.
            if os.environ.get("SIMVLA_TRACE_LIFT", "0") == "1" and global_frames % 20 == 0:
                for _arm, _object_name, _eef, _closed in (
                    ("l", args_cli.obj_name_l, l_eef_idx, new_gripper_commands_L),
                    ("r", args_cli.obj_name, r_eef_idx, new_gripper_commands_R),
                ):
                    if _object_name not in env.scene.rigid_objects:
                        continue
                    _trace_gripper = env.action_manager.get_term(
                        "gripperL_action" if _arm == "l" else "gripperR_action")
                    for _env in active_env_indices.tolist():
                        print("[carry-trace] " + json.dumps({
                            "frame": global_frames, "env_id": _env, "arm": _arm,
                            "step": int(env_goal_indices[_env]),
                            "closed": bool(_closed[_env]),
                            "finger_joint_pos": robot.data.joint_pos[_env, _trace_gripper._joint_ids].tolist(),
                            "finger_joint_targets": robot.data.joint_pos_target[_env, _trace_gripper._joint_ids].tolist(),
                            "base_pos_w": robot.data.body_pos_w[_env, base_link_idx].tolist(),
                            "eef_pos_w": robot.data.body_pos_w[_env, _eef].tolist(),
                            "object_pos_w": env.scene.rigid_objects[_object_name].data.body_pos_w[_env, 0].tolist(),
                            "finger_load_n": _mug_finger_contact_forces(_env, _arm),
                        }), flush=True)
            #env.scene.articulations["robot"]._joint_pos_target_sim (N,K)
            # ['base_prismatic_x_joint', 'base_prismatic_y_joint', 'base_revolute_z_joint', 'arm1_base_link_joint', 'arm2_base_link_joint', 'link11_joint', 'link21_joint', 'link12_joint', 'link22_joint', 'link13_joint', 'link23_joint', 'link14_joint', 'link24_joint', 'link15_joint', 'link25_joint', 'gripper1R_joint', 'gripper1_joint', 'gripper2R_joint', 'gripper2_joint']

            if args_cli.dump_frames:
                _dump_frame(env)

            # PER-STEP TILT TRACE THROUGH THE APPROACH, env 0 only.
            #
            # tilt_at_plan=0.0 and obj_tilt=90.0 say the bottle goes over between planning the
            # grasp and closing on it, but two endpoints cannot say WHICH MOTION does it. Seven
            # fixes were aimed at candidates inferred from those endpoints -- the planner, the
            # jog, the gripper, the approach axis -- and none of them was right. A per-step trace
            # names the frame it happens on and where the hand was at the time, which is the
            # difference between knowing and guessing.
            #
            # Prints only while it CHANGES, so a stable bottle costs one line, not thousands.
            if os.environ.get("SIMVLA_TILT_TRACE"):
                try:
                    _tt_ob = env.scene.rigid_objects[args_cli.obj_name]
                    _tt_q = _tt_ob.data.body_quat_w[0, 0]
                    _tt_up = 1.0 - 2.0 * (float(_tt_q[1]) ** 2 + float(_tt_q[2]) ** 2)
                    _tt = math.degrees(math.acos(max(-1.0, min(1.0, _tt_up))))
                    if abs(_tt - _tilt_trace_prev[0]) > 2.0:
                        _tt_o = _tt_ob.data.body_pos_w[0, 0] - env.scene.env_origins[0]
                        _tt_e = (robot.data.body_pos_w[0, r_eef_idx]
                                 - env.scene.env_origins[0])
                        # THE NEAREST GRIPPER BODY, not ee_link1. eef_to_obj was 0.158 m when
                        # the bottle started going over, and I read that as "nothing is in
                        # contact" -- but ee_link1 is a frame near the wrist, and the fingers
                        # reach well past it. A stability probe with the robot idle showed the
                        # bottle perfectly stable for 600 steps (worst tilt 0.1 deg), so the
                        # robot IS what knocks it, and this says which part gets close enough.
                        # Measured from the sim's body poses, which resolve correctly where the
                        # USD's authored link transforms do not.
                        # EVERY BODY, NOT SEVEN. The tuple below was right-hand only --
                        # five gripper links plus arm_r_link6/7 -- so the mobile base, the
                        # torso, the head, the whole LEFT arm and arm_r_link1-5 were all
                        # invisible to it. It reported "nearest=l2@0.6669" (l2 is a right-gripper
                        # FINGER) while the bottle went from 6.9 to 97.7 degrees in nine steps,
                        # and I read that as "nothing is touching it". It only ever meant the
                        # right hand was not. Scanning the full articulation is the same cost
                        # once per printed line and cannot lie by omission.
                        _tt_near, _tt_who = 9.9, "?"
                        _tt_rank = []
                        try:
                            _all_bp = robot.data.body_pos_w[0] - env.scene.env_origins[0]
                            _all_bn = robot.body_names
                            _d = torch.linalg.norm(_all_bp - _tt_o, dim=-1)
                            _ord = torch.argsort(_d)[:3].tolist()
                            _tt_rank = [(_all_bn[_i], float(_d[_i])) for _i in _ord]
                            _tt_who, _tt_near = _tt_rank[0][0], _tt_rank[0][1]
                        except Exception:                               # noqa: BLE001
                            pass
                        print(f"[tilttrace] t={int(timestep.max().item())} "
                              f"step={int(env_goal_indices[0])} tilt={_tt:.1f}deg "
                              f"obj=({float(_tt_o[0]):+.3f},{float(_tt_o[1]):+.3f},"
                              f"{float(_tt_o[2]):+.3f}) "
                              f"eef=({float(_tt_e[0]):+.3f},{float(_tt_e[1]):+.3f},"
                              f"{float(_tt_e[2]):+.3f}) "
                              f"eef_to_obj={float(torch.linalg.norm(_tt_e - _tt_o)):.4f} "
                              f"nearest={_tt_who}@{_tt_near:.4f} "
                              + "near3=[" + ",".join(f"{_n}@{_v:.3f}"
                                                     for _n, _v in _tt_rank) + "] "
                              # WHICH MOTION IS RUNNING. arm_r_retry is 0 while cuRobo's plan is
                              # executing and counts up once the jog takes over, so this says
                              # whether the bottle goes over under the PLAN -- which has no
                              # collision awareness, every obstacle sitting at 1e10 m -- or under
                              # the jog, which does approach along the tool axis. The plan
                              # standoff was removed earlier on a 1-success-versus-0 comparison,
                              # which is noise; this decides it on mechanism instead.
                              f"phase={'JOG' if int(arm_r_retry[0]) > 0 else 'PLAN'}"
                              f"(retry={int(arm_r_retry[0])})",
                              flush=True)
                        _tilt_trace_prev[0] = _tt
                except Exception:                                       # noqa: BLE001
                    pass

            # =================== Goal State Progression ===================
            if torch.any(finished_mask):
                finished_indices_in_active = torch.where(finished_mask[active_env_indices])[0]
                
                # Filter out finished environments from the lists of plans and indices
                for i in finished_indices_in_active.tolist():
                    global_idx = active_env_indices[i].item()
                    env_cmd_plans_r[global_idx] = None
                    env_cmd_plans_l[global_idx] = None
                
                env_cmd_indices_r[finished_mask] = 0
                env_cmd_indices_l[finished_mask] = 0
                env_r_gripper_dist_prev[finished_mask] = 10.0
                env_l_gripper_dist_prev[finished_mask] = 10.0
                
                # Increment goal index for finished environments
                env_goal_indices[finished_mask] += 1

            # Update and display demo count
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                success_env = env.reset_buf
                success_idx = torch.where(success_env)[0] 
                # The 3-stage buffer's third stage is otherwise dead: on_full_success is defined on
                # goal_mgr but called nowhere in this file, so full_good_goals never grows, F stays
                # 0/N in the [Goal3Stage] line, and stage=final_only is unreachable. task1/task2
                # data_generator both call it here; this file never did.
                # NOTE: success_idx above is env.reset_buf, which TerminationManager.compute() sets
                # to the UNION of every registered termination term (success OR retry/OOB — see
                # termination_manager.py's `self._truncated_buf | self._terminated_buf`), so it can
                # include envs that just failed out-of-bounds, not only envs that succeeded. Feeding
                # that union to on_full_success would let a same-step failure get permanently banked
                # into full_good_goals via _save_full(). Read the "success" term's own mask instead,
                # the same pattern the "retry" mask read below uses
                # (env.termination_manager.get_term("retry").clone()).
                success_term_mask = env.termination_manager.get_term("success").clone()
                full_success_idx = torch.where(success_term_mask)[0]
                goal_mgr.on_full_success(full_success_idx)
                _reset_envs(env, goal_mgr, num_envs, resample_goals_for_envs, args_cli.target_idx, env_goal_indices, env_cmd_indices_r, env_cmd_indices_l, grasp_checked_r, grasp_checked_l, nav_phase, env_cmd_pause_l, env_cmd_pause_trig_l, env_cmd_pause_r, env_cmd_pause_trig_r, drawer, pot_first, first_reset, env_sub_saved_r, env_sub_saved_l, new_gripper_commands_L, new_gripper_commands_R, timestep, env_cmd_plans_r, env_cmd_plans_l, ik_goals_r, ik_goals_l, success_idx, reason="success") 
                # THIS is where a recorded demonstration actually lands, and it is NOT the HDF5.
                #
                # env.recorder_manager's own file (dataset_export_dir_path/dataset_filename) stays
                # near-empty: it holds a `data` group with total=0, and its size jumps 96 -> 8192
                # the moment that group is created, whether or not an episode was ever written.
                # So `find -size +2k` reports demonstrations that do not exist, and counting the
                # groups under `data` reports zero when they DO. Both mislead; I lost time to each.
                #
                # writer is the TwoPhaseEpisodeWriter, and it dumps <stage>/<NNNNNN>/ holding
                # arrays.npz plus observation.images.{front,wrist_left,wrist_right}.npy. That
                # directory is exactly what offline_finalize_lerobot.py --stage reads. To count
                # demonstrations, count those numbered directories.
                _exported_episodes = obv[-1]
                _fallback_env_ids = recorded_episode_env_ids(
                    len(_exported_episodes), success_idx.tolist(), full_success_idx.tolist())
                if any(_env_id is None for _env_id in _fallback_env_ids):
                    print("[record] episode-to-env association unavailable; initial object "
                          "poses will not be attached", flush=True)
                for _slot, demo_data in enumerate(_exported_episodes):
                    _episode_succeeded = bool(demo_data.success)
                    write_success_if_missing(
                        env.recorder_manager,
                        demo_data,
                        succeeded=_episode_succeeded,
                        expected_success_count=current_recorded_demo_count,
                    )
                    if _episode_succeeded:
                        _raw_env_id = demo_data.env_id
                        _env_id = (int(_raw_env_id) if _raw_env_id is not None
                                   else _fallback_env_ids[_slot])
                        demo_data.env_id = _env_id
                        if demo_data.seed is None:
                            demo_data.seed = args_cli.seed
                        writer.write_episode(
                            demo_data,
                            initial_objects=initial_objects_by_env.get(_env_id),
                        )

            # Final check for exiting the loop
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break
            if rate_limiter:
                rate_limiter.sleep(env)
        # close the simulator
        if _ct_names:
            _chairtrace_summary()
        env.close()
        if not args_cli.skip_finalize:
            writer.finalize_lerobot(
                output_path=str(
                    simvla_paths.lerobot_root() / args_cli.task_language / args_cli.task
                ),
                task_json=str(file_path),
                # The per-step sentence used to live only in the .reloadable.json twin. A v2 goal
                # carries it itself and no twin is written, so point this at whichever file exists.
                # (finalize_lerobot json.loads it and throws it away; infer_task_language reads
                # goals[0][0]["language"], which a v2 file has.)
                goal_json=str(reloadable_file_path if reloadable_file_path.exists() else file_path),
                fps=20,

                # new behavior on by default:
                export_all=True,          # same as before
                export_subtasks=args_cli.save_each_subtask,   # export each sub_idx
                export_groups=args_cli.export_groups,
                min_segment_len=10,        # optional: ignore tiny segments

                repo_prefix=os.environ.get('HF_USER', 'local'),
                push=False,
                task_language=args_cli.task_language,
            )

if __name__ == "__main__":

    # PRINT THE TRACEBACK OURSELVES, BEFORE KIT GETS A CHANCE TO SWALLOW IT.
    #
    # An uncaught exception in here does not produce a traceback: the log ends with
    #
    #     Error in sys.excepthook:
    #
    #     Original exception was:
    #
    # and nothing else, because Kit has already torn down enough of the interpreter that the
    # default excepthook cannot run. A run that died after 27 minutes and eleven successful
    # post-grasp lifts left exactly that and no indication of what failed. Kit's teardown also
    # resets the process exit status, so the job's own guards cannot see it either.
    #
    # Printing to stdout with an explicit flush happens while the interpreter is still whole.
    exit_code = 0
    try:
        main()
    except BaseException as _exc:
        # Imported locally and by their real names: this module aliases sys as _simvla_sys, and
        # a NameError raised INSIDE the crash handler would lose the very traceback it exists to
        # print.
        import traceback as _tb
        print("=== simvla_gen died; traceback follows ===", flush=True)
        # FORMAT TO A STRING, THEN print(). traceback.print_exc(file=sys.stdout) wrote NOTHING
        # under Kit -- the marker above appeared and the traceback did not -- so the stream it
        # writes through is not usable by then even though print() still works.
        #
        # Each piece is guarded separately and the identity line comes last, so that a failure
        # while formatting the frames still leaves the exception type and message behind. Losing
        # the traceback is survivable; losing the exception itself is what cost the last two runs.
        try:
            print(_tb.format_exc(), flush=True)
        except BaseException:
            pass
        try:
            print(f"=== exception: {type(_exc).__name__}: {_exc!r}", flush=True)
        except BaseException:
            print("=== exception: (repr failed)", flush=True)
        exit_code = 1
    finally:
        # Exceptions can bypass env.close(); unsubscribe the STOP callback before
        # closing the stage, otherwise this fork waits forever rendering it.
        from isaaclab.sim import SimulationContext
        exit_code = close_with_status(
            exit_code, SimulationContext.clear_instance, simulation_app.close)

    raise SystemExit(exit_code)
