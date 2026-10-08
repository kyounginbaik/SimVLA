"""
SimVLA: Demo Replay & SimVQA generation
"""

import argparse
import importlib.util

# Check before Kit installs its exception handlers or initializes a GPU. A
# missing replay dependency must produce an actionable terminal error.
if importlib.util.find_spec("datasets") is None:
    raise SystemExit("Replay requires datasets; install env/simulator-raw.in in the simulator environment.")

from isaaclab.app import AppLauncher
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

import simvla_paths
from collector_profile import apply_ik_joint_deadzone, configure_physics_substeps
from workflow_status import record_status
from hdf5_compat import write_success_if_missing
from replay_targets import hold_uncontrolled_lift, base_tracking_correction, joint_target_order, replay_settle_steps

SAMPLE_PLANS: dict[str, dict[int, int]] = {
    "mug2sink": {0: 3, 1: 2, 3: 1, 4: 3, 5: 2, 7: 1},
    "bowl2sink": {0: 3, 1: 2, 3: 1, 4: 1, 5: 1, 7: 2, 8: 2, 10: 1, 11: 2},
    "pourwater": {0: 3, 1: 2, 3: 1, 4: 3, 5: 2, 7: 1, 10: 2},
}
import os

parser = argparse.ArgumentParser(description="SimVLA: Demo Replay")
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v01-01", help="Name of the task.")
parser.add_argument("--robot", type=str, default="anubis", help="Which robot to use in the task.")
parser.add_argument("--task_language", type=str, default=None,
                    help="Task prompt. Default: the goal file's run_config.task_language (v2), else the "
                         "LeRobot dataset's meta/tasks.parquet; a typed value overrides both.")
parser.add_argument("--dataset_file", type=str, default=os.environ.get("SIMVLA_LEROBOT_ROOT"), help="Local LeRobot file path")
parser.add_argument("--step_hz", type=int, default=20, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed episode selection, simulator initialization, and SimVQA sampling (default: 0).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Envs.")
parser.add_argument("--max_passes", type=int, default=6,
                    help="Maximum complete replay attempts before stopping (default: 6).")
parser.add_argument(
    "--bbox_obj_json",
    type=str,
    default=None,
    help="OVERRIDE: JSON mapping goal_idx -> target object name (e.g., {\"0\":\"mug0\"}). By default "
         "the target is the leaf of each step's prim_path in the goal file (v2, or the v1 .reloadable twin).",
)
parser.add_argument(
    "--subtask_json",
    type=str,
    default=None,
    help="OVERRIDE: JSON mapping goal_idx -> subtask string. By default each step's own `language` "
         "in the goal file (v2, or the v1 .reloadable twin).",
)
parser.add_argument("--simvqa", action="store_true", default=False, help="Save SimVQA data during demo replay")
parser.add_argument("--vqa_output_dir", default=None, help="New output directory for this VQA capture")
parser.add_argument(
    "--vqa_task",
    type=str,
    default=None,
    choices=sorted(SAMPLE_PLANS.keys()),
    help="A named per-subtask frame-sampling plan (only with --simvqa). Default: every subtask the "
         "episode visits gets --frames_per_subtask frames, whatever the task.",
)
parser.add_argument("--frames_per_subtask", type=int, default=2,
                    help="Frames sampled per subtask when no --vqa_task plan is named (the subtask's "
                         "last frame is always one of them).")
parser.add_argument("--local", action="store_true", default=False, help="LeRobot data in local")
parser.add_argument("--episode_index", type=int, default=None,
                    help="Exact recorded episode ID to replay; default: seeded random selection.")
parser.add_argument(
    "--obj_fix_init",
    action="store_true",
    default=False,
    help="Spawn the object at its scene default instead of randomizing it. Use on BOTH collection "
         "and replay for older datasets lacking meta/initial_objects.json; new local datasets "
         "restore their recorded object reset poses automatically.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_passes < 1:
    parser.error("--max_passes must be positive")
if args_cli.episode_index is not None and args_cli.episode_index < 0:
    parser.error("--episode_index must be nonnegative")
if args_cli.simvqa:
    args_cli.vqa_output_dir = args_cli.vqa_output_dir or f"./simvqa/{args_cli.robot}/{args_cli.task}"
    if _SimvlaPath(args_cli.vqa_output_dir, "vqa.jsonl").exists():
        parser.error("VQA records already exist; choose a new --vqa_output_dir to avoid duplicate IDs")
app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pathlib import Path
import gymnasium as gym
import numpy as np, math
import json
import time
import random
from datetime import datetime
import contextlib
import imageio
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from datasets import load_dataset
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.datasets import TwoPhaseEpisodeWriter
from isaaclab.utils.math import euler_xyz_from_quat
import torch.nn.functional as F
from isaaclab.utils.math import quat_from_matrix, quat_mul, quat_inv
from isaaclab.simvla import variable
from isaaclab.simvla.rot import (
    compose_base_world_and_eef_local,
    quat_delta_axis_angle,
    rot6d_to_R,
    sixd_to_quat_wxyz,
    translate_in_eef_frame,
)
from isaaclab.simvla.noise import (
    add_camera_noise_to_scene_cfg_once,
    add_light_noise_to_scene_cfg_once,
)
from isaaclab.simvla.simvqa import (
    _env_key,
    bbox_from_rgba_seg,
    bbox_to_paligemma_loc,
    clear_simvqa_buffers_for_envs,
    clear_simvqa_records_for_envs,
    find_instance_keys_by_name,
    flush_simvqa_on_success,
    load_goalidx_to_text,
    load_target_obj_bbox,
    describe_steps,
    gripper_closed,
    object_type,
    rel_pos_in_base,
    sample_frame_objects,
    sample_subtask_frames,
    sample_subtask_ticks,
    task_language_of,
    target_prim_path,
    uniform_sample_plan,
    vqa_task_maps,
)
from scene_spec import OBJECT_TYPES
from replay_initial_state import (
    apply_initial_objects, initial_objects_for_episode,
    apply_initial_robot_root, initial_robot_root_for_episode,
)

#: Objects each SimVQA frame asks about: the target plus this many sampled others (present in the
#: scene, or an absent vocabulary type so "is the apple visible? -> no" exists in the data).
VQA_OBJECTS_PER_FRAME = 3

class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.05, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

def _to_hwc_uint8(t):
    x = t.detach().cpu().numpy()
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

def pose_error_xyyaw(goal_xyyaw, cur_xyyaw):
    goal = torch.as_tensor(goal_xyyaw, device=cur_xyyaw.device, dtype=cur_xyyaw.dtype)
    dx_w = goal[0] - cur_xyyaw[0]
    dy_w = goal[1] - cur_xyyaw[1]

    yaw_c = cur_xyyaw[2]
    cy = torch.cos(yaw_c)
    sy = torch.sin(yaw_c)

    dx_local =  cy * dx_w + sy * dy_w
    dy_local = -sy * dx_w + cy * dy_w

    dyaw = goal[2] - yaw_c
    dyaw = torch.remainder(dyaw + math.pi, 2 * math.pi) - math.pi

    return torch.stack([dx_local, dy_local, dyaw]), torch.stack([dx_w, dy_w, dyaw])

def world2base(env, ee_pos_w, ee_quat_w, env_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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

def _load_calibration(robot: str) -> dict:
    """Load scripts/simvla/calibration/<robot>.json (frame offsets, init pose).

    Resolved relative to this file so it works regardless of CWD; mirrors
    the loader in real2sim.py so both scripts share one calibration source.
    """
    path = _SimvlaPath(__file__).resolve().parent / "calibration" / f"{robot}.json"
    if not path.exists():
        raise FileNotFoundError(f"No calibration JSON for robot={robot!r} at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Resolve embodiment configuration before constructing the expensive scene.
    calib = _load_calibration(args_cli.robot)
    for field in ("real_to_sim_frame_offset_xyz", "eef_to_grasp_offset_local_xyz"):
        values = calib.get(field)
        if not isinstance(values, list) or len(values) != 3 or not all(
                type(v) in (int, float) and np.isfinite(v) for v in values):
            raise ValueError(f"calibration/{args_cli.robot}.json needs three finite values for {field}")
    # Replay samples a source episode, camera/light randomization, and SimVQA frames/objects.
    # Pin each RNG so a saved dataset and command reproduce those random inputs.
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    print(f"[seed] {args_cli.seed}", flush=True)

    rate_limiter = RateLimiter(args_cli.step_hz)

    root = f"./demo_replay/{args_cli.robot}"
    output_dir = os.path.join(root, args_cli.task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    simvqa_save = False

    if args_cli.simvqa:
        output_dir_vqa = args_cli.vqa_output_dir
        if not os.path.exists(output_dir_vqa):
            os.makedirs(output_dir_vqa)
        simvqa_images_dir = os.path.join(output_dir_vqa, "images")
        simvqa_seg_images_dir = os.path.join(output_dir_vqa, "seg_images")
        os.makedirs(simvqa_images_dir, exist_ok=True)
        os.makedirs(simvqa_seg_images_dir, exist_ok=True)
        simvqa_jsonl_path = os.path.join(output_dir_vqa, "vqa.jsonl")
        simvqa_save = True

    output_file_name = f"{args_cli.task}.hdf5"
    output_path = os.path.join(output_dir, output_file_name)

    device = "cuda:0"
    num_envs = args_cli.num_envs
    base_tracking = os.environ.get("SIMVLA_REPLAY_BASE_TRACKING", "0")
    if base_tracking not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_BASE_TRACKING must be 0 or 1")
    base_tracking = base_tracking == "1"
    print(f"[replay] base mode: {'recorded-pose velocity feedback' if base_tracking else 'action-only velocity'}", flush=True)
    joint_replay = os.environ.get("SIMVLA_REPLAY_JOINT_TARGETS", "0")
    if joint_replay not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_JOINT_TARGETS must be 0 or 1")
    joint_replay = joint_replay == "1"
    settle_steps = replay_settle_steps(os.environ.get("SIMVLA_REPLAY_SETTLE_STEPS", "0"), joint_replay=joint_replay)
    if settle_steps:
        print(f"[replay] diagnostic terminal hold: at most {settle_steps} extra control steps; NOT original timing", flush=True)
    adaptive_gripper = os.environ.get("SIMVLA_REPLAY_ADAPTIVE_GRIPPER", "0")
    if adaptive_gripper not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_ADAPTIVE_GRIPPER must be 0 or 1")
    adaptive_gripper = adaptive_gripper == "1"
    if adaptive_gripper and not joint_replay:
        raise ValueError("adaptive-gripper joint replay requires SIMVLA_REPLAY_JOINT_TARGETS=1")
    substep_replay = os.environ.get("SIMVLA_REPLAY_JOINT_SUBSTEPS", "0")
    if substep_replay not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_JOINT_SUBSTEPS must be 0 or 1")
    substep_replay = substep_replay == "1"
    if substep_replay and (not joint_replay or args_cli.episode_index is None):
        raise ValueError("Motor substep replay requires joint targets and explicit episode_index")
    if joint_replay and not args_cli.local:
        raise ValueError("recorded joint replay currently requires a local dataset with joint-name metadata")
    print(f"[replay] arm/gripper mode: {'recorded joint motor targets' if joint_replay else 'Cartesian IK and binary gripper'}", flush=True)
    if adaptive_gripper:
        print("[replay] gripper override: live binary/contact control, NOT recorded finger motor targets", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=num_envs)
    from sink_collision import configure_sink_collisions
    configure_sink_collisions(env_cfg)
    configure_physics_substeps(env_cfg)
    env_cfg.seed = args_cli.seed
    simvla_paths.validate_robot_asset(args_cli.robot, env_cfg.scene.robot.spawn.usd_path)
    env_cfg.env_name = args_cli.task
    env_cfg.observations.policy.concatenate_terms = False
    if args_cli.simvqa:
        # Most task configs request RGB only. VQA also needs instance IDs and
        # their label map; enable the annotator before the sensors are created.
        for camera_name in ("front", "wrist_left", "wrist_right"):
            camera = getattr(env_cfg.scene, camera_name)
            camera.data_types = list(dict.fromkeys([*camera.data_types, "instance_id_segmentation_fast"]))
            camera.colorize_instance_id_segmentation = True
    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    writer = TwoPhaseEpisodeWriter(staging_dir=output_dir, robot=args_cli.robot)

    if not args_cli.local:
        repo_id = f"{os.environ['HF_USER']}/{args_cli.task}" 
        ds = load_dataset(repo_id, split="train")
    else:
        repo_id = args_cli.dataset_file 
        data_files = os.path.join(repo_id, "data", "**", "*.parquet")
        ds = load_dataset("parquet", data_files=data_files, split="train")

    print(f"Loading dataset: {repo_id}")
    is_start = np.array(ds["is_first"])
    from replay_initial_state import episode_start_indices
    eligible_starts = episode_start_indices(is_start, ds["episode_index"], args_cli.episode_index)
    start_idx = np.random.choice(eligible_starts)
    print(f"[replay] selected episode={int(ds['episode_index'][int(start_idx)])} row={start_idx}", flush=True)
    start_idx_init = start_idx.copy()
    is_last = np.array(ds["is_last"])
    last_indices = np.where(is_last == 1)[0]  # all last indices
    end_idx = last_indices[last_indices >= start_idx][0]
    recorded_initial_objects = (
        initial_objects_for_episode(args_cli.dataset_file, int(ds["episode_index"][start_idx]))
        if args_cli.local and not args_cli.obj_fix_init else None
    )
    recorded_robot_root = (
        initial_robot_root_for_episode(args_cli.dataset_file, int(ds["episode_index"][start_idx]))
        if args_cli.local else None
    )
    restore_joint_state = os.environ.get("SIMVLA_REPLAY_INITIAL_JOINT_STATE", "0")
    if restore_joint_state not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_INITIAL_JOINT_STATE must be 0 or 1")
    recorded_joint_state = None
    if restore_joint_state == "1":
        if not args_cli.local:
            raise ValueError("initial joint-state replay requires a local dataset")
        from replay_initial_state import initial_robot_joints_for_episode, apply_initial_robot_joints
        recorded_joint_state = initial_robot_joints_for_episode(
            args_cli.dataset_file, int(ds["episode_index"][start_idx]))
    replay_initial_step = os.environ.get("SIMVLA_REPLAY_INITIAL_STEP", "0")
    if replay_initial_step not in ("0", "1"):
        raise ValueError("SIMVLA_REPLAY_INITIAL_STEP must be 0 or 1")
    if replay_initial_step == "1" and recorded_joint_state is None:
        raise ValueError("Initial-step replay requires recorded initial joint-state restoration")

    video_dir = Path(f"./demo_replay/{args_cli.robot}/{args_cli.task}/{start_idx}_{end_idx}")
    video_dir.mkdir(parents=True, exist_ok=True)

    timestep = torch.zeros(num_envs, dtype=torch.int32, device=device)

    @torch.no_grad()
    def simvqa(env, sampled_ts_tensor, timestep, env_goal_indices, targets, subtasks, steps_desc, ds, start_idx_init, action_data):
        ANNOT = "instance_id_segmentation_fast"
        hit_mask = torch.isin(timestep, sampled_ts_tensor)
        if not bool(hit_mask.any().item()):
            return
        hit_env_ids = torch.nonzero(hit_mask, as_tuple=False).squeeze(-1).tolist()

        for eid in hit_env_ids:
            k = _env_key(eid)
            gidx = int(env_goal_indices[eid].item())

            obj = targets.get(gidx)
            sub = subtasks.get(gidx)
            step = steps_desc[gidx] if 0 <= gidx < len(steps_desc) else {"action": None, "goal": None}

            front_rgb = env.scene.sensors["front"].data.output["rgb"][eid].clone()
            wl_rgb    = env.scene.sensors["wrist_left"].data.output["rgb"][eid].clone()
            wr_rgb    = env.scene.sensors["wrist_right"].data.output["rgb"][eid].clone()

            segs = {name: env.scene.sensors[name].data.output[ANNOT][eid].clone()   # (H,W,4)
                    for name in ("front", "wrist_left", "wrist_right")}
            front_seg, wl_seg, wr_seg = segs["front"], segs["wrist_left"], segs["wrist_right"]
            robot_data = env.scene.articulations['robot'].data
            base_pos = robot_data.body_pos_w[eid, base_body_idx]
            base_quat = robot_data.body_quat_w[eid, base_body_idx]
            env_origins = env.scene.env_origins[eid]

            def bboxes_of(name):
                out = {}
                for sensor_name, seg in segs.items():
                    info = env.scene.sensors[sensor_name].data.info
                    H, W = seg.shape[:2]
                    path = target_prim_path(steps_desc, gidx, name)
                    keys = find_instance_keys_by_name(info, path, annot_key=ANNOT, eid=eid)
                    bbox = bbox_from_rgba_seg(seg, keys)   # union over every mesh of the prim
                    out[sensor_name] = None if bbox is None else bbox_to_paligemma_loc(*bbox, H, W)
                return out

            # The target's fields stay at the top level of the record (legacy readers); every
            # object the frame asks about, target included, is also listed under "objects".
            present = [n for n in env.scene.rigid_objects.keys() if object_type(n) in OBJECT_TYPES]
            objects = []
            for o in sample_frame_objects(obj, present, OBJECT_TYPES, k=VQA_OBJECTS_PER_FRAME, rng=random):
                if o["present"]:
                    # A target may be a prim that is not a rigid object (a door handle): it still
                    # has a segmentation label, so it gets bboxes, but no body pose -> no distance.
                    # "How far from the robot base" is answered IN THE BASE FRAME, like the
                    # gripper goal below -- a world-axes offset means nothing to the policy.
                    dist = None
                    if o["name"] in env.scene.rigid_objects:
                        o_pos = env.scene.rigid_objects[o["name"]].data.body_pos_w.squeeze(1)[eid]
                        dist = rel_pos_in_base(o_pos, base_pos, base_quat)
                    objects.append({**o, "bboxes": bboxes_of(o["name"]), "distance_base": dist})
                else:
                    objects.append({**o, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
                                    "distance_base": None})
            has_target = obj is not None and objects and objects[0]["name"] == obj
            bbox_tokens = objects[0]["bboxes"] if has_target else {"front": None, "wrist_left": None, "wrist_right": None}
            distance_base = objects[0]["distance_base"] if has_target else None
            
            goal_state_mobile = None
            goal_state_gripper = None
            gripper = None

            if step["action"] == "N_s" and step["goal"] is not None:
                # (x, y, yaw) of the base in the env frame -- the yaw comes from the quaternion;
                # handing the position's z in the yaw slot made every dyaw equal the goal yaw.
                _, _, base_yaw = euler_xyz_from_quat(base_quat.unsqueeze(0))
                cur_xyyaw = torch.stack([(base_pos - env_origins)[0], (base_pos - env_origins)[1], base_yaw[0]])
                goal_state_mobile = pose_error_xyyaw(step["goal"], cur_xyyaw)[0]

            elif step["action"] == "A_r":
                target = int(env_goal_indices[eid].item())
                subtask = ds["subtask_index"]
                gripper = "right"
                idx = [i for i, v in enumerate(subtask) if v == target]

                ends = [a for a, b in zip(idx, idx[1:]) if a + 1 != b] + [idx[-1]]
                candidates = [x for x in ends if x > start_idx_init]
                gripper_goal_idx = min(candidates) if candidates else None
                if gripper_goal_idx is not None:
                    goal_pos = action_data[gripper_goal_idx][10:13]
                    goal_quat = sixd_to_quat_wxyz(action_data[gripper_goal_idx][13:19])

                    ee_link_r_idx = env.scene.articulations["robot"].find_bodies(vqa_cfg["ee_links"]["right"])[0][0]
                    ee_link_r_pos = env.scene.articulations["robot"].data.body_pos_w[eid, ee_link_r_idx]
                    ee_link_r_quat = env.scene.articulations["robot"].data.body_quat_w[eid, ee_link_r_idx]
                    ee_link_r_pos_b, ee_link_r_quat_b = world2base(env, ee_link_r_pos, ee_link_r_quat, eid)
                    pos_diff = goal_pos - ee_link_r_pos_b
                    q_diff = quat_mul(goal_quat, quat_inv(ee_link_r_quat_b))
                    goal_state_gripper = torch.cat([pos_diff, q_diff], dim=-1)

            elif step["action"] == "A_l":
                target = int(env_goal_indices[eid].item())
                subtask = ds["subtask_index"]
                gripper = "left"
                idx = [i for i, v in enumerate(subtask) if v == target]

                ends = [a for a, b in zip(idx, idx[1:]) if a + 1 != b] + [idx[-1]]
                candidates = [x for x in ends if x > start_idx_init]
                gripper_goal_idx = min(candidates) if candidates else None
                if gripper_goal_idx is not None:
                    goal_pos = action_data[gripper_goal_idx][0:3]
                    goal_quat = sixd_to_quat_wxyz(action_data[gripper_goal_idx][3:9])

                    ee_link_l_idx = env.scene.articulations["robot"].find_bodies(vqa_cfg["ee_links"]["left"])[0][0]
                    ee_link_l_pos = env.scene.articulations["robot"].data.body_pos_w[eid, ee_link_l_idx]
                    ee_link_l_quat = env.scene.articulations["robot"].data.body_quat_w[eid, ee_link_l_idx]
                    ee_link_l_pos_b, ee_link_l_quat_b = world2base(env, ee_link_l_pos, ee_link_l_quat, eid)
                    pos_diff = goal_pos - ee_link_l_pos_b
                    q_diff = quat_mul(goal_quat, quat_inv(ee_link_l_quat_b))
                    goal_state_gripper = torch.cat([pos_diff, q_diff], dim=-1)

            r_joint = env.scene.articulations['robot'].data.joint_pos[eid, r_gripper_idx]
            l_joint = env.scene.articulations['robot'].data.joint_pos[eid, l_gripper_idx]

            # "closed" = the finger has travelled from its reset (open) reading; see gripper_closed.
            right_gripper = gripper_closed(r_joint, open_joint_r[eid], vqa_cfg["gripper_closed_travel"])
            left_gripper = gripper_closed(l_joint, open_joint_l[eid], vqa_cfg["gripper_closed_travel"])
            simvqa_records[k].append({
                "t": int(timestep[eid].item()),
                "goal_idx": gidx,
                "grasp_target": step.get("grasp_target", False),
                "obj_name": obj,
                "subtask": sub,
                "subtask_end": int(timestep[eid].item()) in subtask_end_ts,
                "gripper_joints": {"left": float(l_joint), "right": float(r_joint),
                                   "left_open": float(open_joint_l[eid]), "right_open": float(open_joint_r[eid])},
                "images": {
                    "front": front_rgb,
                    "wrist_left": wl_rgb,
                    "wrist_right": wr_rgb,
                },
                "seg_images": {
                    "front": front_seg,
                    "wrist_left": wl_seg,
                    "wrist_right": wr_seg,
                },
                "bboxes": {
                    "front": bbox_tokens["front"],
                    "wrist_left": bbox_tokens["wrist_left"],
                    "wrist_right": bbox_tokens["wrist_right"],
                },
                "distance_base": distance_base,
                "objects": objects,
                "goal_state_mobile": goal_state_mobile,
                "goal_state_gripper": goal_state_gripper,
                "left_gripper": left_gripper,
                "right_gripper": right_gripper,
                "gripper": gripper,
            })

    front_frames = []
    wrist_left_frames = []
    wrist_right_frames = []

    file_path = str(simvla_paths.goal_files(args_cli.task)[0])
    with open(file_path, 'r') as file:
        start_init_pose = ds["initial_pose"][start_idx]
        if recorded_robot_root is not None:
            apply_initial_robot_root(env_cfg, recorded_robot_root)
            print(f"[replay] restored recorded robot articulation root: {recorded_robot_root['root_pose']}", flush=True)
        else:
            if args_cli.robot == "aiworker":
                raise ValueError("AI Worker replay requires a recorded initial robot root in "
                                 "meta/scene_states.json; base_link is not the articulation root. "
                                 "Recollect with the current recorder.")
            env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (start_init_pose[0], start_init_pose[0] + 0.000001)
            env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (start_init_pose[1], start_init_pose[1] + 0.000001)
        data = json.load(file)
    from sink_collision import validate_sink_goal_environment
    validate_sink_goal_environment(data)
    # Every per-step fact SimVQA needs is in the goal file: the sentence, the prim the step acts
    # on, the authored nav goal. v2 carries them on the step; a v1 file's .reloadable twin does.
    twin_path = simvla_paths.goal_files(args_cli.task)[1]
    twin = json.load(open(twin_path)) if os.path.exists(twin_path) else None
    steps_desc = describe_steps(data, twin)
    task_language = args_cli.task_language or task_language_of(data, None)   # typed flag wins
    if task_language is None and args_cli.local:
        tasks_pq = os.path.join(args_cli.dataset_file, "meta", "tasks.parquet")
        tasks_jl = os.path.join(args_cli.dataset_file, "meta", "tasks.jsonl")
        if os.path.exists(tasks_pq):
            import pandas as pd
            task_language = str(pd.read_parquet(tasks_pq).index[0])
        elif os.path.exists(tasks_jl):
            with open(tasks_jl) as f:
                task_language = str(json.loads(f.readline())["task"])
    if simvqa_save and not task_language:
        raise SystemExit("--task_language: the goal file has no run_config.task_language and the "
                         "dataset has no meta/tasks.parquet; pass it explicitly.")

    # New datasets carry object reset poses in meta/initial_objects.json. Older datasets do not;
    # their open-loop replay is only reproducible if BOTH collection and replay used --obj_fix_init.
    if recorded_initial_objects is not None and not args_cli.obj_fix_init:
        from replay_initial_state import initial_object_velocities_for_episode
        recorded_object_velocities = initial_object_velocities_for_episode(
            args_cli.dataset_file, int(ds["episode_index"][start_idx]))
        restored = apply_initial_objects(env_cfg, recorded_initial_objects, recorded_object_velocities)
        print(f"[replay] restored recorded initial poses for: {', '.join(restored)}", flush=True)
        if recorded_object_velocities is not None:
            print("[replay] restored recorded initial object linear/angular velocities", flush=True)
    elif not args_cli.obj_fix_init:
        print("[replay] warning: the dataset does not store the object's initial pose; "
              "a randomized replay may not reproduce the collected grasp. "
              "For older datasets, pass --obj_fix_init to both collect and replay.", flush=True)
        env_cfg.events.obj_init_pos.params["pose_range"]["x"] = (-0.02,0.02)
        env_cfg.events.obj_init_pos.params["pose_range"]["y"] = (-0.02,0.02)

    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.05, rot_max_deg=10.0, yaw_only=True, seed=args_cli.seed, cams = "front")
    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.01, rot_max_deg=5.0, yaw_only=False, seed=args_cli.seed, cams = "wrist_right")
    env_cfg = add_camera_noise_to_scene_cfg_once(env_cfg, pos_sigma=0.01, rot_max_deg=5.0, yaw_only=False, seed=args_cli.seed, cams = "wrist_left")

    env_cfg = add_light_noise_to_scene_cfg_once(
       env_cfg,
       intensity_base=3000.0,
       intensity_stop_range=1.5,
       kelvin_choices=(2700, 3200, 4000, 5000, 6500, 8000, 9000),
       rgb_jitter_sigma=0.08,
       seed=args_cli.seed,
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[ik] per-joint DLS deadzone set to {apply_ik_joint_deadzone(env):g} rad", flush=True)
    env.reset()
    replay_robot = env.scene.articulations["robot"]
    if recorded_joint_state is not None:
        apply_initial_robot_joints(replay_robot, recorded_joint_state)
        print("[replay] restored measured initial robot joint positions and velocities", flush=True)
    replay_lift_ids = replay_robot.find_joints(["lift_joint"])[0] if args_cli.robot == "aiworker" else None
    joint_targets = None
    motor_substeps = None
    replay_joint_ids = None
    if joint_replay:
        with open(Path(args_cli.dataset_file) / "meta/info.json") as stream:
            joint_feature = json.load(stream)["features"]["action.joint"]
        order = joint_target_order(joint_feature, replay_robot.data.joint_names)
        joint_targets = torch.tensor(ds["action.joint"], device=device, dtype=torch.float32)[:, order]
        if not bool(torch.isfinite(joint_targets).all()):
            raise ValueError("dataset contains non-finite recorded joint targets")
        if substep_replay:
            from replay_substeps import load_joint_substeps
            substep_start = int(start_idx)
            frames = sum(int(index) == args_cli.episode_index for index in ds["episode_index"])
            motor_substeps = torch.tensor(load_joint_substeps(
                args_cli.dataset_file, args_cli.episode_index, replay_robot.data.joint_names,
                frames=frames, decimation=env_cfg.decimation, control_fps=args_cli.step_hz),
                device=device, dtype=torch.float32)
            print(f"[replay] physics-rate motor targets: {frames} frames x {env_cfg.decimation} substeps", flush=True)
        if adaptive_gripper:
            from replay_targets import non_gripper_joint_ids
            replay_joint_ids = non_gripper_joint_ids(
                len(replay_robot.data.joint_names),
                [env.action_manager.get_term(name)._joint_ids
                 for name in ("gripperL_action", "gripperR_action")])
            # ActionManager accepts full-width targets and applies the selected
            # columns itself; preserve the dataset/articulation shape here.

    initial_step = None
    if replay_initial_step == "1":
        from replay_initial_step import load_initial_step
        initial_step = load_initial_step(
            args_cli.dataset_file, int(ds["episode_index"][start_idx]),
            action_dim=env.action_manager.total_action_dim,
            joint_names=replay_robot.joint_names, decimation=env_cfg.decimation)
        if initial_step["subtask_index"] >= len(steps_desc):
            raise ValueError("Initial-step subtask is outside the goal script")
        if substep_replay and "motor_substeps" not in initial_step:
            raise ValueError("Physics-rate replay requires initial-step motor substeps")

    @torch.inference_mode()
    def run_initial_step():
        if initial_step is None:
            return
        hold_uncontrolled_lift(replay_robot, replay_lift_ids)
        if joint_targets is not None:
            key = "motor_substeps" if substep_replay else "motor_targets"
            initial_motors = torch.tensor(initial_step[key], device=device, dtype=torch.float32)
            initial_motors = initial_motors.unsqueeze(0).expand(num_envs, *initial_motors.shape)
            env.action_manager.set_joint_position_replay_targets(
                initial_motors, joint_ids=replay_joint_ids)
        initial_action = torch.tensor(initial_step["raw_action"], device=device,
                                      dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
        initial_goal = torch.full((num_envs,), initial_step["subtask_index"],
                                  device=device, dtype=torch.long)
        # Success/retry predicates read this shared index, not env.step's argument.
        # Initialize it on the simulator device before the prelude's first physics tick.
        variable.env_goal_indices = initial_goal.clone()
        env.step(initial_action, initial_goal)
        if bool(torch.any(env.reset_buf).item()):
            raise RuntimeError("Environment terminated during the recorded initial-step prelude")
        print("[replay] executed recorded initial control tick before exported frame zero; "
              "this tick is not included in dataset/video/SimVQA frame counts", flush=True)

    run_initial_step()
    current_recorded_demo_count = 0
    gripper_command_L = False
    gripper_command_R = False
    start = 0 

    ox, oy, oz = calib["real_to_sim_frame_offset_xyz"]

    action_data = torch.tensor(ds["action"], device=device, dtype=torch.float32)
    action_data[:, 0]  += ox; action_data[:, 1]  += oy; action_data[:, 2]  += oz
    action_data[:, 10] += ox; action_data[:, 11] += oy; action_data[:, 12] += oz

    # Link / joint names the SimVQA capture reads are per robot and live in the calibration file.
    vqa_cfg = calib.get("vqa")
    if simvqa_save and vqa_cfg is None:
        raise SystemExit(f"calibration/{args_cli.robot}.json has no 'vqa' block (base_body, ee_links, "
                         f"gripper_joints, gripper_closed_travel); SimVQA cannot name this robot's links.")
    if vqa_cfg is None:
        vqa_cfg = {"base_body": "base_link", "ee_links": {"left": "ee_link2", "right": "ee_link1"},
                   "gripper_joints": {"left": "gripper2R_joint", "right": "gripper1R_joint"},
                   "gripper_closed_travel": 0.002}
    r_gripper_idx = env.scene.articulations['robot'].find_joints(vqa_cfg["gripper_joints"]["right"])[0][0]
    l_gripper_idx = env.scene.articulations['robot'].find_joints(vqa_cfg["gripper_joints"]["left"])[0][0]
    base_body_idx = env.scene.articulations['robot'].find_bodies(vqa_cfg["base_body"])[0][0]
    # The grippers are open at reset: this reading is the per-env "open" the closed test measures from.
    open_joint_r = env.scene.articulations['robot'].data.joint_pos[:, r_gripper_idx].clone()
    open_joint_l = env.scene.articulations['robot'].data.joint_pos[:, l_gripper_idx].clone()
    print(f"[SimVQA] gripper open readings at reset: right={open_joint_r[0].item():.4f} left={open_joint_l[0].item():.4f}")

    R_l = rot6d_to_R(action_data[:, 3:9], device=device, dtype=action_data.dtype)
    R_r = rot6d_to_R(action_data[:, 13:19], device=device, dtype=action_data.dtype)

    delta_local = torch.tensor(
        calib["eef_to_grasp_offset_local_xyz"], device=device, dtype=action_data.dtype
    ).view(1, 3, 1)

    delta_world_l = (R_l @ delta_local).squeeze(-1)  # (T, 3) torch cuda
    delta_world_r = (R_r @ delta_local).squeeze(-1)  # (T, 3) torch cuda

    action_data[:, 0:3]  += delta_world_l
    action_data[:, 10:13] += delta_world_r

    subtask_data = torch.tensor(ds["subtask_index"], device=device)
    
    if simvqa_save:

        episode_subtasks = ds[start_idx_init:end_idx]["subtask_index"]
        sample_plan = (SAMPLE_PLANS[args_cli.vqa_task] if args_cli.vqa_task
                       else uniform_sample_plan(episode_subtasks, args_cli.frames_per_subtask))

        local_sampled_idx_by_subtask = {}

        for subtask, n_sample in sample_plan.items():
            candidates = [i for i, v in enumerate(episode_subtasks) if v == subtask]
            local_sampled_idx_by_subtask[subtask] = sample_subtask_ticks(candidates, n_sample, rng=random)

        # Loop ticks start at one; dataset frame indices start at zero.
        # sample_subtask_ticks includes the last frame and performs that conversion.
        subtask_end_ts = {max(lst) for lst in local_sampled_idx_by_subtask.values() if lst}
        local_sampled_idx = [i for lst in local_sampled_idx_by_subtask.values() for i in lst]
        local_sampled_idx = sorted(local_sampled_idx)
        sampled_ts_tensor = torch.tensor(local_sampled_idx, device=device, dtype=timestep.dtype)
        env_keys = [_env_key(i) for i in range(num_envs)]
        simvqa_records = {k: [] for k in env_keys}

        front_img_dict       = {k: [] for k in env_keys}
        wrist_left_img_dict  = {k: [] for k in env_keys}
        wrist_right_img_dict = {k: [] for k in env_keys}
        front_bbox_dict       = {k: [] for k in env_keys}
        wrist_left_bbox_dict  = {k: [] for k in env_keys}
        wrist_right_bbox_dict = {k: [] for k in env_keys}
        subtasks, targets = vqa_task_maps(steps_desc, vocab=OBJECT_TYPES)
        if args_cli.subtask_json:
            subtasks = load_goalidx_to_text(args_cli.subtask_json)
        if args_cli.bbox_obj_json:
            targets = load_target_obj_bbox(args_cli.bbox_obj_json)
        print(f"[SimVQA] task language: {task_language!r}")
        for i in sorted(subtasks):
            print(f"[SimVQA] subtask {i}: {subtasks[i]!r} -> target {targets.get(i)!r}")
        # The expander reads this beside vqa.jsonl, so one command converts any mix of tasks.
        with open(os.path.join(output_dir_vqa, "vqa_meta.json"), "w") as f:
            json.dump({"task": args_cli.task, "task_language": task_language,
                       "subtasks": {str(k): v for k, v in subtasks.items()},
                       "grasp_steps": {str(i): s.get("grasp_target", False) for i, s in enumerate(steps_desc)},
                       "sampled_goal_indices": sorted(int(i) for i, frames in local_sampled_idx_by_subtask.items() if frames),
                       "targets": {str(k): v for k, v in targets.items()}}, f, indent=2)

    subtask = np.array(ds["subtask_index"])
    unique_vals = np.unique(subtask[subtask >= 0])
    variable.max_sequence_length = len(unique_vals)
    num_retry = 0
    replay_succeeded = False
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            timestep+=1
            recorded_idx = min(start_idx, end_idx)
            settling = start_idx > end_idx
            if start_idx == end_idx + 1:
                print("[replay] recorded actions exhausted; beginning explicit terminal motor hold", flush=True)
            env_goal_indices = subtask_data[recorded_idx].clone().view(1,1).expand(num_envs,1)
            variable.env_goal_indices = env_goal_indices.squeeze(1).clone()

            action = action_data[recorded_idx].clone()
            action = action.view(1,23).expand(num_envs, 23)
            if simvqa_save and not settling:
                simvqa(env, sampled_ts_tensor, timestep, env_goal_indices, targets, subtasks, steps_desc, ds, start_idx_init, action_data)
            l_rot6d = action[..., 3:9]
            r_rot6d = action[..., 13:19]
            l_quat = sixd_to_quat_wxyz(l_rot6d)
            r_quat = sixd_to_quat_wxyz(r_rot6d)

            if start == 0:
                d_l_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
                d_r_xyz = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
                d_l_rot = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
                d_r_rot = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
                start += 1
            else:
                d_l_xyz = (action[..., 0:3] - prev_l_xyz).to(device)
                d_r_xyz = (action[..., 10:13] - prev_r_xyz).to(device)
                d_l_rot = (quat_delta_axis_angle( prev_l_quat, l_quat)).to(device)
                d_r_rot = (quat_delta_axis_angle( prev_r_quat, r_quat)).to(device)

            prev_l_xyz = action[..., 0:3]
            prev_r_xyz = action[..., 10:13]
            prev_l_quat = l_quat.clone()
            prev_r_quat = r_quat.clone()

            l_gripper = action[..., 9:10].to(device)
            r_gripper = action[..., 19:20].to(device)
            l_gripper = torch.where(l_gripper < -0.8, 1.0, -1.0)
            r_gripper = torch.where(r_gripper < -0.8, 1.0, -1.0)

            delta_pose_base = action[..., 20:23].to(device)
            base_link_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
            r, p, yaw = euler_xyz_from_quat(env.scene.articulations["robot"].data.body_quat_w[:, base_link_idx])
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            delta_pose_base = action[..., 20:23].to(device)  # (3,)
            vx_local = delta_pose_base[..., 0]
            vy_local = delta_pose_base[..., 1]
            omega    = delta_pose_base[..., 2]
            vx_world = cos_yaw * vx_local - sin_yaw * vy_local
            vy_world = sin_yaw * vx_local + cos_yaw * vy_local

            delta_pose_base_world = torch.stack([vx_world, vy_world, omega], dim=1)
            if settling:
                delta_pose_base_world.zero_()
            if base_tracking:
                reference = ds[int(recorded_idx)]["initial_pose"]
                actual = replay_robot.data.body_state_w[:, base_link_idx, :7]
                corrections = [base_tracking_correction(
                    reference, pose[[0, 1, 3, 4, 5, 6]].tolist()) for pose in actual]
                delta_pose_base_world += torch.tensor(corrections, device=device)
            d_l_xyz = d_l_xyz.view(num_envs, 3)
            d_l_rot = d_l_rot.view(num_envs, 3)
            d_r_xyz = d_r_xyz.view(num_envs, 3)
            d_r_rot = d_r_rot.view(num_envs, 3)
            l_gripper = l_gripper.view(num_envs, 1)
            r_gripper = r_gripper.view(num_envs, 1)
            actions = torch.cat(
                (d_l_xyz, d_l_rot, d_r_xyz, d_r_rot, l_gripper, r_gripper, delta_pose_base_world),
                dim=1,
            )
            hold_uncontrolled_lift(replay_robot, replay_lift_ids)
            if joint_targets is not None:
                motor_targets = (motor_substeps[int(recorded_idx) - substep_start].unsqueeze(0).expand(num_envs, -1, -1)
                           if motor_substeps is not None else
                           joint_targets[recorded_idx].view(1, -1).expand(num_envs, -1))
                if settling and motor_targets.ndim == 3:
                    motor_targets = motor_targets[:, -1, :]
                env.action_manager.set_joint_position_replay_targets(
                    motor_targets,
                    joint_ids=replay_joint_ids)
            obs_tuple = env.step(actions, env_goal_indices.squeeze(1).clone())
            obs_dict = obs_tuple[0]

            if os.environ.get("SIMVLA_TRACE_LIFT", "0") == "1" and int(timestep[0]) % 20 == 0:
                for _arm, _body in (("l", "ee_link2"), ("r", "ee_link1")):
                    _term = env.action_manager.get_term("gripperL_action" if _arm == "l" else "gripperR_action")
                    _eef = replay_robot.find_bodies(_body)[0][0]
                    print("[replay-contact] " + json.dumps({
                        "frame": int(timestep[0]), "arm": _arm,
                        "step": int(env_goal_indices[0]),
                        "base_pose_w": replay_robot.data.body_state_w[0, base_link_idx, :7].tolist(),
                        "eef_pose_w": replay_robot.data.body_state_w[0, _eef, :7].tolist(),
                        "command": _term.raw_actions[0].tolist(),
                        "joint_pos": replay_robot.data.joint_pos[0, _term._joint_ids].tolist(),
                        "finger_load_n": [float(_term._finger_load(i)[0]) for i in (0, 1)]
                            if hasattr(_term, "_finger_load") else None,
                    }), flush=True)

            if os.environ.get("SIMVLA_DEBUG_TRACK"):
                # The replay feeds per-step DELTAS to a relative-mode IK controller, so any
                # drift between the controller's desired pose and the recorded absolute pose
                # accumulates invisibly. Compare where the EEF actually is against where the
                # demo said it should be (action_data holds absolute ee_pos_des after the
                # calibration offsets are undone above).
                # The recorded targets are BASE-frame EEF poses (frame 0 of a demo reads as
                # ~home_r = (0.2257,-0.0988,1.0351)), so the live pose has to be pulled into
                # the base frame before differencing -- comparing against world coordinates
                # reports ~2 m of error while the arm is sitting exactly at home.
                _rob = env.scene.articulations["robot"]
                _r_idx = _rob.find_bodies("ee_link1")[0][0]
                _l_idx = _rob.find_bodies("ee_link2")[0][0]
                _b_idx = _rob.find_bodies("base_link")[0][0]
                _b_pos = _rob.data.body_pos_w[:, _b_idx]
                _b_quat = _rob.data.body_quat_w[:, _b_idx]
                _r_base = math_utils.quat_rotate_inverse(_b_quat, _rob.data.body_pos_w[:, _r_idx] - _b_pos)[0]
                _l_base = math_utils.quat_rotate_inverse(_b_quat, _rob.data.body_pos_w[:, _l_idx] - _b_pos)[0]
                _r_tgt = action_data[recorded_idx, 10:13]
                _l_tgt = action_data[recorded_idx, 0:3]
                print(
                    f"[track] t={int(timestep[0])} goal={int(env_goal_indices[0])} "
                    f"r_now=({float(_r_base[0]):.3f},{float(_r_base[1]):.3f},{float(_r_base[2]):.3f}) "
                    f"r_tgt=({float(_r_tgt[0]):.3f},{float(_r_tgt[1]):.3f},{float(_r_tgt[2]):.3f}) "
                    f"r_err={float(torch.linalg.norm(_r_base - _r_tgt)):.4f} "
                    f"l_err={float(torch.linalg.norm(_l_base - _l_tgt)):.4f}",
                    flush=True,
                )

            front_rgb       = env.scene.sensors["front"].data.output["rgb"]
            wrist_left_rgb  = env.scene.sensors["wrist_left"].data.output["rgb"]
            wrist_right_rgb = env.scene.sensors["wrist_right"].data.output["rgb"]

            front_frames.append(_to_hwc_uint8(front_rgb))
            wrist_left_frames.append(_to_hwc_uint8(wrist_left_rgb))
            wrist_right_frames.append(_to_hwc_uint8(wrist_right_rgb))

            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                if len(obs_tuple) > 5 and isinstance(obs_tuple[5], (list, tuple)):
                    for episode in obs_tuple[5]:
                        write_success_if_missing(
                            env.recorder_manager, episode, succeeded=bool(episode.success),
                            expected_success_count=env.recorder_manager.exported_successful_episode_count)
                replay_succeeded = True
                success_env = env.reset_buf & env.termination_manager.get_term("success")
                success_idx = torch.where(success_env)[0]
                if success_idx.numel() == 0:
                    raise RuntimeError("Recorder reported success without a successful terminal environment")
                first_idx = int(success_idx[0])
                front_success = [x[first_idx] for x in front_frames]   
                wrist_right_success = [x[first_idx] for x in wrist_right_frames]   
                wrist_left_success = [x[first_idx] for x in wrist_left_frames]   
                fps = args_cli.step_hz
                for _cam, _frames in (("front", front_success),
                                      ("wrist_right", wrist_right_success),
                                      ("wrist_left", wrist_left_success)):
                    imageio.mimwrite(video_dir / f"success{num_retry + 1}_{_cam}.mp4",
                                     _frames, fps=fps, codec="libx264")
                print("Demo replay done")
                try:
                    _tm = env.termination_manager
                    # env.step() auto-resets successful environments before returning.
                    # Object transforms here are the *new* episode's spawn poses, not
                    # terminal replay evidence; do not print them as if they proved
                    # the physical outcome of the completed episode.
                    print(f"[replay] success detected: pass={num_retry + 1} timestep={int(timestep[0].item())} "
                          f"start_idx={int(start_idx)} reset_buf={env.reset_buf.tolist()} "
                          f"success_term={_tm.get_term('success').tolist()} "
                          f"exported_success={env.recorder_manager.exported_successful_episode_count} "
                          f"exported_failed={env.recorder_manager.exported_failed_episode_count} "
                          f"goal_idx={variable.env_goal_indices.tolist()} "
                          "objects=unavailable-after-auto-reset", flush=True)
                    print(f"[replay] terminal hold steps used: {max(0, int(start_idx - end_idx))}", flush=True)
                except Exception as _e:
                    print(f"[replay] success diagnostics failed: {_e!r}", flush=True)
                if simvqa_save:
                    flush_simvqa_on_success(
                        success_idx=first_idx,
                        simvqa_records=simvqa_records,
                        simvqa_jsonl_path=simvqa_jsonl_path,
                        simvqa_images_dir=simvqa_images_dir,
                        simvqa_seg_images_dir=simvqa_seg_images_dir,
                        prompt=task_language,
                        task=args_cli.task,
                    )
                break
            if rate_limiter:
                rate_limiter.sleep(env)
            start_idx +=1
            failed_reset = bool(torch.any(env.reset_buf).item())
            if failed_reset or start_idx > end_idx + settle_steps:
                num_retry += 1 
                reason = "environment terminated before success" if failed_reset else "reached the end of the demo without success"
                print(f"[replay] pass {num_retry}: {reason}; timestep={int(timestep[0])}", flush=True)
                # step() auto-resets terminated environments. Never execute the remainder
                # of a recorded episode against the replacement scene or export mixed VQA.
                # Diagnostics for a failed pass: where the objects and the acting gripper ended up,
                # and the pass's own video (success writes one; a failure used to vanish silently).
                try:
                    _rob = env.scene.articulations["robot"]
                    _pos = {n: [round(v, 3) for v in env.scene.rigid_objects[n].data.body_pos_w.squeeze(1)[0].tolist()]
                            for n in env.scene.rigid_objects.keys() if n[-1].isdigit()}
                    _state_label = "post-reset, NOT terminal" if failed_reset else "final live state"
                    print(f"[replay]   objects (env 0, world; {_state_label}): {_pos}", flush=True)
                    print(f"[replay]   gripper joints R={_rob.data.joint_pos[0, r_gripper_idx].item():.4f} "
                          f"L={_rob.data.joint_pos[0, l_gripper_idx].item():.4f}", flush=True)
                    for _cam, _frames in (("front", front_frames), ("wrist_right", wrist_right_frames),
                                          ("wrist_left", wrist_left_frames)):
                        imageio.mimwrite(video_dir / f"fail{num_retry}_{_cam}.mp4", [x[0] for x in _frames],
                                         fps=args_cli.step_hz, codec="libx264")
                    print(f"[replay]   wrote {video_dir}/fail{num_retry}_*.mp4", flush=True)
                except Exception as _e:   # diagnostics must never take the run down
                    print(f"[replay]   diagnostics failed: {_e!r}", flush=True)
                if num_retry >= args_cli.max_passes:
                    print(f"[replay] giving up after {args_cli.max_passes} passes: no episode succeeded, nothing exported", flush=True)
                    break

                env.reset()
                if recorded_joint_state is not None:
                    apply_initial_robot_joints(replay_robot, recorded_joint_state)
                env.action_manager.get_term("armL_action")._ik_controller.reset()
                env.action_manager.get_term("armR_action")._ik_controller.reset()
                run_initial_step()

                front_frames = []
                wrist_right_frames = []
                wrist_left_frames = []
                gripper_command_L = False
                gripper_command_R = False
                start = 0
                timestep = torch.zeros(num_envs, dtype=torch.int32, device=device)
                start_idx = start_idx_init.copy()
                
                env_ids = torch.arange(num_envs, device=device) 
                if simvqa_save: 
                    clear_simvqa_buffers_for_envs(
                        env_ids,
                        front_img_dict, wrist_left_img_dict, wrist_right_img_dict,
                        front_bbox_dict, wrist_left_bbox_dict, wrist_right_bbox_dict,
                    )
                    clear_simvqa_records_for_envs(env_ids, simvqa_records)
        env.close()
    if not replay_succeeded:
        raise RuntimeError(f"Replay did not reproduce task success within {num_retry} completed passes")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        import traceback
        print(traceback.format_exc(), flush=True)
        exit_code = 1
    finally:
        record_status(exit_code)
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
        simulation_app.close()
    raise SystemExit(exit_code)
