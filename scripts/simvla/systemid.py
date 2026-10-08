"""SimVLA integrated system identification.

(Single or Bimanual)
(Static or Mobile)
(Joint control or EEF control)
selected by --robot + --control. 

Per-robot structure lives in calibration/<robot>.json
"""
import argparse
import json
import contextlib
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SimVLA integrated SysID")
parser.add_argument("--disable_fabric", action="store_true", default=False,help="Disable fabric and use USD I/O operations.")
parser.add_argument("--task", type=str, default="Real2Sim-v1", help="Base task id.")
parser.add_argument("--robot", type=str, default="anubis", help="Robot embodiment.")
parser.add_argument("--control", type=str, required=True, choices=["joint", "eef"], help="Absolute joint-position control or absolute EEF-pose control.")
parser.add_argument("--fps", type=int, default=20, help="Env stepping rate (Hz).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs.")
parser.add_argument("--termination_t", type=int, default=20, help="SysID termination window.")
parser.add_argument("--repo_id", type=str, help="LeRobot data repo_id.")
parser.add_argument("--data_file", type=Path, help="Local LeRobot Parquet recording (no Hub access).")
parser.add_argument("--npz", type=str, default=None,
                    help="RB-Y1 recorder log (.npz) to replay instead of --repo_id (joint mode).")
parser.add_argument("--qpos_thres", type=float, default=None,
                    help="Strict acceptance: reset an env whose abs-sum joint error EVER exceeds this "
                         "(rad, summed over the fitted joints). Each reset re-draws that env's gains, so "
                         "the run becomes rejection sampling and the SURVIVORS are the accepted set. "
                         "Default: leave the env cfg's value (2.5, which never fires).")
parser.add_argument("--fixed_gains", action="store_true", default=False,
                    help="Disable gain randomisation and replay with whatever the robot cfg ships "
                         "(rby1.py's own values). Without this or --gains, a replay randomises.")
parser.add_argument("--reference", type=str, default=None,
                    help="Right-arm RB-Y1 LeRobot parquet that records COMMAND and ACHIEVED joints "
                         "(same metric as the teleop logs). Use with --fit_joints right_arm.")
parser.add_argument("--fit_joints", type=str, default=None,
                    help="Comma-separated joint names, or a prefix like 'right_arm', to restrict "
                         "which joints the error is scored over. Held joints otherwise dilute it.")
parser.add_argument("--rainbow", type=str, default=None,
                    help="Rainbow Robotics RB-Y1 LeRobot parquet (real hardware) to replay. Its action "
                         "is EEF, so this commands the RECORDED JOINT trajectory one step ahead -- a "
                         "different, easier metric than the MolmoBot replay; see the loader docstring.")
parser.add_argument("--episode_id", type=int, default=None, help="Which episode_index in --rainbow.")
parser.add_argument("--molmobot", type=str, default=None,
                    help="MolmoBot trajectories_*.h5 (allenai/molmobot-data) to replay instead of --npz.")
parser.add_argument("--traj", type=str, default="traj_0", help="Which trajectory in --molmobot.")
parser.add_argument("--stiffness_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Log-uniform stiffness sampling range (overrides the env cfg).")
parser.add_argument("--damping_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Log-uniform RAW damping range. Giving this switches the sampler back to "
                         "independent K,D (IsaacLab's randomize_actuator_gains) instead of sampling "
                         "the damping ratio.")
parser.add_argument("--zeta_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Damping-ratio sampling range, when not using --damping_range.")
parser.add_argument("--settle_steps", type=int, default=None,
                    help="Steps after each reset during which --qpos_thres is NOT enforced "
                         "(the reset frame's error is one frame of real motion, gain-independent). "
                         "Default: the env cfg's value (20).")
parser.add_argument("--max_done", type=int, default=0,
                    help="Stop once this many envs have completed a full replay (0 = run until killed).")
parser.add_argument("--max_steps", type=int, default=0,
                    help="Bound total simulation steps across retries (0 = unlimited).")
parser.add_argument("--json", type=str, default="real2sim.json", help="Output SysID json.")
parser.add_argument("--record", action="store_true", default=False,help="Dump env 0's camera to --video (needs --enable_cameras).")
parser.add_argument("--cam_pitch_deg", type=float, default=None,
                    help="Override the head camera's downward pitch (default 50). Their front "
                         "camera is near-horizontal, so a low value makes a side-by-side comparable.")
parser.add_argument("--camera", type=str, default="camera", choices=["camera", "scene_cam"],
                    help="Which sensor --record films: 'camera' is the head camera, 'scene_cam' a "
                         "fixed third-person view (better for judging whether a motion looks right).")
parser.add_argument("--video", type=str, default=None, help="Output mp4 for --record (default videos/<robot>.mp4).")
parser.add_argument("--gains", type=str, default=None,
                    help="JSON {stiffness: {joint: K}, damping: {joint: D}}: fix the actuator gains to these and "
                         "disable the gain randomisation (replay with identified gains).")
parser.add_argument("--episode", type=int, default=0, help="Which demo/episode to replay.")
parser.add_argument("--random_episode", action="store_true", default=False,
                    help="Pick a uniform-random episode instead of --episode (logs the choice).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab.simvla as simvla
import isaaclab_tasks  # noqa: F401  (registers envs)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat
from datasets import load_dataset

import systemid_core as sc

CALIB_DIR = Path(__file__).resolve().parent / "calibration"

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


def _to_hwc_uint8(t):
    x = t.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.clip(0, 255).astype(np.uint8)
    return x


# Any event that randomises gains must be off whenever a run is meant to use FIXED gains,
# or it silently overwrites them on every reset. Matching one function name is what broke
# this once: the check looked for "randomize_actuator_gains" and the sampler was renamed to
# randomize_gains_by_damping_ratio, so --gains kept printing "fixed 28 gain values" while the
# event re-randomised them behind it. Match anything gain-shaped instead.
def _disable_gain_randomisation(cfg, why):
    n = 0
    for name in list(vars(cfg.events)):
        term = getattr(cfg.events, name)
        fn = getattr(getattr(term, "func", None), "__name__", "")
        if "gain" in fn.lower():
            setattr(cfg.events, name, None)
            print(f"[systemid] {why}: disabled gain randomisation event {name!r} ({fn})")
            n += 1
    if n == 0:
        print(f"[systemid] {why}: no gain randomisation event found to disable")
    return n


def main():
    mode = args_cli.control
    device = args_cli.device
    rate_limiter = RateLimiter(args_cli.fps)

    cfg = sc.load_robot_cfg(args_cli.robot, CALIB_DIR)
    if mode not in cfg["layouts"]:
        raise ValueError(f"robot {args_cli.robot!r} has no '{mode}' layout in its JSON")
    layout = cfg["layouts"][mode]

    env_id = f"{args_cli.task}-{args_cli.robot}-{mode}"
    if env_id not in gym.registry:
        raise ValueError(
            f"env id {env_id!r} is not registered. Register a "
            f"Real2Sim env cfg for (robot={args_cli.robot}, control={mode})."
        )

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=args_cli.num_envs)
    env_cfg.env_name = env_id
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    # Cameras exist only to feed --record. Left in, --enable_cameras tiles one render per env
    # and a 1024-env sysid ran a 24 GB RTX 4090 out of VULKAN memory (job 2130292) while the
    # physics needed a fraction of that. The scene builder skips None entries.
    for _cam in ("camera", "scene_cam"):
        if getattr(env_cfg.scene, _cam, None) is not None and (
                not args_cli.record or _cam != args_cli.camera):
            setattr(env_cfg.scene, _cam, None)
    if not args_cli.record:
        print("[systemid] no --record: camera sensors dropped from the scene")
    else:
        print(f"[systemid] recording from {args_cli.camera!r}")

    # Fixed gains: write them into the actuator groups that own each joint and switch the
    # randomisation event off, so every env replays with the identified dynamics.
    if args_cli.gains:
        with open(args_cli.gains, "r", encoding="utf-8") as f:
            gains = json.load(f)
        import re as _re
        n_set = 0
        for prop in ("stiffness", "damping"):
            for jname, val in gains.get(prop, {}).items():
                owners = [a for a in env_cfg.scene.robot.actuators.values()
                          if any(_re.fullmatch(expr, jname) for expr in a.joint_names_expr)]
                if len(owners) != 1:
                    raise ValueError(f"--gains: joint {jname!r} is owned by {len(owners)} actuator groups")
                cur = getattr(owners[0], prop)
                if not isinstance(cur, dict):
                    # IsaacLab's resolver raises on a joint matching two keys, so the group's
                    # regex cannot stay next to exact names: the dict becomes exact-name-only.
                    # A joint of this group NOT in the gains file then keeps its USD default.
                    cur = {}
                cur[jname] = float(val)
                setattr(owners[0], prop, cur)
                n_set += 1
        _disable_gain_randomisation(env_cfg, "--gains")
        print(f"[systemid] --gains: fixed {n_set} gain values from {args_cli.gains}")

    # Offline data: a LeRobot repo, or one raw RB-Y1 recorder log resampled to --fps.
    if args_cli.reference:
        offline = sc.load_rby1_reference_parquet(args_cli.reference, args_cli.episode_id, args_cli.fps)
        print(f"[systemid] loaded {args_cli.reference} ep{args_cli.episode_id}: "
              f"{offline['action'].shape[0]} frames")
    elif args_cli.rainbow:
        offline = sc.load_rainbow_parquet(args_cli.rainbow, args_cli.episode_id, args_cli.fps)
        print(f"[systemid] loaded {args_cli.rainbow} ep{args_cli.episode_id}: "
              f"{offline['action'].shape[0]} frames")
    elif args_cli.molmobot:
        offline = sc.load_molmobot_h5(args_cli.molmobot, args_cli.traj, args_cli.fps)
        print(f"[systemid] loaded {args_cli.molmobot}:{args_cli.traj}: "
              f"{offline['action'].shape[0]} frames")
    elif args_cli.npz:
        offline = sc.load_rby1_npz(args_cli.npz, args_cli.fps)
        print(f"[systemid] loaded {args_cli.npz}: {offline['action'].shape[0]} frames at {args_cli.fps} Hz")
    elif args_cli.data_file:
        if not args_cli.data_file.is_file():
            raise FileNotFoundError(args_cli.data_file)
        offline = load_dataset("parquet", data_files=str(args_cli.data_file), split="train")
    elif args_cli.repo_id:
        offline = load_dataset(args_cli.repo_id, split="train")
    else:
        raise ValueError("give --data_file, --repo_id, --npz, --molmobot or --rainbow")
    if args_cli.cam_pitch_deg is not None and getattr(env_cfg.scene, "camera", None) is not None:
        import math
        th = math.radians(args_cli.cam_pitch_deg) / 2
        env_cfg.scene.camera.offset.rot = (math.cos(th), 0.0, math.sin(th), 0.0)
        print(f"[systemid] head camera pitch set to {args_cli.cam_pitch_deg:.1f} deg down")

    # NOTE: an attempt to pitch the head camera by the demo's held head tilt was REVERTED.
    # Adding their tilt to our 50 deg mount assumes their camera mount equals ours; it does not,
    # and the result pointed 92 deg down -- straight at the robot's own body. Their extrinsics
    # (obs/sensor_param/head_camera/cam2world_gl) do not agree with the head-joint values under
    # the obvious sign convention either, so the true offset is unresolved. Our fixed 50 deg is
    # used for every replay: it frames the arms well, and for door-opening demos (head held at
    # 34-42 deg) the two panels of a comparison video simply look in somewhat different
    # directions. Video framing only -- the tracking numbers are joint-space.

    arr = np.array(offline["action"], dtype=np.float64)

    # Resolve which episode to replay, then slice every replayed array to it so
    # the action stream and the sim2ruin targets stay frame-aligned (and a replay
    # can never run past the episode boundary into the next demo).
    ep_index = np.asarray(offline["episode_index"]).reshape(-1)
    if args_cli.random_episode:
        episode = int(np.random.default_rng().integers(0, int(ep_index.max()) + 1))
    else:
        episode = int(args_cli.episode)
    ep_start, ep_end = sc.episode_frame_range(ep_index, episode)
    print(f"[systemid] replaying episode {episode}: frames [{ep_start}, {ep_end}) "
          f"({ep_end - ep_start} steps)")

    ep_len = ep_end - ep_start
    # timestep is compared to termination_t only AFTER being incremented, so a
    # non-positive value never resets and the replay walks off the end just as
    # an over-long one does.
    if not (1 <= args_cli.termination_t <= ep_len):
        raise ValueError(
            f"--termination_t {args_cli.termination_t} is outside episode {episode}'s usable "
            f"range (frames [{ep_start}, {ep_end}), length {ep_len}); the replay would index "
            f"past the sliced arrays. Use 1 <= --termination_t <= {ep_len}."
        )

    arr = arr[ep_start:ep_end]

    # SysID targets (dataset key -> signal contract).
    if mode == "eef":
        qpos_full = np.asarray(offline["qpos"], dtype=np.float64)
        eefpos_full = np.asarray(offline["observation.state"], dtype=np.float64)
        # torch tensors on device: sim2ruin indexes these with a CUDA timestep tensor,
        # which numpy arrays cannot accept.
        env_cfg.terminations.sim2ruin.params["qpos"] = torch.as_tensor(
            qpos_full[ep_start:ep_end], device=device)
        env_cfg.terminations.sim2ruin.params["eefpos"] = torch.as_tensor(
            eefpos_full[ep_start:ep_end], device=device)
        env_cfg.terminations.sim2ruin.params["pos_offset_xyz"] = tuple(
            cfg.get("real_to_sim_frame_offset_xyz", (0.0, 0.0, 0.0)))
    else:
        qpos_full = np.asarray(offline["observation.state"], dtype=np.float64)
        env_cfg.terminations.sim2ruin.params["qpos"] = torch.as_tensor(
            qpos_full[ep_start:ep_end], device=device)
    if args_cli.fixed_gains and not args_cli.gains:
        _disable_gain_randomisation(env_cfg, "--fixed_gains")

    # Gain sampling: ranges, and which coordinate to sample in.
    ev = getattr(env_cfg.events, "arm_gains", None)
    if ev is not None:
        if args_cli.damping_range is not None:
            from isaaclab.envs.mdp import randomize_actuator_gains
            k_range = tuple(args_cli.stiffness_range) if args_cli.stiffness_range else \
                tuple(ev.params.get("stiffness_range", (30.0, 20000.0)))
            ev.func = randomize_actuator_gains
            ev.params = {
                "asset_cfg": ev.params["asset_cfg"],
                "stiffness_distribution_params": k_range,
                "damping_distribution_params": tuple(args_cli.damping_range),
                "operation": "abs",
                "distribution": "log_uniform",
            }
            gain_prior = {"stiffness": list(k_range), "damping": list(args_cli.damping_range),
                          "sampling": "independent K,D"}
            print(f"[systemid] independent gain sampling: K {k_range}, D {tuple(args_cli.damping_range)}")
        else:
            if args_cli.stiffness_range is not None:
                ev.params["stiffness_range"] = tuple(args_cli.stiffness_range)
            if args_cli.zeta_range is not None:
                ev.params["zeta_range"] = tuple(args_cli.zeta_range)
            gain_prior = {"stiffness": list(ev.params["stiffness_range"]),
                          "zeta": list(ev.params["zeta_range"]), "sampling": "damping ratio"}
            print(f"[systemid] damping-ratio sampling: K {ev.params['stiffness_range']}, "
                  f"zeta {ev.params['zeta_range']}")
        # Recorded in the output json so the analysis knows what prior the survivors came from.
        env_cfg.terminations.sim2ruin.params["gain_prior"] = gain_prior

    if args_cli.settle_steps is not None:
        env_cfg.terminations.sim2ruin.params["settle_steps"] = args_cli.settle_steps
        print(f"[systemid] post-reset grace window: {args_cli.settle_steps} steps")
    if args_cli.qpos_thres is not None:
        env_cfg.terminations.sim2ruin.params["qpos_thres"] = args_cli.qpos_thres
        print(f"[systemid] strict acceptance: qpos_thres = {args_cli.qpos_thres} rad abs-sum")
    if args_cli.fit_joints:
        allj = list(env_cfg.terminations.sim2ruin.params["joint_names"])
        want = args_cli.fit_joints.split(",")
        sel = [j for j in allj if any(j == w or j.startswith(w) for w in want)]
        if not sel:
            raise ValueError(f"--fit_joints {args_cli.fit_joints!r} matched none of {allj}")
        keep = [allj.index(j) for j in sel]
        env_cfg.terminations.sim2ruin.params["joint_names"] = sel
        env_cfg.terminations.sim2ruin.params["qpos_cols"] = keep
        print(f"[systemid] scoring over {len(sel)} joint(s): {sel}")
    env_cfg.terminations.sim2ruin.params["termination_t"] = args_cli.termination_t
    env_cfg.terminations.sim2ruin.params["json_file"] = str(Path(args_cli.json).expanduser().resolve())

    # Initial joint pose. Prefer the episode's own first recorded frame — real
    # episodes each start from a different configuration, and init_state is
    # re-applied by Isaac on every env reset, so resets return to this pose too.
    qpos_init = cfg.get("qpos_init")
    if qpos_init is not None:
        expected_source = "qpos" if mode == "eef" else "observation.state"
        if qpos_init["source"] != expected_source:
            raise ValueError(
                f"qpos_init.source is {qpos_init['source']!r} but {mode} mode seeds the init "
                f"row from {expected_source!r} — they must match, or the sim would be "
                f"initialized from the wrong dataset feature."
            )
    if qpos_init is not None:
        init_joints = sc.init_joints_from_qpos(qpos_full[ep_start], qpos_init)
        # CLAMP TO THE URDF LIMITS. Isaac raises on an out-of-limit initial pose rather than
        # clamping, and MolmoBot's RB-Y1 model is slightly wider than this one -- a door-opening
        # episode starts right_arm_1 at 0.023 rad where this URDF caps 0.017, and the whole task
        # died on 0.006 rad. Clamping is right for a replay, but a LARGE clamp would mean the
        # recorded pose is genuinely unreachable here, so every clamp is reported and a big one
        # is loud.
        limits = cfg.get("joint_limits_rad", {})
        clamped = {}
        for jn, val in list(init_joints.items()):
            lo_hi = limits.get(jn)
            if lo_hi is None:
                continue
            # Clamp a hair INSIDE the limit: landing exactly on it still fails Isaac's check,
            # because the USD stores the limit in float32 and is fractionally tighter than the
            # float64 URDF value ("0.017 not in [-3.142, 0.017]"). 1e-4 rad = 0.006 deg.
            eps = 1e-4
            new_val = min(max(val, lo_hi[0] + eps), lo_hi[1] - eps)
            if new_val != val:
                clamped[jn] = (val, new_val, abs(new_val - val))
                init_joints[jn] = new_val
        if clamped:
            worst = max(clamped.values(), key=lambda t: t[2])[2]
            print(f"[systemid] clamped {len(clamped)} init joint(s) into the URDF limits, "
                  f"worst {worst:.4f} rad: "
                  + ", ".join(f"{k} {v[0]:.4f}->{v[1]:.4f}" for k, v in clamped.items()))
            if worst > 0.05:
                print(f"[systemid] WARNING: a {worst:.3f} rad clamp is large -- this recorded pose "
                      f"is not reachable by this robot model, so the replay starts somewhere else")
        for jname, val in init_joints.items():
            env_cfg.scene.robot.init_state.joint_pos[jname] = val
        print(f"[systemid] init joints from {qpos_init['source']}[{ep_start}] "
              f"({len(init_joints)} joints)")
    else:
        for jname, val in cfg.get("init_joint_pos", {}).items():
            if jname.startswith("_"):
                continue
            env_cfg.scene.robot.init_state.joint_pos[jname] = val
        print("[systemid] no qpos_init in robot JSON — using static init_joint_pos")

    # Select columns; calibrate the full trajectory once (eef only).
    inputs_full = sc.select_inputs(arr, layout["input"])
    if mode == "eef":
        sc.apply_eef_calibration(inputs_full, cfg)

    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()

    if qpos_init is not None:
        sim_joint_names = set(env.scene.articulations["robot"].data.joint_names)
        unknown = sorted(set(init_joints) - sim_joint_names)
        if unknown:
            raise ValueError(
                f"qpos_init maps joints absent from the articulation: {unknown}. "
                f"Available: {sorted(sim_joint_names)}"
            )

    simvla.timestep = torch.zeros(env.num_envs, device=device, dtype=torch.long)

    frames = []
    video_path = Path(args_cli.video) if args_cli.video else Path("videos") / f"{args_cli.robot}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush_video():
        if frames:
            import imageio.v2 as imageio
            imageio.mimwrite(video_path, frames, fps=args_cli.fps, codec="libx264")
            print(f"[systemid] wrote {len(frames)} frames -> {video_path}")
    prev = None
    total_steps = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            idx = simvla.timestep.long().cpu().numpy()
            reset_mask = (simvla.timestep == 0)

            if mode == "eef":
                curr = {f: torch.tensor(inputs_full[f][idx], device=device) for f in inputs_full}
                yaw = None
                if "base_delta" in layout["input"]:
                    base_idx = env.scene.articulations["robot"].find_bodies("base_link")[0][0]
                    _, _, yaw = euler_xyz_from_quat(
                        env.scene.articulations["robot"].data.body_quat_w[:, base_idx])
                fields = sc.build_eef_fields(curr, prev, cfg, yaw, reset_mask, device)
                action = sc.pack_output(fields, layout["output"])
                prev = curr
            else:  # joint passthrough
                joints = torch.tensor(inputs_full["joints"][idx], device=device)
                action = sc.pack_output({"joints": joints}, layout["output"])

            env.step(action, torch.tensor([0], device=device))
            total_steps += 1
            if total_steps % 100 == 0:
                print(f"[systemid] steps={total_steps} completed={getattr(simvla, 'sysid_done_count', 0)} "
                      f"furthest_frame={int(simvla.timestep.max())}", flush=True)
            if args_cli.record:
                # After the step: the frame shows the state that sim2ruin compares with qpos[t+1].
                frames.append(_to_hwc_uint8(env.scene.sensors[args_cli.camera].data.output["rgb"][0].clone()))
                if len(frames) % 200 == 0:
                    _flush_video()
            if env.sim.is_stopped():
                break
            if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
                if getattr(simvla, "sysid_done_count", 0) == 0:
                    raise RuntimeError(f"No gain candidates completed within {total_steps} steps")
                break
            if args_cli.max_done > 0 and getattr(simvla, "sysid_done_count", 0) >= args_cli.max_done:
                print(f"[systemid] {simvla.sysid_done_count} envs completed >= --max_done "
                      f"{args_cli.max_done}; stopping")
                break
            if rate_limiter:
                rate_limiter.sleep(env)

    if args_cli.record:
        _flush_video()
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
