"""Pure SysID transform + config logic for systemid.py.

Stdlib + numpy + torch + isaaclab.simvla.rot only — importing this module
must NOT boot Omniverse, so it stays unit-testable (see test_systemid.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# rot helpers: use the package import when Omniverse is booted (the runner),
# else load rot.py directly by file path — importing the isaaclab.simvla PACKAGE
# runs its __init__ -> .terminations -> isaacsim.core, which needs Omniverse and
# breaks test-time import. The rot.py module itself is Omniverse-free.
try:
    from isaaclab.simvla.rot import sixd_to_quat_wxyz, quat_delta_axis_angle, rot6d_to_R
except Exception:  # noqa: BLE001 - any import-time failure means "no booted Kit"
    import importlib.util as _ilu

    _ROT_PATH = Path(__file__).resolve().parents[2] / "source/isaaclab/isaaclab/simvla/rot.py"
    _spec = _ilu.spec_from_file_location("_simvla_rot", _ROT_PATH)
    _rot = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_rot)
    sixd_to_quat_wxyz = _rot.sixd_to_quat_wxyz
    quat_delta_axis_angle = _rot.quat_delta_axis_angle
    rot6d_to_R = _rot.rot6d_to_R

# Expected column width per semantic field.
FIELD_WIDTHS: dict[str, int] = {
    "l_eef_xyz": 3, "l_rot6d": 6, "l_gripper": 1,
    "r_eef_xyz": 3, "r_rot6d": 6, "r_gripper": 1,
    "base_delta": 3,
}

# Arm prefix -> output field names it produces (eef mode).
ARM_OUT: dict[str, dict[str, str]] = {
    "l": {"dxyz": "dLxyz", "drot": "dLrot", "grip": "Lgrip"},
    "r": {"dxyz": "dRxyz", "drot": "dRrot", "grip": "Rgrip"},
}


def load_robot_cfg(robot: str, calib_dir: Path) -> dict:
    """Read calibration/<robot>.json, strip top-level `_*` comment keys, validate."""
    path = Path(calib_dir) / f"{robot}.json"
    if not path.exists():
        raise FileNotFoundError(f"No robot JSON for robot={robot!r} at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    validate_cfg(cfg)
    return cfg


def validate_cfg(cfg: dict) -> None:
    """Raise ValueError if the config is internally inconsistent."""
    for key in ("arms", "mobile", "layouts"):
        if key not in cfg:
            raise ValueError(f"robot config missing required key {key!r}")
    for mode, layout in cfg["layouts"].items():
        inp = layout["input"]
        # Width check for every known field present.
        for field, idxs in inp.items():
            if field in FIELD_WIDTHS and len(idxs) != FIELD_WIDTHS[field]:
                raise ValueError(
                    f"[{mode}] field {field!r} has width {len(idxs)}, "
                    f"expected {FIELD_WIDTHS[field]}"
                )
        # arms metadata must match presence of r_* fields (eef) / declared arms.
        if mode == "eef":
            has_r = any(k.startswith("r_") for k in inp)
            if cfg["arms"] == 1 and has_r:
                raise ValueError("arms=1 but layout declares right-arm (r_*) fields")
            if cfg["arms"] == 2 and not has_r:
                raise ValueError("arms=2 but layout declares no right-arm (r_*) fields")
            # eef calibration needs the offset keys (even if zero); a missing key must
            # fail loudly rather than silently yield an uncalibrated (wrong) trajectory.
            for key in ("real_to_sim_frame_offset_xyz", "eef_to_grasp_offset_local_xyz"):
                if key not in cfg:
                    raise ValueError(f"eef layout requires top-level {key!r} in robot config")
            # each present arm must carry its full input set — build_eef_fields reads
            # {arm}_rot6d and {arm}_gripper whenever {arm}_eef_xyz is present.
            for arm in ("l", "r"):
                if f"{arm}_eef_xyz" in inp:
                    for req in (f"{arm}_rot6d", f"{arm}_gripper"):
                        if req not in inp:
                            raise ValueError(
                                f"[eef] {arm}_eef_xyz present but {req!r} missing from input"
                            )
        # mobile metadata must match base_delta presence.
        has_base = "base_delta" in inp
        if cfg["mobile"] and not has_base and mode == "eef":
            raise ValueError("mobile=true but eef layout has no base_delta input")
        # Every eef output field must be backed by an input.
        for out in layout.get("output", []):
            if not _output_backed(out, inp):
                raise ValueError(f"[{mode}] output field {out!r} has no backing input")

    qi = cfg.get("qpos_init")
    if qi is not None:
        if "source" not in qi:
            raise ValueError("qpos_init requires a 'source' key naming the dataset feature")
        for name, col in qi.get("joints", {}).items():
            if not isinstance(col, int) or isinstance(col, bool) or col < 0:
                raise ValueError(f"qpos_init joint {name!r}: column must be a non-negative int")
        for name, spec in qi.get("grippers", {}).items():
            for key in ("col", "offset", "scale", "range"):
                if key not in spec:
                    raise ValueError(f"qpos_init gripper {name!r}: missing {key!r}")
            if not isinstance(spec["col"], int) or isinstance(spec["col"], bool) or spec["col"] < 0:
                raise ValueError(f"qpos_init gripper {name!r}: column must be a non-negative int")
            if spec["scale"] == 0:
                raise ValueError(f"qpos_init gripper {name!r}: scale must be non-zero")


def _output_backed(out: str, inp: dict) -> bool:
    """True if output field `out` can be produced from the available inputs."""
    if out == "base":
        return "base_delta" in inp
    if out == "joints":
        return "joints" in inp
    for prefix, names in ARM_OUT.items():
        if out == names["dxyz"]:
            return f"{prefix}_eef_xyz" in inp
        if out == names["drot"]:
            return f"{prefix}_rot6d" in inp
        if out == names["grip"]:
            return f"{prefix}_gripper" in inp
    return True  # unknown fields (future) are not blocked here


def select_inputs(arr: np.ndarray, input_layout: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Slice a (T, D) action array into {field: (T, width)} by column index lists."""
    return {field: arr[:, list(idxs)] for field, idxs in input_layout.items()}


def apply_eef_calibration(inputs: dict[str, np.ndarray], cfg: dict) -> None:
    """In place: shift EEF xyz by the world frame offset, then add the grasp
    offset (local) rotated to world by R(rot6d), for each arm present."""
    offset = np.asarray(cfg.get("real_to_sim_frame_offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    grasp_local = torch.tensor(
        cfg.get("eef_to_grasp_offset_local_xyz", [0.0, 0.0, 0.0]), dtype=torch.float64
    ).view(1, 3, 1)
    for arm in ("l", "r"):
        xyz_key, rot_key = f"{arm}_eef_xyz", f"{arm}_rot6d"
        if xyz_key not in inputs:
            continue
        inputs[xyz_key] = inputs[xyz_key] + offset  # frame offset
        R = rot6d_to_R(inputs[rot_key], dtype=torch.float64)          # (T,3,3)
        delta_world = (R @ grasp_local).squeeze(-1).detach().cpu().numpy()     # (T,3)
        inputs[xyz_key] = inputs[xyz_key] + delta_world               # grasp offset


def binarize_gripper(g: torch.Tensor, threshold: float) -> torch.Tensor:
    """+1 (close) where g < threshold, else -1 (open)."""
    return torch.where(g < threshold, 1.0, -1.0)


def base_local_to_world(base_delta: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate planar base velocity (vx, vy) from base frame to world by yaw;
    angular omega passes through. base_delta: (N,3), yaw: (N,)."""
    vx, vy, omega = base_delta[:, 0], base_delta[:, 1], base_delta[:, 2]
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    vx_w = cos_y * vx - sin_y * vy
    vy_w = sin_y * vx + cos_y * vy
    return torch.stack([vx_w, vy_w, omega], dim=1)


def build_eef_fields(
    curr: dict, prev: dict | None, cfg: dict,
    yaw: torch.Tensor | None, reset_mask: torch.Tensor, device,
) -> dict[str, torch.Tensor]:
    """Produce the eef output fields for one step. Deltas are zeroed for rows
    in reset_mask and for all rows when prev is None (trajectory start)."""
    threshold = cfg["threshold_gripper"]
    input_fields = cfg["layouts"]["eef"]["input"]
    out: dict[str, torch.Tensor] = {}
    reset_mask = reset_mask.to(device)

    for arm, names in ARM_OUT.items():
        xyz_key = f"{arm}_eef_xyz"
        if xyz_key not in input_fields:
            continue
        rot_key, grip_key = f"{arm}_rot6d", f"{arm}_gripper"
        curr_xyz = curr[xyz_key].to(device)
        curr_quat = sixd_to_quat_wxyz(curr[rot_key].to(device))
        N = curr_xyz.shape[0]
        if prev is None:
            d_xyz = torch.zeros((N, 3), device=device, dtype=curr_xyz.dtype)
            d_rot = torch.zeros((N, 3), device=device, dtype=curr_quat.dtype)
        else:
            prev_xyz = prev[xyz_key].to(device)
            prev_quat = sixd_to_quat_wxyz(prev[rot_key].to(device))
            d_xyz = (curr_xyz - prev_xyz)
            d_rot = quat_delta_axis_angle(prev_quat, curr_quat).to(device)
        # zero the reset rows
        d_xyz[reset_mask] = 0.0
        d_rot[reset_mask] = 0.0
        out[names["dxyz"]] = d_xyz
        out[names["drot"]] = d_rot
        out[names["grip"]] = binarize_gripper(curr[grip_key].to(device), threshold)

    if "base_delta" in input_fields and yaw is not None:
        out["base"] = base_local_to_world(curr["base_delta"].to(device), yaw.to(device))
    return out


def pack_output(fields: dict[str, torch.Tensor], output_layout: list[str]) -> torch.Tensor:
    """Concatenate the named fields (dim=1) in the declared output order."""
    missing = [f for f in output_layout if f not in fields]
    if missing:
        raise KeyError(f"output layout names fields not produced: {missing}")
    return torch.cat([fields[f] for f in output_layout], dim=1)


def episode_frame_range(episode_index, episode_n: int) -> tuple[int, int]:
    """Half-open [start, end) frame range of one episode.

    `episode_index` is the dataset's per-frame episode id column. Frames of an
    episode are expected to be contiguous (LeRobot stores them that way); a gap
    means the caller's assumption about the dataset is wrong, so raise.
    """
    ep = np.asarray(episode_index).reshape(-1)
    hits = np.flatnonzero(ep == episode_n)
    if hits.size == 0:
        raise ValueError(f"episode {episode_n} not found in dataset")
    start, end = int(hits[0]), int(hits[-1]) + 1
    if hits.size != end - start:
        raise ValueError(f"episode {episode_n} frames are not contiguous")
    return start, end


def init_joints_from_qpos(qpos_row, qpos_init: dict) -> dict[str, float]:
    """Map one recorded qpos row to {sim_joint_name: value}.

    Arm joints copy their column verbatim. Gripper joints invert the transform
    sim2ruin applies in the other direction (qpos = offset - scale*joint/range).
    """
    row = np.asarray(qpos_row).reshape(-1)
    width = row.shape[0]

    def _col(idx: int, who: str) -> float:
        if not (0 <= idx < width):
            raise ValueError(f"qpos_init {who}: column {idx} out of range for qpos width {width}")
        return float(row[idx])

    out: dict[str, float] = {}
    for name, col in qpos_init.get("joints", {}).items():
        out[name] = _col(int(col), name)
    for name, spec in qpos_init.get("grippers", {}).items():
        val = _col(int(spec["col"]), name)
        out[name] = (spec["offset"] - val) * spec["range"] / spec["scale"]
    return out


# ---------------------------------------------------------------------------
# RB-Y1 raw recorder logs (npz) -> the dataset dict systemid.py replays
# ---------------------------------------------------------------------------

RBY1_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)]
RBY1_TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
# Joints the joint-mode fit compares, in observation.state column order.
RBY1_STATE_JOINTS = RBY1_ARM_JOINTS + RBY1_TORSO_JOINTS


def load_rby1_npz(path, fps: int, sync_tol: float | None = 0.2) -> dict[str, np.ndarray]:
    """Read one RB-Y1 teleop log (the real recorder's npz) as a single-episode dataset.

    The recorder samples at ~38.6 Hz with jitter; the sim steps at `fps`, so every
    channel is linearly interpolated onto a uniform 1/fps grid spanning the log.

    `sync_tol`: the log starts with the follower CATCHING UP to wherever the leader was
    when teleop engaged (0.4-0.5 rad apart, ~4 s, rate-limited -- not PD dynamics). A stiff
    sim env snaps to the leader at once, overshoots the real follower and gets rejected, so
    the replay begins at the first frame where every arm joint is within `sync_tol` rad of
    its leader. None keeps the whole log.

    Columns (both arrays: right arm first, then left, matching the recorder):
      action (N,22)            = [leader R0..6, leader L0..6, torso_0..5 (recorded, held),
                                  gripR, gripL]   grip: -1 close (command > 0.5) / +1 open
      observation.state (N,20) = [position R0..6, position L0..6, torso_0..5]
      episode_index (N,)       = 0
    """
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["joint_names"]]
    col = {n: i for i, n in enumerate(names)}
    for j in RBY1_STATE_JOINTS:
        if j not in col:
            raise ValueError(f"{path}: joint_names has no {j!r}; got {names}")
    # The leader stream carries no names; it is documented right-then-left, so the
    # follower must be in the same order or the two streams would be paired crosswise.
    right = [col[f"right_arm_{i}"] for i in range(7)]
    left = [col[f"left_arm_{i}"] for i in range(7)]
    if right != list(range(right[0], right[0] + 7)) or left != list(range(left[0], left[0] + 7)) \
            or left[0] < right[0]:
        raise ValueError(
            f"{path}: expected right_arm_0..6 then left_arm_0..6 as contiguous blocks in "
            f"joint_names (the leader_position layout); got {names}")

    t = np.asarray(z["time_s"], dtype=np.float64)
    pos = np.asarray(z["position"], dtype=np.float64)
    leader = np.asarray(z["leader_position"], dtype=np.float64)
    grip = np.asarray(z["gripper_command"], dtype=np.float64)
    if leader.shape[1] != 14 or grip.shape[1] != 2:
        raise ValueError(f"{path}: leader_position {leader.shape}, gripper_command {grip.shape}")

    if sync_tol is not None:
        arm_cols = [col[j] for j in RBY1_ARM_JOINTS]
        gap = np.abs(pos[:, arm_cols] - leader).max(axis=1)
        synced = np.flatnonzero(gap < sync_tol)
        if synced.size == 0:
            raise ValueError(f"{path}: follower never comes within {sync_tol} rad of the leader "
                             f"(min gap {gap.min():.3f}); pass a larger sync_tol")
        i0 = int(synced[0])
        if i0 > 0:
            print(f"[load_rby1_npz] {path}: skipping {i0} frames ({t[i0] - t[0]:.2f} s) of "
                  f"leader/follower catch-up (gap {gap[0]:.3f} -> <{sync_tol} rad)", file=sys.stderr)
        t, pos, leader, grip = t[i0:], pos[i0:], leader[i0:], grip[i0:]

    grid = np.arange(0.0, t[-1] - t[0] + 1e-9, 1.0 / fps) + t[0]

    def _resample(x: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(grid, t, x[:, k]) for k in range(x.shape[1])], axis=1)

    state = _resample(pos[:, [col[j] for j in RBY1_STATE_JOINTS]])
    torso = state[:, 14:20]
    grip_act = np.where(_resample(grip) > 0.5, -1.0, 1.0)
    action = np.concatenate([_resample(leader), torso, grip_act], axis=1)
    return {
        "action": action,
        "observation.state": state,
        "episode_index": np.zeros(grid.shape[0], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# MolmoBot (allenai/molmobot-data) RB-Y1 trajectories
# ---------------------------------------------------------------------------

def load_molmobot_h5(path, traj: str = "traj_0", fps: int = 20,
                     grip_closed_below: float | None = None) -> dict[str, np.ndarray]:
    """One MolmoBot RB-Y1 trajectory in the same layout `load_rby1_npz` returns.

    `allenai/molmobot-data` stores each field as a per-frame JSON blob padded into a uint8
    row. `actions/joint_pos` is the ABSOLUTE commanded joint position and `obs/agent/qpos`
    the achieved one -- the same command/response pair the teleop logs give as
    leader_position/position, so it replays identically.

    Two things to know about the comparison it supports. The data is SIMULATED, in MuJoCo
    (`mujoco_thor`), so a replay measures agreement with THEIR simulator, not with hardware.
    And their arm tracks its own command to ~0.014 rad, far tighter than the real robot's
    0.016-0.025, so it is an easy trajectory to follow: matching it is weak evidence, while
    failing to match it would be strong evidence against a gain set.

    The first frame carries an empty action dict and is dropped. `fps` is assumed, not
    recorded in the file; it only sets the replay rate.
    """
    import h5py

    def _dec(ds, i):
        raw = ds[i].tobytes().rstrip(b"\x00")
        return json.loads(raw.decode()) if raw else {}

    with h5py.File(path, "r") as f:
        if traj not in f:
            raise ValueError(f"{path}: no {traj!r} (have {sorted(f.keys())[:8]})")
        a_ds, q_ds = f[f"{traj}/actions/joint_pos"], f[f"{traj}/obs/agent/qpos"]
        n = a_ds.shape[0]
        acts = [_dec(a_ds, i) for i in range(n)]
        qs = [_dec(q_ds, i) for i in range(n)]

    keep = [i for i in range(n) if acts[i].get("right_arm") is not None
            and qs[i].get("right_arm") is not None]
    if not keep:
        raise ValueError(f"{path}:{traj} has no frame with both a command and a state")

    def _grip(d, side):
        v = d.get(f"{side}_gripper")
        return float(np.asarray(v).reshape(-1)[0]) if v is not None else np.nan

    graw = np.array([[_grip(acts[i], "right"), _grip(acts[i], "left")] for i in keep])
    if grip_closed_below is None:
        # No documented convention: split each hand's own command range at its midpoint.
        finite = graw[np.isfinite(graw).all(axis=1)]
        grip_closed_below = float((np.nanmin(finite) + np.nanmax(finite)) / 2.0) if finite.size \
            else 0.0
    grip = np.where(np.nan_to_num(graw, nan=grip_closed_below + 1.0) < grip_closed_below, -1.0, 1.0)

    arms_a = np.array([np.concatenate([acts[i]["right_arm"], acts[i]["left_arm"]]) for i in keep])
    arms_q = np.array([np.concatenate([qs[i]["right_arm"], qs[i]["left_arm"]]) for i in keep])
    torso_q = np.array([np.asarray(qs[i].get("torso", np.zeros(6)), dtype=float) for i in keep])
    # TORSO COMMAND COMES FROM THE OBSERVED qpos, NOT THE ACTION. Different MolmoBot tasks
    # abstract the torso differently -- pick-and-place commands all 6 joints, door opening
    # commands a single value while still reporting 6 in qpos -- so the action's torso width is
    # not a fixed 6 and building the action from it produced a 17-wide array that the joint
    # layout could not index. The torso is held, not fitted (the arms are what is being
    # identified), so commanding it to its recorded posture is both robust and what we want.
    torso_a = torso_q

    action = np.concatenate([arms_a, torso_a, grip], axis=1)          # (N, 22)
    state = np.concatenate([arms_q, torso_q], axis=1)                 # (N, 20)
    if action.shape[1] != 22 or state.shape[1] != 20:
        raise ValueError(f"{path}:{traj} built action {action.shape} / state {state.shape}; "
                         f"expected (N,22) and (N,20) -- the joint layout cannot index anything else")
    # The recorded HEAD TILT, so a replay can aim its head camera where theirs was aimed. It is
    # constant within an episode but differs by task: pick-and-place sits at ~0.5 deg while
    # door-opening holds 34-42 deg, and a fixed camera makes the two panels of a comparison video
    # look at different things while the joints themselves agree.
    head = np.array([np.asarray(qs[i].get("head", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
                     for i in keep])
    head_tilt = float(np.median(head[:, 1])) if head.size else 0.0
    print(f"[load_molmobot_h5] {path}:{traj} -> {len(keep)} frames at an assumed {fps} Hz "
          f"(gripper closed below {grip_closed_below:.3f}, head tilt {head_tilt:.3f} rad)",
          file=sys.stderr)
    return {"action": action, "observation.state": state,
            "episode_index": np.zeros(len(keep), dtype=np.int64),
            "head_tilt": head_tilt}


# ---------------------------------------------------------------------------
# Rainbow Robotics RB-Y1 LeRobot datasets (real hardware)
# ---------------------------------------------------------------------------

def load_rainbow_parquet(path, episode: int | None = None, fps: int = 20,
                         grip_closed_above: float = 0.5) -> dict[str, np.ndarray]:
    """One episode of a Rainbow Robotics RB-Y1 LeRobot v3 dataset, resampled to `fps`.

    THE METRIC THIS SUPPORTS IS NOT THE MOLMOBOT ONE. Those datasets record a commanded JOINT
    trajectory next to the achieved one, so a replay measures our dynamics against theirs.
    Rainbow's `action` is END-EFFECTOR poses (torso/right/left 6-DoF + grippers + base velocity),
    so no joint command exists to replay. What is available is `observation.state`, which does
    carry every joint: torso_0..5, right_arm_0..6, left_arm_0..6 and the two grippers.

    So this commands the RECORDED joint trajectory one step ahead -- action[t] = q[t+1] -- and
    the termination compares the post-step sim against that same q[t+1]. The error is therefore
    purely OUR arm's inability to follow a trajectory the real robot actually executed, with no
    contribution from the real robot's own tracking lag. It is a fair dynamics-fidelity measure
    on real hardware data, but it is a different quantity from the MolmoBot numbers and must not
    be tabulated beside them without saying so.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if episode is not None:
        df = df[df["episode_index"] == episode]
        if df.empty:
            raise ValueError(f"{path}: no episode {episode} (have {sorted(pd.read_parquet(path)['episode_index'].unique())[:8]})")
    else:
        episode = int(df["episode_index"].iloc[0])
        df = df[df["episode_index"] == episode]
    df = df.sort_values("frame_index")

    state = np.stack(df["observation.state"].values).astype(np.float64)
    t = np.stack(df["timestamp"].values).reshape(-1).astype(np.float64)

    # Column order is fixed by the dataset's meta/info.json feature names.
    TORSO = list(range(0, 6))
    RIGHT = list(range(6, 13))
    LEFT = list(range(13, 20))
    GR, GL = 20, 21

    grid = np.arange(0.0, t[-1] - t[0] + 1e-9, 1.0 / fps) + t[0]

    def _rs(cols):
        return np.stack([np.interp(grid, t, state[:, c]) for c in cols], axis=1)

    arms = np.concatenate([_rs(RIGHT), _rs(LEFT)], axis=1)      # (N,14) right then left
    torso = _rs(TORSO)                                          # (N,6)
    grip = np.where(_rs([GR, GL]) > grip_closed_above, -1.0, 1.0)

    q = np.concatenate([arms, torso], axis=1)                   # (N,20) the fit target
    # command = the NEXT recorded pose; the last frame repeats so lengths match
    cmd = np.concatenate([q[1:], q[-1:]], axis=0)
    action = np.concatenate([cmd, grip], axis=1)                # (N,22)
    print(f"[load_rainbow_parquet] {path} ep{episode} -> {len(grid)} frames at {fps} Hz "
          f"(from {len(t)} at {1/np.median(np.diff(t)):.0f} Hz); commands the recorded "
          f"trajectory one step ahead", file=sys.stderr)
    return {"action": action, "observation.state": q,
            "episode_index": np.zeros(len(grid), dtype=np.int64)}


# Torso posture both real RB-Y1 datasets are recorded in (teleop logs and Rainbow's door set).
RBY1_WORKING_TORSO = (0.0, 0.785, -1.571, 0.785, 0.0, 0.0)


def load_rby1_reference_parquet(path, episode: int | None = None, fps: int = 20,
                                torso=RBY1_WORKING_TORSO,
                                grip_closed_above: float = 0.5) -> dict[str, np.ndarray]:
    """One episode of a right-arm-only RB-Y1 LeRobot set that records COMMAND and ACHIEVED.

    Unlike the Rainbow door dataset, `action` here is joint positions (right_arm_0..6 plus the
    gripper) rather than end-effector poses, so this supports the SAME metric as the teleop logs
    and the MolmoBot replays: drive the sim with the recorded command and score it against the
    recorded achieved state. The real arm's own command-to-achieved error is 0.027-0.038 rad,
    which is the figure our replay should be read against.

    Only the right arm is recorded. The left arm is held at the sim's own initial pose and the
    torso at `torso` -- the bent working posture both other real datasets sit in. That posture is
    an ASSUMPTION here (this dataset does not record it) and it matters, because the torso angle
    sets how gravity loads the arm. Restrict the comparison to the right arm (systemid.py
    --fit_joints) or the held joints will dilute the error toward zero.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if episode is None:
        episode = int(df["episode_index"].iloc[0])
    df = df[df["episode_index"] == episode].sort_values("frame_index")
    if df.empty:
        raise ValueError(f"{path}: no episode {episode}")

    act = np.stack(df["action"].values).astype(np.float64)
    obs = np.stack(df["observation.state"].values).astype(np.float64)
    t = np.stack(df["timestamp"].values).reshape(-1).astype(np.float64)
    grid = np.arange(0.0, t[-1] - t[0] + 1e-9, 1.0 / fps) + t[0]
    rs = lambda a, c: np.interp(grid, t, a[:, c])                      # noqa: E731

    n = len(grid)
    cmd_r = np.stack([rs(act, i) for i in range(7)], axis=1)           # commanded right arm
    ach_r = np.stack([rs(obs, i) for i in range(7)], axis=1)           # achieved right arm
    left = np.zeros((n, 7))                                            # held; filled by the runner's init
    torso_a = np.tile(np.asarray(torso, dtype=float), (n, 1))
    grip_r = np.where(rs(act, 7) > grip_closed_above, -1.0, 1.0)
    grip = np.stack([grip_r, np.ones(n)], axis=1)                      # left hand stays open

    action = np.concatenate([cmd_r, left, torso_a, grip], axis=1)      # (N,22)
    state = np.concatenate([ach_r, left, torso_a], axis=1)             # (N,20)
    print(f"[load_rby1_reference] {path} ep{episode} -> {n} frames at {fps} Hz; right arm only, "
          f"left held, torso at {tuple(round(v,3) for v in torso)}", file=sys.stderr)
    return {"action": action, "observation.state": state,
            "episode_index": np.zeros(n, dtype=np.int64)}
