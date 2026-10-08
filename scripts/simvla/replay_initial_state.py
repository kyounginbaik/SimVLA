"""Restore per-episode object reset poses recorded alongside a LeRobot dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path


def episode_start_indices(is_first, episode_indices, requested=None):
    """Return eligible row starts, optionally selecting an exact recorded episode ID."""
    if len(is_first) != len(episode_indices):
        raise ValueError("episode boundary and ID columns have different lengths")
    if requested is not None and (type(requested) is not int or requested < 0):
        raise ValueError("episode index must be a nonnegative integer")
    starts = [i for i, first in enumerate(is_first)
              if first == 1 and (requested is None or episode_indices[i] == requested)]
    if not starts:
        raise ValueError(f"no episode starts match requested episode {requested}")
    if requested is not None and len(starts) != 1:
        raise ValueError(f"episode {requested} has multiple start rows")
    return starts


def initial_robot_root_for_episode(dataset_root: str | Path, episode_index: int) -> dict | None:
    """Read the articulation-root reset pose, not the post-step base_link observation."""
    path = Path(dataset_root) / "meta" / "scene_states.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("frame") != "environment":
        raise ValueError(f"unsupported scene state schema/frame in {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not 0 <= episode_index < len(episodes):
        raise ValueError(f"episode {episode_index} has no scene state in {path}")
    initial = episodes[episode_index].get("initial")
    if initial is None:
        return None  # older recordings can have terminal state only
    root = initial.get("articulation", {}).get("robot")
    if not isinstance(root, dict):
        raise ValueError(f"initial scene state has no robot articulation in {path}")
    for key, size in (("root_pose", 7), ("root_velocity", 6)):
        values = root.get(key)
        if (not isinstance(values, list) or len(values) != size
                or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
            raise ValueError(f"invalid robot {key} in {path}")
    if abs(sum(v * v for v in root["root_pose"][3:]) - 1.0) > 1e-3:
        raise ValueError(f"invalid robot root quaternion in {path}")
    return {key: root[key] for key in ("root_pose", "root_velocity")}


def apply_initial_robot_root(env_cfg, root: dict) -> None:
    """Restore the recorded root frame without assuming it coincides with base_link."""
    pose, velocity = root["root_pose"], root["root_velocity"]
    init = env_cfg.scene.robot.init_state
    # Preserve the authored spawn frame (including fixed virtual-base joint
    # anchors). Collection moves the root through this reset event, not by
    # changing the USD's spawn transform. Use the same mechanism for replay.
    a, b, c, d = init.rot
    w, x, y, z = pose[3:]
    qw, qx, qy, qz = (
        a*w+b*x+c*y+d*z, a*x-b*w-c*z+d*y,
        a*y+b*z-c*w-d*x, a*z-b*y+c*x-d*w,
    )  # conjugate(default) * recorded; quaternions are wxyz
    angles = (
        math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)),
        math.asin(max(-1., min(1., 2*(qw*qy-qz*qx)))),
        math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)),
    )
    offsets = [pose[i] - init.pos[i] for i in range(3)] + list(angles)
    default_velocity = tuple(init.lin_vel) + tuple(init.ang_vel)
    params = env_cfg.events.robot_init_pos.params
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    params["pose_range"] = {axis: (value, value) for axis, value in zip(axes, offsets)}
    params["velocity_range"] = {
        axis: (value-default, value-default)
        for axis, value, default in zip(axes, velocity, default_velocity)
    }


def initial_robot_joints_for_episode(dataset_root: str | Path, episode_index: int) -> dict:
    """Read measured joint state in the recorder's named native articulation order."""
    root = Path(dataset_root)
    # This mode requires an initial snapshot and named motor-target metadata.
    if initial_robot_root_for_episode(root, episode_index) is None:
        raise ValueError("initial joint-state replay requires a recorded initial scene state")
    payload = json.loads((root / "meta/scene_states.json").read_text())
    robot = payload["episodes"][episode_index]["initial"]["articulation"]["robot"]
    info = json.loads((root / "meta/info.json").read_text())
    feature = info.get("features", {}).get("action.joint", {})
    names = feature.get("names", {})
    if isinstance(names, dict):
        names = names.get("action.joint")
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names) or feature.get("shape") != [len(names)]):
        raise ValueError("initial joint-state replay requires unique native joint names")
    result = {"names": names}
    for key in ("joint_position", "joint_velocity"):
        values = robot.get(key)
        if (not isinstance(values, list) or len(values) != len(names)
                or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
            raise ValueError(f"invalid initial robot {key}")
        result[key] = values
    return result


def apply_initial_robot_joints(robot, state: dict) -> None:
    """Restore measured state only; velocities are NOT velocity-control targets."""
    current = list(robot.joint_names)
    if len(set(current)) != len(current) or set(current) != set(state["names"]):
        raise ValueError("recorded and runtime robot joint names differ")
    order = [state["names"].index(name) for name in current]
    positions = robot.data.joint_pos.new_tensor([state["joint_position"][i] for i in order])
    velocities = robot.data.joint_vel.new_tensor([state["joint_velocity"][i] for i in order])
    robot.write_joint_state_to_sim(positions.expand_as(robot.data.joint_pos).clone(),
                                   velocities.expand_as(robot.data.joint_vel).clone())


def initial_objects_for_episode(dataset_root: str | Path, episode_index: int) -> dict | None:
    path = Path(dataset_root) / "meta" / "initial_objects.json"
    if not path.exists():
        return None  # older datasets did not record this state
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported initial object pose schema in {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not 0 <= episode_index < len(episodes):
        raise ValueError(f"episode {episode_index} has no initial object pose in {path}")
    objects = episodes[episode_index]
    if not isinstance(objects, dict):
        raise ValueError(f"episode {episode_index} has no initial object pose in {path}")
    for name, pose in objects.items():
        if not isinstance(name, str) or not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"invalid initial pose for {name!r} in {path}")
        if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in pose):
            raise ValueError(f"non-finite initial pose for {name!r} in {path}")
    return objects


def initial_object_velocities_for_episode(dataset_root: str | Path, episode_index: int) -> dict | None:
    """Reset snapshots can contain residual momentum; poses alone are insufficient."""
    path = Path(dataset_root) / "meta" / "scene_states.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("frame") != "environment":
        raise ValueError(f"unsupported scene state schema/frame in {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not 0 <= episode_index < len(episodes):
        raise ValueError(f"episode {episode_index} has no scene state in {path}")
    initial = episodes[episode_index].get("initial")
    if initial is None:
        return None
    result = {}
    for name, state in initial.get("rigid_object", {}).items():
        velocity = state.get("root_velocity")
        if (not isinstance(velocity, list) or len(velocity) != 6
                or not all(type(v) in (int, float) and math.isfinite(v) for v in velocity)):
            raise ValueError(f"invalid initial velocity for {name!r} in {path}")
        result[name] = velocity
    return result


def apply_initial_objects(env_cfg, objects: dict, velocities: dict | None = None) -> list[str]:
    """Set known scene objects' reset poses and disable their uniform reset offsets."""
    reset_events = []
    for event_name in dir(env_cfg.events):
        if event_name.startswith("_"):
            continue
        event = getattr(env_cfg.events, event_name)
        params = getattr(event, "params", None)
        if isinstance(params, dict) and isinstance(params.get("pose_range"), dict):
            reset_events.append(params)

    resettable = {getattr(params.get("asset_cfg"), "name", None) for params in reset_events}
    applied = []
    for name, pose in objects.items():
        if name not in resettable:
            continue
        asset = getattr(env_cfg.scene, name, None)
        if asset is None or not hasattr(asset, "init_state"):
            continue
        asset.init_state.pos = tuple(pose[:3])
        asset.init_state.rot = tuple(pose[3:])
        if velocities is not None and name in velocities:
            asset.init_state.lin_vel = tuple(velocities[name][:3])
            asset.init_state.ang_vel = tuple(velocities[name][3:])
        applied.append(name)

    for params in reset_events:
        asset_cfg = params.get("asset_cfg")
        if getattr(asset_cfg, "name", None) not in applied:
            continue
        pose_range = params.get("pose_range")
        if isinstance(pose_range, dict):
            for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                if axis in pose_range:
                    pose_range[axis] = (0.0, 0.0)
        if velocities is not None and getattr(asset_cfg, "name", None) in velocities:
            for axis in params.get("velocity_range", {}):
                params["velocity_range"][axis] = (0.0, 0.0)
    return applied
