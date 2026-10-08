"""An open-loop replay must restore the object's recorded reset pose."""

import json
import math
from types import SimpleNamespace

import pytest

from replay_initial_state import (
    apply_initial_objects, initial_objects_for_episode,
    apply_initial_robot_root, initial_robot_root_for_episode,
    episode_start_indices,
    initial_object_velocities_for_episode,
    initial_robot_joints_for_episode, apply_initial_robot_joints,
)


def joint_snapshot(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    state = {"root_pose": [0., 0., 0., 1., 0., 0., 0.], "root_velocity": [0.] * 6,
             "joint_position": [.1, .2], "joint_velocity": [.3, -.4]}
    (meta / "scene_states.json").write_text(json.dumps({
        "schema_version": 1, "frame": "environment",
        "episodes": [{"initial": {"articulation": {"robot": state}}}]}))
    (meta / "info.json").write_text(json.dumps({"features": {"action.joint": {
        "shape": [2], "names": {"action.joint": ["second", "first"]}}}}))
    return meta


def test_initial_robot_joints_require_named_finite_state(tmp_path):
    meta = joint_snapshot(tmp_path)
    result = initial_robot_joints_for_episode(tmp_path, 0)
    assert result == {"names": ["second", "first"], "joint_position": [.1, .2],
                      "joint_velocity": [.3, -.4]}
    info = json.loads((meta / "info.json").read_text())
    info["features"]["action.joint"]["names"]["action.joint"] = ["same", "same"]
    (meta / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="unique native joint names"):
        initial_robot_joints_for_episode(tmp_path, 0)


@pytest.mark.parametrize("value", [None, [], [1.], [True, 0.], [float("nan"), 0.]])
def test_initial_robot_joint_velocity_is_not_silently_dropped(tmp_path, value):
    meta = joint_snapshot(tmp_path)
    payload = json.loads((meta / "scene_states.json").read_text())
    payload["episodes"][0]["initial"]["articulation"]["robot"]["joint_velocity"] = value
    (meta / "scene_states.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="joint_velocity"):
        initial_robot_joints_for_episode(tmp_path, 0)


def test_restore_joint_state_reorders_names_without_setting_drive_targets(tmp_path):
    torch = pytest.importorskip("torch")
    joint_snapshot(tmp_path)
    written = []
    robot = SimpleNamespace(joint_names=["first", "second"],
                            data=SimpleNamespace(joint_pos=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2)),
                            write_joint_state_to_sim=lambda q, qd: written.append((q, qd)))
    state = initial_robot_joints_for_episode(tmp_path, 0)
    apply_initial_robot_joints(robot, state)
    assert torch.allclose(written[0][0], torch.tensor([[.2, .1], [.2, .1]]))
    assert torch.allclose(written[0][1], torch.tensor([[-.4, .3], [-.4, .3]]))
    robot.joint_names = ["different", "second"]
    with pytest.raises(ValueError, match="names differ"):
        apply_initial_robot_joints(robot, state)


def test_explicit_replay_episode_selection():
    first = [True, False, True, False, True]
    ids = [0, 0, 1, 1, 2]
    assert episode_start_indices(first, ids) == [0, 2, 4]
    assert episode_start_indices(first, ids, 1) == [2]
    assert episode_start_indices(first, ids, 0) == [0]
    with pytest.raises(ValueError, match="no episode starts"):
        episode_start_indices(first, ids, 3)
    with pytest.raises(ValueError, match="multiple start"):
        episode_start_indices([True, True], [0, 0], 0)


@pytest.mark.parametrize("requested", [-1, True, 0.0, "0"])
def test_replay_episode_index_rejects_invalid_values(requested):
    with pytest.raises(ValueError, match="nonnegative integer"):
        episode_start_indices([True], [0], requested)


def test_replay_episode_columns_must_align():
    with pytest.raises(ValueError, match="different lengths"):
        episode_start_indices([True], [])


def test_older_dataset_has_no_object_pose(tmp_path):
    assert initial_objects_for_episode(tmp_path, 0) is None
    assert initial_robot_root_for_episode(tmp_path, 0) is None
    assert initial_object_velocities_for_episode(tmp_path, 0) is None


def test_object_momentum_is_restored_before_the_first_replay_step(tmp_path):
    velocity = [.066, 0., -.014, -.022, 1.953, -.046]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/scene_states.json").write_text(json.dumps({
        "schema_version": 1, "frame": "environment", "episodes": [{"initial": {
            "rigid_object": {"bowl0": {"root_velocity": velocity}}}}]}))
    restored = initial_object_velocities_for_episode(tmp_path, 0)
    asset = SimpleNamespace(init_state=SimpleNamespace())
    event = SimpleNamespace(params={"asset_cfg": SimpleNamespace(name="bowl0"),
                                    "pose_range": {"x": (-1., 1.)},
                                    "velocity_range": {"yaw": (-1., 1.)}})
    cfg = SimpleNamespace(scene=SimpleNamespace(bowl0=asset),
                          events=SimpleNamespace(obj_init_pos=event))
    assert apply_initial_objects(cfg, {"bowl0": [0, 0, 1, 1, 0, 0, 0]}, restored) == ["bowl0"]
    assert asset.init_state.lin_vel == tuple(velocity[:3])
    assert asset.init_state.ang_vel == tuple(velocity[3:])
    assert event.params["velocity_range"]["yaw"] == (0., 0.)


@pytest.mark.parametrize("velocity", [None, [0.] * 5, [float("nan")] * 6, [True] * 6])
def test_initial_object_momentum_rejects_corrupt_snapshots(tmp_path, velocity):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/scene_states.json").write_text(json.dumps({
        "schema_version": 1, "frame": "environment", "episodes": [{"initial": {
            "rigid_object": {"bowl0": {"root_velocity": velocity}}}}]}))
    with pytest.raises(ValueError, match="invalid initial velocity"):
        initial_object_velocities_for_episode(tmp_path, 0)


def test_robot_root_is_not_base_link_and_zeroes_reset_offsets(tmp_path):
    root = {"root_pose": [.8471096754, -2.1963748932, .01, 1, 0, 0, 0],
            "root_velocity": [0, 0, 0, 0, 0, 0]}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/scene_states.json").write_text(json.dumps({
        "schema_version": 1, "frame": "environment",
        "episodes": [{"initial": {"articulation": {"robot": root}}}],
    }))
    restored = initial_robot_root_for_episode(tmp_path, 0)
    cfg = SimpleNamespace(
        scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace(
            pos=(0, 0, .01), rot=(1, 0, 0, 0), lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)))),
        events=SimpleNamespace(robot_init_pos=SimpleNamespace(params={
            "pose_range": {"x": (.711586, .711587)}, "velocity_range": {"x": (-1, 1)}})))
    apply_initial_robot_root(cfg, restored)
    assert cfg.scene.robot.init_state.pos == (0, 0, .01)
    assert cfg.scene.robot.init_state.rot == (1, 0, 0, 0)
    assert cfg.scene.robot.init_state.lin_vel == (0, 0, 0)
    params = cfg.events.robot_init_pos.params
    assert params["pose_range"]["x"] == (root["root_pose"][0],) * 2
    assert params["pose_range"]["y"] == (root["root_pose"][1],) * 2
    assert params["pose_range"]["z"] == (0., 0.)
    assert all(v == (0., 0.) for v in params["velocity_range"].values())


@pytest.mark.parametrize("corruption", ["frame", "version", "pose", "velocity", "quaternion", "episode"])
def test_robot_root_rejects_invalid_snapshot(tmp_path, corruption):
    root = {"root_pose": [1, 2, .01, 1, 0, 0, 0], "root_velocity": [0] * 6}
    payload = {"schema_version": 1, "frame": "environment",
               "episodes": [{"initial": {"articulation": {"robot": root}}}]}
    if corruption == "frame": payload["frame"] = "world"
    elif corruption == "version": payload["schema_version"] = 2
    elif corruption == "pose": root["root_pose"][0] = float("nan")
    elif corruption == "velocity": root["root_velocity"][0] = True
    elif corruption == "quaternion": root["root_pose"][3] = 0
    elif corruption == "episode": payload["episodes"] = []
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/scene_states.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        initial_robot_root_for_episode(tmp_path, 0)


def test_terminal_only_snapshot_does_not_invent_initial_root(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/scene_states.json").write_text(json.dumps({
        "schema_version": 1, "frame": "environment", "episodes": [{"initial": None, "terminal": {}}],
    }))
    assert initial_robot_root_for_episode(tmp_path, 0) is None


def test_robot_root_offsets_preserve_nonidentity_spawn_frame():
    init = SimpleNamespace(pos=(1., 2., .1), rot=(2**-.5, 0., 0., 2**-.5),
                           lin_vel=(.1, .2, .3), ang_vel=(.4, .5, .6))
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(init_state=init)),
                          events=SimpleNamespace(robot_init_pos=SimpleNamespace(params={})))
    apply_initial_robot_root(cfg, {"root_pose": [2., 4., .2, 0., 0., 0., 1.],
                                  "root_velocity": [0.] * 6})
    params = cfg.events.robot_init_pos.params
    assert params["pose_range"]["x"] == (1., 1.)
    assert params["pose_range"]["yaw"] == pytest.approx((math.pi/2,) * 2)
    assert params["velocity_range"]["yaw"] == (-.6, -.6)
    assert init.pos == (1., 2., .1)


def test_pose_sidecar_selects_episode_and_rejects_bad_values(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    path = meta / "initial_objects.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "episodes": [{"bowl0": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]}],
    }))
    assert initial_objects_for_episode(tmp_path, 0)["bowl0"][:3] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="episode 1"):
        initial_objects_for_episode(tmp_path, 1)
    path.write_text(json.dumps({"schema_version": 1, "episodes": [{"bowl0": [float("nan")] * 7}]}))
    with pytest.raises(ValueError, match="non-finite"):
        initial_objects_for_episode(tmp_path, 0)


def test_apply_pose_disables_object_reset_randomization():
    asset = SimpleNamespace(init_state=SimpleNamespace(pos=(0, 0, 0), rot=(1, 0, 0, 0)))
    static = SimpleNamespace(init_state=SimpleNamespace(pos=(4, 5, 6), rot=(1, 0, 0, 0)))
    event = SimpleNamespace(params={
        "asset_cfg": SimpleNamespace(name="bowl0"),
        "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1, 1)},
    })
    cfg = SimpleNamespace(scene=SimpleNamespace(bowl0=asset, countertop=static), events=SimpleNamespace(obj_init_pos=event))
    assert apply_initial_objects(cfg, {
        "bowl0": [1, 2, 3, 1, 0, 0, 0],
        "countertop": [7, 8, 9, 1, 0, 0, 0],
    }) == ["bowl0"]
    assert asset.init_state.pos == (1, 2, 3)
    assert static.init_state.pos == (4, 5, 6)
    assert event.params["pose_range"]["x"] == (0.0, 0.0)
    assert event.params["pose_range"]["yaw"] == (0.0, 0.0)
