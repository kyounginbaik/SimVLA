"""Synthetic export integrity checks; these are NOT successful robot episodes."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot")
from isaaclab.utils.datasets import TwoPhaseEpisodeWriter
from isaaclab.utils.datasets.robot_schemas import get_robot_schema


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_offline_export_all_robot_schemas_without_isaac(tmp_path, robot):
    schema = get_robot_schema(robot)
    frames = 20
    images = np.empty((frames, 240, 320, 3), dtype=np.uint8)
    for i in range(frames):
        images[i] = i * 10
    episode = SimpleNamespace(
        is_empty=lambda: False, seed=0, env_id=0, success=True,
        data={
            "raw_actions": np.zeros((frames, 17)),
            "initial_state": {"rigid_object": {"mug0": {"root_pose": np.array([[1., 2., 3., 1., 0., 0., 0.]])}}},
            "terminal_state": {"rigid_object": {"mug0": {"root_pose": np.array([[4., 5., 6., 1., 0., 0., 0.]])}}},
            "observations": {
                "images": dict.fromkeys(("front", "wrist_left", "wrist_right"), images),
                "ee_6D_pos": np.repeat(np.arange(frames)[:, None], 20, axis=1),
                "joint_angles": np.repeat(np.arange(frames)[:, None], len(schema["joint_angles"]), axis=1),
                "base_vel": np.zeros((frames, 3)),
                "mobile_base_world_frame": np.zeros((frames, 6)),
                "subtask_index": np.zeros(frames),
            },
            "actions": {
                "ee_6D_pos": np.repeat((100 + np.arange(frames))[:, None], 20, axis=1),
                "base": np.zeros((frames, 3)),
                "joint_pos": np.zeros((frames, len(schema["action_joint"]))),
                "joint_substeps": np.repeat(
                    np.arange(frames * 6, dtype=np.float32).reshape(frames, 6, 1),
                    len(schema["action_joint"]), axis=2),
            },
        })
    episode.data["actions"]["ee_6D_pos"] = episode.data["actions"]["ee_6D_pos"].astype(float)
    episode.data["actions"]["ee_6D_pos"][:, [9, 19]] = -1.6
    stage = tmp_path / "stage"
    TwoPhaseEpisodeWriter(str(stage), robot=robot, record_initial_step=True).write_episode(episode)
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"kitchen_num": 813, "kitchen_sub_num": 0,
                               "kitchen_type": "l_shaped", "language": "Synthetic export test."}))
    script = Path(__file__).resolve().parents[1] / "tools/offline_finalize_lerobot.py"
    result = subprocess.run([
        sys.executable, str(script), "--stage", str(stage), "--output", str(tmp_path / "export"),
        "--goal", str(goal), "--robot", robot], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads((tmp_path / "export/all/meta/info.json").read_text())
    assert info["total_episodes"] == 1
    # The collector's historical contract removes the initial frame from ALL streams.
    assert info["total_frames"] == frames - 1
    motor_meta = json.loads((tmp_path / "export/all/meta/joint_substeps.json").read_text())
    assert motor_meta["joint_names"] == schema["action_joint"]
    assert motor_meta["control_fps"] == 20
    sequence = np.load(tmp_path / "export/all/meta" / motor_meta["episodes"][0]["file"])
    assert sequence.shape == (frames - 1, 6, len(schema["action_joint"]))
    assert sequence[:, :, 0].reshape(-1).tolist() == list(range(6, frames * 6))
    scene_states = json.loads((tmp_path / "export/all/meta/scene_states.json").read_text())
    assert scene_states["frame"] == "environment"
    prelude = scene_states["episodes"][0]["initial_step"]
    assert prelude["initial_state_phase"] == "before_first_action"
    assert prelude["raw_action"] == [0.] * 17
    assert prelude["motor_targets"] == [0.] * len(schema["action_joint"])
    assert [row[0] for row in prelude["motor_substeps"]] == list(range(6))
    assert scene_states["episodes"][0]["initial"]["rigid_object"]["mug0"]["root_pose"][:3] == [1., 2., 3.]
    assert scene_states["episodes"][0]["terminal"]["rigid_object"]["mug0"]["root_pose"][:3] == [4., 5., 6.]
    import pyarrow.parquet as pq
    data = next((tmp_path / "export/all/data").rglob("*.parquet"))
    joints = pq.read_table(data, columns=["joint_angles"])["joint_angles"].to_pylist()
    assert [row[0] for row in joints] == list(range(1, frames))
    aligned = pq.read_table(data, columns=["observation.state", "action"]).to_pydict()
    assert [row[0] for row in aligned["observation.state"]] == list(range(1, frames))
    assert [row[0] for row in aligned["action"]] == list(range(101, 100 + frames))
    import av
    video = next((tmp_path / "export/all/videos/observation.images.front").rglob("*.mp4"))
    with av.open(str(video)) as container:
        first = next(container.decode(video=0)).to_ndarray(format="rgb24")
    assert abs(float(first.mean()) - 10) <= 3  # Frame 1, allowing lossy codec rounding.
    assert "Export verified" in result.stdout
    # Optional physics-rate sidecars are part of export integrity, not unchecked
    # diagnostic files. A corrupt sample must fail the same public validator.
    valid_sequence = sequence.copy()
    sequence[0, 0, 0] = np.nan
    np.save(tmp_path / "export/all/meta" / motor_meta["episodes"][0]["file"], sequence)
    validator = script.with_name("validate_lerobot_collection.py")
    corrupt = subprocess.run([sys.executable, str(validator), str(tmp_path / "export/all")],
                             capture_output=True, text=True, timeout=30)
    assert corrupt.returncode != 0
    assert "Invalid motor substep array" in corrupt.stderr
    np.save(tmp_path / "export/all/meta" / motor_meta["episodes"][0]["file"], valid_sequence)
    scene_states["episodes"][0]["initial_step"]["raw_action"][0] = float("nan")
    (tmp_path / "export/all/meta/scene_states.json").write_text(json.dumps(scene_states))
    corrupt_prelude = subprocess.run([sys.executable, str(validator), str(tmp_path / "export/all")],
                                    capture_output=True, text=True, timeout=30)
    assert corrupt_prelude.returncode != 0
    assert "Invalid initial-step raw action" in corrupt_prelude.stderr
