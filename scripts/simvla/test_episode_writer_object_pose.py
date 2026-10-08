"""The local LeRobot export must preserve the reset scene needed for replay."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab.utils.datasets import hdf5_dataset_file_handler as writer_module


def _episode():
    length = 2
    images = np.zeros((length, 2, 2, 3), dtype=np.uint8)
    return SimpleNamespace(
        is_empty=lambda: False,
        seed=3,
        env_id=0,
        success=True,
        data={
            "observations": {
                "images": {"front": images, "wrist_left": images, "wrist_right": images},
                "ee_6D_pos": np.zeros((length, 20)),
                "joint_angles": np.zeros((length, 14)),
                "base_vel": np.zeros((length, 3)),
                "mobile_base_world_frame": np.zeros((length, 6)),
                "subtask_index": np.zeros(length),
            },
            "actions": {
                "ee_6D_pos": np.zeros((length, 20)),
                "base": np.zeros((length, 3)),
                "joint_pos": np.zeros((length, 19)),
            },
        },
    )


def test_writer_keeps_initial_object_pose_in_stage_and_dataset(tmp_path, monkeypatch):
    writer = writer_module.TwoPhaseEpisodeWriter(str(tmp_path / "stage"), robot="anubis")
    pose = {"bowl0": [0.9, -0.3, 0.9, 1.0, 0.0, 0.0, 0.0]}
    writer.write_episode(_episode(), initial_objects=pose)
    ep_dir = tmp_path / "stage" / "000000"
    assert json.loads((ep_dir / "meta.json").read_text())["initial_objects"] == pose

    class FakeDataset:
        @classmethod
        def create(cls, **kwargs):
            (kwargs["root"] / "meta").mkdir(parents=True)
            return cls()

        def finalize(self):
            pass

    monkeypatch.setattr(writer_module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(writer, "_export_segments_to_dataset", lambda **kwargs: None)
    out = tmp_path / "dataset"
    writer._build_and_push_dataset(
        repo_id="local/test",
        out_root=out,
        fps=20,
        features={},
        task_data={"kitchen_num": 813, "kitchen_sub_num": 0, "kitchen_type": "l_shaped"},
        kmap={"l_shaped": 1},
        segments_by_episode={ep_dir: [(0, 1)]},
        episode_cache={ep_dir: {}},
        push=False,
        task_language="Put bowl inside drawer.",
    )
    sidecar = json.loads((out / "meta" / "initial_objects.json").read_text())
    assert sidecar == {"schema_version": 1, "episodes": [pose]}


def test_stage_reuse_never_overwrites_prior_episode(tmp_path):
    first = writer_module.TwoPhaseEpisodeWriter(str(tmp_path), robot="anubis")
    first.write_episode(_episode())
    path = tmp_path / "000000/arrays.npz"
    original = path.read_bytes()
    second = writer_module.TwoPhaseEpisodeWriter(str(tmp_path), robot="anubis")
    with pytest.raises(FileExistsError):
        second.write_episode(_episode())
    assert path.read_bytes() == original


def test_initial_step_requires_raw_action_and_genuine_reset_snapshot(tmp_path):
    episode = _episode()
    writer = writer_module.TwoPhaseEpisodeWriter(str(tmp_path), robot="anubis",
                                                record_initial_step=True)
    with pytest.raises(ValueError, match="reset state and raw actions"):
        writer.write_episode(episode)
    assert not list(tmp_path.iterdir())
    episode.data["initial_state"] = {"robot": {"position": np.zeros((1, 3))}}
    episode.data["raw_actions"] = np.zeros((2, 17))
    episode.data["raw_actions"][0, 0] = np.nan
    with pytest.raises(ValueError, match="initial-step raw"):
        writer.write_episode(episode)
    assert not list(tmp_path.iterdir())


def test_initial_step_keeps_discarded_action_without_changing_exported_rows(tmp_path):
    episode = _episode()
    episode.data["initial_state"] = {"robot": {"position": np.zeros((1, 3))}}
    episode.data["raw_actions"] = np.arange(34, dtype=float).reshape(2, 17)
    writer = writer_module.TwoPhaseEpisodeWriter(str(tmp_path), robot="anubis",
                                                record_initial_step=True)
    writer.write_episode(episode)
    meta = json.loads((tmp_path / "000000/meta.json").read_text())
    assert meta["num_samples"] == 1
    assert meta["initial_step"]["raw_action"] == list(range(17))
    assert meta["initial_step"]["initial_state_phase"] == "before_first_action"
    assert len(np.load(tmp_path / "000000/arrays.npz")["action__eef_pos"]) == 1


def test_finalize_never_deletes_existing_dataset(tmp_path):
    writer = writer_module.TwoPhaseEpisodeWriter(str(tmp_path / "stage"), robot="anubis")
    out = tmp_path / "dataset"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("previous dataset")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        writer._build_and_push_dataset(
            repo_id="local/test", out_root=out, fps=20, features={},
            task_data={}, kmap={}, segments_by_episode={}, episode_cache={},
            push=False, task_language="Put bowl inside drawer.")
    assert marker.read_text() == "previous dataset"


def test_writer_rejects_misaligned_streams_before_writing(tmp_path):
    episode = _episode()
    episode.data["observations"]["joint_angles"] = np.zeros((1, 14))
    writer = writer_module.TwoPhaseEpisodeWriter(str(tmp_path), robot="anubis")
    with pytest.raises(ValueError, match="unequal lengths"):
        writer.write_episode(episode)
    assert not list(tmp_path.iterdir())
