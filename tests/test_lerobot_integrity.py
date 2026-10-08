"""Tiny real Parquet/video fixtures for collection integrity (optional data dependencies)."""
import importlib.util
import json
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
SPEC = importlib.util.spec_from_file_location(
    "collection_validator", Path(__file__).resolve().parents[1] /
    "scripts/tools/validate_lerobot_collection.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def video(path, count=6, fps=20, width=16):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, 16, "yuv420p"
        for _ in range(count):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, width, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def dataset(tmp_path):
    for directory in ("data/chunk-000", "meta/episodes/chunk-000", "videos/observation.images.front/chunk-000"):
        (tmp_path / directory).mkdir(parents=True)
    info = {"fps": 20, "total_frames": 6, "total_episodes": 2, "features": {
        "action": {"dtype": "float32", "shape": [2]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "observation.images.front": {"dtype": "video", "shape": [16, 16, 3]},
    }}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    rows = [{"index": i, "episode_index": i // 3, "frame_index": i % 3,
             "timestamp": (i % 3) / 20, "action": [0., 1.], "observation.state": [1., 2.]}
            for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "length": 3},
                                       {"episode_index": 1, "length": 3}]),
                   tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    video(tmp_path / "videos/observation.images.front/chunk-000/file-000.mp4")
    return tmp_path


def test_valid_export(dataset):
    assert validator.validate(dataset) == {"episodes": 2, "frames": 6, "video_keys": 1}


def test_action_gripper_channels_are_commands_not_motor_positions(dataset):
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [23]
    info_path.write_text(json.dumps(info))
    path = dataset / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["action"] = [0.] * 23
        row["action"][9], row["action"][19] = -1.6, .1
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert validator.validate(dataset)["frames"] == 6
    rows[0]["action"][9] = 1.8
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="binary gripper command"):
        validator.validate(dataset)


@pytest.mark.parametrize("field", ["action.joint", "joint_angles"])
def test_joint_streams_are_validated_when_declared(dataset, field):
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][field] = {"dtype": "float32", "shape": [2]}
    info_path.write_text(json.dumps(info))
    path = dataset / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row[field] = [0., 1.]
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert validator.validate(dataset)["frames"] == 6
    rows[1][field] = [float("nan"), 0.]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="invalid " + field):
        validator.validate(dataset)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("timestamp", 0.2, "timestamp/fps"),
    ("timestamp", float("nan"), "timestamp/fps"),
    ("frame_index", 9, "frame_index"),
    ("episode_index", 9, "episode_index"),
    ("index", 9, "global index"),
    ("action", [float("nan"), 1.], "invalid action"),
    ("observation.state", [1.], "invalid observation.state"),
])
def test_rejects_corrupt_row(dataset, field, value, message):
    path = dataset / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1][field] = value
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match=message):
        validator.validate(dataset)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"fps": 10}, "fps mismatch"), ({"count": 5}, "decoded 5 frames"),
    ({"width": 32}, "dimensions mismatch"),
])
def test_rejects_corrupt_video(dataset, kwargs, message):
    video(dataset / "videos/observation.images.front/chunk-000/file-000.mp4", **kwargs)
    with pytest.raises(ValueError, match=message):
        validator.validate(dataset)


def test_rejects_per_episode_length_mismatch_even_when_total_matches(dataset):
    path = dataset / "meta/episodes/chunk-000/file-000.parquet"
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "length": 2},
                                       {"episode_index": 1, "length": 4}]), path)
    with pytest.raises(ValueError, match="per-episode"):
        validator.validate(dataset)
