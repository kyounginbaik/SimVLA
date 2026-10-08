#!/usr/bin/env python3
"""Verify a finalized SimVLA LeRobot ``all`` dataset, including decoded videos.

Run with the simulator Python environment, which provides pyarrow, numpy, and PyAV.
This checks export integrity; it does not prove that the simulated task succeeded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(dataset: Path) -> dict[str, int]:
    try:
        import av
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("install pyarrow, numpy, and av in the validation environment") from exc

    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"missing finalized LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text())
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    if episodes < 1 or frames < 1:
        raise ValueError(f"empty LeRobot metadata: {info_path}")
    fps = float(info.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("metadata fps must be finite and positive")

    data_files = sorted((dataset / "data").rglob("*.parquet"))
    episode_files = sorted((dataset / "meta/episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError("missing data or episode parquet files")
    data_rows = sum(pq.read_metadata(path).num_rows for path in data_files)
    episode_rows = sum(pq.read_metadata(path).num_rows for path in episode_files)
    lengths = sum(
        sum(int(value) for value in pq.read_table(path, columns=["length"])["length"].to_pylist())
        for path in episode_files
    )
    if (data_rows, episode_rows, lengths) != (frames, episodes, frames):
        raise ValueError(
            f"count mismatch: info={episodes} episodes/{frames} frames, "
            f"parquet={episode_rows} episodes/{data_rows} rows, episode lengths={lengths}"
        )

    expected_sizes = {field: int(np.prod(info["features"][field]["shape"]))
                      for field in ("observation.state", "action")}
    # Joint-space consumers need the same guarantees as Cartesian consumers.
    # Older/public synthetic fixtures may legitimately omit these optional fields.
    for field in ("joint_angles", "action.joint"):
        if field in info["features"]:
            expected_sizes[field] = int(np.prod(info["features"][field]["shape"]))
    columns = [*expected_sizes, "index", "episode_index", "frame_index", "timestamp"]
    # Stream bounded batches: large collection shards can contain millions of rows.
    observed_lengths = {}
    global_index = 0
    for path in data_files:
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"missing data columns in {path}: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for row in batch.to_pylist():
                for field, expected_size in expected_sizes.items():
                    array = np.asarray(row[field])
                    if array.size != expected_size or not np.isfinite(array).all():
                        raise ValueError(f"invalid {field} at {path} global row {global_index}")
                if expected_sizes.get("action") == 23:
                    grips = np.asarray(row["action"])[[9, 19]]
                    if not np.all(np.isclose(grips, -1.6, atol=1e-5)
                                  | np.isclose(grips, .1, atol=1e-5)):
                        raise ValueError(f"invalid binary gripper command encoding at row {global_index}")
                episode = row["episode_index"]
                frame = row["frame_index"]
                if row["index"] != global_index:
                    raise ValueError(f"non-contiguous global index at row {global_index}")
                if not isinstance(episode, int) or not 0 <= episode < episodes:
                    raise ValueError(f"invalid episode_index at row {global_index}")
                expected_frame = observed_lengths.get(episode, 0)
                if frame != expected_frame:
                    raise ValueError(f"non-contiguous frame_index in episode {episode}")
                timestamp = float(row["timestamp"])
                if not math.isfinite(timestamp) or abs(timestamp - frame / fps) > 2e-4:
                    raise ValueError(f"timestamp/fps mismatch in episode {episode}, frame {frame}")
                observed_lengths[episode] = expected_frame + 1
                global_index += 1
    declared_lengths = {}
    for path in episode_files:
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            episode = row["episode_index"]
            if episode in declared_lengths:
                raise ValueError(f"duplicate episode metadata: {episode}")
            declared_lengths[episode] = row["length"]
    if observed_lengths != declared_lengths:
        raise ValueError("per-episode data counts do not match episode metadata")

    motor_entries = None
    substep_manifest = dataset / "meta/joint_substeps.json"
    if substep_manifest.is_file():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
        from replay_substeps import load_joint_substeps
        manifest = json.loads(substep_manifest.read_text())
        entries = manifest.get("episodes")
        motor_entries = entries
        if not isinstance(entries, list) or len(entries) != episodes:
            raise ValueError("Motor substep episode count does not match dataset")
        names = info["features"].get("action.joint", {}).get("names")
        if isinstance(names, dict):
            names = names.get("action.joint")
        if not isinstance(names, list):
            raise ValueError("Motor substeps require named action.joint metadata")
        for index, entry in enumerate(entries):
            if entry is None:  # mixed old/new staged episodes remain explicitly absent
                continue
            if (not isinstance(entry, dict) or type(entry.get("substeps")) is not int
                    or not 1 <= entry["substeps"] <= 24):
                raise ValueError("Invalid motor substep episode entry")
            load_joint_substeps(dataset, index, names, frames=declared_lengths[index],
                                decimation=entry["substeps"], control_fps=fps)

    scene_path = dataset / "meta/scene_states.json"
    if scene_path.is_file():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
        from replay_initial_step import load_initial_step
        states = json.loads(scene_path.read_text()).get("episodes")
        if not isinstance(states, list) or len(states) != episodes:
            raise ValueError("Scene state episode count does not match dataset")
        for index, state in enumerate(states):
            if not isinstance(state, dict):
                raise ValueError("Invalid scene state episode entry")
            if state.get("initial_step") is None:
                continue  # Legacy recordings and sliced segments have no prelude.
            names = info["features"].get("action.joint", {}).get("names")
            if isinstance(names, dict):
                names = names.get("action.joint")
            if not isinstance(names, list):
                raise ValueError("Initial-step replay requires named motor metadata")
            motor_entry = motor_entries[index] if motor_entries is not None else None
            decimation = motor_entry["substeps"] if motor_entry else 1
            load_initial_step(dataset, index, action_dim=17, joint_names=names,
                              decimation=decimation)

    video_keys = sorted(key for key, feature in info["features"].items()
                        if feature.get("dtype") == "video")
    if not video_keys:
        raise ValueError("dataset declares no video observations")
    for key in video_keys:
        paths = sorted((dataset / "videos" / key).rglob("*.mp4"))
        if not paths:
            raise ValueError(f"missing videos for {key}")
        decoded = 0
        expected_shape = info["features"][key]["shape"]
        if len(expected_shape) != 3 or expected_shape[2] != 3:
            raise ValueError(f"unsupported RGB video shape for {key}: {expected_shape}")
        for path in paths:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if stream.average_rate is None or abs(float(stream.average_rate) - fps) > 1e-4:
                    raise ValueError(f"video fps mismatch: {path}; expected {fps}")
                previous_time = None
                for frame in container.decode(video=0):
                    if [frame.height, frame.width] != expected_shape[:2]:
                        raise ValueError(f"video dimensions mismatch: {path}")
                    current_time = frame.time
                    if current_time is None or (previous_time is not None and
                            abs(current_time - previous_time - 1 / fps) > 2e-4):
                        raise ValueError(f"non-contiguous video timestamps: {path}")
                    previous_time = current_time
                    decoded += 1
        if decoded != frames:
            raise ValueError(f"{key} decoded {decoded} frames; metadata says {frames}")

    return {"episodes": episodes, "frames": frames, "video_keys": len(video_keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="path to finalized LeRobot all/ directory")
    args = parser.parse_args()
    result = validate(args.dataset)
    print(f"verified {args.dataset}: {result['episodes']} episodes, "
          f"{result['frames']} data rows, {result['video_keys']} synchronized videos")


if __name__ == "__main__":
    main()
