# Datasets

The [SimVLA_data Hugging Face collection](https://huggingface.co/collections/kyounginbaik/simvla-data)
currently contains four real-robot, single-task LeRobot v3 datasets. The values below come from each
repository's committed `meta/info.json` and `meta/tasks.parquet`, checked on 2026-09-30. Use the
listed commit when an exact snapshot matters; `main` can change.

| Dataset | Task label | Episodes | Frames / Parquet rows | Revision | License |
| --- | --- | ---: | ---: | --- | --- |
| [`kyounginbaik/real_task1`](https://huggingface.co/datasets/kyounginbaik/real_task1) | `Put Mug to Sink.` | 200 | 150,510 | [`8a81568`](https://huggingface.co/datasets/kyounginbaik/real_task1/tree/8a815683a867b654cf07556c72c476285c35042e) | **Not specified; no dataset card** |
| [`kyounginbaik/real_task2`](https://huggingface.co/datasets/kyounginbaik/real_task2) | `Put Bowl to Drawer.` | 200 | 189,629 | [`888f4ca`](https://huggingface.co/datasets/kyounginbaik/real_task2/tree/888f4ca68ee7a7240ebea0aff53099a7fe119a6e) | Apache-2.0 |
| [`kyounginbaik/real_task3`](https://huggingface.co/datasets/kyounginbaik/real_task3) | `Pour Water from Bottle to Mug` | 200 | 250,453 | [`41d0ddb`](https://huggingface.co/datasets/kyounginbaik/real_task3/tree/41d0ddb6422a8d3faa34d2706fdda80791b620a4) | Apache-2.0 |
| [`kyounginbaik/real_task4`](https://huggingface.co/datasets/kyounginbaik/real_task4) | `Pour Water from Bottle to Mug.` | 200 | 150,201 | [`cf2fa2b`](https://huggingface.co/datasets/kyounginbaik/real_task4/tree/cf2fa2b51b67885393bc059e22b1d602c9bbe3f6) | Apache-2.0 |

The task labels above preserve the published capitalization and punctuation. Each dataset declares
one task, a `train` split covering episodes `0:200`, and a top-level rate of 20 FPS. The metadata's
embedded video descriptors say 30 FPS, so consumers should inspect timestamps instead of assuming
the video descriptor is authoritative. `robot_type` is `null` in all four repositories, and the
cards do not identify the hardware. The cards for tasks 2–4 categorize the data as `robotics`, but
otherwise contain only the generated LeRobot structure and placeholder citation fields.

## Shared schema

All four `meta/info.json` files declare the same features:

| Feature | Type and shape | Contents described by the published names |
| --- | --- | --- |
| `qpos` | `float32[17]` | 14 arm/gripper joints and 3 base values |
| `observation.state` | `float32[23]` | Left and right end-effector pose/gripper state plus 3 base values |
| `action` | `float32[23]` | Left and right end-effector pose/gripper command plus 3 base velocity commands |
| `action_quat` | `float32[19]` | Quaternion form of the two end-effector commands plus grippers and base velocity |
| `observation.images.front` | video `[240, 320, 3]` | Front RGB stream |
| `observation.images.wrist_left` | video `[240, 320, 3]` | Left-wrist RGB stream |
| `observation.images.wrist_right` | video `[240, 320, 3]` | Right-wrist RGB stream |
| `timestamp` | `float32` | Frame timestamp |
| `frame_index`, `episode_index`, `index`, `task_index` | `int64` | Frame, episode, global-row, and task indices |

Hugging Face's viewer reports only 12 rows for `real_task1`: without a card/config it auto-detects
the twelve MP4 files as the dataset. Its committed LeRobot metadata and Parquet file instead declare
150,510 trajectory frames. The other three repositories have cards that map `data/*/*.parquet`, so
their viewer row counts agree with `total_frames`.

## Download and load

Download a reproducible snapshot, including Parquet, metadata, and video:

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    repo_id="kyounginbaik/real_task2",
    repo_type="dataset",
    revision="888f4ca68ee7a7240ebea0aff53099a7fe119a6e",
)
print(root)
```

For tasks 2–4, load the frame table without decoding video:

```python
from datasets import load_dataset

frames = load_dataset(
    "kyounginbaik/real_task2",
    split="train",
    revision="888f4ca68ee7a7240ebea0aff53099a7fe119a6e",
)
print(frames.column_names, len(frames))
```

Use a snapshot download or a LeRobot loader for `real_task1`; `datasets.load_dataset` currently
loads its auto-detected video listing rather than its trajectory Parquet table. Review the license
on the dataset page before redistribution: tasks 2–4 declare Apache-2.0, while task 1 publishes no
license metadata.
