# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import h5py
import numpy as np
import os
import torch
from collections.abc import Iterable
from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData
from .robot_schemas import get_robot_schema

from pathlib import Path
import json, shutil
from typing import Any, Dict, Sequence, Optional

# Collection/staging must not require the offline LeRobot/Rerun dependency graph.
# Resolve the exporter only when finalization is explicitly requested.
LeRobotDataset = None
#


import shutil
import torch
from typing import Any, Dict, Sequence, Optional, List, Tuple, Iterable


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _stack_time_series(leaf):
    """
    Accepts:
      - list[torch.Tensor or np.ndarray] with leading T as list length
      - torch.Tensor/np.ndarray with leading T (T,...) already
    Returns:
      np.ndarray with shape (T, ...) for numeric arrays.
    """
    if isinstance(leaf, list):
        arrs = []
        for v in leaf:
            if isinstance(v, torch.Tensor):
                arrs.append(v.detach().cpu().numpy())
            else:
                arrs.append(np.asarray(v))
        return np.stack(arrs, axis=0)
    if isinstance(leaf, torch.Tensor):
        return leaf.detach().cpu().numpy()
    return np.asarray(leaf)

def _stack_frames(leaf):
    """
    Stack frames EXACTLY as-is.
                    "joint_angles" : joint_angles[t],
    - No dtype conversion (no uint8 checks)
    - No CHW<->HWC conversion
    """
    if isinstance(leaf, list):
        frames = []
        for f in leaf:
            if isinstance(f, torch.Tensor):
                frames.append(f.detach().cpu().numpy())
            else:
                frames.append(np.asarray(f))
        if len(frames) == 0:
            return np.zeros((0, 240, 320, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)

    if isinstance(leaf, torch.Tensor):
        return leaf.detach().cpu().numpy()

    return np.asarray(leaf)

def _get(d, path_tuple):
    cur = d
    for k in path_tuple:
        cur = cur[k]
    return cur


PATHS = {
    "img_front":      ("observations","images","front"),
    "img_wl":         ("observations","images","wrist_left"),
    "img_wr":         ("observations","images","wrist_right"),
    "eef_pose":       ("observations","ee_6D_pos"),
    "joint_angles":   ("observations","joint_angles"),
    "base_vel":      ("observations","base_vel"),
    "mobile_base6":   ("observations","mobile_base_world_frame"),
    "subtask_index":  ("observations","subtask_index"),
    "language":       ("observations","language"),
    "act_eef":        ("actions","ee_6D_pos"),
    "act_base":       ("actions","base"),
    "act_joint":      ("actions","joint_pos"),
}


# -------- segmentation helpers --------
def _mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert boolean mask (T,) into list of contiguous (start, end) segments,
    where end is exclusive.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got {mask.shape}")
    T = mask.shape[0]
    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for t in range(T):
        if mask[t] and not in_seg:
            in_seg = True
            seg_start = t
        if in_seg and (not mask[t] or t == T - 1):
            seg_end = t if not mask[t] else t + 1
            segments.append((seg_start, seg_end))
            in_seg = False
    return segments

def _segments_for_value(sub_idx: np.ndarray, value: int) -> List[Tuple[int, int]]:
    return _mask_to_segments(sub_idx == value)

def _segments_for_set(sub_idx: np.ndarray, values: Iterable[int]) -> List[Tuple[int, int]]:
    values = list(values)
    return _mask_to_segments(np.isin(sub_idx, values))


def _assign_short_runs_to_previous(sub_idx: np.ndarray, min_len: int) -> np.ndarray:
    """
    Build an 'assigned' array used ONLY for export bucketing.

    Rule:
      - If a contiguous run is shorter than min_len,
        assign its frames to the PREVIOUS run's bucket label.
      - If it's the first run and too short, assign to NEXT run's label.
      - We DO NOT change the original sub_idx stored in frames.
    """
    sub_idx = np.asarray(sub_idx, dtype=np.int64)
    T = sub_idx.shape[0]
    if T == 0 or min_len <= 1:
        return sub_idx.copy()

    # Run-length encoding: runs = [(value, start, end_excl), ...]
    runs: List[Tuple[int, int, int]] = []
    cur = int(sub_idx[0])
    start = 0
    for t in range(1, T + 1):
        if t == T or int(sub_idx[t]) != cur:
            runs.append((cur, start, t))
            if t < T:
                cur = int(sub_idx[t])
                start = t

    # Decide export bucket label per run
    buckets: List[int] = []
    for i, (val, s, e) in enumerate(runs):
        length = e - s
        if length >= min_len:
            buckets.append(val)
        else:
            if i > 0:
                # absorb into previous bucket (could already be absorbed)
                buckets.append(buckets[i - 1])
            elif len(runs) > 1:
                # first run too short -> absorb forward into next run's raw label
                buckets.append(runs[i + 1][0])
            else:
                # only one run total
                buckets.append(val)

    assigned = np.empty_like(sub_idx)
    for (val, s, e), b in zip(runs, buckets):
        assigned[s:e] = b
    return assigned

import json
import shutil
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple

import numpy as np


class TwoPhaseEpisodeWriter:
    """
    Call write_episode(ep) whenever an env finishes.
    After all envs: finalize_lerobot(...).

    Phase 1 (write_episode):
      - Dump raw npy/npz for each episode (fast).

    Phase 2 (finalize_lerobot):
      - Read all episodes once.
      - Precompute obs_state/action per episode.
      - Reuse cached arrays/images for:
          * full dataset
          * per-subtask datasets
          * grouped-subtask datasets
      - Use LeRobot's async image writer (multi-process).
    """

    def __init__(
        self,
        staging_dir: str,
        *,
        robot: str,
        video_keys: Optional[Sequence[str]] = None,
        record_initial_step: bool = False,
    ):
        # Validate robot eagerly so misconfiguration fails at writer
        # construction, not 6 minutes later at finalize.
        get_robot_schema(robot)
        self.robot = robot
        self.record_initial_step = record_initial_step
        self.stage = Path(staging_dir)
        self.stage.mkdir(parents=True, exist_ok=True)
        self._demo_count = 0
        self._total_steps = 0
        self.video_keys = (
            list(video_keys)
            if video_keys
            else [
                "observation.images.front",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            ]
        )

    # --------------------------------------------------------
    # Phase 1: per-episode dumping (unchanged logic, fast)
    # --------------------------------------------------------
    def write_episode(self, episode: "EpisodeData", *, initial_objects: Optional[Dict[str, List[float]]] = None):
        if episode is None or episode.is_empty():
            return

        d = episode.data

        # images -> stacked as-is (no dtype/layout fixes)
        img_front = _stack_frames(_get(d, PATHS["img_front"]))
        img_wl    = _stack_frames(_get(d, PATHS["img_wl"]))
        img_wr    = _stack_frames(_get(d, PATHS["img_wr"]))

        # arrays -> (T, ...)
        eef_pose     = _stack_time_series(_get(d, PATHS["eef_pose"])).astype(np.float32, copy=False)
        joint_angles = _stack_time_series(_get(d, PATHS["joint_angles"])).astype(np.float32, copy=False)
        base_vel     = _stack_time_series(_get(d, PATHS["base_vel"])).astype(np.float32, copy=False)
        mobile6      = _stack_time_series(_get(d, PATHS["mobile_base6"])).astype(np.float32, copy=False)
        sub_idx      = _stack_time_series(_get(d, PATHS["subtask_index"])).astype(np.int64,   copy=False)
        act_eef      = _stack_time_series(_get(d, PATHS["act_eef"])).astype(np.float32, copy=False)
        act_base     = _stack_time_series(_get(d, PATHS["act_base"])).astype(np.float32, copy=False)
        act_joint     = _stack_time_series(_get(d, PATHS["act_joint"])).astype(np.float32, copy=False)


        start = 1  # <-- skip index 0 everywhere
        Ts = [
            x.shape[0] - start
            for x in (
                img_front, img_wl, img_wr,
                eef_pose, base_vel, joint_angles, mobile6,
                act_eef, act_base, act_joint,
                sub_idx,
            )
        ]
        if len(set(Ts)) > 1:
            raise ValueError(f"Episode streams have unequal lengths after initial-frame removal: {Ts}")
        T = int(min(Ts)) if Ts else 0
        if T <= 0:
            return

        initial_step = None
        if self.record_initial_step:
            # Export removes the first observation/action pair, whose reset
            # camera buffers can be stale. Preserve its physical action so replay
            # can reach the state preceding exported frame zero without teleporting.
            # Only current collectors with a genuine pre-action snapshot enable this.
            if "initial_state" not in d or "raw_actions" not in d:
                raise ValueError("Initial-step recording requires reset state and raw actions")
            raw = _stack_time_series(d["raw_actions"])
            if (raw.ndim != 2 or len(raw) != len(act_joint)
                    or not np.isfinite(raw).all() or not np.isfinite(act_joint[0]).all()):
                raise ValueError("Invalid initial-step raw actions or motor targets")
            initial_step = {
                "schema_version": 1,
                "initial_state_phase": "before_first_action",
                "raw_action": raw[0].tolist(),
                "subtask_index": int(sub_idx[0]),
                "motor_targets": act_joint[0].tolist(),
            }

        s = start
        e = start + T

        ep_id = self._demo_count
        ep_dir = self.stage / f"{ep_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=False)

        if "joint_substeps" in d["actions"]:
            substeps = _stack_time_series(d["actions"]["joint_substeps"]).astype(np.float32)
            if (substeps.ndim != 3 or substeps.shape[0] != act_joint.shape[0]
                    or substeps.shape[2] != act_joint.shape[1] or substeps.shape[1] < 1
                    or not np.isfinite(substeps).all()):
                raise ValueError("Invalid physics-substep motor recording")
            np.save(ep_dir / "joint-substeps.npy", substeps[s:e], allow_pickle=False)
            if initial_step is not None:
                initial_step["motor_substeps"] = substeps[0].tolist()

        # ---- save raw video stacks (skip first frame) ----
        np.save(ep_dir / "observation.images.front.npy", img_front[s:e])
        np.save(ep_dir / "observation.images.wrist_left.npy", img_wl[s:e])
        np.save(ep_dir / "observation.images.wrist_right.npy", img_wr[s:e])

        # ---- save non-video arrays (skip first step) ----
        np.savez_compressed(
            ep_dir / "arrays.npz",
            obs__eef_pose=eef_pose[s:e],
            joint_angles=joint_angles[s:e],
            obs__base_pose=base_vel[s:e],
            # if you actually want only the very first pose, use mobile6[0]
            initial_pose=mobile6[s:e],      # or mobile6[0] depending on your design
            subtask_index=sub_idx[s:e],
            action__eef_pos=act_eef[s:e],
            action__base_pos=act_base[s:e],
            action__joint=act_joint[s:e],
        )

        meta = {
            "episode_id": ep_id,
            "num_samples": T,     # now equals len of data after skipping index 0
            "seed": episode.seed,
            "env_id": episode.env_id,
            "success": bool(episode.success) if episode.success is not None else None,
        }
        if initial_objects is not None:
            for name, pose in initial_objects.items():
                if len(pose) != 7 or not np.isfinite(pose).all():
                    raise ValueError(f"initial object pose for {name!r} must be seven finite numbers")
            meta["initial_objects"] = initial_objects
        def scene_snapshot(value):
            if isinstance(value, dict):
                return {key: scene_snapshot(item) for key, item in value.items()}
            array = _stack_time_series(value)
            if len(array) != 1 or not np.isfinite(array).all():
                raise ValueError("episode scene snapshots must contain one finite state")
            return array[0].tolist()

        for key in ("initial_state", "terminal_state"):
            if key in d:
                meta[key] = scene_snapshot(d[key])
        if initial_step is not None:
            meta["initial_step"] = initial_step
        (ep_dir / "meta.json").write_text(json.dumps(meta))

        self._demo_count += 1
        self._total_steps += T

    # --------------------------------------------------------
    # Internal helpers for Phase 2 (fast export)
    # --------------------------------------------------------
    def _export_segments_to_dataset(
        self,
        ds: LeRobotDataset,
        cache: Dict[str, Any],
        start: int,
        end: int,
        kitchen_num_arr: np.ndarray,
        kitchen_sub_num_arr: np.ndarray,
        kitchen_type_arr: np.ndarray,
        task_language: str,
        full_task: bool,
    ):
        """
        Add frames [start:end] from a single episode cache to `ds` as
        one LeRobot episode. `end` is exclusive.
        """
        arr          = cache["arr"]
        cams         = cache["cams"]
        eef          = cache["obs_state"]
        obs_state    = np.concatenate([eef[..., 0:9], eef[..., 18:19], eef[..., 9:18], eef[..., 19:23]], axis=-1)
        joint_angles = cache["joint_angles"]
        act_full     = cache["act"]
        act_joint     = cache["act_joint"]
        seg_len      = end - start

        if seg_len <= 0:
            return

        # Sanity guard: verify the cached arrays match the schema declared in
        # ds.features. Lerobot's validate_frame doesn't surface the writer's
        # robot identity, which makes shape mismatches hard to diagnose;
        # raise here with self.robot in the message before any add_frame call.
        expected_act_joint = ds.features["action.joint"]["shape"][0]
        expected_joint_ang = ds.features["joint_angles"]["shape"][0]
        actual_act_joint = act_joint.shape[-1]
        actual_joint_ang = joint_angles.shape[-1]
        if actual_act_joint != expected_act_joint or actual_joint_ang != expected_joint_ang:
            raise ValueError(
                f"Robot schema mismatch (writer.robot={self.robot!r}): "
                f"action.joint actual={actual_act_joint} expected={expected_act_joint}, "
                f"joint_angles actual={actual_joint_ang} expected={expected_joint_ang}. "
                f"Check robot_schemas.py and the staged arrays.npz."
            )

        for i, t in enumerate(range(start, end)):
            is_first = 1 if i == 0 else 0
            is_last  = 1 if i == seg_len - 1 else 0
            
            if full_task:
                sample = {
                    "kitchen_num":      kitchen_num_arr,
                    "kitchen_sub_num":  kitchen_sub_num_arr,
                    "kitchen_type":     kitchen_type_arr,
                    "initial_pose":     arr["initial_pose"][t],
                    "is_first":         np.array([is_first], dtype=np.int64),
                    "is_last":          np.array([is_last],  dtype=np.int64),
                    "task":             task_language,
                    "subtask_index":    np.array([arr["subtask_index"][t]], dtype=np.int64),
                    "observation.state": obs_state[t],
                    "joint_angles" : joint_angles[t],
                    "action":            act_full[t],
                    "action.joint":            act_joint[t],
                }
            else:
                sample = {
                    "kitchen_num":      kitchen_num_arr,
                    "kitchen_sub_num":  kitchen_sub_num_arr,
                    "kitchen_type":     kitchen_type_arr,
                    "initial_pose":     arr["initial_pose"][t],
                    "is_first":         np.array([is_first], dtype=np.int64),
                    "is_last":          np.array([is_last],  dtype=np.int64),
                    "task":             str(arr["subtask_language"][t]),
                    "subtask_index":    np.array([arr["subtask_index"][t]], dtype=np.int64),
                    "observation.state": obs_state[t],
                    "joint_angles" : joint_angles[t],
                    "action":            act_full[t],
                    "action.joint":            act_joint[t],
                }

            # images: THWC uint8 (we rely on them already being correct)
            for cam_key, stack in cams.items():
                sample[cam_key] = stack[t]

            ds.add_frame(sample)

        ds.save_episode()

    def _build_and_push_dataset(
            self,
            repo_id: str,
            out_root: Path,
            fps: int,
            features: Dict[str, Any],
            task_data: Dict[str, Any],
            kmap: Dict[str, int],
            segments_by_episode: Dict[Path, List[Tuple[int, int]]],
            episode_cache: Dict[Path, Dict[str, Any]],
            push: bool = True,
            image_writer_processes: int = 8,
            image_writer_threads: int = 4,
            # NO MUG DEFAULT. finalize_lerobot resolves the language from the goal file and every
            # caller here passes it now; the old default is what let the SUBTASK and GROUP exports
            # below -- which omitted the argument -- label their datasets "Put Mug to Sink" while
            # the "all" dataset beside them was labelled correctly. None raises here rather than
            # writing a wrong string into meta/tasks.parquet an hour later.
            task_language: Optional[str] = None,
            full_task: bool = True,
            use_videos: bool = True,
        ):
        """
        Generic builder: uses precomputed episode_cache to fill a LeRobotDataset.
        """
        if not task_language:
            raise ValueError(
                "task_language is required: it becomes the instruction every frame of this dataset "
                "is trained against. finalize_lerobot reads it from the goal file; a builder called "
                "without one has no way to know what the episodes show."
            )
        if out_root.exists():
            raise FileExistsError(f"Refusing to overwrite existing LeRobot dataset: {out_root}")

        global LeRobotDataset
        if LeRobotDataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset.create(
            repo_id=repo_id,
            root=out_root,
            fps=fps,
            use_videos=use_videos,
            features=features,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )

        # Constant arrays per dataset
        kitchen_num_arr     = np.array([task_data["kitchen_num"]],        dtype=np.int64)
        kitchen_sub_num_arr = np.array([int(task_data["kitchen_sub_num"])], dtype=np.int64)
        kitchen_type_arr    = np.array([kmap[task_data["kitchen_type"]]], dtype=np.int64)

        for ep_dir, segs in segments_by_episode.items():
            cache = episode_cache[ep_dir]
            for (start, end) in segs:
                if end <= start:
                    continue
                self._export_segments_to_dataset(
                    ds=ds,
                    cache=cache,
                    start=start,
                    end=end,
                    kitchen_num_arr=kitchen_num_arr,
                    kitchen_sub_num_arr=kitchen_sub_num_arr,
                    kitchen_type_arr=kitchen_type_arr,
                    task_language=task_language,
                    full_task=full_task
                )

        # finalize() will wait on image writer, encode videos, and write metadata
        ds.finalize()
        # The trajectory alone cannot replay a randomized object: its initial pose is not an
        # action or robot observation. Keep an episode-indexed sidecar next to LeRobot metadata.
        # Iteration order is the exact order in which add_frame built the output episodes.
        initial_poses = []
        scene_states = []
        motor_substeps = []
        for ep_dir, segs in segments_by_episode.items():
            episode_meta = json.loads((ep_dir / "meta.json").read_text())
            objects = episode_meta.get("initial_objects")
            for start, end in segs:
                if end > start:
                    substep_path = ep_dir / "joint-substeps.npy"
                    substep_entry = None
                    if substep_path.is_file():
                        sequence = np.load(substep_path, allow_pickle=False, mmap_mode="r")[start:end]
                        if (sequence.ndim != 3 or len(sequence) != end - start
                                or sequence.shape[1] < 1
                                or sequence.shape[2] != len(get_robot_schema(self.robot)["action_joint"])
                                or not np.isfinite(sequence).all()):
                            raise ValueError("Invalid staged physics-substep motor recording")
                        relative = f"joint_substeps/episode_{len(motor_substeps):06d}.npy"
                        destination = out_root / "meta" / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        np.save(destination, sequence, allow_pickle=False)
                        substep_entry = {"file": relative, "frames": len(sequence),
                                         "substeps": sequence.shape[1]}
                    motor_substeps.append(substep_entry)
                    initial_poses.append(objects)
                    # A sliced skill segment does not start at the scene reset.
                    # Never label the full episode's terminal state as its own.
                    scene_states.append({
                        "initial": episode_meta.get("initial_state") if start == 0 else None,
                        "initial_step": episode_meta.get("initial_step") if start == 0 else None,
                        "terminal": episode_meta.get("terminal_state")
                                    if end == episode_meta["num_samples"] else None,
                    })
        if any(objects is not None for objects in initial_poses):
            (out_root / "meta" / "initial_objects.json").write_text(json.dumps({
                "schema_version": 1,
                "episodes": initial_poses,
            }))
        if any(entry is not None for entry in motor_substeps):
            (out_root / "meta" / "joint_substeps.json").write_text(json.dumps({
                "schema_version": 1, "control_fps": fps,
                "joint_names": get_robot_schema(self.robot)["action_joint"],
                "episodes": motor_substeps,
            }, allow_nan=False))
        if any(state["initial"] is not None or state["terminal"] is not None for state in scene_states):
            (out_root / "meta" / "scene_states.json").write_text(json.dumps({
                "schema_version": 1, "frame": "environment", "episodes": scene_states,
            }, allow_nan=False))
        if push:
            ds.push_to_hub()

        print(f"Built LeRobot dataset at: {out_root}")
        if push:
            print(f"Uploaded to: {repo_id}")
        else:
            print(f"Saved locally (upload disabled): {repo_id}")

    # --------------------------------------------------------
    # Phase 2: finalize_lerobot (now heavily optimized)
    # --------------------------------------------------------

    def finalize_lerobot(
            self,
            output_path: str,
            task_json: str,
            goal_json: Optional[str] = None,
            fps: int = 20,
            # knobs:
            export_all: bool = True,
            export_subtasks: bool = False,
            export_groups: Optional[Sequence[Sequence[int]]] = None,
            min_segment_len: int = 2,
            repo_prefix: Optional[str] = None,
            push: bool = True,
            # parallelism knobs:
            image_writer_processes: int = 8,
            image_writer_threads: int = 4,
            # storage mode:
            use_videos: bool = True,
            task_language: Optional[str] = None,
        ):
        """
        Phase-2 export: build one or more LeRobot datasets from staged npy/npz.
        If use_videos=False, images are embedded directly (no mp4 encoding).

        `task_language` is the string that lands in the dataset's meta/tasks.parquet and becomes
        the instruction a policy is conditioned on. None means TAKE IT FROM THE GOAL FILE, which
        is what `task_json` already is.

        IT USED TO DEFAULT TO "Put Mug to Sink. The Mug is filled with water so move slow." and
        offline_finalize_lerobot.py never passed anything else, so EVERY dataset this path has
        produced carries that string -- including the 32-episode push-chair set at
        campaign_out/pushchair/deliverable/Isaac-Kitchen-v1218-00, whose meta says
        "Put Mug to Sink" over recordings of a robot pushing a chair. Nothing errors, the dataset
        loads, and the label is only wrong where it matters: at training time.

        A MISSING LANGUAGE IS NOW AN ERROR, not a fallback. Falling back to a plausible default is
        exactly the mechanism that produced the mislabelled corpus -- the caller cannot tell the
        difference between "it read my task" and "it used the mug".
        """

        repo_prefix = repo_prefix or os.environ.get("HF_USER")
        if not repo_prefix:
            if push:
                raise ValueError("Set repo_prefix or HF_USER when uploading a dataset")
            repo_prefix = "local"

        out  = Path(output_path)
        task = output_path.split("/")[-1]

        with open(task_json, "r") as f:
            task_data = json.load(f)

        if task_language is None:
            task_language = ((task_data.get("run_config") or {}).get("task_language")
                             or task_data.get("language"))
        if not task_language:
            raise ValueError(
                f"no task language: {task_json} carries neither run_config.task_language nor a "
                f"top-level 'language', and none was passed. That string becomes the instruction "
                f"the policy is conditioned on, so guessing it silently mislabels every episode "
                f"in the dataset."
            )
        if goal_json is not None:
            # Parsed and discarded -- `_` has never had a consumer in this method. Kept as an
            # opt-in validity check rather than deleted, but no longer REQUIRED: the twin is a
            # GUI-authoring artifact (goal_generator.list_reloadable_templates), and task_emit
            # -- which writes the corpora that actually get driven -- emits only `<task>.json`.
            # Demanding it here made every task_emit corpus unexportable, and the failure landed
            # after the whole collection run, with every episode already staged.
            with open(goal_json, "r") as f:
                _ = json.load(f)

        features = _make_feature_schema(fps, self.robot, use_videos=use_videos)
        # Preserve the historical research exporter's encoding. A layout missing here
        # is a KeyError at the end of a
        # collection run, after every episode is already staged. New layouts take the next
        # free index; 0-4 are what the existing corpus was written with.
        kmap = {
            "island":      0,
            "l_shaped":    1,
            "peninsula":   2,
            "u_shaped":    3,
            "single_wall": 4,
            "galley":      5,
        }

        ep_dirs = sorted([p for p in self.stage.iterdir() if p.is_dir()])

        # ---- Preload and cache per-episode data (single pass) ----
        raw_sub_idx_cache:      Dict[Path, np.ndarray] = {}
        assigned_sub_idx_cache: Dict[Path, np.ndarray] = {}
        episode_lengths:        Dict[Path, int]         = {}
        episode_cache:          Dict[Path, Dict[str, Any]] = {}

        for ep_dir in ep_dirs:
            meta = json.loads((ep_dir / "meta.json").read_text())
            T    = int(meta["num_samples"])

            # Load arrays once
            arr = np.load(ep_dir / "arrays.npz")

            raw_sub_idx      = arr["subtask_index"][:T].astype(np.int64, copy=False)
            assigned_sub_idx = _assign_short_runs_to_previous(raw_sub_idx, min_segment_len)

            raw_sub_idx_cache[ep_dir]      = raw_sub_idx
            assigned_sub_idx_cache[ep_dir] = assigned_sub_idx
            episode_lengths[ep_dir]        = T

            # Precompute obs_state and action once per episode
            obs_state = np.concatenate(
                [
                    arr["obs__eef_pose"][:T],
                    arr["obs__base_pose"][:T],
                ],
                axis=-1,
            ).astype(np.float32, copy=False)
          
            joint_angles = arr["joint_angles"][:T]

            act_full = np.concatenate(
                [
                    arr["action__eef_pos"][:T],
                    arr["action__base_pos"][:T],
                ],
                axis=-1,
            ).astype(np.float32, copy=False)
            act_joint = np.concatenate(
                [
                    arr["action__joint"][:T],
                ],
                axis=-1,
            ).astype(np.float32, copy=False)
            # Memory-map images once per episode
            cams = {
                "observation.images.front": np.load(
                    ep_dir / "observation.images.front.npy", mmap_mode="r"
                ),
                "observation.images.wrist_left": np.load(
                    ep_dir / "observation.images.wrist_left.npy", mmap_mode="r"
                ),
                "observation.images.wrist_right": np.load(
                    ep_dir / "observation.images.wrist_right.npy", mmap_mode="r"
                ),
            }

            episode_cache[ep_dir] = {
                "arr":       arr,
                "obs_state": obs_state,
                "joint_angles": joint_angles,
                "act":       act_full,
                "act_joint":       act_joint,
                "cams":      cams,
                "T":         T,
            }

        # ---------------------------------------------------
        # 1) Export ALL (full episodes)
        # ---------------------------------------------------
        if export_all:
            all_repo_id  = f"{repo_prefix}/{task}"
            all_out_root = out / "all"
            segments_all: Dict[Path, List[Tuple[int, int]]] = {
                ep_dir: [(0, episode_lengths[ep_dir])]
                for ep_dir in ep_dirs
            }
            self._build_and_push_dataset(
                repo_id=all_repo_id,
                out_root=all_out_root,
                fps=fps,
                features=features,
                task_data=task_data,
                kmap=kmap,
                segments_by_episode=segments_all,
                episode_cache=episode_cache,
                push=push,
                image_writer_processes=image_writer_processes,
                image_writer_threads=image_writer_threads,
                task_language=task_language,
                full_task=True
            )

        # ---------------------------------------------------
        # 2) Export per-subtask datasets (using assigned buckets)
        # ---------------------------------------------------
        if export_subtasks:
            unique_assigned = sorted(
                set(int(x) for a in assigned_sub_idx_cache.values() for x in np.unique(a))
            )

            for k in unique_assigned:
                segments_k: Dict[Path, List[Tuple[int, int]]] = {}

                for ep_dir, assigned in assigned_sub_idx_cache.items():
                    segs = _segments_for_value(assigned, k)
                    if segs:
                        segments_k[ep_dir] = segs

                if not segments_k:
                    continue

                repo_id_k  = f"{repo_prefix}/{task}_sub{k}"
                out_root_k = out / f"sub{k}"
                self._build_and_push_dataset(
                    repo_id=repo_id_k,
                    out_root=out_root_k,
                    fps=fps,
                    features=features,
                    task_data=task_data,
                    kmap=kmap,
                    segments_by_episode=segments_k,
                    episode_cache=episode_cache,
                    push=push,
                    image_writer_processes=image_writer_processes,
                    image_writer_threads=image_writer_threads,
                    full_task = False,
                    task_language=task_language,
                )

        # ---------------------------------------------------
        # 3) Export grouped/pair subtasks datasets (assigned buckets)
        # ---------------------------------------------------
        if export_groups:
            for group in export_groups:
                group      = [int(x) for x in group]
                group_name = "-".join(str(x) for x in group)

                segments_g: Dict[Path, List[Tuple[int, int]]] = {}

                for ep_dir, assigned in assigned_sub_idx_cache.items():
                    segs = _segments_for_set(assigned, group)
                    if segs:
                        segments_g[ep_dir] = segs

                if not segments_g:
                    continue

                repo_id_g  = f"{repo_prefix}/{task}_sub{group_name}"
                out_root_g = out / f"sub{group_name}"
                self._build_and_push_dataset(
                    repo_id=repo_id_g,
                    out_root=out_root_g,
                    fps=fps,
                    features=features,
                    task_data=task_data,
                    kmap=kmap,
                    segments_by_episode=segments_g,
                    episode_cache=episode_cache,
                    push=push,
                    image_writer_processes=image_writer_processes,
                    image_writer_threads=image_writer_threads,
                    task_language=task_language,
                )

        # ---------------------------------------------------
        # Cleanup staging dirs
        # ---------------------------------------------------
#        for p in sorted(self.stage.iterdir()):
#            if p.is_dir() and p.name.isdigit() and len(p.name) == 6:
#                shutil.rmtree(p)


# Your _make_feature_schema stays the same as you posted,
# unless you also want to tweak dims/names.

def _make_feature_schema(fps: int, robot: str, use_videos: bool = True) -> Dict[str, Any]:
    state_names = [
        "l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6",
        "r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6",
        "l_gripper","r_gripper",
        "v_x","v_y","omega",
    ]
    action_names = [
        "l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6","l_gripper",
        "r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6","r_gripper",
        "v_x","v_y","omega",
    ]

    schema = get_robot_schema(robot)
    joint_names = schema["joint_angles"]
    action_joint_names = schema["action_joint"]

    img_dtype = "video" if use_videos else "image"

    return {
        "kitchen_num":      {"dtype": "int64",   "shape": (1,),  "names": None},
        "kitchen_sub_num":  {"dtype": "int64",   "shape": (1,),  "names": None},
        "kitchen_type":     {"dtype": "int64",   "shape": (1,),  "names": None},
        "initial_pose": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"pose": ["x", "y", "qw", "qx", "qy", "qz"]},
        },
        "is_first":         {"dtype": "int64",   "shape": (1,),  "names": None},
        "is_last":          {"dtype": "int64",   "shape": (1,),  "names": None},
        "subtask_index":    {"dtype": "int64",   "shape": (1,),  "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"state": state_names},
        },
        "observation.images.front": {
            "dtype": img_dtype,
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": 240,
                "video.width": 320,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.images.wrist_left": {
            "dtype": img_dtype,
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": 240,
                "video.width": 320,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.images.wrist_right": {
            "dtype": img_dtype,
            "shape": [240, 320, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": 240,
                "video.width": 320,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "joint_angles":{
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"joint_names": joint_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"action": action_names},
        },
        "action.joint": {
            "dtype": "float32",
            "shape": (len(action_joint_names),),
            "names": {"action.joint": action_joint_names},
        },
    }



#def _make_feature_schema(fps: int) -> Dict[str, Any]:
#   state_names = [
#       "l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6",
#       "r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6",
#       "l_gripper","r_gripper",
##      'arm1_base_link_joint', 'arm2_base_link_joint', 'link11_joint', 'link21_joint',
##  'link12_joint', 'link22_joint', 'link13_joint', 'link23_joint',
##      'link14_joint', 'link24_joint', 'link15_joint', 'link25_joint',
#       "v_x","v_y","omega",
#   ]
#   action_names = [
#       "l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6",
#       "r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6",
#       "l_gripper","r_gripper",
#       "v_x","v_y","omega",
#   ]
#
#   return {
#       "kitchen_num":      {"dtype": "int64",   "shape": (1,),  "names": None},
#       "kitchen_sub_num":  {"dtype": "int64",   "shape": (1,),  "names": None},
#       "kitchen_type":     {"dtype": "int64",   "shape": (1,),  "names": None},
#       "initial_pose": {
#           "dtype": "float32",
#           "shape": (6,),
#           "names": {"pose": ["x", "y", "qw", "qx", "qy", "qz"]},
#       },
#       "is_first":         {"dtype": "int64",   "shape": (1,),  "names": None},
#       "is_last":          {"dtype": "int64",   "shape": (1,),  "names": None},
#       "subtask_index":    {"dtype": "int64",   "shape": (1,),  "names": None},
#       "observation.state": {
#           "dtype": "float32",
#           "shape": (len(state_names),),
#           "names": {"state": state_names},
#       },
#       "observation.images.front": {
#           "dtype": "video",
#           "shape": [224, 224, 3],
#           "names": ["height", "width", "channels"],
#           "info": {
#               "video.fps": fps,
#               "video.height": 224,
#               "video.width": 224,
#               "video.channels": 3,
#               "video.codec": "h264",
#               "video.pix_fmt": "yuv420p",
#               "video.is_depth_map": False,
#               "has_audio": False,
#           },
#       },
#       "observation.images.wrist_left": {
#           "dtype": "video",
#           "shape": [224, 224, 3],
#           "names": ["height", "width", "channels"],
#           "info": {
#               "video.fps": fps,
#               "video.height": 224,
#               "video.width": 224,
#               "video.channels": 3,
#               "video.codec": "h264",
#               "video.pix_fmt": "yuv420p",
#               "video.is_depth_map": False,
#               "has_audio": False,
#           },
#       },
#       "observation.images.wrist_right": {
#           "dtype": "video",
#           "shape": [224, 224, 3],
#           "names": ["height", "width", "channels"],
#           "info": {
#               "video.fps": fps,
#               "video.height": 224,
#               "video.width": 224,
#               "video.channels": 3,
#               "video.codec": "h264",
#               "video.pix_fmt": "yuv420p",
#               "video.is_depth_map": False,
#               "has_audio": False,
#           },
#       },
#       "action": {
#           "dtype": "float32",
#           "shape": (len(action_names),),
#           "names": {"action": action_names},
#       },
#   }
#

class HDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the HDF5 dataset file handler."""
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._hdf5_file_stream = h5py.File(file_path, "w")

        # set up a data group in the file
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0

        # set environment arguments
        # the environment type (we use gym environment type) is set to be compatible with robomimic
        # Ref: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L15
        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        """Destructor for the file handler."""
        self.close()

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        return self._demo_count

    """
    Operations.
    """

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def load_dataset_helper(group):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key])
                else:
                    # Converting group[key] to numpy array greatly improves the performance
                    # when converting to torch tensor
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]

        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()

        return episode

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        # Use custom demo id if provided, otherwise use default naming
        if demo_id is not None:
            episode_group_name = f"demo_{demo_id}"
        else:
            episode_group_name = f"demo_{self._demo_count}"

        # create episode group with the specified name
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        # store number of steps taken
        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            """Helper method to create dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        # increment total step counts
        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

        # Only increment demo count if using default indexing
        if demo_id is None:
            # increment total demo counts
            self._demo_count += 1

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()

        self._hdf5_file_stream.flush()

    def close(self):
        """Close the dataset file handler."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")
