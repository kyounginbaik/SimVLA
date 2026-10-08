"""SimVQA serialization + segmentation helpers.

Logic that was previously inlined in ``scripts/simvla/simvla_replay.py``.
Two main areas:

- per-env record buffering (``_env_key``, ``clear_simvqa_*``,
  ``flush_simvqa_on_success``) for accumulating samples and writing
  one JSONL line per (env, timestep) on episode success.
- segmentation → bbox → PaliGemma loc-token conversion
  (``find_instance_key_by_name``, ``bbox_from_rgba_seg``,
  ``bbox_to_paligemma_loc``).

A couple of generic image / JSON helpers used by the flush path
(``_to_pil_rgb``, ``to_jsonable``) live here too — they have no
caller outside SimVQA serialization.
"""

from __future__ import annotations

import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image


def _env_key(i: int) -> str:
    """Stable per-env key used across record / image / bbox dicts."""
    return f"env_{i:02d}"


def load_target_obj_bbox(path: str) -> dict[int, str]:
    """Load ``{int_env_id: obj_name}`` from a JSON file."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): str(v["obj"] if isinstance(v, dict) else v) for k, v in raw.items()}


def load_goalidx_to_text(path: str) -> dict[int, str]:
    """Load ``{goal_idx: subtask_text}`` from a JSON file."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def build_target_text(subtask: str | None) -> str:
    """Render a per-sample target string from a subtask label (None → 'unknown')."""
    return f"Subtask: {subtask if subtask is not None else 'unknown'}"


def object_type(name: str) -> str:
    """``mug0`` -> ``mug``: the scene names objects ``<type><index>``."""
    return re.sub(r"\d+$", "", str(name))


def sample_frame_objects(target: str, present: list[str], vocab, k: int = 3, rng=None) -> list[dict]:
    """Pick the ``k`` objects a frame's questions are asked about.

    The target manipuland (when the step has one) is always first. The remaining slots alternate
    between the OTHER objects present in the scene (names whose type is in ``vocab`` -- furniture
    is a rigid body too, and must not be asked about) and the vocab types ABSENT from the scene,
    present first, so a kitchen with one other object always gets it asked about and the dataset
    still carries "is the apple visible? -> no" negatives. Drawing uniformly from the union was
    measured to starve the present objects: 1 of ~21 pool entries, never sampled in 16 frames.
    Each entry is ``{"name", "present"}``; an absent entry's ``name`` is the bare type.
    """
    rng = rng or random
    vocab = set(vocab)
    present_types = {object_type(n) for n in present}
    others = [{"name": n, "present": True} for n in present
              if n != target and object_type(n) in vocab]
    exclude = {object_type(target)} if target is not None else set()
    absent = [{"name": t, "present": False} for t in sorted(vocab - present_types - exclude)]
    rng.shuffle(others); rng.shuffle(absent)
    head = [{"name": target, "present": True}] if target is not None else []
    picked = []
    pools = [others, absent]
    turn = 0
    while len(head) + len(picked) < k and (others or absent):
        pool = pools[turn % 2] if pools[turn % 2] else pools[(turn + 1) % 2]
        picked.append(pool.pop())
        turn += 1
    return head + picked


def gripper_closed(joint, open_joint, travel: float = 0.002) -> bool:
    """Closed = the finger joint moved more than ``travel`` in its native units from OPEN.

    The open reading is taken from the robot at reset, so nothing here depends on which end of
    the joint range is open or on the object's width: a fat mug that stops the fingers 4 mm
    from fully open is still a closed gripper (a fixed 0.035 threshold called it open). Prismatic
    grippers use metres; revolute grippers use radians.
    """
    return bool(abs(float(joint) - float(open_joint)) > travel)


def sample_subtask_frames(candidates: list[int], n: int, rng=None) -> list[int]:
    """``n`` frame indices from one subtask, always including its LAST frame.

    Include the transition frame for coverage. It is not automatically a positive
    reach/grasp label: the converter still checks the recorded pose measurement.
    """
    rng = rng or random
    if not candidates or n <= 0:
        return []
    last = candidates[-1]
    rest = rng.sample(candidates[:-1], min(n - 1, len(candidates) - 1))
    return sorted(rest + [last])


def describe_steps(goal_data: dict, reloadable_data: dict | None = None) -> list[dict]:
    """Flatten a goal file into ``[{language, prim_path, action, goal}]``, one per step.

    v2 files carry the sentence and the prim on the step. A v1 file carries neither; its
    ``<task>.reloadable.json`` twin does (``language`` / ``parameters.prim_path``), so pass it
    when there is one. ``goal`` is the authored nav goal ``[x, y, yaw]`` for ``N_s`` steps and
    None for everything else: arm goals are read from the demo, never from the file.
    """
    steps = goal_data["goals"][0]
    twin = (reloadable_data or {}).get("goals", [[]])[0] if reloadable_data else []
    out = []
    for i, st in enumerate(steps):
        if isinstance(st, dict):                       # v2
            action, goal = st["action"], st.get("goal")
            language, prim = st.get("language", ""), (st.get("params") or {}).get("prim_path")
        else:                                          # v1: [action, payload]
            action, payload = st[0], st[1]
            goal = list(payload[:3]) if action == "N_s" else None
            t = twin[i] if i < len(twin) else {}
            language, prim = t.get("language", ""), (t.get("parameters") or {}).get("prim_path")
        out.append({"language": language or "", "prim_path": prim,
                    "action": action, "goal": goal if action == "N_s" else None})
    for i, step in enumerate(out):
        step["grasp_target"] = False
        if step["action"] not in {"A_l", "A_r"} or i + 1 >= len(steps):
            continue
        following = steps[i + 1]
        if isinstance(following, dict):
            next_action = following["action"]
            close = (following.get("params") or {}).get("grasp", following.get("goal"))
        else:
            next_action, close = following
        step["grasp_target"] = next_action == "G_" + step["action"][-1] and close is True
    return out


def sample_subtask_ticks(candidates: list[int], n: int, rng=None) -> list[int]:
    """Map zero-based episode frame indices to replay's one-based loop ticks."""
    return [index + 1 for index in sample_subtask_frames(candidates, n, rng)]


def vqa_task_maps(steps: list[dict], vocab=None) -> tuple[dict[int, str], dict[int, str | None]]:
    """``(goal_idx -> subtask text, goal_idx -> target object)`` from :func:`describe_steps`.

    A nav step (``N_s``) targets its destination -- the leaf of its prim path
    (``/world/sink_cabinet`` -> ``sink_cabinet``): "is the robot close enough" is about where it
    is driving. Every other step targets the MANIPULAND: the last ``vocab``-typed object an arm
    step acted on (``/world/mug0`` -> ``mug0``), so "is the mug graspable" stays about the mug
    while it is carried to the sink and released there -- never "is the sink graspable". Before
    any arm step names one, the last vocab object seen on any step serves; a script that names
    no vocab object at all (a door task) falls back to carrying the last prim leaf.
    Arm targets are tracked separately: a second arm's handle reach/closure and
    articulation navigation target that handle, not the first arm's held object.
    """
    vocab = set(vocab) if vocab is not None else None
    subtasks = {i: st["language"] for i, st in enumerate(steps)}
    leaves = [(st["prim_path"] or "").rstrip("/").split("/")[-1] or None for st in steps]
    is_obj = lambda leaf: bool(leaf) and (vocab is None or object_type(leaf) in vocab)

    targets: dict[int, str | None] = {}
    manipuland = next((l for l in leaves if is_obj(l)), None)
    carried = next((l for l in leaves if l), None)
    arm_targets = {}
    handle_target = None
    for i, (st, leaf) in enumerate(zip(steps, leaves)):
        if leaf:
            carried = leaf
        if is_obj(leaf) and (st["action"] != "N_s" or manipuland is None):
            manipuland = leaf
        if st["action"] == "N_s" and leaf:
            targets[i] = leaf
        elif st["action"] in {"A_l", "A_r", "G_l", "G_r"}:
            arm = st["action"][-1]
            # A second arm may operate a handle while the first carries an
            # object. Do not label the left drawer reach as grasping that bowl.
            if st["action"].startswith("A_") and leaf:
                if is_obj(leaf) or "handle" in leaf:
                    arm_targets[arm] = leaf
                if "handle" in leaf:
                    handle_target = leaf
            targets[i] = arm_targets.get(arm, manipuland or carried)
        elif st["action"] == "N" and handle_target:
            targets[i] = handle_target
        else:
            targets[i] = manipuland if manipuland is not None else carried
    return subtasks, targets


def target_prim_path(steps: list[dict], goal_index: int, name: str) -> str:
    """Disambiguate repeated handle names using the last authored target path."""
    for step in reversed(steps[:goal_index + 1]):
        path = (step.get("prim_path") or "").rstrip("/")
        if path.rsplit("/", 1)[-1] == name:
            return path
    return name


def task_language_of(goal_data: dict, fallback: str | None) -> str | None:
    """The task sentence a v2 file records in ``run_config``; else ``fallback``."""
    return (goal_data.get("run_config") or {}).get("task_language") or fallback


def uniform_sample_plan(subtask_indices, frames_per_subtask: int) -> dict[int, int]:
    """``{subtask_idx: n}`` for every subtask (>= 0) the episode visits."""
    return {int(i): int(frames_per_subtask) for i in sorted({int(v) for v in subtask_indices if int(v) >= 0})}


def rel_pos_in_base(p_w: torch.Tensor, base_p_w: torch.Tensor, base_q_wxyz: torch.Tensor) -> torch.Tensor:
    """A world point expressed in the robot base frame: ``R(q)^T (p - base)``."""
    d = p_w - base_p_w
    w, x, y, z = base_q_wxyz.unbind(-1)
    # conjugate rotation applied to d (unit quaternion assumed)
    qv = torch.stack([-x, -y, -z])
    t = 2.0 * torch.cross(qv, d, dim=-1)
    return d + w * t + torch.cross(qv, t, dim=-1)


def _to_pil_rgb(img_tensor) -> Image.Image:
    """Isaac Lab image tensor → PIL RGB image (drops alpha, scales [0,1]→[0,255])."""
    x = img_tensor.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.moveaxis(x, 0, -1)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255).astype(np.uint8)
        else:
            x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def to_jsonable(x):
    """Recursively coerce torch / numpy values into JSON-serializable Python types."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()

    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    return x


def _normalize_success_envs(success_idx) -> list[int]:
    """Coerce env-id input (None / int / tensor / iterable) to ``list[int]``."""
    if success_idx is None:
        return []

    if isinstance(success_idx, torch.Tensor):
        if success_idx.numel() == 0:
            return []
        if success_idx.ndim == 0:
            return [int(success_idx.item())]
        return [int(x) for x in success_idx.detach().cpu().tolist()]

    if isinstance(success_idx, int):
        return [success_idx]

    try:
        return [int(x) for x in success_idx]
    except TypeError:
        return [int(success_idx)]


def flush_simvqa_on_success(
    success_idx,
    simvqa_records: dict,
    simvqa_jsonl_path: str,
    simvqa_images_dir: str,
    simvqa_seg_images_dir: str,
    prompt: str,
    task: str | None = None,
):
    """Append SimVQA records for the given envs to the JSONL output.

    Saves only the envs in ``success_idx`` (which can be ``int``,
    ``list[int]``, or ``torch.Tensor``). Does not clear the buffers —
    the caller's per-env reset handles that.
    """
    env_list = _normalize_success_envs(success_idx)
    if not env_list:
        return

    os.makedirs(simvqa_images_dir, exist_ok=True)
    os.makedirs(simvqa_seg_images_dir, exist_ok=True)

    with open(simvqa_jsonl_path, "a") as f:
        for eid in env_list:
            k = _env_key(int(eid))
            recs = simvqa_records.get(k, [])
            if not recs:
                continue

            for r in recs:
                t = int(r["t"])
                obj = r.get("obj_name", None)
                sub = r.get("subtask", None)
                gripper = r.get("gripper", None)
                r_gripper = r.get("right_gripper", None)
                l_gripper = r.get("left_gripper", None)
                goal_state_gripper = r.get("goal_state_gripper", None)
                goal_state_mobile = r.get("goal_state_mobile", None)
                distance_base = r.get("distance_base", None)

                target_text = build_target_text(sub)
                subtask_end = bool(r.get("subtask_end", False))

                img_paths = {}
                seg_img_paths = {}
                for cam in ("front", "wrist_left", "wrist_right"):
                    fn = f"{k}_t{t:04d}_{cam}.png"

                    out_path = os.path.join(simvqa_images_dir, fn)
                    _to_pil_rgb(r["images"][cam]).save(out_path)
                    img_paths[cam] = out_path

                    seg_out_path = os.path.join(simvqa_seg_images_dir, fn)
                    _to_pil_rgb(r["seg_images"][cam]).save(seg_out_path)
                    seg_img_paths[cam] = seg_out_path

                item = {
                    "id": f"{k}_t{t:04d}",
                    "frame_index": t - 1,
                    "task": task,
                    "prompt": prompt,
                    "images": img_paths,
                    "seg_images": seg_img_paths,
                    "target_text": target_text,
                    "subtask_end": subtask_end,
                    "objects": r.get("objects"),
                    "front_bbox": r["bboxes"].get("front"),
                    "wrist_left_bbox": r["bboxes"].get("wrist_left"),
                    "wrist_right_bbox": r["bboxes"].get("wrist_right"),
                    "meta": {
                        "goal_idx": int(r["goal_idx"]),
                        "grasp_target": r.get("grasp_target", False),
                        "obj_name": obj,
                        "gripper": gripper,
                        "r_gripper": r_gripper,
                        "l_gripper": l_gripper,
                        "goal_state_gripper": goal_state_gripper,
                        "goal_state_mobile": goal_state_mobile,
                        "distance_base": distance_base,
                        "gripper_joints": r.get("gripper_joints"),
                    },
                }
                f.write(json.dumps(to_jsonable(item)) + "\n")


def find_instance_keys_by_name(
    sensor_info: dict,
    name: str,
    annot_key: str = "instance_id_segmentation_fast",
    eid: int = 0,
) -> list:
    """Every RGBA key whose label lives under env ``eid`` and has ``name`` as a whole path segment.

    A qualified authored path also matches, excluding its leading ``/world``;
    this distinguishes identical handle names on different doors.
    A rigid object usually has one label; a fixture (``sink_cabinet``) has one per mesh -- door,
    basin, corpus -- and its bbox is the union of all of them, not the first one found.
    Segment matching, never substrings: "can0" is inside "sodacan0", "env_1" inside "env_10".
    """
    id_to_labels = sensor_info[annot_key]["idToLabels"]
    segments = name.lower().strip("/").split("/")
    if segments[0] == "world":
        segments = segments[1:]
    env_seg = f"env_{eid}"
    keys = []
    for rgba_key, label in id_to_labels.items():
        segs = str(label).lower().split("/")
        if env_seg in segs and any(segs[i:i + len(segments)] == segments
                                  for i in range(len(segs) - len(segments) + 1)):
            keys.append(rgba_key)
    return keys


def find_instance_key_by_name(sensor_info: dict, name: str, annot_key: str = "instance_id_segmentation_fast", eid: int = 0):
    """First match of :func:`find_instance_keys_by_name` as ``(rgba_key, label)``; ``(None, None)`` if none."""
    keys = find_instance_keys_by_name(sensor_info, name, annot_key=annot_key, eid=eid)
    if not keys:
        return None, None
    return keys[0], sensor_info[annot_key]["idToLabels"][keys[0]]


def bbox_from_rgba_seg(seg_rgba: torch.Tensor, rgba_keys):
    """Pixel bbox ``(x1, y1, x2, y2)`` of every pixel matching ANY of ``rgba_keys`` (one key or a list).

    Returns ``None`` if no key appears in the segmentation.
    """
    if rgba_keys is None:
        return None
    if not isinstance(rgba_keys, (list, tuple)) or (rgba_keys and not isinstance(rgba_keys[0], (list, tuple))):
        rgba_keys = [rgba_keys]
    if len(rgba_keys) == 0:
        return None
    mask = torch.zeros(seg_rgba.shape[:-1], dtype=torch.bool, device=seg_rgba.device)
    for key in rgba_keys:
        t = torch.tensor(key, device=seg_rgba.device, dtype=seg_rgba.dtype)
        mask |= (seg_rgba == t).all(dim=-1)

    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def bbox_to_paligemma_loc(x1: int, y1: int, x2: int, y2: int, H: int, W: int) -> str:
    """Convert a pixel bbox to PaliGemma's ``<locYYYY><locXXXX>`` token form.

    PaliGemma loc-token order is ``ymin, xmin, ymax, xmax`` mapped to
    ``[0, 1024]``.
    """
    def to_loc_x(x: int) -> int:
        return int(round(x * 1024.0 / W))

    def to_loc_y(y: int) -> int:
        return int(round(y * 1024.0 / H))

    ymin = to_loc_y(y1)
    xmin = to_loc_x(x1)
    ymax = to_loc_y(y2)
    xmax = to_loc_x(x2)

    return f"<loc{ymin:04d}><loc{xmin:04d}><loc{ymax:04d}><loc{xmax:04d}>"


def clear_simvqa_records_for_envs(env_ids: torch.Tensor, simvqa_records: dict) -> None:
    """Empty the per-env record buffers for the given envs."""
    if env_ids is None or env_ids.numel() == 0:
        return
    for eid in env_ids.detach().cpu().tolist():
        k = _env_key(int(eid))
        if k in simvqa_records:
            simvqa_records[k].clear()


def clear_simvqa_buffers_for_envs(
    env_ids: torch.Tensor,
    front_img_dict: dict,
    wrist_left_img_dict: dict,
    wrist_right_img_dict: dict,
    front_bbox_dict: dict,
    wrist_left_bbox_dict: dict,
    wrist_right_bbox_dict: dict,
) -> None:
    """Drop references to old per-env image / bbox tensors before next episode."""
    if env_ids is None or env_ids.numel() == 0:
        return
    if env_ids.dtype != torch.long:
        env_ids = env_ids.to(torch.long)
    for eid in env_ids.detach().cpu().tolist():
        k = _env_key(int(eid))

        if k in front_img_dict:
            front_img_dict[k].clear()
        if k in wrist_left_img_dict:
            wrist_left_img_dict[k].clear()
        if k in wrist_right_img_dict:
            wrist_right_img_dict[k].clear()

        if k in front_bbox_dict:
            front_bbox_dict[k].clear()
        if k in wrist_left_bbox_dict:
            wrist_left_bbox_dict[k].clear()
        if k in wrist_right_bbox_dict:
            wrist_right_bbox_dict[k].clear()
