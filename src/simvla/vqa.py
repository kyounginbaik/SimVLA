#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def subtask_list_from(data: Any) -> List[str]:
    """A subtask list from either a JSON list or a ``{goal_idx: text}`` map (ordered by index)."""
    if isinstance(data, dict):
        return [str(data[k]) for k in sorted(data, key=int)]
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("subtask list must be a JSON list or a {goal_idx: text} map")


def load_subtask_list(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        return subtask_list_from(json.loads(text))
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def load_root_meta(root: Path) -> Optional[Dict[str, Any]]:
    """The ``vqa_meta.json`` the capture writes beside ``vqa.jsonl`` (task, language, subtasks,
    targets), or None for a root from before it existed."""
    p = root / "vqa_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_root(input_path: Path, extract_dir: Optional[Path]) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError("input_path must be a directory or .zip file")
    if extract_dir is None:
        extract_dir = input_path.parent / f"{input_path.stem}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zf:
        zf.extractall(extract_dir)
    candidates = [p.parent for p in extract_dir.rglob("vqa.jsonl")]
    if not candidates:
        raise FileNotFoundError("Could not find vqa.jsonl in extracted zip")
    candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def normalize_rel_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = p.replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def resolve_path(root: Path, maybe_rel: Optional[str]) -> Optional[str]:
    if maybe_rel is None:
        return None
    rel = normalize_rel_path(maybe_rel)
    if rel is None:
        return None

    explicit_camera_path = rel.startswith(("images/", "seg_images/"))
    for marker in ("/images/", "/seg_images/"):
        if marker in rel:
            _, right = rel.split(marker, 1)
            rel = marker.strip("/") + "/" + right
            explicit_camera_path = True
            break

    p = root / rel
    # RGB and segmentation deliberately share basenames. A missing RGB must
    # never silently resolve to its segmentation image (or vice versa).
    if p.exists() or explicit_camera_path:
        return str(p.resolve())

    name = Path(rel).name
    matches = list(root.rglob(name))
    if len(matches) == 1:
        return str(matches[0].resolve())
    return str(p.resolve())


def fmt_float(x: float, ndigits: int = 4) -> str:
    s = f"{float(x):.{ndigits}f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s if s else "0"


#: "yes" gates for object_reachable / object_graspable. The nav gate mirrors nav_tuning's
#: goal_reached_distance / goal_reached_yaw (0.02 m / 0.1 rad): the base is "close enough" where
#: the planner itself stops driving. The gripper gate uses the same numbers: 2 cm and ~6 deg is the
#: band inside which the grasp planner's pose is met and closing the fingers is the next action.
REACH_POS_TOL_M = 0.02
REACH_YAW_TOL_RAD = 0.1
GRASP_POS_TOL_M = 0.02
GRASP_ROT_TOL_RAD = 0.1


def quat_angle_rad(qw: float, qx: float, qy: float, qz: float) -> float:
    """Rotation angle of a (w, x, y, z) quaternion, sign-insensitive."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(qw) / n)))


def safe_bool_to_yes_no(x: Any) -> Optional[str]:
    if x is None:
        return None
    return "yes" if bool(x) else "no"


def clean_obj_name(name: Optional[str]) -> Optional[str]:
    """Scene name -> noun for the question: ``mug0`` -> ``mug``, ``door_handle`` -> ``door handle``."""
    if name is None:
        return None
    return re.sub(r"\d+$", "", str(name)).replace("_", " ").strip()


def get_current_subtask(goal_idx: Optional[int], subtask_list: List[str]) -> Optional[str]:
    if goal_idx is None:
        return None
    idx = int(goal_idx)
    if 0 <= idx < len(subtask_list):
        return subtask_list[idx]
    return None


def get_next_subtask(goal_idx: Optional[int], subtask_list: List[str]) -> Optional[str]:
    if goal_idx is None:
        return None
    idx = int(goal_idx) + 1
    if 0 <= idx < len(subtask_list):
        return subtask_list[idx]
    return None


def parse_subtask_from_target_text(target_text: Optional[str]) -> Optional[str]:
    if not target_text:
        return None
    prefix = "Subtask:"
    if target_text.startswith(prefix):
        return target_text[len(prefix):].strip()
    return None


def is_navigation(meta: Dict[str, Any]) -> bool:
    return (
        meta.get("goal_state_mobile") is not None
        and meta.get("goal_state_gripper") is None
        and meta.get("gripper") is None
    )


def is_manipulation(meta: Dict[str, Any]) -> bool:
    return (
        meta.get("goal_state_mobile") is None
        and meta.get("goal_state_gripper") is not None
        and meta.get("gripper") in ("left", "right")
    )


def make_base(row: Dict[str, Any], root: Path) -> Dict[str, Any]:
    images = row.get("images", {}) or {}
    seg_images = row.get("seg_images", {}) or {}

    out_images = {
        "front": resolve_path(root, images.get("front")),
        "wrist_left": resolve_path(root, images.get("wrist_left")),
        "wrist_right": resolve_path(root, images.get("wrist_right")),
    }
    out_seg_images = {
        "front": resolve_path(root, seg_images.get("front")),
        "wrist_left": resolve_path(root, seg_images.get("wrist_left")),
        "wrist_right": resolve_path(root, seg_images.get("wrist_right")),
    }

    # Raw ids are env/timestep only; the task (kitchen) name makes them unique across a
    # multi-kitchen concatenation. New captures record it, older roots are named after it.
    task = row.get("task") or root.name
    return {
        "source_id": f'{task}/{row.get("id")}',
        "prompt": row.get("prompt"),
        "image": out_images,
        "seg_image": out_seg_images,
        "image_mask": {k: bool(v) for k, v in out_images.items()},
        "seg_image_mask": {k: bool(v) for k, v in out_seg_images.items()},
        "action_loss_mask": False,
    }


def build_examples_for_row(row: Dict[str, Any], subtask_list: List[str], root: Path) -> List[Dict[str, Any]]:
    base = make_base(row, root)
    meta = row.get("meta", {}) or {}

    goal_idx = meta.get("goal_idx")
    obj_name = clean_obj_name(meta.get("obj_name"))
    gripper = meta.get("gripper")
    r_gripper = meta.get("r_gripper")
    l_gripper = meta.get("l_gripper")
    goal_state_gripper = meta.get("goal_state_gripper")
    goal_state_mobile = meta.get("goal_state_mobile")
    distance_base = meta.get("distance_base")

    front_bbox = row.get("front_bbox")
    wrist_left_bbox = row.get("wrist_left_bbox")
    wrist_right_bbox = row.get("wrist_right_bbox")

    # One entry per object the frame asks about: the target first, then the sampled others
    # (present or absent). Records from before the per-object capture have only the target's
    # top-level fields; they are folded into the same shape.
    objects = row.get("objects") or [{
        "name": meta.get("obj_name"),
        "present": True,
        "bboxes": {"front": front_bbox, "wrist_left": wrist_left_bbox, "wrist_right": wrist_right_bbox},
        "distance_base": distance_base,
    }]

    current_subtask = get_current_subtask(goal_idx, subtask_list) or parse_subtask_from_target_text(row.get("target_text"))
    next_subtask = get_next_subtask(goal_idx, subtask_list)

    nav = is_navigation(meta)
    manip = is_manipulation(meta)

    examples: List[Dict[str, Any]] = []

    def add(vqa_function: str, question: str, answer: Optional[str], subject: Optional[str] = None) -> None:
        if not question or answer is None or answer == "":
            return
        ex = dict(base)
        ex["id"] = f'{base["source_id"]}::{vqa_function}' + (f"::{subject}" if subject else "")
        ex["vqa_function"] = vqa_function
        ex["cotrain_prompt"] = f"Question: {question}\nAnswer:"
        ex["cotrain_answer"] = answer
        examples.append(ex)

    if base.get("prompt") and current_subtask:
        add(
            "high_level_subtask",
            f'To complete the task "{base["prompt"]}", what should the robot do now?',
            current_subtask,
        )

    if current_subtask and next_subtask:
        add(
            "next_subtask",
            f'After current subtask "{current_subtask}", what is the next subtask?',
            next_subtask,
        )

    for o in objects:
        if not o.get("name"):
            continue
        noun = clean_obj_name(o["name"])
        bb = o.get("bboxes") or {}
        o_front, o_wl, o_wr = bb.get("front"), bb.get("wrist_left"), bb.get("wrist_right")
        present = bool(o.get("present", True))

        add(
            "object_detection",
            f"Is the {noun} visible in the front camera?",
            f"yes. bbox: {o_front}" if (present and o_front is not None) else "no",
            subject=o["name"],
        )
        if not present:
            continue

        d = o.get("distance_base")
        if isinstance(d, list) and len(d) >= 3:
            dx, dy, dz = d[:3]
            add(
                "object_3d_information",
                f"How far is the {noun} from the robot base?",
                f"dx {fmt_float(dx)} dy {fmt_float(dy)} dz {fmt_float(dz)}",
                subject=o["name"],
            )

        if o_front is not None and (o_wl is not None or o_wr is not None):
            left_ans = o_wl if o_wl is not None else "none"
            right_ans = o_wr if o_wr is not None else "none"
            add(
                "view_correspondence",
                f"Given front bbox {o_front}, what are the wrist camera bboxes?",
                f"left {left_ans} right {right_ans}",
                subject=o["name"],
            )

    if nav and obj_name and isinstance(goal_state_mobile, list) and len(goal_state_mobile) >= 3:
        dx, dy, dyaw = goal_state_mobile[:3]
        reached = math.hypot(dx, dy) <= REACH_POS_TOL_M and abs(dyaw) <= REACH_YAW_TOL_RAD
        add(
            "object_reachable",
            f"Is the robot close enough to manipulate the {obj_name}?",
            "yes" if reached else f"no. move base dx {fmt_float(dx)} dy {fmt_float(dy)} dyaw {fmt_float(dyaw)}",
        )

    if (manip and meta.get("grasp_target") is True and obj_name
            and gripper in ("left", "right") and isinstance(goal_state_gripper, list)
            and len(goal_state_gripper) >= 7):
        dx, dy, dz, qw, qx, qy, qz = goal_state_gripper[:7]
        graspable = (math.sqrt(dx * dx + dy * dy + dz * dz) <= GRASP_POS_TOL_M
                     and quat_angle_rad(qw, qx, qy, qz) <= GRASP_ROT_TOL_RAD)
        add(
            "object_graspable",
            f"Is the {obj_name} graspable with {gripper} gripper?",
            "yes" if graspable else
            f"no. move gripper dx {fmt_float(dx)} dy {fmt_float(dy)} dz {fmt_float(dz)} "
            f"dquat {fmt_float(qw)} {fmt_float(qx)} {fmt_float(qy)} {fmt_float(qz)}",
        )

    l_ans = safe_bool_to_yes_no(l_gripper)
    if l_ans is not None:
        add(
            "left_gripper_open",
            "Is the robot's left gripper closed?",
            l_ans,
        )

    r_ans = safe_bool_to_yes_no(r_gripper)
    if r_ans is not None:
        add(
            "right_gripper_open",
            "Is the robot's right gripper closed?",
            r_ans,
        )

    return examples


_KITCHEN_RE = re.compile(r"Isaac-Kitchen-v(\d+)-\d+")


def parse_kitchen_ranges(spec: str) -> List[range]:
    """Parse a comma-separated range spec like '472-493,616-677' into range objects."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ranges.append(range(int(lo), int(hi) + 1))
        else:
            n = int(part)
            ranges.append(range(n, n + 1))
    return ranges


def kitchen_num_in_ranges(name: str, ranges: List[range]) -> bool:
    m = _KITCHEN_RE.search(name)
    if not m:
        return False
    n = int(m.group(1))
    return any(n in r for r in ranges)


def find_dataset_roots(input_path: Path, kitchen_ranges: Optional[List[range]] = None) -> List[Path]:
    """Return list of dataset roots (dirs containing vqa.jsonl).

    If input_path itself contains vqa.jsonl it is treated as a single dataset.
    Otherwise every immediate subdirectory that contains vqa.jsonl is collected,
    which covers the case where input_path is a parent of many Isaac-Kitchen-* folders.
    If kitchen_ranges is provided, only folders whose kitchen number falls in one of
    the ranges are included.
    """
    if (input_path / "vqa.jsonl").exists():
        return [input_path]

    roots = sorted(
        p.parent for p in input_path.glob("*/vqa.jsonl")
    )
    if kitchen_ranges is not None:
        roots = [r for r in roots if kitchen_num_in_ranges(r.name, kitchen_ranges)]
    if not roots:
        raise FileNotFoundError(
            f"No vqa.jsonl found directly in {input_path} or its immediate subdirectories"
            + (f" matching ranges" if kitchen_ranges else "")
        )
    return roots


def convert_root(root: Path, subtask_list: Optional[List[str]], out) -> int:
    """Expand one dataset root. The subtask texts come from the root's own ``vqa_meta.json``
    when the capture wrote one (any task, no per-run flag), else from ``subtask_list``."""
    vqa_path = root / "vqa.jsonl"
    if not vqa_path.exists():
        raise FileNotFoundError(f"Could not find {vqa_path}")
    meta = load_root_meta(root)
    if meta is not None and meta.get("subtasks") is not None:
        subtask_list = subtask_list_from(meta["subtasks"])
    if subtask_list is None:
        raise FileNotFoundError(
            f"{root}: no vqa_meta.json beside vqa.jsonl and no --subtask-list given; one of the "
            f"two has to say what each goal_idx means."
        )
    count = 0
    with vqa_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for ex in build_examples_for_row(row, subtask_list, root):
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
    return count


def convert(input_path: Path, subtask_list_path: Optional[Path], output_jsonl: Path, extract_dir: Optional[Path], kitchen_ranges_spec: Optional[str] = None) -> int:
    subtask_list = load_subtask_list(subtask_list_path) if subtask_list_path else None

    kitchen_ranges = parse_kitchen_ranges(kitchen_ranges_spec) if kitchen_ranges_spec else None

    if input_path.suffix.lower() == ".zip":
        # Single zip — existing behaviour
        root = ensure_root(input_path, extract_dir)
        roots = [root]
    else:
        roots = find_dataset_roots(input_path, kitchen_ranges)

    total = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("x", encoding="utf-8") as out:
        for root in roots:
            n = convert_root(root, subtask_list, out)
            print(f"  {root.name}: {n} examples")
            total += n
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Single dataset root dir (containing vqa.jsonl), a zip file, "
            "or a parent directory containing multiple Isaac-Kitchen-* dataset folders"
        ),
    )
    parser.add_argument("--subtask-list", type=Path, default=None,
                        help="txt/json file: index == goal_idx. Only needed for roots captured before "
                             "vqa_meta.json existed; a root's own meta always wins.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output cotrain jsonl")
    parser.add_argument("--extract-dir", type=Path, default=None, help="Optional extract dir for zip input")
    parser.add_argument(
        "--kitchen-ranges",
        default=None,
        help="Comma-separated kitchen number ranges to include, e.g. '472-493,616-677,699-737'",
    )
    args = parser.parse_args()

    n = convert(args.input_path, args.subtask_list, args.output_jsonl, args.extract_dir, args.kitchen_ranges)
    print(f"Wrote {n} cotrain examples total to {args.output_jsonl}")


if __name__ == "__main__":
    main()
