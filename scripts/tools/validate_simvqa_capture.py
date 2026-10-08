#!/usr/bin/env python3
"""Validate captured frames/labels before treating a replay as usable SimVQA data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from simvla.vqa import resolve_path


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {text}")
    return value


def validate(root: Path) -> dict:
    from PIL import Image

    result = {"schema_version": 1, "passed": False, "records": 0,
              "images": 0, "goal_indices": [], "errors": [],
              "scope": "Capture integrity and annotation consistency, not physical task success."}
    try:
        metadata = json.loads((root / "vqa_meta.json").read_text(), parse_constant=reject_constant,
                              parse_float=finite_float)
        subtasks, targets = metadata["subtasks"], metadata["targets"]
        grasp_steps = metadata["grasp_steps"]
        expected = set(metadata["sampled_goal_indices"])
        if not expected:
            raise ValueError("no sampled goal indices declared")
        ids, observed = set(), set()
        for line in (root / "vqa.jsonl").read_text().splitlines():
            row = json.loads(line, parse_constant=reject_constant, parse_float=finite_float)
            if row["id"] in ids:
                raise ValueError(f"duplicate record id: {row['id']}")
            ids.add(row["id"])
            frame = row["frame_index"]
            if type(frame) is not int or frame < 0:
                raise ValueError("frame_index must be a nonnegative integer")
            meta = row["meta"]
            index = meta["goal_idx"]
            key = str(index)
            if not subtasks[key].strip() or row["target_text"] != "Subtask: " + subtasks[key]:
                raise ValueError(f"missing/mismatched language at goal {index}")
            if meta["obj_name"] != targets[key]:
                raise ValueError(f"mismatched target at goal {index}")
            if type(meta["grasp_target"]) is not bool or meta["grasp_target"] != grasp_steps[key]:
                raise ValueError(f"missing/mismatched grasp annotation at goal {index}")
            for name, size in (("goal_state_mobile", 3), ("goal_state_gripper", 7), ("distance_base", 3)):
                values = meta.get(name)
                if values is not None and (not isinstance(values, list) or len(values) != size
                                           or any(type(v) not in (int, float) for v in values)):
                    raise ValueError(f"malformed {name}")
            dimensions = {}
            for kind in ("images", "seg_images"):
                for camera in ("front", "wrist_left", "wrist_right"):
                    path = Path(resolve_path(root, row[kind][camera]))
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f"image outside capture: {path}")
                    with Image.open(path) as img:
                        if min(img.size) < 2:
                            raise ValueError(f"empty image: {path}")
                        if kind == "images":
                            dimensions[camera] = img.size
                        elif img.size != dimensions[camera]:
                            raise ValueError(f"RGB/segmentation size mismatch: {camera}")
                        img.verify()
                    result["images"] += 1
            observed.add(index)
            result["records"] += 1
        result["goal_indices"] = sorted(observed)
        if not ids or observed != expected:
            raise ValueError(f"sample coverage mismatch: expected {sorted(expected)}, observed {sorted(observed)}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    result["passed"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.capture)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
