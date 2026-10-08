"""Select one authored grasp pose from a legacy or v2 goal, preserving the task.

This makes grasp-candidate ablations explicit and repeatable. Collection normally explores all
authored poses; use this only when diagnosing or reproducing a specific pose.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def side_approach_indices(poses: list[list[float]], min_horizontal: float = 0.65) -> list[int]:
    """Indices whose BoDex tool +Z axis has a substantial horizontal component.

    Goal quaternions are stored wxyz. Their tool +Z axis is the third rotation
    matrix column; this uses only stdlib so it can be checked before Isaac starts.
    """
    if not 0.0 < min_horizontal <= 1.0:
        raise ValueError("min_horizontal must be in (0, 1]")
    selected = []
    for index, pose in enumerate(poses):
        if len(pose) != 7 or not all(math.isfinite(v) for v in pose):
            raise ValueError(f"candidate {index} must be a finite seven-number pose")
        w, x, y, z = pose[3:7]
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-8:
            raise ValueError(f"candidate {index} has a zero quaternion")
        w, x, y, z = (v / norm for v in (w, x, y, z))
        axis_x = 2.0 * (x*z + w*y)
        axis_y = 2.0 * (y*z - w*x)
        if math.hypot(axis_x, axis_y) >= min_horizontal:
            selected.append(index)
    return selected


def select_candidate(goal: dict[str, Any], arm: str, index: int) -> dict[str, Any]:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    if index < 0:
        raise ValueError("candidate index must be nonnegative")
    scripts = goal.get("goals")
    if not isinstance(scripts, list) or not scripts:
        raise ValueError("goal must contain a non-empty 'goals' list")
    action = "A_l" if arm == "left" else "A_r"
    # Anubis's public example still uses [channel, payload] legacy steps;
    # RB-Y1 and AI Worker use dictionaries. Match pose banks, not arm.pose steps.
    matches = []
    for script_index, script in enumerate(scripts):
        if not isinstance(script, list):
            continue
        for step_index, step in enumerate(script):
            if isinstance(step, dict):
                channel, poses = step.get("action"), step.get("goal")
            elif isinstance(step, list) and len(step) == 2:
                channel, poses = step
            else:
                continue
            if (channel == action and isinstance(poses, list) and poses
                    and isinstance(poses[0], list)):
                matches.append((script_index, step_index, poses))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {action} step, found {len(matches)}")
    script_index, step_index, poses = matches[0]
    if not isinstance(poses, list) or index >= len(poses):
        count = len(poses) if isinstance(poses, list) else 0
        raise ValueError(f"{action} candidate index {index} is out of range (have {count})")
    pose = poses[index]
    if (not isinstance(pose, list) or len(pose) != 7
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pose)):
        raise ValueError(f"{action} candidate {index} must be a finite [x,y,z,qw,qx,qy,qz] pose")
    result = json.loads(json.dumps(goal))
    step = result["goals"][script_index][step_index]
    if isinstance(step, dict):
        step["goal"] = [list(pose)]
    else:
        step[1] = [list(pose)]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("left", "right"))
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        goal = json.loads(args.input.read_text())
        selected = select_candidate(goal, args.arm, args.index)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, allow_nan=True) + "\n")
    print(f"selected {args.arm} grasp candidate {args.index}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
