#!/usr/bin/env python3
"""Test a leftward base offset for RBY1's right-arm mug reach (not a validated recipe)."""
import argparse
import copy
import json
import math
from pathlib import Path


def with_lateral_parking_offset(goal, offset_m=.15):
    if not math.isfinite(offset_m) or not 0 < offset_m <= .3:
        raise ValueError("offset must be finite and in (0, 0.3] metres")
    result = copy.deepcopy(goal)
    first = result["goals"][0][0]
    if (first.get("skill") != "nav.to_prim" or first.get("action") != "N_s"
            or first.get("params", {}).get("which_arm") != "Right"):
        raise ValueError("expected a right-arm nav.to_prim first step")
    x, y, yaw = first["goal"]
    first["goal"] = [x - math.sin(yaw)*offset_m, y + math.cos(yaw)*offset_m, yaw]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2]
                        / "examples/goals/Isaac-Kitchen-v813r-00.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-m", type=float, default=.15)
    args = parser.parse_args()
    result = with_lateral_parking_offset(json.loads(args.source.read_text()), args.offset_m)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
