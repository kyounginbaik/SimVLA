#!/usr/bin/env python3
"""Prepare a kitchen-813 diagnostic base waypoint before the left drawer reach.

This is a feasibility experiment, not a validated replacement collection goal.
The original bowl grasp and both drawer orientations are preserved.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys


def add_step_metadata(goal, reloadable):
    """Upgrade only encoding/labels; keep the legacy executor interpretation."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    from goal_format import decode_v1

    source_steps = copy.deepcopy(reloadable["goals"][0])
    if len(source_steps) != 12 or len(goal["goals"][0]) != 13:
        raise ValueError("metadata requires the original 12-step reloadable and 13-step diagnostic")
    source_steps.insert(4, {"language": "Align the left arm with the drawer handle",
                            "parameters": {"prim_path": "/world/base_cabinet/drawer_0_0"}})
    result = copy.deepcopy(goal)
    described = []
    for (action, payload), metadata in zip(result["goals"][0], source_steps):
        step = decode_v1(action, payload)
        params = dict(step.params)
        prim_path = metadata.get("parameters", {}).get("prim_path")
        if prim_path:
            params["prim_path"] = prim_path
        described.append({"skill": step.skill, "action": step.action, "params": params,
                          "goal": step.goal, "language": metadata["language"]})
    result["goals"] = [described]
    result["version"] = 2
    return result


def with_drawer_waypoint(goal, lateral_shift_m=.12, *, swap_drawer_jaws=False, backoff_m=0.):
    if not math.isfinite(lateral_shift_m) or not 0 < lateral_shift_m <= .3:
        raise ValueError("lateral shift must be finite and in (0, 0.3] metres")
    if not math.isfinite(backoff_m) or not 0 <= backoff_m <= .2:
        raise ValueError("backoff must be finite and in [0, 0.2] metres")
    result = copy.deepcopy(goal)
    steps = result["goals"][0]
    if (len(steps) != 12 or steps[0][0] != "N_s"
            or [step[0] for step in steps[3:7]] != ["A_r", "A_l", "A_l", "G_l"]):
        raise ValueError("expected the unmodified 12-step Anubis kitchen-813 drawer goal")
    if swap_drawer_jaws:
        # Local tool-Z half-turn swaps the parallel jaws without changing their
        # closing axis or the handle position. This is a diagnostic alternative
        # wrist branch, not a change to BoDex object grasps.
        for index in (4, 5):
            pose = steps[index][1]
            w, x, y, z = pose[3:7]
            pose[3:7] = [-z, y, -x, w]
    waypoint = list(steps[0][1])
    waypoint[0] += lateral_shift_m
    waypoint[0] -= math.cos(waypoint[2]) * backoff_m
    waypoint[1] -= math.sin(waypoint[2]) * backoff_m
    # N_s is the legacy literal-navigation channel; N is reserved for articulation skills.
    steps.insert(4, ["N_s", waypoint])
    result["run_config"]["export_groups"] = "4,5,6,7,8;9,10,11;12"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2]
                        / "examples/goals/Isaac-Kitchen-v813-00.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lateral-shift-m", type=float, default=.12)
    parser.add_argument("--swap-drawer-jaws", action="store_true")
    parser.add_argument("--backoff-m", type=float, default=0.)
    parser.add_argument("--with-metadata", action="store_true",
                        help="Embed v2 language/target metadata from the source's reloadable twin")
    args = parser.parse_args()
    result = with_drawer_waypoint(json.loads(args.source.read_text()), args.lateral_shift_m,
                                 swap_drawer_jaws=args.swap_drawer_jaws, backoff_m=args.backoff_m)
    if args.with_metadata:
        twin = args.source.with_name(args.source.stem + ".reloadable.json")
        result = add_step_metadata(result, json.loads(twin.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
