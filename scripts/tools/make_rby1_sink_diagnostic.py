#!/usr/bin/env python3
"""Kitchen-813 delivery diagnostic: align the right hand with the sink basin."""
import argparse
import copy
import json
import math
from pathlib import Path


def make_goal(source, *, parking_advance_m=0., forward_m=.40, place_position=None,
              final_approach_m=0.):
    if (not math.isfinite(parking_advance_m) or not 0 <= parking_advance_m <= .15
            or not math.isfinite(forward_m) or not .15 <= forward_m <= .40):
        raise ValueError("parking advance must be in [0, .15] and reach in [.15, .40] metres")
    if not math.isfinite(final_approach_m) or not 0 <= final_approach_m <= .08:
        raise ValueError("final approach must be finite and in [0, .08] metres")
    if final_approach_m and place_position is None:
        raise ValueError("final approach requires an explicit absolute placement target")
    goal = copy.deepcopy(source)
    steps = goal["goals"][0]
    if (goal.get("kitchen_num") != 813 or len(steps) != 8
            or steps[4].get("skill") != "nav.to_prim"
            or steps[4].get("params", {}).get("prim_path") != "/world/sink_cabinet"
            or steps[5].get("action") != "A_r"
            or steps[5].get("skill") != "arm.bowl_place"):
        raise ValueError("requires the eight-step kitchen-813 right-arm mug-to-sink task")
    # The right carry hand sits ~0.22 m to the robot's right. Park left of the
    # basin centre so the hand, not the base, aligns with it. Keep the safe
    # x standoff for the rectangular base's final rotation.
    steps[4]["goal"][1] = -2.04
    steps[4]["goal"][0] += parking_advance_m
    # Cross the counter edge before releasing; do not descend into its rim.
    steps[5]["params"].update(forward_m=forward_m, down_m=0., roll_deg=0.)
    if place_position is not None:
        if len(place_position) != 3 or not all(math.isfinite(v) for v in place_position):
            raise ValueError("place position must contain three finite world coordinates")
        # Experimental alternative: the authored sink target does not move when
        # the sampled parking pose or the retained carry wrist changes. arm.place
        # preserves the measured wrist orientation; it still requires planning,
        # physical grasp retention, release and the original terminal predicate.
        steps[5].update(skill="arm.place", params={"prim_path": "/world/sink_cabinet"},
                        goal=[float(v) for v in place_position] + [1., 0., 0., 0.])
    if final_approach_m:
        # Complete the corner sweep at the original parking waypoint, then move
        # along the parked heading. Moving both base and release point preserves
        # arm reach while giving a larger basin margin. This is an experimental
        # authored kitchen-813 route, not a general navigation safety guarantee.
        approach = copy.deepcopy(steps[4])
        yaw = approach["goal"][2]
        approach["goal"][0] += math.cos(yaw) * final_approach_m
        approach["goal"][1] += math.sin(yaw) * final_approach_m
        approach["language"] = "Move straight closer to the sink after completing the turn"
        steps.insert(5, approach)
        goal["run_config"]["export_groups"] = "3;6,7,8"
        goal["diagnostic_final_approach"] = {
            "distance_m": final_approach_m, "accepted_episode": False,
            "scope": "Kitchen-813 straight approach after the original sink parking turn",
        }
    return goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parking-advance-m", type=float, default=0.)
    parser.add_argument("--forward-m", type=float, default=.40)
    parser.add_argument("--place-position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Experimental absolute world EEF target instead of a relative reach")
    parser.add_argument("--final-approach-m", type=float, default=0.,
                        help="Experimental straight advance after completing the original parking turn")
    args = parser.parse_args()
    result = make_goal(json.loads(args.source.read_text()), parking_advance_m=args.parking_advance_m,
                       forward_m=args.forward_m, place_position=args.place_position,
                       final_approach_m=args.final_approach_m)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
