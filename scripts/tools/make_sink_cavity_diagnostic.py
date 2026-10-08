"""Kitchen-813/00 diagnostic with below-rim acceptance, not legacy sink proximity."""
import argparse
import copy
import json
import math
from pathlib import Path


def make_goal(source, eef_position):
    if (source.get("kitchen_num"), source.get("kitchen_sub_num")) != (813, 0):
        raise ValueError("Measured basin region is valid only for kitchen 813 rotation 00")
    if len(eef_position) != 3 or not all(math.isfinite(v) for v in eef_position):
        raise ValueError("EEF position must contain three finite world coordinates")
    goal = copy.deepcopy(source)
    steps = goal["goals"][0]
    placements = [steps[i - 1] for i, step in enumerate(steps) if i > 0
                  and step.get("skill") == "gripper.set"
                  and step.get("params", {}).get("grasp") is False
                  and steps[i - 1].get("skill") in ("arm.bowl_place", "arm.place", "arm.pose")]
    if len(placements) != 1:
        raise ValueError("Expected exactly one sink placement in this diagnostic")
    placement = placements[0]
    placement.update(skill="arm.place", params={"prim_path": "/world/sink_cabinet"},
                     goal=[*map(float, eef_position), 1., 0., 0., 0.],
                     language="Release the mug above the open sink basin")
    # Authored sink bounds: x=[1.5160,2.0664], y=[-2.5473,-1.9769],
    # z=[.7209,.8656]. The cabinet root XY=(1.8686,-2.2621).
    # This small disk lies inside the basin's XY bounds, and the height band
    # excludes the .9011 m countertop and floor. It checks the OBJECT ROOT,
    # not full mesh containment, stability over time, or arbitrary kitchens.
    goal["diagnostic_sink_cavity"] = {
        "requires_environment": {"SIMVLA_SINK_CAVITY_COLLISIONS": "1",
                                 "SIMVLA_KITCHEN813_SINK_INTERIOR_GATE": "1"},
        "accepted_episode": False,
        "predicate_scope": "Object root below basin rim and within an interior XY disk; not full mesh containment",
        "legacy_proximity_replaced": True,
    }
    return goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eef-position", nargs=3, type=float, required=True)
    args = parser.parse_args()
    result = make_goal(json.loads(args.source.read_text()), args.eef_position)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
