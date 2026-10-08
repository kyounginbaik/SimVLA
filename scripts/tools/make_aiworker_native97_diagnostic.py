"""Recreate a kitchen-813 native-BoDex diagnostic goal, NOT an accepted episode.

The collector recalibrates translation from live pad geometry. This script preserves
the original trial's nominal translation and exact native-bank orientation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
EXPECTED_BANK_SHA256 = "cb630aee81c2701a7774292c7e52ee70f075e8a5651101e92c57216c8f3b4909"


def sink_clearance_goal(goal: dict, *, wall_clearance: bool = False) -> None:
    """Experimental kitchen-813 parking clearance; preserve grasp and task gates."""
    steps = goal["goals"][0]
    if (goal.get("kitchen_num") != 813 or len(steps) != 8
            or steps[4].get("skill") != "nav.to_prim"
            or steps[4].get("params", {}).get("prim_path") != "/world/sink_cabinet"
            or steps[5].get("skill") != "arm.bowl_place"
            or steps[5].get("action") != "A_l"):
        raise ValueError("requires the eight-step kitchen-813 left-arm sink task")
    # Leave space for the final base rotation; align the carried left hand,
    # not the base centre, with the basin. Reach over its rim before release.
    steps[4]["goal"][:2] = [.90, -2.40]
    steps[5]["params"].update(forward_m=.40, down_m=0., roll_deg=0.)
    if wall_clearance:
        steps[4]["goal"][:2] = [1.0, -2.10]
        steps[5]["params"].update(forward_m=.30, lateral_m=-.25)


def make_goal(bank: Path, source: Path, *, preserve_grasp_carry: bool = False,
              sink_clearance: bool = False, wall_clearance: bool = False,
              carry_forward_m: float = 0.0, angled_sink: bool = False,
              return_home: bool = False, aisle_waypoint: bool = False,
              raised_home_clearance: bool = False, absolute_sink_placement: bool = False,
              candidate_index: int = 97) -> dict:
    if type(candidate_index) is not int or not 0 <= candidate_index < 100:
        raise ValueError("candidate_index must be an integer from 0 through 99")
    if absolute_sink_placement and not (raised_home_clearance and angled_sink):
        raise ValueError("absolute sink placement requires raised home clearance and angled sink")
    if raised_home_clearance and not aisle_waypoint:
        raise ValueError("raised home clearance requires the loaded-home aisle waypoint")
    if aisle_waypoint and not (return_home and wall_clearance):
        raise ValueError("aisle waypoint requires loaded return home and wall clearance")
    if return_home and preserve_grasp_carry:
        raise ValueError("return_home and preserve_grasp_carry are mutually exclusive")
    if not math.isfinite(carry_forward_m) or not 0.0 <= carry_forward_m <= .20:
        raise ValueError("carry_forward_m must be finite and between 0 and .20 m")
    if carry_forward_m and not (preserve_grasp_carry and wall_clearance):
        raise ValueError("carry extension requires preserved grasp and wall clearance")
    if angled_sink and not ((preserve_grasp_carry or return_home) and wall_clearance):
        raise ValueError("angled sink requires an explicit carry mode and wall clearance")
    # NumPy object arrays use pickle. Never deserialize an arbitrary downloaded bank.
    digest = hashlib.sha256(bank.read_bytes()).hexdigest()
    if digest != EXPECTED_BANK_SHA256:
        raise ValueError("native97 diagnostic requires the exact hash-verified public left-hand bank")
    import numpy as np
    from scipy.spatial.transform import Rotation
    from simvla.skills import grasp_tool_frame

    data = np.load(bank, allow_pickle=True).item()
    if candidate_index >= data["robot_pose"].shape[1]:
        raise ValueError("candidate_index is outside the verified bank")
    pose = data["robot_pose"][0, candidate_index, 0, :7]
    offset = np.asarray(grasp_tool_frame("aiworker")["q_offset"])
    rotation = (Rotation.from_quat(pose[3:][[1, 2, 3, 0]])
                * Rotation.from_quat(offset[[1, 2, 3, 0]]))
    if rotation.apply([1, 0, 0])[2] <= .9:
        raise ValueError("native grasp wrist-camera-up frame check failed")
    goal = json.loads(source.read_text())
    if goal.get("kitchen_num") != 813 or goal.get("task_name") != "mug_to_sink_left":
        raise ValueError("this diagnostic is only calibrated for kitchen 813 mug_to_sink_left")
    step = goal["goals"][0][1]
    if step.get("skill") != "arm.grasp" or step.get("action") != "A_l":
        raise ValueError("expected left-arm grasp at step 1")
    position = [0.996, -0.3492, 0.9681] - rotation.apply([0, 0, -0.0235])
    step["goal"] = [[*position.tolist(), *rotation.as_quat()[[3, 0, 1, 2]].tolist()]]
    if return_home:
        carry = goal["goals"][0][3]
        if carry.get("skill") != "arm.reset" or carry.get("action") != "A_l":
            raise ValueError("expected the left-arm post-lift reset at step 3")
        # Keep the actual home-reset skill, not a Cartesian carry substitute.
        # No gripper command is inserted: the preceding close remains active.
        carry["language"] = "Return the left arm home while holding the mug"
    if preserve_grasp_carry:
        # A carry is not a reset: preserve the grasp's wrist orientation and
        # retract above the counter instead of rolling the loaded wrist home.
        # This diagnostic is kitchen-813-specific, just like its grasp position.
        carry = goal["goals"][0][3]
        if carry.get("skill") != "arm.reset":
            raise ValueError("expected the post-lift reset at step 3")
        carry.update(skill="arm.pose", params={},
                     # At this counter the robot faces world +Y. Extend in
                     # that direction, not world +X (which folds the arm inward).
                     goal=[float(position[0]), -0.55 + carry_forward_m, 1.14,
                           *step["goal"][0][3:]],
                     language="Retract the mug above the counter without changing its grasp orientation")
        goal["goals"][0][5]["params"]["roll_deg"] = 0.0
    if sink_clearance or wall_clearance:
        if not (preserve_grasp_carry or return_home):
            raise ValueError("sink clearance requires an explicit carry mode")
        sink_clearance_goal(goal, wall_clearance=wall_clearance)
        # Extend the carry to unfold the arm, but keep the basin release target
        # approximately unchanged. Both remain collision-checked arm motions.
        if carry_forward_m:
            goal["goals"][0][5]["params"]["forward_m"] -= carry_forward_m
        if angled_sink:
            # Keep the basin on the left side in the robot's frame instead of
            # asking the left arm to cross its lateral joint limit. The world
            # placement offsets still use the resolver's nearest cardinal axis.
            goal["goals"][0][4]["goal"][2] = -.50
            goal["goals"][0][5]["params"].update(
                forward_m=.33-carry_forward_m, lateral_m=-.03)
    if aisle_waypoint:
        # The loaded folded wrist sweeps near the range during the direct
        # clockwise turn. Test a westward aisle detour first; this is an authored
        # kitchen-specific waypoint, NOT a general collision-free navigation claim.
        goal["goals"][0].insert(4, {
            "skill": "nav.to_prim", "action": "N_s",
            "params": {"prim_path": "/world/sink_cabinet", "which_arm": "Left", "safety": .15},
            "goal": [.80, -.88, math.pi / 2],
            "language": "Move into the aisle before turning toward the sink",
        })
        goal["run_config"]["export_groups"] = "3;4,5,6,7,8"
        if raised_home_clearance:
            # The actual symmetric home is reached first. After the aisle move,
            # raise the folded hand 20 cm without changing its orientation so
            # the held mug clears the range edge during the sink approach.
            # Position/quaternion come from the public USD's lift=-.30 home FK
            # at the authored aisle base pose (.80, -.88, pi/2). This is a
            # kitchen-specific diagnostic, not a general navigation guarantee.
            goal["goals"][0].insert(5, {
                "skill": "arm.pose", "action": "A_l", "params": {},
                "goal": [.40693794, -.55506558, 1.22299793,
                         -.52290019, .50872755, .57963543, .36303503],
                "language": "Raise the held mug above the counter edge before approaching the sink",
            })
            goal["run_config"]["export_groups"] = "3;4,5,6,7,8,9"
    if absolute_sink_placement:
        # A relative extension calibrated for the earlier extended-carry pose
        # releases beside the basin after actual home. Use an explicit world
        # target over kitchen 813's basin, preserving the folded wrist's
        # orientation at the authored sink heading. The live collision planner
        # and legacy sink-proximity success check remain authoritative. This
        # check does not establish containment inside the physical basin.
        home_q = np.asarray([-.52290019, .50872755, .57963543, .36303503])
        sink_rotation = (Rotation.from_euler("z", -.5 - math.pi / 2)
                         * Rotation.from_quat(home_q[[1, 2, 3, 0]]))
        placement = goal["goals"][0][7]
        if placement.get("skill") != "arm.bowl_place":
            raise ValueError("expected the loaded-home sink placement at step 7")
        placement.update(skill="arm.pose", params={},
                         goal=[1.685, -2.240, 1.100, *sink_rotation.as_quat()[[3, 0, 1, 2]].tolist()],
                         language="Place the held mug over the sink basin")
    goal["diagnostic_grasp_source"] = {
        "bank": bank.name, "bank_sha256": digest, "native_hand": "left", "native_index": candidate_index,
        "orientation": "BoDex pose composed with AI Worker tool frame; no extra rotation",
        "translation": "measured wrist-to-jaw offset; cylindrical contact center 65 mm above mug root",
        "accepted_episode": False,
        "preserve_grasp_carry": preserve_grasp_carry,
        "return_home": return_home,
        "sink_clearance": sink_clearance,
        "wall_clearance": wall_clearance,
        "carry_forward_m": carry_forward_m,
        "angled_sink": angled_sink,
        "aisle_waypoint": aisle_waypoint,
        "raised_home_clearance": raised_home_clearance,
        "absolute_sink_placement": absolute_sink_placement,
    }
    return goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int, default=97,
                        help="Experimental native bank index (0..99); default 97 preserves historical goals")
    parser.add_argument("--source-goal", type=Path,
                        default=REPO / "examples/goals/Isaac-Kitchen-v813a-00.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preserve-grasp-carry", action="store_true")
    parser.add_argument("--return-home", action="store_true",
                        help="Return the loaded left arm home after lift, keeping the gripper closed")
    parser.add_argument("--sink-clearance", action="store_true")
    parser.add_argument("--wall-clearance", action="store_true")
    parser.add_argument("--carry-forward-m", type=float, default=0.0)
    parser.add_argument("--angled-sink", action="store_true")
    parser.add_argument("--aisle-waypoint", action="store_true",
                        help="Experimental kitchen-813 aisle detour after loaded home")
    parser.add_argument("--raised-home-clearance", action="store_true",
                        help="Experimental 20-cm clearance lift after actual home and aisle motion")
    parser.add_argument("--absolute-sink-placement", action="store_true",
                        help="Experimental world-frame basin release pose for the raised-home route")
    args = parser.parse_args()
    goal = make_goal(args.bank, args.source_goal, preserve_grasp_carry=args.preserve_grasp_carry,
                     sink_clearance=args.sink_clearance, wall_clearance=args.wall_clearance,
                     carry_forward_m=args.carry_forward_m, angled_sink=args.angled_sink,
                     return_home=args.return_home, aisle_waypoint=args.aisle_waypoint,
                     raised_home_clearance=args.raised_home_clearance,
                     absolute_sink_placement=args.absolute_sink_placement,
                     candidate_index=args.candidate_index)
    with args.output.open("x") as stream:
        json.dump(goal, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote diagnostic only (not a successful demonstration): {args.output}")


if __name__ == "__main__":
    main()
