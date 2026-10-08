"""Conservative carried-object geometry for planning, never a physics attachment."""
from itertools import product
import math

PAYLOAD_LINK = "simvla_carried_object"


def box_sphere_cover(lower, upper, padding=0.003):
    """Eight spheres cover every point of the box, including its corners."""
    if (len(lower) != 3 or len(upper) != 3
            or not all(math.isfinite(float(x)) for x in [*lower, *upper, padding])
            or padding < 0 or any(hi <= lo for lo, hi in zip(lower, upper))):
        raise ValueError("payload bounds must be finite, nondegenerate and ordered")
    half_cell = [(hi - lo) / 4 for lo, hi in zip(lower, upper)]
    radius = math.sqrt(sum(x * x for x in half_cell)) + padding
    axes = [[lo + h, hi - h] for lo, hi, h in zip(lower, upper, half_cell)]
    return [[*center, radius] for center in product(*axes)]


def configure_payload_proxy(robot_cfg):
    """Reserve graph-stable spheres on the tool; only intentional hand contact is ignored."""
    kin = robot_cfg["kinematics"]
    if PAYLOAD_LINK in kin["collision_link_names"]:
        raise ValueError("payload proxy is already configured")
    tool = kin["ee_link"]
    hand = [name for name in kin["collision_link_names"] if name.startswith("gripper_l_")]
    if tool != "ee_link2" or not hand:
        raise ValueError("payload proxy currently requires the AI Worker left-hand model")
    kin["collision_link_names"].append(PAYLOAD_LINK)
    kin.setdefault("extra_links", {})[PAYLOAD_LINK] = {
        "parent_link_name": tool, "link_name": PAYLOAD_LINK,
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
        "joint_type": "FIXED", "joint_name": "simvla_payload_joint",
    }
    kin.setdefault("extra_collision_spheres", {})[PAYLOAD_LINK] = 8
    kin.setdefault("self_collision_ignore", {})[PAYLOAD_LINK] = hand
    kin.setdefault("self_collision_buffer", {})[PAYLOAD_LINK] = 0.0
