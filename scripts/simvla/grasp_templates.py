"""Generate grasp-only task templates for the campaign.

Each template reduces a task to nav -> grasp/squeeze -> [close] -> lift, so a run exercises only
grasping. The three shapes mirror the two vase archetypes (templates/vase_neck.json,
templates/vase_press.json):
  single + graspable -> arm.grasp A_r (uses the type's BODex grasp data) + gripper close
  single + clutter   -> arm.squeeze A_r (geometric single-hand grasp) + gripper close
  bimanual           -> arm.squeeze A_b, no gripper close (held by friction)
Pure — no sim imports.
"""
from __future__ import annotations

import json
from pathlib import Path

from scene_spec import GRASPABLE_TYPES

_ROLES = [
    {"name": "target", "object_type": None, "articulation_with": None, "handle_of": None},
    {"name": "container", "object_type": None, "articulation_with": "drawer", "handle_of": None},
    {"name": "container_handle", "object_type": None, "articulation_with": None, "handle_of": "container"},
]

_NAV = {"skill": "nav.to_prim", "action": "N_s",
        "params": {"prim_path": "@target", "which_arm": "Right"}, "language": "Move to object"}


def _scene(obj_type: str) -> list[dict]:
    # A single distractor of a DIFFERENT type from the target, so its "<type>0" name never collides
    # with the target's (scene_spec rejects duplicate object names). mug is the default distractor;
    # when the target IS a mug, use a cup instead.
    distractor = "cup" if obj_type == "mug" else "mug"
    return [
        {"name": f"{obj_type}0", "object_type": obj_type,
         "size": {"center": 1.0, "spread": 0.02}, "placement": "dishwasher",
         "lift": {"center": 0.775, "spread": 0.025}},
        {"name": f"{distractor}0", "object_type": distractor,
         "size": {"center": 1.0, "spread": 0.05}, "placement": "island", "lift": None},
    ]


def build_grasp_template(obj_type: str, strategy: str) -> dict:
    if strategy not in ("single", "bimanual"):
        raise ValueError(f"unknown strategy {strategy!r}")

    roles = [dict(r) for r in _ROLES]
    roles[0]["object_type"] = obj_type

    if strategy == "bimanual":
        steps = [
            _NAV,
            {"skill": "arm.squeeze", "action": "A_b", "params": {"prim_path": "@target"},
             "language": "Both open hands press the object's opposite faces"},
            {"skill": "arm.reset", "action": "A_b", "params": {},
             "language": "Lift the object between both hands (no gripper close)"},
        ]
        groups = [[2]]
    else:
        grab = "arm.grasp" if obj_type in GRASPABLE_TYPES else "arm.squeeze"
        steps = [
            _NAV,
            {"skill": grab, "action": "A_r", "params": {"prim_path": "@target"},
             "language": "Right hand grasps the object"},
            {"skill": "gripper.set", "action": "G_r", "params": {"grasp": True},
             "language": "Close the gripper"},
            {"skill": "arm.reset", "action": "A_r", "params": {}, "language": "Lift the object"},
        ]
        groups = [[3]]

    return {
        "name": f"graspcamp_{obj_type}_{strategy}",
        "language": f"Grasp the {obj_type} and lift it.",
        "roles": roles,
        "steps": steps,
        "subtask_groups": groups,
        "scene": _scene(obj_type),
    }


def write_grasp_template(obj_type: str, strategy: str, out_dir: str) -> str:
    t = build_grasp_template(obj_type, strategy)
    path = Path(out_dir) / f"{t['name']}.json"
    path.write_text(json.dumps(t, indent=2))
    return str(path)
