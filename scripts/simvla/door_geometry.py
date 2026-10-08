"""SimVLA: measure a hinged door — hinge, handle, radius, and which side the hinge is on.

IMPORTABLE WITHOUT pxr; THE MEASUREMENT FUNCTIONS NEED IT. No omni, no isaaclab, no torch, no
scene_synthesizer, and pxr itself is imported lazily rather than at module scope. That matters for
two different readers of this module: test_door_geometry.py wants pxr (a plain `usd-core` install)
so it can measure a real stage in about a second, instead of paying test_kitchen_usd_load.py's
53-second Isaac Sim boot; fridge_template.arm_for() wants only the pure mirror logic
(handle_side / arm_for_handle / arm_for_hinge / _OPPOSITE), which is dict lookups and trig with no
USD dependency at all, and must keep importing under plain env_isaaclab, which has no pxr until
Isaac Sim boots (see _require_pxr below). A caller that needs the USD-measuring functions
(measure_door, max_base_sweep_deg, the __main__ CLI) without pxr available gets a clear ImportError
naming which interpreter has it, not an AttributeError on None three calls deep.

WHY IT EXISTS. The hinge-to-handle distance of the refrigerator door — 0.717 m on the kitchen
1300/1400 asset — is written as the literal `0.7` in three unrelated functions:

    simvla_data_generator.py:1195   refrigerator_trans = 0.7        (where nav parks the base)
    simvla_data_generator.py:1933   vec3d[1] + 0.7 - 0.06           (where the hand reaches)
    and, before this module, the arc radius would have been a fourth.

Three copies of one physical quantity, none of them measured, none of them a fact about the asset
in front of them. The grasp one is provably wrong even on the asset it was tuned for: it reads the
handle prim's ExtractTranslation(), which returns the DOOR XFORM'S PIVOT — the hinge, at floor
level, (1.0884, -3.0351, 0.0) — and then walks 0.7 m from there, landing at y = -2.3951 against a
handle bar spanning y ∈ [-2.3568, -2.2868]. That is 3.8 cm off the end of the bar.

THE HINGE IS READ FROM THE JOINT, NOT FROM THE PRIM. A door Xform's pivot often sits at the hinge,
but that is a convention of one exporter, not a guarantee. The authority is the revolute joint's
`localPos0` — an anchor in body0's (the corpus's) frame — carried to world. On kitchen 1300 the two
agree to 0.3 mm, and the test asserts that agreement rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Deferred, not module-scope: `from __future__ import annotations` above already makes every
# `-> Gf.Matrix4d`-style annotation a string that's never evaluated, so the only thing a module-
# scope pxr import would buy is failing THIS import for callers who never touch a stage --
# fridge_template.arm_for() among them, under plain env_isaaclab, which has no pxr until Isaac Sim
# boots. Try once at import time so the common case (pxr present) pays nothing extra; every
# USD-touching function below calls _require_pxr() first so the failure is a named ImportError,
# not an AttributeError on None from deep inside GetPrimAtPath.
try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    _PXR_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as _exc:
    Gf = Usd = UsdGeom = UsdPhysics = None  # type: ignore[assignment]
    _PXR_IMPORT_ERROR = _exc


def _require_pxr() -> None:
    """Raise a clear, named error if pxr did not import, instead of letting a None leak into a USD
    call and surface as an opaque AttributeError. Call this first in every function that touches
    Usd/Gf/UsdGeom/UsdPhysics."""
    if Gf is None:
        raise ImportError(
            "door_geometry's USD-measuring functions need pxr, which this interpreter does not "
            "have. Use python (usd-core), or run "
            "under a booted Isaac Sim app, where env_isaaclab's own pxr becomes importable. The "
            "pure mirror logic (handle_side, arm_for_handle, arm_for_hinge) needs none of this."
        ) from _PXR_IMPORT_ERROR


class DoorGeometryError(ValueError):
    """This prim is not a measurable hinged door. Named, so a caller can skip a kitchen for it
    without also swallowing a bug in this module."""


@dataclass(frozen=True)
class DoorGeometry:
    door_path: str
    joint_path: str
    joint_name: str
    hinge_world: tuple[float, float, float]
    handle_path: str
    handle_world: tuple[float, float, float]
    #: Hinge-to-handle distance PERPENDICULAR TO THE HINGE LINE. The arc radius, and the quantity
    #: the three hardcoded 0.7s are all approximating. For a vertical hinge -- every door this
    #: module is meant for -- that is exactly the XY distance it has always reported; for a
    #: drop-down door (measurable only with require_vertical=False) it is a mostly-vertical drop,
    #: which an XY distance would have understated by half.
    radius_m: float
    limit_lower_deg: float
    limit_upper_deg: float
    #: The hinge axis as a UNIT VECTOR IN WORLD, not the `physics:axis` token. See
    #: _hinge_axis_world for why the token alone is not the axis.
    axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: True when a POSITIVE joint angle swings the door counter-clockwise seen from above, i.e.
    #: when axis_world points along world +Z. False when the joint frame is flipped.
    opens_ccw: bool = True


def _local_to_world(stage, path: str) -> Gf.Matrix4d:
    _require_pxr()
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise DoorGeometryError(f"no prim at {path!r}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _bbox_centre(stage, path: str) -> tuple[float, float, float]:
    _require_pxr()
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    return ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5)


def _revolute_joint_for(stage, door_path: str):
    """The revolute joint whose body1 is this door.

    Searched under the door's PARENT (the furniture root), because that is where the exporter puts
    joints: /world/refrigerator/door_joint is a SIBLING of /world/refrigerator/door, not a child of
    it. A search rooted at the door itself finds nothing.
    """
    _require_pxr()
    door = stage.GetPrimAtPath(door_path)
    if not door:
        raise DoorGeometryError(f"no prim at {door_path!r}")
    for prim in Usd.PrimRange(door.GetParent()):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        targets = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        if door_path in targets:
            return joint
    raise DoorGeometryError(
        f"no revolute joint under {door.GetParent().GetPath()} names {door_path!r} as its body1, "
        f"so this door has no hinge to arc about. A prismatic drawer is not a door: it travels in "
        f"a straight line, and nav.open_articulation already handles that case correctly."
    )


#: `physics:axis` tokens as vectors in the JOINT'S frame.
_AXIS_TOKENS = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _hinge_axis_world(stage, joint, body0_path: str) -> tuple[float, float, float]:
    """The hinge axis as a unit vector in WORLD.

    THE TOKEN IS NOT THE AXIS, and reading it as one is the defect this function exists to remove.
    `physics:axis` names an axis of the JOINT'S OWN FRAME, and that frame is oriented by
    `physics:localRot0` inside body0, which is itself placed by body0's local-to-world. Every door
    in this corpus authors the token "Z", so a reader that stops there calls all of them vertical.

    Measured on a freshly built kitchen, token "Z" throughout:

        refrigerator/door        localRot0 (1,0,0,0)              -> world (0, 0, +1)  vertical
        sink_cabinet/door_0_1    localRot0 (0,0,1,0)              -> world (0, 0, -1)  vertical, flipped
        dishwasher/door_0_1      localRot0 (.7071,0,.7071,0)      -> world (0, -1, 0)  HORIZONTAL
        range/door_0_1           localRot0 (.7071,0,.7071,0)      -> world (0, -1, 0)  HORIZONTAL

    The dishwasher and the oven are drop-down doors. Read as vertical they report a hinge-to-handle
    radius of 0.343 m against a true 0.682 m, a `hinge_side` that means nothing, and an arc that
    drives the base sideways for a door that opens downward.
    """
    _require_pxr()
    token = joint.GetAxisAttr().Get()
    local = _AXIS_TOKENS.get(token)
    if local is None:
        raise DoorGeometryError(
            f"{joint.GetPrim().GetPath()} has unrecognised physics:axis {token!r}"
        )
    rot = joint.GetPrim().GetAttribute("physics:localRot0").Get()
    if rot is not None:
        turned = Gf.Rotation(
            Gf.Quatd(float(rot.GetReal()), Gf.Vec3d(*[float(v) for v in rot.GetImaginary()]))
        ).TransformDir(Gf.Vec3d(*local))
    else:
        turned = Gf.Vec3d(*local)
    world = _local_to_world(stage, body0_path).TransformDir(turned).GetNormalized()
    return (world[0], world[1], world[2])


#: How far off horizontal the axis may tilt and still count as a vertical hinge. Every door
#: measured is within 1e-6 of exactly vertical or exactly flat, so this only has to separate two
#: well-separated populations, not adjudicate a marginal one.
_VERTICAL_MIN_Z = 0.9


def measure_door(stage, door_path: str, handle_name: str = "door_handle", *,
                 require_vertical: bool = True) -> DoorGeometry:
    """Measure the hinged door at `door_path`. Raises DoorGeometryError if it is not one.

    `require_vertical` refuses a drop-down door (a horizontal hinge). It defaults to True because
    every consumer of the result -- nav.open_door_arc, hinge_side/arm_for_handle, the arc clearance
    gate -- is written for a door that swings about a vertical axis, and each of them returns a
    plausible-looking number for a door that does not move that way. Pass False to REPORT a door's
    shape without arcing about it; the kitchen survey does exactly that.
    """
    _require_pxr()
    joint = _revolute_joint_for(stage, door_path)
    joint_path = str(joint.GetPrim().GetPath())

    body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
    if not body0:
        raise DoorGeometryError(f"{joint_path!r} has no body0, so its anchor has no frame")
    anchor = joint.GetLocalPos0Attr().Get()
    if anchor is None:
        raise DoorGeometryError(f"{joint_path!r} has no localPos0 anchor")
    hinge = _local_to_world(stage, body0[0]).Transform(
        Gf.Vec3d(float(anchor[0]), float(anchor[1]), float(anchor[2]))
    )

    handle_path = f"{door_path.rstrip('/')}/{handle_name}"
    if not stage.GetPrimAtPath(handle_path):
        raise DoorGeometryError(
            f"no handle at {handle_path!r}. task_bind resolves a handle role through this exact "
            f"fixed convention (f'{{parent.prim_path}}/door_handle'), so a door without one cannot "
            f"be grasped by role either."
        )
    handle = _bbox_centre(stage, handle_path)

    axis = _hinge_axis_world(stage, joint, body0[0])
    if require_vertical and abs(axis[2]) < _VERTICAL_MIN_Z:
        raise DoorGeometryError(
            f"{door_path!r} hinges about the horizontal axis "
            f"({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}) in world -- it is a DROP-DOWN door, "
            f"not a swing door, even though its physics:axis token reads "
            f"{joint.GetAxisAttr().Get()!r} (the token is in the joint's own frame; localRot0 lays "
            f"it flat). dishwasher/door_0_1 and range/door_0_1 are authored this way in every "
            f"kitchen in this corpus. Arcing a base about it would drive the robot sideways for a "
            f"door that opens downward. Pass require_vertical=False to measure it anyway."
        )

    # PERPENDICULAR TO THE HINGE LINE, not an xy distance. The two agree exactly for a vertical
    # axis (the case every existing caller is in, so no number moves), and differ by a factor of
    # two for a drop-down door, whose handle travels mostly in z.
    delta = tuple(handle[i] - hinge[i] for i in range(3))
    along = sum(delta[i] * axis[i] for i in range(3))
    radius = math.sqrt(max(sum((delta[i] - along * axis[i]) ** 2 for i in range(3)), 0.0))

    lower = joint.GetLowerLimitAttr().Get()
    upper = joint.GetUpperLimitAttr().Get()

    return DoorGeometry(
        door_path=door_path,
        joint_path=joint_path,
        joint_name=joint_path.rsplit("/", 1)[-1],
        hinge_world=(hinge[0], hinge[1], hinge[2]),
        handle_path=handle_path,
        handle_world=handle,
        radius_m=radius,
        limit_lower_deg=float(lower) if lower is not None else 0.0,
        limit_upper_deg=float(upper) if upper is not None else 0.0,
        axis_world=axis,
        opens_ccw=axis[2] > 0.0,
    )


def face_normal(stage, geom: DoorGeometry) -> tuple[tuple[float, float], tuple[float, float]]:
    """((outward unit xy), (along-the-door unit xy, away from the hinge)) for a shut door.

    FROM THE PANEL'S THIN AXIS, which is exact, and not from (handle - corpus centre) made
    perpendicular to the door, which is not. A door is a thin slab on one face of its corpus, so its
    thin axis IS the facing axis and which side of the corpus it sits on gives the sign -- the same
    derivation fridge_kitchen.fridge_front uses, and for the same reason: kitchen furniture snaps to
    the four cardinal directions, so the true normal is axis-aligned to floating-point.

    The perpendicular-to-the-hinge-line construction is off by however far the handle sits from the
    door's mid-width -- 4 degrees on a base cabinet door, which is 31 mm of base position at a
    0.45 m standoff. That is a quarter of the clearance budget, and it put the gate's measured pose
    and the planner's authored pose 35 mm apart. Caught by test_nav_to_door_handle.

    `along` still comes from hinge -> handle, which is exact: both lie on the door.
    """
    _require_pxr()
    door = stage.GetPrimAtPath(geom.door_path)
    if not door:
        raise DoorGeometryError(f"no prim at {geom.door_path!r}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                              useExtentsHint=False)

    # The SLAB, not the door group: the group's bbox includes the handle, which stands proud on the
    # very axis being measured and can make the slab look thicker than it is.
    panel, best = None, 0.0
    for child in door.GetChildren():
        if child.GetName() == geom.handle_path.rsplit("/", 1)[-1]:
            continue
        rng = cache.ComputeWorldBound(child).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        size = rng.GetMax() - rng.GetMin()
        vol = size[0] * size[1] * size[2]
        if vol > best:
            panel, best = rng, vol
    if panel is None:
        raise DoorGeometryError(f"{geom.door_path!r} has no panel to read a facing axis from")

    corpus = cache.ComputeWorldBound(door.GetParent()).ComputeAlignedRange().GetMidpoint()
    lo, hi = panel.GetMin(), panel.GetMax()
    mid = panel.GetMidpoint()
    if (hi[0] - lo[0]) <= (hi[1] - lo[1]):          # thin in x -> faces +-x
        nrm = (1.0 if mid[0] > corpus[0] else -1.0, 0.0)
    else:                                           # thin in y -> faces +-y
        nrm = (0.0, 1.0 if mid[1] > corpus[1] else -1.0)

    dx = geom.handle_world[0] - geom.hinge_world[0]
    dy = geom.handle_world[1] - geom.hinge_world[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        raise DoorGeometryError(f"{geom.door_path!r}: the handle sits on its own hinge line")
    return nrm, (dx / n, dy / n)


def hinge_side(geom: DoorGeometry, approach_yaw_rad: float) -> str:
    """Which side of the approaching robot the hinge is on: "Right" or "Left".

    `approach_yaw_rad` is the robot's heading — the direction it FACES, which is the direction
    nav.to_prim leaves it pointing. A robot at yaw 0 faces +x, so its right hand points at -y, i.e.
    (sin yaw, -cos yaw). The hinge is on that side when (hinge - handle) has a positive component
    along it.

    Measured against kitchen 1300 at yaw 0: hinge - handle = (0.076, -0.713), right = (0, -1),
    dot = +0.713 -> "Right", and therefore the LEFT arm reaches the handle (see arm_for_hinge).
    """
    right = (math.sin(approach_yaw_rad), -math.cos(approach_yaw_rad))
    delta = (geom.hinge_world[0] - geom.handle_world[0],
             geom.hinge_world[1] - geom.handle_world[1])
    return "Right" if delta[0] * right[0] + delta[1] * right[1] > 0.0 else "Left"


#: The two sides, as each other's mirror. ONE dict, so "opposite" is written down exactly once —
#: every other function here reads it rather than restating {"Right": "Left", "Left": "Right"}.
_OPPOSITE = {"Right": "Left", "Left": "Right"}


def _checked_side(side: str, what: str) -> str:
    """Validate `side` against _OPPOSITE and return it unchanged; raise DoorGeometryError otherwise.

    Not written as the dict-lookup-as-validator idiom (`_OPPOSITE[side] and side`): that relies on
    a KeyError from the lookup for the bad-input path and on Python's `and` falling through to
    `side` for the good one, which is correct but reads as a bug on first pass. A plain membership
    check says the same thing without asking the reader to notice it's leaning on truthiness.
    """
    if side not in _OPPOSITE:
        raise DoorGeometryError(f"{what} must be 'Right' or 'Left', not {side!r}")
    return side


def handle_side(geom: DoorGeometry, approach_yaw_rad: float) -> str:
    """Which side of the approaching robot the HANDLE is on: "Left" or "Right".

    The frame the rule is actually stated in. From the user: "if the door handle of the refridge is
    in the left (robot 기준) the robot should grasp with left arm and move back a bit and move right
    and back 대각선".

    The handle sits at the door's free edge, which is by construction opposite the hinge, so this
    is `hinge_side`'s mirror and is DERIVED from it rather than measured a second way. Two
    independent measurements of the same quantity can disagree; a mirror cannot.
    """
    return _OPPOSITE[hinge_side(geom, approach_yaw_rad)]


def arm_for_handle(side: str) -> str:
    """Which arm grasps a handle on the given side. The identity, and that IS the rule.

    A handle on the robot's left is reached across the body by the right arm only by fighting it;
    the left arm reaches it directly. Written as a function rather than inlined so arm_for_hinge
    can be defined through it, which is what stops the two from drifting apart.
    """
    return _checked_side(side, "handle side")


def arm_for_hinge(side: str) -> str:
    """Which arm should grasp the handle, given which side the HINGE is on.

    Unchanged behaviour: hinge Right -> Left arm. Now routed through arm_for_handle and _OPPOSITE
    so the mirror lives in one place. nav.open_door_arc's resolver derives the end-effector it
    reads from the same fact rather than taking a second, desynchronisable param.
    """
    return arm_for_handle(_OPPOSITE[_checked_side(side, "hinge side")])


def max_base_sweep_deg(stage, geom, base_xy, min_clearance_m: float = 0.35,
                       cap_deg: float = 120.0, ignore=()) -> tuple:
    """(max_sweep_deg, clearance_m, sense) the base can arc about this hinge on THIS stage.

    Measured against the BUILT kitchen -- its real walls and furniture -- rather than against the
    recipe's worst-case room model. fridge_kitchen.max_clear_sweep_deg has to assume the tightest
    room the generator can roll (ROOM_SHELL_FLUSH_M 0.02 / ROOM_SHELL_WALKWAY_M 0.20) because it
    runs before any USD exists, and that conservatism costs real opening angle: on kitchen 1400 it
    reported 65 degrees where the built stage allows 85 at the same 0.35 m margin.

    Both numbers are right for what they are. The recipe's is the gate that picks a seed safely;
    this is the one the task should actually be driven to, once the kitchen is on disk.
    """
    _require_pxr()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    skip = set(ignore) | {"refrigerator", "GroundPlane", "defaultGroundPlane", "Looks", "ground"}
    boxes = []
    world = stage.GetPrimAtPath("/world")
    for prim in world.GetChildren():
        name = prim.GetName()
        if name in skip or name.startswith("bottle"):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        if hi[2] < 0.05:                       # the floor is not an obstacle
            continue
        boxes.append(((lo[0], lo[1]), (hi[0], hi[1])))

    hx, hy = geom.hinge_world[0], geom.hinge_world[1]
    dx, dy = base_xy[0] - hx, base_xy[1] - hy

    def clearance(sweep_deg, sense):
        worst = float("inf")
        for i in range(41):
            t = math.radians(sweep_deg) * i / 40.0 * sense
            px = hx + math.cos(t) * dx - math.sin(t) * dy
            py = hy + math.sin(t) * dx + math.cos(t) * dy
            for (lo_x, lo_y), (hi_x, hi_y) in boxes:
                d = math.hypot(max(lo_x - px, 0.0, px - hi_x), max(lo_y - py, 0.0, py - hi_y))
                if d < worst:
                    worst = d
        return worst

    best = (0.0, 0.0, 1)
    for deg in range(5, int(cap_deg) + 1, 5):
        room, sense = max(((clearance(deg, s), s) for s in (1, -1)), key=lambda x: x[0])
        if room >= min_clearance_m:
            best = (float(deg), room, sense)
        else:
            break
    return best


def main() -> int:
    """CLI: measure a door and print it. This is how a new appliance gets onboarded.

        python door_geometry.py <kitchen.usd> /world/refrigerator/door
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Measure a hinged door from a USD stage.")
    parser.add_argument("usd")
    parser.add_argument("door_path")
    parser.add_argument("--handle-name", default="door_handle")
    parser.add_argument("--approach-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    # Guarded explicitly, ahead of measure_door's own guard: Usd.Stage.Open below is an argument
    # expression, evaluated before measure_door is ever entered, so without this a missing pxr
    # would surface as AttributeError: 'NoneType' object has no attribute 'Stage' instead of the
    # named ImportError.
    _require_pxr()
    geom = measure_door(Usd.Stage.Open(args.usd), args.door_path, args.handle_name)
    side = hinge_side(geom, math.radians(args.approach_yaw_deg))
    out = {**geom.__dict__, "hinge_side": side, "arm": arm_for_hinge(side)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
