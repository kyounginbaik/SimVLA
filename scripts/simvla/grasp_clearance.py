"""Make an authored grasp reachable in a world cuRobo can actually see.

WHY THIS EXISTS. plan_arm_grasp takes BODex's contact points on the object and turns them into
ee_link1 targets. It asks nothing about what the object is STANDING ON. While cuRobo planned in an
empty world (every obstacle re-expressed relative to a robot at 1e10 m) that cost nothing, because
nothing was ever refused. With the world switched on it is the whole failure: on kitchen 1201, ten
of the fifteen authored candidates put the gripper's own collision spheres INSIDE the kitchen at
the goal pose -- median -15.0 mm, worst -49.5 mm -- so two thirds of the executor's uniform draws
are unplannable by construction, and Anubis went 4/32 and 2/32 sustained lifts to 0/32.

The number that matters is NOT the goal origin's height above the counter. That is +4.2 mm at
worst and reads like a near miss. The gripper is not a point: on Anubis the palm sphere sits
0.1096 m behind the tool origin with a 0.040 m radius (buffer included) and the jaws span
[-0.045, +0.050] along the tool axis, so a target 4 mm clear of the slab can have 49 mm of gripper
inside it. Everything here is measured on the SPHERES, in the frame and with the buffer cuRobo
itself uses.

WHAT IT DOES, and why not the other two options.

  * Retract along the candidate's own -approach axis. MEASURED AND REJECTED: 5/15 -> 6/15 at 10 cm
    for Anubis, 3/15 -> 3/15 for RB-Y1, and by 4 cm the jaws are off the object. It cannot work,
    because the offending approaches are horizontal or point UP -- three Anubis candidates approach
    at +0.85..+0.99 in world z, i.e. the hand reaching up out of the closed base cabinet. Backing
    away along such an axis does not move the hand off the countertop.
  * A pre-grasp standoff waypoint. Does not apply: the GOAL pose itself is the pose in collision.
    An extra waypoint in front of it changes nothing about whether cuRobo will accept the goal.
  * Raise vertically, by the smallest amount that clears, and drop what still does not clear.
    MEASURED: Anubis 5/15 -> 12/15 at 4 cm, RB-Y1 3/15 -> 12/15 at 8 cm, jaws still on the object.
    This is the mechanism.

WHAT IT COSTS. The grasp moves UP the object -- by construction only as far as it must, and never
past the point where the jaws would leave the object's own bounding box. For a tall object (the
1201 mug is 0.152 m) that is free; for a short one the cap bites first and the mechanism degrades
into a pure filter, which is the honest answer for an object whose only grasps intersect the thing
it is standing on. Candidates that cannot be cleared at any allowed height are DROPPED, so the
candidate list shrinks: on 1201 Anubis loses the three that approach from inside the cabinet.

ASSUMPTIONS, stated rather than hidden:
  * Over the raise band the object is treated as prismatic -- the same grasp orientation is assumed
    to remain valid a few centimetres higher up the body. True for the mug/cup/bottle bodies this
    corpus grasps; the cap and the object-bbox test keep the error bounded.
  * Leaf-prim world AABBs stand in for cuRobo's obstacle geometry. That is CONSERVATIVE for meshes
    and rotated cubes (an AABB is never smaller than what it bounds), so this filter can drop a
    candidate cuRobo would have taken, but cannot keep one it refuses. The dominant obstacle here,
    the countertop slab, is axis aligned, so for it the proxy is exact.

Pure python/numpy/yaml/xml: no pxr, no torch, no Omniverse. The caller supplies the scene AABBs,
because reading them needs the USD stage.

Run: cd scripts/simvla && pytest test_grasp_clearance.py -v
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

#: Metres of clearance every tool collision sphere must have from the scene at the authored pose.
#: It must EXCEED cuRobo's collision_activation_distance -- 1e-3, set at simvla_video.py's two
#: MotionGenConfig.load_from_robot_config calls -- because at exactly the activation distance the
#: pose is on the boundary and the planner may still refuse it. The rest is margin for the AABB
#: proxy above and for the few millimetres the object settles under gravity before the arm plans.
#: SIMVLA_GRASP_CLEARANCE overrides in metres; 0 disables the whole mechanism.
REQUIRED_CLEARANCE_M = 0.005

#: Metres the grasp may be raised. A candidate needing more than this is not a grasp that wants a
#: nudge, it is a grasp for a different scene -- dropped rather than dragged up the object. The
#: object's own bounding box caps it further, and usually first.
MAX_RAISE_M = 0.06

#: Metres per step of the raise search. Clearance is NOT monotone in the raise (going up can meet
#: an overhead cabinet), so the search is a scan for the SMALLEST clearing height, not a bisection.
RAISE_STEP_M = 0.002


def required_clearance() -> float:
    """The configured clearance, or 0.0 when the mechanism is switched off."""
    v = os.environ.get("SIMVLA_GRASP_CLEARANCE")
    return REQUIRED_CLEARANCE_M if v is None else float(v)


# -----------------------------------------------------------------------------------------------
# The tool's collision spheres, in the ee_link1 frame the goal pose names.
# -----------------------------------------------------------------------------------------------
def _rpy_to_R(rpy):
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return (np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))


def quat_wxyz_to_R(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _abs_path(p, repo_root):
    p = os.path.expandvars(p)
    if "$" in p:
        raise KeyError(f"unset variable in cuRobo config path: {p}")
    return p if os.path.isabs(p) else os.path.join(repo_root, p)


def _sphere_path(path, assets_root, repo_root):
    """Resolve a sphere file from either the source-tree or portable bundle layout.

    Checked-in cuRobo configs spell this as ``source/isaaclab_assets/data/...`` while the external
    bundle roots that same data at ``assets/``. Prefer the portable asset root, but keep the source
    checkout fallback for users with the in-tree assets layout.
    """
    path = os.path.expandvars(path)
    if "$" in path:
        raise KeyError(f"unset variable in cuRobo sphere path: {path}")
    if os.path.isabs(path):
        return path

    asset_prefix = os.path.join("source", "isaaclab_assets", "data")
    candidates = []
    if path == asset_prefix or path.startswith(asset_prefix + os.sep):
        stripped = path[len(asset_prefix):].lstrip(os.sep)
        candidates.extend((os.path.join(assets_root, stripped), os.path.join(repo_root, path)))
    else:
        candidates.extend((os.path.join(assets_root, path), os.path.join(repo_root, path)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"cuRobo sphere file {path!r} not found; checked: {', '.join(candidates)}")


def tool_spheres(robot: str, arm: str, repo_root: str):
    """The collision spheres RIGIDLY attached to `robot`'s ee_link1, in that link's frame.

    Returns (centres (N,3), radii (N,) with collision_sphere_buffer already added, jaw mask (N,)).
    `jaw mask` marks the spheres on the two fingers -- identified as the children of the LOCKED
    PRISMATIC joints, which is what a parallel jaw is in both robots' cuRobo configs, rather than
    by matching link names.

    Links reached through a free joint are excluded: their pose depends on the arm configuration,
    which an authoring-time check does not have. Those are the upper-arm links, and they are not
    what a grasp pose puts against a countertop.
    """
    assets_root = os.environ.get(
        "SIMVLA_ASSETS_DIR", os.path.join(repo_root, "source/isaaclab_assets/data")
    )
    # Match the collector: AI Worker and RB-Y1 planner geometry is versioned in
    # this checkout. The asset bundle can contain older sphere approximations.
    checked_in_cfg = os.path.join(repo_root, "configs/curobo/robot", f"{robot}_{arm}_arm.yml")
    cfg = (checked_in_cfg if robot in {"aiworker", "rby1"} and os.path.isfile(checked_in_cfg)
           else os.path.join(assets_root, "curobo/robot", f"{robot}_{arm}_arm.yml"))
    if not os.path.exists(cfg):
        raise FileNotFoundError(
            f"no cuRobo config {cfg}. grasp clearance needs the same tool geometry the planner "
            f"uses; add the config or set SIMVLA_GRASP_CLEARANCE=0 to author without the check."
        )
    kin = yaml.safe_load(open(cfg))["robot_cfg"]["kinematics"]
    buf = float(kin.get("collision_sphere_buffer") or 0.0)
    spheres = kin["collision_spheres"]
    if isinstance(spheres, str):
        spheres = yaml.safe_load(open(_sphere_path(spheres, assets_root, repo_root)))["collision_spheres"]
    lock = kin.get("lock_joints") or {}
    ee = kin["ee_link"]

    model_root = os.environ.get("SIMVLA_ROBOT_MODELS_DIR", repo_root)
    urdf_path = (os.environ.get("SIMVLA_AIWORKER_URDF_PATH") if robot == "aiworker" else None)
    root = ET.parse(os.path.expanduser(urdf_path) if urdf_path
                    else _abs_path(kin["urdf_path"], model_root)).getroot()
    up = {}
    for j in root.findall("joint"):
        o, ax = j.find("origin"), j.find("axis")
        up[j.find("child").get("link")] = dict(
            name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"),
            xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
            if o is not None else np.zeros(3),
            rpy=np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()])
            if o is not None else np.zeros(3),
            axis=np.array([float(v) for v in ax.get("xyz").split()])
            if ax is not None else np.zeros(3))

    jaw_links = {c for c, j in up.items() if j["type"] == "prismatic" and j["name"] in lock}

    def step(link):
        """(parent, R, t, is_rigid) for link -> parent, at the locked joint value."""
        j = up[link]
        R, t = _rpy_to_R(j["rpy"]), j["xyz"].copy()
        q = float(lock.get(j["name"], 0.0))
        if j["type"] == "prismatic":
            t = t + R @ (j["axis"] * q)
        elif j["type"] in ("revolute", "continuous") and abs(q) > 1e-12:
            a = j["axis"] / np.linalg.norm(j["axis"])
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = R @ (np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K)
        return j["parent"], R, t, (j["type"] == "fixed" or j["name"] in lock)

    def to_root(link):
        chain, cur = [link], link
        while cur in up:
            cur = up[cur]["parent"]
            chain.append(cur)
        return chain

    ee_chain = to_root(ee)

    def rel_to_ee(link):
        chain = to_root(link)
        anc = next((a for a in chain if a in ee_chain), None)
        if anc is None:
            return None
        R, t, cur, rigid = np.eye(3), np.zeros(3), link, True
        while cur != anc:
            cur, Rj, tj, r = step(cur)
            R, t, rigid = Rj @ R, Rj @ t + tj, rigid and r
        Re, te, cur = np.eye(3), np.zeros(3), ee
        while cur != anc:
            cur, Rj, tj, r = step(cur)
            Re, te, rigid = Rj @ Re, Rj @ te + tj, rigid and r
        return (Re.T @ R, Re.T @ (t - te)) if rigid else None

    cen, rad, jaw = [], [], []
    for link in kin["collision_link_names"]:
        rt = rel_to_ee(link)
        if rt is None or link not in spheres:
            continue
        R, t = rt
        for s in spheres[link]:
            cen.append(R @ np.array(s["center"], dtype=float) + t)
            rad.append(float(s["radius"]) + buf)
            jaw.append(link in jaw_links)
    if not cen:
        raise ValueError(f"{robot}/{arm}: no collision spheres rigidly attached to {ee}")
    return np.asarray(cen), np.asarray(rad), np.asarray(jaw, dtype=bool)


# -----------------------------------------------------------------------------------------------
# The raise-and-filter itself.
# -----------------------------------------------------------------------------------------------
def sphere_clearance(centres_w, radii, box_lo, box_hi):
    """Smallest (distance from a sphere's surface to a box) over every sphere and every box.

    Negative means penetration, measured to the nearest face -- the same sign convention cuRobo's
    signed distance uses, so it can be compared against collision_activation_distance directly.
    """
    if len(box_lo) == 0:
        return float("inf")
    p = centres_w[:, None, :]
    out = np.maximum(np.maximum(box_lo[None] - p, p - box_hi[None]), 0.0)
    dist = np.linalg.norm(out, axis=-1)
    inside = dist == 0.0
    if inside.any():
        depth = np.minimum(p - box_lo[None], box_hi[None] - p).min(-1)
        dist = np.where(inside, -np.maximum(depth, 0.0), dist)
    return float((dist - radii[:, None]).min())


def clear_grasps(poses, spheres, box_lo, box_hi, obj_lo, obj_hi,
                 clearance=None, max_raise=MAX_RAISE_M, step=RAISE_STEP_M):
    """Raise each authored grasp by the least it needs to clear the scene; drop what cannot clear.

    poses    -- iterable of 7-float [x,y,z,qw,qx,qy,qz] ee_link1 targets, world/env frame.
    spheres  -- (centres, radii, jaw_mask) from tool_spheres(), in the ee_link1 frame.
    box_lo/hi-- (B,3) world AABBs of the scene's colliders, WITHOUT the grasp target itself.
    obj_lo/hi-- (3,) world AABB of the grasp target, which caps the raise: the point the jaws close
                on must stay inside it, or the hand has been lifted off the object.

    Returns (kept_poses, notes) where notes is one dict per INPUT pose:
    {"index", "clearance_before", "raise", "clearance_after", "kept"}.
    """
    clearance = required_clearance() if clearance is None else clearance
    cen, rad, jaw = spheres
    pinch = cen[jaw].mean(axis=0) if jaw.any() else cen.mean(axis=0)
    box_lo, box_hi = np.asarray(box_lo, dtype=float), np.asarray(box_hi, dtype=float)
    obj_lo, obj_hi = np.asarray(obj_lo, dtype=float), np.asarray(obj_hi, dtype=float)

    kept, notes = [], []
    for i, pose in enumerate(poses):
        pose = [float(v) for v in pose]
        p0 = np.asarray(pose[:3])
        Rm = quat_wxyz_to_R(pose[3:7])
        world = (Rm @ cen.T).T
        pinch_z0 = float((Rm @ pinch)[2] + p0[2])
        # As far as the jaws may travel up the object and still be on it.
        cap = min(max_raise, max(0.0, float(obj_hi[2]) - pinch_z0))

        before = sphere_clearance(world + p0, rad, box_lo, box_hi)
        chosen, after = None, before
        d = 0.0
        while True:
            c = sphere_clearance(world + p0 + np.array([0.0, 0.0, d]), rad, box_lo, box_hi)
            if c >= clearance:
                chosen, after = d, c
                break
            if d >= cap:
                break
            d = min(d + step, cap)
        notes.append({"index": i, "clearance_before": before, "raise": chosen,
                      "clearance_after": after if chosen is not None else before,
                      "kept": chosen is not None})
        if chosen is not None:
            kept.append([pose[0], pose[1], pose[2] + chosen] + pose[3:7])
    return kept, notes


def summarize(notes) -> str:
    """One line for the emit log: what the check saw and what it did."""
    n = len(notes)
    if not n:
        return "grasp clearance: no candidates"
    before = sorted(x["clearance_before"] for x in notes)
    kept = [x for x in notes if x["kept"]]
    raises = sorted(x["raise"] for x in kept)
    ok0 = sum(1 for x in notes if x["raise"] == 0.0)
    return (f"grasp clearance: {n} authored, {ok0} already clear, {len(kept)} kept, "
            f"{n - len(kept)} dropped | clearance before "
            f"min={before[0]:+.4f} med={before[n // 2]:+.4f} max={before[-1]:+.4f} m | raise "
            + (f"med={raises[len(raises) // 2]:.3f} max={raises[-1]:.3f} m" if raises else "n/a"))
