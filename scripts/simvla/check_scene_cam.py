"""Does scene_cam actually SEE the task, or is it filming a wall?

Two full GPU runs have been spent on this camera -- once from a hand-written quaternion, once
from an eye placed outside the room -- and each cost ~20 minutes of queue plus thousands of
useless frames before anyone looked at a single PNG. A third failure was caught on the ground:
kitchen 1218's push-chair task, where the pose is BOTH outside the room and aimed at the counter
run while the table, both chairs and the robot are elsewhere, so every video of that task had to
be shot from the robot's own fisheye instead. All three RENDER FINE -- a camera inside a wall,
below the floor or facing away produces a plausible picture of the wrong thing, and nothing but
looking catches it.

So this does not eyeball coordinates. It measures the room and the task's own prims off the USD
and answers three questions the numbers alone never do:

  1. is the EYE inside the room, and not inside a cabinet? (the 2nd failure above)
  2. does every corner of every prim that has to be in shot project inside the image? (the 3rd)
  3. is the line from the eye to those prims CLEAR of every other prim in the stage? -- a camera
     can hold something in its frustum and still see nothing but the wall cabinet in front of it.

Isaac Lab's convention="world" camera looks along its own +X with +Y left and +Z up, so a point
is visible when x_cam > 0 and its normalised image coordinates both fall inside [-1, 1].

    python check_scene_cam.py                          # kitchen 1400's pose against 1400's shell
    python check_scene_cam.py --kitchen <path.usd>     # that pose against ANY kitchen's shell
    python check_scene_cam.py --num 1218 --kitchen <path.usd> \\
        --see table,chair_0,chair_1 --moved chair_0=0.456,-2.29 --moved chair_1=1.2519,-2.20

--num takes the pose from env_cfg_emit.scene_cam_values, which is the SAME function the emitter
bakes into the task file, so this can no longer drift from what was emitted. Without it, the
corpus default (kitchen 1400's) is checked -- which is what every un-overridden kitchen gets.

THE DEFAULT POSE'S EYE IS FIXED WHILE KITCHENS ARE NOT. Kitchen 1215's interior is x in
(-0.593, 2.899) and kitchen 1218's is x in (-0.855, 2.938); the default eye at x = -1.25 is
behind wall__x in both. Pass --kitchen for the kitchen you are actually about to run.
"""

from __future__ import annotations

import argparse
import math

FOCAL_MM = 14.0
H_APERTURE_MM = 20.955
WIDTH, HEIGHT = 640, 480
V_APERTURE_MM = H_APERTURE_MM * HEIGHT / WIDTH

#: Kitchen 1400's room shell, measured off the USD. An eye outside this is behind a wall.
ROOM_X = (-1.60, 2.26)
ROOM_Y = (-3.12, 0.41)
CEILING_Z = 2.65

#: What kitchen 1400's shot has to contain, measured off the USD. Used only when no --kitchen
#: prims are named: these are POINTS, and a point test cannot see occlusion.
TARGETS = {
    "fridge door (shut)":   (0.00, -0.36, 1.00),
    "door handle":          (-0.27, -0.37, 0.98),
    "hinge":                (0.34, -0.36, 1.00),
    "door tip at 90 deg":   (0.34, -0.98, 1.00),
    "robot base, parked":   (-0.27, -0.85, 0.50),
    "robot base, retreated":(0.30, -1.60, 0.50),
}

#: The prims a kitchen is checked against when --see is not given: whatever of these the stage
#: has. Named rather than "everything" because most of a kitchen is scenery -- the question is
#: whether the TASK is in shot.
SEE_DEFAULTS = ("table", "chair_0", "chair_1")

#: The room shell's four walls. Not obstacles: they are the edge of the room, and a sightline
#: that ends on the far wall behind the target is not occluded by it.
SHELL_NAMES = ("wall_x", "wall__x", "wall_y", "wall__y")

#: How far inside the shell the eye must sit, and how far clear of any furniture box. A camera
#: 1 cm off a wall is a camera whose near plane is inside it on the next rebuild.
EYE_CLEARANCE_M = 0.10

#: Fraction of the half-frame a target must stay inside. 1.0 is the frame edge, where a prim is
#: half out of shot and the framing has no room for the chair to overshoot or the robot to
#: swing wide.
FRAME_MARGIN = 0.92


def pose_for(kitchen_num=None):
    """(pos, rot) for `kitchen_num`, from the same function the emitter bakes into the task file.

    None means the corpus default -- what every kitchen with no entry in SCENE_CAM_AIMS is
    emitted with. Imported lazily so `--help` and the pure-geometry helpers below stay importable
    without env_cfg_emit's kitchen_build/trimesh dependency chain.
    """
    import env_cfg_emit

    if kitchen_num is None:
        return env_cfg_emit.SCENE_CAM_DEFAULT_POS, env_cfg_emit.SCENE_CAM_DEFAULT_ROT, None
    values = env_cfg_emit.scene_cam_values(kitchen_num)
    return values["pos"], values["rot"], values["aim"]


def _matrix(q):
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def project(point, pos, rot, focal_mm=FOCAL_MM):
    """(u, v, depth) in normalised image coords, where |u|,|v| <= 1 is inside the frame."""
    m = _matrix(rot)
    d = [point[i] - pos[i] for i in range(3)]
    # Columns of m are the camera's axes in world, so m^T maps world -> camera.
    cam = [sum(m[r][c] * d[r] for r in range(3)) for c in range(3)]
    fwd, left, up = cam
    if fwd <= 1e-6:
        return None, None, fwd
    u = (-left / fwd) * focal_mm / (H_APERTURE_MM / 2.0)
    v = (up / fwd) * focal_mm / (V_APERTURE_MM / 2.0)
    return u, v, fwd


def box_corners(box):
    """The 8 corners of ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
    lo, hi = box
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def box_samples(box, heights=None):
    """Points on a box worth ray-testing: a 3x3 grid in xy at its middle and top height.

    Corners alone under-report occlusion -- a wall cabinet can hide a chair's middle while
    leaving its corners visible. The FLOOR LEVEL is deliberately not sampled: everything in a
    kitchen stands on the floor, so at z = 0 every object in front of another occludes it, and a
    check that counted that would reject every camera position there is. What decides whether an
    object reads on video is its identifying half -- a tabletop, a chair's seat and backrest, a
    robot's torso and head -- which is what these two heights are.
    """
    lo, hi = box
    cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    if heights is None:
        heights = ((lo[2] + hi[2]) / 2.0, hi[2] - 0.01)
    return [(x, y, z)
            for x in (lo[0], cx, hi[0]) for y in (lo[1], cy, hi[1]) for z in heights]


def point_in_box(box, p, pad=0.0):
    lo, hi = box
    return all(lo[i] - pad <= p[i] <= hi[i] + pad for i in range(3))


def segment_hits_box(p0, p1, box, pad=0.0):
    """Slab test: does the SEGMENT p0->p1 pass through `box` (grown by `pad`)?

    Segment and not ray: a target behind the box's far side is occluded, a box behind the TARGET
    is not, and a ray test cannot tell those apart.
    """
    lo, hi = box
    t0, t1 = 0.0, 1.0
    for i in range(3):
        d = p1[i] - p0[i]
        a, b = lo[i] - pad, hi[i] + pad
        if abs(d) < 1e-12:
            if p0[i] < a or p0[i] > b:
                return False
        else:
            ta, tb = (a - p0[i]) / d, (b - p0[i]) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


def visibility(pos, rot, box, obstacles, focal_mm=FOCAL_MM, margin=FRAME_MARGIN):
    """(framed, unoccluded, worst_u, worst_v) for one target box.

    `framed` is the fraction of the box's CORNERS inside the frame -- the box is fully in shot
    only at 1.0, and worst_u/worst_v say by how much it misses. `unoccluded` is the fraction of
    its sample points with a clear line to the eye. `obstacles` is {name: box} and must already
    have the target itself (and any other pose of the same prim) removed.
    """
    corners = box_corners(box)
    inside = 0
    worst_u = worst_v = 0.0
    for p in corners:
        u, v, _d = project(p, pos, rot, focal_mm)
        if u is None:
            worst_u = worst_v = float("inf")
            continue
        worst_u, worst_v = max(worst_u, abs(u)), max(worst_v, abs(v))
        inside += abs(u) <= margin and abs(v) <= margin
    samples = box_samples(box)
    clear = sum(
        0 if any(segment_hits_box(pos, p, o, pad=-0.01) for o in obstacles.values()) else 1
        for p in samples
    )
    return inside / len(corners), clear / len(samples), worst_u, worst_v


def stage_boxes(usd_path):
    """{prim_name: ((min_x,min_y,min_z), (max_x,max_y,max_z))} for every depth-3 prim.

    Traverses instance proxies because these stages are made instanceable, and re-execs into an
    interpreter that can import pxr when the current one cannot.
    """
    import importlib.util, os, sys
    from pathlib import Path

    if importlib.util.find_spec("pxr") is None:
        if os.environ.get("_CSC_REEXEC"):
            raise SystemExit("pxr not importable after re-exec")
        spec = importlib.util.find_spec("isaacsim")
        libs = next((c for root in (spec.submodule_search_locations if spec else [])
                     for c in sorted(Path(root).glob("extscache/omni.usd.libs-*"))
                     if (c / "pxr").is_dir() and (c / "bin").is_dir()), None)
        if libs is None:
            raise SystemExit("no isaacsim here; run under conda -n env_isaaclab")
        env = dict(os.environ); env["_CSC_REEXEC"] = "1"
        env["PYTHONPATH"] = os.pathsep.join([str(libs), env.get("PYTHONPATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"),
                                                  env.get("LD_LIBRARY_PATH", "")])
        sys.stdout.flush()
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit(f"{usd_path}: could not open")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    boxes = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()).count("/") != 2:
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        boxes[prim.GetName()] = (tuple(round(v, 4) for v in r.GetMin()),
                                 tuple(round(v, 4) for v in r.GetMax()))
    return boxes


def shell_from_boxes(boxes):
    """(x_range, y_range, ceiling) from the four wall prims' measured boxes.

    The shells are four walls named wall_x / wall__x / wall_y / wall__y and NO ceiling, so the
    interior is bounded by the INNER face of each wall: the far wall's min and the near wall's
    max. The "ceiling" is the LOWEST wall top -- above that the camera is over a wall, which is
    outside the room in every way that matters even though nothing stops it being placed there.
    """
    missing = set(SHELL_NAMES) - set(boxes)
    if missing:
        raise SystemExit(f"no wall prims named {sorted(missing)}; cannot measure the shell, so "
                         f"this check would be a guess")
    return ((boxes["wall__x"][1][0], boxes["wall_x"][0][0]),
            (boxes["wall__y"][1][1], boxes["wall_y"][0][1]),
            min(boxes[n][1][2] for n in SHELL_NAMES))


def shell_from_usd(usd_path):
    """Back-compat shim: (x_range, y_range, ceiling) measured off a kitchen's own wall prims."""
    return shell_from_boxes(stage_boxes(usd_path))


def moved(box, dx, dy):
    lo, hi = box
    return ((lo[0] + dx, lo[1] + dy, lo[2]), (hi[0] + dx, hi[1] + dy, hi[2]))


def check(pos, rot, room, boxes, targets, focal_mm=FOCAL_MM, log=print):
    """The whole verdict. `targets` is {label: (prim_name, box)}. Returns a problem count."""
    room_x, room_y, ceiling = room
    bad = 0

    inside = (room_x[0] + EYE_CLEARANCE_M < pos[0] < room_x[1] - EYE_CLEARANCE_M
              and room_y[0] + EYE_CLEARANCE_M < pos[1] < room_y[1] - EYE_CLEARANCE_M
              and pos[2] < ceiling)
    log(f"[cam] eye {tuple(round(v, 4) for v in pos)} inside the room: {inside}"
        f"   (x {tuple(round(v, 3) for v in room_x)}, "
        f"y {tuple(round(v, 3) for v in room_y)}, wall top {round(ceiling, 3)})")
    if not inside:
        bad += 1
        log("[cam] the eye is OUTSIDE this room -- it will film the back of a wall.")

    # An eye inside a cabinet renders that cabinet's interior, which is a plausible dark frame.
    furniture = {n: b for n, b in boxes.items() if n not in SHELL_NAMES}
    in_prim = [n for n, b in furniture.items() if point_in_box(b, pos, pad=EYE_CLEARANCE_M)]
    if in_prim:
        bad += 1
        log(f"[cam] the eye is INSIDE {sorted(in_prim)} -- it will film the inside of it.")

    hfov = 2 * math.degrees(math.atan(H_APERTURE_MM / (2 * focal_mm)))
    vfov = 2 * math.degrees(math.atan(V_APERTURE_MM / (2 * focal_mm)))
    log(f"[cam] focal {focal_mm} mm -> FOV {hfov:.0f} x {vfov:.0f} deg, "
        f"targets must stay inside {FRAME_MARGIN:.2f} of the half-frame\n")

    log(f"  {'target':<22} {'in frame':>9} {'unoccluded':>11} {'|u|max':>7} {'|v|max':>7}")
    for label, (prim, box) in targets.items():
        # Every OTHER prim occludes; the target's own prim never does, at any of its poses.
        obstacles = {n: b for n, b in furniture.items() if n != prim}
        obstacles.update({l: b for l, (p, b) in targets.items()
                          if p != prim and l != label})
        framed, clear, wu, wv = visibility(pos, rot, box, obstacles, focal_mm)
        ok = framed >= 1.0 and clear >= 0.85
        bad += not ok
        wus = "  behind" if wu == float("inf") else f"{wu:7.2f}"
        wvs = "  behind" if wv == float("inf") else f"{wv:7.2f}"
        log(f"  {label:<22} {framed:9.2f} {clear:11.2f} {wus} {wvs}   {'' if ok else 'NO'}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", default="",
                    help="kitchen number: take scene_cam's pose from env_cfg_emit."
                         "scene_cam_values(num), the same function the emitter bakes into the "
                         "task file. Omitted, the corpus default pose is checked.")
    ap.add_argument("--kitchen", default="",
                    help="measure the room shell AND the target prims off THIS kitchen USD. "
                         "Without it only kitchen 1400's hard-coded shell and points are used, "
                         "and no occlusion can be tested.")
    ap.add_argument("--see", default="",
                    help=f"comma-separated prim names that must be in shot "
                         f"(default: whichever of {','.join(SEE_DEFAULTS)} the stage has)")
    ap.add_argument("--moved", action="append", default=[], metavar="PRIM=X,Y",
                    help="also require PRIM to be in shot with its box translated so its centre "
                         "is at (X, Y) -- a chair at its pushed-in pose, say. Repeatable.")
    ap.add_argument("--focal", type=float, default=FOCAL_MM,
                    help=f"lens, mm (default {FOCAL_MM}, the template's)")
    args = ap.parse_args()

    pos, rot, aim = pose_for(args.num or None)
    if args.num:
        print(f"[cam] kitchen {args.num}: pose from env_cfg_emit.scene_cam_values"
              f"{'' if aim is None else f' -- look-at {aim}'}")
    if aim is None:
        print("[cam] this is the CORPUS DEFAULT pose (kitchen 1400's). No SCENE_CAM_AIMS entry.")

    if not args.kitchen:
        print("[cam] no --kitchen: kitchen 1400's shell and points, no occlusion test\n")
        room = (ROOM_X, ROOM_Y, CEILING_Z)
        bad = 0 if (room[0][0] < pos[0] < room[0][1] and room[1][0] < pos[1] < room[1][1]
                    and pos[2] < room[2]) else 1
        print(f"  {'point':<24} {'u':>7} {'v':>7} {'depth':>7}   in frame")
        for name, p in TARGETS.items():
            u, v, depth = project(p, pos, rot, args.focal)
            if u is None:
                print(f"  {name:<24} {'--':>7} {'--':>7} {depth:7.2f}   BEHIND CAMERA")
                bad += 1
                continue
            ok = abs(u) <= 1.0 and abs(v) <= 1.0
            bad += not ok
            print(f"  {name:<24} {u:7.2f} {v:7.2f} {depth:7.2f}   {'yes' if ok else 'NO'}")
    else:
        boxes = stage_boxes(args.kitchen)
        room = shell_from_boxes(boxes)
        print(f"[cam] shell and {len(boxes)} prims measured off {args.kitchen}")
        wanted = ([s.strip() for s in args.see.split(",") if s.strip()]
                  or [n for n in SEE_DEFAULTS if n in boxes])
        if not wanted:
            raise SystemExit(
                f"none of {SEE_DEFAULTS} is in this stage and --see was not given; there is "
                f"nothing to check the camera against, and 'no problems' would be a lie")
        targets = {}
        for name in wanted:
            if name not in boxes:
                raise SystemExit(f"--see names {name!r}, which is not a prim in {args.kitchen}")
            targets[name] = (name, boxes[name])
        for spec in args.moved:
            name, _, xy = spec.partition("=")
            name = name.strip()
            if name not in boxes:
                raise SystemExit(f"--moved names {name!r}, which is not a prim in {args.kitchen}")
            try:
                mx, my = (float(v) for v in xy.split(","))
            except ValueError:
                raise SystemExit(f"--moved {spec!r}: expected PRIM=X,Y")
            lo, hi = boxes[name]
            targets[f"{name}@{mx},{my}"] = (
                name, moved(boxes[name], mx - (lo[0] + hi[0]) / 2, my - (lo[1] + hi[1]) / 2))
        bad = check(pos, rot, room, boxes, targets, args.focal)

    print()
    if bad:
        print(f"[cam] FAIL: {bad} problem(s). Do not spend a GPU run on this pose.")
    else:
        print("[cam] OK: the eye is inside the room and every target is in shot and unoccluded.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
