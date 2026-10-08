"""Author-time base clearance for nav.to_prim: is the park pose clear of the furniture, and if
not, where else can the base stand?

Stdlib only (math, dataclasses), like fridge_siting: it runs under env_isaaclab on the emit node
and under plain python3 on the login node, where test_nav_clearance.py pins it against dumped
kitchen geometry. plan_nav_to_prim (simvla_data_generator.py) gathers the /world AABBs with
UsdGeom.BBoxCache and hands them in as Box records; nothing here touches Omniverse.

WHY THIS EXISTS. plan_nav_to_prim parks the base at `furniture_face - safety - wheel_r` on whichever
side of the furniture the free-space walk finds first. Neither step asks whether the base FITS there.
On kitchen 1570 the mug sits in the corner where the base cabinet meets the refrigerator, the
kitchen's only free-space band is SOUTH of the fridge, so the walk chooses the south side and the
pose it authors is inside the refrigerator's footprint (y -1.893, fridge y [-2.204, -1.423]).
Three generation runs timed out identically with the base grinding on the fridge door.

THE POLICY IS "AS USUAL UNLESS IT COLLIDES". A pose whose clearance is >= 0 is returned untouched,
byte for byte -- the proven kitchens (1550 measures +0.087 m) cannot move because of this module.
A pose that penetrates is re-sited by the same search the fridge and cabinet-door tasks use
(fridge_siting / cabinet_kitchen.pick_park_pose): enumerate candidate poses, keep the ones that
clear every obstacle, rank them. Here the candidates are the four sides of the supporting furniture
at the same standoff, with a lateral slide along the face and extra standoff off it, and the rank
is REACH FIRST -- how far ahead of the base the object sits -- because arm reach, not clearance, is
RB-Y1's binding constraint (test_object_to_plate, rby1-nav-park-too-close).

THE BOX, NOT A CIRCLE. wheel_r = 0.23 is a hand-typed stand-in for the whole base that undershoots
every real footprint (RB-Y1 reaches 0.345 m forward, 0.300 to the side). The base is checked as its
measured oriented box against axis-aligned furniture boxes by the separating-axis test, so a base
facing a counter is measured by its front and one driving past it by its flank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ----------------------------------------------------------------------------------------------
# Footprints
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Footprint:
    """The base's collision box in its own frame (x forward, y left), metres from the link origin.

    Asymmetric on purpose: the nav goal is expressed at the link origin, and the AI Worker's base
    reaches 0.403 m behind that point but only 0.225 ahead of it."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def corners(self, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
        c, s = math.cos(yaw), math.sin(yaw)
        return [(x + c * bx - s * by, y + s * bx + c * by)
                for bx, by in ((self.x_min, self.y_min), (self.x_max, self.y_min),
                               (self.x_max, self.y_max), (self.x_min, self.y_max))]

    @property
    def radius_m(self) -> float:
        """The farthest corner from the origin: what a turn in place sweeps."""
        return max(math.hypot(bx, by) for bx in (self.x_min, self.x_max)
                   for by in (self.y_min, self.y_max))


#: MEASURED 2026-09-02 from each robot's own USD, base_link, all four bbox purposes (proxy/guide/
#: render included -- anubis_simvla.usd's colliders bind to nothing under default_ alone, see
#: robot_footprint._bbox_cache):
#:   rby1      rby1m/models/rby1m/urdf/model/model.usd    x [-0.350, +0.345]  y [-0.300, +0.300]
#:   anubis    anubis_simvla.usd                          0.465 x 0.4916, centred (cabinet_kitchen)
#:   aiworker  Robots/MM/aiworker/ffw_sg2.usd             x [-0.403, +0.225]  y [-0.301, +0.301]
FOOTPRINTS: dict[str, Footprint] = {
    "rby1": Footprint(-0.350, 0.345, -0.300, 0.300),
    "anubis": Footprint(-0.2325, 0.2325, -0.2458, 0.2458),
    "aiworker": Footprint(-0.403, 0.225, -0.301, 0.301),
}


def footprint_for(robot: str) -> Footprint:
    """An unknown robot raises: a borrowed footprint certifies parks the real base does not fit."""
    if robot not in FOOTPRINTS:
        raise KeyError(f"no base footprint for robot {robot!r}; have {', '.join(sorted(FOOTPRINTS))}")
    return FOOTPRINTS[robot]


# ----------------------------------------------------------------------------------------------
# Obstacles
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """A scene prim's world-space AABB, as UsdGeom.BBoxCache.ComputeWorldBound(...).ComputeAlignedRange()
    reports it. Kitchen furniture snaps to the four cardinal directions, so the AABB is the box."""
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    z_min: float = 0.0
    z_max: float = 1.0

    def corners(self) -> list[tuple[float, float]]:
        return [(self.x_min, self.y_min), (self.x_max, self.y_min),
                (self.x_max, self.y_max), (self.x_min, self.y_max)]


#: Anything whose UNDERSIDE is below this can meet the base column. Countertops (0.911), things on
#: them (microwave 0.95) and wall cabinets (1.27) are above it. Not a per-robot base height: a
#: countertop overhangs the cabinet the base actually meets, so in plan its box CONTAINS every
#: parking spot and any clearance to it is negative -- for every robot and every layout
#: (cabinet_kitchen.floor_colliders learned the same lesson).
BASE_CLEAR_Z_M = 0.85
#: The floor itself, if it is a prim, has z_max ~0.
FLOOR_Z_M = 0.02


def floor_obstacles(boxes, exclude=()) -> list[Box]:
    """The boxes the base can hit: underside below BASE_CLEAR_Z_M, not the floor, not `exclude`."""
    return [b for b in boxes
            if b.z_min < BASE_CLEAR_Z_M and b.z_max > FLOOR_Z_M and b.name not in exclude]


#: kitchen_build.room_shell_prim_names(), copied because that module imports scene_synthesizer
#: and this one must stay stdlib. NEVER a "wall_" prefix test: wall_cabinet, wall_cabinet_0 .. _3
#: are wall-MOUNTED cabinets, furniture with a footprint on the floor plan -- and read as walls
#: they shrink the room to the strip under them. The wall_0N names are the run-time walls
#: make_walls_from_bounds adds to a kitchen that has no shell of its own.
SHELL_WALL_NAMES = frozenset({"wall_x", "wall_-x", "wall__x", "wall_y", "wall_-y", "wall__y",
                              "wall_01", "wall_02", "wall_03", "wall_04"})


def is_shell_wall(name: str) -> bool:
    return name in SHELL_WALL_NAMES


#: kitchen_env_cfg / isaaclab.simvla.utils.make_walls_from_bounds: a kitchen with no shell in its
#: USD gets four 0.1 m walls at RUN time, centred 0.2 m outside the furniture union.
RUNTIME_WALL_OFFSET_M = 0.2
RUNTIME_WALL_THICKNESS_M = 0.1


def room_interior(boxes) -> tuple[tuple[float, float, float, float], list[Box]]:
    """((x_min, y_min, x_max, y_max) the base must stay inside, the wall boxes to check against).

    With a room shell in the USD the interior is the walls' inner faces. Without one, the walls
    the env config adds at run time are synthesised here, so the check sees what the run will."""
    walls = [b for b in boxes if is_shell_wall(b.name)]
    furniture = [b for b in boxes if not is_shell_wall(b.name)]
    fx0 = min(b.x_min for b in furniture)
    fy0 = min(b.y_min for b in furniture)
    fx1 = max(b.x_max for b in furniture)
    fy1 = max(b.y_max for b in furniture)
    if walls:
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        x_lo = max((w.x_max for w in walls if w.x_max <= cx), default=-math.inf)
        x_hi = min((w.x_min for w in walls if w.x_min >= cx), default=math.inf)
        y_lo = max((w.y_max for w in walls if w.y_max <= cy), default=-math.inf)
        y_hi = min((w.y_min for w in walls if w.y_min >= cy), default=math.inf)
        return (x_lo, y_lo, x_hi, y_hi), walls
    o, t = RUNTIME_WALL_OFFSET_M, RUNTIME_WALL_THICKNESS_M / 2
    x0, y0, x1, y1 = fx0 - o, fy0 - o, fx1 + o, fy1 + o
    walls = [Box("wall_01", x0 - t, y0 - t, x1 + t, y0 + t, 0.0, 3.0),
             Box("wall_02", x1 - t, y0 - t, x1 + t, y1 + t, 0.0, 3.0),
             Box("wall_03", x0 - t, y0 - t, x0 + t, y1 + t, 0.0, 3.0),
             Box("wall_04", x0 - t, y1 - t, x1 + t, y1 + t, 0.0, 3.0)]
    return (x0 + t, y0 + t, x1 - t, y1 - t), walls


# ----------------------------------------------------------------------------------------------
# Separating-axis clearance
# ----------------------------------------------------------------------------------------------

def _separation(poly_a, poly_b, axes) -> float:
    """The widest gap over `axes` (positive: apart by that much; negative: overlap depth)."""
    best = -math.inf
    for ax, ay in axes:
        pa = [px * ax + py * ay for px, py in poly_a]
        pb = [px * ax + py * ay for px, py in poly_b]
        gap = max(min(pb) - max(pa), min(pa) - max(pb))
        if gap > best:
            best = gap
    return best


def clearance(x: float, y: float, yaw: float, footprint: Footprint, obstacles) -> tuple[float, str | None]:
    """(min separation, name of the nearest box). inf and None when there is nothing to hit.

    Four candidate axes: the world axes (the boxes' own) and the base's two. That is the full
    separating-axis set for two rectangles in the plane."""
    base = footprint.corners(x, y, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    axes = ((1.0, 0.0), (0.0, 1.0), (c, s), (-s, c))
    best, who = math.inf, None
    for ob in obstacles:
        gap = _separation(base, ob.corners(), axes)
        if gap < best:
            best, who = gap, ob.name
    return best, who


# ----------------------------------------------------------------------------------------------
# The park search
# ----------------------------------------------------------------------------------------------

#: A candidate must clear everything by this much. The nav goal band is +-0.02 m, so a pose that
#: clears by less makes contact on the runs that stop at the near edge of the band.
MIN_CLEAR_M = 0.02
#: Lateral slides along the furniture face, tried nearest-first.
SLIDES_M = (0.0, 0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20, 0.25, -0.25, 0.30, -0.30)
#: Extra standoff off the face. Each step costs reach directly, which is why it ranks last.
EXTRAS_M = (0.0, 0.05, 0.10, 0.15)
#: plan_nav_to_prim's four approach letters: (which furniture face, sign of the standoff, yaw).
#: N parks off x_min facing +x; S off x_max facing -x; W off y_min facing +y; E off y_max facing -y.
SIDES = ("N", "S", "W", "E")


def lateral_for(side: str, coord: float, arm_bias: float) -> float:
    """The lateral coordinate that parks the base `arm_bias` metres to ITS OWN LEFT of `coord`.

    nav_tuning.arm_lateral_bias is "how far LEFT of the target the base parks" (positive = the
    base's +y), so a right arm's -0.05 puts the target on the robot's right. Left is +y facing
    +x (N) and -y facing -x (S), +x facing -y (E) and -x facing +y (W): the sign of the world
    offset flips between the two members of each pair. plan_nav_to_prim applied `+ arm_bias` on
    all four sides, so N and E parked the base on the wrong side of the target by 0.10 m."""
    return coord - arm_bias if side in ("N", "E") else coord + arm_bias


def park_for_arm(side: str, furniture: Box, object_xy, arm_bias: float, standoff_m: float):
    """(x, y, yaw) of the usual park on `side`, with the arm bias applied the same way on every side."""
    along = object_xy[0] if side in ("W", "E") else object_xy[1]
    return park_on_side(side, furniture, lateral_for(side, along, arm_bias), standoff_m)


def park_on_side(side: str, furniture: Box, lateral: float, standoff_m: float) -> tuple[float, float, float]:
    """(x, y, yaw) of the base parked `standoff_m` off `furniture`'s `side`, at `lateral` along it.
    THE SAME FOUR FORMULAS AS plan_nav_to_prim: plan_arm_grasp keys the grasp half-plane on the
    letter, so a letter here must mean what it means there."""
    if side == "N":
        return furniture.x_min - standoff_m, lateral, 0.0
    if side == "S":
        return furniture.x_max + standoff_m, lateral, math.pi
    if side == "W":
        return lateral, furniture.y_min - standoff_m, math.pi / 2
    if side == "E":
        return lateral, furniture.y_max + standoff_m, -math.pi / 2
    raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")


@dataclass(frozen=True)
class Park:
    side: str
    x: float
    y: float
    yaw: float
    clearance_m: float
    nearest: str | None
    #: How far ahead of the base origin the object sits, along the approach axis. The reach cost.
    ahead_m: float
    slide_m: float
    extra_m: float
    #: Index into `starts` of the first start whose drive to this pose is clear; None when no
    #: starts were given, or none of them reaches it (then this is the best CLEAR pose only).
    start: int | None = None


def inside(x: float, y: float, yaw: float, footprint: Footprint, bounds) -> bool:
    if bounds is None:
        return True
    x0, y0, x1, y1 = bounds
    return all(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in footprint.corners(x, y, yaw))


def ranked_parks(object_xy, furniture: Box, standoff_m: float, arm_bias: float,
                 footprint: Footprint, obstacles, prefer_side: str | None = None,
                 min_clear: float = MIN_CLEAR_M, bounds=None, starts=None) -> list:
    """Every clear candidate park, best first.

    Rank: (ahead_m, |slide|, side != prefer_side, extra, distance to the nearest start). Reach
    first, then the least lateral offset -- a slide that happens to cancel arm_bias is not a better
    pose, it is a different one -- then the side the free-space walk chose, then standoff (already
    inside ahead_m), then the side the base is already nearer to. `bounds` rejects poses outside
    the room: a wall is thin, and without this the far side of it is "clear"."""
    ox, oy = object_xy
    ranked = []
    for side in SIDES:
        along = 0 if side in ("W", "E") else 1        # the lateral coordinate is x for W/E, y for N/S
        for extra in EXTRAS_M:
            for slide in SLIDES_M:
                lateral = lateral_for(side, ox if along == 0 else oy, arm_bias) + slide
                x, y, yaw = park_on_side(side, furniture, lateral, standoff_m + extra)
                if not inside(x, y, yaw, footprint, bounds):
                    continue
                gap, who = clearance(x, y, yaw, footprint, obstacles)
                if gap < min_clear:
                    continue
                ahead = (ox - x) * math.cos(yaw) + (oy - y) * math.sin(yaw)
                # Last key: the nearest start. A FIXTURE target is its own centre, so its
                # opposite sides tie on everything above; the base should take the side it is
                # already on rather than the first letter in SIDES.
                to_start = min((math.hypot(x - s[0], y - s[1]) for s in (starts or ())), default=0.0)
                key = (round(ahead, 6), abs(slide), side != prefer_side, extra, round(to_start, 6))
                ranked.append((key, Park(side, x, y, yaw, gap, who, ahead, slide, extra)))
    ranked.sort(key=lambda kp: kp[0])
    return [p for _, p in ranked]


def pick_clear_park(object_xy, furniture: Box, standoff_m: float, arm_bias: float,
                    footprint: Footprint, obstacles, prefer_side: str | None = None,
                    min_clear: float = MIN_CLEAR_M, bounds=None, starts=None) -> Park | None:
    """The clear park that keeps the object nearest, or None if no side, slide or standoff clears.

    `starts` are (x, y, yaw) poses the drive begins from (spawn band centres, or the previous
    park); given, the best-ranked pose that SOME start reaches without contact wins, and only if
    none is reachable does the best clear pose come back with start=None. A park 4 cm off a fridge
    front is clear when aligned, but the rear corner sweeps into the fridge while turning to face
    the counter -- the drive is part of whether a pose is usable."""
    ranked = ranked_parks(object_xy, furniture, standoff_m, arm_bias, footprint, obstacles,
                          prefer_side, min_clear, bounds, starts)
    if not ranked:
        return None
    if starts is None:
        return ranked[0]
    # Lazily: the drive is the expensive part. ONE pass, with the turn tolerance. An earlier
    # version preferred a park whose final turn kept the full margin over one that needed the
    # tolerance, and on kitchen 1550's rotation 11 that slid the mug park 0.20 m along the
    # counter to avoid a 1.4 cm turn scrape -- trading reach, the binding constraint, for the
    # very thing TURN_SLOP_M exists to accept.
    for park in ranked:
        for i, (sx, sy, syaw) in enumerate(starts):
            if path_clear((sx, sy), syaw, (park.x, park.y), park.yaw, footprint, obstacles,
                          min_clear=min_clear)[0]:
                return Park(**{**park.__dict__, "start": i})
    return ranked[0]


# ----------------------------------------------------------------------------------------------
# The drive
# ----------------------------------------------------------------------------------------------

#: How often the base box is sampled along the straight drive. 0.05 m is a sixth of the footprint's
#: shortest half-extent, so nothing narrower than a chair leg can slip between samples.
PATH_STEP_M = 0.05
#: How far a TURN IN PLACE may overlap a box's AABB. Kitchen 1550's proven mug park (18 recorded
#: demos, main 3608cc99) is 0.087 m off the base cabinet's AABB, and the chain's next nav turns
#: the base 180 deg right there: by this model the corner sweeps 0.024 m into the box. The box
#: includes handles that protrude 0.038 m, and the runs say the turn is fine. The straight
#: drive keeps the full margin; only turns get this.
TURN_SLOP_M = 0.03


#: Turn-in-place sampling. The far corner of the base is 0.46 m out, so 5 degrees moves it 0.04 m
#: between samples -- coarser and a corner can pass through a 0.02 m margin unseen.
TURN_STEP_RAD = math.radians(5.0)


def _turn_samples(yaw_from: float, yaw_to: float):
    d = math.atan2(math.sin(yaw_to - yaw_from), math.cos(yaw_to - yaw_from))
    n = max(1, int(math.ceil(abs(d) / TURN_STEP_RAD)))
    return [yaw_from + d * k / n for k in range(n + 1)]


def path_clear(start_xy, start_yaw: float, park_xy, park_yaw: float, footprint: Footprint,
               obstacles, step_m: float = PATH_STEP_M, min_clear: float = MIN_CLEAR_M,
               turn_slop_m: float = TURN_SLOP_M) -> tuple[bool, str | None]:
    """Does the base clear everything on the drive simvla_video actually performs: turn in place to
    the bearing, drive straight along it, turn in place to the park yaw?

    Returns (clear, name of the first blocker). The same margin as the park: a drive that passes
    3 mm from a fridge is not collision-free under a +-0.02 m goal band and a 20 Hz controller."""
    sx, sy = start_xy
    px, py = park_xy
    dist = math.hypot(px - sx, py - sy)
    bearing = math.atan2(py - sy, px - sx) if dist > 1e-9 else start_yaw
    # The START turn always gets the tolerance: the base is where it is, and it must make this
    # turn whichever park is chosen, so its clearance cannot be a reason to prefer one park
    # over another. `turn_slop_m` governs the FINAL turn only.
    for yaw in _turn_samples(start_yaw, bearing):
        gap, who = clearance(sx, sy, yaw, footprint, obstacles)
        if gap < -TURN_SLOP_M:
            return False, who
    # The segment's INTERIOR only: its two end poses are the last sample of the first turn and
    # the first sample of the second, and those carry the turn tolerance. Checking the same pose
    # again with the drive margin refused every drive that starts beside furniture.
    n = max(1, int(math.ceil(dist / step_m)))
    for k in range(1, n):
        t = k / n
        gap, who = clearance(sx + (px - sx) * t, sy + (py - sy) * t, bearing, footprint, obstacles)
        if gap < min_clear:
            return False, who
    for yaw in _turn_samples(bearing, park_yaw):
        gap, who = clearance(px, py, yaw, footprint, obstacles)
        if gap < -turn_slop_m:
            return False, who
    return True, None


def band_centre(band) -> tuple[float, float]:
    """A spawn band is [["x", lo, hi], ["y", lo, hi]] for the base ORIGIN; the drive is checked
    from its CENTRE. The emitted bands are inset for the 0.23 m stand-in, so a real base already
    grazes furniture from a band's edges (RB-Y1 on kitchen 1550's east edge, the dishwasher) --
    a spawn-band defect this gate is not for. What it is for is a band on the WRONG SIDE of the
    furniture, and the centre tells that."""
    (_, x0, x1), (_, y0, y1) = band[0], band[1]
    return ((x0 + x1) / 2, (y0 + y1) / 2)


#: The run spawns the base facing +x (initial_rot_yaw_range is +-10 deg about 0).
SPAWN_YAW = 0.0


def bands_with_clear_path(bands, park, footprint: Footprint, obstacles) -> list[int]:
    """Indices of the bands from whose centre the drive to `park` is clear."""
    px, py, pyaw = park
    return [i for i, band in enumerate(bands)
            if path_clear(band_centre(band), SPAWN_YAW, (px, py), pyaw, footprint, obstacles)[0]]


# ----------------------------------------------------------------------------------------------
# Repairing the spawn band
# ----------------------------------------------------------------------------------------------

#: _generate_env_config's own numbers: a rectangle counts as a band when both sides exceed
#: 2 * robot_radius after `safe` is taken off, and the band is the rectangle inset by
#: robot_radius + 0.05.
GRID_ROBOT_RADIUS_M = 0.23
GRID_SAFE_M = 0.101
GRID_INSET_M = GRID_ROBOT_RADIUS_M + 0.05
#: How far two rows' x-extents may differ and still stack (see free_rectangles).
STACK_TOL_M = 0.05


def free_rectangles(furniture, bounds=None) -> list[tuple[float, float, float, float]]:
    """The free cells of the obstacle-edge grid, merged ROWS FIRST.

    Same grid as isaaclab.simvla.utils.find_free_spaces_grid (every box edge becomes a grid line,
    a cell is free when its centre is in no box). That module's merge_free_spaces then merges
    greedily in the order the cells were listed -- column-major -- so it fuses each column top to
    bottom first, and the floor in front of a counter run comes out as columns as wide as the gaps
    between the obstacle x-edges: 0.43-0.56 m on kitchen 1570, every one under the size gate.
    Merging each row across first keeps that floor as a strip the width of the room."""
    if bounds is None:
        bounds = (min(b.x_min for b in furniture), min(b.y_min for b in furniture),
                  max(b.x_max for b in furniture), max(b.y_max for b in furniture))
    xs = sorted({bounds[0], bounds[2], *(v for b in furniture for v in (b.x_min, b.x_max))})
    ys = sorted({bounds[1], bounds[3], *(v for b in furniture for v in (b.y_min, b.y_max))})

    def free(cx, cy):
        return not any(b.x_min < cx < b.x_max and b.y_min < cy < b.y_max for b in furniture)

    strips = []
    for j in range(len(ys) - 1):
        y0, y1 = ys[j], ys[j + 1]
        run = None
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            if free((x0 + x1) / 2, (y0 + y1) / 2):
                run = [x0, y0, x1, y1] if run is None else [run[0], y0, x1, y1]
            elif run is not None:
                strips.append(run)
                run = None
        if run is not None:
            strips.append(run)
    # Then stack rows whose x-extents agree within STACK_TOL_M, on their COMMON extent. Exact
    # agreement is defeated by the furniture itself: kitchen 1550's dishwasher, sink cabinet and
    # range fronts sit at x 1.4159 / 1.4179 / 1.4159, so the three rows in front of them -- each
    # under the 0.80 m gate on its own -- never stacked, and the floor in front of the counter run
    # was not a band. The intersection can only shrink a rectangle, so it stays free.
    merged = []
    for s in sorted(strips, key=lambda r: (r[1], r[0])):
        for m in merged:
            if abs(m[3] - s[1]) < 1e-9 and abs(m[0] - s[0]) <= STACK_TOL_M and abs(m[2] - s[2]) <= STACK_TOL_M:
                m[0], m[2], m[3] = max(m[0], s[0]), min(m[2], s[2]), s[3]
                break
        else:
            merged.append(list(s))
    return [tuple(r) for r in merged]


def spawn_bands(rects, robot_radius: float = GRID_ROBOT_RADIUS_M, safe: float = GRID_SAFE_M,
                footprint: Footprint | None = None):
    """_generate_env_config's size gate and inset, applied to `rects`.

    With a `footprint`, the radius is its longest extent instead of the 0.23 m stand-in: a band
    authored here is for the base that will spawn in it. (The emitted bands keep the stand-in,
    and a real base grazes furniture from their edges -- a defect this module leaves as found.)"""
    if footprint is not None:
        robot_radius = max(abs(footprint.x_min), footprint.x_max, abs(footprint.y_min), footprint.y_max)
    inset = robot_radius + 0.05
    out = []
    for x0, y0, x1, y1 in rects:
        if (x1 - x0 - safe) > 2 * robot_radius and (y1 - y0 - safe) > 2 * robot_radius:
            out.append([["x", x0 + inset, x1 - inset], ["y", y0 + inset, y1 - inset]])
    return out


def _band_distance(band, xy) -> float:
    (_, x0, x1), (_, y0, y1) = band[0], band[1]
    dx = max(x0 - xy[0], 0.0, xy[0] - x1)
    dy = max(y0 - xy[1], 0.0, xy[1] - y1)
    return math.hypot(dx, dy)


def repair_spawn_band(furniture, park, footprint: Footprint, obstacles):
    """The nearest row-merged band with a clear straight drive to `park`, or None.

    Called only when NO emitted band can reach the park -- a kitchen whose bands work keeps them."""
    px, py, _ = park
    bands = spawn_bands(free_rectangles(furniture))
    clear = bands_with_clear_path(bands, park, footprint, obstacles)
    if not clear:
        return None
    return min((bands[i] for i in clear), key=lambda b: _band_distance(b, (px, py)))


# ----------------------------------------------------------------------------------------------
# Routing: via-points around furniture
# ----------------------------------------------------------------------------------------------

#: Lattice resolution for the route search. Fine enough that a 0.6 m base threads a 0.85 m corridor.
ROUTE_STEP_M = 0.05
#: A bend costs this much driving, so the search prefers fewer turns over a marginally shorter path.
ROUTE_TURN_COST_M = 0.30
#: The four lattice headings: (dx, dy, yaw). Kitchen furniture is axis-aligned, so routes are too.
ROUTE_HEADINGS = ((1, 0, 0.0), (0, 1, math.pi / 2), (-1, 0, math.pi), (0, -1, -math.pi / 2))


def _axis_box(x: float, y: float, h: int, fp: Footprint):
    """The base's AABB at lattice heading h (0 +x, 1 +y, 2 -x, 3 -y). Exact for these yaws."""
    if h == 0:
        return (x + fp.x_min, y + fp.y_min, x + fp.x_max, y + fp.y_max)
    if h == 1:
        return (x - fp.y_max, y + fp.x_min, x - fp.y_min, y + fp.x_max)
    if h == 2:
        return (x - fp.x_max, y - fp.y_max, x - fp.x_min, y - fp.y_min)
    return (x + fp.y_min, y - fp.x_max, x + fp.y_max, y - fp.x_min)


def _aabb_gap(a, b: Box) -> float:
    return max(b.x_min - a[2], a[0] - b.x_max, b.y_min - a[3], a[1] - b.y_max)


def _point_box_distance(x: float, y: float, b: Box) -> float:
    return math.hypot(max(b.x_min - x, 0.0, x - b.x_max), max(b.y_min - y, 0.0, y - b.y_max))


def route(start, park, footprint: Footprint, obstacles, bounds, step_m: float = ROUTE_STEP_M):
    """Via-points that take the base from `start` (x, y, yaw) to `park` (x, y, yaw) clear of every
    obstacle: [] when the straight drive is already clear, None when no route exists.

    A* over a lattice with a heading state. A cell is open for a heading when the base's box at
    that heading clears everything by MIN_CLEAR_M and stays in the room; a bend is allowed only
    where the base can turn in place (its circumscribed circle overlaps nothing by more than
    TURN_SLOP_M), except at the start and the park, whose turns the exact sweep judges. Bends
    become via-points, each facing the next leg; every leg is then verified with path_clear -- the
    exact oriented sweep -- and via-points whose removal keeps every leg clear are pruned, so a
    detour is the fewest legs the furniture allows, not a staircase of lattice moves.

    The via-points are emitted as extra nav steps: the run's nav is turn-drive-turn per step, and
    the runtime advances to the next step when one finishes, so a polyline is a chain of steps."""
    sx, sy, syaw = start
    px, py, pyaw = park
    if path_clear((sx, sy), syaw, (px, py), pyaw, footprint, obstacles)[0]:
        return []
    x0, y0, x1, y1 = bounds
    if not all(math.isfinite(v) for v in bounds):
        xs = [b.x_min for b in obstacles] + [b.x_max for b in obstacles] + [sx, px]
        ys = [b.y_min for b in obstacles] + [b.y_max for b in obstacles] + [sy, py]
        x0, x1 = min(xs) - 1.0, max(xs) + 1.0
        y0, y1 = min(ys) - 1.0, max(ys) + 1.0
    nx = int((x1 - x0) / step_m) + 1
    ny = int((y1 - y0) / step_m) + 1
    if nx * ny > 400_000:
        raise ValueError(f"route lattice of {nx}x{ny} cells is too large; bounds {bounds}")

    def cx(i):
        return x0 + i * step_m

    def cy(j):
        return y0 + j * step_m

    def cell_of(x, y):
        return (min(max(int(round((x - x0) / step_m)), 0), nx - 1),
                min(max(int(round((y - y0) / step_m)), 0), ny - 1))

    open_memo = {}
    turn_memo = {}
    radius = footprint.radius_m

    def is_open(i, j, h):
        key = (i, j, h)
        v = open_memo.get(key)
        if v is None:
            box = _axis_box(cx(i), cy(j), h, footprint)
            v = (box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1
                 and all(_aabb_gap(box, ob) >= MIN_CLEAR_M for ob in obstacles))
            open_memo[key] = v
        return v

    def turn_safe(i, j):
        key = (i, j)
        v = turn_memo.get(key)
        if v is None:
            x, y = cx(i), cy(j)
            v = all(_point_box_distance(x, y, ob) >= radius - TURN_SLOP_M for ob in obstacles)
            turn_memo[key] = v
        return v

    import heapq
    si, sj = cell_of(sx, sy)
    gi, gj = cell_of(px, py)
    best = {}
    came = {}
    frontier = []
    for h in range(4):
        # The start turn is always allowed (the base is where it is); the start cell may be tight.
        best[(si, sj, h)] = 0.0
        came[(si, sj, h)] = None
        heapq.heappush(frontier, ((abs(gi - si) + abs(gj - sj)) * step_m, 0.0, si, sj, h))
    goal_state = None
    while frontier:
        _, g, i, j, h = heapq.heappop(frontier)
        if g > best.get((i, j, h), math.inf):
            continue
        if (i, j) == (gi, gj):
            goal_state = (i, j, h)
            break
        for d, (dx, dy, _yaw) in enumerate(ROUTE_HEADINGS):
            ni, nj = i + dx, j + dy
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if d != h and (i, j) != (si, sj) and not turn_safe(i, j):
                continue
            # the park cell is judged by the exact sweep, like the start
            if (ni, nj) != (gi, gj) and not is_open(ni, nj, d):
                continue
            ng = g + step_m + (ROUTE_TURN_COST_M if d != h else 0.0)
            if ng < best.get((ni, nj, d), math.inf):
                best[(ni, nj, d)] = ng
                came[(ni, nj, d)] = (i, j, h)
                heapq.heappush(frontier, (ng + (abs(gi - ni) + abs(gj - nj)) * step_m, ng, ni, nj, d))
    if goal_state is None:
        return None
    # walk back; a via-point is where the heading changes
    path = []
    s = goal_state
    while s is not None:
        path.append(s)
        s = came[s]
    path.reverse()
    via = []
    for k in range(1, len(path)):
        i, j, h = path[k]
        pi, pj, ph = path[k - 1]
        if h != ph and k > 1:
            via.append((cx(pi), cy(pj), ROUTE_HEADINGS[h][2]))
    via = _prune_via((sx, sy, syaw), via, (px, py, pyaw), footprint, obstacles)
    return via


def _with_bearings(start, via, park):
    """Each via-point faces the next leg, so the arrival turn at one is the departure heading."""
    pts = list(via) + [park]
    out = []
    prev = start
    for k, v in enumerate(via):
        nxt = pts[k + 1]
        yaw = math.atan2(nxt[1] - v[1], nxt[0] - v[0])
        out.append((v[0], v[1], yaw))
        prev = v
    return out


def _legs_ok(start, via, park, footprint, obstacles) -> bool:
    poses = [start] + list(via) + [park]
    return all(path_clear(a[:2], a[2], b[:2], b[2], footprint, obstacles)[0]
               for a, b in zip(poses, poses[1:]))


def _prune_via(start, via, park, footprint: Footprint, obstacles):
    """Drop via-points whose removal keeps every leg clear under the exact sweep; None if even
    the lattice route fails the exact sweep (it should not, but exactness is the sweep's)."""
    via = _with_bearings(start, list(via), park)
    if not _legs_ok(start, via, park, footprint, obstacles):
        return None
    changed = True
    while changed and via:
        changed = False
        for k in range(len(via)):
            trial = _with_bearings(start, via[:k] + via[k + 1:], park)
            if _legs_ok(start, trial, park, footprint, obstacles):
                via = trial
                changed = True
                break
    return via


#: A trimmed band smaller than this is a spawn POINT, not a band (kitchen 1550's rotation-11 band
#: trimmed to a 5 cm square): no spawn diversity, and a repaired band serves better.
MIN_TRIMMED_BAND_AREA_M2 = 0.01


def shrink_band(band, footprint: Footprint, obstacles, bounds, step_m: float = ROUTE_STEP_M):
    """The largest part of a spawn band where the real base fits and can turn in place, or None.

    The emitted bands are inset for the 0.23 m stand-in. Kitchen 1570's reference band puts
    RB-Y1's front 4.5 cm inside the fridge at its north edge and 1.6 cm inside the wall at its
    east edge, and a base that spawns in contact cannot be routed anywhere. A cell is usable when
    the base at the spawn heading clears everything and its turning circle overlaps nothing by
    more than TURN_SLOP_M; the result is the largest axis-aligned rectangle of usable cells, as a
    band. A band that is usable everywhere comes back unchanged (the same object)."""
    (_, bx0, bx1), (_, by0, by1) = band[0], band[1]
    x0, y0, x1, y1 = bounds
    radius = footprint.radius_m
    nx = max(1, int(round((bx1 - bx0) / step_m)) + 1)
    ny = max(1, int(round((by1 - by0) / step_m)) + 1)
    xs = [bx0 + (bx1 - bx0) * i / (nx - 1) if nx > 1 else bx0 for i in range(nx)]
    ys = [by0 + (by1 - by0) * j / (ny - 1) if ny > 1 else by0 for j in range(ny)]

    def usable(x, y):
        box = _axis_box(x, y, 0, footprint)
        if not (box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1):
            return False
        if any(_aabb_gap(box, ob) < 0.0 for ob in obstacles):
            return False
        return all(_point_box_distance(x, y, ob) >= radius - TURN_SLOP_M for ob in obstacles)

    grid = [[usable(xs[i], ys[j]) for j in range(ny)] for i in range(nx)]
    if all(all(col) for col in grid):
        return band
    # maximal rectangle of True cells (histogram method over columns of x)
    best, best_area = None, 0
    heights = [0] * ny
    for i in range(nx):
        for j in range(ny):
            heights[j] = heights[j] + 1 if grid[i][j] else 0
        stack = []
        for j in range(ny + 1):
            h = heights[j] if j < ny else 0
            start = j
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (j - s)
                if area > best_area:
                    best_area = area
                    best = (i - sh + 1, s, i, j - 1)        # i0, j0, i1, j1 inclusive
                start = s
            stack.append((start, h))
    if best is None:
        return None
    i0, j0, i1, j1 = best
    if (xs[i1] - xs[i0]) * (ys[j1] - ys[j0]) < MIN_TRIMMED_BAND_AREA_M2:
        return None
    return [["x", xs[i0], xs[i1]], ["y", ys[j0], ys[j1]]]


def band_corners(band):
    (_, x0, x1), (_, y0, y1) = band[0], band[1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


class NavRoute(list):
    """A nav planner's result: the park pose as a plain [x, y, yaw] list -- so a GUI save or a
    json dump sees exactly the pose it always did -- carrying the via-points the base drives
    through first. task_emit.plan_steps writes each via-point as its own nav step ahead of the
    park; every other consumer sees a list of three floats."""

    def __init__(self, pose, via=()):
        super().__init__(float(v) for v in pose)
        self.via = [[float(v) for v in p] for p in via]


# ----------------------------------------------------------------------------------------------
# The whole decision
# ----------------------------------------------------------------------------------------------

class NavClearanceError(ValueError):
    """No side, slide or standoff parks the base clear of this kitchen's furniture. task_emit
    treats a planner's ValueError as PlanFailure and skips the kitchen BY NAME, which is the
    right outcome: a run authored here would grind on furniture for its whole wall-clock."""


@dataclass(frozen=True)
class Resolution:
    x: float
    y: float
    yaw: float
    side: str
    #: True when the pose is NOT the usual one.
    changed: bool
    clearance_m: float
    nearest: str | None
    #: A replacement for kitchen_data["initial_pos_ranges"], or None to keep the emitted bands.
    bands: list | None
    #: What happened, one line each, for the emit log.
    notes: tuple
    #: Via-points (x, y, yaw) to drive through BEFORE the park, emitted as extra nav steps. Empty
    #: when the straight drive is clear.
    via: tuple = ()


def scene_obstacles(boxes, target_name=None):
    """(room bounds, obstacles) for a kitchen's /world AABBs: floor-level furniture plus the
    shell walls -- synthesised when the USD has none, as the run adds them."""
    bounds, walls = room_interior(boxes)
    have = {b.name for b in boxes}
    obstacles = floor_obstacles(boxes, exclude=() if target_name is None else (target_name,))
    obstacles += [w for w in walls if w.name not in have]
    return bounds, obstacles


#: How many ranked candidates the routing pass tries before giving up. A route search costs
#: ~0.1 s; a kitchen with no route anywhere would otherwise search once per candidate.
ROUTE_TRIES = 8


def resolve_park(usual, usual_side, object_xy, furniture: Box, standoff_m: float, arm_bias: float,
                 footprint: Footprint, boxes, bands, target_name=None, prev_park=None) -> Resolution:
    """plan_nav_to_prim's CASE 2 decision, given the pose it would have authored.

    usual       -- (x, y, yaw) the existing formula produced, or None if the free-space walk found
                   no side at all (it used to raise there; now the search runs from scratch).
    bands       -- kitchen_data["initial_pos_ranges"] as emitted; drives are checked from their
                   centres when this is the chain's FIRST nav (prev_park None).
    prev_park   -- where the base already stands for a later nav in the chain.

    AS USUAL UNLESS IT COLLIDES -- and a clear park whose DRIVE collides is a collision. In order:
      1. the usual pose, clear, with a clear straight drive from a start      -> as usual
      2. the usual pose, clear, with a ROUTE (via-points) from a start         -> as usual + via
      3. a re-sited clear pose a start reaches straight                        -> changed
      4. a re-sited clear pose a start reaches by a route                      -> changed + via
      5. (first nav) the usual/re-sited pose from a REPAIRED spawn band        -> + bands
      6. refuse, naming every blocked drive
    The spawn band is kept whenever a drive or a route exists from it: the run starts where it used
    to and drives AROUND the furniture. Replacing the band is the last resort before refusal."""
    bounds, obstacles = scene_obstacles(boxes, target_name)
    furniture_boxes = [b for b in boxes if not is_shell_wall(b.name)]
    first_nav = prev_park is None
    notes = []
    # EVERY spawn band is trimmed to where the real base fits and can turn. The emitted bands are
    # inset for the 0.23 m stand-in: 1550's puts RB-Y1's front 6.5 cm into the base cabinet at
    # its north edge and 1570's 4.5 cm into the fridge. A spawn in contact is a collision before
    # the drive starts. A band that fits already comes back the same object, so a kitchen whose
    # bands fit keeps them untouched.
    trimmed = [(b, shrink_band(b, footprint, obstacles, bounds)) for b in bands] if first_nav else []
    usable = [tb for _, tb in trimmed if tb is not None]
    band_changed = any(tb is not orig for orig, tb in trimmed)
    if first_nav and band_changed:
        dropped = sum(1 for _, tb in trimmed if tb is None)
        notes.append("spawn band(s) trimmed to where the base fits and can turn: "
                     + "; ".join(f"x[{b[0][1]:.3f}, {b[0][2]:.3f}] y[{b[1][1]:.3f}, {b[1][2]:.3f}]" for b in usable)
                     + (f" ({dropped} dropped: no part fits)" if dropped else ""))
    rep = None

    def repaired_bands():
        nonlocal rep
        if rep is None:
            rep = spawn_bands(free_rectangles(furniture_boxes), footprint=footprint)
        return rep

    def straight_from(starts_, pose):
        """(index of the first start with a clear straight drive, {start index: blocker})."""
        blocked = {}
        for i, (sx, sy, syaw) in enumerate(starts_):
            ok, who = path_clear((sx, sy), syaw, pose[:2], pose[2], footprint, obstacles)
            if ok:
                return i, blocked
            blocked[i] = who
        return None, blocked

    if first_nav and bands and not usable:
        # No part of any emitted band fits the base: spawn from a repaired band, or refuse.
        usable = list(repaired_bands())
        if not usable:
            raise NavClearanceError(
                f"no spawn band fits the base ({footprint.x_max - footprint.x_min:.3f} x "
                f"{footprint.y_max - footprint.y_min:.3f} m) with room to turn: every emitted band "
                f"trims to nothing and the free-space grid offers no rectangle wide enough")
        notes.append("no part of any emitted spawn band fits the base; spawning from a repaired band: "
                     + "; ".join(f"x[{b[0][1]:.3f}, {b[0][2]:.3f}] y[{b[1][1]:.3f}, {b[1][2]:.3f}]" for b in usable))
    starts = [(*band_centre(b), SPAWN_YAW) for b in usable] if first_nav else [tuple(prev_park)]
    authored_bands = list(usable) if (first_nav and band_changed and usable) else None
    kwargs = dict(standoff_m=standoff_m, arm_bias=arm_bias, footprint=footprint,
                  obstacles=obstacles, prefer_side=usual_side, bounds=bounds)
    def route_from(starts_, pose):
        """(index of the first start with a route, its via-points) or (None, None).

        For the first nav the starts are the trimmed bands' centres, and a route counts only if
        its first leg is also clear from the band's four corners -- the run spawns anywhere in it."""
        for i, s in enumerate(starts_):
            via = route(s, pose, footprint, obstacles, bounds)
            if not via:
                continue
            if first_nav:
                first = via[0]
                if not all(path_clear(c, SPAWN_YAW, first[:2], first[2], footprint, obstacles)[0]
                           for c in band_corners(usable[i])):
                    continue
            return i, tuple(via)
        return None, None

    def routed_bands(i):
        """The band list to author when route i (first nav) is used."""
        return authored_bands

    def band_text(b):
        return f"x[{b[0][1]:.3f}, {b[0][2]:.3f}] y[{b[1][1]:.3f}, {b[1][2]:.3f}]"

    def via_text(via):
        return " -> ".join(f"({v[0]:.2f}, {v[1]:.2f})" for v in via)

    def give_up(target, blocked, rep_blocked):
        where = "the previous park" if not first_nav else "the emitted spawn band(s)"
        parts = [f"from {where} {tuple(round(v, 3) for v in starts[i][:2])}: {who}"
                 for i, who in blocked.items()]
        parts += [f"from repaired band {band_text(repaired_bands()[i])}: {who}"
                  for i, who in rep_blocked.items()]
        return NavClearanceError(
            f"no straight drive or route reaches {target} on any side of {furniture.name} without "
            f"meeting furniture -- " + "; ".join(parts) + f". Blocked drives: {' | '.join(parts) or 'none tried'}")

    def keep(pose, gap, who, new_bands=None, via=()):
        return Resolution(pose[0], pose[1], pose[2], usual_side, False, gap, who,
                          authored_bands if new_bands is None else new_bands, tuple(notes), tuple(via))

    def changed(park, new_bands=None, via=()):
        notes.append(f"re-sited to side {park.side} at ({park.x:.3f}, {park.y:.3f}, "
                     f"{math.degrees(park.yaw):.0f} deg): slide {park.slide_m:+.2f} m, extra standoff "
                     f"{park.extra_m:.2f} m, clears {park.nearest} by {park.clearance_m:.3f} m, object "
                     f"{park.ahead_m:.3f} m ahead")
        return Resolution(park.x, park.y, park.yaw, park.side, True, park.clearance_m, park.nearest,
                          authored_bands if new_bands is None else new_bands, tuple(notes), tuple(via))

    def routed_candidate(starts_):
        """(best-ranked clear park some start reaches by a route, its route, the start index)."""
        ranked = ranked_parks(object_xy, furniture, starts=starts_, **kwargs)
        for park in ranked[:ROUTE_TRIES]:
            i, via = route_from(starts_, (park.x, park.y, park.yaw))
            if via:
                return park, via, i
        return None, None, None

    if usual is not None:
        gap, who = clearance(usual[0], usual[1], usual[2], footprint, obstacles)
        pose_text = f"park ({usual[0]:.3f}, {usual[1]:.3f}, {math.degrees(usual[2]):.0f} deg)"
        if gap >= 0.0:
            notes.append(f"{pose_text} clears {who} by {gap:.3f} m -- kept as usual")
            if not starts:
                return keep(usual, gap, who)
            idx, blocked = straight_from(starts, usual)
            if idx is not None:
                return keep(usual, gap, who)
            blocker = blocked.get(0)
            notes.append(f"no emitted spawn band drives straight to the park without meeting {blocker}"
                         if first_nav else f"the straight drive from the previous park meets {blocker}")
            ridx, via = route_from(starts, usual)
            if via:
                notes.append(f"routed around it: {len(via)} via-point(s) {via_text(via)}")
                return keep(usual, gap, who, routed_bands(ridx), via=via)
            park = pick_clear_park(object_xy, furniture, starts=starts, **kwargs)
            if park is not None and park.start is not None:
                notes.append("no route to the usual park; re-siting to one a straight drive reaches")
                return changed(park)
            park, via, ridx = routed_candidate(starts)
            if park is not None:
                notes.append(f"no route to the usual park; re-siting to one a route reaches: "
                             f"{len(via)} via-point(s) {via_text(via)}")
                return changed(park, routed_bands(ridx), via=via)
            rep_blocked = {}
            if first_nav:
                rep_starts = [(*band_centre(b), SPAWN_YAW) for b in repaired_bands()]
                ridx, rep_blocked = straight_from(rep_starts, usual)
                if ridx is not None:
                    best = repaired_bands()[ridx]
                    notes.append(f"no route from the emitted band either; spawn band replaced by {band_text(best)}")
                    return keep(usual, gap, who, [best])
                again = pick_clear_park(object_xy, furniture, starts=rep_starts, **kwargs)
                if again is not None and again.start is not None:
                    best = repaired_bands()[again.start]
                    notes.append(f"no route from the emitted band either; spawn band replaced by {band_text(best)}")
                    return changed(again, [best])
            raise give_up(f"the clear {pose_text} or any other clear park", blocked, rep_blocked)
        notes.append(f"{pose_text} is INSIDE {who} by {-gap:.3f} m -- re-siting")
    else:
        notes.append("free-space walk found no side; searching all four")

    park = pick_clear_park(object_xy, furniture, starts=starts if starts else None, **kwargs)
    if park is None:
        raise NavClearanceError(
            f"no clear park for the base ({footprint.x_max - footprint.x_min:.3f} x "
            f"{footprint.y_max - footprint.y_min:.3f} m) on any side of {furniture.name}: every "
            f"side, slide up to {max(SLIDES_M):.2f} m and extra standoff up to {max(EXTRAS_M):.2f} m "
            f"meets furniture or leaves the room")
    if not starts or park.start is not None:
        return changed(park)
    _, blocked = straight_from(starts, (park.x, park.y, park.yaw))
    rpark, via, ridx = routed_candidate(starts)
    if rpark is not None:
        notes.append(f"no straight drive to a clear park; routed: {len(via)} via-point(s) {via_text(via)}")
        return changed(rpark, routed_bands(ridx), via=via)
    rep_blocked = {}
    if first_nav:
        rep_starts = [(*band_centre(b), SPAWN_YAW) for b in repaired_bands()]
        again = pick_clear_park(object_xy, furniture, starts=rep_starts, **kwargs)
        if again is not None and again.start is not None:
            best = repaired_bands()[again.start]
            notes.append(f"no emitted spawn band drives or routes to a clear park; spawn band "
                         f"replaced by {band_text(best)}")
            return changed(again, [best])
        _, rep_blocked = straight_from(rep_starts, (park.x, park.y, park.yaw))
    raise give_up(f"any clear park (best: side {park.side} at ({park.x:.3f}, {park.y:.3f}))",
                  blocked, rep_blocked)
