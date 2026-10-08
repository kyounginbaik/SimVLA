"""Can this kitchen host the open-the-fridge task? Four conditions, not one.

WHY THIS EXISTS. fridge_kitchen.best_arc_clearance certifies a ROTATION OF THE BASE ABOUT THE
HINGE. fridge_template.SWING_SWEEP_DEG is 0.0, so the robot runs a straight diagonal retreat
instead. The gate passed kitchen 1400 at 65 deg with 0.355 m of room for a motion the robot never
performs; the authored diagonal has 0.252 m, and the runtime independently measured 0.240 m to
base_cabinet_0 at the final pose.

It also never checked the thing that actually caps the task: the base parks at the handle's own
lateral coordinate, INSIDE the door's swept arc, and the slab jams against it well short of a
usable open angle -- see door_blocked_deg's docstring below for the measured comparison against
the OBB/SAT authority.

PURE GEOMETRY ON PURPOSE. fridge_kitchen needs scene_synthesizer (env_isaaclab only) and
door_sweep_blocker needs pxr (dexdreamer only). A module importing both could not run in either
environment. Numbers in, verdict out; the callers do the measuring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class SitingError(ValueError):
    """This kitchen cannot host the task, and says why."""


#: Anubis's base footprint radius, measured 2026-08-28 from anubis_simvla.usd's
#: /anubis_for_random_parts/base_link colliders (0.465 x 0.492 m). NOT the nav standoff
#: constant WHEEL_RADIUS_M = 0.23, which is a different quantity: where nav.to_prim chooses to
#: park, versus how big the robot is. Conflating them authors an offset that does not clear
#: the door -- on kitchen 1400 at standoff 0.375, base_radius_m=0.230 gives min_past_for(60) =
#: 0.14 (the door still jams there; OBB/SAT says it is free only from past >= 0.30), while
#: base_radius_m=0.340 gives 0.27.
ANUBIS_BASE_RADIUS_M = 0.340


@dataclass(frozen=True)
class Site:
    """Everything the four conditions need, all of it measured elsewhere.

    hinge_xy/handle_xy/radius_m  fridge_kitchen.hinge_and_handle
    face_normal                  fridge_kitchen.fridge_front -- unit, points out of the door face
    standoff_m                   NAV_SAFETY_M + the base radius; what nav.to_prim parks at
    base_radius_m                robot_footprint.measure(...)['radius_m']
    reach_m                      the robot's forward reach at handle height (anubis: 0.555)
    """
    hinge_xy: tuple
    handle_xy: tuple
    radius_m: float
    face_normal: tuple
    standoff_m: float
    base_radius_m: float
    reach_m: float

    @property
    def away_from_hinge(self) -> tuple:
        """Unit vector from the hinge toward the handle, i.e. the direction 'past the handle'."""
        vx = self.handle_xy[0] - self.hinge_xy[0]
        vy = self.handle_xy[1] - self.hinge_xy[1]
        n = math.hypot(vx, vy) or 1.0
        return (vx / n, vy / n)


def base_xy(site: Site, past_m: float) -> tuple:
    """Where nav.to_prim parks, offset `past_m` beyond the handle away from the hinge.

    Reproduces plan_nav_to_prim: the LATERAL coordinate is the handle's own (plus the offset), the
    NORMAL coordinate is the door face plus the standoff.
    """
    ax, ay = site.away_from_hinge
    if abs(site.face_normal[0]) > abs(site.face_normal[1]):     # door faces +-x
        return (site.handle_xy[0] + site.face_normal[0] * site.standoff_m,
                site.handle_xy[1] + past_m * ay)
    return (site.handle_xy[0] + past_m * ax,
            site.handle_xy[1] + site.face_normal[1] * site.standoff_m)


def _door_edge(site: Site, deg: float) -> tuple:
    """The door's free edge (the handle end) after swinging `deg` toward the robot."""
    hx, hy = site.hinge_xy
    vx, vy = site.handle_xy[0] - hx, site.handle_xy[1] - hy
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return (hx + c * vx - s * vy, hy + s * vx + c * vy)


def _sense(site: Site) -> float:
    """+1 or -1: the rotational sense that swings the door TOWARD the parked robot.

    Decided by which sense moves the free edge closer to the base after one degree, rather than
    assumed from a sign convention -- the hinge is on either side depending on the kitchen, and a
    fixed sign parks the whole analysis on the wrong half of the circle.
    """
    b = base_xy(site, 0.0)
    hx, hy = site.hinge_xy
    vx, vy = site.handle_xy[0] - hx, site.handle_xy[1] - hy
    out = []
    for s in (1.0, -1.0):
        t = math.radians(1.0) * s
        c, sn = math.cos(t), math.sin(t)
        e = (hx + c * vx - sn * vy, hy + sn * vx + c * vy)
        out.append((math.dist(e, b), s))
    return min(out)[1]


def door_blocked_deg(site: Site, past_m: float, cap_deg: float = 180.0,
                     step_deg: float = 0.5) -> float | None:
    """SCREEN ONLY -- NOT AN AUTHORITY. The angle at which the door's free EDGE POINT first reaches
    a CIRCLE of base_radius_m about the parked base, or None.

    THIS MODEL ERRS IN BOTH DIRECTIONS AND MAY NOT DECIDE WHETHER A KITCHEN IS USABLE.

    THE MEASURED COMPARISON TABLE (kitchen 1400, at the true parked pose: base y =
    handle_bbox_min - 0.35 = -0.7435, matching the goal file the GPU actually consumes). This is
    the ONE copy of these numbers -- the module docstring and min_past_for refer here rather than
    repeating them, so a re-measurement only has to change one place:

        past   OBB/SAT authority   screen (this function, base_radius_m=0.340)
        0.00        13.0 deg              3.5 deg
        0.20        13.0 deg             10.5 deg
        0.28        13.0 deg             free        <- the screen's blind spot
        0.30      free to 90             free
        0.40      free to 90             free

    (13.0 deg predicted vs 13.1 deg measured on GPU.)

    CAUTION: 10.5 appears above as the SCREEN's own reading at past=0.20 -- a different number,
    from a different pose, than an earlier draft's (wrong, 25-mm-off-pose) OBB/SAT reading of
    10.5 at every past. They are unrelated and coincide only by accident; do not conflate them if
    a "10.5" resurfaces somewhere.

    A circumscribed circle cannot collide later than the box it contains, so the error is not in
    the base model -- it is in the DOOR model. This function tracks one point at radius R. The real
    door carries two collision prims, and the handle protrudes toward the robot, so it strikes the
    base at offsets where the edge point is still clear. No choice of base_radius_m fixes that.

    Use this to rank and to fail fast. Condition (a) is decided by door_sweep_blocker.sweep(...,
    extra_boxes=[box_at(...)]) on the built USD, and nothing else.
    """
    b = base_xy(site, past_m)
    sense = _sense(site)
    n = int(round(cap_deg / step_deg))
    for i in range(n + 1):
        deg = i * step_deg
        if math.dist(_door_edge(site, deg * sense), b) <= site.base_radius_m:
            return deg
    return None


def reach_m_at(site: Site, past_m: float) -> float:
    """How far the parked base is from the handle it has to grasp."""
    return math.dist(base_xy(site, past_m), site.handle_xy)


def min_past_for(site: Site, target_sweep_deg: float, step_m: float = 0.01,
                 cap_m: float = 1.0, margin_m: float = 0.0) -> float:
    """A LOWER BOUND on the offset that frees the door, plus an optional margin, and a hard check
    that the value actually RETURNED -- margin included -- is in reach.

    Built on door_blocked_deg, so it inherits that function's point-model blindness: the real
    minimum is at or ABOVE this value, never below -- see door_blocked_deg's docstring for the
    measured gap between this function's answer and the OBB authority's on kitchen 1400. Treat
    the result as the start of the search, and confirm with the OBB sweep before committing a
    kitchen.

    RAISES rather than returning the least bad. A fridge that can only be cleared from out of
    reach cannot host this task, and returning a number anyway spends a GPU allocation to find
    that out -- the discipline build_fridge_template already applies at MIN_OPEN_DEG.

    margin_m IS ADDED BEFORE THE REACH CHECK, not after. This function used to check reach only
    on the raw door-clearing lower bound, and a caller (fridge_park_past_m) added a margin to
    that value afterward with no re-check -- so the offset actually returned and baked into a
    goal file could be out of reach without ever raising. Reproduced: a Site with
    standoff_m=0.465 (0.09 m deeper than kitchen 1400's handle -- plausible on another fridge)
    gives a raw lower bound of 0.260 m, reach 0.533 m, which passes; adding a 0.05 m margin
    afterward gives 0.310 m, reach 0.559 m -- 4 mm past a 0.555 m reach, and nothing raised. So
    there is exactly ONE reach check in this function, it runs on offset = lower_bound + margin_m
    -- the value this function returns -- and there is no second place downstream for a caller to
    add a margin and reopen that hole.
    """
    n = int(round(cap_m / step_m))
    first_clear = None
    for i in range(n + 1):
        past = i * step_m
        blocked = door_blocked_deg(site, past, cap_deg=target_sweep_deg)
        if blocked is None:
            first_clear = past
            break
    if first_clear is None:
        raise SitingError(
            f"no offset up to {cap_m:.2f} m clears the door to {target_sweep_deg:.0f} deg; the "
            f"base is inside the swept arc wherever it stands in front of this handle"
        )
    offset = first_clear + margin_m
    got = reach_m_at(site, offset)
    if got > site.reach_m:
        raise SitingError(
            f"clearing the door to {target_sweep_deg:.0f} deg needs the base {offset:.3f} m past "
            f"the handle ({first_clear:.3f} m lower bound + {margin_m:.3f} m margin), which puts "
            f"the handle {got:.3f} m away -- past the robot's {site.reach_m:.3f} m reach. This "
            f"kitchen cannot host the task."
        )
    return offset


def chord_diag_deg(target_sweep_deg: float) -> float:
    """The handle's chord direction, in degrees off straight-back, for a given total sweep.

    The handle starts at radius r from the hinge and ends rotated by theta, so its displacement is
    r*(1-cos theta, -sin theta) -- a chord lying at theta/2 off the initial tangent. DIAG_DEG =
    45.0 is therefore the chord of a NINETY-degree swing, and was never the right constant for the
    60-degree target this task actually aims at.
    """
    return target_sweep_deg / 2.0


def retreat_path(site: Site, past_m: float, diag_deg: float, pull_m: float,
                 total_m: float, samples: int = 120) -> list:
    """The base path the template actually authors: `pull_m` straight back, then `total_m` at
    `diag_deg` off straight-back, toward the hinge side.

    Sampled densely rather than at the six waypoints: consecutive waypoints are driven as straight
    lines, so the swept volume is the polyline, and a six-point sample walks straight through a
    cabinet that sits between two waypoints.

    The diagonal's basis is (face_normal, its in-plane perpendicular) -- NOT
    (face_normal, away_from_hinge). away_from_hinge is only APPROXIMATELY perpendicular to
    face_normal: hinge and handle are rarely exactly level (kitchen 1400's differ by 4.1 mm in y),
    so a basis built from it is not orthonormal, and the authored total_m leg comes out short
    (0.8966 m measured on kitchen 1400, for an authored 0.90 m). face_normal is unit by
    construction (fridge_front), so its exact in-plane perpendicular gives an orthonormal basis and
    the leg lengths this function claims to author are the ones it actually produces.

    The `samples=120` default was chosen and exercised only on kitchen 1400; it has not been
    checked against a layout with a tighter gap, and a coarser-than-needed sample could step
    straight over a cabinet that a denser sample would have caught.
    """
    nx, ny = site.face_normal
    perp = (-ny, nx)
    hx, hy = site.hinge_xy
    to_hinge = (hx - site.handle_xy[0], hy - site.handle_xy[1])
    if perp[0] * to_hinge[0] + perp[1] * to_hinge[1] < 0.0:
        perp = (-perp[0], -perp[1])              # pick the perpendicular that points to the hinge
    back = (nx, ny)
    b = base_xy(site, past_m)
    pts = [b]
    p = (b[0] + back[0] * pull_m, b[1] + back[1] * pull_m)
    pts.append(p)
    t = math.radians(diag_deg)
    d = (back[0] * math.cos(t) + perp[0] * math.sin(t),
         back[1] * math.cos(t) + perp[1] * math.sin(t))
    for i in range(samples):
        s = (i + 1) * total_m / samples
        pts.append((p[0] + d[0] * s, p[1] + d[1] * s))
    return pts
