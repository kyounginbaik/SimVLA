"""nav.push_prim's base pose, as arithmetic.

Its own module, stdlib only, for the reason skill_contract gives: the planner body has to live in
simvla_data_generator.py (that is where @register_planner injects it, and where the stage is), and
that module boots Omniverse at import. Nothing in there can be unit-tested. So the stage work --
two bbox lookups and a projection -- stays there, and the arithmetic lives here, where a laptop can
check it.
"""

from __future__ import annotations

import math


def push_prim_base_pose(prim_xy, toward_xy, half_extent_along_u: float,
                        wheel_r: float, safety: float, offset_m: float):
    """Where the base stands to push `prim_xy` toward `toward_xy`, and which way it faces.

    Returns (x, y, yaw) in the kitchen frame -- the same frame and the same 3-slot N_s payload
    plan_nav_to_prim returns.

    The base sits on the line through both centres, on the FAR side of the prim from the target, at
    `half_extent_along_u + wheel_r + safety` from the prim's centre -- just clear of the chair
    rather than inside it -- and yawed to look along the push direction. `offset_m` slides it along
    that same line toward the target: 0.0 is the approach pose and a positive value is the push,
    which is why both legs of one push must be computed from ONE axis. Two poses derived from
    different directions would turn the base mid-push and slew the chair off the line.
    """
    dx = float(prim_xy[0]) - float(toward_xy[0])
    dy = float(prim_xy[1]) - float(toward_xy[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        # nav.to_prim's canonical-snap fallback returns (0, 0, 0) here and the robot drives to the
        # env-frame origin with no error. Refusing instead is the whole reason this skill exists.
        raise ValueError(
            f"push target {tuple(prim_xy)} is on top of {tuple(toward_xy)}; there is no outward "
            f"direction to push along"
        )
    ux, uy = dx / n, dy / n
    stand = float(half_extent_along_u) + float(wheel_r) + float(safety) - float(offset_m)
    # The heading is the bearing from the prim TO the target, and it is taken from that difference
    # rather than from (-uy, -ux). Negating a component that is exactly 0.0 gives -0.0, and
    # atan2(-0.0, -1.0) is -pi where atan2(+0.0, -1.0) is +pi. Both name the same heading, but
    # which one you get would be decided by the sign of a zero -- and plan_nav_to_prim writes
    # +math.radians(180.0) for this heading (simvla_data_generator.py:1252, 1386), so a sibling
    # planner that sometimes emitted -pi would disagree with the file it sits next to for no
    # reason a reader could see. `t - p` is +0.0 whenever t == p, so this is stable.
    return (float(prim_xy[0]) + ux * stand,
            float(prim_xy[1]) + uy * stand,
            math.atan2(float(toward_xy[1]) - float(prim_xy[1]),
                       float(toward_xy[0]) - float(prim_xy[0])))


def half_extent_along(local_min, local_max, yaw: float, ux: float, uy: float) -> float:
    """Half the plan-view width of an ORIENTED box, measured along the world unit vector (ux, uy).

    `local_min` / `local_max` are the box's extents in ITS OWN frame and `yaw` is how that frame is
    turned in the world. The support is then taken in the box's frame:

        u_local = ( u . e_x,  u . e_y )        e_x = (cos yaw, sin yaw), e_y = (-sin yaw, cos yaw)
        support = |u_local_x| * hx + |u_local_y| * hy

    THE YAW IS AN ARGUMENT BECAUSE OMITTING IT WAS A BUG, and an expensive one. This function used
    to take a bbox and two components, and both planners fed it the prim's WORLD AABB -- the
    axis-aligned box AROUND the oriented one. For anything yawed off the world axes that counts the
    rotation TWICE: once in inflating the AABB, and again in the support taken along a direction
    that is itself rotated.
    
    MEASURED ON KITCHEN 1218. Its chairs are 0.3832 m square, and face_chair_toward yaws each to
    look at its table, so each chair is SQUARE-ON to its own push and the true support is 0.1916 m
    for both. The old form returned 0.1916 for chair_1 (yaw -90 deg, axis-aligned, AABB tight) and
    0.3827 for chair_0 (yaw -47.31 deg) -- a 0.191 m difference invented entirely by the frame the
    box was measured in. push_prim_base_pose adds that to the standoff, so chair_0 parked 0.191 m
    further back than chair_1 behind an identical chair, which is what made one chair transfer 86%
    of its push and the other jam at 8%. See pushchair_task's SHIPPED_DATASET for the constants that
    were tuned against the broken number.
    
    A yaw of 0 reproduces the old behaviour exactly, which is what makes the axis-aligned tests
    above still hold: for an unrotated box its own frame IS the world's.
    
    THE SIGNATURE CHANGE IS THE POINT. Adding `yaw` with a default of 0.0 would have left every old
    four-argument call compiling and silently wrong; taking it positionally in the middle makes the
    old call a TypeError at the first frame instead.
    """
    hx = (float(local_max[0]) - float(local_min[0])) * 0.5
    hy = (float(local_max[1]) - float(local_min[1])) * 0.5
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    lx = float(ux) * c + float(uy) * s
    ly = -float(ux) * s + float(uy) * c
    return abs(lx) * hx + abs(ly) * hy


# =================================================================================================
# arm.push_pose -- where the two HANDS go, so the chair is pushed by its backrest and not shoved
# by the chassis at a leg.
# =================================================================================================

#: The hand orientation the 1600-episode pushchair dataset used, and the base heading it was
#: authored at. goals/Isaac-SceneSmith-001.json's second step is
#:
#:     ["A_b", [[2.576, 2.88, 0.85, 0.70711, -0.70711, 0.0, 0.0,
#:               2.776, 2.88, 0.85, 0.70711, -0.70711, 0.0, 0.0]]]
#:
#: with the N_s before it at yaw 1.5708 and the N_s after it 0.77 m further along +y. Both hands
#: carry the SAME quaternion, so there is one rotation here and not two.
#:
#: WHAT IT IS, read off rather than guessed. (w, x, y, z) = (cos 45, -sin 45, 0, 0) is Rx(-90 deg),
#: and applied to the ee_link1 frame -- whose +Z is the approach and whose +-X is the jaw travel
#: (skills.GRASP_TOOL_FRAME["anubis"]) -- it sends
#:
#:     ee +X -> world +x      the robot's RIGHT, since forward is +y at heading 1.5708
#:     ee +Y -> world -z      DOWN  (which is GRASP_TOOL_FRAME's y_down: True for this robot)
#:     ee +Z -> world +y      FORWARD, i.e. exactly the direction the following N_s drives
#:
#: So the reference is not an arbitrary rotation: it is "palm square to the push, jaws horizontal,
#: wrist rolled so the frame's +Y is down". That is why it can be carried to another heading at
#: all -- there is a relationship to preserve.
PUSH_HANDS_Q_REF = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)

#: The base heading PUSH_HANDS_Q_REF was authored against, in radians. The reference file writes
#: 1.5708; this is the pi/2 it is a five-figure rounding of, because the rounding is the goal
#: file's, not the geometry's.
PUSH_HANDS_Q_REF_HEADING = math.pi / 2.0


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions. Spelled out rather than imported: this
    module is stdlib-only on purpose (see the file docstring), and the alternative is a numpy
    dependency for eight multiplies."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def push_pose_quat(heading_rad: float):
    """The hand orientation for a push along `heading_rad`, as (w, x, y, z) in the world/env frame.

    THE REFERENCE, TURNED. PUSH_HANDS_Q_REF is a working orientation for ONE heading (pi/2) and
    every chair in this task has its own -- kitchen 1218 measured -0.826 for chair_0 and -1.571 for
    chair_1, because face_chair_toward aims each chair at the table and nav.push_prim takes its
    axis from the chair->table vector. What must be preserved across those headings is the
    RELATIONSHIP the reference has to the push direction (palm square to it, jaws horizontal, +Y
    down), and the rotation that carries one heading to another about the world's vertical is
    Rz(heading - PUSH_HANDS_Q_REF_HEADING). So:
    
        q(heading) = Rz(heading - pi/2) (x) q_ref
    
    PRE-multiplied, not post-. Rz here is a rotation of the WORLD-frame result -- it turns the
    already-oriented hand about the world's z -- and world-frame rotations compose on the left.
    Post-multiplying would instead spin the hand about its OWN z, which is the approach axis: the
    palm would keep facing +y whatever the chair's heading and the push would arrive sideways.
    
    THE ONE FIXED POINT is a test, not a comment: q(pi/2) is PUSH_HANDS_Q_REF exactly, so the
    heading the 1600 episodes were recorded at reproduces their quaternion bit for bit.
    
    WHY NOT DERIVE IT FROM THE TOOL FRAME INSTEAD. It can be -- (ee X, Y, Z) = (right, down,
    forward) is a complete basis and the quaternion follows from it -- and doing so gives the same
    rotation, which is what the reference-reproduction test above checks. What it would NOT give is
    the provenance: 1600 recorded episodes say this hand pose survives contact with a chair, and a
    basis assembled from first principles says only that it is self-consistent. The reference is
    kept as the anchor for that reason, with the basis as its explanation.
    """
    half = 0.5 * (float(heading_rad) - PUSH_HANDS_Q_REF_HEADING)
    return _quat_mul((math.cos(half), 0.0, 0.0, math.sin(half)), PUSH_HANDS_Q_REF)


def push_pose_hands(prim_xy, toward_xy, half_extent_along_u: float,
                    clearance_m: float, height_m: float, span_m: float):
    """Both hands, placed to press on `prim_xy`'s far face and drive it toward `toward_xy`.

    Returns the A_b payload the executor reads: 14 floats, `[:7]` the LEFT hand and `[7:14]` the
    RIGHT, each [x, y, z, qw, qx, qy, qz]. That layout is goal_format's (`_decode_arm_both`) and
    simvla_gen's (`arm_b_goals[:, :7]` / `[:, 7:14]`), and the flat list is returned rather than a
    (left, right) pair so that there is no second place for the halves to be concatenated in the
    wrong order.

    THE FRAME IS THE CALLER'S. Pass env-frame centres and the poses come back in the env frame,
    which is what an A_b goal has to be: simvla_gen subtracts env_origins when it FILLS a reset
    target and leaves an authored payload alone, so the payload is env-local. Identical to what
    push_prim_base_pose promises about its own return, and for the same reason -- during authoring
    the one kitchen on the stage sits at the env origin.

    THE GEOMETRY, on the same axis push_prim_base_pose uses so the hands and the base that carries
    them cannot disagree about which way the push goes:

        u    = unit(prim - toward)             outward, away from the target
        mid  = prim + u * (half_extent + clearance_m)
        p    = (u_y, -u_x)                     u turned -90 deg
        left = mid + p * span_m/2,  right = mid - p * span_m/2,  both at z = height_m

    `p` IS THE ROBOT'S LEFT, and that is the whole reason it is u turned MINUS 90 and not plus.
    The base drives along -u, so its heading is atan2(-u_y, -u_x) and its left hand side is that
    turned +90, i.e. (u_y, -u_x). Turning u the other way names the robot's RIGHT, which would put
    the left hand on the right of the push axis and cross the arms -- and the reference file
    settles it independently: with the base at x 2.676 facing +y its left hand is at x 2.576, on
    the -x side, which is the robot's left.

    `clearance_m` IS MEASURED PAST THE PRIM'S SUPPORTING PLANE, and it has to be positive because
    cuRobo's world contains the chair: a goal pose inside it is refused, and the step never plans.
    `half_extent_along_u` is the AABB's support along u, so the plane at prim + u*half_extent has
    the whole box behind it; every returned point projects onto u exactly `clearance_m` beyond that
    plane, whatever the span puts it sideways, because p . u == 0. It is a bound on the mesh, not
    the mesh, so the true gap to the backrest is clearance_m OR MORE.

    CONTACT IS THE BASE'S JOB, NOT THIS POSE'S -- which is why the clearance is positive rather
    than negative. The step order is A_b then N_s, exactly as the reference file has it: the hands
    go out to a pose that plans, and the push leg then drives the base (and the arms holding their
    configuration) into the chair.
    """
    dx = float(prim_xy[0]) - float(toward_xy[0])
    dy = float(prim_xy[1]) - float(toward_xy[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        raise ValueError(
            f"push target {tuple(prim_xy)} is on top of {tuple(toward_xy)}; there is no outward "
            f"direction to place hands along"
        )
    if float(clearance_m) <= 0.0:
        # Not a taste question. At 0.0 the hands sit ON the AABB's supporting plane and cuRobo
        # refuses the goal; below it they are authored inside the chair. Failing here names the
        # parameter, where failing there is a planner refusal an hour into a GPU job.
        raise ValueError(
            f"clearance_m={clearance_m}; the hands must be authored CLEAR of the prim -- cuRobo's "
            f"world contains it and refuses a goal in collision. The push leg makes the contact."
        )
    if float(span_m) < 0.0:
        raise ValueError(
            f"span_m={span_m}; a negative span swaps the hands across the push axis and crosses "
            f"the arms. Left is left."
        )

    ux, uy = dx / n, dy / n
    stand = float(half_extent_along_u) + float(clearance_m)
    mid_x = float(prim_xy[0]) + ux * stand
    mid_y = float(prim_xy[1]) + uy * stand
    # u turned -90 degrees: the robot's LEFT once it is yawed to the push heading. See above.
    px, py = uy, -ux
    reach = 0.5 * float(span_m)
    z = float(height_m)

    # The SAME expression push_prim_base_pose builds its yaw from, and for the same reason: `t - p`
    # is +0.0 when the two agree, where negating a component would give -0.0 and flip atan2's pi to
    # -pi. The base pose and the hand orientation MUST come from one heading -- a hand square to
    # one direction while the base drives along another is a push that arrives at an angle.
    yaw = math.atan2(float(toward_xy[1]) - float(prim_xy[1]),
                     float(toward_xy[0]) - float(prim_xy[0]))
    qw, qx, qy, qz = push_pose_quat(yaw)

    return [mid_x + px * reach, mid_y + py * reach, z, qw, qx, qy, qz,
            mid_x - px * reach, mid_y - py * reach, z, qw, qx, qy, qz]


def support_along_points(points_xy, centre_xy, ux: float, uy: float) -> float:
    """How far `points_xy` reach past `centre_xy` in the direction (ux, uy). Exact, and frameless.

    THE FRAMELESS ONE, and that is the whole reason it exists. half_extent_along needs to be told
    which way the box is turned, and on kitchen 1218 the planner told it 0.0 for a chair whose prim
    carries a +42.687 deg orient -- because it was reading the frame UsdGeom.BBoxCache's world bound
    comes back in (identity, range already world-aligned), not the prim's own transform, and the log
    printed that under the label "prim yaw". A support taken over the points themselves cannot be
    defeated by any of that: wherever the rotation lives, the points are already where they are.

    THE ROTATION IS ON THE PRIM, NOT IN THE VERTICES. This docstring used to say the USD export bakes
    it into the vertices and leaves the prim identity. It does not -- /world/chair_0 in the shipped
    kitchen_1218_00.usd carries xformOp:translate + xformOp:orient (0.9314151, 0, 0, 0.36395872) and
    its child mesh inherits that -- and believing otherwise sent the campaign hunting a quaternion
    convention bug in env_cfg_emit that does not exist. See test_chair_orientation.py.

    IT IS ALSO TIGHTER THAN ANY BOX. Measured on this task's chair (112 vertices): along its own
    axes the mesh gives 0.1916, the same as the box; along a diagonal it gives 0.2344 where the
    box's own support gives 0.2710 and the world AABB's gives 0.3827. A box is a bound; this is the
    thing itself.

    SIGNED, NOT SYMMETRIC. push_prim_base_pose stands the base at `centre + u * (support + ...)`,
    so what it needs is how far the prim reaches on the +u SIDE -- not the larger of the two sides.
    A chair whose backrest overhangs its legs at the back is exactly the case where those differ,
    and the base has to clear the back, not the average.

    Refuses an empty point set rather than returning 0.0, which would park the base inside the prim.
    """
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    best = None
    for p in points_xy:
        d = (float(p[0]) - cx) * float(ux) + (float(p[1]) - cy) * float(uy)
        if best is None or d > best:
            best = d
    if best is None:
        raise ValueError(
            "support_along_points was given no points; the prim's geometry could not be read, and "
            "returning 0.0 would stand the base at the prim's own centre"
        )
    return best


def axis_offset_deg(ux: float, uy: float) -> float:
    """How far the direction (ux, uy) is from the nearest world axis, in degrees: 0 to 45.

    The condition the extent fallback is refused on. A box measured at a yaw of 0 is the WORLD AABB,
    and for an axis-aligned push the AABB is tight and that fallback is exactly right; for anything
    off-axis it is inflated -- by (|cos|+|sin|)^2 on a square, i.e. up to double at 45 degrees. So
    this is the number that separates "the fallback is fine" from "the fallback is a known lie".

    Folded to a quarter turn because a box has four equivalent axes: +x, +y, -x and -y are all
    "aligned", and so is any direction within a degree of them.
    """
    off = math.degrees(math.atan2(float(uy), float(ux))) % 90.0
    return min(off, 90.0 - off)


# =================================================================================================
# HOW HIGH THE HANDS GO. The chair says how high it CAN be pressed; the arm says how low it CAN
# reach holding the push orientation. A height that satisfies only one of those is the defect this
# section exists to remove -- see ARM_REACH_FLOOR_M.
# =================================================================================================

#: THE ARM'S FLOOR: the lowest z, in metres above the floor, at which Anubis can hold the EXACT
#: push orientation (palm square to the push, jaws horizontal) with its hand `forward` metres in
#: front of base_link. Each row is (forward_m, lowest_z_m).
#:
#: MEASURED, TWICE, INDEPENDENTLY. Damped-least-squares IK over the arm1 chain read straight out of
#: anubis/anubis_final.urdf (arm1_base_link_joint at z 0.82336, then 0.062 / 0.300 / 0.320 / 0.040 /
#: 0.027 link offsets and ee_fixed_joint1's 0.10956), joint limits clamped, converged to 2 mm and
#: 1 degree, 6 seeds per sample, warm-started down the z sweep. It reproduces the table
#: pushchair_task.PUSH_HEIGHT_M's comment recorded from the earlier campaign row for row.
#:
#: WHY IT IS A FLOOR AND NOT A PREFERENCE. A previous collector build set cuRobo's endpoint
#: thresholds to 10 m / 10 rotation-metric units, so an unreachable goal could be returned as a
#: successful nearest configuration. The collector now uses 0.02 m / 0.05 thresholds and rejects
#: such a path. This authoring-side floor remains useful: it rejects an impossible press before a
#: GPU run instead of leaving the user with a planner failure or a retry loop. At the shipped
#: 0.60 m with the hands 0.25 m forward, the old nearest configuration was 0.22 m below the floor,
#: with the elbow over the shoulder and palm pitched 17 degrees over. This table prevents that
#: unreachable pose from being authored at all.
#:
#: THE SHAPE OF IT. The shoulder sits at 0.823 m, so a hand held close in front of the base cannot
#: get below shoulder height at all without breaking the wrist out of the orientation; extending
#: forward buys downward room fast, and past ~0.40 m the limit stops being the wrist and starts
#: being total reach. That is why the fix for "the hands must go lower" is to stand the base
#: FURTHER BACK, not to ask the arm for more.
ARM_REACH_FLOOR_M = (
    (0.20, 0.86),
    (0.25, 0.82),
    (0.30, 0.78),
    (0.35, 0.70),
    (0.40, 0.48),
    (0.45, 0.48),
)


def arm_reach_floor_m(forward_m: float) -> float:
    """The lowest z the hand can hold the push orientation at, `forward_m` in front of base_link.

    Linear between the measured rows. OUTSIDE them it does not extrapolate -- it clamps -- and the
    two ends clamp for different reasons, so neither is a convenience:

      * Below 0.20 m forward the arm has no solution at ANY height (the sweep found none), so
        returning the 0.20 row's 0.86 is the honest answer: nothing lower is reachable there
        either. Extrapolating the slope upward would invent a floor above the shoulder.
      * Above 0.45 m the limit stops being the wrist and becomes total reach, which curves back
        UP (0.50 measured 0.50, 0.55 measured 0.54). Extrapolating the 0.35->0.40 slope downward
        would promise 0.0 at 0.46 m -- a floor at the floor -- which is the exact shape of error
        this whole table exists to stop.
    """
    f = float(forward_m)
    rows = ARM_REACH_FLOOR_M
    if f <= rows[0][0]:
        return rows[0][1]
    if f >= rows[-1][0]:
        return rows[-1][1]
    for (f0, z0), (f1, z1) in zip(rows, rows[1:]):
        if f0 <= f <= f1:
            return z0 + (z1 - z0) * (f - f0) / (f1 - f0)
    raise AssertionError("ARM_REACH_FLOOR_M is not sorted by forward distance")


def min_hands_forward_for_height(height_m: float) -> float | None:
    """The smallest hand-forward distance at which `height_m` is inside the arm's envelope.

    None when no row reaches that low: the height is under the arm's global floor and no standoff
    saves it. Returning None rather than the largest forward distance is the point -- a caller that
    got 0.45 back would stand the base 20 cm further away and STILL plunge, having been told the
    problem was solved.

    Read off the table rather than inverted analytically, then refined linearly inside the bracket
    it lands in, so it is exactly consistent with arm_reach_floor_m: the two disagreeing by even a
    millimetre would let a height pass this check and fail that one.
    """
    h = float(height_m)
    rows = ARM_REACH_FLOOR_M
    if h >= rows[0][1]:
        return rows[0][0]
    for (f0, z0), (f1, z1) in zip(rows, rows[1:]):
        # The table descends; the bracket is the first pair straddling h.
        if z1 <= h <= z0:
            if abs(z0 - z1) < 1e-12:
                return f0
            return f0 + (f1 - f0) * (z0 - h) / (z0 - z1)
    return None


def _triangle_plane_crossing(tri_tsz, s_plane: float):
    """Where one triangle, given as three (t, s, z) tuples, crosses the plane s = `s_plane`.

    Returns [] or [(t, z), (t, z)] -- the endpoints of the segment the plane cuts out of it.

    A PLANE AND NOT A POINT TEST, and that distinction is the whole reason this function exists.
    The first version of press_bands asked "is there a VERTEX within 3 cm of where the hand goes",
    and measured the mesh's tessellation rather than its surface: a folding chair's backrest is a
    flat panel carrying four corner vertices and NOTHING at the hand's lateral offset, so the panel
    read as air at every height but the one row where an edge happened to pass. Measured on
    b43f9098: the point test found the panel at z 0.84 and missed it at 0.81, 0.75 and 0.72, all of
    which are solid. A triangle either crosses the hand's centreline or it does not, whatever its
    vertices are doing.

    Degenerate cases are dropped rather than special-cased: a triangle lying IN the plane has all
    three vertices on it and contributes no crossing, which is right -- a panel edge-on to the hand
    is not a face to press. A triangle touching it at one vertex likewise.
    """
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ta, sa, za = tri_tsz[a]
        tb, sb, zb = tri_tsz[b]
        da, db = sa - s_plane, sb - s_plane
        if da == 0.0 and db == 0.0:
            continue
        if (da < 0.0 and db < 0.0) or (da > 0.0 and db > 0.0):
            continue
        if da == db:
            continue
        f = da / (da - db)
        if f < 0.0 or f > 1.0:
            continue
        out.append((ta + (tb - ta) * f, za + (zb - za) * f))
        if len(out) == 2:
            return out
    return []


def press_bands_from_triangles(triangles, prim_xy, toward_xy, span_m: float,
                               slab_m: float = 0.03, depth_tol_m: float = 0.08):
    """The height bands where BOTH hands would land on the mesh, not on air.

    `triangles` is an iterable of ((x, y, z), (x, y, z), (x, y, z)) in the caller's frame -- the
    env frame at authoring time, exactly as push_pose_hands' inputs and outputs are. Returns a list
    of (z_lo, z_hi) in ascending order.

    THE QUESTION IT ANSWERS is not "how tall is the chair" -- the AABB answers that, and answering
    it is what put the reference goal file's hands 36 mm above the shipped chair's highest point.
    It is "at this height, is there material under EACH hand, on the face they will arrive at".
    Three things have to hold for a slab of height to qualify, and each is a way a cruder test is
    fooled:

      1. The mesh crosses the plane at lateral offset +span/2, AND the one at -span/2. A
         spindle-back chair has plenty of material at the right height and nothing at all under a
         hand at +-0.08 -- it passes between the spindles. One hand landing is not enough either:
         a single contact off the axis is a couple, which is the yaw that costs the chassis push a
         third of its travel.
      2. That material within `depth_tol_m` of the prim's own support plane along the push axis. A
         chair's SEAT is material at seat height under both hands and sits a third of a metre
         forward of the backrest; hands authored just clear of the support plane would never reach
         it and the base would arrive alone. Depth is what separates a face that can be pressed
         from material that merely exists at that height.
      3. `slab_m` of vertical resolution, and a slab is judged on the crossings that fall in it.

    FRAMELESS, exactly as support_along_points is and for the same reason: the rotation of a USD
    prim may live on the prim, on a parent, or baked into the vertices, and every one of those has
    been true of a chair in this library. Points are where they are.
    """
    dx = float(prim_xy[0]) - float(toward_xy[0])
    dy = float(prim_xy[1]) - float(toward_xy[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        raise ValueError(
            f"press band: {tuple(prim_xy)} is on top of {tuple(toward_xy)}; there is no push axis "
            f"to measure a face against"
        )
    if float(span_m) < 0.0:
        raise ValueError(f"span_m={span_m}; a negative span crosses the hands. Left is left.")
    if float(slab_m) <= 0.0:
        raise ValueError(f"slab_m={slab_m}; the vertical resolution must be positive")

    ux, uy = dx / n, dy / n
    px, py = uy, -ux                      # push_pose_hands' lateral, by the same expression
    cx, cy = float(prim_xy[0]), float(prim_xy[1])
    reach = 0.5 * float(span_m)
    slab = float(slab_m)

    tris = []
    support = None
    for tri in triangles:
        conv = []
        for q in tri:
            vx, vy = float(q[0]) - cx, float(q[1]) - cy
            t = vx * ux + vy * uy          # outward, away from the target
            s = vx * px + vy * py          # lateral, + is the robot's left
            conv.append((t, s, float(q[2])))
            if support is None or t > support:
                support = t
        tris.append(conv)
    if support is None:
        raise ValueError(
            "press band: the prim exposed no triangles. An empty band would read as 'this chair "
            "cannot be pressed anywhere', which is indistinguishable from 'nobody looked'."
        )

    floor_t = support - float(depth_tol_m)
    hits = {}
    for hand, s_plane in ((0, reach), (1, -reach)):
        for tri in tris:
            seg = _triangle_plane_crossing(tri, s_plane)
            if len(seg) != 2:
                continue
            (t0, z0), (t1, z1) = seg
            if max(t0, t1) < floor_t:
                continue
            k0 = int(math.floor(min(z0, z1) / slab))
            k1 = int(math.floor(max(z0, z1) / slab))
            for k in range(k0, k1 + 1):
                # The segment's own t at this slab, not the segment's maximum: a backrest that
                # leans back is nearer the support plane at the top than at the bottom, and taking
                # the maximum would credit every slab it spans with the top's depth.
                if z1 == z0:
                    t_here = max(t0, t1)
                else:
                    zc = min(max((k + 0.5) * slab, min(z0, z1)), max(z0, z1))
                    t_here = t0 + (t1 - t0) * (zc - z0) / (z1 - z0)
                if t_here >= floor_t:
                    got = hits.setdefault(k, [False, False])
                    got[hand] = True

    good = sorted(k for k, (l, r) in hits.items() if l and r)
    bands = []
    for k in good:
        if bands and k == bands[-1][1]:
            bands[-1][1] = k + 1
        else:
            bands.append([k, k + 1])
    return [(lo * slab, hi * slab) for lo, hi in bands]


def choose_push_height(triangles, prim_xy, toward_xy, span_m: float, hands_forward_m: float,
                       top_margin_m: float = 0.05, slab_m: float = 0.03,
                       depth_tol_m: float = 0.08):
    """How high to press, and why. Returns (height_m, note).

    AS HIGH AS THE PRIM ALLOWS, WHICH IS THE POINT. The bands the hands can press are taken from
    the prim's own triangles, the HIGHEST band is chosen, and the hands go `top_margin_m` below
    the prim's top -- clamped INTO that band, never above it and never below it. That is "just
    enough to push it": the arm never reaches further down than the thing being pushed makes it.

    THE DEFECT THIS REPLACES. `height_m` was a constant, 0.60, measured once against one chair's
    occupancy grid. Two independent things have to be true of a push height and a constant can only
    ever have checked one of them: it must be ON the chair, and it must be somewhere the arm can
    hold the push orientation. 0.60 was on that chair and 0.22 m under the arm's floor for the
    standoff it was used at. The old 10 m / 10 rotation-metric cuRobo thresholds accepted the
    nearest configuration instead of refusing it: elbow over the shoulder, palm pitched 17 degrees
    over, the whole arm diving at the chair. The collector now rejects that bad endpoint; this
    function moves the rejection earlier, while authoring the goal.

    THE TRADE IT MAKES, named rather than left implicit: a horizontal push at height h tips instead
    of sliding once friction exceeds d/h, d being the plan-view distance from the contact line to
    the chair's front edge -- so pressing high costs tipping margin. What makes it the right call
    anyway is that the alternative was never a lower HAND. It was the same hand at the same height
    with the arm folded under itself, because the goal was below the floor. The tipping margin is
    identical either way; only the arm's posture differs. If a run does tip, the knobs are
    `top_margin_m` and the chair, and the chairtrace's `tilt max / z min` fields are what say so.

    REFUSES rather than clamps when the best band is below the arm's floor, and the message carries
    the standoff that WOULD reach it -- because the fix there is to stand the base further back
    (nav.push_prim's `safety`), not to accept a plunging arm, and a caller told only "too low" will
    reach for the wrong knob.
    """
    bands = press_bands_from_triangles(triangles, prim_xy, toward_xy, span_m,
                                       slab_m=slab_m, depth_tol_m=depth_tol_m)
    if not bands:
        raise ValueError(
            f"press band: no height has material under BOTH hands at +-{0.5 * float(span_m):.3f} m "
            f"on the face this push arrives at. Either the span is wider than the thing being "
            f"pushed, or its pressable face is not on the push axis -- a chair yawed 90 degrees "
            f"presents its side, which is mostly air between the legs. A chassis push "
            f"(nav.push_prim with no arm.push_pose leg) is what such a shape supports."
        )
    z_lo, z_hi = bands[-1]
    z_top = max(float(q[2]) for tri in triangles for q in tri)
    # The margin is measured from the PRIM'S top, not the band's: what it buys is that the hands
    # cannot ride over the top edge, and that edge belongs to the chair. Clamped into the band
    # afterwards, because a band is where the hands actually touch -- a height just under the top
    # of a chair whose top 3 cm is the only pressable part must land IN those 3 cm.
    height = min(z_top - float(top_margin_m), z_hi - 0.5 * float(slab_m))
    height = max(height, z_lo + 0.5 * float(slab_m))
    floor = arm_reach_floor_m(hands_forward_m)
    # LIFTED TO THE FLOOR WHEN THE BAND STILL HAS ROOM, before any refusal. The top margin is a
    # preference -- keep the hands off the top edge -- and the arm's floor is a hard limit, so on a
    # chair where they disagree and the band reaches above the floor, the answer is the floor, not
    # a refusal. Without this a chair with a 0.75-0.90 m band and a 0.88 m arm floor would be turned
    # away for the sake of a 5 cm preference, which is the same over-strict shape as the constant
    # this function replaced -- just failing safe instead of failing quiet.
    lifted = ""
    if height < floor <= z_hi - 0.5 * float(slab_m):
        lifted = f" LIFTED from {height:.3f} to clear the arm's floor"
        height = floor
    if height < floor:
        want = min_hands_forward_for_height(height)
        cure = (f"standing the base back so the hands sit {want:.3f} m in front of it "
                f"(nav.push_prim safety {want + 0.10 - 0.23:.3f} at clearance_m 0.10) reaches it"
                if want is not None else
                "no standoff reaches it: this height is below the arm's floor at every forward "
                "distance measured")
        raise ValueError(
            f"press band: the highest face this prim offers both hands is [{z_lo:.3f}, {z_hi:.3f}] "
            f"(prim top {z_top:.3f}), so the hands would go to {height:.3f} m -- but "
            f"{hands_forward_m:.3f} m in front of the base the arm cannot hold the push "
            f"orientation below {floor:.3f} m. The collector's strict cuRobo endpoint gate will "
            f"reject this pose instead of accepting a distant nearest configuration. {cure}."
        )
    note = (f"press bands {[(round(a, 3), round(b, 3)) for a, b in bands]} top={z_top:.3f} "
            f"-> height {height:.3f} (margin {float(top_margin_m):.3f}, arm floor {floor:.3f} at "
            f"{hands_forward_m:.3f} m forward){lifted}")
    return height, note


#: The yaw corrections best_press_yaw will consider, in radians. Quarter turns only.
#:
#: WHY QUARTER TURNS AND NOT A SEARCH. The chair is already SEATED -- a layout put it at a pose that
#: is square to the table and clear of the furniture, and that pose is not in question. What is in
#: question is which of the chair's own faces is presented, and a chair's faces are its own frame's
#: axes. Anything between two of them would leave the chair standing askew at its table for the sake
#: of a face that is not there.
PRESS_YAW_CANDIDATES = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)


def _rotate_triangles_about(triangles, centre_xy, yaw: float):
    """`triangles` turned `yaw` radians about the vertical line through `centre_xy`."""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    out = []
    for tri in triangles:
        out.append(tuple(
            (cx + (float(q[0]) - cx) * c - (float(q[1]) - cy) * s,
             cy + (float(q[0]) - cx) * s + (float(q[1]) - cy) * c,
             float(q[2]))
            for q in tri))
    return out


def best_press_yaw(triangles, prim_xy, toward_xy, span_m: float, hands_forward_m: float,
                   slab_m: float = 0.03, depth_tol_m: float = 0.08):
    """Which quarter turn presents this prim's best pressable face to the push. Returns (yaw, rows).

    `rows` is one (yaw, top_of_highest_band_or_None) per candidate, so the caller can log what it
    chose over. `yaw` is the winner: the candidate whose highest two-handed band tops out highest,
    ties broken toward 0.0 so a prim that is already right is never turned.

    THE HEURISTIC THIS REPLACES, AND HOW IT FAILED. kitchen_build.face_chair_toward aims a chair by
    facing_direction: the centroid of the mesh's upper half against the centroid of the whole, the
    offset pointing away from the backrest. It is skipped entirely below FACING_CONFIDENCE_CUT
    (0.02) because on a near-symmetric chair that offset is noise. MEASURED on b43f9098, a folding
    chair with an unmistakable solid backrest: the offset is 0.0100 m on a 0.554 m extent, so the
    confidence is 0.0181 -- UNDER the cut -- and the chair is left exactly as the seat layout
    dropped it, 99 degrees off the table. Nothing reports this. The chair looks seated, the push
    plans, and the hands close on the gap between the seat and the backrest.

    WHY THE HEURISTIC UNDERSCORES A CHAIR THAT OBVIOUSLY HAS A BACK: it is a centroid over
    VERTICES, unweighted by area. A folding chair's backrest is one flat panel carrying four
    corners, and the legs and seat frame carry hundreds of vertices between them, so the panel
    contributes almost nothing to the mean. It is the same mistake in the same shape as the vertex
    press test this module used to have -- both were measuring tessellation.

    SO THIS DOES NOT GUESS WHERE THE BACK IS. It asks the question that actually matters, four
    times: at this yaw, how high can both hands press on the face the push arrives at? The tallest
    answer IS the backrest, on any chair that has one, because nothing else on a chair is both high
    and broad. On a chair that has none -- a stool -- every candidate scores its seat and the
    winner is a tie broken toward 0.0, which leaves the seating alone.

    `hands_forward_m` is accepted but not used to REJECT candidates here: this function's job is to
    turn the chair to its best face, and whether even that face is high enough for the arm is
    choose_push_height's refusal to make, with its own message about standoffs. Filtering here
    would turn "this chair is too short" into "no yaw is any good", which names the wrong problem.
    """
    rows = []
    for yaw in PRESS_YAW_CANDIDATES:
        turned = _rotate_triangles_about(triangles, prim_xy, yaw)
        try:
            bands = press_bands_from_triangles(turned, prim_xy, toward_xy, span_m,
                                               slab_m=slab_m, depth_tol_m=depth_tol_m)
        except ValueError:
            bands = []
        rows.append((yaw, bands[-1][1] if bands else None))
    best = None
    for yaw, top in rows:
        if top is None:
            continue
        if best is None or top > best[1] + 1e-9:
            best = (yaw, top)
    return (0.0 if best is None else best[0]), rows
