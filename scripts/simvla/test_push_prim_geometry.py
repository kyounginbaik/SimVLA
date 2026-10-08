"""nav.push_prim's base-pose geometry, with no Omniverse and no stage.

The planner body has to live in simvla_data_generator.py -- that is where @register_planner
injects it, and where the stage is -- and that module boots Omniverse at import, so nothing in it
can be unit-tested. The GEOMETRY is therefore a pure function taking numbers and returning numbers,
and this file gates it on CPU. What stays in the planner is two bbox lookups feeding these
arguments.
"""

import math

import pytest

from push_prim_geometry import half_extent_along, push_prim_base_pose

WHEEL_R = 0.23
SAFETY = 0.12
HALF = 0.19          # a chair's half-extent along the push axis


def test_the_base_parks_on_the_far_side_of_the_chair_from_the_target():
    """The whole point. Park between chair and table and the push drives the chair AWAY."""
    # table at the origin, chair 1.5 m along +y -> the base must be BEYOND the chair, at y > 1.5
    x, y, _yaw = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    assert y > 1.5, f"base at y={y:.3f} is on the table's side of a chair at y=1.5"
    assert y == pytest.approx(1.5 + HALF + WHEEL_R + SAFETY)
    assert x == pytest.approx(0.0)


def test_the_base_faces_the_target():
    """yaw points from the base toward the table, i.e. the direction the push travels."""
    _x, _y, yaw = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    assert yaw == pytest.approx(math.radians(-90.0))          # facing -y, toward the table
    _x, _y, yaw = push_prim_base_pose((1.5, 0.0), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    assert yaw == pytest.approx(math.radians(180.0))          # facing -x


def test_offset_moves_the_base_toward_the_target_along_the_same_axis():
    """The push leg is the SAME pose with offset_m subtracted from the standoff.

    Same line and same yaw is what makes the push straight: a second pose computed from a different
    direction would turn the base mid-push and slew the chair off it.
    """
    a = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    b = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.60)
    assert b[1] == pytest.approx(a[1] - 0.60)
    assert b[0] == pytest.approx(a[0])
    assert b[2] == pytest.approx(a[2])


def test_a_diagonal_chair_is_handled_on_its_own_axis():
    """The failure nav.to_prim has: a chair on a diagonal matches no canonical snap and gets (0,0,0).

    Here the axis is computed, so a 45-degree chair is no different from an axis-aligned one.
    """
    d = math.sqrt(0.5)
    x, y, yaw = push_prim_base_pose((d, d), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    stand = 1.0 + HALF + WHEEL_R + SAFETY
    assert x == pytest.approx(d * stand)
    assert y == pytest.approx(d * stand)
    assert yaw == pytest.approx(math.radians(-135.0))


def test_a_negative_offset_backs_the_base_out_along_the_same_axis_for_the_retreat():
    """The retreat leg. `stand` subtracts offset_m, so a NEGATIVE offset adds to the standoff.

    Same line and same heading as the approach and the push, so the base backs straight off the
    chair rather than arcing around it -- and the clearance it opens is `safety + RETREAT_M`, with
    the half-extent cancelling, which is what lets one constant serve every chair size.
    """
    RETREAT = 0.25
    approach = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)
    retreat = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), HALF, WHEEL_R, SAFETY, -RETREAT)

    assert retreat[1] == pytest.approx(approach[1] + RETREAT)   # further from the table
    assert retreat[0] == pytest.approx(approach[0])
    assert retreat[2] == pytest.approx(approach[2])             # unchanged heading

    contact = HALF + WHEEL_R                                    # centres touching
    assert (retreat[1] - 1.5) - contact == pytest.approx(SAFETY + RETREAT)
    assert SAFETY + RETREAT > WHEEL_R, "cannot rotate in place without re-touching the chair"


def test_a_chair_on_top_of_the_target_is_refused_rather_than_dividing_by_zero():
    """No outward direction exists, and a silent (0,0,0) is the bug this whole skill replaces."""
    with pytest.raises(ValueError):
        push_prim_base_pose((0.0, 0.0), (0.0, 0.0), HALF, WHEEL_R, SAFETY, 0.0)


def test_half_extent_is_the_boxs_support_along_the_axis():
    """A wide chair must be cleared by more than a narrow one, so the standoff reads the box.

    Along +x a 0.8 x 0.4 box gives 0.4; along the diagonal it gives 0.4*d + 0.2*d. At yaw 0 the
    box's own frame IS the world's, so this is the arithmetic the function always had.
    """
    lo, hi = (-0.4, -0.2), (0.4, 0.2)
    assert half_extent_along(lo, hi, 0.0, 1.0, 0.0) == pytest.approx(0.4)
    assert half_extent_along(lo, hi, 0.0, 0.0, 1.0) == pytest.approx(0.2)
    d = math.sqrt(0.5)
    assert half_extent_along(lo, hi, 0.0, d, d) == pytest.approx(0.4 * d + 0.2 * d)
    assert half_extent_along(lo, hi, 0.0, -1.0, 0.0) == pytest.approx(0.4)   # sign-independent


#: Kitchen 1218's chair, from the chair manifest: 0.3832 m square in plan.
CHAIR_HALF = 0.3832 / 2


def _push_axis(yaw):
    """The chair's own forward direction in world, which IS its push axis.

    face_chair_toward yaws every chair to look at its table, and nav.push_prim takes the push axis
    from the chair->table vector, so the two are the same direction by construction. That is what
    makes the test below a statement about the same physical push seen from two yaws.
    """
    return math.cos(yaw), math.sin(yaw)


def test_the_SAME_CHAIR_AT_ANY_YAW_HAS_THE_SAME_EXTENT_ALONG_ITS_OWN_PUSH():
    """THE BUG THIS FUNCTION WAS CHANGED FOR, and it fails against the old four-argument form.

    A chair is the same chair whichever way it is turned. Measured in the world AABB it is not:
    kitchen 1218 seats two copies of one 0.3832 m square mesh, and the old code returned 0.1916 for
    the one at yaw -90 and 0.3827 for the one at yaw -47.31 -- a 0.191 m difference invented by the
    frame the box was measured in, and added straight onto one chair's standoff.

    Asserted across a sweep and against the ONE true value, not merely as "the two agree": two
    equally wrong numbers agree too.
    """
    lo, hi = (-CHAIR_HALF, -CHAIR_HALF), (CHAIR_HALF, CHAIR_HALF)
    for yaw_deg in (0.0, -47.31, -90.0, -131.17, 17.0, 45.0, 180.0, 359.0):
        yaw = math.radians(yaw_deg)
        got = half_extent_along(lo, hi, yaw, *_push_axis(yaw))
        assert got == pytest.approx(CHAIR_HALF, abs=1e-9), (
            f"a chair yawed {yaw_deg} deg measures {got:.4f} along its own push, not "
            f"{CHAIR_HALF:.4f}; the extent is being read in the wrong frame")


def test_the_world_AABB_is_what_it_used_to_return_and_it_is_bigger():
    """The artefact, stated as a number so the fix is not merely asserted to have happened.

    The AABB around a square yawed by psi has half-extents a(|cos|+|sin|) on BOTH axes, and taking
    the support of THAT along a direction which is itself yawed multiplies the factor again -- so
    at 45 degrees it doubles. The planner logs both values side by side for this reason.
    """
    a = CHAIR_HALF
    for yaw_deg in (-47.31, 45.0, -131.17):
        yaw = math.radians(yaw_deg)
        ux, uy = _push_axis(yaw)
        infl = a * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
        aabb_support = half_extent_along((-infl, -infl), (infl, infl), 0.0, ux, uy)
        assert aabb_support > half_extent_along((-a, -a), (a, a), yaw, ux, uy)
        assert aabb_support == pytest.approx(a * (abs(math.cos(yaw)) + abs(math.sin(yaw))) ** 2)
    # and at an axis-aligned yaw there is no artefact at all -- which is why chair_1 was correct
    assert half_extent_along((-a, -a), (a, a), math.radians(-90.0), *_push_axis(math.radians(-90.0))
                             ) == pytest.approx(a)


def test_a_non_square_box_still_reads_its_own_two_extents():
    """The oriented support is not "always the x half-extent" -- it is a support function, and a
    push that meets a rectangular chair on the diagonal of its OWN frame still gets both terms."""
    lo, hi = (-0.4, -0.2), (0.4, 0.2)
    yaw = math.radians(30.0)
    # a push along the box's own +y, seen in the world
    ux, uy = math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)
    assert half_extent_along(lo, hi, yaw, ux, uy) == pytest.approx(0.2)
    # and along the box's own diagonal
    d = math.sqrt(0.5)
    ux, uy = math.cos(yaw + math.pi / 4), math.sin(yaw + math.pi / 4)
    assert half_extent_along(lo, hi, yaw, ux, uy) == pytest.approx(0.4 * d + 0.2 * d)


# =================================================================================================
# arm.push_pose -- the two HANDS. Same file, because it is the same axis: both planners take
# u = unit(chair - table) from the same two bboxes, and a hand square to one direction while the
# base drives along another is a push that arrives at an angle.
# =================================================================================================

from push_prim_geometry import (PUSH_HANDS_Q_REF, PUSH_HANDS_Q_REF_HEADING,  # noqa: E402
                                push_pose_hands, push_pose_quat)

HEIGHT = 0.60
SPAN = 0.16
CLEAR = 0.10


def _R(q):
    """(w, x, y, z) -> the 3x3 rotation, as three COLUMNS: the images of ee +X, +Y, +Z.

    Written out rather than imported: this file gates a stdlib-only module and scipy would make
    the gate heavier than the thing it gates.
    """
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),      # row 0
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),      # row 1
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),      # row 2
    )


def _col(q, j):
    R = _R(q)
    return (R[0][j], R[1][j], R[2][j])


def _hands(prim, toward, half=HALF, clearance=CLEAR, height=HEIGHT, span=SPAN):
    p = push_pose_hands(prim, toward, half, clearance_m=clearance, height_m=height, span_m=span)
    return p, p[:7], p[7:14]


def test_the_payload_is_fourteen_floats_split_left_then_right():
    """The layout goal_format._decode_arm_both and simvla_gen both read: [:7] LEFT, [7:14] RIGHT.

    A short payload is not an error the executor reports -- it zero-pads to 14 and commands the
    right arm to the env origin -- so the length is asserted here, where it can be seen.
    """
    payload, left, right = _hands((0.0, 1.5), (0.0, 0.0))
    assert len(payload) == 14, payload
    assert all(isinstance(v, float) for v in payload), payload
    assert len(left) == 7 and len(right) == 7
    assert left[3:7] == right[3:7], "both hands take ONE orientation, as the reference file does"


def test_the_hands_are_symmetric_about_the_push_axis_and_span_apart():
    """The couple that stops the chair yawing is what two hands buy over one chassis corner, and it
    needs them equally placed either side of the line the push travels along."""
    payload, left, right = _hands((0.0, 1.5), (0.0, 0.0))

    # span apart
    sep = math.hypot(left[0] - right[0], left[1] - right[1])
    assert sep == pytest.approx(SPAN)

    # and their midpoint is ON the chair->table line, i.e. symmetric about it
    mid = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    assert mid[0] == pytest.approx(0.0)                       # the axis here is x = 0
    assert mid[1] == pytest.approx(1.5 + HALF + CLEAR)


def test_both_hands_sit_at_height_m():
    _payload, left, right = _hands((0.0, 1.5), (0.0, 0.0), height=0.72)
    assert left[2] == pytest.approx(0.72)
    assert right[2] == pytest.approx(0.72)


def test_every_hand_clears_the_prims_supporting_plane_by_clearance_m():
    """WHY THE STEP PLANS AT ALL. cuRobo's world contains the chair and refuses a goal inside it,
    so the pose has to be authored clear and the PUSH leg makes the contact.

    Asserted as a projection onto u, not as a distance from the centre, because that is the
    property that survives the span: half_extent_along is the ORIENTED box's support along u, so the plane
    at centre + u*half has the whole box behind it, and p . u == 0 means the sideways offset moves
    a hand ALONG that plane. Both hands are therefore exactly clearance_m beyond it however wide
    the span goes.
    """
    # a diagonal chair, so nothing is accidentally axis-aligned
    prim, toward = (1.0, 1.0), (0.0, 0.0)
    d = math.sqrt(0.5)
    for span in (0.0, 0.16, 0.60):
        _payload, left, right = _hands(prim, toward, span=span)
        for hand in (left, right):
            along_u = (hand[0] - prim[0]) * d + (hand[1] - prim[1]) * d
            assert along_u == pytest.approx(HALF + CLEAR), (span, hand)


def test_the_left_hand_is_on_the_robots_left_and_not_across_the_axis():
    """Crossed arms are the failure a sign error here produces, and it is silent: the payload is
    still 14 well-formed floats.

    The reference file settles which side is which without any derivation -- base at x 2.676 facing
    +y, left hand at x 2.576 -- so the left hand is on the -x side, the robot's left. Checked here
    against push_prim_base_pose's OWN heading rather than against a remembered sign.
    """
    for prim, toward in (((0.0, 1.5), (0.0, 0.0)),
                         ((1.0, 1.0), (0.0, 0.0)),
                         ((-0.7, 2.2), (1.3, 0.4))):
        _x, _y, yaw = push_prim_base_pose(prim, toward, HALF, WHEEL_R, SAFETY, 0.0)
        left_x, left_y = -math.sin(yaw), math.cos(yaw)          # the base's left, from its heading
        _payload, left, right = _hands(prim, toward)
        across = (left[0] - right[0]) * left_x + (left[1] - right[1]) * left_y
        assert across == pytest.approx(SPAN), (prim, toward, across)


def test_the_hand_axis_is_the_base_axis():
    """One heading for both, or the hands are square to a direction the base is not driving along.

    push_pose_hands builds its yaw from the same `toward - prim` difference push_prim_base_pose
    does, for the same -0.0 reason; this asserts they agree rather than trusting that they do.
    """
    for prim, toward in (((0.0, 1.5), (0.0, 0.0)), ((1.0, 1.0), (0.0, 0.0)),
                         ((-1.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 1.5))):
        _x, _y, yaw = push_prim_base_pose(prim, toward, HALF, WHEEL_R, SAFETY, 0.0)
        _payload, left, _right = _hands(prim, toward)
        forward = _col(left[3:7], 2)                            # the ee frame's +Z, the approach
        assert forward[0] == pytest.approx(math.cos(yaw), abs=1e-9)
        assert forward[1] == pytest.approx(math.sin(yaw), abs=1e-9)
        assert forward[2] == pytest.approx(0.0, abs=1e-9)       # horizontal, never tipped


def test_the_quaternion_reproduces_the_reference_at_the_reference_heading():
    """The anchor. 1600 recorded episodes used PUSH_HANDS_Q_REF at heading 1.5708, so the
    construction has to give that quaternion back there -- otherwise it is a new pose wearing the
    reference's provenance."""
    q = push_pose_quat(PUSH_HANDS_Q_REF_HEADING)
    assert q == pytest.approx(PUSH_HANDS_Q_REF, abs=1e-12)
    # and the file's own five-figure rounding, which is what a reader will compare against
    assert q == pytest.approx((0.70711, -0.70711, 0.0, 0.0), abs=1e-5)


def test_the_quaternion_turns_with_the_heading_and_keeps_the_hand_square_to_the_push():
    """WHAT IS PRESERVED across headings: ee +Z along the push, ee +Y down, ee +X to the robot's
    right. Kitchen 1218's two chairs measured -0.826 and -1.571, so this is exercised at real
    values and not only at the reference's pi/2."""
    for yaw in (PUSH_HANDS_Q_REF_HEADING, -0.826, -1.571, 0.0, math.pi, -math.pi / 4, 2.9):
        q = push_pose_quat(yaw)
        assert math.hypot(math.hypot(q[0], q[1]), math.hypot(q[2], q[3])) == pytest.approx(1.0)
        right, down, forward = _col(q, 0), _col(q, 1), _col(q, 2)
        assert forward == pytest.approx((math.cos(yaw), math.sin(yaw), 0.0), abs=1e-9)
        assert down == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
        # right = forward turned -90 degrees about world z
        assert right == pytest.approx((math.sin(yaw), -math.cos(yaw), 0.0), abs=1e-9)


def test_the_quaternion_is_pre_multiplied_and_not_post():
    """The two compose to different rotations everywhere except the reference heading, and the
    wrong one is not obviously wrong: it is still unit, still smooth in the heading, and still
    reproduces the reference at pi/2. Post-multiplying spins the hand about its OWN approach axis,
    so the palm keeps facing +y whatever the chair's heading.
    """
    from push_prim_geometry import _quat_mul

    yaw = -0.826
    half = 0.5 * (yaw - PUSH_HANDS_Q_REF_HEADING)
    rz = (math.cos(half), 0.0, 0.0, math.sin(half))
    assert push_pose_quat(yaw) == pytest.approx(_quat_mul(rz, PUSH_HANDS_Q_REF), abs=1e-12)

    wrong = _quat_mul(PUSH_HANDS_Q_REF, rz)
    assert _col(wrong, 2) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9), (
        "post-multiplying should leave the approach on +y -- if it does not, this test no longer "
        "demonstrates what it claims"
    )
    assert _col(push_pose_quat(yaw), 2) != pytest.approx(_col(wrong, 2), abs=1e-6)


def test_a_clearance_that_would_author_the_hands_inside_the_prim_is_refused():
    """cuRobo refuses a goal in collision, an hour into a GPU job and by prim path. Refusing here
    names the PARAMETER instead."""
    for bad in (0.0, -0.05):
        with pytest.raises(ValueError, match="clearance_m"):
            push_pose_hands((0.0, 1.5), (0.0, 0.0), HALF,
                            clearance_m=bad, height_m=HEIGHT, span_m=SPAN)


def test_a_negative_span_is_refused_rather_than_crossing_the_arms():
    with pytest.raises(ValueError, match="span_m"):
        push_pose_hands((0.0, 1.5), (0.0, 0.0), HALF,
                        clearance_m=CLEAR, height_m=HEIGHT, span_m=-0.16)


def test_a_prim_on_top_of_the_target_is_refused_here_too():
    with pytest.raises(ValueError):
        push_pose_hands((0.0, 0.0), (0.0, 0.0), HALF,
                        clearance_m=CLEAR, height_m=HEIGHT, span_m=SPAN)


def test_the_hands_sit_between_the_prim_and_the_base_that_will_drive_them_into_it():
    """The invariant that makes ONE clearance serve every chair in the library: the base parks at
    half + wheel_r + safety and the hands at half + clearance, so the gap between them is
    wheel_r + safety - clearance and THE HALF-EXTENT CANCELS.

    It is also the reach budget -- 0.25 m in front of the base at the template's values -- and the
    sign is what says the hands are in FRONT of the base rather than behind it.
    """
    for half in (0.19, 0.271, 0.35):
        base = push_prim_base_pose((0.0, 1.5), (0.0, 0.0), half, WHEEL_R, SAFETY, 0.0)
        _payload, left, right = _hands((0.0, 1.5), (0.0, 0.0), half=half)
        mid_y = (left[1] + right[1]) * 0.5
        assert base[1] - mid_y == pytest.approx(WHEEL_R + SAFETY - CLEAR)
        assert base[1] > mid_y > 1.5, (half, base[1], mid_y)


# =================================================================================================
# The FRAMELESS extent. half_extent_along has to be told which way the prim is turned, and on
# kitchen 1218 nobody could tell it correctly -- the USD export bakes the rotation into the
# vertices and leaves the prim transform identity, so the emit log read `prim yaw 0.0deg` off a
# chair yawed -47.31 degrees. These are the tests for the form that does not need to be told.
# =================================================================================================

from push_prim_geometry import axis_offset_deg, support_along_points  # noqa: E402


def _rot(points, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


#: The chair as a plan-view square, at the manifest's measured 0.3832 m.
SQUARE = [(-CHAIR_HALF, -CHAIR_HALF), (CHAIR_HALF, -CHAIR_HALF),
          (CHAIR_HALF, CHAIR_HALF), (-CHAIR_HALF, CHAIR_HALF)]


def test_the_support_is_the_reach_past_the_centre_along_u():
    assert support_along_points(SQUARE, (0.0, 0.0), 1.0, 0.0) == pytest.approx(CHAIR_HALF)
    assert support_along_points(SQUARE, (0.0, 0.0), 0.0, -1.0) == pytest.approx(CHAIR_HALF)
    d = math.sqrt(0.5)
    assert support_along_points(SQUARE, (0.0, 0.0), d, d) == pytest.approx(CHAIR_HALF * 2 * d)


def test_ROTATING_THE_POINTS_CANNOT_CHANGE_THE_ANSWER_AND_NO_YAW_IS_PASSED():
    """THE PROPERTY THE WHOLE CHANGE IS FOR. Rotate the chair and rotate the push with it -- it is
    the same physical push -- and the number must not move. Nothing here is told the yaw, so there
    is nothing for a baked-in rotation to be missing from.

    This is what half_extent_along could not give: it needs a yaw, the yaw was 0 on a prim rotated
    -47.31 degrees, and the answer came back 0.3827 instead of 0.1916.
    """
    for yaw_deg in (0.0, -47.31, -90.0, -131.17, 17.0, 45.0, 180.0, 359.0):
        yaw = math.radians(yaw_deg)
        got = support_along_points(_rot(SQUARE, yaw), (0.0, 0.0), *_push_axis(yaw))
        assert got == pytest.approx(CHAIR_HALF, abs=1e-9), yaw_deg


def test_the_points_are_TIGHTER_than_any_box_around_them():
    """A box is a bound; the points are the thing itself. Measured on this task's real chair (112
    vertices): along its own axes the mesh gives 0.1916, the same as the box, but along a diagonal
    it gives 0.2344 where the box's own support gives 0.2710 and the world AABB's gives 0.3827.

    An octagon stands in for that here, because a filled square touches its box at the corner and
    would show no difference -- which is exactly the case that would make this test vacuous.
    """
    # Vertices at 22.5, 67.5, ... so NONE sits on +x: an octagon with a vertex on the push axis
    # touches its own box there and would show no difference, which is the vacuous case.
    oct_pts = [(math.cos(math.radians(a)) * CHAIR_HALF, math.sin(math.radians(a)) * CHAIR_HALF)
               for a in (22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5)]
    mesh = support_along_points(oct_pts, (0.0, 0.0), 1.0, 0.0)
    box = half_extent_along((-CHAIR_HALF, -CHAIR_HALF), (CHAIR_HALF, CHAIR_HALF), 0.0, 1.0, 0.0)
    assert box == pytest.approx(CHAIR_HALF)
    assert mesh < box, (mesh, box)
    assert mesh == pytest.approx(CHAIR_HALF * math.cos(math.radians(22.5)))


def test_the_support_is_SIGNED_so_an_overhang_at_the_back_is_what_the_base_clears():
    """push_prim_base_pose stands the base at centre + u * (support + ...), so what it needs is the
    reach on the +u SIDE -- not the larger of the two sides. A chair whose backrest overhangs its
    legs is exactly where those differ, and the base has to clear the back."""
    lopsided = [(-0.10, -0.19), (0.30, -0.19), (0.30, 0.19), (-0.10, 0.19)]
    assert support_along_points(lopsided, (0.0, 0.0), 1.0, 0.0) == pytest.approx(0.30)
    assert support_along_points(lopsided, (0.0, 0.0), -1.0, 0.0) == pytest.approx(0.10)


def test_the_centre_is_the_one_the_standoff_is_measured_from():
    """A support taken about a different centre than push_prim_base_pose uses would stand the base
    off by the difference, silently."""
    assert support_along_points(SQUARE, (0.05, 0.0), 1.0, 0.0) == pytest.approx(CHAIR_HALF - 0.05)


def test_no_points_is_refused_rather_than_returning_zero():
    """0.0 would stand the base at the prim's own centre -- inside it."""
    with pytest.raises(ValueError, match="no points"):
        support_along_points([], (0.0, 0.0), 1.0, 0.0)


@pytest.mark.parametrize("deg,expected", [
    (0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (-90.0, 0.0), (270.0, 0.0),
    (45.0, 45.0), (-45.0, 45.0), (135.0, 45.0),
    (-47.31, 42.69), (0.5, 0.5), (89.5, 0.5),
])
def test_the_axis_offset_is_what_the_extent_fallback_is_refused_on(deg, expected):
    """A box measured at yaw 0 IS the world AABB. For an axis-aligned push that is tight and the
    fallback is right; off-axis it is inflated by up to double, so the planner refuses it there.

    kitchen 1218's two headings are -0.826 rad (-47.31 deg, 42.7 off axis -> REFUSED without a
    mesh) and -1.571 rad (-90.01 deg, 0.01 off axis -> allowed). The second is not exactly 90: the
    seat is placed at -pi/2 and the goal file rounds it, which is why the threshold is a degree
    rather than an exact test.
    """
    yaw = math.radians(deg)
    assert axis_offset_deg(math.cos(yaw), math.sin(yaw)) == pytest.approx(expected, abs=1e-6)


def test_the_two_kitchen_1218_headings_land_on_opposite_sides_of_the_refusal():
    """The one that produced emit job 2109085 must be the one that is refused."""
    for heading, refused in ((-0.826, True), (-1.571, False)):
        off = axis_offset_deg(math.cos(heading), math.sin(heading))
        assert (off > 1.0) is refused, (heading, off)
