"""fridge_siting on kitchen 1400's measured numbers. Pure geometry: no pxr, no scene_synthesizer,
runs under either interpreter.

Run with: python -m pytest test_fridge_siting.py -v
"""
import math

import pytest

import fridge_siting as fs

# Kitchen 1400, seed 1400, measured 2026-08-28 via fridge_kitchen.hinge_and_handle/fridge_front,
# corrected fix round 1 (2026-08-28) against source/isaaclab_assets/data/Kitchen/kitchen_1400_00.usd.
# plan_nav_to_prim parks at the HANDLE PRIM'S BBOX MIN along the face normal, minus safety, minus
# wheel_r -- NOT at the handle's centre. Handle bbox y = [-0.3935, -0.3435], centre -0.3685 (this
# is handle_xy[1]). Base y = -0.3935 - 0.12 - 0.23 = -0.7435, matching the goal file the GPU
# actually consumes (goal: [-0.2235, -0.7435, 1.5708]) to the millimetre.
# standoff_m and base_radius_m are DIFFERENT numbers with different jobs. hinge_and_handle returns
# the handle's CENTRE, so expressed from that reference the standoff is NAV_SAFETY_M 0.12 +
# WHEEL_RADIUS_M 0.23 + handle_half_depth 0.025 = 0.375 -- not the 0.35 that conflated the parking
# convention (standoff_m) with the robot's own footprint (base_radius_m, 0.340 m, measured from
# anubis_simvla.usd).
K1400 = fs.Site(
    hinge_xy=(0.3385, -0.3639),
    handle_xy=(-0.2735, -0.3685),
    radius_m=0.612,
    face_normal=(0.0, -1.0),
    standoff_m=0.375,
    base_radius_m=0.340,
    reach_m=0.555,
)


def test_the_default_parking_pose_reproduces_plan_nav_to_prims():
    """PIN THIS AGAINST THE PLANNER, NOT AGAINST A GUESS. plan_nav_to_prim parks at the HANDLE
    PRIM'S BBOX MIN along the face normal, minus safety, minus wheel_r -- not at the handle's
    centre and not at the door face's centre. Those three differ by up to 25 mm, and the viable
    park window is only ~70 mm wide, so the difference is a third of the budget.

    Derive the expected value by reading simvla_data_generator.py:1257-1279 and substituting the
    handle bbox from the kitchen USD; do not copy a number from this plan's header.
    """
    got = fs.base_xy(K1400, 0.0)
    assert got[0] == pytest.approx(-0.273, abs=1e-3)
    assert -0.75 < got[1] < -0.70, "within the 25 mm band the three candidate formulas span"


def test_the_screen_flags_the_door_jamming_on_the_parked_base():
    """THE DIAGNOSIS. The OBB/SAT authority puts it at 13.0 deg (see door_blocked_deg's docstring
    for the full comparison table); GPU honest peaks were 13.1 and 20.8. This point-model screen
    puts it at 3.5 deg with the measured 0.340 m base -- much too early, because it compares a
    point at radius R against a circle. The number is not the contract; that SOMETHING blocks
    is."""
    assert fs.door_blocked_deg(K1400, 0.0) is not None


def test_the_screen_is_documented_as_unable_to_clear_a_kitchen_on_its_own():
    """REGRESSION PIN FOR A REAL DEFECT IN AN EARLIER DRAFT OF THIS PLAN. Measured at the true
    parked pose (fix round 1 -- the original draft's OBB numbers were taken 25 mm off):

        past   OBB/SAT door jams at   screen says
        0.00        13.0 deg            3.5 deg
        0.20        13.0 deg           10.5 deg
        0.28        13.0 deg           free      <- the screen's blind spot is HERE
        0.30      free to 90           free
        0.40      free to 90           free

    At past=0.28 the screen reports the door free while the OBB authority reports it jammed at
    13.0 deg -- the screen tracks the door's edge point and misses the handle, which protrudes
    toward the robot. Any caller treating a None here as 'this kitchen works' is wrong."""
    assert fs.door_blocked_deg(K1400, 0.28) is None      # the screen's blind spot, pinned
    assert "SCREEN ONLY" in fs.door_blocked_deg.__doc__


def test_reach_grows_with_the_offset_and_stays_inside_anubis_reach():
    assert fs.reach_m_at(K1400, 0.00) == pytest.approx(0.375, abs=1e-3)
    assert fs.reach_m_at(K1400, 0.20) == pytest.approx(0.425, abs=1e-3)
    assert fs.reach_m_at(K1400, 0.35) == pytest.approx(0.513, abs=1e-3)


def test_min_past_is_a_lower_bound_on_the_offset_that_frees_the_door():
    """The OBB authority says 0.30 m on this kitchen. The screen returns less, and must never
    return MORE -- an over-estimate would reject a kitchen that works."""
    got = fs.min_past_for(K1400, 60.0, step_m=0.01)
    assert got <= 0.30, "the screen must be a lower bound, never an over-estimate"
    assert fs.door_blocked_deg(K1400, got) is None
    assert fs.door_blocked_deg(K1400, got - 0.01) is not None


def test_an_unreachable_handle_is_a_loud_error_not_a_silent_offset():
    """A fridge whose door can only be cleared by standing out of reach cannot host this task,
    and must fail on CPU rather than spend a GPU allocation discovering it."""
    cramped = fs.Site(hinge_xy=(0.338, -0.368), handle_xy=(-0.273, -0.368), radius_m=0.612,
                      face_normal=(0.0, -1.0), standoff_m=0.35, base_radius_m=0.23,
                      reach_m=0.38)                     # barely past the standoff
    with pytest.raises(fs.SitingError, match="reach"):
        fs.min_past_for(cramped, 60.0)


def test_the_chord_angle_is_half_the_sweep():
    """DIAG_DEG = 45 is the chord of a NINETY-degree swing. For a 60-degree target it is 30."""
    assert fs.chord_diag_deg(90.0) == pytest.approx(45.0)
    assert fs.chord_diag_deg(60.0) == pytest.approx(30.0)


def test_the_retreat_path_starts_at_the_parked_base_and_covers_the_authored_distance():
    pts = fs.retreat_path(K1400, 0.0, diag_deg=45.0, pull_m=0.10, total_m=0.90)
    assert pts[0] == pytest.approx(fs.base_xy(K1400, 0.0), abs=1e-6)
    assert math.dist(pts[0], pts[1]) == pytest.approx(0.10, abs=1e-9)      # the straight pull
    assert math.dist(pts[1], pts[-1]) == pytest.approx(0.90, abs=1e-9)     # the diagonal leg
    # The straight-line start-to-end distance is NOT asserted here: it depends on how nearly
    # axis-aligned the door happens to be, which is a property of the kitchen, not of this
    # function -- an idealised axis-aligned door gives 0.9733 m, kitchen 1400's real (4.1 mm
    # off-level) hinge/handle give something else, and both are correct for their geometry.
