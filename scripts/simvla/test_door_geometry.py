"""door_geometry against the real kitchen 1300 refrigerator, and a synthetic mirrored door.

Run with the usd-core interpreter, NOT env_isaaclab:

    python -m pytest test_door_geometry.py -v

env_isaaclab has no importable `pxr` until Isaac Sim is booted (test_kitchen_usd_load.py pays 53
seconds for exactly that). door_geometry.py imports pxr + stdlib only, precisely so this file can
run in about a second under a plain usd-core install.

The expected numbers below are MEASUREMENTS, read off kitchen_1300_00.usd on 2026-08-25 and
recorded in the design spec's section 3. They are not tolerances chosen to make the test pass -- if
this file goes red, either the asset changed or the module is wrong, and both deserve a look.
"""

import math
import os

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import door_geometry as dg

KITCHEN_1300 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/isaaclab_assets/data/Kitchen/kitchen_1300_00.usd",
)

#: Scoped to the tests that need the real asset. NOT a module-level pytestmark: the synthetic-door
#: tests below build their own stage and must still run on a machine without the kitchen corpus.
needs_1300 = pytest.mark.skipif(
    not os.path.exists(KITCHEN_1300), reason=f"{KITCHEN_1300} not present"
)


@pytest.fixture(scope="module")
def fridge():
    if not os.path.exists(KITCHEN_1300):
        pytest.skip(f"{KITCHEN_1300} not present")
    stage = Usd.Stage.Open(KITCHEN_1300)
    return dg.measure_door(stage, "/world/refrigerator/door")


@needs_1300
def test_the_hinge_is_the_joint_anchor_not_the_prim_pivot(fridge):
    """The whole point of the module. The door Xform's pivot happens to coincide with the hinge
    here, but the AUTHORITY is the revolute joint's body0 anchor, and the two are read
    independently: localPos0 (0.38914537, -0.30728954, 0) in the corpus frame, carried to world,
    must land on the same point. They agree to 0.3 mm on this asset, which is what makes the read
    trustworthy rather than merely plausible."""
    assert fridge.joint_name == "door_joint"
    assert fridge.joint_path == "/world/refrigerator/door_joint"
    assert fridge.hinge_world[0] == pytest.approx(1.0884, abs=2e-3)
    assert fridge.hinge_world[1] == pytest.approx(-3.0351, abs=2e-3)


@needs_1300
def test_the_handle_is_its_bbox_centre_not_its_pivot(fridge):
    """plan_arm_fridge_handle_grasp read ExtractTranslation() here and got (1.0884, -3.0351, 0.0)
    -- the door Xform's pivot, i.e. the hinge at floor level, 0.71 m from the handle and 1.03 m
    below it."""
    assert fridge.handle_path == "/world/refrigerator/door/door_handle"
    assert fridge.handle_world[0] == pytest.approx(1.0128, abs=2e-3)
    assert fridge.handle_world[1] == pytest.approx(-2.3218, abs=2e-3)
    assert fridge.handle_world[2] == pytest.approx(1.0268, abs=2e-3)


@needs_1300
def test_the_radius_is_the_hardcoded_zero_point_seven_measured(fridge):
    """0.717, not 0.7. Three functions in this repo write the literal 0.7 for this quantity."""
    assert fridge.radius_m == pytest.approx(0.717, abs=5e-3)


@needs_1300
def test_the_joint_limits_come_from_the_asset(fridge):
    assert fridge.limit_lower_deg == pytest.approx(0.0, abs=1e-6)
    assert fridge.limit_upper_deg == pytest.approx(180.0, abs=1e-6)


@needs_1300
def test_hinge_side_and_arm_for_a_robot_facing_the_door(fridge):
    """Approaching along +x (yaw 0), the robot's right is -y. The hinge is at y=-3.035 and the
    handle at y=-2.322, so the hinge is to the RIGHT and the handle to the LEFT -- which is why the
    LEFT arm grasps it. hero_template.py chose A_l independently; this derives it."""
    assert dg.hinge_side(fridge, approach_yaw_rad=0.0) == "Right"
    assert dg.arm_for_hinge("Right") == "Left"


@needs_1300
def test_hinge_side_flips_with_the_approach(fridge):
    """Same door, robot turned around: the hinge is now on its left. Guards against a convention
    that happens to be right for one approach and silently wrong for the other."""
    assert dg.hinge_side(fridge, approach_yaw_rad=math.pi) == "Left"
    assert dg.arm_for_hinge("Left") == "Right"


def _synthetic_door(tmp_path, handle_y, name="synthetic.usda"):
    """A minimal fridge: corpus, door, handle, one revolute joint. Hinge at the world origin.

    `handle_y` places the handle relative to that hinge, which is the only thing that decides
    which side the hinge falls on for a robot facing +x.
    """
    path = str(tmp_path / name)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.Xform.Define(stage, "/world")
    UsdGeom.Xform.Define(stage, "/world/fridge")
    UsdGeom.Xform.Define(stage, "/world/fridge/corpus")
    UsdGeom.Xform.Define(stage, "/world/fridge/door")
    handle = UsdGeom.Cube.Define(stage, "/world/fridge/door/door_handle")
    handle.GetSizeAttr().Set(0.1)
    UsdGeom.Xformable(handle).AddTranslateOp().Set(Gf.Vec3d(0.0, handle_y, 1.0))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/world/fridge/door_joint")
    joint.GetAxisAttr().Set("Z")
    joint.GetLowerLimitAttr().Set(0.0)
    joint.GetUpperLimitAttr().Set(150.0)
    joint.GetBody0Rel().SetTargets(["/world/fridge/corpus"])
    joint.GetBody1Rel().SetTargets(["/world/fridge/door"])
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    return stage


def test_a_synthetic_door_measures_its_own_radius_and_limits(tmp_path):
    stage = _synthetic_door(tmp_path, handle_y=0.6)
    geom = dg.measure_door(stage, "/world/fridge/door")
    assert geom.radius_m == pytest.approx(0.6, abs=1e-3)
    assert geom.limit_upper_deg == pytest.approx(150.0, abs=1e-6)
    assert geom.joint_name == "door_joint"


def test_a_mirrored_door_reports_the_other_hinge_side(tmp_path):
    """The Left branch, on a door built to exercise it -- the real kitchen only has a Right one.

    Handle at y = -0.6 with the hinge at the origin puts the handle on the robot's RIGHT (facing
    +x, +y is its left), so the hinge is on its LEFT and the RIGHT arm reaches the handle. This is
    the mirror of kitchen 1300, whose handle sits at higher y than its hinge.
    """
    stage = _synthetic_door(tmp_path, handle_y=-0.6, name="mirrored.usda")
    geom = dg.measure_door(stage, "/world/fridge/door")
    assert dg.hinge_side(geom, approach_yaw_rad=0.0) == "Left"
    assert dg.arm_for_hinge("Left") == "Right"


def test_the_unmirrored_synthetic_door_matches_kitchen_1300s_handedness(tmp_path):
    """Handle at higher y than the hinge -> hinge on the right -> left arm. Same handedness the
    real fridge measures, reproduced on geometry small enough to reason about by hand."""
    stage = _synthetic_door(tmp_path, handle_y=0.6, name="righthand.usda")
    geom = dg.measure_door(stage, "/world/fridge/door")
    assert dg.hinge_side(geom, approach_yaw_rad=0.0) == "Right"
    assert dg.arm_for_hinge("Right") == "Left"


def test_a_door_with_no_joint_is_refused(tmp_path):
    """Silence here would produce a radius of 0 and an arc that spins the base in place."""
    stage = _synthetic_door(tmp_path, handle_y=0.6, name="nojoint.usda")
    stage.RemovePrim("/world/fridge/door_joint")
    with pytest.raises(dg.DoorGeometryError, match="no revolute joint"):
        dg.measure_door(stage, "/world/fridge/door")


def test_a_missing_handle_is_refused(tmp_path):
    stage = _synthetic_door(tmp_path, handle_y=0.6, name="nohandle.usda")
    stage.RemovePrim("/world/fridge/door/door_handle")
    with pytest.raises(dg.DoorGeometryError, match="no handle"):
        dg.measure_door(stage, "/world/fridge/door")


def test_an_unknown_hinge_side_is_refused(tmp_path):
    with pytest.raises(dg.DoorGeometryError, match="Right"):
        dg.arm_for_hinge("Middle")


def test_the_handle_is_always_opposite_the_hinge(fridge):
    """The handle sits at the door's FREE EDGE, which is by construction opposite the hinge.

    Kitchen 1300's fridge measures hinge on the Right at yaw 0, so the handle is on the Left.
    """
    assert dg.hinge_side(fridge, 0.0) == "Right"
    assert dg.handle_side(fridge, 0.0) == "Left"


def test_handle_side_mirrors_hinge_side_at_every_approach(fridge):
    """Not just at yaw 0. The two are mirrors at every heading, or one of them is a coincidence."""
    for yaw_deg in range(0, 360, 15):
        yaw = math.radians(yaw_deg)
        hinge, handle = dg.hinge_side(fridge, yaw), dg.handle_side(fridge, yaw)
        assert {hinge, handle} == {"Left", "Right"}, f"yaw {yaw_deg}: {hinge}/{handle}"


def test_the_arm_is_the_handles_own_side():
    """The rule, in one line: the handle on the left is reached by the LEFT arm.

    arm_for_handle is deliberately the identity. Writing it as a function anyway is what lets
    arm_for_hinge be defined THROUGH it, so the mirror is stated exactly once.
    """
    assert dg.arm_for_handle("Left") == "Left"
    assert dg.arm_for_handle("Right") == "Right"


def test_arm_for_hinge_still_answers_exactly_as_before():
    """The regression guard. This function has callers; its results must not move."""
    assert dg.arm_for_hinge("Right") == "Left"
    assert dg.arm_for_hinge("Left") == "Right"


def test_both_arm_helpers_agree_through_the_mirror(fridge):
    for yaw_deg in range(0, 360, 15):
        yaw = math.radians(yaw_deg)
        assert (dg.arm_for_handle(dg.handle_side(fridge, yaw))
                == dg.arm_for_hinge(dg.hinge_side(fridge, yaw)))


def test_a_bad_handle_side_is_a_loud_error():
    with pytest.raises(dg.DoorGeometryError):
        dg.arm_for_handle("Middle")


# ==================================================================================================
# The hinge AXIS, which is not the axis token
# ==================================================================================================
#
# `physics:axis` is a token in the JOINT'S OWN FRAME, and that frame is oriented by localRot0. Every
# door in this corpus authors the token "Z", so reading it alone says "vertical hinge" for all of
# them -- including the dishwasher and the oven, whose localRot0 lays the axis flat. Measured across
# a freshly built kitchen: refrigerator, sink_cabinet, base_cabinet, wall_cabinet and microwave come
# out vertical; dishwasher/door_0_1 and range/door_0_1 come out HORIZONTAL, i.e. drop-down doors
# with the hinge along the floor. Their radius read as an xy distance is meaningless (0.343 m for a
# dishwasher whose handle is really 0.682 m from its hinge line), and an arc skill pointed at one
# would drive the base sideways for a door that opens downward.
#
# The SIGN matters too, and independently. Half the vertical doors author localRot0 = (0,0,1,0) -- a
# 180-degree flip -- so their world axis is (0,0,-1) and a positive joint angle swings the door
# CLOCKWISE. A sweep that always rotates counter-clockwise models those doors opening into the wall
# they are mounted on.


def _reoriented_door(tmp_path, local_rot, name):
    """_synthetic_door with a joint FRAME rotation applied, the axis token left at "Z"."""
    stage = _synthetic_door(tmp_path, handle_y=0.6, name=name)
    joint = stage.GetPrimAtPath("/world/fridge/door_joint")
    joint.GetAttribute("physics:localRot0").Set(local_rot)
    joint.GetAttribute("physics:localRot1").Set(local_rot)
    return stage


def test_a_vertical_hinge_reports_its_world_axis_and_opening_sense(tmp_path):
    """The plain case: no frame rotation, so the token IS the world axis and +angle opens CCW."""
    stage = _synthetic_door(tmp_path, handle_y=0.6, name="upright.usda")
    geom = dg.measure_door(stage, "/world/fridge/door")
    assert geom.axis_world == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
    assert geom.opens_ccw is True


def test_a_flipped_frame_reports_a_clockwise_opening(tmp_path):
    """localRot0 = (0,0,1,0) is a 180-degree turn about Y: the axis token "Z" lands on world -Z.
    The door is still vertical, but a POSITIVE joint angle now swings it the other way -- which is
    the difference between a door opening into the room and one opening into the cabinet beside it.
    sink_cabinet/door_0_1 is authored exactly like this in every kitchen measured."""
    stage = _reoriented_door(tmp_path, Gf.Quatf(0.0, 0.0, 1.0, 0.0), "flipped.usda")
    geom = dg.measure_door(stage, "/world/fridge/door")
    assert geom.axis_world == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert geom.opens_ccw is False
    # The radius is a perpendicular distance to the hinge LINE, so flipping the line's direction
    # cannot change it.
    assert geom.radius_m == pytest.approx(0.6, abs=1e-3)


def test_a_horizontal_hinge_is_refused_as_a_drop_down_door(tmp_path):
    """localRot0 = (0.7071, 0, 0.7071, 0) is 90 degrees about Y, laying the axis flat. This is how
    dishwasher/door_0_1 and range/door_0_1 are authored. Refused rather than measured: every
    consumer of DoorGeometry (nav.open_door_arc, the arc clearance gate, hinge_side/arm_for_hinge)
    assumes a vertical hinge, and each would return a plausible number for a motion that does not
    exist."""
    stage = _reoriented_door(tmp_path, Gf.Quatf(0.70710677, 0.0, 0.70710677, 0.0), "flat.usda")
    with pytest.raises(dg.DoorGeometryError, match="horizontal"):
        dg.measure_door(stage, "/world/fridge/door")


def test_a_drop_down_door_can_still_be_measured_on_request(tmp_path):
    """require_vertical=False is for a caller that wants to REPORT the door's shape (the kitchen
    survey does), not arc about it. The radius is then the perpendicular distance to the hinge
    LINE, which for a flat axis is a vertical drop, not an xy distance."""
    stage = _reoriented_door(tmp_path, Gf.Quatf(0.70710677, 0.0, 0.70710677, 0.0), "flat2.usda")
    geom = dg.measure_door(stage, "/world/fridge/door", require_vertical=False)
    assert abs(geom.axis_world[2]) < 1e-6
    # Handle at (0, 0.6, 1.0), hinge line along world x through the origin: the perpendicular
    # distance is hypot(0.6, 1.0).
    assert geom.radius_m == pytest.approx(math.hypot(0.6, 1.0), abs=1e-3)


@needs_1300
def test_the_fridge_radius_is_unchanged_by_the_axis_work(fridge):
    """A regression guard, not a new fact. kitchen 1300's fridge has an identity joint frame, so
    the perpendicular-to-the-line radius must equal the xy radius this module always reported."""
    assert fridge.radius_m == pytest.approx(0.7173, abs=2e-3)
    assert fridge.axis_world[2] == pytest.approx(1.0, abs=1e-6) or \
           fridge.axis_world[2] == pytest.approx(-1.0, abs=1e-6)
