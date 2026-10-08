"""Tests for nav_clearance: the author-time base-clearance gate behind nav.to_prim.

Run with: pytest scripts/simvla/test_nav_clearance.py -v
Stdlib only -- no sim, no torch, no Omniverse. The kitchen geometry below is the world-space AABB
of every /world child, dumped from kitchen_1570_00.usd and kitchen_1550_00.usd with
UsdGeom.BBoxCache (the same read plan_nav_to_prim makes), so what these tests pin is what the
planner sees.

The behaviour being pinned: a park pose that is CLEAR is returned untouched, and a pose that
PENETRATES furniture is re-sited to the nearest clear side -- which on kitchen 1570 is the
difference between three identical timed-out runs (the base authored inside the refrigerator) and
a run that reaches the mug.
"""

import math

import pytest

import nav_clearance as nc
from nav_clearance import Box

KITCHEN_1570 = [
    Box('range', -0.4222, -0.3934, 0.4222, 0.3534, 0.0, 0.99),
    Box('dishwasher', 0.4222, -0.3934, 1.0811, 0.3534, -0.0, 0.9111),
    Box('sink_cabinet', 1.0811, -0.3914, 1.7874, 0.3534, -0.0, 0.95),
    Box('corner', 1.7874, -0.4234, 2.5643, 0.3534, -0.0, 0.9111),
    Box('base_cabinet', 1.8194, -1.4234, 2.5643, -0.4234, -0.0, 0.9111),
    Box('refrigerator', 1.771, -2.2045, 2.5643, -1.4234, -0.0, 1.821),
    Box('range_hood', -0.3972, -0.1568, 0.3972, 0.3534, 1.2677, 2.1708),
    Box('wall_cabinet', 2.1729, -1.4234, 2.5643, -0.4234, 1.2677, 2.0308),
    Box('wall_cabinet_0', 0.4222, -0.038, 1.0811, 0.3534, 1.2677, 2.0308),
    Box('wall_cabinet_1', 1.0811, -0.038, 1.7874, 0.3534, 1.2677, 2.0308),
    Box('wall_cabinet_2', 1.7874, -0.038, 2.1759, 0.3534, 1.2677, 2.0308),
    Box('wall_cabinet_3', 2.1729, -0.4234, 2.5643, -0.035, 1.2677, 2.0308),
    Box('countertop_dishwasher', 0.4222, -0.3534, 1.0811, 0.3534, 0.9111, 0.95),
    Box('countertop_base_cabinet', 1.8574, -1.4234, 2.5643, -0.4234, 0.9111, 0.95),
    Box('countertop_corner', 1.7874, -0.4234, 2.5643, 0.3534, 0.9111, 0.95),
    Box('microwave', 2.0946, -1.2134, 2.5643, -0.6334, 0.95, 1.21),
    Box('table', 0.521, -3.7545, 1.621, -3.0545, 0.0, 0.74),
    Box('chair_0', 0.0047, -4.3708, 0.5454, -3.8301, 0.0, 0.8143),
    Box('chair_1', 1.5966, -4.3708, 2.1373, -3.8301, 0.0, 0.8143),
    Box('wall__x', -0.9588, -4.889, -0.8231, 0.5569, -0.1543, 2.3251),
    Box('wall_y', -0.9773, 0.4026, 2.7678, 0.5383, -0.1543, 2.3251),
    Box('wall__y', -0.9773, -4.8704, 2.7678, -4.7347, -0.1543, 2.3251),
    Box('wall_x', 2.6135, -5.0432, 2.7492, 0.7112, -0.3085, 2.4793),
    Box('mug0', 1.8675, -1.1207, 1.9473, -1.0141, 0.952, 1.1038),
    Box('plate0', 2.0597, -0.3516, 2.2009, -0.2113, 0.952, 0.9842),
]
KITCHEN_1550 = [
    Box('refrigerator', -0.3859, -0.3881, 0.3859, 0.3575, -0.0, 1.581),
    Box('base_cabinet', 0.3859, -0.3833, 1.3859, 0.3575, -0.0, 0.9119),
    Box('corner', 1.3859, -0.4153, 2.1587, 0.3575, -0.0, 0.9119),
    Box('dishwasher', 1.4159, -1.0685, 2.1587, -0.4153, -0.0, 0.9119),
    Box('sink_cabinet', 1.4179, -1.7392, 2.1587, -1.0685, -0.0, 0.95),
    Box('range', 1.4159, -2.4903, 2.1587, -1.7392, 0.0, 0.99),
    Box('range_hood', 1.5971, -2.4653, 2.1587, -1.7642, 1.3343, 2.3008),
    Box('wall_cabinet', 0.3859, -0.0319, 1.3859, 0.3575, 1.3343, 2.086),
    Box('wall_cabinet_0', 1.7693, -1.0685, 2.1587, -0.4153, 1.3343, 2.086),
    Box('wall_cabinet_1', 1.7693, -1.7392, 2.1587, -1.0685, 1.3343, 2.086),
    Box('wall_cabinet_2', 1.3859, -0.0319, 1.7723, 0.3575, 1.3343, 2.086),
    Box('wall_cabinet_3', 1.7693, -0.4153, 2.1587, -0.0289, 1.3343, 2.086),
    Box('countertop_dishwasher', 1.4559, -1.0685, 2.1587, -0.4153, 0.9119, 0.95),
    Box('countertop_base_cabinet', 0.3859, -0.3453, 1.3859, 0.3575, 0.9119, 0.95),
    Box('countertop_corner', 1.3859, -0.4153, 2.1587, 0.3575, 0.9119, 0.95),
    Box('microwave', 0.5959, -0.1121, 1.1759, 0.3575, 0.95, 1.21),
    Box('table', 0.3364, -4.0403, 1.4364, -3.3403, 0.0, 0.74),
    Box('chair_0', -0.1799, -4.6566, 0.3608, -4.1159, 0.0, 0.8143),
    Box('chair_1', 1.412, -4.6566, 1.9527, -4.1159, 0.0, 0.8143),
    Box('wall_x', 2.246, -5.5941, 2.3283, 0.7562, -0.3113, 2.6121),
    Box('wall__x', -1.1313, -5.5941, -1.049, 0.7562, -0.3113, 2.6121),
    Box('wall__y', -1.3603, -5.3651, 2.5573, -5.2828, -0.3113, 2.6121),
    Box('wall_y', -1.6716, 0.4449, 2.8686, 0.5272, -0.6225, 2.9233),
    Box('mug0', 0.8454, -0.371, 0.9251, -0.2644, 0.952, 1.1038),
    Box('plate0', 0.7554, -4.0718, 0.8966, -3.9315, 0.742, 0.7742),
]

#: Kitchen 1570's emitted nav goal, Isaac-Kitchen-v1570r-00.json step 0: park y is the base
#: cabinet's y_min (-1.4234) minus safety 0.12 minus wheel_r 0.23 minus the RB-Y1 object standoff
#: 0.12. The refrigerator spans y [-2.2045, -1.4234] at that x.
PARK_1570 = (1.8517831563949585, -1.893428921699524, math.pi / 2)
MUG_1570 = (1.901783141378007, -1.0628432758625448)
#: Kitchen 1550's, after the 0.12 object gap (main 3608cc99): measured clearance 0.087 m to the
#: base cabinet, whose body and handles protrude 0.038 m past the countertop the park is measured
#: from.
PARK_1550 = (0.835, -0.8153, math.pi / 2)
MUG_1550 = (0.885, -0.318)

RBY1 = nc.footprint_for("rby1")
#: 1570's re-sited park (N side, right arm): the base 0.05 m to its own LEFT of the mug, so the mug
#: sits 0.05 m to the robot's RIGHT -- the right arm's side. (Before the sign fix the N/E branches
#: parked the base on the WRONG side of the target, by the same 0.05.)
PARK_1570_N = (1.3874, MUG_1570[1] + 0.05, 0.0)


def box(name, x0, y0, x1, y1, z0=0.0, z1=0.9):
    return Box(name, x0, y0, x1, y1, z0, z1)


# ----------------------------------------------------------------------------------------------
# Footprints
# ----------------------------------------------------------------------------------------------

def test_footprints_are_the_measured_base_boxes():
    """Measured 2026-09-02 from each robot's own USD base_link (all four bbox purposes):
    RB-Y1 rby1m model.usd x[-0.350, +0.345] y[+-0.300]; Anubis anubis_simvla.usd 0.465 x 0.4916
    (cabinet_kitchen.BASE_X/Y); AI Worker ffw_sg2.usd x[-0.403, +0.225] y[+-0.301]."""
    r = nc.footprint_for("rby1")
    assert (r.x_min, r.x_max, r.y_min, r.y_max) == pytest.approx((-0.350, 0.345, -0.300, 0.300))
    a = nc.footprint_for("anubis")
    assert (a.x_max - a.x_min, a.y_max - a.y_min) == pytest.approx((0.465, 0.4916))
    w = nc.footprint_for("aiworker")
    assert (w.x_min, w.x_max, w.y_min, w.y_max) == pytest.approx((-0.403, 0.225, -0.301, 0.301))


def test_an_unknown_robot_raises_rather_than_borrowing_a_footprint():
    with pytest.raises(KeyError):
        nc.footprint_for("tesla_optimus")


# ----------------------------------------------------------------------------------------------
# Clearance of one pose
# ----------------------------------------------------------------------------------------------

def test_clearance_is_the_gap_between_the_base_front_and_the_nearest_box():
    wall = box("counter", 0.5, -1.0, 1.0, 1.0)
    gap, who = nc.clearance(0.0, 0.0, 0.0, RBY1, [wall])
    assert gap == pytest.approx(0.5 - 0.345)
    assert who == "counter"


def test_penetration_is_reported_as_a_negative_gap():
    wall = box("counter", 0.3, -1.0, 1.0, 1.0)
    gap, who = nc.clearance(0.0, 0.0, 0.0, RBY1, [wall])
    assert gap == pytest.approx(0.3 - 0.345)
    assert who == "counter"


def test_clearance_turns_with_the_base():
    """Facing +y the base's 0.300 half-WIDTH is what points at a box on +x, not its 0.345 front."""
    wall = box("counter", 0.5, -1.0, 1.0, 1.0)
    gap, _ = nc.clearance(0.0, 0.0, math.pi / 2, RBY1, [wall])
    assert gap == pytest.approx(0.5 - 0.300)


def test_clearance_with_no_obstacles_is_unbounded():
    gap, who = nc.clearance(0.0, 0.0, 0.0, RBY1, [])
    assert gap == math.inf and who is None


def test_a_diagonal_base_is_checked_as_a_rotated_box_not_a_circle():
    """At 45 deg the corner reaches sqrt(0.345^2 + 0.3^2) = 0.457 along the diagonal but only
    (0.345 + 0.300)/sqrt(2) = 0.456 along x; a circle model would say 0.457 on every axis."""
    wall = box("counter", 0.5, -1.0, 1.0, 1.0)
    gap, _ = nc.clearance(0.0, 0.0, math.pi / 4, RBY1, [wall])
    assert gap == pytest.approx(0.5 - (0.345 + 0.300) / math.sqrt(2.0), abs=1e-6)


# ----------------------------------------------------------------------------------------------
# Which boxes count
# ----------------------------------------------------------------------------------------------

def test_floor_obstacles_keep_what_the_base_column_can_hit_and_drop_the_rest():
    names = {b.name for b in nc.floor_obstacles(KITCHEN_1570)}
    assert {"refrigerator", "base_cabinet", "table", "chair_0", "wall_x", "wall__y"} <= names
    # Countertops (underside 0.911), the microwave (0.95) and wall cabinets (1.27) sit above the
    # base; in plan their boxes CONTAIN the parking spot, so counting them makes every park a hit.
    assert not {"countertop_base_cabinet", "microwave", "wall_cabinet", "range_hood"} & names


def test_floor_obstacles_can_exclude_the_target_by_name():
    names = {b.name for b in nc.floor_obstacles(KITCHEN_1570, exclude=("table",))}
    assert "table" not in names and "chair_0" in names


# ----------------------------------------------------------------------------------------------
# The two campaign kitchens
# ----------------------------------------------------------------------------------------------

def test_kitchen_1570_usual_park_is_inside_the_refrigerator():
    obstacles = nc.floor_obstacles(KITCHEN_1570)
    gap, who = nc.clearance(*PARK_1570, RBY1, obstacles)
    assert who == "refrigerator"
    assert gap < -0.25          # deep inside, not a graze


def test_kitchen_1550_usual_park_is_clear_by_the_measured_0_087_m():
    obstacles = nc.floor_obstacles(KITCHEN_1550)
    gap, who = nc.clearance(*PARK_1550, RBY1, obstacles)
    assert who == "base_cabinet"
    assert gap == pytest.approx(0.087, abs=0.002)


# ----------------------------------------------------------------------------------------------
# The room
# ----------------------------------------------------------------------------------------------

def test_room_interior_is_the_inner_faces_of_the_shell_walls():
    bounds, walls = nc.room_interior(KITCHEN_1570)
    assert bounds == pytest.approx((-0.8231, -4.7347, 2.6135, 0.4026))
    assert {w.name for w in walls} == {"wall_x", "wall__x", "wall_y", "wall__y"}


def test_wall_mounted_cabinets_are_furniture_not_walls():
    """kitchen_build.room_shell_prim_names: never a "wall_" prefix test. Read as walls,
    wall_cabinet_0 (x_min 1.08) and wall_cabinet (y_min -1.42) shrink 1570's room to the strip
    under them and every park west of the cabinet is "outside"."""
    assert not nc.is_shell_wall("wall_cabinet") and not nc.is_shell_wall("wall_cabinet_0")
    assert nc.is_shell_wall("wall__x") and nc.is_shell_wall("wall_-x") and nc.is_shell_wall("wall_01")


def test_a_kitchen_without_a_shell_gets_the_runtime_walls_the_env_config_adds():
    """make_walls_from_bounds: four 0.1 m walls centred 0.2 m outside the furniture union, so
    the inner faces sit 0.15 m out."""
    furniture = [box("a", 0.0, 0.0, 1.0, 1.0), box("b", 2.0, -1.0, 3.0, 0.5)]
    bounds, walls = nc.room_interior(furniture)
    assert bounds == pytest.approx((-0.15, -1.15, 3.15, 1.15))
    assert len(walls) == 4 and all(w.name.startswith("wall") for w in walls)
    assert nc.clearance(1.5, -0.5, 0.0, RBY1, walls)[0] > 0.0                 # inside: no hit
    assert nc.clearance(1.5, -1.0, 0.0, RBY1, walls)[0] < 0.0                 # flank in the -y wall


# ----------------------------------------------------------------------------------------------
# Re-siting: the island-style search
# ----------------------------------------------------------------------------------------------

def _furniture(kitchen, name):
    return next(b for b in kitchen if b.name == name)


def _scene(kitchen):
    bounds, walls = nc.room_interior(kitchen)
    return bounds, nc.floor_obstacles(kitchen) + [w for w in walls if w.name not in {b.name for b in kitchen}]


def test_pick_clear_park_resites_1570_to_the_open_side_of_the_cabinet():
    """The only clear side of 1570's corner cabinet is -x (N): E is the corner run, S is the wall,
    W is the refrigerator. Parked there the mug is 0.51 m ahead instead of 0.83."""
    bounds, obstacles = _scene(KITCHEN_1570)
    park = nc.pick_clear_park(MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                              standoff_m=0.47, arm_bias=-0.05, footprint=RBY1,
                              obstacles=obstacles, prefer_side="W", bounds=bounds)
    assert park is not None
    assert park.side == "N"
    assert park.yaw == pytest.approx(0.0)
    assert park.x == pytest.approx(1.8574 - 0.47)
    assert park.y == pytest.approx(MUG_1570[1] + 0.05)      # base LEFT of the mug: mug on the right
    assert park.slide_m == 0.0 and park.extra_m == 0.0
    assert park.clearance_m >= nc.MIN_CLEAR_M
    assert park.ahead_m == pytest.approx(MUG_1570[0] - park.x)
    assert park.start is None                                 # no starts given: not asked


def test_pick_clear_park_prefers_the_nearest_reach_over_the_original_side():
    """Two clear sides: the object is 0.10 m from the +x face and 0.60 m from the -x face."""
    furniture = box("table", 0.0, -0.5, 0.7, 0.5)
    park = nc.pick_clear_park((0.6, 0.0), furniture, standoff_m=0.47, arm_bias=0.0,
                              footprint=RBY1, obstacles=[], prefer_side="N")
    assert park.side == "S"
    assert park.ahead_m == pytest.approx(0.57)


def test_pick_clear_park_keeps_the_side_and_adds_standoff_when_that_is_all_it_takes():
    """A cabinet body that protrudes 0.15 m past the countertop the park is measured from: the
    usual pose's front sits 0.125 m off the countertop face, so it is 0.025 m INTO the body.
    One 0.05 m step back clears it by 0.025 -- the same shape as kitchen 1550's real defect."""
    top = box("countertop", 0.0, -1.0, 1.0, 0.0, 0.9, 0.95)
    body = box("cabinet", 0.0, -1.15, 1.0, 0.0, 0.0, 0.9)
    obstacles = nc.floor_obstacles([top, body])
    usual = nc.clearance(0.5, -1.0 - 0.47, math.pi / 2, RBY1, obstacles)[0]
    assert usual == pytest.approx(-0.025)
    park = nc.pick_clear_park((0.5, -0.9), top, standoff_m=0.47, arm_bias=0.0,
                              footprint=RBY1, obstacles=obstacles, prefer_side="W")
    assert park.side == "W"
    assert park.slide_m == 0.0
    assert park.extra_m == pytest.approx(0.05)
    assert park.clearance_m == pytest.approx(0.025)


def test_pick_clear_park_slides_along_the_face_to_get_out_of_a_corner():
    """Object 0.05 m from the room's +x wall: the straight-on park's flank is 0.25 m into the
    wall, the far side of the wall is outside the room, so the only way out is sideways."""
    top = box("countertop", 0.0, 0.0, 2.0, 0.6, 0.9, 0.95)
    wall = box("wall", 2.0, -3.0, 2.1, 3.0, 0.0, 2.5)
    obstacles = nc.floor_obstacles([top, wall])
    park = nc.pick_clear_park((1.95, 0.3), top, standoff_m=0.47, arm_bias=0.0,
                              footprint=RBY1, obstacles=obstacles, prefer_side="W",
                              bounds=(-3.0, -3.0, 2.0, 3.0))
    assert park is not None and park.side == "W"
    assert park.slide_m < 0.0
    assert park.x + RBY1.y_max <= 2.0 - nc.MIN_CLEAR_M + 1e-9
    assert park.extra_m == 0.0


def test_pick_clear_park_never_parks_outside_the_room():
    """A thin wall alone would let the search stand the base on the far side of it."""
    top = box("countertop", 0.0, 0.0, 2.0, 0.6, 0.9, 0.95)
    wall = box("wall", 2.0, -3.0, 2.1, 3.0, 0.0, 2.5)
    obstacles = nc.floor_obstacles([top, wall])
    outside = nc.pick_clear_park((1.95, 0.3), top, standoff_m=0.47, arm_bias=0.0,
                                 footprint=RBY1, obstacles=obstacles, prefer_side="W")
    assert outside.side == "S" and outside.x > 2.1                   # what happens without bounds
    inside = nc.pick_clear_park((1.95, 0.3), top, standoff_m=0.47, arm_bias=0.0,
                                footprint=RBY1, obstacles=obstacles, prefer_side="W",
                                bounds=(-3.0, -3.0, 2.0, 3.0))
    assert inside.side != "S"


def test_pick_clear_park_returns_none_when_boxed_in():
    top = box("countertop", 0.0, 0.0, 1.0, 1.0, 0.9, 0.95)
    ring = [box(f"wall{i}", *r, 0.0, 2.5) for i, r in enumerate(
        [(-1.5, -1.5, -0.2, 2.5), (1.2, -1.5, 2.5, 2.5), (-1.5, -1.5, 2.5, -0.2), (-1.5, 1.2, 2.5, 2.5)])]
    assert nc.pick_clear_park((0.5, 0.5), top, standoff_m=0.47, arm_bias=0.0,
                              footprint=RBY1, obstacles=ring, prefer_side="N") is None


def test_park_on_side_reproduces_plan_nav_to_prims_four_poses():
    """The letters, standoff sign and yaw MUST match plan_nav_to_prim's own branch, because
    plan_arm_grasp keys the grasp half-plane on app.N_dir."""
    f = box("f", 1.0, 2.0, 3.0, 4.0)
    assert nc.park_on_side("N", f, lateral=2.5, standoff_m=0.5) == pytest.approx((0.5, 2.5, 0.0))
    assert nc.park_on_side("S", f, lateral=2.5, standoff_m=0.5) == pytest.approx((3.5, 2.5, math.pi))
    assert nc.park_on_side("W", f, lateral=1.5, standoff_m=0.5) == pytest.approx((1.5, 1.5, math.pi / 2))
    assert nc.park_on_side("E", f, lateral=1.5, standoff_m=0.5) == pytest.approx((1.5, 4.5, -math.pi / 2))


# ----------------------------------------------------------------------------------------------
# The drive to the park
# ----------------------------------------------------------------------------------------------

def test_path_clear_sweeps_the_base_along_the_straight_drive():
    post = box("post", 0.9, -0.1, 1.1, 0.1)
    ok, who = nc.path_clear((0.0, 0.0), 0.0, (2.0, 0.0), 0.0, RBY1, [post])
    assert not ok and who == "post"
    ok, who = nc.path_clear((0.0, 1.0), 0.0, (2.0, 1.0), 0.0, RBY1, [post])
    assert ok and who is None


def test_path_clear_checks_the_turn_in_place_at_both_ends():
    """A box beside the start clears the flank at the spawn heading, but the base's rear corner
    sweeps into it while turning from the spawn heading to the bearing."""
    beside = box("beside", -0.6, 0.33, -0.2, 0.9)       # 0.03 outside the flank at yaw 0
    assert nc.clearance(0.0, 0.0, 0.0, RBY1, [beside])[0] >= nc.MIN_CLEAR_M
    ok, who = nc.path_clear((0.0, 0.0), 0.0, (0.0, -2.0), -math.pi / 2, RBY1, [beside])
    assert not ok and who == "beside"


def test_path_clear_carries_the_same_margin_as_the_park():
    """A drive that passes 3 mm from the fridge is not collision-free under a +-0.02 m goal band."""
    post = box("post", 0.0, 0.31, 0.5, 0.9)             # 0.01 outside the flank
    ok, _ = nc.path_clear((-1.0, 0.0), 0.0, (1.0, 0.0), 0.0, RBY1, [post])
    assert not ok


def test_kitchen_1570_spawn_band_cannot_reach_the_resited_park_in_a_straight_line():
    """The emitted band x[2.051, 2.284] y[-3.474, -2.484] is SOUTH of the refrigerator; a straight
    drive from it to any park west of the cabinet cuts the fridge's front corner (1.771, -2.2045)."""
    bounds, obstacles = _scene(KITCHEN_1570)
    band = [["x", 2.050999186771115, 2.2842807701293313], ["y", -3.4744721989858536, -2.4844722201865928]]
    park = nc.pick_clear_park(MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                              standoff_m=0.47, arm_bias=-0.05, footprint=RBY1,
                              obstacles=obstacles, prefer_side="W", bounds=bounds,
                              starts=[(*nc.band_centre(band), nc.SPAWN_YAW)])
    assert park is not None and park.side == "N"
    assert park.start is None                 # the best clear park, but nothing reaches it
    assert nc.bands_with_clear_path([band], (park.x, park.y, park.yaw), RBY1, obstacles) == []


def test_kitchen_1550_spawn_band_reaches_its_park_in_a_straight_line():
    """From the band's centre, not its corners: the emitted bands are inset for a 0.23 m stand-in,
    so a 0.35 m base already grazes the dishwasher from this band's east edge. That is a spawn
    defect the gate is not for; the gate is for drives that go THROUGH furniture."""
    bounds, obstacles = _scene(KITCHEN_1550)
    band = [["x", 0.386 + 0.28, 1.386 - 0.28], ["y", -3.34 + 0.28, -0.383 - 0.28]]
    assert nc.bands_with_clear_path([band], PARK_1550, RBY1, obstacles) == [0]


def test_the_park_beside_the_fridge_front_is_reachable_with_the_margin_from_the_repaired_band():
    """The nearest clear park on 1570 (x 1.387, 0.089 m off the fridge's front plane) is taken:
    from the repaired band's centre the approach is nearly aligned and the final turn stays
    within the turn tolerance. (Hand geometry said the rear corner would sweep into the fridge;
    the separating-axis test says a box turning beside a box's CORNER clears it -- believe the SAT.)"""
    bounds, obstacles = _scene(KITCHEN_1570)
    furniture = _furniture(KITCHEN_1570, "countertop_base_cabinet")
    bands = nc.spawn_bands(nc.free_rectangles(_furniture_boxes(KITCHEN_1570)), footprint=RBY1)
    starts = [(*nc.band_centre(b), nc.SPAWN_YAW) for b in bands]
    park = nc.pick_clear_park(MUG_1570, furniture, standoff_m=0.47, arm_bias=-0.05,
                              footprint=RBY1, obstacles=obstacles, prefer_side="W",
                              bounds=bounds, starts=starts)
    assert park.side == "N" and park.start is not None
    assert park.slide_m == 0.0 and park.extra_m == 0.0
    assert park.clearance_m == pytest.approx(0.087, abs=0.001)          # now the base cabinet, in x
    assert nc.path_clear(starts[park.start][:2], nc.SPAWN_YAW, (park.x, park.y), park.yaw,
                         RBY1, obstacles)[0]
    assert bands[park.start][0][2] <= 1.771                        # a band west of the fridge


def test_a_final_turn_within_the_tolerance_costs_no_reach():
    """Arriving along the counter (bearing 0) the base turns 90 deg in place at the park; at the
    usual standoff its corner passes 0.014 m from the counter -- inside the 0.02 margin, well
    within the turn tolerance. That must NOT push the park a step back: reach is the binding
    constraint and the tolerance exists to accept exactly this."""
    counter = box("counter", -1.0, 0.0, 4.0, 1.0)
    along = (0.3, -0.47, 0.0)
    usual = nc.park_on_side("W", counter, 1.5, 0.47)
    assert not nc.path_clear(along[:2], along[2], usual[:2], usual[2], RBY1, [counter],
                             turn_slop_m=-nc.MIN_CLEAR_M)[0]
    park = nc.pick_clear_park((1.5, 0.2), counter, standoff_m=0.47, arm_bias=0.0, footprint=RBY1,
                              obstacles=[counter], prefer_side="W", starts=[along])
    assert park.extra_m == 0.0 and park.slide_m == 0.0 and park.start == 0


def test_1550_rotation_11_gets_the_same_mug_park_as_the_other_rotations():
    """The free-space walk finds no side there (the rotated mug's box hangs 2.7 mm below the
    cabinet face and fragments the grid) and the emitted band is east of the range. The search
    from scratch must land on the pose rotations 00-10 author -- not 0.20 m along the counter --
    with a repaired band in front of the counters."""
    east_band = [["x", 1.696, 1.879], ["y", -3.06, -2.77]]
    r = nc.resolve_park(None, None, MUG_1550, _furniture(KITCHEN_1550, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1550,
                        bands=[east_band], target_name="mug0")
    assert r.side == "W"
    assert (r.x, r.y, r.yaw) == pytest.approx(PARK_1550)
    # the east band trims to nothing, so the run spawns from repaired bands -- one of them the
    # floor in front of the counters, which is the one the campaign's nearest-band pin selects
    assert r.bands is not None
    assert any(b[1][1] < -1.5 and b[0][2] < 1.4 for b in r.bands), r.bands


def test_the_start_turn_always_gets_the_tolerance():
    """The base is where it is: a start whose own turn grazes a table must not push the choice
    to a worse park. From a start 0.006 m short of the table's box, the pick equals the pick
    from an unobstructed start."""
    bounds, obstacles = _scene(KITCHEN_1570)
    furniture = _furniture(KITCHEN_1570, "countertop_base_cabinet")
    grazing = (0.7, -2.6, nc.SPAWN_YAW)               # rear corner sweeps 0.006 m into the table
    worst = min(nc.clearance(0.7, -2.6, math.radians(a), RBY1, obstacles)[0] for a in range(0, 70, 5))
    assert -nc.TURN_SLOP_M < worst < 0.0
    without_table = [o for o in obstacles if o.name != "table"]
    pick = lambda obs: nc.pick_clear_park(MUG_1570, furniture, standoff_m=0.47, arm_bias=-0.05,
                                          footprint=RBY1, obstacles=obs, prefer_side="W",
                                          bounds=bounds, starts=[grazing])
    a, b = pick(obstacles), pick(without_table)
    assert (a.x, a.y, a.yaw, a.side) == (b.x, b.y, b.yaw, b.side) and a.start == 0


# ----------------------------------------------------------------------------------------------
# Repairing the spawn band
# ----------------------------------------------------------------------------------------------

def _furniture_boxes(kitchen):
    """What _generate_env_config feeds the free-space grid: every /world child except the shell."""
    return [b for b in kitchen if not nc.is_shell_wall(b.name)]


def test_row_major_free_rectangles_recover_the_floor_in_front_of_the_1570_counters():
    """The column-first merge in isaaclab.simvla.utils fragments that floor into 0.43-0.56 m
    columns, all under the 2*0.23+0.101 size gate, so the emitted file has ONE band, south of
    the fridge. Merging rows first keeps the 2.2 m strip between the counters and the fridge."""
    bands = nc.spawn_bands(nc.free_rectangles(_furniture_boxes(KITCHEN_1570)))
    # a band west of the fridge's front plane that covers the floor beside it
    front = [b for b in bands if b[0][2] <= 1.771 + 1e-6 and b[1][1] < -1.4234 and b[1][2] > -2.2045]
    assert front, bands


def test_spawn_bands_apply_the_generators_size_gate_and_inset():
    """_generate_env_config: keep (w - 0.101) > 0.46 and (h - 0.101) > 0.46, inset by 0.23 + 0.05."""
    rects = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 3.0)]
    assert nc.spawn_bands(rects) == [[["x", 0.28, 0.72], ["y", 0.28, 0.72]]]


def test_spawn_bands_for_a_real_footprint_inset_by_its_longest_extent():
    """A band authored HERE is inset for the base that will spawn in it, not for the 0.23 m
    stand-in: RB-Y1 reaches 0.350 m behind its origin, so a 0.28 m inset spawns its corner
    0.07 m into whatever bounds the rectangle."""
    rects = [(0.0, 0.0, 0.8, 0.8), (0.0, 0.0, 2.0, 2.0)]        # 0.8 - 0.101 < 2 * 0.35: dropped
    bands = nc.spawn_bands(rects, footprint=RBY1)
    assert len(bands) == 1
    assert bands[0][0][1:] == pytest.approx([0.40, 1.60]) and bands[0][1][1:] == pytest.approx([0.40, 1.60])


def test_a_repaired_band_keeps_the_real_base_out_of_the_furniture_at_its_corners():
    _, obstacles = _scene(KITCHEN_1570)
    for band in nc.spawn_bands(nc.free_rectangles(_furniture_boxes(KITCHEN_1570)), footprint=RBY1):
        (_, x0, x1), (_, y0, y1) = band
        for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            gap, who = nc.clearance(cx, cy, nc.SPAWN_YAW, RBY1, obstacles)
            assert gap >= 0.0, (band, cx, cy, who, gap)
def test_repair_spawn_band_hands_a_clear_park_the_nearest_band_that_reaches_it():
    bounds, obstacles = _scene(KITCHEN_1570)
    park = (1.8574 - 0.47, MUG_1570[1] - 0.05 + 0.20, 0.0)
    band = nc.repair_spawn_band(_furniture_boxes(KITCHEN_1570), park, RBY1, obstacles)
    assert band is not None
    assert band[0][2] <= 1.771                       # west of the refrigerator
    assert nc.bands_with_clear_path([band], park, RBY1, obstacles) == [0]


def test_repair_spawn_band_is_none_when_nothing_reaches_the_park():
    top = box("countertop", 0.0, 0.0, 1.0, 1.0, 0.9, 0.95)
    ring = [box(f"wall{i}", *r, 0.0, 2.5) for i, r in enumerate(
        [(-1.5, -1.5, -0.2, 2.5), (1.2, -1.5, 2.5, 2.5), (-1.5, -1.5, 2.5, -0.2), (-1.5, 1.2, 2.5, 2.5)])]
    assert nc.repair_spawn_band([top] + ring, (0.5, -0.47, math.pi / 2), RBY1,
                                nc.floor_obstacles([top] + ring)) is None


# ----------------------------------------------------------------------------------------------
# The whole decision, as plan_nav_to_prim calls it
# ----------------------------------------------------------------------------------------------

BAND_1570 = [["x", 2.050999186771115, 2.2842807701293313], ["y", -3.4744721989858536, -2.4844722201865928]]
BAND_1550 = [["x", 0.386 + 0.28, 1.386 - 0.28], ["y", -3.34 + 0.28, -0.383 - 0.28]]


def test_resolve_keeps_a_clear_park_byte_for_byte():
    r = nc.resolve_park(PARK_1550, "W", MUG_1550, _furniture(KITCHEN_1550, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1550,
                        bands=[BAND_1550], target_name="mug0")
    assert (r.x, r.y, r.yaw) == PARK_1550           # the very same floats, not approx
    assert r.side == "W" and not r.changed and r.via == ()
    assert r.bands is None or _inside_band(r.bands[0], BAND_1550)     # the band may be trimmed
    assert r.clearance_m == pytest.approx(0.087, abs=0.002)


def _inside_band(inner, outer, tol=1e-6):
    return (outer[0][1] - tol <= inner[0][1] and inner[0][2] <= outer[0][2] + tol
            and outer[1][1] - tol <= inner[1][1] and inner[1][2] <= outer[1][2] + tol)


def test_resolve_resites_1570_and_trims_its_spawn_band_to_where_the_base_can_turn():
    """The emitted band is inset for the 0.23 m stand-in: at its north edge the real base's front
    is 4.5 cm inside the fridge, at its east edge 1.6 cm inside the wall. The band is TRIMMED to
    the part where the base fits and can turn in place, and stays inside the original."""
    r = nc.resolve_park(PARK_1570, "W", MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1570,
                        bands=[BAND_1570], target_name="mug0")
    assert r.changed and r.side == "N" and r.yaw == pytest.approx(0.0)
    assert r.clearance_m >= nc.MIN_CLEAR_M
    assert r.bands is not None and len(r.bands) == 1
    assert _inside_band(r.bands[0], BAND_1570)
    (_, x0, x1), (_, y0, y1) = r.bands[0]
    assert x1 <= 2.6135 - RBY1.radius_m + nc.TURN_SLOP_M + 1e-6          # can turn beside the wall
    assert y1 <= -2.2045 - RBY1.radius_m + nc.TURN_SLOP_M + 1e-6         # ...and beside the fridge
    assert r.via
    assert any("refrigerator" in n for n in r.notes)


def test_shrink_band_trims_1570s_band_to_the_turn_safe_part():
    bounds, obstacles = _scene(KITCHEN_1570)
    b = nc.shrink_band(BAND_1570, RBY1, obstacles, bounds)
    assert b is not None and _inside_band(b, BAND_1570)
    (_, x0, x1), (_, y0, y1) = b
    # west column: 1 mm short of the turning radius against the table; east: the wall; north:
    # the fridge; south: chair_1 (its seat ends at y -3.83)
    assert x0 == pytest.approx(2.098, abs=0.03) and x1 == pytest.approx(2.144, abs=0.03)
    assert y0 == pytest.approx(-3.375, abs=0.03) and y1 == pytest.approx(-2.682, abs=0.03)
    for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), ((x0 + x1) / 2, (y0 + y1) / 2)):
        worst = min(nc.clearance(cx, cy, math.radians(a), RBY1, obstacles)[0] for a in range(0, 180, 5))
        assert worst >= -nc.TURN_SLOP_M - 1e-9, (cx, cy, worst)


def test_shrink_band_leaves_a_band_alone_when_it_already_fits():
    bounds, obstacles = _scene(KITCHEN_1550)
    small = [["x", 0.80, 0.95], ["y", -2.5, -1.5]]
    assert nc.shrink_band(small, RBY1, obstacles, bounds) == small


def test_shrink_band_is_none_when_no_part_of_the_band_fits():
    bounds, obstacles = _scene(KITCHEN_1570)
    inside_fridge = [["x", 1.9, 2.3], ["y", -2.1, -1.6]]
    assert nc.shrink_band(inside_fridge, RBY1, obstacles, bounds) is None


def test_resolve_routes_around_a_fixture_and_keeps_the_pose_and_the_band():
    """A clear park whose only spawn band is on the far side of a fixture: the pose is "as
    usual", the band stays, and via-points take the base around the fixture."""
    counter = box("counter", 0.0, 1.0, 3.0, 1.6)
    fixture = box("fixture", 1.0, -1.0, 2.0, 0.0)
    stub = box("stub", 0.0, -3.0, 3.0, -2.9)                          # extends the room south
    boxes = [counter, fixture, stub]
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    far_band = [["x", 1.3, 1.7], ["y", -1.8, -1.5]]                  # south of the fixture
    r = nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=boxes, bands=[far_band])
    assert not r.changed and (r.x, r.y, r.yaw) == usual
    assert r.via
    assert r.bands is None or _inside_band(r.bands[0], far_band)
    _, obstacles = nc.scene_obstacles(boxes)
    start = (*nc.band_centre(r.bands[0] if r.bands else far_band), nc.SPAWN_YAW)
    assert _legs_clear(start, r.via, usual, obstacles)[0]


def test_resolve_raises_when_no_park_clears():
    top = box("countertop", 0.0, 0.0, 1.0, 1.0, 0.9, 0.95)
    ring = [box(f"wall{i}", *r, 0.0, 2.5) for i, r in enumerate(
        [(-1.5, -1.5, -0.2, 2.5), (1.2, -1.5, 2.5, 2.5), (-1.5, -1.5, 2.5, -0.2), (-1.5, 1.2, 2.5, 2.5)])]
    with pytest.raises(nc.NavClearanceError):
        nc.resolve_park((0.5, -0.47, math.pi / 2), "W", (0.5, 0.5), top, standoff_m=0.47,
                        arm_bias=0.0, footprint=RBY1, boxes=[top] + ring, bands=[])


def test_resolve_from_a_previous_park_never_touches_the_bands():
    """The second nav of a chain starts where the first parked, not in a spawn band."""
    r = nc.resolve_park(PARK_1570, "W", MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1570,
                        bands=[BAND_1570], target_name="mug0", prev_park=(0.5, -2.6, math.pi / 2))
    assert r.changed and r.side == "N"
    assert r.bands is None


def test_resolve_searches_from_scratch_when_the_free_space_walk_found_no_side():
    r = nc.resolve_park(None, None, MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1570,
                        bands=[BAND_1570], target_name="mug0")
    assert r.changed and r.side == "N"


# ----------------------------------------------------------------------------------------------
# Turning in place next to furniture, and ties between sides
# ----------------------------------------------------------------------------------------------

def test_a_turn_in_place_tolerates_a_few_cm_of_box_overlap():
    """Kitchen 1550's PROVEN mug park (18 recorded demos) is 0.087 m off the base cabinet's AABB,
    and the chain's next nav turns the base 180 deg right there. By this model that turn sweeps
    the corner 0.024 m INTO the cabinet's box -- a box that includes handles protruding 0.038 m.
    The drive keeps the full margin; a turn is allowed TURN_SLOP_M of overlap."""
    _, obstacles = _scene(KITCHEN_1550)
    x, y, _ = PARK_1550
    worst = min(nc.clearance(x, y, math.radians(a), RBY1, obstacles)[0] for a in range(90, -91, -5))
    assert -nc.TURN_SLOP_M < worst < 0.0
    table_park = (0.836, -2.85, -math.pi / 2)
    ok, who = nc.path_clear((x, y), math.pi / 2, table_park[:2], table_park[2], RBY1, obstacles)
    assert ok, who


def test_pick_clear_park_breaks_a_reach_tie_toward_the_start():
    """A fixture target is its own centre, so opposite sides tie on reach. Prefer the one the base
    is already nearer to -- the other one's straight drive goes through the fixture."""
    table = box("table", 0.3, -4.0, 1.4, -3.3)
    park_n = nc.pick_clear_park((0.85, -3.65), table, standoff_m=0.49, arm_bias=0.0,
                                footprint=RBY1, obstacles=[table], starts=[(0.85, -0.8, math.pi / 2)])
    assert park_n.side == "E" and park_n.start == 0
    park_s = nc.pick_clear_park((0.85, -3.65), table, standoff_m=0.49, arm_bias=0.0,
                                footprint=RBY1, obstacles=[table], starts=[(0.85, -6.0, math.pi / 2)])
    assert park_s.side == "W" and park_s.start == 0


def test_resolve_from_the_1550_mug_park_takes_the_near_side_of_the_table():
    """The second nav of 1550's chain, as rotation 11 exercised it (free-space walk found no
    side): the table's E side is what rotations 00-10 park at, and it is reachable."""
    table = _furniture(KITCHEN_1550, "table")
    r = nc.resolve_park(None, None, (0.886, -3.69), table, standoff_m=0.49, arm_bias=0.0,
                        footprint=RBY1, boxes=KITCHEN_1550, bands=[BAND_1550],
                        prev_park=PARK_1550)
    assert r.side == "E"
    assert (r.x, r.y) == pytest.approx((0.886, -3.3403 + 0.49))
    assert not any("WARNING" in n for n in r.notes)


def test_the_unreachable_fallback_still_prefers_the_side_nearest_the_start():
    """When NO start reaches any candidate, the best clear pose comes back; between two sides
    that tie on reach, the one nearer the start -- not the first in SIDES order."""
    table = box("table", 0.3, -4.0, 1.4, -3.3)
    cage = [box("cage_n", -1.0, -2.0, 3.0, -1.9), box("cage_s", -1.0, -5.4, 3.0, -5.3)]
    park = nc.pick_clear_park((0.85, -3.65), table, standoff_m=0.49, arm_bias=0.0,
                              footprint=RBY1, obstacles=[table] + cage,
                              starts=[(0.85, -0.8, math.pi / 2)])
    assert park.start is None and park.side == "E"


# ----------------------------------------------------------------------------------------------
# A clear park whose DRIVE collides is still a collision
# ----------------------------------------------------------------------------------------------

def test_resolve_resites_a_later_nav_whose_drive_from_the_previous_park_is_blocked():
    """1570's second leg as the emit authored it: the mug park is now west of the cabinet, the
    table park (kept "as usual") is at the table's +x side, and the straight line between them
    cuts the refrigerator's front corner. The table's +y side is equally near and reachable."""
    table = _furniture(KITCHEN_1570, "table")
    mug_park = (1.3874, -1.0628, 0.0)
    usual = (2.111, -3.4545, math.pi)
    r = nc.resolve_park(usual, "S", (1.071, -3.4045), table, standoff_m=0.49, arm_bias=0.0,
                        footprint=RBY1, boxes=KITCHEN_1570, bands=[BAND_1570], prev_park=mug_park)
    assert r.changed and r.side == "E"
    assert r.y == pytest.approx(-3.0545 + 0.49)
    _, obstacles = _scene(KITCHEN_1570)
    assert nc.path_clear(mug_park[:2], mug_park[2], (r.x, r.y), r.yaw, RBY1, obstacles)[0]
    assert r.bands is None


def test_resolve_keeps_a_clear_park_whose_drive_is_clear_even_when_nearer_sides_exist():
    """As usual unless it collides: 1550's table park is not the nearest side, and stays."""
    table = _furniture(KITCHEN_1550, "table")
    r = nc.resolve_park((0.836, -2.85, -math.pi / 2), "E", (0.886, -3.69), table, standoff_m=0.49,
                        arm_bias=0.0, footprint=RBY1, boxes=KITCHEN_1550, bands=[BAND_1550],
                        prev_park=PARK_1550)
    assert not r.changed and (r.x, r.y, r.yaw) == (0.836, -2.85, -math.pi / 2)


def _corridor():
    """A counter reached only through a 0.8 m slot between two pillars -- 0.8 - 0.101 is under the
    2 * 0.35 gate, so no repaired band can exist in it -- with a post across the slot between the
    only emitted band and the park. The park itself is clear by 0.1 m on each side."""
    counter = box("counter", 0.0, 1.0, 3.0, 1.6)
    pillar_l = box("pillar_l", -1.0, -0.5, 1.1, 1.0)
    pillar_r = box("pillar_r", 1.9, -0.5, 3.0, 1.0)
    post = box("post", 1.4, -0.1, 1.6, 0.05)
    south = [["x", 1.45, 1.55], ["y", -0.3, -0.2]]
    return counter, [pillar_l, pillar_r, post], south


def test_resolve_refuses_a_kitchen_whose_spawn_bands_cannot_hold_the_base():
    """The corridor is 0.8 m wide: no spawn there can turn in place, so every emitted band trims
    to nothing and the free-space grid offers no rectangle either."""
    counter, blocks, south = _corridor()
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    assert nc.clearance(*usual, RBY1, nc.floor_obstacles([counter] + blocks))[0] == pytest.approx(0.1)
    with pytest.raises(nc.NavClearanceError, match="spawn band"):
        nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=[counter] + blocks, bands=[south])


def test_resolve_raises_when_the_previous_park_cannot_reach_any_clear_park():
    counter, blocks, _ = _corridor()
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    with pytest.raises(nc.NavClearanceError, match="drive"):
        nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=[counter] + blocks, bands=[], prev_park=(1.5, -0.4, math.pi / 2))


def test_resolve_names_what_blocked_each_start_when_it_gives_up():
    counter, blocks, _ = _corridor()
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    with pytest.raises(nc.NavClearanceError, match="post"):
        nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=[counter] + blocks, bands=[], prev_park=(1.5, -0.25, math.pi / 2))


def test_free_rectangles_stack_rows_whose_x_extents_agree_within_a_few_cm():
    """Kitchen 1550's counter run: dishwasher x_min 1.4159, sink cabinet 1.4179, range 1.4159.
    The rows in front of them are each under the 0.80 m gate, and exact-extent stacking is
    defeated by 2 mm, so the floor in front of the counters never became a band (rotation 11's
    refusal). Rows whose extents agree within STACK_TOL_M stack on their common extent."""
    rects = nc.free_rectangles(_furniture_boxes(KITCHEN_1550))
    front = [r for r in rects if r[1] <= -2.49 + 1e-6 and r[3] >= -0.4153 - 1e-6 and r[2] <= 1.4159 + 1e-6]
    assert front, rects
    bands = nc.spawn_bands(rects, footprint=RBY1)
    assert any(b[1][1] < -1.5 and b[1][2] > -1.0 for b in bands), bands


# ----------------------------------------------------------------------------------------------
# Routing: via-points around furniture, so the spawn band stays where it was
# ----------------------------------------------------------------------------------------------

def _legs_clear(start, via, park, obstacles):
    poses = [start] + list(via) + [park]
    for a, b in zip(poses, poses[1:]):
        ok, who = nc.path_clear(a[:2], a[2], b[:2], b[2], RBY1, obstacles)
        if not ok:
            return False, who
    return True, None


def test_route_is_empty_when_the_straight_drive_is_clear():
    bounds, obstacles = _scene(KITCHEN_1550)
    start = (*nc.band_centre(BAND_1550), nc.SPAWN_YAW)
    assert nc.route(start, PARK_1550, RBY1, obstacles, bounds) == []


def test_route_goes_around_the_fridge_from_1570s_south_band():
    """The reference runs' spawn: south of the fridge, x 2.05-2.28. To reach the park west of the
    corner cabinet the base has to leave the band westward through the 0.85 m corridor between
    the table and the fridge, then drive north past the fridge front. Every leg must pass the
    exact box sweep, and the base must be clear of the fridge's front plane before it turns north."""
    bounds, obstacles = _scene(KITCHEN_1570)
    start = (*nc.band_centre(BAND_1570), nc.SPAWN_YAW)
    park = PARK_1570_N
    via = nc.route(start, park, RBY1, obstacles, bounds)
    assert via, "no route found"
    assert 1 <= len(via) <= 3, via
    ok, who = _legs_clear(start, via, park, obstacles)
    assert ok, who
    last = via[-1]
    assert last[0] + max(RBY1.y_max, RBY1.x_max) <= 1.771, last          # west of the fridge front


def test_route_works_from_the_corners_of_1570s_band_too():
    """The run spawns anywhere in the (trimmed) band, so the detour must hold from its corners."""
    bounds, obstacles = _scene(KITCHEN_1570)
    (_, x0, x1), (_, y0, y1) = nc.shrink_band(BAND_1570, RBY1, obstacles, bounds)
    park = PARK_1570_N
    for sx, sy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        start = (sx, sy, nc.SPAWN_YAW)
        via = nc.route(start, park, RBY1, obstacles, bounds)
        assert via, (sx, sy)
        assert _legs_clear(start, via, park, obstacles)[0], (sx, sy)


def test_route_is_none_when_the_only_corridor_is_posted():
    counter, blocks, south = _corridor()
    bounds, obstacles = nc.scene_obstacles([counter] + blocks)
    start = (*nc.band_centre(south), nc.SPAWN_YAW)
    assert nc.route(start, (1.5, 0.53, math.pi / 2), RBY1, obstacles, bounds) is None


def test_resolve_keeps_1570s_band_and_authors_via_points():
    """With a route available the emitted band is kept -- the whole point: the run starts where
    it used to and drives AROUND the fridge instead of into it."""
    r = nc.resolve_park(PARK_1570, "W", MUG_1570, _furniture(KITCHEN_1570, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1570,
                        bands=[BAND_1570], target_name="mug0")
    assert r.changed and r.side == "N"
    assert r.bands is None or _inside_band(r.bands[0], BAND_1570)
    assert r.via, r.notes
    _, obstacles = _scene(KITCHEN_1570)
    band = r.bands[0] if r.bands else BAND_1570
    for c in nc.band_corners(band) + [nc.band_centre(band)]:
        assert _legs_clear((*c, nc.SPAWN_YAW), r.via, (r.x, r.y, r.yaw), obstacles)[0], c
    assert any("via" in n for n in r.notes)


def test_resolve_routes_a_later_nav_before_resiting_it():
    """A later nav whose usual park is clear but straight-blocked from the previous park keeps
    the park AS USUAL with a detour; re-siting is only for when no detour exists."""
    counter = box("counter", 0.0, 1.0, 3.0, 1.6)
    fixture = box("fixture", 1.0, -1.0, 2.0, 0.0)
    stub = box("stub", 0.0, -3.0, 3.0, -2.9)                          # extends the room south
    boxes = [counter, fixture, stub]
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    prev = (1.5, -1.6, 0.0)
    r = nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=boxes, bands=[], prev_park=prev)
    assert not r.changed and (r.x, r.y, r.yaw) == usual
    assert r.via and r.bands is None
    _, obstacles = nc.scene_obstacles(boxes)
    assert _legs_clear(prev, r.via, usual, obstacles)[0]


def test_1570s_second_leg_is_resited_because_its_usual_park_cannot_even_be_turned_into():
    """The usual table park (east side, facing the table) is clear, but arriving from any
    direction the final turn in place sweeps 5 cm into chair_1 -- no route ends there. The
    table's +y side is equally near, clear, and a straight drive from the mug park."""
    table = _furniture(KITCHEN_1570, "table")
    mug_park = PARK_1570_N
    usual = (2.111, -3.4545, math.pi)
    bounds, obstacles = _scene(KITCHEN_1570)
    assert nc.route(mug_park, usual, RBY1, obstacles, bounds) is None
    r = nc.resolve_park(usual, "S", (1.071, -3.4045), table, standoff_m=0.49, arm_bias=0.0,
                        footprint=RBY1, boxes=KITCHEN_1570, bands=[BAND_1570], prev_park=mug_park)
    assert r.changed and r.side == "E" and r.via == ()


def test_resolution_via_is_empty_when_nothing_was_routed():
    r = nc.resolve_park(PARK_1550, "W", MUG_1550, _furniture(KITCHEN_1550, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1550,
                        bands=[BAND_1550], target_name="mug0")
    assert r.via == () and not r.changed



# ----------------------------------------------------------------------------------------------
# The arm bias means the same thing on every side
# ----------------------------------------------------------------------------------------------

def _object_in_base_frame(park, obj):
    x, y, yaw = park
    dx, dy = obj[0] - x, obj[1] - y
    return (dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw))


def test_a_right_arm_park_puts_the_object_on_the_robots_right_on_every_side():
    """nav_tuning.arm_lateral_bias: how far LEFT of the target the base parks, so the named arm
    does not reach across the body. plan_nav_to_prim applied `+ arm_bias` to the lateral
    coordinate on all four sides, and "left" is +y facing +x but -y facing -x: the N and E sides
    parked the base on the WRONG side of the target by the full 0.10 m."""
    furniture = box("f", 0.0, 0.0, 1.0, 1.0)
    obj = (0.5, 0.5)
    for side in nc.SIDES:
        right = nc.park_for_arm(side, furniture, obj, arm_bias=-0.05, standoff_m=0.47)
        _, lateral = _object_in_base_frame(right, obj)
        assert lateral == pytest.approx(-0.05), (side, right)              # on the robot's RIGHT
        left = nc.park_for_arm(side, furniture, obj, arm_bias=+0.05, standoff_m=0.47)
        _, lateral = _object_in_base_frame(left, obj)
        assert lateral == pytest.approx(+0.05), (side, left)


def test_the_search_uses_the_same_lateral_convention():
    furniture = box("f", 0.0, 0.0, 1.0, 1.0)
    obj = (0.5, 0.5)
    for side in nc.SIDES:
        ranked = [p for p in nc.ranked_parks(obj, furniture, 0.47, -0.05, RBY1, [], prefer_side=side)
                  if p.side == side and p.slide_m == 0.0 and p.extra_m == 0.0]
        assert ranked, side
        _, lateral = _object_in_base_frame((ranked[0].x, ranked[0].y, ranked[0].yaw), obj)
        assert lateral == pytest.approx(-0.05), side


# ----------------------------------------------------------------------------------------------
# Every spawn band is trimmed for the real base, not only the ones a route starts from
# ----------------------------------------------------------------------------------------------

def test_resolve_trims_1550s_band_where_the_real_base_spawns_in_contact():
    """1550's emitted band puts RB-Y1's front 6.5 cm into the base cabinet at its north edge and
    its flank 3.5 cm into the dishwasher at its east edge (the band is inset for the 0.23 m
    stand-in). The park is untouched; the band is trimmed to where the base fits and can turn."""
    r = nc.resolve_park(PARK_1550, "W", MUG_1550, _furniture(KITCHEN_1550, "countertop_base_cabinet"),
                        standoff_m=0.47, arm_bias=-0.05, footprint=RBY1, boxes=KITCHEN_1550,
                        bands=[BAND_1550], target_name="mug0")
    assert (r.x, r.y, r.yaw) == PARK_1550 and not r.changed and r.via == ()
    assert r.bands is not None and len(r.bands) == 1 and _inside_band(r.bands[0], BAND_1550)
    _, obstacles = _scene(KITCHEN_1550)
    for c in nc.band_corners(r.bands[0]):
        worst = min(nc.clearance(c[0], c[1], math.radians(a), RBY1, obstacles)[0] for a in range(0, 180, 5))
        assert worst >= -nc.TURN_SLOP_M - 1e-9, (c, worst)
    assert any("trimmed" in n for n in r.notes)


def test_a_band_that_already_fits_is_returned_untouched():
    counter = box("counter", 0.0, 1.0, 3.0, 1.6)
    stub = box("stub", 0.0, -3.0, 3.0, -2.9)
    usual = (1.5, 1.0 - 0.47, math.pi / 2)
    roomy = [["x", 1.2, 1.8], ["y", -1.6, -1.2]]
    r = nc.resolve_park(usual, "W", (1.5, 1.2), counter, standoff_m=0.47, arm_bias=0.0,
                        footprint=RBY1, boxes=[counter, stub], bands=[roomy])
    assert not r.changed and r.bands is None
