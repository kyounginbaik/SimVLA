"""The raise-and-filter that makes an authored grasp reachable once cuRobo's world is switched on.

Two kinds of test. The synthetic ones pin the logic on geometry small enough to check by hand. The
robot ones read the REAL vendored cuRobo configs and URDFs, because the whole point of the module
is to measure the gripper the planner measures -- a sphere set derived from some other file would
agree with itself and disagree with cuRobo.

Run: cd scripts/simvla && pytest test_grasp_clearance.py -v
"""
import os

import numpy as np
import pytest

import grasp_clearance as gc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IDENT = [1.0, 0.0, 0.0, 0.0]

#: One sphere of radius r at the tool origin, one jaw sphere either side of it. Small enough that
#: every expected number below is arithmetic.
TOY = (np.array([[0.0, 0.0, 0.0], [-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]),
       np.array([0.01, 0.01, 0.01]),
       np.array([False, True, True]))

#: A slab whose top face is at z = 1.0, wide in x/y.
SLAB_LO = np.array([[-1.0, -1.0, 0.5]])
SLAB_HI = np.array([[1.0, 1.0, 1.0]])

#: An object standing on the slab, 0.2 m tall.
OBJ_LO, OBJ_HI = np.array([-0.05, -0.05, 1.0]), np.array([0.05, 0.05, 1.2])


def test_sphere_clearance_is_signed_distance_to_the_nearest_face():
    # centre 0.03 above the slab top, radius 0.01 -> 0.02 of air
    c = gc.sphere_clearance(np.array([[0.0, 0.0, 1.03]]), np.array([0.01]), SLAB_LO, SLAB_HI)
    assert c == pytest.approx(0.02)
    # centre 0.005 BELOW the top: 0.005 in, plus the radius
    c = gc.sphere_clearance(np.array([[0.0, 0.0, 0.995]]), np.array([0.01]), SLAB_LO, SLAB_HI)
    assert c == pytest.approx(-0.015)
    # beside the slab, not above it: the distance is horizontal, not vertical
    c = gc.sphere_clearance(np.array([[1.05, 0.0, 0.9]]), np.array([0.01]), SLAB_LO, SLAB_HI)
    assert c == pytest.approx(0.04)


def test_no_boxes_is_infinite_clearance():
    assert gc.sphere_clearance(np.zeros((1, 3)), np.array([1.0]),
                               np.zeros((0, 3)), np.zeros((0, 3))) == float("inf")


def test_a_clear_grasp_is_left_exactly_alone():
    pose = [0.0, 0.0, 1.10] + IDENT
    kept, notes = gc.clear_grasps([pose], TOY, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI, clearance=0.005)
    assert kept == [pose]
    assert notes[0]["raise"] == 0.0 and notes[0]["kept"]


def test_a_grasp_inside_the_slab_is_raised_by_the_least_that_clears():
    # tool origin 0.005 above the top: the r=0.01 spheres are 0.005 INSIDE.
    pose = [0.0, 0.0, 1.005] + IDENT
    kept, notes = gc.clear_grasps([pose], TOY, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI,
                                  clearance=0.005, step=0.001)
    assert notes[0]["clearance_before"] == pytest.approx(-0.005)
    # needs the sphere bottom 0.005 clear of z=1.0 -> origin at 1.015 -> a raise of 0.010, and the
    # scan may take one more step of float slack rather than sit exactly on the boundary.
    assert 0.010 <= notes[0]["raise"] <= 0.010 + 2 * 0.001
    assert kept[0][2] == pytest.approx(1.005 + notes[0]["raise"])
    assert notes[0]["clearance_after"] >= 0.005


def test_the_raise_never_moves_the_jaws_off_the_object():
    # A slab so deep that clearing it would need the jaws above the object's top.
    tall_lo, tall_hi = np.array([[-1.0, -1.0, 0.5]]), np.array([[1.0, 1.0, 1.19]])
    pose = [0.0, 0.0, 1.10] + IDENT
    kept, notes = gc.clear_grasps([pose], TOY, tall_lo, tall_hi, OBJ_LO, OBJ_HI, clearance=0.005)
    assert kept == []
    assert not notes[0]["kept"]


def test_a_grasp_needing_more_than_the_cap_is_dropped_not_dragged():
    pose = [0.0, 0.0, 1.005] + IDENT
    kept, _ = gc.clear_grasps([pose], TOY, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI,
                              clearance=0.005, max_raise=0.002)
    assert kept == []


def test_orientation_is_never_touched():
    q = [0.7071, 0.0, 0.7071, 0.0]
    kept, _ = gc.clear_grasps([[0.0, 0.0, 1.005] + q], TOY, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI,
                              clearance=0.005)
    assert kept and kept[0][3:] == q


def test_the_sphere_set_is_ROTATED_by_the_grasp_before_it_is_measured():
    """Same position, same spheres, opposite orientation -- and the answer must differ, because
    where the jaws actually are depends on the grasp's rotation. Measuring the ee-frame offsets
    unrotated would give both poses the same verdict."""
    sph = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.10], [0.0, 0.0, 0.10]]),
           np.array([0.01, 0.01, 0.01]), np.array([False, True, True]))
    pose_up = [0.0, 0.0, 1.005] + IDENT               # jaws 0.10 ABOVE the origin -> at 1.105, clear
    pose_dn = [0.0, 0.0, 1.005, 0.0, 1.0, 0.0, 0.0]   # 180 about x: jaws 0.10 BELOW -> 0.895 deep in
    _, n_up = gc.clear_grasps([pose_up], sph, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI, clearance=0.005)
    _, n_dn = gc.clear_grasps([pose_dn], sph, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI, clearance=0.005)
    assert n_up[0]["kept"] and n_up[0]["raise"] < 0.02      # only the palm was in the slab
    assert not n_dn[0]["kept"]                              # the jaws are 0.105 deep; past the cap


def test_the_cap_is_set_by_where_the_ROTATED_jaws_sit_on_the_object():
    """Two tools differing only in how far the jaws lead the tool origin. The one whose jaws are
    already at the object's top has no headroom left and must be dropped; the one whose jaws are at
    the origin has 0.195 m of object above it and is raised the 0.011 it needs."""
    palm_in_slab = [0.0, 0.0, 1.005] + IDENT
    near_top = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.195], [0.0, 0.0, 0.195]]),
                np.array([0.01, 0.01, 0.01]), np.array([False, True, True]))
    at_origin = (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                 np.array([0.01, 0.01, 0.01]), np.array([False, True, True]))
    _, n_top = gc.clear_grasps([palm_in_slab], near_top, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI,
                               clearance=0.005)
    _, n_org = gc.clear_grasps([palm_in_slab], at_origin, SLAB_LO, SLAB_HI, OBJ_LO, OBJ_HI,
                               clearance=0.005)
    assert not n_top[0]["kept"]
    assert n_org[0]["kept"] and n_org[0]["raise"] < 0.02


def test_switch_off_is_the_env_var_not_a_code_edit(monkeypatch):
    monkeypatch.setenv("SIMVLA_GRASP_CLEARANCE", "0")
    assert gc.required_clearance() == 0.0
    monkeypatch.setenv("SIMVLA_GRASP_CLEARANCE", "0.01")
    assert gc.required_clearance() == 0.01
    monkeypatch.delenv("SIMVLA_GRASP_CLEARANCE")
    assert gc.required_clearance() == gc.REQUIRED_CLEARANCE_M


def test_required_clearance_exceeds_curobos_activation_distance():
    """1e-3 is collision_activation_distance at simvla_video.py's two MotionGenConfig calls. A
    required clearance at or below it authors poses the planner is entitled to refuse."""
    assert gc.REQUIRED_CLEARANCE_M > 1e-3


# ---------------------------------------------------------------------------------------------
# The real robots.
# ---------------------------------------------------------------------------------------------
def test_anubis_tool_spheres_match_the_vendored_config():
    if not os.path.isfile(os.path.join(
        os.environ.get("SIMVLA_ASSETS_DIR", os.path.join(REPO, "source/isaaclab_assets/data")),
        "curobo/robot/anubis_right_arm.yml",
    )):
        pytest.skip("Anubis cuRobo configs are external; set SIMVLA_ASSETS_DIR to run this integration check")
    cen, rad, jaw = gc.tool_spheres("anubis", "right", REPO)
    # 1 palm + 5 per jaw, and the buffer is IN the radii (0.035 palm + 0.005).
    assert len(rad) == 11
    assert jaw.sum() == 10
    assert rad[~jaw][0] == pytest.approx(0.040)
    # ee_fixed_joint1 puts ee_link1 0.10956 along +z of gripper_base_link, so the palm sphere sits
    # that far BEHIND the tool origin.
    assert cen[~jaw][0] == pytest.approx([0.0, 0.0, -0.10956], abs=1e-6)
    # the two jaws straddle the tool axis: gripper1_joint locks at 0.04 on -x, gripper1R on +x.
    assert cen[jaw][:, 0].min() < 0 < cen[jaw][:, 0].max()


def test_anubis_LEFT_arm_has_no_jaw_spheres_and_that_does_not_crash():
    """A pre-existing asymmetry in the vendored configs, not something this module introduces:
    anubis_left_arm.yml carries no lock_joints and lists only gripper2_base_link as a collision
    link, so cuRobo cannot see that hand's fingers at all. The check has to measure exactly what
    the planner measures -- one palm sphere, no jaws -- rather than invent finger geometry the
    planner will not enforce, and the jawless cap must fall back rather than produce NaN."""
    if not os.path.isfile(os.path.join(
        os.environ.get("SIMVLA_ASSETS_DIR", os.path.join(REPO, "source/isaaclab_assets/data")),
        "curobo/robot/anubis_left_arm.yml",
    )):
        pytest.skip("Anubis cuRobo configs are external; set SIMVLA_ASSETS_DIR to run this integration check")
    cen, rad, jaw = gc.tool_spheres("anubis", "left", REPO)
    assert not jaw.any() and len(rad) == 1
    # The one sphere is the palm, 0.1096 m behind the tool origin with a 0.040 m radius. A grasp
    # whose origin is at the slab top buries it; one 0.16 m up clears it. Both verdicts must be
    # finite -- the jawless cap falling through to NaN would make every candidate "clear".
    deep, high = [0.0, 0.0, 1.005] + IDENT, [0.0, 0.0, 1.165] + IDENT
    for pose, want in ((deep, False), (high, True)):
        kept, notes = gc.clear_grasps([pose], (cen, rad, jaw), SLAB_LO, SLAB_HI,
                                      OBJ_LO, OBJ_HI, clearance=0.005)
        assert np.isfinite(notes[0]["clearance_before"])
        assert notes[0]["kept"] is want


def test_rby1_tool_spheres_come_out_of_its_own_urdf():
    if not os.environ.get("SIMVLA_RBY1M_DIR") or not os.path.isfile(os.path.join(
        os.environ.get("SIMVLA_ASSETS_DIR", os.path.join(REPO, "source/isaaclab_assets/data")),
        "curobo/robot/rby1_right_arm.yml",
    )):
        pytest.skip("RB-Y1 model/config assets are external; set SIMVLA_RBY1M_DIR and SIMVLA_ASSETS_DIR")
    cen, rad, jaw = gc.tool_spheres("rby1", "right", REPO)
    assert len(rad) > 0 and jaw.any()
    # Only what is rigid w.r.t. the tool: the arm links above the wrist must not be here.
    assert np.abs(cen).max() < 0.5


def test_relative_sphere_file_resolves_from_external_assets_root(tmp_path, monkeypatch):
    """External robot bundles keep the cuRobo-relative spheres inside their assets root."""
    assets = tmp_path / "portable-assets"
    models = tmp_path / "portable-models"
    (assets / "curobo/robot/spheres").mkdir(parents=True)
    models.mkdir()
    (assets / "curobo/robot/test_right_arm.yml").write_text(
        "robot_cfg:\n  kinematics:\n"
        "    collision_spheres: curobo/robot/spheres/test_spheres.yml\n"
        "    collision_sphere_buffer: 0.005\n"
        "    urdf_path: test.urdf\n"
        "    ee_link: tool\n"
        "    collision_link_names: [tool]\n"
    )
    (assets / "curobo/robot/spheres/test_spheres.yml").write_text(
        "collision_spheres:\n  tool:\n    - center: [0.0, 0.0, 0.0]\n      radius: 0.02\n"
    )
    (models / "test.urdf").write_text(
        '<robot name="test"><link name="base"/><link name="tool"/>'
        '<joint name="mount" type="fixed"><parent link="base"/>'
        '<child link="tool"/></joint></robot>'
    )
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(assets))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(models))

    centres, radii, jaws = gc.tool_spheres("test", "right", str(tmp_path / "wrong-checkout"))

    assert centres.tolist() == [[0.0, 0.0, 0.0]]
    assert radii.tolist() == pytest.approx([0.025])
    assert not jaws.any()


def test_aiworker_uses_checked_in_geometry_over_stale_asset_config(tmp_path, monkeypatch):
    assets, repo, models = (tmp_path / name for name in ("assets", "repo", "models"))
    (assets / "curobo/robot").mkdir(parents=True)
    (repo / "configs/curobo/robot").mkdir(parents=True)
    models.mkdir()
    (assets / "curobo/robot/aiworker_left_arm.yml").write_text("stale asset config")
    (repo / "configs/curobo/robot/aiworker_left_arm.yml").write_text(
        "robot_cfg:\n  kinematics:\n    ee_link: tool\n    urdf_path: test.urdf\n"
        "    collision_link_names: [tool]\n    collision_sphere_buffer: 0.005\n"
        "    collision_spheres:\n      tool:\n        - center: [0, 0, 0]\n          radius: 0.01\n")
    (models / "test.urdf").write_text(
        '<robot name="test"><link name="base"/><link name="tool"/>'
        '<joint name="mount" type="fixed"><parent link="base"/>'
        '<child link="tool"/></joint></robot>')
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(assets))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(models))
    centres, radii, _ = gc.tool_spheres("aiworker", "left", str(repo))
    assert centres.tolist() == [[0, 0, 0]]
    assert radii.tolist() == pytest.approx([0.015])


def test_the_authored_1201_mug_grasps_are_mostly_unreachable_and_the_raise_recovers_them():
    """The regression this module exists for, on the real Anubis tool geometry and the real
    countertop slab. The numbers are the measured ones: a 0.950 m countertop top, a mug from 0.952
    to 1.104, and the fifteen authored candidates of kitchen 1201 rotation 00."""
    goals = "campaign_out/mugchain2/goals/Isaac-Kitchen-v1201-00.json"
    if not os.path.exists(goals):
        pytest.skip("the 1201 goal corpus is not on this filesystem")
    import json
    step = [s for s in json.load(open(goals))["goals"][0] if s["skill"] == "arm.grasp"][0]
    poses = step["goal"]
    sph = gc.tool_spheres("anubis", "right", REPO)
    # The countertop slab alone -- the one obstacle that is exactly an AABB.
    lo = np.array([[1.953, -1.430, 0.915]])
    hi = np.array([[2.673, -0.430, 0.950]])
    obj_lo, obj_hi = np.array([1.940, -1.206, 0.952]), np.array([2.020, -1.099, 1.104])

    _, off = gc.clear_grasps(poses, sph, lo, hi, obj_lo, obj_hi, clearance=0.0, max_raise=0.0)
    blocked = [x for x in off if x["clearance_before"] < 1e-3]
    assert len(blocked) >= 8, "the authored candidates should mostly be inside the countertop"
    assert min(x["clearance_before"] for x in off) < -0.01

    kept, notes = gc.clear_grasps(poses, sph, lo, hi, obj_lo, obj_hi)
    assert len(kept) > len([x for x in off if x["clearance_before"] >= gc.REQUIRED_CLEARANCE_M])
    for k, n in zip(kept, [x for x in notes if x["kept"]]):
        assert n["clearance_after"] >= gc.REQUIRED_CLEARANCE_M
    # and nothing was raised off the mug
    assert all(k[2] <= obj_hi[2] for k in kept)
