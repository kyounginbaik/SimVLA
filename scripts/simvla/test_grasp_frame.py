"""Moving a BODex grasp out of its synthesis frame and into the placed one.

The dataset tests are the real ones: they check the correction against BODex's own recorded contact
points, which are the two places the gripper's jaws touch the object. A grasp expressed in the right
frame puts those points ON the placed mesh; one expressed in the wrong frame puts them millimetres
inside or outside it. Nothing here needs Omniverse.

Run: cd scripts/simvla && pytest test_grasp_frame.py -v
"""
import glob
import os

import numpy as np
import pytest

import grasp_frame
from mesh_orientation import resolved_up

AXES = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

#: The two poses BODex actually used, as they appear in world_cfg (curobo wxyz).
POSE_Y_UP = [0.0, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0]      # core_*/ddg_*: +90 deg about X
POSE_Z_UP = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]            # sem_*: identity


def _data(pose):
    return {"world_cfg": [{"mesh": {"whatever/scale010": {"pose": pose}}}]}


# --- the rotation itself -----------------------------------------------------------------------

@pytest.mark.parametrize("up", AXES)
def test_rotation_to_z_stands_the_axis_up(up):
    assert np.allclose(grasp_frame.rotation_to_z(up) @ np.array(up, float), [0, 0, 1], atol=1e-9)


@pytest.mark.parametrize("up", AXES)
def test_rotation_to_z_is_a_proper_rotation(up):
    """A reflection would mirror every grasp -- left hand for right -- while still passing the
    test above."""
    m = grasp_frame.rotation_to_z(up)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(m), 1.0)


# --- reading the synthesis pose ----------------------------------------------------------------

def test_synthesis_rotation_reads_the_two_poses_the_dataset_uses():
    assert np.allclose(grasp_frame.synthesis_rotation(_data(POSE_Z_UP)), np.eye(3), atol=1e-4)
    y_up = grasp_frame.synthesis_rotation(_data(POSE_Y_UP))
    assert np.allclose(y_up @ np.array([0.0, 1.0, 0.0]), [0, 0, 1], atol=1e-3)


def test_synthesis_rotation_accepts_the_stringified_form():
    """world_cfg round-trips through str() in some files and stays an array in others."""
    as_text = grasp_frame.synthesis_rotation(_data("[0.     0.     0.     0.7071 0.7071 0. 0.]"))
    assert np.allclose(as_text, grasp_frame.synthesis_rotation(_data(POSE_Y_UP)), atol=1e-9)


@pytest.mark.parametrize("broken", [
    {}, {"world_cfg": []}, {"world_cfg": [{}]}, {"world_cfg": [{"mesh": {}}]},
    _data([0.0, 0.0, 0.0]), _data([0, 0, 0, 0, 0, 0, 0]), _data("nonsense"),
])
def test_an_unreadable_pose_degrades_to_the_old_behaviour(broken):
    """Identity means 'assume the grasps are already placed', which is exactly what
    plan_arm_grasp did before this module. A raise here would kill goal generation halfway."""
    assert grasp_frame.is_identity(grasp_frame.synthesis_rotation(broken))


# --- recovering the object type from the prim ----------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("bottle0", "bottle"), ("mug0", "mug"), ("apple0", "apple"), ("sodacan0", "sodacan"),
    ("cerealbox0", "cerealbox"), ("toiletpaper0", "toiletpaper"),
    ("mug", "mug"),                 # already bare
    ("mug00", "mug"),               # object_id_generator's doubled suffix
    ("", None), (None, None), ("0", None),
])
def test_object_type_is_recovered_from_the_prim_name(name, expected):
    assert grasp_frame.object_type_from_prim_name(name) == expected


def test_every_offered_type_survives_the_round_trip():
    """build_kitchen names a placed object <type><n>; if any type ended in a digit this recovery
    would silently truncate it and the apple/sodacan exception would be missed."""
    from scene_spec import CATEGORIES

    for obj_type in CATEGORIES:
        assert grasp_frame.object_type_from_prim_name(f"{obj_type}0") == obj_type


# --- the correction ----------------------------------------------------------------------------

def test_the_correction_vanishes_when_the_object_is_placed_as_it_was_synthesised():
    """This is what makes it safe to apply unconditionally instead of to a list of known-bad
    meshes -- a list would go stale the next time a label changes."""
    assert grasp_frame.is_identity(grasp_frame.correction((0, 1, 0), _data(POSE_Y_UP)))
    assert grasp_frame.is_identity(grasp_frame.correction((0, 0, 1), _data(POSE_Z_UP)))


def test_the_correction_is_a_half_turn_for_a_plate_labelled_z_down():
    """The plates are the population case: hand-labelled (0,0,-1), synthesised at (0,0,1), so
    their grasps arrive upside down."""
    c = grasp_frame.correction((0, 0, -1), _data(POSE_Z_UP))
    assert not grasp_frame.is_identity(c)
    assert np.allclose(c @ np.array([0.0, 0.0, 1.0]), [0, 0, -1], atol=1e-9)


def test_apply_returns_the_input_untouched_under_identity():
    xyz = np.array([[0.1, 0.2, 0.3]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    out_xyz, out_quat = grasp_frame.apply(np.eye(3), xyz, quat)
    assert np.allclose(out_xyz, xyz) and np.allclose(out_quat, quat)


def test_apply_turns_the_gripper_with_the_object():
    """Position and orientation have to move together: rotating only the position would slide the
    gripper around the object while it kept facing the way it did before."""
    c = grasp_frame.correction((0, 0, -1), _data(POSE_Z_UP))        # half turn about X
    xyz = np.array([[0.0, 0.0, 0.05]])
    quat = np.array([[1.0, 0.0, 0.0, 0.0]])                          # identity orientation
    out_xyz, out_quat = grasp_frame.apply(c, xyz, quat)

    assert np.allclose(out_xyz, [[0.0, 0.0, -0.05]], atol=1e-9)
    from scipy.spatial.transform import Rotation
    approach = Rotation.from_quat(out_quat[:, [1, 2, 3, 0]]).as_matrix()[0] @ np.array([0, 0, 1.0])
    assert np.allclose(approach, [0, 0, -1], atol=1e-9), "the gripper did not turn with the object"


def test_apply_keeps_quaternions_normalised():
    c = grasp_frame.correction((1, 0, 0), _data(POSE_Z_UP))
    quat = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]])
    _xyz, out = grasp_frame.apply(c, np.zeros((2, 3)), quat)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


# --- against the real dataset -------------------------------------------------------------------

DATASET = os.environ.get("BODEX_OBJ_DIR", "BODex_obj")
pytestmark_dataset = pytest.mark.skipif(
    not os.path.isdir(f"{DATASET}/use_data"), reason="no BODex dataset here"
)


def _placed_up(name):
    from scene_spec import CATEGORIES, MESH_EXCLUDE
    obj_type = None
    if name not in MESH_EXCLUDE:
        for t, prefixes in CATEGORIES.items():
            if name.startswith(prefixes):
                obj_type = t
                break
    return resolved_up(obj_type, f"{DATASET}/use_data/{name}/mesh/simplified.obj")


def _grasp_data(name, side="right"):
    folder = f"{DATASET}/graspdata_final/sim_parallel/{name}/floating"
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.endswith(f"_{side}.npy")]
    if not files:
        return None
    return np.load(os.path.join(folder, files[0]), allow_pickle=True).item()


def _grasped_meshes():
    out = []
    for path in sorted(glob.glob(f"{DATASET}/use_data/*")):
        name = os.path.basename(path)
        if os.path.isdir(f"{DATASET}/graspdata_final/sim_parallel/{name}/floating"):
            out.append(name)
    return out


@pytestmark_dataset
def test_every_grasp_file_records_one_of_the_two_known_synthesis_frames():
    """The correction is only as good as this: a file whose pose cannot be read falls back to
    identity and silently keeps the old, possibly-wrong frame. Measured at 357 of 357 readable,
    and only ever +Y up or +Z up — so a failure here means the dataset changed shape, not that
    the fallback is quietly carrying meshes."""
    meshes = _grasped_meshes()
    assert len(meshes) == 357, f"expected 357 grasped meshes, found {len(meshes)}"

    y_up = grasp_frame.rotation_to_z((0, 1, 0))
    seen = {}
    for name in meshes:
        rot = grasp_frame.synthesis_rotation(_grasp_data(name))
        which = ("+Z" if grasp_frame.is_identity(rot)
                 else "+Y" if np.allclose(rot, y_up, atol=1e-3) else "other")
        seen.setdefault(which, []).append(name)
    assert "other" not in seen, f"unrecognised synthesis frame: {seen.get('other', [])[:3]}"
    assert sorted(seen) == ["+Y", "+Z"]
    assert len(seen["+Y"]) == 227 and len(seen["+Z"]) == 130


@pytestmark_dataset
def test_the_correction_is_identity_wherever_the_frames_already_agree():
    """340 of the 357 -- everything the old rot_z-only path already got right must be left exactly
    as it was, or this 'fix' is a regression for the overwhelming majority.

    The 17 that do move are all label-vs-synthesis disagreements, and 7 of them are meshes
    scene_spec has WITHDRAWN (MESH_EXCLUDE), so only 10 can reach a scene. They are corrected
    anyway: the correction is derived per mesh, not from a list, so a mesh coming back into the
    registry does not need anyone to remember this."""
    changed = [n for n in _grasped_meshes()
               if not grasp_frame.is_identity(grasp_frame.correction(_placed_up(n), _grasp_data(n)))]
    assert len(changed) == 17, f"expected the 17 known mismatches, got {len(changed)}: {changed}"

    from scene_spec import MESH_EXCLUDE
    placeable = [n for n in changed if n not in MESH_EXCLUDE]
    assert len(placeable) == 10, f"expected 10 placeable mismatches, got {placeable}"


@pytestmark_dataset
def test_corrected_contacts_land_on_the_object_as_placed():
    """The end-to-end check, and the one that would catch a transposed or mirrored correction.

    contact_point holds where the two jaws touch. Rotated into the placed frame they must lie ON
    the placed mesh; measured, every mesh comes out at ~0.1 mm, which is mesh resolution. The
    uncorrected points are up to 6.9 mm out on the mismatched meshes -- small in absolute terms
    only because these objects are not much bigger than that error.
    """
    import trimesh

    worst_after = 0.0
    improved = 0
    for name in _grasped_meshes():
        data = _grasp_data(name)
        c = grasp_frame.correction(_placed_up(name), data)
        if grasp_frame.is_identity(c):
            continue                                    # covered by the test above
        contacts = data["contact_point"][0, :, 0, :, :].reshape(-1, 3)

        mesh = trimesh.load(f"{DATASET}/use_data/{name}/mesh/simplified.obj", force="mesh")
        mesh.apply_scale(0.1)
        placed = np.eye(4)
        placed[:3, :3] = grasp_frame.rotation_to_z(_placed_up(name))
        mesh.apply_transform(placed)

        def median_gap(points):
            _pt, dist, _tri = trimesh.proximity.closest_point(mesh, points)
            return float(np.median(dist))

        before, after = median_gap(contacts), median_gap(contacts @ c.T)
        worst_after = max(worst_after, after)
        improved += after <= before
        assert after < 0.001, f"{name}: corrected contacts sit {after * 1000:.2f} mm off the mesh"

    assert improved == 17, "the correction made some mesh worse"
    assert worst_after < 0.001, f"worst corrected gap {worst_after * 1000:.2f} mm"
