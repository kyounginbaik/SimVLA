"""Per-mesh up axis, and the two values that must follow from it.

Run: cd scripts/simvla && pytest test_mesh_orientation.py -v
"""
import os

import pytest

from mesh_orientation import MESH_UP, front_for, origin_for, up_for

AXES = {(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)}

#: What kitchen_build._object_pose returned before this table existed. front_for/origin_for must
#: reproduce both, or the change is not a generalisation of the old behaviour but a replacement.
LEGACY_APPLE = ((0, 0, 1), (0, 1, 0), ("com", "com", "bottom"))
LEGACY_OTHER = ((0, 1, 0), (0, 0, -1), ("com", "bottom", "com"))


def test_every_recorded_up_is_an_axis_aligned_unit_vector():
    """front_for and origin_for both assume this; a diagonal entry would break them silently."""
    assert {tuple(v) for v in MESH_UP.values()} <= AXES


def test_front_is_perpendicular_to_up():
    """scene_synthesizer aligns front and up independently; a front parallel to up is degenerate,
    and the object comes out at whatever angle the library falls back to."""
    for up in AXES:
        front = front_for(up)
        assert sum(a * b for a, b in zip(up, front)) == 0, f"{front} is not perpendicular to {up}"


def test_origin_puts_the_world_lowest_point_on_the_support():
    """Measured: up=(0,0,-1) with 'bottom' hangs the object 12.6 cm BELOW the countertop, and
    up=(0,0,1) with the old ('com','bottom','com') sinks it 6.1 cm in. The word has to follow the
    sign of the up component, on the up axis."""
    assert origin_for((0, 0, 1)) == ("com", "com", "bottom")
    assert origin_for((0, 0, -1)) == ("com", "com", "top")
    assert origin_for((0, 1, 0)) == ("com", "bottom", "com")
    assert origin_for((0, -1, 0)) == ("com", "top", "com")
    assert origin_for((1, 0, 0)) == ("bottom", "com", "com")
    assert origin_for((-1, 0, 0)) == ("top", "com", "com")


def test_the_rules_reproduce_the_two_poses_that_existed_before():
    """This change generalises _object_pose rather than replacing it."""
    up, front, origin = LEGACY_APPLE
    assert (front_for(up), origin_for(up)) == (front, origin)
    up, front, origin = LEGACY_OTHER
    assert (front_for(up), origin_for(up)) == (front, origin)


@pytest.mark.parametrize("bad", [(0, 0, 0), (1, 1, 0), (0, 0, 2), (0.5, 0, 0.5)])
def test_a_non_axis_up_is_refused_rather_than_quietly_mishandled(bad):
    """Deriving a front from a diagonal up would produce a non-perpendicular pair, and the object
    would stand at an angle — the exact symptom this table exists to remove."""
    with pytest.raises(ValueError):
        front_for(bad)
    with pytest.raises(ValueError):
        origin_for(bad)


def test_up_for_reads_the_table_by_the_same_name_the_planner_uses():
    """object_name's '/-3' derivation is shared with load_grasp_file; drift means the table
    describes one mesh while the scene places another."""
    name = next(iter(MESH_UP))
    assert up_for(f"/data/use_data/{name}/mesh/simplified.obj") == MESH_UP[name]


def test_a_mesh_absent_from_the_table_has_no_override():
    assert up_for("/data/use_data/core_mug_1038e4ea/mesh/simplified.obj") is None
    assert up_for(None) is None


def test_no_mug_or_bowl_mesh_is_overridden():
    """The only objects shown to grasp today are core_* mugs and bowls. The table is all sem_*,
    so this change cannot reach them — that is what makes it safe to ship before the grasp-frame
    question is settled."""
    assert not [n for n in MESH_UP if n.startswith(("core_mug", "core_bowl"))]


# --- the object pose, as kitchen_build hands it to scene_synthesizer -------------------------

def test_object_pose_is_unchanged_for_a_mesh_that_is_not_listed():
    from kitchen_build import _object_pose
    assert _object_pose("apple") == LEGACY_APPLE
    assert _object_pose("sodacan") == LEGACY_APPLE
    assert _object_pose("mug") == LEGACY_OTHER
    assert _object_pose("mug", "/data/use_data/core_mug_1038e4ea/mesh/simplified.obj") == LEGACY_OTHER


def test_the_placer_and_the_picker_resolve_up_through_the_same_call():
    """kitchen_gallery stands its tiles on resolved_up and kitchen_build places on _object_pose.
    A second copy of the fallback rule in either one is how the picker came to disagree with the
    counter in the first place, so the placer must take its answer from resolved_up and nowhere
    else."""
    from mesh_orientation import resolved_up
    from kitchen_build import _object_pose

    listed = next(iter(MESH_UP))
    for obj_type, path in [
        ("mug", None),                                              # type fallback
        ("apple", None),                                            # the +Z exception
        ("sodacan", None),
        ("nosuchtype", None),                                       # unknown type
        ("vase", f"/data/use_data/{listed}/mesh/simplified.obj"),   # per-mesh override
        ("mug", "/data/use_data/core_mug_1038e4ea/mesh/simplified.obj"),
    ]:
        assert _object_pose(obj_type, path)[0] == resolved_up(obj_type, path), (obj_type, path)


def test_resolved_up_always_answers_with_an_axis():
    """The picker rotates by whatever comes back and the placer derives front and origin from it;
    a None or a diagonal would be a silent tilt in one and a raise in the other."""
    from mesh_orientation import resolved_up

    for obj_type in ("mug", "apple", "sodacan", "cerealbox", "", None):
        assert resolved_up(obj_type) in AXES


def test_object_pose_follows_the_table_for_a_listed_mesh():
    from kitchen_build import _object_pose
    name = next(n for n, v in MESH_UP.items() if tuple(v) == (0, 0, 1))
    up, front, origin = _object_pose("vase", f"/data/use_data/{name}/mesh/simplified.obj")
    assert (up, front, origin) == ((0, 0, 1), (0, 1, 0), ("com", "com", "bottom"))


# --- real dataset ---------------------------------------------------------------------------

DATASET = os.environ.get("BODEX_OBJ_DIR", "BODex_obj")


#: Correctly upright, resting on LESS of their surface than the old up=(0,1,0) fallback did, and
#: no taller for it either -- so the "it was stood up" rule in the test below cannot vouch for
#: them. In every case the fallback balanced the mesh on an edge or a rim that happens to carry a
#: lot of vertices, and a vertex-count contact measure reads that as a big footprint:
#:   Cap             -- correctly on its brim
#:   DrinkingUtensil -- a goblet, correctly on its foot
#:   Toaster         -- the fallback stood it on its narrow end; upright is the wide, low stance
#:   RiceCooker      -- the fallback tipped it onto its lid rim; upright is base down, knob up
#:   SoapBar         -- the fallback stood the bar on edge; upright is lying flat
#: Contact area was only ever the heuristic that found candidates; the human labels are the
#: criterion, and they overruled it here. Pinned by name so a sixth mesh joining them fails this
#: test instead of passing unnoticed.
LOW_CONTACT_BY_DESIGN = {
    "sem_Cap_c2cf2cf35d08662945c5fa74440a4519",
    "sem_DrinkingUtensil_2e8d37b3dd65c312b8f0377fb16daf9d",
    "sem_Toaster_2fb52b43697b9341b785a4ac4a0dbd73",
    "sem_RiceCooker_86175eb31a23a452757a72d9e80d1698",
    "sem_SoapBar_b3718bdc7025004813f5ec8466b8a488",
}


@pytest.mark.skipif(not os.path.isdir(f"{DATASET}/use_data"), reason="no BODex dataset here")
def test_no_offered_sem_mesh_is_left_to_the_fallback():
    """A sem_* mesh with no entry takes up=(0,1,0), the core_* convention, and is placed on its
    side. Nothing downstream catches it: the mesh picker draws the raw file frame corrected by
    THIS table, so an unlabelled Z-up mesh looks upright in the gallery right up until the scene
    lays it down. Coverage is asserted rather than documented, because the failure is silent —
    a sem_* mesh added to CATEGORIES fails here until somebody labels it.
    """
    import glob

    from grasp_manifest import object_name
    from kitchen_build import _object_pose, matching_meshes
    from scene_spec import CATEGORIES

    mesh_files = sorted(glob.glob(f"{DATASET}/use_data/*/mesh/simplified.obj"))
    assert mesh_files, "dataset is present but no meshes matched"

    laid_down = sorted({
        object_name(path)
        for obj_type in CATEGORIES
        for path in matching_meshes(obj_type, mesh_files)
        if object_name(path).startswith("sem_") and _object_pose(obj_type, path)[0] == (0, 1, 0)
    })
    assert not laid_down, f"{len(laid_down)} sem_* mesh(es) on the core_* fallback: {laid_down[:5]}"


@pytest.mark.skipif(not os.path.isdir(f"{DATASET}/use_data"), reason="no BODex dataset here")
def test_every_listed_mesh_rests_on_the_support():
    """The mechanical half, and the one that is not a matter of judgement: whatever `up` says, the
    object's lowest point must land exactly on the surface. Getting this wrong is invisible in a
    render and fatal in physics -- the old origin sinks a corrected mesh 6.1 cm into the counter,
    and a Z-down one hangs 12.6 cm below it."""
    from kitchen_build import _object_pose
    from scene_synthesizer.assets import Asset

    floating = []
    for name in MESH_UP:
        path = f"{DATASET}/use_data/{name}/mesh/simplified.obj"
        if not os.path.exists(path):
            continue
        up, front, origin = _object_pose("", path)
        mesh = Asset(fname=path, scale=0.1, up=up, front=front, origin=origin, align=True).mesh()
        low = float(mesh.vertices[:, 2].min())
        if abs(low) > 1e-6:
            floating.append((name, round(low * 100, 2)))
    assert not floating, f"not resting on the support (cm): {floating[:5]}"


@pytest.mark.skipif(not os.path.isdir(f"{DATASET}/use_data"), reason="no BODex dataset here")
def test_the_table_puts_the_population_on_its_feet():
    """Measured through the real asset pipeline over all 126: median bottom contact 2.8% -> 13.7%,
    90 improved outright. Asserted on the population rather than per mesh, because some meshes are
    correctly upright with LESS contact and a per-mesh rule would encode the heuristic as the
    criterion -- which is the mistake that produced the wrong answer three times.

    Losing contact is only allowed two ways. Standing UP is the ordinary one: a cereal box on its
    base touches far less of the counter than the fallback did laying it on its largest face, and
    31 of the tall packages are in exactly that position (the smallest of them gains 25% height).
    Height separates that from being tipped over, so it needs no list. The remaining 5 are pinned
    by name in LOW_CONTACT_BY_DESIGN."""
    import statistics

    from kitchen_build import _object_pose
    from scene_synthesizer.assets import Asset

    def contact(mesh):
        z = mesh.vertices[:, 2]
        return float((z < z.min() + 0.002).sum()) / len(z)

    def height(mesh):
        z = mesh.vertices[:, 2]
        return float(z.max() - z.min())

    before, after, worse = [], [], set()
    for name in MESH_UP:
        path = f"{DATASET}/use_data/{name}/mesh/simplified.obj"
        if not os.path.exists(path):
            continue
        old = Asset(fname=path, scale=0.1, up=(0, 1, 0), front=(0, 0, -1),
                    origin=("com", "bottom", "com"), align=True).mesh()
        up, front, origin = _object_pose("", path)
        new = Asset(fname=path, scale=0.1, up=up, front=front, origin=origin, align=True).mesh()
        before.append(contact(old))
        after.append(contact(new))
        if contact(new) < contact(old) and height(new) <= height(old):
            worse.add(name)

    assert statistics.median(after) > 3 * statistics.median(before)
    assert worse == LOW_CONTACT_BY_DESIGN, f"unexpected regressions: {worse - LOW_CONTACT_BY_DESIGN}"
