"""Tests for the table manifest and the table furniture type.

Run with: conda run -n env_isaaclab python -m pytest scripts/simvla/test_table_manifest.py -v

Never touches the network. Synthetic fixtures are trimesh primitives; the real-mesh fixtures are
the uids the desk gate names for exactly this purpose ("Fixture uids for Task 3's tests"),
already cached by that gate's own download run.
"""

import glob
import os
import sys
from pathlib import Path

import numpy
import pytest
import trimesh

import table_manifest

# build_table_manifest.py lives in scripts/tools, a sibling of scripts/simvla with no package
# __init__.py to bridge them -- mirrors test_chair_manifest.py's own sys.path insert.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_chair_manifest as pipeline           # noqa: E402  the shared engine
import build_table_manifest as B                  # noqa: E402
from build_table_manifest import TABLE, DEFAULT_THRESHOLDS, TABLE_SURFACE_HEIGHT_M  # noqa: E402

#: The two caches this cluster keeps Objaverse GLBs in. The desk gate downloaded into the first;
#: OBJAVERSE_PATH points fetch at the second, on /lustre, because /home is a shared quota.
OBJAVERSE_CACHE_GLOBS = (
    os.path.join(os.environ.get("OBJAVERSE_PATH", os.path.expanduser("~/.objaverse")),
                 "hf-objaverse-v1/glbs/*/{uid}.glb"),
    os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs/*/{uid}.glb"),
    "objaverse_cache/hf-objaverse-v1/glbs/*/{uid}.glb",
)

# Fixture uids, and the numbers the gate published for each. Kept together so a disagreement is
# visible as a disagreement rather than as a mysterious failure.
ACCEPT_UID = "2568b63db4c9410a863efe11467e6851"          # plain 4-leg table, 876 faces
LOW_POLY_UID = "b874f4ffbafd4152b3f28868e2fa79f4"        # 60 faces -- under the CHAIR floor of 100
HEAVY_UID = "cccefbf8c7354944a32545abebafc6cb"           # 486,616 faces -- over the CHAIR ceiling
OFFICE_SCENE_UID = "07dd49275fc84617b5c6d3bd6a7d30a1"    # cubicle scene, ws_frac 0.60
DINING_SET_UID = "b16d2b8158ef4adca2a0f7a917928ec4"      # round table + chairs, overhang 2.02
NO_SURFACE_UID = "df1aa375f775420b90a20c1eaec9e016"      # L-shaped counter: no qualifying plane
COFFEE_TABLE_UID = "9e19bc36beef4093a067f6cc1f1aced8"    # 2.59 m long at a 0.74 m work surface


def _cached_glb(uid):
    """Path to a real mesh already downloaded into one of the two caches. Never downloads -- glob
    only. Fails loudly (not a skip) if the cache is gone: these uids are named as fixtures by the
    findings doc, and silently skipping would remove the only real-mesh coverage these tests
    have."""
    matches = [m for pattern in OBJAVERSE_CACHE_GLOBS
               for m in glob.glob(pattern.format(uid=uid))]
    assert matches, (
        f"{uid}.glb not found in any objaverse cache ({', '.join(OBJAVERSE_CACHE_GLOBS)}) -- this "
        "uid is named as a fixture by the desk findings doc. Restore the cache "
        "(objaverse.load_objects([uid])) rather than skipping."
    )
    return matches[0]


def _curated(uid, path):
    return pipeline.curate({"tables": [{"uid": uid, "raw_path": path}]},
                           spec=TABLE)["tables"][0]


# --- the manifest module ------------------------------------------------------------------------

def test_the_table_root_honours_its_own_environment_variable(monkeypatch, tmp_path):
    """TABLE_OBJ_DIR, not CHAIR_OBJ_DIR: the two libraries are separate installs and pointing one
    at the other's root would have them overwrite each other's manifest."""
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path))
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path / "chairs"))
    assert table_manifest.root() == str(tmp_path)
    assert table_manifest.manifest_path() == str(tmp_path / "table_manifest.json")


def test_an_absent_table_manifest_gives_a_skeleton_keyed_on_tables(monkeypatch, tmp_path):
    """Mutation: reusing the chair skeleton's "chairs" key makes every stage silently operate on an
    empty list while the real entries sit under a key nothing reads."""
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path / "nothing-here"))
    assert table_manifest.load() == {
        "objaverse_version": None, "generated_from": None, "thresholds": None, "tables": [],
    }


def test_the_table_manifest_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path))
    table_manifest.save({"tables": [{"uid": "u", "accepted": True}, {"uid": "v"}]})
    loaded = table_manifest.load()
    assert [e["uid"] for e in table_manifest.accepted(loaded)] == ["u"]


# --- the work-surface measurement, against the gate's published numbers ---------------------------

@pytest.mark.parametrize("uid,ws_frac,overhang", [
    (ACCEPT_UID, 1.00, 1.00),
    (OFFICE_SCENE_UID, 0.60, 1.12),
    (DINING_SET_UID, 0.95, 2.02),
    (COFFEE_TABLE_UID, 1.00, 1.00),
])
def test_the_work_surface_reproduces_the_gates_published_measurements(uid, ws_frac, overhang):
    """The gate measured these by hand-written script in a scratchpad; this is the reimplementation,
    and it has to agree or every threshold derived from those numbers is being applied to a
    different quantity. Tolerance 0.025 -- the gate published two decimal places and measures the
    band's height slightly differently, which shifts a fraction by up to 0.02.

    Mutation: summing |n . up| x area across the whole band instead of taking the larger of the two
    facings double-counts a thin slab, and finds a "work surface" halfway up a staircase.
    """
    loaded = trimesh.load(_cached_glb(uid), process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    found = B._work_surface(mesh, 1)                       # 1 = Y, glTF's declared up

    assert found is not None
    assert found["work_surface_frac"] == pytest.approx(ws_frac, abs=0.025)
    assert found["overhang"] == pytest.approx(overhang, abs=0.03)


def test_a_mesh_with_no_qualifying_plane_measures_as_having_none():
    """`df1aa375f775` is an L-shaped reception counter: no horizontal band anywhere along Y covers
    half its own footprint. The gate records "—" for it, and that absence is the up-axis validator's
    whole signal."""
    loaded = trimesh.load(_cached_glb(NO_SURFACE_UID), process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    assert B._work_surface(mesh, 1) is None


# --- curate, on real meshes ----------------------------------------------------------------------

def test_a_real_table_is_accepted():
    entry = _curated(ACCEPT_UID, _cached_glb(ACCEPT_UID))
    assert entry["accepted"] is True, entry.get("rejected_because")
    assert entry["up_axis"] == "Y" and entry["up_axis_source"] == "gltf-y-up"


def test_an_office_scene_is_rejected_because_something_sits_above_the_work_surface():
    """The `desk` category's whole failure mode -- a monitor, hutch, partition or lamp above the
    tabletop pushes the work-surface fraction down. 17 of the gate's 24 `desk` meshes contain at
    least one object that is not the desk."""
    entry = _curated(OFFICE_SCENE_UID, _cached_glb(OFFICE_SCENE_UID))
    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("work surface")


def test_a_dining_set_is_rejected_because_its_chairs_stick_out_past_the_table():
    """Fatal for this project specifically: the wizard places its own chairs on a ring derived from
    the table's bounds, so a table arriving with its own chairs would be surrounded by a second,
    misaligned set."""
    entry = _curated(DINING_SET_UID, _cached_glb(DINING_SET_UID))
    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("overhang")


def test_a_mesh_with_no_work_surface_at_all_is_rejected_first():
    """Ordered first because every later measurement is meaningless without a plane to measure
    against -- including the export scale, which divides by work_surface_frac."""
    entry = _curated(NO_SURFACE_UID, _cached_glb(NO_SURFACE_UID))
    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("no work surface")


def test_a_coffee_table_is_rejected_for_the_footprint_it_would_ship_at():
    """`9e19bc36beef` is a clean, correct, hand-labelled-usable ornate table. Its proportions are a
    COFFEE table's (h/L 0.29), so raising its top to a 0.74 m desk height makes it 2.59 x 1.21 m --
    a banquet table. The `table` LVIS category mixes the two and one height constant cannot serve
    both, which is why the footprint band exists at all."""
    entry = _curated(COFFEE_TABLE_UID, _cached_glb(COFFEE_TABLE_UID))
    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("footprint")
    assert max(entry["export_footprint_m"]) > DEFAULT_THRESHOLDS["footprint_max_m"]


def test_the_face_band_is_the_tables_own_and_not_the_chairs():
    """Both of these are hand-labelled usable and BOTH would be thrown away by the chair pipeline's
    inherited band -- 60 faces is under its floor of 100, 486,616 is over its ceiling of 200,000.
    Four of the gate's 20 usable meshes, a fifth of the yield, were at stake here.

    Mutation: putting the chair numbers back into DEFAULT_THRESHOLDS reddens both halves.
    """
    low = _curated(LOW_POLY_UID, _cached_glb(LOW_POLY_UID))
    heavy = _curated(HEAVY_UID, _cached_glb(HEAVY_UID))

    assert low["faces"] < pipeline.DEFAULT_THRESHOLDS["face_floor"]
    assert heavy["faces"] > pipeline.DEFAULT_THRESHOLDS["face_ceiling"]
    assert low["accepted"] is True, low.get("rejected_because")
    assert heavy["accepted"] is True, heavy.get("rejected_because")


def test_a_table_records_no_facing_axis():
    """The deliberate non-port. Chairs separate 170x against their own noise floor on the asymmetry
    measure and tables 9x; 13 of 15 usable tables fall below the chair gate's own confidence cut.
    Recording a facing axis anyway would produce exactly the confident nonsense
    FACING_CONFIDENCE_CUT exists to prevent.

    Mutation: adding _facing_axis to TableType.measure_extras reddens this.
    """
    entry = _curated(ACCEPT_UID, _cached_glb(ACCEPT_UID))
    assert "facing_axis" not in entry
    assert entry["work_surface_frac"] is not None and entry["overhang"] is not None


# --- curate, on shapes built to isolate one criterion each ----------------------------------------

def _table_scene(*, width=1.2, depth=0.8, height=0.75, top=0.05):
    """A plain 4-leg table, Y-up as glTF declares, as a multi-geometry Scene."""
    scene = trimesh.Scene()
    slab = trimesh.creation.box(extents=(width, top, depth))
    scene.add_geometry(slab, transform=trimesh.transformations.translation_matrix(
        (0, height - top / 2, 0)))
    for x in (-width / 2 + 0.06, width / 2 - 0.06):
        for z in (-depth / 2 + 0.06, depth / 2 - 0.06):
            leg = trimesh.creation.box(extents=(0.06, height - top, 0.06))
            scene.add_geometry(leg, transform=trimesh.transformations.translation_matrix(
                (x, (height - top) / 2, z)))
    return scene


def _glb(scene, tmp_path, name):
    path = tmp_path / f"{name}.glb"
    scene.export(path)
    return str(path)


def test_a_plain_synthetic_table_is_accepted(tmp_path):
    entry = _curated("plain", _glb(_table_scene(), tmp_path, "plain"))
    assert entry["accepted"] is True, entry.get("rejected_because")
    assert entry["work_surface_frac"] == pytest.approx(1.0, abs=1e-6)
    assert entry["overhang"] == pytest.approx(1.0, abs=1e-6)


def test_a_monitor_welded_onto_the_tabletop_is_rejected(tmp_path):
    """The `desk` failure mode, isolated: the mesh is a perfectly good table plus one object above
    its work surface. Nothing about its aspect ratio, face count or footprint is unusual -- which is
    precisely why the chair pipeline's criteria would admit it."""
    scene = _table_scene()
    panel = trimesh.creation.box(extents=(0.5, 0.35, 0.03))
    scene.add_geometry(panel, transform=trimesh.transformations.translation_matrix(
        (0, 0.75 + 0.175, 0)))
    entry = _curated("monitor", _glb(scene, tmp_path, "monitor"))

    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("work surface")
    assert entry["work_surface_frac"] < DEFAULT_THRESHOLDS["work_surface_frac_min"]


def test_a_stool_welded_beside_the_table_is_rejected_for_overhang(tmp_path):
    """The `dining_table` failure mode, isolated, and kept SHORTER than the table on purpose so the
    work-surface fraction stays at 1.00 and this test can only pass or fail on overhang."""
    scene = _table_scene()
    stool = trimesh.creation.box(extents=(0.4, 0.7, 0.4))
    scene.add_geometry(stool, transform=trimesh.transformations.translation_matrix((0.8, 0.35, 0)))
    entry = _curated("stool", _glb(scene, tmp_path, "stool"))

    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("overhang")
    assert entry["work_surface_frac"] == pytest.approx(1.0, abs=1e-6)


def test_a_slab_lying_in_the_horizontal_plane_is_rejected_for_the_size_it_would_ship_at(tmp_path):
    """The billboard case the chair pipeline's _aspect_ratio catches -- and the reason THAT
    criterion was blind for the pipeline's whole life is the trap this test guards against: while
    the up axis was argmax(extents), normalized_extents[up] was identically 1.0 by construction and
    no horizontal could ever exceed the band.

    Nothing here is 1.0 by construction. A 2 x 2 x 0.02 slab has a real work surface and an
    overhang of exactly 1.0, and is rejected because scaling its 0.02 "height" to a 0.74 m work
    surface would make it 74 m across.
    """
    slab = trimesh.creation.box(extents=(2.0, 0.02, 2.0))
    slab = slab.subdivide().subdivide()                    # clear the 50-face floor
    path = tmp_path / "slab.glb"
    slab.export(path)
    entry = _curated("slab", str(path))

    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("footprint")
    assert entry["work_surface_frac"] == pytest.approx(1.0, abs=1e-6)


def test_a_thin_standing_panel_is_a_KNOWN_FALSE_ACCEPT_only_the_contact_sheet_can_catch(tmp_path):
    """Recorded, not hidden, and asserted in the direction it actually behaves.

    A panel standing on its edge has a horizontal top face that covers its own (very thin)
    footprint, so it has a work surface at 1.00 and an overhang of 1.00, and its LONGER horizontal
    extent lands inside the footprint band. The chair pipeline rejects such a thing on its aspect
    band; the desk gate measured that band and found it does not transfer -- usable h/L spans
    [0.285, 0.893] and rejects span [0.204, 1.145], with 34 of 40 rejects inside the usable range --
    so porting it would cost real tables to catch a case the gate's own sample does not contain.

    Bounding the SHORTER horizontal extent would catch it, and is deliberately not done: nothing
    has measured where that bound goes, and this project's ledger forbids a threshold chosen rather
    than measured. Both extents are recorded on every entry so a later measurement can add one
    without re-loading a mesh. Until then this is MANUAL_REJECTS' job, which is why the contact
    sheet is not optional.

    Mutation: this test reddens the day someone adds a shorter-extent bound -- at which point it
    should be replaced by that bound's own test, not deleted.
    """
    panel = trimesh.creation.box(extents=(1.5, 1.0, 0.02)).subdivide().subdivide()
    path = tmp_path / "panel.glb"
    panel.export(path)
    entry = _curated("panel", str(path))

    assert entry["accepted"] is True
    assert min(entry["export_footprint_m"]) < 0.05


# --- the hand verdicts ----------------------------------------------------------------------------

#: A room shell -- a floor and three walls -- labelled `kitchen_table`, and one of five
#: architectural models the first sheet of fifty caught.
ROOM_SHELL_UID = "cc1e1a66e7bc431487a05388c24b1449"


def test_an_architectural_model_passes_every_criterion_and_is_hand_rejected_anyway():
    """The measurement that justifies MANUAL_REJECTS existing at all, asserted rather than claimed.

    A building's top floor slab genuinely IS a horizontal plane covering its own footprint with
    nothing above it and nothing outside it -- which is the criterion's definition of a work
    surface. Five of the first fifty were structural frames and room shells, all from
    `kitchen_table`, all passing. The desk gate found the same thing in `desk`, where three of its
    sixty meshes were staircases and an escalator.

    Mutation: dropping this uid from MANUAL_REJECTS puts a room shell back in the table gallery,
    and NO threshold change would find it -- which is the whole point.
    """
    assert ROOM_SHELL_UID in B.MANUAL_REJECTS
    entry = _curated(ROOM_SHELL_UID, _cached_glb(ROOM_SHELL_UID))

    assert entry["accepted"] is False
    assert entry["manual_reject"] is True
    assert entry["rejected_because"].startswith("manual reject: not a table")
    # The evidence the verdict rests on: every geometric criterion says yes.
    assert [criterion(entry, DEFAULT_THRESHOLDS, ()) for criterion in B.CRITERIA] == [None] * 6
    assert entry["work_surface_frac"] >= DEFAULT_THRESHOLDS["work_surface_frac_min"]
    assert entry["overhang"] <= DEFAULT_THRESHOLDS["overhang_max"]


def test_a_hand_verdict_retracts_a_table_a_previous_run_accepted_and_survives_a_recurate():
    """A hand verdict has to outlive the next rebuild or the next selection picks the mesh straight
    back. curate applies MANUAL_REJECTS before its already-judged skip, so this retracts an entry
    that is already accepted, exported and converted.

    Mutation: moving the manual pass below `if "accepted" in entry: continue` leaves the accepted
    verdict standing.
    """
    manifest = {"tables": [{"uid": ROOM_SHELL_UID, "raw_path": _cached_glb(ROOM_SHELL_UID),
                            "accepted": True, "selected": True, "obj_path": "/nonexistent.obj"}]}
    entry = pipeline.curate(manifest, spec=TABLE)["tables"][0]

    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("manual reject")
    # And it carries the measurements it was judged beside, so the verdict has evidence attached.
    assert entry["work_surface_frac"] is not None and "faces" in entry


# --- export ---------------------------------------------------------------------------------------

def _exported(uid, raw_path, tmp_path, monkeypatch):
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path))
    manifest = pipeline.curate({"tables": [{"uid": uid, "raw_path": raw_path}]}, spec=TABLE)
    manifest["tables"][0]["accepted"] = True               # export judges nothing; curate does
    manifest = pipeline.export(manifest, spec=TABLE)
    entry = manifest["tables"][0]
    return entry, trimesh.load(entry["obj_path"], process=False, force="mesh")


def test_export_puts_the_work_surface_at_074_not_the_bounding_box_top(tmp_path, monkeypatch):
    """The measured consequence of getting this wrong: on the two usable meshes whose work surface
    is not the top of the bounding box (fractions 0.65 and 0.775), a bounding-box scale puts the
    work surface at 0.48 m and 0.57 m -- a doll's table under a real chair.

    Mutation: returning mesh.bounding_box.extents[2] from TableType.export_scale (the chair rule)
    leaves the overall height at 0.74 and drops the surface to 0.60.
    """
    scene = _table_scene()
    panel = trimesh.creation.box(extents=(0.5, 0.35, 0.03))
    scene.add_geometry(panel, transform=trimesh.transformations.translation_matrix(
        (0, 0.75 + 0.175, 0)))
    entry, mesh = _exported("hutch", _glb(scene, tmp_path, "hutch"), tmp_path, monkeypatch)

    assert entry["surface_height_m"] == pytest.approx(TABLE_SURFACE_HEIGHT_M, abs=1e-6)
    assert float(mesh.bounding_box.extents[2]) > TABLE_SURFACE_HEIGHT_M + 0.2
    assert entry["height_m"] == pytest.approx(entry["export_height_m"], abs=1e-6)


def test_export_centres_the_table_over_the_origin_and_sits_it_on_the_floor(tmp_path, monkeypatch):
    """Carried across from the chair library's live placement bug, not re-derived: add_chair (and
    add_table) compose the placement transform with the asset's own frame, so an author's arbitrary
    origin passes straight through. A chair 9.95 m from its origin landed 9.953 m from the seat the
    ring computed for it, outside the kitchen, touching nothing -- so the overlap gate saw nothing.

    Mutation: restoring a z-only translation leaves the first assert red and the floor assert
    green, which is exactly how it survived in the chair library.
    """
    scene = _table_scene()
    scene.apply_translation([9.95, -3.0, 12.0])
    entry, mesh = _exported("stray", _glb(scene, tmp_path, "stray"), tmp_path, monkeypatch)

    low, high = mesh.bounds
    assert (low[:2] + high[:2]) / 2 == pytest.approx([0.0, 0.0], abs=1e-6)
    assert low[2] == pytest.approx(0.0, abs=1e-6)


def test_curate_predicts_the_footprint_export_actually_writes(tmp_path, monkeypatch):
    """The claim that lets a post-scale criterion (_footprint) live in curate at all, checked end to
    end on a real cached mesh: curate predicts, export writes an OBJ, the OBJ is measured back.

    Mutation: dropping the / work_surface_frac from either TableType.export_geometry or
    TableType.export_scale breaks the agreement -- and _footprint would then be judging a table
    that ships at some other size.
    """
    entry, mesh = _exported(ACCEPT_UID, _cached_glb(ACCEPT_UID), tmp_path, monkeypatch)
    actual = sorted(float(v) for v in mesh.bounding_box.extents[:2])

    assert actual == pytest.approx(entry["export_footprint_m"], abs=1e-6)
    assert float(mesh.bounding_box.extents[2]) == pytest.approx(entry["export_height_m"], abs=1e-6)


def test_two_tables_of_the_same_shape_export_to_the_same_height(tmp_path, monkeypatch):
    """No per-uid Gaussian draw. Chairs sample N(0.85, 0.05) because real chairs genuinely vary;
    real work surfaces do not, and a fixed constant is what keeps mesh tables consistent with the
    nine procedural variants that are all exactly 0.74.

    Mutation: reusing _sample_height in TableType.export_scale reddens this, and would also make
    every mesh table a different height from its procedural neighbours in the same gallery.
    """
    path = _glb(_table_scene(), tmp_path, "twin")
    first, _ = _exported("uid-one", path, tmp_path / "a", monkeypatch)
    second, _ = _exported("uid-two", path, tmp_path / "b", monkeypatch)

    assert first["height_m"] == pytest.approx(second["height_m"], abs=1e-9)
    assert first["surface_height_m"] == pytest.approx(TABLE_SURFACE_HEIGHT_M, abs=1e-6)


# --- the shape descriptor --------------------------------------------------------------------------

def test_the_shape_grid_is_normalised_so_a_wide_table_is_not_clipped_to_its_border_cells():
    """_shape_grid's box is x, y in [-0.5, 0.5] and z in [0, 1], and its own docstring states the
    invariant that keeps a mesh inside it: "x and y stay inside [-0.5, 0.5] for free, since curate
    defines up_axis as argmax(extents)". That stopped being true for chairs when the up axis started
    coming from the file format, and it is false for EVERY table -- a table's height/longer-footprint
    ratio runs 0.285 to 0.893, so height-normalising one pushes most of its width outside the grid,
    where np.clip stacks it into the border columns. The descriptor would then describe a central
    slice of the table rather than the table.

    This asserts the invariant directly rather than an IoU, because an IoU between two clipped
    descriptors is not evidence about anything: measured on this pair it happens to come out LOWER
    under clipping (0.32 against 0.58), which says only that clipping changes the answer, not that
    it degrades it in a predictable direction.

    Mutation: reverting TableType.grid_scale to the chair rule reddens the first assert -- 63% of
    this table's width lands outside the grid.
    """
    table = _table_scene(width=2.0, depth=0.7, height=0.75).to_geometry()

    def half_width(spec):
        mesh = table.copy()
        pipeline._stand_up(mesh, 1)
        mesh.apply_scale(1.0 / spec.grid_scale(mesh))
        return float(numpy.abs(numpy.asarray(mesh.vertices)[:, :2]).max())

    assert half_width(TABLE) == pytest.approx(0.5, abs=1e-9)
    assert half_width(pipeline.CHAIR) > 1.3


# --- the seam itself --------------------------------------------------------------------------------

def test_the_table_type_shares_the_engines_up_axis_rule():
    """The one fork the plan predicted and the code does not need. `_up_axis` already trusts glTF's
    declared +Y and falls back to argmax(extents) only where no convention is declared -- which IS
    the desk gate's recommendation, measured 20/20 for Y-up against argmax's 0/20. Forking it would
    have produced two copies of one rule."""
    wide = trimesh.creation.box(extents=(2.0, 0.7, 0.9))     # a table: wider than it is tall
    assert pipeline._up_axis(wide, "/x/y.glb") == (1, "gltf-y-up")
    assert pipeline._up_axis(wide, "/x/y.obj") == (0, "argmax-extents")


def test_the_table_type_carries_no_category_cap():
    """Measured, not inherited. The chair cap exists because deck_chair supplied 13 of 50 from a
    24-uid list whose tail was junk, and its own docstring records that a cap is category-BLIND, so
    it cannot prefer good categories. Here the yields are 94% for `table` against 17% for `desk` and
    5% for `dining_table`: a cap would push the selection toward office scenes to buy a balance
    nobody asked for."""
    assert TABLE.category_cap is None
    assert pipeline.CHAIR.category_cap == pipeline.DEFAULT_CATEGORY_CAP


def test_dining_table_is_not_a_source_category():
    """15 of the gate's 20 `dining_table` meshes are dining SETS carrying 1-8 of their own chairs.
    The wizard seats its own chairs on a ring derived from the table's bounds, so such a table
    arrives surrounded by a second, misaligned set. This is not a yield problem to be outvoted by a
    bigger sample -- it is disqualifying."""
    assert "dining_table" not in B.DEFAULT_CATEGORIES
    assert B.DEFAULT_CATEGORIES[0] == "table"


def test_the_engine_still_defaults_to_chairs_when_no_spec_is_given():
    """Every stage takes `spec=None` and falls back to CHAIR, which is what keeps the chair suite
    passing unedited. Mutation: making `spec` required reddens the whole chair suite at once."""
    manifest = pipeline.curate({"chairs": []})
    assert manifest["thresholds"] == pipeline.DEFAULT_THRESHOLDS
    assert pipeline.offered({"chairs": [{"accepted": True, "uid": "u"}]}) == [
        {"accepted": True, "uid": "u"}]


# --- the CLI ------------------------------------------------------------------------------------

def _fake_downloader(tmp_path):
    """Three tables of DIFFERENT proportion. Identical ones would be rejected as re-uploads of each
    other by _distinctness, which is correct and would make this a test of that instead."""
    shapes = [(1.2, 0.8), (1.7, 0.7), (0.9, 0.9)]

    def download(uids, download_processes=1):
        paths = {}
        for uid, (width, depth) in zip(uids, shapes):
            path = tmp_path / f"{uid}.glb"
            _table_scene(width=width, depth=depth).export(path)
            paths[uid] = str(path)
        return paths
    return download


def test_the_cli_drives_the_table_spec_end_to_end(tmp_path, monkeypatch, capsys):
    """argv in, table_manifest.json out, under "tables" and in TABLE_OBJ_DIR. No network, no Isaac.

    Mutation: forgetting spec=TABLE in build_table_manifest.main writes the tables into the CHAIR
    manifest -- which is silent, and would be found only by a wizard offering chairs that are
    tables.
    """
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path))
    B.main(["fetch", "--count", "3"], download=_fake_downloader(tmp_path),
           uids=["uid-a", "uid-b", "uid-c"])
    B.main(["curate"])
    B.main(["select", "--target", "2"])
    B.main(["export"])

    manifest = table_manifest.load()
    assert [e["uid"] for e in manifest["tables"]] == ["uid-a", "uid-b", "uid-c"]
    assert manifest["thresholds"] == DEFAULT_THRESHOLDS
    assert len(pipeline.offered(manifest, spec=TABLE)) == 2
    assert all(os.path.exists(e["obj_path"]) for e in pipeline.offered(manifest, spec=TABLE))
    assert "tables: 3 total, 3 accepted" in capsys.readouterr().out


def test_the_table_contact_sheet_uses_a_window_wide_enough_for_a_two_metre_table(
        tmp_path, monkeypatch):
    """A table may ship at the full 2.00 m footprint bound; the chair library's 0.7 m window would
    crop every cell. Mutation: inheriting CONTACT_SHEET_HALF_EXTENT_M from the chair module puts
    this table in `overflowed`."""
    monkeypatch.setenv("TABLE_OBJ_DIR", str(tmp_path))
    manifest = pipeline.curate(
        {"tables": [{"uid": "wide", "raw_path": _glb(
            _table_scene(width=2.0, depth=1.0), tmp_path, "wide")}]}, spec=TABLE)
    manifest = pipeline.export(manifest, spec=TABLE)

    report = pipeline.contact_sheet(manifest, columns=1, spec=TABLE)
    assert report["overflowed"] == []
    assert report["drawn"] == ["wide"]
    assert numpy.isclose(TABLE.contact_sheet_half_extent_m, B.CONTACT_SHEET_HALF_EXTENT_M)
