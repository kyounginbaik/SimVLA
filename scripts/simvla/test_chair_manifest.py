"""Tests for the chair manifest and its pipeline stages.

Run with: conda run -n env_isaaclab python -m pytest scripts/simvla/test_chair_manifest.py -v
Never touches the network: downloads are slow and rate-limited, so every test here builds its
own fixtures with trimesh primitives, or reads a mesh already cached under ~/.objaverse by Task
1's download run. The fetch stage is tested through an injected downloader, not against
Objaverse; curate never downloads at all -- it only reads raw_path.
"""

import glob
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy
import pytest
import trimesh

import chair_manifest

# build_chair_manifest.py lives in scripts/tools, a sibling of scripts/simvla with no package
# __init__.py to bridge them -- mirrors test_build_grasp_manifest.py's own sys.path insert for
# the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_the_root_honours_the_environment_variable(monkeypatch, tmp_path):
    """Mirrors BODEX_OBJ_DIR (grasp_manifest.py:18) so a checkout without /lustre can point
    somewhere else rather than failing."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    assert chair_manifest.root() == str(tmp_path)
    assert chair_manifest.manifest_path().startswith(str(tmp_path))


def test_loading_an_absent_manifest_gives_an_empty_skeleton(monkeypatch, tmp_path):
    """Every stage reads the manifest before writing it, so the first run must not have to
    special-case a missing file."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = chair_manifest.load()
    assert manifest["chairs"] == []
    assert "objaverse_version" in manifest


def test_accepted_returns_only_accepted_entries():
    manifest = {"chairs": [
        {"uid": "a", "accepted": True},
        {"uid": "b", "accepted": False, "rejected_because": "aspect ratio"},
        {"uid": "c"},                      # not yet judged
    ]}
    assert [e["uid"] for e in chair_manifest.accepted(manifest)] == ["a"]


def test_the_manifest_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = chair_manifest.load()
    manifest["chairs"].append({"uid": "x", "category": "chair"})
    chair_manifest.save(manifest)
    assert chair_manifest.load()["chairs"] == [{"uid": "x", "category": "chair"}]


import build_chair_manifest
import build_table_manifest
from build_chair_manifest import CRITERIA, DEFAULT_THRESHOLDS, curate


def test_fetch_adds_one_entry_per_new_uid_and_skips_known_ones(monkeypatch, tmp_path):
    """Re-running fetch must not re-download. The downloader is injected because a test that
    hits Objaverse would be slow, rate-limited, and would fail offline."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_download(uids, download_processes=1):
        calls.append(list(uids))
        return {uid: f"/fake/{uid}.glb" for uid in uids}

    manifest = {"chairs": [{"uid": "already_here", "category": "chair"}]}
    manifest = build_chair_manifest.fetch(
        manifest, "chair", 3, download=fake_download,
        uids=["already_here", "new_a", "new_b"],
    )

    assert calls == [["new_a", "new_b"]], "fetch re-downloaded a uid it already had"
    assert {e["uid"] for e in manifest["chairs"]} == {"already_here", "new_a", "new_b"}
    for entry in manifest["chairs"]:
        if entry["uid"] != "already_here":
            assert entry["raw_path"].endswith(".glb")
            assert "accepted" not in entry, "fetch must not judge; that is curate's job"


# --- curate --------------------------------------------------------------------------------
#
# Thresholds and fixture uids below are taken from the chair-findings measurement notes,
# "Proposed thresholds" section -- not invented here. Each test comment cites the specific claim
# it exercises.

#: Both places a fetched GLB can be. Task 1 downloaded into objaverse's own default under ~; the
#: pipeline has since redirected the cache onto /lustre (_redirect_objaverse_cache), so the fifty-
#: chair pool -- and every uid the contact sheet hand-labelled -- is only in the second.
OBJAVERSE_CACHE_GLOBS = (
    os.path.join(os.environ.get("OBJAVERSE_PATH", os.path.expanduser("~/.objaverse")),
                 "hf-objaverse-v1/glbs/*/{uid}.glb"),
    os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs/*/{uid}.glb"),
    os.path.join(chair_manifest.root(), "..", "objaverse_cache/hf-objaverse-v1/glbs/*/{uid}.glb"),
)


def _cached_glb(uid):
    """Path to a real mesh already downloaded into one of the two caches. Never downloads -- glob
    only. Fails loudly (not a skip) if the cache is gone, per the task instruction not to let a
    missing cache quietly remove the only real-mesh coverage these tests have."""
    matches = [m for pattern in OBJAVERSE_CACHE_GLOBS
               for m in glob.glob(pattern.format(uid=uid))]
    assert matches, (
        f"{uid}.glb not found in any objaverse cache ({', '.join(OBJAVERSE_CACHE_GLOBS)}) -- this "
        "uid is named as a fixture by the chair findings docs. If the cache has been cleared, "
        "these real-mesh curate tests cannot run honestly and must not be silently skipped; "
        "restore the cache (objaverse.load_objects([uid])) or update the fixture uid."
    )
    return matches[0]


@pytest.fixture
def chairish_thresholds():
    """Findings doc, "Proposed thresholds":
    1. aspect ratio bounds [0.20, 0.95] -- observed range across 40 real chairs was
       [0.308, 0.811] (width) / [0.350, 0.788] (depth); these bounds sit outside that whole
       cluster with margin.
    2. face count [100, 200_000] -- floor 100 sits just below the lowest observed working face
       count (160); ceiling 200_000 is a compute-cost guard, not a validated quality signal.

    These numbers now live in exactly one place -- build_chair_manifest.DEFAULT_THRESHOLDS -- so
    this fixture imports rather than re-transcribes them; a hand-copied dict here could silently
    drift from what curate's own default (and the CLI's) actually use. A copy is returned (not
    the module dict itself) so no test can mutate the shared DEFAULT_THRESHOLDS by mutating what
    it thought was its own fixture.
    """
    return dict(DEFAULT_THRESHOLDS)


def _chair_like_scene_path(tmp_path):
    """A 10-box stretcher-chair frame (4 legs, 4 stretchers, a seat, a backrest), exported as a
    multi-geometry GLB so trimesh.load() returns a Scene, not a single Trimesh -- exercising the
    findings doc's proposed threshold #3 ("multi-body handling: merge, not reject"; 5/40 real
    chairs were multi-body and 4/5 were good chairs after merging). 10 boxes x 12 faces = 120
    faces, comfortably inside [100, 200_000]; merged extents come out (0.8, 1.5, 0.8) -- both
    non-up-axis normalized extents are 0.533, comfortably inside [0.20, 0.95].

    That 0.533 used to be 0.333 (a 0.5-wide seat under the same 1.5-tall frame), which the
    post-scale width criterion added on 2026-08-13 correctly rejects: a 3:1 tall-to-wide chair
    exports to a 0.28 m footprint at a nominal height, narrower than every one of the 38 real
    accepted chairs and narrower than the two the criterion throws out. The fixture was describing
    an impossible chair -- the real corpus's normalized larger-horizontal extents run 0.345 to
    0.892 -- so it was the fixture that was wrong, not the criterion. 0.533 sits inside that
    measured range and clears the 0.35 m bound at every height in the [0.75, 1.0] clip band.
    """
    scene = trimesh.Scene()
    for x, z in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]:
        leg = trimesh.creation.box(extents=(0.05, 0.9, 0.05))
        scene.add_geometry(leg, transform=trimesh.transformations.translation_matrix((x, 0.45, z)))
    for x, z, ex, ez in [(-0.35, 0, 0.05, 0.7), (0.35, 0, 0.05, 0.7), (0, -0.35, 0.7, 0.05), (0, 0.35, 0.7, 0.05)]:
        stretcher = trimesh.creation.box(extents=(ex, 0.05, ez))
        scene.add_geometry(stretcher, transform=trimesh.transformations.translation_matrix((x, 0.2, z)))
    seat = trimesh.creation.box(extents=(0.8, 0.05, 0.8))
    scene.add_geometry(seat, transform=trimesh.transformations.translation_matrix((0, 0.9, 0)))
    backrest = trimesh.creation.box(extents=(0.8, 0.6, 0.05))
    scene.add_geometry(backrest, transform=trimesh.transformations.translation_matrix((0, 1.2, -0.375)))

    path = tmp_path / "chair_like.glb"
    scene.export(path)
    return str(path)


def test_a_pole_is_rejected_for_aspect_ratio(chairish_thresholds, tmp_path):
    """Findings doc #1: "this plan's own pole fixture, box(0.1, 0.1, 2.0), normalizes to
    (0.05, 0.05, 1.0) and is correctly rejected by the low bound.\""""
    pole = tmp_path / "pole.obj"
    trimesh.creation.box(extents=(0.1, 0.1, 2.0)).export(pole)

    manifest = curate({"chairs": [{"uid": "pole", "raw_path": str(pole)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is False
    assert "aspect" in entry["rejected_because"], (
        f"rejected, but for the wrong reason: {entry['rejected_because']}"
    )


def test_a_near_cubic_slab_is_rejected_for_aspect_ratio(chairish_thresholds, tmp_path):
    """Findings doc #1: "a flat/near-cubic slab is caught by the high bound." A 1x1x1 cube
    normalizes to (1.0, 1.0, 1.0) -- both non-up-axis extents (1.0) exceed the 0.95 ceiling."""
    cube = tmp_path / "cube.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(cube)

    manifest = curate({"chairs": [{"uid": "cube", "raw_path": str(cube)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is False
    assert "aspect" in entry["rejected_because"], (
        f"rejected, but for the wrong reason: {entry['rejected_because']}"
    )


def test_a_12_face_box_is_rejected_for_face_count(chairish_thresholds, tmp_path):
    """Findings doc #2: floor is 100 faces. A bare trimesh box has 12 (chair-proportioned, so
    aspect ratio passes -- this isolates the face-count floor from the aspect criterion)."""
    box = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.5, 1.0, 0.5)).export(box)

    manifest = curate({"chairs": [{"uid": "box12", "raw_path": str(box)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is False
    assert "face" in entry["rejected_because"], (
        f"rejected, but for the wrong reason: {entry['rejected_because']}"
    )
    assert entry["faces"] == 12


def test_a_chair_proportioned_multibody_stack_is_accepted(chairish_thresholds, tmp_path):
    """Accept case for BOTH criteria at once, and for the merge-not-reject handling of
    multi-body scenes (findings doc #3)."""
    manifest = curate(
        {"chairs": [{"uid": "stack", "raw_path": _chair_like_scene_path(tmp_path)}]},
        thresholds=chairish_thresholds,
    )
    entry = manifest["chairs"][0]

    assert entry["accepted"] is True
    assert "rejected_because" not in entry
    assert entry["faces"] == 120


def test_curate_records_measurements_for_re_deciding_without_reloading(chairish_thresholds, tmp_path):
    """curate must record extents, normalized_extents, faces, up_axis, facing_axis on every
    judged entry -- accepted or not -- so a later threshold change can re-decide from the
    manifest alone, without re-loading any mesh."""
    pole = tmp_path / "pole.obj"
    trimesh.creation.box(extents=(0.1, 0.1, 2.0)).export(pole)

    manifest = curate({"chairs": [{"uid": "pole", "raw_path": str(pole)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert len(entry["extents"]) == 3
    assert len(entry["normalized_extents"]) == 3
    assert isinstance(entry["faces"], int)
    assert entry["up_axis"] in ("X", "Y", "Z")
    assert entry["facing_axis"] in ("X", "Y", "Z")
    # The distinctness descriptor is recorded for the same reason: without it on the entry, a
    # later duplicate_iou_max change would have to re-load every mesh in the corpus to re-decide.
    assert entry["shape_grid_n"] == build_chair_manifest.SHAPE_GRID_N
    assert isinstance(entry["shape_grid"], str) and entry["shape_grid"]
    grid = build_chair_manifest._decode_grid(entry["shape_grid"], entry["shape_grid_n"])
    assert grid.shape == (entry["shape_grid_n"],) * 3
    assert grid.any(), "an occupancy grid with nothing in it cannot distinguish anything"
    # And the post-scale pair, for the same reason: without them, a later export_width_min_m
    # change could not be re-decided from the manifest, and no human reading an entry could see
    # how wide the chair it describes actually comes out.
    assert entry["export_height_m"] == pytest.approx(_sample_height("pole"))
    assert len(entry["export_footprint_m"]) == 2
    assert entry["export_footprint_m"] == sorted(entry["export_footprint_m"])


def test_criteria_is_a_list_of_separately_callable_predicates():
    """CRITERIA is exported (not just curate) so a later rejection-histogram tool can label
    which criterion produced each rejection without hardcoding the count -- "31 for aspect
    ratio, 12 for face count" instead of "43 rejected" (task brief)."""
    assert len(CRITERIA) == 4
    assert all(callable(predicate) for predicate in CRITERIA)


def test_distinctness_is_the_last_criterion():
    """CRITERIA order is load-bearing, not incidental (see the comment above CRITERIA): a mesh
    that is both a duplicate AND over the face ceiling must be reported as "face count", because
    that reason is a property of the mesh alone and stays true however the corpus changes, while
    "too similar to <uid>" is only true relative to what else was accepted that day. It also keeps
    a candidate that is not a usable chair at all from becoming the incumbent that shadows a later
    one."""
    assert CRITERIA[-1] is build_chair_manifest._distinctness
    # _export_width is per-mesh, so it goes among the per-mesh criteria and before the pairwise
    # one -- for the same reason face count does: a chair that will not ship at a sittable size
    # must never become the incumbent that shadows a later, better one. It goes LAST among the
    # per-mesh three so no pre-existing rejection changes bucket: the pole fixture fails both
    # aspect ratio and width, and "aspect ratio" is the better reason -- it says the thing is not
    # a chair at all, rather than that it is a small one.
    assert CRITERIA.index(build_chair_manifest._export_width) == 2


def test_curate_records_thresholds_into_manifest_provenance(chairish_thresholds, tmp_path):
    """"Record the thresholds used into the manifest's provenance, so a manifest carries the
    criteria that produced it" (task brief)."""
    manifest = curate({"chairs": []}, thresholds=chairish_thresholds)
    assert manifest["thresholds"] == chairish_thresholds


def test_curate_defaults_to_default_thresholds_when_none_are_given():
    """The CLI's `curate` subcommand calls curate(manifest) with no thresholds argument at all --
    it must fall back to DEFAULT_THRESHOLDS rather than raising, and record exactly that dict
    into provenance (not some CLI-local re-transcription of the same four numbers)."""
    manifest = curate({"chairs": []})
    assert manifest["thresholds"] == DEFAULT_THRESHOLDS


def test_curate_skips_entries_without_a_raw_path(chairish_thresholds):
    """A candidate fetch hasn't downloaded yet has no raw_path; curate must not choke on it."""
    manifest = curate({"chairs": [{"uid": "no_path_yet", "category": "chair"}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]
    assert "accepted" not in entry


def test_curate_does_not_re_judge_an_entry_that_already_has_a_verdict(chairish_thresholds):
    """Idempotent: an entry curate already judged (accepted True/False) is left alone on a
    second run, even if raw_path no longer resolves -- otherwise a re-run with a pruned/moved
    cache would crash re-judging work that's already done."""
    manifest = {"chairs": [{
        "uid": "already_judged", "raw_path": "/does/not/exist.glb", "accepted": True,
    }]}
    curate(manifest, thresholds=chairish_thresholds)
    assert manifest["chairs"][0] == {
        "uid": "already_judged", "raw_path": "/does/not/exist.glb", "accepted": True,
    }


def test_a_real_accepted_chair_mesh_is_accepted(chairish_thresholds):
    """Findings doc #5, real fixture uid: 0144133a874d4fa9881d8947928999e0 -- "clean
    single-geometry armchair, watertight, norm_width/depth 0.433/0.433, 16028 faces --
    comfortably inside every proposed bound.\""""
    uid = "0144133a874d4fa9881d8947928999e0"
    manifest = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is True
    assert "rejected_because" not in entry
    assert entry["faces"] == 16028
    assert entry["up_axis"] == "Y"


def test_a_real_mesh_failing_two_criteria_is_rejected_by_the_first_of_them(chairish_thresholds):
    """Findings doc #5, real fixture uid: 0d3608b47df342dda0ae6549e55b79a0 -- "fails up-axis match
    AND the face-count ceiling; also not watertight." Both signals are real and the verdict is the
    same either way; which one fires first moved on 2026-08-13 and the move is the point.

    It used to be face count, because up_axis was argmax(extents): the up axis WAS the longest axis
    by construction, so normalized_extents[up] was identically 1.0, every horizontal was <= 1.0,
    and _aspect_ratio was structurally incapable of observing "this mesh is longer than it is
    tall". Since _up_axis trusts the file's declared glTF Y-up, this mesh's Z extent normalises to
    1.0 against a shorter Y and the aspect band rejects it first -- which is the findings doc's
    OWN first-named reason, "fails up-axis match", finally expressed as a verdict instead of a
    recorded field a human had to notice.

    Mutation: reverting _up_axis to argmax(extents) sends the reason back to face count and reddens
    the aspect assert, while leaving accepted False -- which is why this asserts the reason and not
    just the verdict.
    """
    uid = "0d3608b47df342dda0ae6549e55b79a0"
    manifest = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is False
    assert "aspect ratio" in entry["rejected_because"], (
        f"rejected, but for the wrong reason: {entry['rejected_because']}"
    )
    assert entry["faces"] == 782996
    assert entry["up_axis"] == "Y" and entry["up_axis_source"] == "gltf-y-up"


def test_the_known_geometry_blind_spot_is_accepted_not_rejected(chairish_thresholds):
    """Findings doc's headline finding and this task's explicit instruction: uid
    0730fabb2c8341aaaf303351f2d644c7 is a pedestal-base office/swivel chair that a human labeled
    a REJECT, but it "passes every measured geometric criterion" -- aspect ratio, face count,
    up-axis, facing, watertight. "No geometry-only criterion in this plan will catch this
    rejection." This test documents that curate honestly accepts it -- a real false accept, not
    a bug to be papered over with a contorted threshold. See the module docstring for how this
    blind spot is handled: by recording enough measurements that a human reviewing the final
    accepted set can still catch it, not by tuning a criterion to exclude one mesh.
    """
    uid = "0730fabb2c8341aaaf303351f2d644c7"
    manifest = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)}]},
                      thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is True, (
        "if this now fails, a criterion was tuned to catch this specific mesh -- see the task "
        "brief: 'Do not try to [close it with a threshold].' Revert that criterion change."
    )


# --- distinctness ------------------------------------------------------------------------------
#
# Fixture uids and every number below come from the chair-distinctness measurement notes --
# measured over all 703 pairs of the 38 accepted chairs before the threshold was written, not
# invented here. Both pairs are
# real meshes already cached under ~/.objaverse; nothing here downloads.

#: The measured exact-duplicate pair: the SAME MESH uploaded to Objaverse twice under two uids.
#: Findings doc, "Result 2 -- what does separate cleanly: identical re-uploads": identical SHA-256
#: of canonical vertices, identical 6764 face count, identical 164232-byte GLBs whose md5s differ,
#: shape IoU 1.0000, indistinguishable when rendered side by side.
DUPLICATE_PAIR = ("053039b95c314104b5fe0dfb19f43cb3", "fab9443d48e24fbfa309187df78d58e6")

#: The measured CLOSEST NON-IDENTICAL pair in the whole corpus: shape IoU 0.9531 at N=12 (findings
#: doc, "Result 2", top-of-ranking table -- rank 3 of 703, described there as "same design family,
#: different back and arms"). Deliberately the tightest available keep case: any threshold that
#: rejects these two rejects a genuinely different chair, and 0.98 is the number that does not.
CLOSEST_DISTINCT_PAIR = ("19b2c1893f084653ab226cc3faf27c96", "9173450a6303469fa0fa0c5fbfd9fdac")


def _curated(uids, thresholds):
    """curate() over real cached meshes for `uids`, returned as {uid: entry}."""
    manifest = curate(
        {"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)} for uid in uids]},
        thresholds=thresholds,
    )
    return {entry["uid"]: entry for entry in manifest["chairs"]}


def test_a_measured_identical_pair_is_rejected_as_a_duplicate(chairish_thresholds):
    """The criterion's whole job, on the real pair it was measured from. Their shape IoU is
    1.0000 -- they are byte-identical geometry -- against a 0.98 bound.

    Also asserts the SURVIVOR and the REASON TEXT, not just "one of them was rejected":
    "too similar to <uid>" is actionable and "rejected" is not (task brief), and which one
    survives is the tie-break rule curate's docstring documents -- uid order, so 053039... is
    judged first and keeps its place.

    Reddens if `duplicate_iou_max` is raised above 1.0, or if _distinctness is dropped from
    CRITERIA: either way both meshes are accepted and the library ships the same chair twice.
    """
    first, second = DUPLICATE_PAIR
    entries = _curated(DUPLICATE_PAIR, chairish_thresholds)

    assert entries[first]["accepted"] is True, "the lexicographically first uid must survive"
    assert entries[second]["accepted"] is False
    assert entries[second]["rejected_because"] == (
        f"distinctness: too similar to {first} (shape IoU 1.0000 above 0.98)"
    ), entries[second]["rejected_because"]
    assert "rejected_because" not in entries[first]


def test_the_closest_measured_non_identical_pair_is_kept(chairish_thresholds):
    """The other side of the bound, on the tightest real case the corpus contains: shape IoU
    0.9531, the highest score of any non-identical pair at the shipped grid resolution.

    This is the test that stops the threshold being quietly tuned downward to make a chair count
    come out right. The findings doc is explicit that below 1.0 the distribution is a continuum
    with no gap in it, so any bound under ~0.97 is a number chosen rather than measured.

    Reddens if `duplicate_iou_max` is lowered to 0.9531 or below -- e.g. to somewhere in the
    0.6-0.95 range that would "catch the pod chairs" -- which rejects a visibly different chair.
    """
    first, second = CLOSEST_DISTINCT_PAIR
    entries = _curated(CLOSEST_DISTINCT_PAIR, chairish_thresholds)

    for uid in CLOSEST_DISTINCT_PAIR:
        assert entries[uid]["accepted"] is True, entries[uid].get("rejected_because")

    measured = build_chair_manifest._shape_iou(
        entries[first]["shape_grid"], entries[second]["shape_grid"],
        entries[first]["shape_grid_n"],
    )
    assert measured == pytest.approx(0.9531, abs=5e-4), (
        f"findings doc records 0.9531 for this pair at N={build_chair_manifest.SHAPE_GRID_N}; "
        f"measured {measured:.4f}. The descriptor changed -- re-measure before trusting the "
        "threshold, which was set against that measurement."
    )
    assert chairish_thresholds["duplicate_iou_max"] > measured


def test_distinctness_verdicts_do_not_depend_on_manifest_order(chairish_thresholds):
    """A pairwise criterion's verdict depends on what else is accepted, so the order of judging
    decides which chair survives. curate judges in uid order precisely so that order is not the
    manifest's list order -- which is FETCH order, i.e. however Objaverse happened to return the
    LVIS uid list and however many batches have been run.

    Reddens if curate iterates manifest["chairs"] in list order instead of sorted uid order: the
    reversed manifest then keeps fab9443d... and rejects 053039b9..., so the same four candidates
    fetched in a different order would ship a different library.
    """
    uids = list(DUPLICATE_PAIR) + list(CLOSEST_DISTINCT_PAIR)
    orders = [uids, list(reversed(uids)), [uids[2], uids[0], uids[3], uids[1]]]

    verdicts = []
    for order in orders:
        entries = _curated(order, chairish_thresholds)
        verdicts.append({uid: (entries[uid]["accepted"], entries[uid].get("rejected_because"))
                         for uid in uids})

    assert verdicts[0] == verdicts[1] == verdicts[2], verdicts
    # and the surviving member is the one uid order picks, not the one that happened to be listed
    # first -- the reversed order lists fab9443d... before 053039b9...
    assert verdicts[1][DUPLICATE_PAIR[0]][0] is True
    assert verdicts[1][DUPLICATE_PAIR[1]][0] is False


def test_the_manifests_list_order_is_left_alone(chairish_thresholds):
    """curate judges in uid order but must not REORDER the manifest: entries are mutated in place,
    so anything downstream holding an index into manifest["chairs"] is undisturbed.

    Reddens if curate assigns the sorted list back to manifest["chairs"].
    """
    uids = list(reversed(DUPLICATE_PAIR))
    manifest = curate(
        {"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)} for uid in uids]},
        thresholds=chairish_thresholds,
    )
    assert [entry["uid"] for entry in manifest["chairs"]] == uids


def test_an_entry_rejected_by_an_earlier_criterion_never_shadows_a_duplicate(
    chairish_thresholds, tmp_path
):
    """Rule 2 of curate's ordering: only ACCEPTED entries are incumbents. A candidate thrown out
    for face count must not take an identical later candidate down with it -- and since both are
    unusable here for the same per-mesh reason, both must say so.

    Two identical 12-face boxes: chair-proportioned (aspect passes) and below the 100-face floor,
    so the first fails face count and never becomes an incumbent. The second is then judged on its
    own merits and fails face count too -- NOT "too similar to box_a".

    Reddens two ways: if curate appends rejected entries to its incumbent list, or if
    _distinctness is moved ahead of _face_count in CRITERIA. Either way box_b is reported as a
    duplicate of a chair that is not in the library, which is a lie the histogram would repeat.
    """
    paths = {}
    for uid in ("box_a", "box_b"):
        path = tmp_path / f"{uid}.obj"
        trimesh.creation.box(extents=(0.5, 1.0, 0.5)).export(path)
        paths[uid] = str(path)

    manifest = curate({"chairs": [{"uid": uid, "raw_path": p} for uid, p in paths.items()]},
                      thresholds=chairish_thresholds)

    for entry in manifest["chairs"]:
        assert entry["accepted"] is False
        assert entry["rejected_because"].startswith("face count"), entry["rejected_because"]


def test_a_duplicate_of_an_incumbent_from_a_previous_run_is_still_caught(chairish_thresholds):
    """The incumbent set is not only what this call accepted -- it is every entry already carrying
    accepted: True, including one a previous curate run judged and that export/convert may already
    have written files for. That is rule 4: a chair already on disk always wins, whatever its uid.

    Here the survivor is fab9443d..., which uid order would NOT have picked -- proving the
    incumbent came from the manifest rather than from this call's own sorting.

    Reddens if the incumbent list is built only from entries accepted during this call.
    """
    first, second = DUPLICATE_PAIR
    manifest = curate({"chairs": [
        {"uid": second, "raw_path": _cached_glb(second), "accepted": True,
         "shape_grid_n": build_chair_manifest.SHAPE_GRID_N,
         "shape_grid": build_chair_manifest._measure(_cached_glb(second))["shape_grid"]},
        {"uid": first, "raw_path": _cached_glb(first)},
    ]}, thresholds=chairish_thresholds)
    entries = {entry["uid"]: entry for entry in manifest["chairs"]}

    assert entries[second]["accepted"] is True
    assert entries[first]["accepted"] is False
    assert entries[first]["rejected_because"] == (
        f"distinctness: too similar to {second} (shape IoU 1.0000 above 0.98)"
    )


def test_an_incumbent_without_a_descriptor_fails_open_rather_than_guessing(chairish_thresholds):
    """A manifest curated before this criterion existed has entries with no shape_grid at all, and
    one curated under a different SHAPE_GRID_N has grids that cannot be compared with today's.
    _distinctness skips those incumbents rather than comparing at a mismatched resolution -- it
    fails OPEN, accepting a duplicate rather than inventing a similarity number.

    The honest consequence, recorded so nobody is surprised by it: re-curating an old manifest in
    place does not de-duplicate it. Clearing the verdicts (or curating a fresh manifest) does.

    Reddens if _distinctness compares grids of differing shape_grid_n, which raises or -- worse --
    silently reshapes one of them.
    """
    first, second = DUPLICATE_PAIR
    stale = {"uid": second, "accepted": True}                       # pre-criterion entry
    wrong_n = {"uid": "other", "accepted": True, "shape_grid_n": 8, "shape_grid": "AAAA"}
    measurements = build_chair_manifest._measure(_cached_glb(first))

    assert build_chair_manifest._distinctness(
        measurements, chairish_thresholds, [stale, wrong_n]) is None


def test_the_shape_descriptor_is_reproducible_across_separate_interpreter_processes():
    """The descriptor is recorded into a manifest, so it must not depend on process state. It uses
    trimesh's subdivide_to_size rather than surface sampling for exactly this reason (see
    _shape_grid): a seeded sampler would put a per-process number into a durable artefact, the
    same trap _sample_height's docstring describes.

    Mirrors test_sample_height_is_stable_across_separate_interpreter_processes: real separate
    `python -c` invocations with PYTHONHASHSEED varied, not a same-process loop.

    Reddens if _shape_grid is reimplemented on top of mesh.sample()/np.random without a seed, or
    with a seed derived from hash().
    """
    uid = DUPLICATE_PAIR[0]
    script = (
        "import sys, hashlib;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'tools')!r});"
        "import build_chair_manifest as B;"
        f"print(hashlib.sha256(B._measure({_cached_glb(uid)!r})['shape_grid'].encode()).hexdigest())"
    )
    digests = set()
    for seed in ("0", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        digests.add(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   check=True, env=env).stdout.strip())
    assert len(digests) == 1, f"descriptor differed across processes: {digests}"


# --- post-scale export width --------------------------------------------------------------------
#
# Fixture uids and every number below come from the chair-export-width measurement notes -- the
# exported horizontal extents of all 38 accepted chairs, predicted analytically and cross-checked
# against the 38 OBJs
# on disk to 9.5e-09 m, measured before the bound was written. Both fixtures are real meshes
# already cached under ~/.objaverse; nothing here downloads.
#
# This is the one criterion that judges a chair AFTER scaling. It is still a curate criterion,
# because the scale is known in closed form at curate time -- which is exactly what
# test_the_predicted_export_footprint_is_what_export_actually_writes below checks.

#: The TIGHTEST REJECT case in the corpus: exported footprint 0.334 x 0.285 m, the wider of its two
#: horizontal extents just 0.016 m below the 0.35 m bound (findings doc, "Result 1 -- the
#: distribution"). A tall ladder-back side chair -- a correct model whose proportions make it doll
#: furniture once export normalises it to a life-sized height. It is one of the offered twenty, so
#: this criterion genuinely removes a chair that shipped.
TOO_NARROW_UID = "99ef536b8f174b9496516bed69368f46"

#: The TIGHTEST KEEP case: exported footprint 0.374 x 0.359 m, 0.024 m above the bound and the
#: narrowest chair of the 38 that the criterion keeps. Rendered (findings doc, "Result 2") it is an
#: ordinary oval-back armchair with arms and cabriole legs. Deliberately the tightest available keep
#: case, exactly as CLOSEST_DISTINCT_PAIR is for distinctness: any bound that rejects this rejects a
#: real chair.
NARROWEST_KEPT_UID = "d2d5000b1e7a4d9a8a4a637267418061"


def test_a_measured_too_narrow_chair_is_rejected_for_export_width(chairish_thresholds):
    """The criterion's whole job, on the narrowest chair that survived every pre-existing
    criterion. Asserts the REASON TEXT too, not just that it was rejected: the reason has to carry
    the footprint and the height it was judged at, or a human reading the histogram cannot tell a
    chair that is 1 mm under the bound from one that is 50 mm under.

    Note which criterion fires: aspect ratio does NOT catch this chair (its normalized extents are
    0.325/0.380, well inside [0.20, 0.95]) and neither does face count (424, well inside
    [100, 200_000]). Proportion is scale-free, which is the whole reason a post-scale criterion had
    to exist.

    Reddens if `export_width_min_m` is lowered to 0.334 or below, or if _export_width is dropped
    from CRITERIA: either way the library ships a chair 0.334 m across at its widest.
    """
    entries = _curated([TOO_NARROW_UID], chairish_thresholds)
    entry = entries[TOO_NARROW_UID]

    assert entry["accepted"] is False
    assert entry["rejected_because"] == (
        "export width: 0.334 m x 0.285 m footprint at its 0.878 m export height -- "
        "larger horizontal extent below 0.35 m"
    ), entry["rejected_because"]
    # The measurements the verdict rests on, recorded whatever the verdict -- and matching the
    # findings doc, so a descriptor change cannot silently invalidate the threshold.
    assert entry["export_footprint_m"] == pytest.approx([0.285, 0.334], abs=5e-4)
    assert build_chair_manifest.rejection_histogram({"chairs": [entry]}) == {"export width": 1}


def test_the_narrowest_measured_chair_that_still_ships_is_kept(chairish_thresholds):
    """The other side of the bound, on the tightest real case the corpus contains: 0.374 m, the
    narrowest of the 36 kept.

    This is the test that stops the bound being quietly raised to the 0.40-0.55 m folk figure for
    real dining chairs, which the task brief names as the number NOT to use. 0.40 sits in the
    middle of the dense band and would reject this chair and three others that render as ordinary
    armchairs; the measured empty interval is (0.334, 0.374), and 0.35 is the number inside it.

    Reddens if `export_width_min_m` is raised to 0.375 or above -- e.g. to 0.40.
    """
    entries = _curated([NARROWEST_KEPT_UID], chairish_thresholds)
    entry = entries[NARROWEST_KEPT_UID]

    assert entry["accepted"] is True, entry.get("rejected_because")
    assert "rejected_because" not in entry
    assert entry["export_footprint_m"] == pytest.approx([0.359, 0.374], abs=5e-4), (
        f"findings doc records a 0.374 x 0.359 m footprint for this chair; measured "
        f"{entry['export_footprint_m']}. Re-measure before trusting the bound, which was set "
        "against that measurement."
    )
    assert max(entry["export_footprint_m"]) > chairish_thresholds["export_width_min_m"]


def test_the_predicted_export_footprint_is_what_export_actually_writes(
    chairish_thresholds, tmp_path, monkeypatch,
):
    """The load-bearing claim that lets a POST-SCALE criterion live in curate at all: curate can
    know the exported footprint without exporting anything, because export scales uniformly to a
    height drawn from the uid and _stand_up's rotation is cardinal, so exported horizontal extent
    is exactly normalized_extents[i] x _sample_height(uid).

    Checked end to end on a real cached mesh -- curate predicts, export writes an OBJ, the OBJ is
    loaded back and measured. Verified across all 38 accepted chairs at 9.5e-09 m (findings doc);
    1e-6 m here leaves room for OBJ's text precision.

    Reddens if export stops scaling uniformly (e.g. per-axis normalisation), if _stand_up gains a
    non-cardinal rotation, or if _export_footprint and export ever stop sharing _sample_height as
    their one source of truth -- any of which would make every width verdict a guess about a chair
    that ships at some other size.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    uid = NARROWEST_KEPT_UID

    manifest = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)}]},
                      thresholds=chairish_thresholds)
    predicted = manifest["chairs"][0]["export_footprint_m"]
    predicted_height = manifest["chairs"][0]["export_height_m"]

    manifest = build_chair_manifest.export(manifest)
    out = trimesh.load(manifest["chairs"][0]["obj_path"], process=False, force="mesh")
    actual = sorted(float(v) for v in out.bounding_box.extents[:2])

    assert actual == pytest.approx(predicted, abs=1e-6), (
        f"curate predicted a {predicted} m footprint; export actually wrote {actual}"
    )
    assert float(out.bounding_box.extents[2]) == pytest.approx(predicted_height, abs=1e-6)
    assert manifest["chairs"][0]["height_m"] == pytest.approx(predicted_height, abs=1e-6)


def test_export_width_judges_the_chairs_own_height_draw_not_the_nominal_height(
    chairish_thresholds, tmp_path,
):
    """The structural decision, made explicit: export height is a uid-seeded Gaussian clipped to
    [0.75, 1.0], so two chairs of IDENTICAL proportion ship at different sizes and the criterion
    must judge the size each one actually ships at.

    Both fixtures below are the same chair-proportioned shape; only their uids differ, and the uids
    were chosen for their draws (findings doc, "The structural question"):

      * width-fixture-151 draws 0.7552 m. At 0.44 normalized it exports 0.332 m -- REJECT.
        Judged at the nominal 0.85 m it would read 0.374 m and be kept, and a chair 0.332 m across
        would ship.
      * width-fixture-6 draws 0.9814 m. At 0.39 normalized it exports 0.383 m -- KEEP.
        Judged at the nominal 0.85 m it would read 0.332 m and be thrown away for a size it never
        has.

    So the two rules disagree in BOTH directions, which is why the choice is not cosmetic. On the
    38 real accepted chairs they happen to agree -- that is measured in the findings doc, and is
    why this test is built rather than drawn from the corpus.

    This is not non-determinism: one uid has exactly one draw, in every process (see
    test_sample_height_is_stable_across_separate_interpreter_processes). It is a coupling, and the
    findings doc records it: changing TARGET_HEIGHT_M, HEIGHT_SIGMA_M or the clip band re-decides
    every width verdict.

    Curated one at a time on purpose: the two fixtures differ only in proportion, so curating them
    together would let _distinctness weigh in on a test about width.

    Reddens if _export_footprint substitutes TARGET_HEIGHT_M for _sample_height(uid).
    """
    def verdict(uid, normalized_width):
        path = tmp_path / f"{uid}.obj"
        mesh = trimesh.creation.box(extents=(normalized_width, 1.0, normalized_width))
        mesh.subdivide().subdivide().export(path)          # 192 faces, clears the 100 floor
        manifest = curate({"chairs": [{"uid": uid, "raw_path": str(path)}]},
                          thresholds=chairish_thresholds)
        return manifest["chairs"][0]

    low_draw = verdict("width-fixture-151", 0.44)
    assert low_draw["export_height_m"] == pytest.approx(0.7552, abs=5e-4)
    assert max(low_draw["export_footprint_m"]) == pytest.approx(0.332, abs=5e-4)
    assert low_draw["accepted"] is False, (
        "a chair that ships 0.332 m across was kept -- the criterion is reading the nominal "
        "height, not this chair's own draw"
    )
    assert low_draw["rejected_because"].startswith("export width")
    assert 0.44 * build_chair_manifest.TARGET_HEIGHT_M > (
        chairish_thresholds["export_width_min_m"]
    ), "fixture no longer distinguishes the two rules; re-pick its normalized width"

    high_draw = verdict("width-fixture-6", 0.39)
    assert high_draw["export_height_m"] == pytest.approx(0.9814, abs=5e-4)
    assert max(high_draw["export_footprint_m"]) == pytest.approx(0.383, abs=5e-4)
    assert high_draw["accepted"] is True, high_draw.get("rejected_because")
    assert 0.39 * build_chair_manifest.TARGET_HEIGHT_M < (
        chairish_thresholds["export_width_min_m"]
    ), "fixture no longer distinguishes the two rules; re-pick its normalized width"


def test_export_width_fails_open_when_no_footprint_was_measured(chairish_thresholds):
    """_measure without a uid cannot know the export height, so it records no footprint. The
    criterion then declines to judge rather than substituting a height -- the same fail-open choice
    _distinctness makes for a missing shape descriptor, and stated here so nobody is surprised that
    a manifest curated before this criterion existed does not get re-judged for width in place.

    Reddens if _export_width falls back to TARGET_HEIGHT_M when the footprint is absent.
    """
    measurements = build_chair_manifest._measure(_cached_glb(TOO_NARROW_UID))

    assert "export_footprint_m" not in measurements
    assert build_chair_manifest._export_width(measurements, chairish_thresholds, []) is None
    # ...and with the uid, the same mesh is rejected.
    with_uid = build_chair_manifest._measure(_cached_glb(TOO_NARROW_UID), TOO_NARROW_UID)
    assert build_chair_manifest._export_width(with_uid, chairish_thresholds, []) is not None


# --- export ----------------------------------------------------------------------------------


def test_export_stands_the_chair_up_and_scales_it_to_life_size(tmp_path, monkeypatch):
    """Objaverse carries no units and no reliable up-axis. An exported chair must be upright,
    life-sized, and sitting on z=0 -- the same bottom-alignment the kitchen's own table needed,
    where an origin-centred asset ended up half-buried in the floor, invisible to every test
    because nothing asserted z."""
    import trimesh

    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    # A deliberately lying fixture: chair-proportioned, but Y-up and 40 units tall.
    raw = trimesh.creation.box(extents=(16.0, 40.0, 16.0))
    raw_path = tmp_path / "liar.obj"
    raw.export(raw_path)

    manifest = {"chairs": [{
        "uid": "liar", "category": "chair", "raw_path": str(raw_path),
        "accepted": True, "up_axis": 1,          # Y-up, as curate measured it
    }]}

    manifest = build_chair_manifest.export(manifest)
    entry = manifest["chairs"][0]
    out = trimesh.load(entry["obj_path"])

    assert 0.75 <= out.extents[2] <= 1.0, f"height {out.extents[2]} is not chair-sized"
    assert out.extents[2] == max(out.extents), "the chair is not standing up"
    assert out.bounds[0][2] == pytest.approx(0.0, abs=1e-6), (
        f"underside sits at z={out.bounds[0][2]}, not on the floor"
    )
    assert entry["scale_applied"] > 0 and entry["height_m"] == pytest.approx(out.extents[2])


def test_export_stands_a_real_cached_chair_up_and_scales_it_to_life_size(
    chairish_thresholds, tmp_path, monkeypatch,
):
    """A synthetic box stack can prove the arithmetic works but cannot prove export correctly
    consumes curate's REAL up-axis measurement. uid 0144133a874d4fa9881d8947928999e0 is the
    findings doc's clean single-geometry armchair fixture (Y-up, watertight, 16028 faces) --
    curate measures it for real, reading only the ~/.objaverse cache (no network), and export
    must stand it up, scale it, and bottom-align it exactly as the synthetic case above."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    uid = "0144133a874d4fa9881d8947928999e0"

    manifest = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid)}]},
                      thresholds=chairish_thresholds)
    assert manifest["chairs"][0]["accepted"] is True
    raw_extents = manifest["chairs"][0]["extents"]
    raw_up_axis = manifest["chairs"][0]["up_axis"]

    manifest = build_chair_manifest.export(manifest)
    entry = manifest["chairs"][0]
    out = trimesh.load(entry["obj_path"])

    print(f"\nreal mesh {uid}: raw extents {raw_extents} up_axis={raw_up_axis} "
          f"-> out extents {list(out.extents)} bounds[0][2]={out.bounds[0][2]}")

    assert 0.75 <= out.extents[2] <= 1.0, f"height {out.extents[2]} is not chair-sized"
    assert out.extents[2] == max(out.extents), "the chair is not standing up"
    assert out.bounds[0][2] == pytest.approx(0.0, abs=1e-6), (
        f"underside sits at z={out.bounds[0][2]}, not on the floor"
    )
    assert entry["scale_applied"] > 0
    assert entry["height_m"] == pytest.approx(out.extents[2])
    assert entry["facing_axis"] in ("X", "Y", "Z")


# --- export: material handling (bugfix, 2026-08-11 findings) -----------------------------
#
# Real finding: trimesh's OBJ exporter writes materials to a FIXED filename, material.mtl,
# regardless of which mesh is being exported. Every accepted chair shares the same obj_dir, so
# 38 accepted chairs meant 38 clobbering writes to the one obj_dir/material.mtl -- confirmed on
# disk: 38 .obj files, exactly 1 .mtl, holding only the LAST chair's material. A plain
# trimesh.creation.box() (ColorVisuals, no material) never triggers this at all -- these tests
# build a mesh with an actual TextureVisuals material attached, matching what a real Objaverse
# GLB carries, so the fixture can actually exercise (and catch a regression of) the fix.


def _liar_box_with_material():
    """Same lying fixture as test_export_stands_the_chair_up_and_scales_it_to_life_size (Y-up,
    40 units tall, chair-proportioned) but with a real material attached, so exporting it can
    actually prove whether a .mtl gets written."""
    mesh = trimesh.creation.box(extents=(16.0, 40.0, 16.0))
    material = trimesh.visual.material.SimpleMaterial(diffuse=[200, 50, 50, 255])
    mesh.visual = trimesh.visual.texture.TextureVisuals(material=material)
    return mesh


def test_export_writes_no_material_file_even_when_the_source_mesh_has_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    manifest = {"chairs": []}
    for uid in ("liar1", "liar2"):
        raw_path = tmp_path / f"{uid}.obj"
        _liar_box_with_material().export(raw_path)  # a real .mtl-bearing source, on purpose
        manifest["chairs"].append({
            "uid": uid, "category": "chair", "raw_path": str(raw_path),
            "accepted": True, "up_axis": 1,
        })

    manifest = build_chair_manifest.export(manifest)

    obj_dir = os.path.join(str(tmp_path), "obj")
    mtl_files = glob.glob(os.path.join(obj_dir, "*.mtl"))
    assert mtl_files == [], f"export must write no material file at all, found {mtl_files}"

    for entry in manifest["chairs"]:
        out = trimesh.load(entry["obj_path"], process=False, force="mesh")
        assert len(out.faces) > 0, f"{entry['uid']}'s exported OBJ failed to load real geometry"


def test_export_records_the_floor_material_group_decision(tmp_path, monkeypatch):
    """The user's decision -- chairs take the kitchen's own MATERIALS['floor'] pool
    (scripts/simvla/kitchen_build.py) rather than any material of their own -- recorded on the
    manifest so a later placement stage does not have to rediscover it. Set even when there are
    no accepted chairs to actually export, same as curate() always records manifest["thresholds"]
    regardless of whether anything got judged."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = build_chair_manifest.export({"chairs": []})
    assert manifest["material_group"] == "floor"


# --- per-chair height sampling ------------------------------------------------------------
#
# export no longer scales every chair to a single fixed TARGET_HEIGHT_M -- it draws a per-chair
# target from a Gaussian centred on TARGET_HEIGHT_M, seeded from the chair's own uid, then clips
# to the same [0.75, 1.0] band the export tests above already assert on. These tests exercise
# the sampler (_sample_height) directly rather than through export: export loads and transforms
# a real mesh, which is slow, and none of these properties depend on mesh content at all.

import subprocess

from build_chair_manifest import HEIGHT_CLIP_MAX_M, HEIGHT_CLIP_MIN_M, _sample_height


def test_sample_height_is_reproducible_for_the_same_uid():
    """The load-bearing property: re-running export must give the same chair the same height,
    or the manifest's recorded height_m is a lie about anything but the last run."""
    assert _sample_height("0144133a874d4fa9881d8947928999e0") == _sample_height(
        "0144133a874d4fa9881d8947928999e0"
    )


def test_sample_height_differs_across_uids():
    """Otherwise the Gaussian is decorative -- every chair would still land on the same height,
    just reached through a more roundabout calculation."""
    heights = {_sample_height(f"uid-{i}") for i in range(10)}
    assert len(heights) > 1, "10 different uids all sampled the same height"


def test_sampled_heights_stay_within_the_clip_band():
    """A Gaussian's tails are unbounded; drive several hundred synthetic uids through the
    sampler (not export -- exporting is slow, and this is a property of the sampler alone) and
    confirm none escape the hard clip band."""
    for i in range(500):
        height = _sample_height(f"synthetic-uid-{i}")
        assert HEIGHT_CLIP_MIN_M <= height <= HEIGHT_CLIP_MAX_M, (
            f"synthetic-uid-{i} sampled {height}, outside [{HEIGHT_CLIP_MIN_M}, {HEIGHT_CLIP_MAX_M}]"
        )


def test_sampled_heights_are_not_all_pinned_to_the_clip_edges():
    """A degenerate sampler could satisfy the band test above by always returning exactly one
    edge of the clip. Confirm the draws actually spread across the interior too."""
    heights = [_sample_height(f"synthetic-uid-{i}") for i in range(500)]
    interior = [h for h in heights if HEIGHT_CLIP_MIN_M < h < HEIGHT_CLIP_MAX_M]
    assert len(interior) > 400, (
        f"only {len(interior)}/500 draws landed strictly inside the clip band -- "
        "sigma may be too large relative to the band"
    )


def test_sample_height_is_stable_across_separate_interpreter_processes():
    """The property that actually matters for reproducibility: not just "same value if called
    twice in this process" (which a process-local counter would also satisfy) but "same value
    in a totally fresh interpreter." Runs two `python -c` subprocesses -- deliberately with
    different PYTHONHASHSEED values, since that is exactly the thing that must NOT affect the
    result (see _sample_height's docstring: builtin hash() of a str, and a tuple seed, are both
    salted by PYTHONHASHSEED and would fail this test)."""
    uid = "0144133a874d4fa9881d8947928999e0"
    tools_dir = str(Path(__file__).resolve().parents[1] / "tools")
    script = (
        f"import sys; sys.path.insert(0, {tools_dir!r}); "
        f"from build_chair_manifest import _sample_height; "
        f"print(repr(_sample_height({uid!r})))"
    )

    results = []
    for hashseed in ("0", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
        )
        results.append(proc.stdout.strip())

    assert results[0] == results[1], (
        f"same uid produced different heights across processes with different PYTHONHASHSEED: "
        f"{results}"
    )
    assert results[0] == repr(_sample_height(uid)), (
        "cross-process height does not even match this in-process one"
    )


# --- convert -------------------------------------------------------------------------------
#
# scripts/tools/convert_mesh.py imports isaaclab.app.AppLauncher at module scope and boots Isaac
# Sim the moment it runs -- it cannot be imported into this pytest process, and a second AppLauncher
# cannot be booted inside a process that hasn't booted one itself. So convert() shells out to it,
# exactly as fetch shells out to (rather than imports) objaverse's own network calls -- and takes
# an injectable `runner` for the same testability reason fetch takes an injectable `download`.
# None of these tests ever import or invoke convert_mesh.py.


def test_convert_invokes_the_runner_once_per_accepted_entry_with_the_right_paths(tmp_path, monkeypatch):
    """The core wiring: one runner call per ACCEPTED entry, input = obj_path, output =
    <CHAIR_OBJ_DIR>/usd/<uid>.usd -- and the entry ends up carrying that usd_path."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_runner(input_path, output_path):
        calls.append((input_path, output_path))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = {"chairs": [
        {"uid": "a", "accepted": True, "obj_path": "/fake/a.obj"},
    ]}
    manifest = build_chair_manifest.convert(manifest, runner=fake_runner)

    expected_usd = os.path.join(str(tmp_path), "usd", "a.usd")
    assert calls == [("/fake/a.obj", expected_usd)]
    assert manifest["chairs"][0]["usd_path"] == expected_usd


def test_convert_skips_an_entry_that_already_carries_a_usd_path(tmp_path, monkeypatch):
    """Re-running convert after convert has already run must not redo work -- mirrors export's
    own idempotence guarantee for obj_path."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_runner(input_path, output_path):
        calls.append((input_path, output_path))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = {"chairs": [{
        "uid": "b", "accepted": True, "obj_path": "/fake/b.obj",
        "usd_path": "/already/b.usd", "hull_count": 5,
    }]}
    manifest = build_chair_manifest.convert(manifest, runner=fake_runner)

    assert calls == [], "convert re-ran an entry that already had a usd_path"
    assert manifest["chairs"][0]["usd_path"] == "/already/b.usd"
    assert manifest["chairs"][0]["hull_count"] == 5


def test_convert_never_touches_a_rejected_or_not_yet_judged_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_runner(input_path, output_path):
        calls.append((input_path, output_path))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = {"chairs": [
        {"uid": "rejected", "accepted": False, "rejected_because": "aspect ratio: ..."},
        {"uid": "unjudged"},
    ]}
    manifest = build_chair_manifest.convert(manifest, runner=fake_runner)

    assert calls == [], "convert must never shell out for a rejected or not-yet-judged entry"
    assert "usd_path" not in manifest["chairs"][0]
    assert "usd_path" not in manifest["chairs"][1]


def test_convert_extracts_a_hull_count_when_the_runner_reports_one(tmp_path, monkeypatch):
    """Best-effort: if convert_mesh.py's stdout ever reports a hull/convex-shape count, convert
    must capture it rather than silently drop it on the floor."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    def fake_runner(input_path, output_path):
        return types.SimpleNamespace(stdout="...\nNumber of convex hulls: 37\n...", returncode=0)

    manifest = {"chairs": [{"uid": "a", "accepted": True, "obj_path": "/fake/a.obj"}]}
    manifest = build_chair_manifest.convert(manifest, runner=fake_runner)

    assert manifest["chairs"][0]["hull_count"] == 37


def test_convert_records_hull_count_as_none_rather_than_a_fabricated_number(tmp_path, monkeypatch):
    """convert_mesh.py, as it stands today, never prints a hull count. Verified by reading
    isaaclab's MeshConverter._convert_asset (source/isaaclab/isaaclab/sim/converters/mesh_converter.py):
    convex decomposition is recorded only as a MeshCollisionAPI 'approximation' ATTRIBUTE on the
    mesh prim; the actual per-hull decomposition is cooked lazily by PhysX the first time a
    RUNNING physics scene touches the collider -- omni.physx's own get_nb_convex_mesh_data()
    binding says outright "Does work only when simulation is running." Reading the produced USD
    back cannot surface a hull count either: nothing about hull count is written to the file at
    conversion time, and even opening the file at all needs the `pxr` module, which is not
    importable in this checkout outside a booted Isaac Sim/Kit process (verified: no standalone
    `pxr` site-package exists; every copy of it lives inside an individual Isaac Sim extension,
    added to sys.path by AppLauncher). So convert must record None here, not a fabricated
    number -- exactly the case the task brief calls out by name."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    def fake_runner(input_path, output_path):
        return types.SimpleNamespace(
            stdout="ordinary convert_mesh.py output, no hull count anywhere", returncode=0,
        )

    manifest = {"chairs": [{"uid": "a", "accepted": True, "obj_path": "/fake/a.obj"}]}
    manifest = build_chair_manifest.convert(manifest, runner=fake_runner)

    assert manifest["chairs"][0]["hull_count"] is None


# --- repair_collisions -----------------------------------------------------------------------
#
# Bugfix (2026-08-11 findings): MeshConverter._convert_asset (vendored, not edited by this fix)
# has a race between writing its converted USD and re-opening that same path, in the same
# process, to author collision on it -- 17/20 real converted chairs came out with no
# UsdPhysics.MeshCollisionAPI at all, and re-converting a chair that had previously succeeded,
# fresh, reproducibly failed 5/5 times. repair_collisions() shells out to
# scripts/tools/repair_chair_collision.py (a NEW tool, not the vendored one) to open each
# ALREADY-WRITTEN usd in a fresh process and author collision directly if it's missing. These
# tests never boot Isaac: `runner` is injected exactly as convert()'s is, and returns fake stdout
# in repair_chair_collision.py's own "RESULT <uid> <True/False>" format for
# _parse_repair_results to parse.


def test_repair_collisions_calls_the_runner_once_with_every_converted_entrys_usd_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_runner(usd_paths):
        calls.append(dict(usd_paths))
        lines = "\n".join(f"RESULT {uid} True" for uid in usd_paths)
        return types.SimpleNamespace(stdout=lines, returncode=0)

    manifest = {"chairs": [
        {"uid": "a", "accepted": True, "obj_path": "/fake/a.obj", "usd_path": "/fake/a.usd"},
        {"uid": "b", "accepted": True, "obj_path": "/fake/b.obj", "usd_path": "/fake/b.usd"},
        {"uid": "c", "accepted": True, "obj_path": "/fake/c.obj"},  # not converted yet
        {"uid": "d", "accepted": False, "rejected_because": "aspect ratio: ..."},
    ]}
    build_chair_manifest.repair_collisions(manifest, runner=fake_runner)

    assert calls == [{"a": "/fake/a.usd", "b": "/fake/b.usd"}], (
        "repair_collisions must call the runner once, with exactly the converted entries' usd_paths"
    )


def test_repair_collisions_records_has_collision_from_parsed_results(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    def fake_runner(usd_paths):
        return types.SimpleNamespace(
            stdout="RESULT a True  # repaired=True\nRESULT b False  # no Mesh prim found\n",
            returncode=0,
        )

    manifest = {"chairs": [
        {"uid": "a", "accepted": True, "obj_path": "/fake/a.obj", "usd_path": "/fake/a.usd"},
        {"uid": "b", "accepted": True, "obj_path": "/fake/b.obj", "usd_path": "/fake/b.usd"},
    ]}
    manifest = build_chair_manifest.repair_collisions(manifest, runner=fake_runner)

    assert manifest["chairs"][0]["has_collision"] is True
    assert manifest["chairs"][1]["has_collision"] is False


def test_repair_collisions_never_calls_the_runner_when_nothing_has_been_converted(tmp_path, monkeypatch):
    """No usd_path anywhere -- e.g. convert() hasn't run yet -- must not shell out at all (and
    must not boot Isaac via the real default runner if this test forgot to inject one)."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_runner(usd_paths):
        calls.append(dict(usd_paths))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = {"chairs": [
        {"uid": "a", "accepted": True, "obj_path": "/fake/a.obj"},
        {"uid": "b", "accepted": False, "rejected_because": "aspect ratio: ..."},
    ]}
    build_chair_manifest.repair_collisions(manifest, runner=fake_runner)

    assert calls == [], "repair_collisions shelled out despite no converted entry existing"


def test_repair_collisions_leaves_has_collision_unset_for_a_uid_missing_from_the_output(tmp_path, monkeypatch):
    """If the subprocess crashed before printing a RESULT line for some uid, repair_collisions
    must not guess True or False on its behalf -- absence of a result is not a verdict."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))

    def fake_runner(usd_paths):
        return types.SimpleNamespace(stdout="RESULT a True\n", returncode=0)  # nothing for "b"

    manifest = {"chairs": [
        {"uid": "a", "accepted": True, "obj_path": "/fake/a.obj", "usd_path": "/fake/a.usd"},
        {"uid": "b", "accepted": True, "obj_path": "/fake/b.obj", "usd_path": "/fake/b.usd"},
    ]}
    manifest = build_chair_manifest.repair_collisions(manifest, runner=fake_runner)

    assert manifest["chairs"][0]["has_collision"] is True
    assert "has_collision" not in manifest["chairs"][1]


# --- the CLI (Part 2: the gap Task 5's brief left) -----------------------------------------
#
# Task 6 needs something to invoke to "run the pipeline to twenty accepted." These tests drive
# main() the same way a shell invocation would -- argv in -- but inject `download`/`uids` (fetch's
# own injection points), `runner` (convert's), and `repair_runner` (repair_collisions') so nothing
# here touches the network or boots Isaac. curate/export need no injection: they already work off
# on-disk mesh files, so the fake `download` below writes REAL trimesh fixtures to tmp_path
# instead of fake path strings, exactly as _chair_like_scene_path does for the curate tests above
# -- that's what lets a full fetch -> curate -> export -> convert -> repair_collisions loop run
# end to end inside a test.


def _fake_repair_runner(calls=None):
    """A repair_runner that reports every uid it's given as already collision-OK, in
    repair_chair_collision.py's own "RESULT <uid> True" stdout format. `calls`, if passed, is
    appended to on every invocation so a test can assert on exactly what it was given."""
    def repair_runner(usd_paths):
        if calls is not None:
            calls.append(dict(usd_paths))
        lines = "\n".join(f"RESULT {uid} True" for uid in usd_paths)
        return types.SimpleNamespace(stdout=lines, returncode=0)
    return repair_runner


def _write_good_chair_mesh(path, uid=""):
    """A chair-proportioned mesh with enough faces to clear every curate criterion. Subdividing a
    box (12 faces) twice quadruples the face count each time (12 -> 48 -> 192), clearing the
    100-face floor, while leaving the box's extents -- and therefore its aspect ratio -- exactly
    as they were: subdivision only adds vertices along existing geometry.

    Every generated chair is then displaced by a per-vertex offset seeded from its own `uid`, so a
    batch of these is a batch of DIFFERENT chairs. Without that they were all byte-identical, and
    the distinctness criterion correctly rejected 29 of 30 as duplicates of each other -- the
    fixture, not the criterion, was describing an impossible corpus (one Objaverse category
    returning the same mesh thirty times). sigma is 2% of height because the distinctness findings
    measured that displacement at IoU 0.7163 against the unperturbed mesh
    (2026-08-13-chair-distinctness-findings.md, "What this threshold does and does not buy"),
    comfortably under the 0.98 bound, while leaving aspect ratio and face count untouched.

    Seeded from the uid string via SHA-256, not hash(), for the reason _sample_height's docstring
    spells out: Python's hash() of a str is salted per process.

    The 0.5 footprint (was 0.4) clears the post-scale width bound at EVERY height in the [0.75, 1.0]
    clip band: 0.5 x 0.75 = 0.375 m, against a 0.35 m bound. At 0.4 the exported width would have
    been 0.30-0.40 m depending on each fixture uid's own draw, so roughly half a batch of these
    would have been rejected for width and half kept -- a fixture whose verdict is decided by the
    height sampler is a fixture that tests the sampler, not the criterion under test.
    """
    mesh = trimesh.creation.box(extents=(0.5, 1.0, 0.5))
    mesh = mesh.subdivide().subdivide()
    if uid:
        seed = int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16)
        rng = numpy.random.default_rng(seed)
        mesh.vertices = numpy.asarray(mesh.vertices) + rng.normal(0.0, 0.02, mesh.vertices.shape)
    mesh.export(path)


def _write_bad_pole_mesh(path):
    """Findings doc's own pole fixture (see test_a_pole_is_rejected_for_aspect_ratio above): fails
    the aspect-ratio floor immediately, before face count is even considered (CRITERIA order)."""
    trimesh.creation.box(extents=(0.1, 0.1, 2.0)).export(path)


def _fake_downloader(tmp_path, kinds):
    """Builds a `download` callable for fetch: `kinds` maps uid -> "good"/"bad". Writes a REAL
    mesh file per uid under tmp_path so curate (which loads and measures the mesh) and export
    (which loads and transforms it) have real content to work with, not just a path string that
    resolves to nothing."""
    def download(uids, download_processes=1):
        paths = {}
        for uid in uids:
            path = tmp_path / f"{uid}.obj"
            if kinds[uid] == "good":
                _write_good_chair_mesh(path, uid)
            else:
                _write_bad_pole_mesh(path)
            paths[uid] = str(path)
        return paths
    return download


def test_cli_fetch_subcommand_uses_the_injected_downloader_and_uids(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []

    def fake_download(uids, download_processes=1):
        calls.append(list(uids))
        return {uid: f"/fake/{uid}.glb" for uid in uids}

    manifest = build_chair_manifest.main(
        ["fetch", "--category", "chair", "--count", "2"],
        download=fake_download, uids=["u1", "u2", "u3"],
    )

    assert calls == [["u1", "u2"]]
    assert {e["uid"] for e in manifest["chairs"]} == {"u1", "u2"}
    assert "accepted" not in manifest["chairs"][0], "fetch must not judge; that is curate's job"


def test_cli_curate_subcommand_defaults_to_default_thresholds(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    good_path = tmp_path / "good.obj"
    _write_good_chair_mesh(good_path)
    chair_manifest.save({"chairs": [{"uid": "u1", "raw_path": str(good_path)}]})

    manifest = build_chair_manifest.main(["curate"])

    assert manifest["chairs"][0]["accepted"] is True
    assert manifest["thresholds"] == build_chair_manifest.DEFAULT_THRESHOLDS


def test_cli_export_subcommand_exports_the_accepted_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    raw_path = tmp_path / "raw.obj"
    _write_good_chair_mesh(raw_path)
    chair_manifest.save({"chairs": [{
        "uid": "u1", "accepted": True, "raw_path": str(raw_path), "up_axis": "Y",
    }]})

    manifest = build_chair_manifest.main(["export"])

    assert "obj_path" in manifest["chairs"][0]


def test_cli_convert_subcommand_uses_the_injected_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    chair_manifest.save({"chairs": [{"uid": "u1", "accepted": True, "obj_path": "/fake/u1.obj"}]})
    calls = []

    def fake_runner(input_path, output_path):
        calls.append((input_path, output_path))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = build_chair_manifest.main(["convert"], runner=fake_runner)

    expected_usd = os.path.join(str(tmp_path), "usd", "u1.usd")
    assert calls == [("/fake/u1.obj", expected_usd)]
    assert manifest["chairs"][0]["usd_path"] == expected_usd


def test_cli_repair_collision_subcommand_uses_the_injected_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    chair_manifest.save({"chairs": [{
        "uid": "u1", "accepted": True, "obj_path": "/fake/u1.obj", "usd_path": "/fake/u1.usd",
    }]})
    calls = []

    manifest = build_chair_manifest.main(
        ["repair-collision"], repair_runner=_fake_repair_runner(calls),
    )

    assert calls == [{"u1": "/fake/u1.usd"}]
    assert manifest["chairs"][0]["has_collision"] is True


def test_cli_all_stops_once_twenty_are_accepted_without_exhausting_the_pool(tmp_path, monkeypatch):
    """The default target is twenty (task brief: 'an all that runs them in order until twenty are
    accepted'). Give it a pool of 30 good chairs, fetched five at a time, and confirm it stops the
    moment the 20th is accepted -- proving `all` does not keep fetching/converting past target,
    not merely that it eventually reaches it."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    uids = [f"good-{i}" for i in range(30)]
    kinds = {uid: "good" for uid in uids}
    download = _fake_downloader(tmp_path, kinds)
    convert_calls = []

    def fake_runner(input_path, output_path):
        convert_calls.append((input_path, output_path))
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = build_chair_manifest.main(
        ["all", "--category", "chair", "--batch-size", "5"],
        download=download, uids=uids, runner=fake_runner, repair_runner=_fake_repair_runner(),
    )

    accepted = chair_manifest.accepted(manifest)
    assert len(accepted) == 20, f"expected exactly 20 accepted, got {len(accepted)}"
    assert len(manifest["chairs"]) < 30, "all fetched the entire pool instead of stopping at target"
    assert len(convert_calls) == 20
    for entry in accepted:
        assert "usd_path" in entry
        assert entry["has_collision"] is True


def test_cli_all_stops_and_reports_rather_than_loosening_when_the_pool_runs_dry(tmp_path, monkeypatch, capsys):
    """A pool that can never reach twenty must not push the CLI to widen a threshold to get there
    -- that is the user's decision, with the histogram in front of them (task instructions).
    Confirms: (1) it stops once the fixed pool is exhausted, having fetched every uid, not fewer;
    (2) the saved thresholds are untouched (still exactly DEFAULT_THRESHOLDS); (3) the CLI prints
    the shortfall and a per-reason rejection histogram rather than just returning quietly."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    uids = [f"good-{i}" for i in range(3)] + [f"bad-{i}" for i in range(7)]
    kinds = {uid: ("good" if uid.startswith("good") else "bad") for uid in uids}
    download = _fake_downloader(tmp_path, kinds)

    def fake_runner(input_path, output_path):
        return types.SimpleNamespace(stdout="", returncode=0)

    manifest = build_chair_manifest.main(
        ["all", "--category", "chair", "--batch-size", "4"],
        download=download, uids=uids, runner=fake_runner, repair_runner=_fake_repair_runner(),
    )

    assert len(manifest["chairs"]) == 10, "pool exhausted -- every uid should have been fetched"
    accepted = chair_manifest.accepted(manifest)
    assert len(accepted) == 3, f"expected exactly the 3 good uids accepted, got {len(accepted)}"
    assert manifest["thresholds"] == build_chair_manifest.DEFAULT_THRESHOLDS, (
        "all must not widen a threshold to reach target -- that is the user's decision"
    )

    histogram = build_chair_manifest.rejection_histogram(manifest)
    assert histogram == {"aspect ratio": 7}, histogram

    out = capsys.readouterr().out
    assert "3/20" in out, f"CLI did not report the shortfall plainly: {out!r}"
    assert "aspect ratio: 7" in out, f"CLI did not print the rejection histogram: {out!r}"


# --- facing_direction ------------------------------------------------------------------------
#
# facing_axis (recorded by export(), see _facing_axis) is an AXIS ("X" or "Y"), not a signed
# DIRECTION -- it cannot tell a chair facing a table from one with its back to it. Chair-placement
# Task 1's findings measured the sign is recoverable from the same backrest-asymmetry signal
# _facing_axis already uses: 36/38 real exported chairs' dominant offset axis agreed with the
# recorded facing_axis, and agreement was 30/30 (100%) once restricted to the 30 chairs with a
# confident (non-near-symmetric) signal -- the only two disagreements were both in the 8 weak-
# signal chairs, one of which is a mesh already independently flagged (by eye) as not really
# having a seat/back split at all.


def test_facing_direction_points_away_from_the_backrest():
    """A chair faces away from its back. The manifest records only an AXIS, so the sign has
    to be recovered from the geometry -- and getting it wrong turns every chair around,
    which looks worse than not facing them at all.

    Built rather than downloaded: a seat slab plus a backrest at +Y, so the answer is known.
    """
    import numpy as np, trimesh

    seat = trimesh.creation.box(extents=(0.45, 0.45, 0.05))
    seat.apply_translation([0.0, 0.0, 0.42])
    back = trimesh.creation.box(extents=(0.45, 0.05, 0.45))
    back.apply_translation([0.0, 0.22, 0.65])          # backrest at +Y
    chair = trimesh.util.concatenate([seat, back])

    fx, fy = build_chair_manifest.facing_direction(chair)
    assert abs(fx) < abs(fy), "picked the wrong axis"
    assert fy < 0, f"faces +Y, toward its own backrest, instead of away: ({fx}, {fy})"
    assert abs((fx ** 2 + fy ** 2) ** 0.5 - 1.0) < 1e-6, "not a unit vector"


def test_facing_direction_flips_with_the_backrest():
    """The negative control. If the function ignored the geometry and always returned the
    same vector, the test above would still pass."""
    import trimesh

    def chair_with_back_at(y):
        seat = trimesh.creation.box(extents=(0.45, 0.45, 0.05))
        seat.apply_translation([0.0, 0.0, 0.42])
        back = trimesh.creation.box(extents=(0.45, 0.05, 0.45))
        back.apply_translation([0.0, y, 0.65])
        return trimesh.util.concatenate([seat, back])

    _, plus = build_chair_manifest.facing_direction(chair_with_back_at(+0.22))
    _, minus = build_chair_manifest.facing_direction(chair_with_back_at(-0.22))
    assert plus * minus < 0, "facing did not flip when the backrest moved to the other side"


# --- seeded deep sampling ----------------------------------------------------------------------
#
# Why this exists at all: the shipped twenty came out of the HEAD of the LVIS `chair` list (the 40
# fetched uids are exactly chair[:40], verified against the real annotations), and the head is one
# Objaverse furniture pack -- roughly 20 of the 38 accepted were the same shell-back/cantilever
# design (2026-08-13-chair-distinctness-findings.md, "And the real archetype is much larger than
# six"). _sampled_uids replaces "take the head" with a reproducible draw over the whole category.
# Nothing here touches the network: the pool is synthetic, which is enough because the property
# under test is about the ORDERING RULE, not about Objaverse.

from build_chair_manifest import SAMPLE_SEED, _sampled_uids


def _synthetic_pool(n=1000):
    return [f"{i:032x}" for i in range(n)]


def test_a_seeded_draw_is_the_same_draw_every_time():
    """The whole point of a seed. Mutation: seeding from random.random() (or from nothing at all)
    reddens this immediately."""
    pool = _synthetic_pool()
    assert _sampled_uids(pool, "chair") == _sampled_uids(pool, "chair")


def test_a_seeded_draw_survives_a_separate_interpreter_process():
    """random.Random(<str>) seeds from a SHA-512 of the bytes, with no PYTHONHASHSEED salt -- the
    same property _sample_height's docstring rests on, and the trap this project already hit once
    with random.Random(hash(uid)). A draw that differs per process cannot be "recorded so it can be
    repeated", which is what the plan asks of the sampling rule.

    Mutation: seeding from hash(f"{seed}:{category}") instead of the string reddens this under a
    varied PYTHONHASHSEED while leaving every in-process test above green.
    """
    script = (
        "import sys; sys.path.insert(0, %r);"
        "import build_chair_manifest as B;"
        "print(','.join(B._sampled_uids([f'{i:032x}' for i in range(200)], 'chair')[:10]))"
        % str(Path(__file__).resolve().parents[1] / "tools")
    )
    outputs = set()
    for hashseed in ("0", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outputs.add(subprocess.run([sys.executable, "-c", script], check=True,
                                   capture_output=True, text=True, env=env).stdout.strip())
    assert len(outputs) == 1, f"the draw moved across processes: {outputs}"


def test_a_seeded_draw_ignores_the_order_the_category_list_arrives_in():
    """objaverse.load_lvis_annotations() returns whatever order its JSON happens to carry, and
    which chairs a library gets must not depend on that. Mutation: dropping the sorted() inside
    _sampled_uids reddens this."""
    pool = _synthetic_pool(200)
    shuffled = list(reversed(pool))
    assert _sampled_uids(shuffled, "chair") == _sampled_uids(pool, "chair")


def test_a_seeded_draw_does_not_take_the_head():
    """The defect the seed exists to fix. Over a 1000-uid pool a uniform draw should put about 5%
    of the original first 50 into its own first 50; taking the head puts 100% of them there.

    Mutation: `return sorted(set(uids))` (i.e. forgetting to shuffle) reddens this -- it would put
    all 50 head uids in the first 50 -- while leaving the reproducibility tests above green.
    """
    pool = _synthetic_pool()
    head = set(pool[:50])
    drawn = _sampled_uids(pool, "chair")[:50]
    assert len(head & set(drawn)) <= 10, (
        f"{len(head & set(drawn))} of the pool's first 50 landed in the seeded first 50 -- "
        "the draw is still concentrated at the head"
    )
    assert set(drawn) != head


def test_a_seeded_draw_differs_per_category():
    """Seeded per-category so adding a category later cannot re-order the draws of the ones already
    fetched. Mutation: seeding on `seed` alone reddens this."""
    pool = _synthetic_pool(200)
    assert _sampled_uids(pool, "chair") != _sampled_uids(pool, "armchair")


def test_fetch_without_a_seed_still_takes_the_head():
    """Backwards compatibility, stated as a test rather than assumed: a manifest built before the
    seed existed must be extendable exactly as it was."""
    downloaded = []

    def fake_download(uids, download_processes=1):
        downloaded.extend(uids)
        return {uid: f"/fake/{uid}.glb" for uid in uids}

    pool = _synthetic_pool(100)
    build_chair_manifest.fetch({"chairs": []}, "chair", 5,
                               download=fake_download, uids=pool)
    assert downloaded == pool[:5]


def test_fetch_with_a_seed_draws_beyond_the_head():
    """The CLI path the fifty-chair pool was actually drawn with. Mutation: ignoring `seed` in
    fetch() reddens this."""
    downloaded = []

    def fake_download(uids, download_processes=1):
        downloaded.extend(uids)
        return {uid: f"/fake/{uid}.glb" for uid in uids}

    pool = _synthetic_pool(500)
    build_chair_manifest.fetch({"chairs": []}, "chair", 20,
                               download=fake_download, uids=pool, seed=SAMPLE_SEED)
    assert downloaded == _sampled_uids(pool, "chair", SAMPLE_SEED)[:20]
    assert downloaded != pool[:20]


# --- select: fifty by greedy max-min diversity --------------------------------------------------
#
# Task 1 measured that shape IoU cannot THRESHOLD design similarity on this corpus -- below exact
# identity it is a continuum with no gap anywhere (2026-08-13-chair-distinctness-findings.md,
# "Result 1"). Ranking over it needs no gap, which is why the plan's amendment spends the measure
# here instead of cutting it. These tests pin the ranking rule, not a number.


def _cube_grid(side, n=None):
    """A solid `side`-cell cube, centred in x and y, sitting on z = 0, as a shape descriptor.

    Centred and axis-aligned so it is invariant under all 8 symmetries _grid_iou maximises over --
    which makes the IoU of two of these exactly (small/large)**3, an arithmetic fact rather than
    something that has to be re-measured whenever the descriptor changes.
    """
    n = n or build_chair_manifest.SHAPE_GRID_N
    grid = numpy.zeros((n, n, n), dtype=bool)
    lo = (n - side) // 2
    grid[lo:lo + side, lo:lo + side, 0:side] = True
    return build_chair_manifest._encode_grid(grid)


def _nested_cube_manifest(sides):
    """One accepted entry per cube side, uids "a", "b", "c", ... in the order given."""
    return {"chairs": [
        {"uid": uid, "accepted": True,
         "shape_grid": _cube_grid(side), "shape_grid_n": build_chair_manifest.SHAPE_GRID_N}
        for uid, side in zip("abcdefgh", sides)
    ]}


def test_selection_takes_the_least_similar_candidate_at_every_step():
    """The greedy max-min rule itself, on a pool whose every pairwise IoU is known in closed form.

    Four nested cubes, deliberately NOT in uid order: "a" = 10, "b" = 11, "c" = 2, "d" = 3. IoU is
    (small/large)**3, so a-b = 0.751, a-c = 0.008, a-d = 0.027, b-c = 0.006, b-d = 0.020,
    c-d = 0.296.

    * First pick: lowest max-IoU-against-the-pool. "a" and "b" are each other's near-twin at
      0.751; "c" and "d" tie at 0.296, so the tie-break on smallest uid opens with **"c"** -- not
      with "a", which is what the smallest uid alone would have given. That is the whole reason
      the sides are shuffled here.
    * Then "b" (0.006 against {c}), then "d" (0.296), then "a".

    Mutations: max()->min() in the greedy step reddens the order; seeding the selection from the
    first uid instead of the pool outlier gives "a" first; dropping the uid tie-break makes the
    first pick depend on dict iteration order.
    """
    manifest = build_chair_manifest.select(_nested_cube_manifest([10, 11, 2, 3]), target=4)
    ranks = {e["uid"]: e["selection_rank"] for e in manifest["chairs"]}
    assert sorted(ranks, key=ranks.get) == ["c", "b", "d", "a"]


def test_selection_stops_at_the_target_and_marks_the_rest_unselected():
    """target is a cap, not a suggestion: what is NOT selected must be marked, because export and
    convert read `selected` and a missing key means "no selection was made at all"."""
    manifest = build_chair_manifest.select(_nested_cube_manifest([10, 11, 2, 3]), target=2)
    selected = [e["uid"] for e in manifest["chairs"] if e["selected"]]
    assert selected == ["b", "c"]
    assert all("selected" in e for e in manifest["chairs"])
    assert all("selection_rank" not in e for e in manifest["chairs"] if not e["selected"])


def test_selection_is_reproducible_and_independent_of_manifest_order():
    """The plan's own requirement -- "Seed the selection deterministically and test that it is
    reproducible" -- and the same defect curate's uid-order rule guards against: manifest list
    order is FETCH order, which depends on how many batches were run, and must not decide which
    chairs the library ships.

    Real cached meshes, not cubes, so this exercises the descriptor as well as the ranking.
    Mutation: iterating manifest["chairs"] instead of sorting by uid reddens the shuffled case.
    """
    uids = list(DUPLICATE_PAIR[:1]) + list(CLOSEST_DISTINCT_PAIR) + [
        "304253851afd493d958fc8e256c189df", "83c7586454d24b82ab7fa697efd4b9af",
    ]
    entries = list(_curated(uids, DEFAULT_THRESHOLDS).values())

    def picked(order):
        manifest = build_chair_manifest.select(
            {"chairs": [dict(e) for e in order]}, target=3)
        chosen = [e for e in manifest["chairs"] if e.get("selected")]
        return [e["uid"] for e in sorted(chosen, key=lambda e: e["selection_rank"])]

    first = picked(entries)
    assert len(first) == 3
    assert first == picked(list(reversed(entries)))
    assert first == picked(sorted(entries, key=lambda e: e["faces"]))


def test_selection_never_reintroduces_a_similarity_threshold():
    """Task 1's finding, held in place: there is no gap in design similarity, so nothing here may
    reject a chair for being similar. select() RANKS -- given a target as large as the pool it must
    return the whole pool, however alike its members are.

    Mutation: adding any "skip candidates above IoU x" guard to the greedy loop reddens this.
    """
    manifest = build_chair_manifest.select(_nested_cube_manifest([10, 11]), target=2)
    assert [e["uid"] for e in manifest["chairs"] if e["selected"]] == ["a", "b"]
    assert manifest["selection"]["selected"] == 2


def test_selection_falls_open_for_a_chair_with_no_descriptor():
    """An accepted chair from a manifest curated before shape_grid existed cannot be ranked. It
    must not be silently dropped from the library -- fail open, the same choice _distinctness makes
    for a missing descriptor. Mutation: filtering unrankable entries out entirely reddens this."""
    manifest = _nested_cube_manifest([2, 3])
    manifest["chairs"].append({"uid": "z", "accepted": True})
    manifest = build_chair_manifest.select(manifest, target=3)
    assert {e["uid"] for e in manifest["chairs"] if e["selected"]} == {"a", "b", "z"}


def test_export_and_convert_touch_only_the_selection(tmp_path, monkeypatch):
    """The reason conversion moved after selection at all: one Isaac boot per mesh, so a chair that
    did not make the cut must never be converted. Mutation: leaving convert()'s loop on
    accepted-ness rather than offered() reddens this."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []
    manifest = {"chairs": [
        {"uid": "in", "accepted": True, "selected": True, "obj_path": "/fake/in.obj"},
        {"uid": "out", "accepted": True, "selected": False, "obj_path": "/fake/out.obj"},
    ]}
    build_chair_manifest.convert(manifest, runner=lambda i, o: calls.append((i, o)))
    assert [c[0] for c in calls] == ["/fake/in.obj"]


def test_convert_still_converts_everything_accepted_when_nothing_was_selected(tmp_path, monkeypatch):
    """offered() must be a no-op on a manifest that predates select(). Mutation: treating a missing
    `selected` key as False would quietly convert nothing at all."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []
    manifest = {"chairs": [{"uid": "a", "accepted": True, "obj_path": "/fake/a.obj"}]}
    build_chair_manifest.convert(manifest, runner=lambda i, o: calls.append((i, o)))
    assert len(calls) == 1


def test_convert_limit_stops_and_the_next_call_resumes(tmp_path, monkeypatch):
    """A fifty-chair convert is over half an hour of Isaac boots and the CLI saves once, at the
    end. --limit turns the manifest into a checkpoint. Mutation: counting SKIPPED entries against
    the limit as well reddens the resume half."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    calls = []
    manifest = {"chairs": [
        {"uid": u, "accepted": True, "obj_path": f"/fake/{u}.obj"} for u in "abcde"
    ]}
    build_chair_manifest.convert(manifest, runner=lambda i, o: calls.append(i), limit=2)
    assert len(calls) == 2
    build_chair_manifest.convert(manifest, runner=lambda i, o: calls.append(i), limit=2)
    assert calls == ["/fake/a.obj", "/fake/b.obj", "/fake/c.obj", "/fake/d.obj"]


# --- prune ---------------------------------------------------------------------------------------


def test_prune_deletes_the_artefacts_of_a_chair_the_library_no_longer_offers():
    """export/convert are idempotent by skipping what exists, which is right for GROWING a library
    and wrong for RE-DECIDING one. kitchen_build offers exactly the chairs carrying a usd_path, so
    a chair that a re-curate rejected -- or that select() passed over -- would otherwise keep being
    offered. Mutation: pruning only rejected entries (ignoring `selected`) reddens the second half.
    """
    removed = []
    manifest = {"chairs": [
        {"uid": "keep", "accepted": True, "selected": True,
         "obj_path": "/o/keep.obj", "usd_path": "/u/keep.usd", "has_collision": True},
        {"uid": "unselected", "accepted": True, "selected": False,
         "obj_path": "/o/unselected.obj", "usd_path": "/u/unselected.usd", "has_collision": True},
        {"uid": "rejected", "accepted": False, "rejected_because": "face count: too many",
         "obj_path": "/o/rejected.obj", "faces": 9},
    ]}
    manifest = build_chair_manifest.prune(manifest, remove=removed.append)

    assert sorted(removed) == ["/o/rejected.obj", "/o/unselected.obj", "/u/unselected.usd"]
    kept, unselected, rejected = manifest["chairs"]
    assert kept["usd_path"] == "/u/keep.usd" and kept["has_collision"] is True
    assert "usd_path" not in unselected and "has_collision" not in unselected
    assert "obj_path" not in rejected


def test_prune_keeps_every_measurement_and_verdict():
    """The point of keeping a rejected entry is that a later threshold change can see what it
    excluded without re-fetching (chair_manifest.py's module docstring). Only artefact POINTERS
    go. Mutation: clearing the whole entry, or dropping shape_grid with it, reddens this."""
    manifest = {"chairs": [{
        "uid": "gone", "accepted": False, "rejected_because": "export width: 0.3 m",
        "obj_path": "/o/gone.obj", "faces": 1234, "shape_grid": "AAAA",
        "export_footprint_m": [0.28, 0.30],
    }]}
    entry = build_chair_manifest.prune(manifest, remove=lambda p: None)["chairs"][0]
    assert entry["faces"] == 1234
    assert entry["shape_grid"] == "AAAA"
    assert entry["rejected_because"].startswith("export width")


def test_prune_is_not_upset_by_a_file_that_is_already_gone():
    """prune is a reconciliation; being asked to delete what is already deleted is the state it is
    trying to reach. Mutation: dropping the OSError suppression reddens this."""
    def exploding_remove(path):
        raise FileNotFoundError(path)

    manifest = {"chairs": [{"uid": "x", "accepted": False, "obj_path": "/o/x.obj"}]}
    assert "obj_path" not in build_chair_manifest.prune(
        manifest, remove=exploding_remove)["chairs"][0]


# --- curate's I/O boundary ----------------------------------------------------------------------


def test_an_unreadable_candidate_is_rejected_rather_than_aborting_the_batch(chairish_thresholds,
                                                                            tmp_path):
    """curate judges a couple of hundred arbitrary Objaverse uploads in one call and the CLI saves
    the manifest once, at the end -- so one malformed GLB throwing would discard every measurement
    taken before it. Mutation: removing the try/except turns this test into an error, and the
    surviving good chair below is what proves the batch continued rather than merely not crashing.
    """
    broken = tmp_path / "broken.glb"
    broken.write_bytes(b"this is not a GLB")
    good = _chair_like_scene_path(tmp_path)

    manifest = curate({"chairs": [
        {"uid": "aaa_broken", "raw_path": str(broken)},
        {"uid": "bbb_good", "raw_path": str(good)},
    ]}, thresholds=chairish_thresholds)

    bad, ok = manifest["chairs"]
    assert bad["accepted"] is False
    assert bad["rejected_because"].startswith("unreadable: ")
    assert "faces" not in bad, "recorded measurements for a mesh it never managed to load"
    assert ok["accepted"] is True, "a later candidate was lost when an earlier one failed to load"
    assert build_chair_manifest.rejection_histogram(manifest) == {"unreadable": 1}


# --- convert's retry ------------------------------------------------------------------------------


def test_the_convert_runner_retries_a_flaky_conversion(monkeypatch):
    """MeshConverter re-opens the USD it has just written, and on this cluster's Lustre storage
    that reopen loses the race against its own write. The 2026-08-11 investigation caught the quiet
    form (no collider, repaired afterwards); the fifty-chair batch caught the loud one -- "Accessed
    invalid null prim", same command, same node, attempt 1 fine, attempt 2 dead, attempt 3 fine.

    Mutation: dropping the retry loop makes a fifty-chair batch die on its first flake.
    """
    attempts = []

    def flaky(cmd, **kwargs):
        attempts.append(cmd)
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(1, cmd, stderr="Accessed invalid null prim")
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(build_chair_manifest.subprocess, "run", flaky)
    assert build_chair_manifest._default_convert_runner("/in.obj", "/out.usd").returncode == 0
    assert len(attempts) == 3


def test_the_convert_runner_gives_up_rather_than_retrying_forever(monkeypatch):
    """A mesh the converter genuinely cannot handle must still fail, loudly, and not become an
    infinite loop. Mutation: `while True` instead of a bounded range hangs the suite."""
    attempts = []

    def always_fails(cmd, **kwargs):
        attempts.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(build_chair_manifest.subprocess, "run", always_fails)
    with pytest.raises(subprocess.CalledProcessError):
        build_chair_manifest._default_convert_runner("/in.obj", "/out.usd")
    assert len(attempts) == build_chair_manifest._CONVERT_ATTEMPTS


def test_export_refreshes_facing_axis_on_a_chair_it_skips(tmp_path, monkeypatch):
    """facing_axis is the one field two stages both write: curate measures it in the RAW mesh's
    frame, export rewrites it in the EXPORTED frame, and _stand_up can turn curate's "Z" into
    export's "X". So re-curating a manifest whose OBJs already exist -- exactly what the
    fifty-chair rebuild did -- overwrites the field with a raw-frame value that export's own
    obj_path skip would then preserve forever. Measured on the three chairs carried over from the
    previous library: two ended up recording "Z", which is not a possible answer in the exported
    frame at all.

    Mutation: dropping the _refresh_facing_axis call from export()'s skip branch leaves "Z" in
    place and reddens this.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    seat = trimesh.creation.box(extents=(0.45, 0.45, 0.05))
    seat.apply_translation([0.0, 0.0, 0.42])
    back = trimesh.creation.box(extents=(0.45, 0.05, 0.45))
    back.apply_translation([0.0, 0.22, 0.65])
    obj_path = tmp_path / "already.obj"
    trimesh.util.concatenate([seat, back]).export(str(obj_path), include_texture=False)

    manifest = build_chair_manifest.export({"chairs": [{
        "uid": "already", "accepted": True, "obj_path": str(obj_path),
        # what curate wrote, in the raw frame -- impossible in the exported one
        "facing_axis": "Z", "up_axis": "Z", "height_m": 0.875,
    }]})

    entry = manifest["chairs"][0]
    assert entry["facing_axis"] == "Y", "the backrest is on Y; the raw-frame 'Z' survived export"
    assert entry["obj_path"] == str(obj_path), "export re-wrote an OBJ it was meant to skip"


# --- the contact sheet ----------------------------------------------------------------------------


def _chairish_obj(path, *, width=0.45, height=0.9):
    """A box-and-backrest chair, exported the way export() leaves one: Z-up, life-sized, on z=0."""
    seat = trimesh.creation.box(extents=(width, width, 0.05))
    seat.apply_translation([0.0, 0.0, height * 0.47])
    back = trimesh.creation.box(extents=(width, 0.05, height * 0.5))
    back.apply_translation([0.0, width / 2, height * 0.75])
    legs = trimesh.creation.box(extents=(width * 0.9, width * 0.9, height * 0.45))
    legs.apply_translation([0.0, 0.0, height * 0.225])
    mesh = trimesh.util.concatenate([seat, back, legs])
    mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])
    mesh.export(str(path), include_texture=False)
    return str(path)


def _sheet_manifest(tmp_path, count, *, columns_hint=None):
    chairs = []
    for i in range(count):
        uid = f"{i:032x}"
        chairs.append({
            "uid": uid, "category": "chair", "accepted": True, "selected": True,
            "selection_rank": i, "faces": 12, "height_m": 0.9,
            "export_footprint_m": [0.45, 0.45],
            "obj_path": _chairish_obj(tmp_path / f"{uid}.obj"),
        })
    return {"chairs": chairs}


def test_the_contact_sheet_draws_every_offered_chair_even_when_the_grid_does_not_divide(
        tmp_path, monkeypatch):
    """The failure this whole check exists for: five chairs into a three-column grid. Both scratch
    versions of this renderer looped `zip(np.ravel(axes), uids)`, which stops at the shorter of the
    two and produces a sheet that LOOKS complete while omitting whatever ran off the end -- and an
    omitted chair is precisely the one nobody ever inspects.

    Mutation: computing the row count as `len(entries) // columns` instead of rounding up drops the
    last two here, and the RuntimeError in contact_sheet() would fire before the sheet is trusted.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = _sheet_manifest(tmp_path, 5)

    report = build_chair_manifest.contact_sheet(manifest, columns=3)

    assert report["drawn"] == [e["uid"] for e in manifest["chairs"]]
    assert report["failed"] == {}
    assert os.path.getsize(report["path"]) > 0


def test_the_contact_sheet_draws_each_chair_from_two_distinct_viewpoints(tmp_path, monkeypatch):
    """One viewpoint is not enough, and that is measured rather than stylistic: a pedestal stand
    reads as a chair from one azimuth, and a welded-in floor plane or backdrop can hide edge-on.
    Both earlier visual passes used two.

    Mutation: rendering only azimuths[0] -- or passing the same azimuth twice -- reddens this.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = _sheet_manifest(tmp_path, 2)
    seen = {}

    real = build_chair_manifest._flat_shade

    def spy(ax, mesh, *, azim, **kw):
        seen.setdefault(round(float(mesh.bounding_box.extents[2]), 6), []).append(azim)
        return real(ax, mesh, azim=azim, **kw)

    monkeypatch.setattr(build_chair_manifest, "_flat_shade", spy)
    build_chair_manifest.contact_sheet(manifest, columns=2)

    angles = list(build_chair_manifest.CONTACT_SHEET_AZIMUTHS)
    assert len(angles) == len(set(angles)) >= 2
    assert list(seen.values()) == [angles * 2]     # both chairs are the same box, so one key


def test_a_chair_whose_mesh_will_not_load_gets_a_failed_cell_rather_than_no_cell(
        tmp_path, monkeypatch):
    """One unreadable OBJ must not cost the other forty-nine their inspection, and must not quietly
    shorten the sheet either. Mutation: `continue`-ing past the bad entry reddens both asserts."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = _sheet_manifest(tmp_path, 3)
    manifest["chairs"][1]["obj_path"] = str(tmp_path / "does_not_exist.obj")

    report = build_chair_manifest.contact_sheet(manifest, columns=3)

    assert report["drawn"] == [e["uid"] for e in manifest["chairs"]]
    assert list(report["failed"]) == [manifest["chairs"][1]["uid"]]


def test_the_contact_sheet_renders_the_offered_selection_not_the_whole_accepted_pool(
        tmp_path, monkeypatch):
    """161 accepted, 50 offered. A sheet of the pool would bury the shipped chairs in it. Mutation:
    looping over chair_manifest.accepted() instead of offered() reddens this."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = _sheet_manifest(tmp_path, 3)
    manifest["chairs"][2]["selected"] = False
    manifest["chairs"][2].pop("selection_rank")

    report = build_chair_manifest.contact_sheet(manifest, columns=3)

    assert report["drawn"] == [e["uid"] for e in manifest["chairs"][:2]]


def test_every_cell_shares_one_world_window_so_a_doll_sized_chair_looks_doll_sized(
        tmp_path, monkeypatch):
    """Fitting each cell to its own mesh -- what both scratch renderers did -- normalises scale
    away, and "implausible scale" is one of the things a human is meant to catch here. Mutation:
    reinstating the per-mesh `set_xlim(cx - r, cx + r)` fit makes both cells identical and reddens
    the ratio assert.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    plt, _ = build_chair_manifest._agg_pyplot()
    fig, (big_ax, small_ax) = plt.subplots(1, 2)

    big = trimesh.load(_chairish_obj(tmp_path / "big.obj", width=0.45, height=0.9), force="mesh")
    small = trimesh.load(_chairish_obj(tmp_path / "small.obj", width=0.15, height=0.3),
                         force="mesh")
    assert build_chair_manifest._flat_shade(big_ax, big, azim=35.0)
    assert build_chair_manifest._flat_shade(small_ax, small, azim=35.0)

    assert big_ax.get_xlim() == small_ax.get_xlim() == (-0.7, 0.7)
    drawn = [ax.collections[0].get_paths()[0].vertices for ax in (big_ax, small_ax)]
    assert numpy.ptp(drawn[0][:, 1]) > 2 * numpy.ptp(drawn[1][:, 1])
    plt.close(fig)


def test_a_mesh_that_overflows_the_shared_window_is_reported_rather_than_silently_cropped(
        tmp_path, monkeypatch):
    """A fixed window means something can fall outside it -- a mesh carrying a welded-in floor
    plane, say. Cropping it without saying so would hide exactly the defect the sheet is for.
    Mutation: returning True unconditionally from _flat_shade reddens this."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    plt, _ = build_chair_manifest._agg_pyplot()
    fig, ax = plt.subplots()
    slab = trimesh.creation.box(extents=(4.0, 4.0, 0.02))
    assert build_chair_manifest._flat_shade(ax, slab, azim=35.0) is False
    plt.close(fig)

    manifest = _sheet_manifest(tmp_path, 1)
    slab.export(manifest["chairs"][0]["obj_path"], include_texture=False)
    report = build_chair_manifest.contact_sheet(manifest, columns=1)
    assert report["overflowed"] == [manifest["chairs"][0]["uid"]]


def test_the_cli_contact_sheet_subcommand_writes_where_it_is_told(tmp_path, monkeypatch, capsys):
    """Step 2's whole point: a repeatable stage, not a scratch script that goes stale."""
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    chair_manifest.save(_sheet_manifest(tmp_path, 2))
    out = tmp_path / "sheet.png"

    build_chair_manifest.main(["contact-sheet", "--out", str(out), "--columns", "2"])

    assert out.exists()
    assert "2 offered chair(s), 2 viewpoints each" in capsys.readouterr().out


def test_a_chair_exported_off_origin_is_still_drawn_and_its_offset_reported(tmp_path, monkeypatch):
    """Rendering the real fifty measured this: export() bottom-aligns z but never centres x/y, so
    31 of the offered fifty sit more than 5 cm off-origin and one is 9.95 m out. Framing on world
    x=y=0 would have produced 31 blank cells -- a sheet that is worse than useless, because it
    looks complete. The sheet slides each mesh onto the origin to draw it and reports the offset as
    a number instead.

    Mutation: dropping _horizontal_centre from _flat_shade blanks the cell (the projected mesh
    lands entirely outside the window, so the overflow flag fires and nothing is visible);
    dropping it from contact_sheet's own bookkeeping loses the report.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    manifest = _sheet_manifest(tmp_path, 2)
    strayed = trimesh.load(manifest["chairs"][0]["obj_path"], force="mesh")
    strayed.apply_translation([9.95, 0.0, 0.0])
    strayed.export(manifest["chairs"][0]["obj_path"], include_texture=False)

    report = build_chair_manifest.contact_sheet(manifest, columns=2)

    uid = manifest["chairs"][0]["uid"]
    assert report["overflowed"] == [], "an off-origin chair was cropped instead of recentred"
    assert report["off_centre_m"][uid] == pytest.approx(9.95, abs=1e-3)
    assert manifest["chairs"][1]["uid"] not in report["off_centre_m"]


# --- manual rejects, the up-axis rule, centring, and the category cap -------------------------------


def _grid_of(i):
    """A distinct-enough descriptor per index. The cap tests are about WHICH candidates the loop
    is allowed to take, not about the ranking, so cube sides may repeat -- ties break on uid and
    stay deterministic."""
    return _cube_grid(2 + (i % 9))


def test_a_manual_reject_retracts_a_chair_a_previous_run_accepted(chairish_thresholds):
    """Every one of the 21 was accepted, selected AND converted before a human saw it, so a hand
    verdict that only applied to unjudged entries would have changed nothing. curate applies
    MANUAL_REJECTS before its already-judged skip, and retracts.

    Mutation: moving the manual-reject pass below the `if "accepted" in entry: continue` skip
    leaves the entry accepted and reddens this.
    """
    uid = next(iter(build_chair_manifest.MANUAL_REJECTS))
    manifest = curate({"chairs": [
        {"uid": uid, "raw_path": "/gone.glb", "accepted": True, "selected": True},
    ]}, thresholds=chairish_thresholds)
    entry = manifest["chairs"][0]

    assert entry["accepted"] is False
    assert entry["manual_reject"] is True
    assert entry["rejected_because"].startswith("manual reject: ")
    assert build_chair_manifest.MANUAL_REJECTS[uid] in entry["rejected_because"]


def test_a_manual_reject_survives_a_re_curate_from_clean(chairish_thresholds):
    """The durability requirement: strip the verdicts and judge again -- the geometric criteria
    accepted the licence-text placard and the railing last time and would again. Mutation: storing
    the verdict only on the manifest (rather than in the module) loses it on a strip-and-recurate.
    """
    uid = "39059edc24d547f393c91856606e25da"        # the licence-text placard
    stripped = {"chairs": [{"uid": uid, "raw_path": _cached_glb(uid), "category": "chair"}]}
    entry = curate(stripped, thresholds=chairish_thresholds)["chairs"][0]

    assert entry["accepted"] is False
    assert "licence-text placard" in entry["rejected_because"]


def test_a_manual_reject_stops_shadowing_duplicates_of_itself(chairish_thresholds):
    """_distinctness rejects a candidate for duplicating an ACCEPTED incumbent. If a retracted
    chair stayed in accepted_so_far, it would keep taking a real chair down with it -- and the
    retracted one is the copy nobody wants. Mutation: building accepted_so_far before the manual
    pass instead of after reddens this.
    """
    uid = next(iter(build_chair_manifest.MANUAL_REJECTS))
    grid = "A" * 100
    manifest = curate({"chairs": [
        {"uid": uid, "raw_path": "/gone.glb", "accepted": True,
         "shape_grid": grid, "shape_grid_n": build_chair_manifest.SHAPE_GRID_N},
    ]}, thresholds=chairish_thresholds)

    assert [e for e in manifest["chairs"] if e.get("accepted")] == []


def test_the_up_axis_comes_from_the_file_format_not_from_the_longest_extent():
    """A lounger is longest along the floor, so argmax(extents) called its length "up" and export
    stood it on end -- 8 of the 50 offered. glTF declares +Y up and the desk gate measured that
    declaration correct 20/20 against argmax's 0/20.

    Mutation: reverting _up_axis to argmax(extents) reddens the .glb case, since this mesh is
    deliberately longest along X.
    """
    lounger = trimesh.creation.box(extents=(2.0, 0.6, 0.8))     # long, low: a lounger's shape

    assert build_chair_manifest._up_axis(lounger, "/x/y.glb") == (1, "gltf-y-up")
    assert build_chair_manifest._up_axis(lounger, "/x/y.GLTF") == (1, "gltf-y-up")
    # No declared convention: measurement is all there is, and it picks the longest axis.
    assert build_chair_manifest._up_axis(lounger, "/x/y.obj") == (0, "argmax-extents")


def test_a_mesh_longer_than_it_is_tall_is_now_rejected_by_the_aspect_band(chairish_thresholds):
    """The rejection the old up-axis rule was structurally incapable of making: under
    argmax(extents) the up axis WAS the longest axis, so normalized_extents[up] was identically 1.0
    and no horizontal could ever exceed the aspect_max. Every lounger therefore passed. This is the
    criterion doing the work, with no new threshold.

    Mutation: reverting _up_axis accepts this mesh outright.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp_glb = os.path.join(d, "lounger.glb")
        trimesh.creation.box(extents=(2.0, 0.6, 0.8)).export(tmp_glb)
        entry = curate({"chairs": [{"uid": "lounger", "raw_path": tmp_glb}]},
                       thresholds=chairish_thresholds)["chairs"][0]

    assert entry["accepted"] is False
    assert entry["rejected_because"].startswith("aspect ratio")
    assert entry["up_axis"] == "Y"


def test_the_predicted_footprint_still_matches_export_for_a_mesh_that_is_not_tallest_up():
    """_export_footprint divides by normalized_extents[up_axis]. That division was a no-op while
    up_axis was argmax; it is load-bearing now. Mutation: dropping it under-predicts this mesh's
    footprint by 39% -- and _export_width judges chairs on that number.
    """
    normalized = [1.0, 0.4, 0.6]        # up (Y) is NOT the largest extent
    height, footprint = build_chair_manifest._export_footprint(normalized, 1, "some-uid")

    assert footprint == pytest.approx([0.6 / 0.4 * height, 1.0 / 0.4 * height])


def test_export_puts_the_chair_over_the_origin_not_just_on_the_floor(tmp_path, monkeypatch):
    """A LIVE PLACEMENT BUG, not a tidy-up: add_chair composes the seat transform with the asset's
    own frame, so an OBJ 9.95 m from its origin landed 9.953 m from the seat the ring computed --
    outside the kitchen, touching nothing, so the overlap gate saw nothing wrong. 31 of the 50
    offered were over 5 cm out.

    Mutation: restoring the z-only translation leaves centre_xy at the source mesh's offset and
    reddens the first assert while leaving the floor assert green -- which is exactly how this
    survived: something did assert z.
    """
    monkeypatch.setenv("CHAIR_OBJ_DIR", str(tmp_path))
    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.9))
    mesh.apply_translation([9.95, -3.0, 12.0])              # an author's arbitrary origin
    raw = tmp_path / "stray.glb"
    mesh.export(str(raw))

    manifest = build_chair_manifest.export({"chairs": [{
        "uid": "stray", "accepted": True, "raw_path": str(raw), "up_axis": "Z",
    }]})

    exported = trimesh.load(manifest["chairs"][0]["obj_path"], force="mesh")
    lo, hi = exported.bounds
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([0.0, 0.0], abs=1e-6)
    assert lo[2] == pytest.approx(0.0, abs=1e-6)


def test_no_category_may_exceed_the_cap():
    """deck_chair supplied 13 of 50 from a 24-uid list whose tail is junk. Mutation: dropping the
    `eligible` filter from select()'s loop lets the dominant category take every slot."""
    chairs = []
    for i in range(12):
        chairs.append({"uid": f"c{i:02d}", "category": "chair", "accepted": True,
                       "shape_grid": _grid_of(i), "shape_grid_n": build_chair_manifest.SHAPE_GRID_N})
    for i in range(6):
        chairs.append({"uid": f"s{i:02d}", "category": "stool", "accepted": True,
                       "shape_grid": _grid_of(50 + i),
                       "shape_grid_n": build_chair_manifest.SHAPE_GRID_N})

    build_chair_manifest.select({"chairs": chairs}, 10, category_cap=0.3)   # ceil(0.3*10) = 3

    picked = [e for e in chairs if e["selected"]]
    counts = {}
    for e in picked:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    assert max(counts.values()) <= 3, counts


def test_the_cap_reports_what_it_selected_per_category():
    """The spread is the product; asserting it is in the manifest is asserting the user can see
    what they got. Mutation: dropping by_category from the provenance reddens this."""
    chairs = [{"uid": f"c{i}", "category": "chair" if i < 4 else "stool", "accepted": True,
               "shape_grid": _grid_of(i), "shape_grid_n": build_chair_manifest.SHAPE_GRID_N}
              for i in range(8)]
    manifest = build_chair_manifest.select({"chairs": chairs}, 8, category_cap=0.5)

    assert manifest["selection"]["by_category"] == {"chair": 4, "stool": 4}
    assert manifest["selection"]["category_cap"] == 0.5
    assert manifest["selection"]["category_cap_count"] == 4


def test_an_uncapped_selection_is_exactly_what_it_was_before_the_cap_existed():
    """category_cap=None must be a true no-op, so a caller that predates the cap is undisturbed.
    Mutation: defaulting `taken` to 0 but still filtering reddens nothing; forgetting the None
    branch in _capped rejects everything and reddens this."""
    chairs = [{"uid": f"c{i}", "category": "chair", "accepted": True,
               "shape_grid": _grid_of(i), "shape_grid_n": build_chair_manifest.SHAPE_GRID_N}
              for i in range(6)]
    build_chair_manifest.select({"chairs": chairs}, 4, category_cap=None)

    assert sum(1 for e in chairs if e["selected"]) == 4


class _RoomSpec(build_chair_manifest.ChairType):
    """A ChairType whose room filter is small enough to reason about by hand."""

    room_categories = ("chair", "stool")
    room_rejects = {"hand": "not kitchen furniture: an upholstered wingback wearing the label"}
    category_cap = None


def _room_pool(*categories, hand=None):
    """One accepted entry per category, uids c0, c1, ...; `hand` inserts the uid _RoomSpec
    hand-rejects at that index, so a test can separate the allow-list from the dict."""
    chairs = [{"uid": f"c{i}", "category": category, "accepted": True,
               "shape_grid": _grid_of(i), "shape_grid_n": build_chair_manifest.SHAPE_GRID_N}
              for i, category in enumerate(categories)]
    if hand is not None:
        chairs[hand]["uid"] = "hand"
    return {"chairs": chairs}


def test_a_category_outside_the_room_allow_list_is_excluded_and_says_so():
    """Rule 0. A rocking chair is real, whole, correctly-curated furniture that simply is not
    kitchen furniture, so it keeps accepted=True and is excluded at SELECTION with a readable
    reason. Mutation: dropping the room_categories branch selects it like anything else."""
    manifest = _room_pool("chair", "rocking_chair", "stool")
    build_chair_manifest.select(manifest, 3, spec=_RoomSpec())

    rocker = manifest["chairs"][1]
    assert rocker["accepted"] is True, "rule 0 must not retract a curate verdict"
    assert rocker["selected"] is False
    assert "rocking_chair" in rocker["not_for_room"]
    assert {e["uid"] for e in manifest["chairs"] if e["selected"]} == {"c0", "c2"}


def test_a_hand_excluded_chair_is_excluded_even_from_an_allowed_category():
    """`armchair` covers both a kitchen carver and an upholstered wingback, and no category rule
    separates them -- which is the whole reason room_rejects exists beside the allow-list.
    Mutation: applying only the allow-list and not the dict reddens this."""
    manifest = _room_pool("chair", "armchair", "stool", hand=1)
    build_chair_manifest.select(manifest, 3, spec=_RoomSpec())

    assert manifest["chairs"][1]["selected"] is False
    assert manifest["chairs"][1]["not_for_room"] == _RoomSpec.room_rejects["hand"]
    assert manifest["chairs"][1]["accepted"] is True


def test_the_room_filter_also_governs_the_backfill():
    """THE load-bearing property, and the one a post-hoc filter over the chosen fifty would not
    have: excluding a category from the OFFERED set but not from the POOL means the replacement
    draw walks straight back to another one of the same kind. Here two of five are rocking chairs
    and the target is 3 -- if rule 0 ran after the pick instead of before it, the selection would
    fall to 1 or 2 rather than backfilling from the allowed categories.

    Mutation: filter `chosen` at the end instead of `accepted_entries` at the start."""
    manifest = _room_pool("rocking_chair", "chair", "rocking_chair", "stool", "chair")
    build_chair_manifest.select(manifest, 3, spec=_RoomSpec())

    picked = {e["uid"] for e in manifest["chairs"] if e["selected"]}
    assert len(picked) == 3, picked
    assert not any(e["category"] == "rocking_chair"
                   for e in manifest["chairs"] if e["selected"])


def test_a_re_selection_retracts_a_stale_room_exclusion_and_a_stale_selection():
    """The mark is rewritten from scratch every select(), in both directions. A chair that leaves
    room_rejects must lose `not_for_room`; a chair that ENTERS it must lose `selected: True`,
    because kitchen_build offers exactly what carries a usd_path and prune() reads `selected` to
    decide what to delete. Mutation: writing `selected` over accepted_entries instead of over every
    judged entry leaves the newly-excluded chair marked selected and still shipped."""
    manifest = _room_pool("chair", "chair", "stool", hand=1)
    unfiltered = build_chair_manifest.ChairType()
    unfiltered.room_categories, unfiltered.room_rejects = (), {}
    build_chair_manifest.select(manifest, 3, category_cap=None, spec=unfiltered)
    assert manifest["chairs"][1]["selected"] is True
    assert "not_for_room" not in manifest["chairs"][1]

    build_chair_manifest.select(manifest, 3, spec=_RoomSpec())
    assert manifest["chairs"][1]["selected"] is False
    assert "not_for_room" in manifest["chairs"][1]

    build_chair_manifest.select(manifest, 3, category_cap=None, spec=unfiltered)
    assert "not_for_room" not in manifest["chairs"][1], "a stale exclusion outlived its rule"
    assert manifest["chairs"][1]["selected"] is True


def test_an_unlabelled_chair_is_never_room_excluded():
    """Fail open on a missing category, for the reason _capped gives: "uncategorised" is one bucket
    holding everything the manifest failed to label, not a category, and excluding it would
    truncate a selection for a reason no user could read. Mutation: dropping the `is not None`
    guard makes every entry in a manifest predating the `category` field unselectable."""
    manifest = {"chairs": [{"uid": "c0", "accepted": True, "shape_grid": _grid_of(0),
                            "shape_grid_n": build_chair_manifest.SHAPE_GRID_N}]}
    build_chair_manifest.select(manifest, 1, spec=_RoomSpec())

    assert manifest["chairs"][0]["selected"] is True
    assert "not_for_room" not in manifest["chairs"][0]


def test_the_room_filter_reports_what_it_excluded():
    """The count and the spread are the product of this pass; the manifest has to carry them or a
    reviewer cannot see what the allow-list cost. Mutation: dropping room_excluded from the
    provenance reddens this."""
    manifest = _room_pool("chair", "rocking_chair", "rocking_chair", "deck_chair", "stool")
    provenance = build_chair_manifest.select(manifest, 5, spec=_RoomSpec())["selection"]

    assert provenance["room_categories"] == ["chair", "stool"]
    assert provenance["room_excluded"] == 3          # two rocking_chair, one deck_chair
    assert provenance["room_excluded_by_category"] == {"deck_chair": 1, "rocking_chair": 2}
    assert provenance["pool"] == 2


def test_the_two_reject_lists_do_not_overlap():
    """manual_rejects is "not furniture, or not one object"; room_rejects is "furniture, wrong
    room". A uid in both would mean the project cannot say which claim it is making, and the two
    land in different fields (`rejected_because` against `not_for_room`). Mutation: moving any
    kitchen verdict into MANUAL_REJECTS reddens this."""
    for module in (build_chair_manifest, build_table_manifest):
        overlap = set(module.MANUAL_REJECTS) & set(module.KITCHEN_REJECTS)
        assert not overlap, f"{module.__name__}: {overlap}"


def test_every_kitchen_reject_names_a_chair_the_manifest_actually_holds():
    """A hand verdict on a uid that is not in the corpus is dead weight nobody will ever notice --
    a typo in a 32-hex string looks exactly like a real entry. Mutation: mistyping any uid in
    KITCHEN_REJECTS reddens this."""
    known = {e["uid"] for e in chair_manifest.load()["chairs"]}
    missing = set(build_chair_manifest.KITCHEN_REJECTS) - known
    assert not missing, missing


def test_a_hand_rejected_chair_still_records_the_measurements_it_was_judged_beside(
        chairish_thresholds):
    """chair_manifest.py's contract is that a rejected entry keeps its numbers so a later reader
    can re-decide from the manifest alone. The hand verdicts are the rejections a sceptic most
    wants to check, so they must not be the only ones with no evidence attached.

    Mutation: dropping the `if manual and "extents" in entry: continue` guard's else-path -- i.e.
    letting the manual pre-pass's `accepted` key skip measurement entirely -- reddens this.
    """
    uid = "39059edc24d547f393c91856606e25da"
    entry = curate({"chairs": [{"uid": uid, "raw_path": _cached_glb(uid), "category": "chair"}]},
                   thresholds=chairish_thresholds)["chairs"][0]

    assert entry["manual_reject"] is True
    assert entry["accepted"] is False
    assert len(entry["extents"]) == 3 and entry["faces"] > 0
    assert entry["up_axis_source"] == "gltf-y-up"


def test_measuring_a_hand_rejected_chair_happens_once_not_every_recurate(chairish_thresholds):
    """One GLB load ever, not one per re-curate -- these entries are never going to be accepted, so
    re-reading them on every run is pure cost. Mutation: dropping the "extents" guard re-measures.
    """
    uid = next(iter(build_chair_manifest.MANUAL_REJECTS))
    loads = []
    manifest = {"chairs": [{"uid": uid, "raw_path": "/gone.glb",
                            "extents": [1.0, 2.0, 1.0], "faces": 12}]}

    real_measure = build_chair_manifest._measure
    build_chair_manifest._measure = lambda *a, **k: (loads.append(a) or real_measure(*a, **k))
    try:
        curate(manifest, thresholds=chairish_thresholds)
    finally:
        build_chair_manifest._measure = real_measure

    assert loads == [], "re-measured a hand-rejected chair that already carries its numbers"
    assert manifest["chairs"][0]["accepted"] is False
