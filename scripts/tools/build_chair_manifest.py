#!/usr/bin/env python3
"""Pipeline that builds scripts/simvla/chair_manifest.json: fetch (Task 2), curate (Task 3),
export (Task 4), convert (Task 5) -- each adding fields to the same candidate entries -- plus a
CLI (also Task 5) that drives them from the command line.

Since 2026-08-13 this module is ALSO the engine the table library runs on
(scripts/tools/build_table_manifest.py), parameterised by FurnitureType -- see that class for what
differs per furniture type and why it is behaviour rather than constants. Two things about that,
stated here because the file's name no longer describes all of it:

  * It kept its name deliberately. scripts/simvla/test_chair_manifest.py monkeypatches
    `build_chair_manifest._flat_shade` and `build_chair_manifest.subprocess`, which pins the
    MODULE IDENTITY the stages resolve their globals through; moving the engine to a neutrally
    named file would have meant editing a test to accommodate a refactor, and on this project that
    is treated as a behaviour change rather than a rename. The chair suite passes here with no
    edit at all.
  * The chair path is unchanged, not merely equivalent. Every chair-specific number and every
    chair-specific step now lives on ChairType and is reached through the same code it always was,
    which is why the manifest on /lustre did not need regenerating: all 207 candidates were
    re-measured through the parameterised path and every recorded field compared equal.

Every stage function takes the manifest dict, mutates its "chairs" entries, and returns the same
dict -- so a caller can chain fetch -> curate -> export -> convert and save once at the end. The
CLI's `all` subcommand (see run_all) does exactly that, in a loop, until enough chairs are
accepted or the candidate pool runs out.

curate's known blind spot -- read before trusting an empty rejection bucket:
Geometry-only criteria (aspect ratio, face count) judge shape and complexity; they cannot tell a
"correct chair, wrong sub-type" from a dining chair. Task 1's hand-labeling found exactly one such
case in 40 downloaded `chair`-category candidates: 0730fabb2c8341aaaf303351f2d644c7, a pedestal-base
office/swivel chair. It passes every criterion below -- normalized width/depth well inside the
aspect-ratio band, face count well inside the floor/ceiling -- and so IS accepted here. That is a
real false accept, not a bug: the findings doc is explicit that "no geometry-only criterion in
this plan will catch this rejection." Rather than contort a threshold to exclude one known mesh
(which would just teach the criterion to overfit one example instead of generalizing), curate
records the measurements it judged on (extents, normalized_extents, faces, up_axis, facing_axis)
on every entry, accepted or not, so a human skimming the final ~20 accepted chairs has enough
per-entry detail to catch what these criteria structurally cannot.

The distinctness criterion added later (_distinctness, 2026-08-13) does NOT close that gap, and
has a measured blind spot of its own worth stating in the same breath: it detects the SAME MESH
re-uploaded under a second uid, and nothing weaker. The six chairs objaverse_chairs.md calls
near-identical "pod chairs" were measured pairwise (all 703 pairs of the 38 accepted, ten grid
resolutions) and are not a separable cluster at all -- their scores span almost the corpus's whole
range, and 608 of the 688
pairs outside the group beat the group's own weakest pair. Below exact identity this corpus is a
continuum, so "0 rejected for distinctness" says nothing about whether the library is varied.
About half of these 38 chairs are one Objaverse furniture pack; the fix for that is a wider LVIS
category and a human looking at a contact sheet, not a lower number here.

The post-scale width criterion (_export_width, 2026-08-13) closes a different gap: every other
normalized-extent criterion judges PROPORTION, which is scale-free, and a chair can be perfectly
proportioned as a model and still come out as doll furniture once export normalizes it to a
life-sized height. Two of the 38 do (0.301 m and 0.334 m across, against a corpus that is otherwise
0.374-0.728 m). It lives in CRITERIA rather than after export because the scale is known in closed
form at curate time -- see _export_footprint -- so "post-scale" describes the quantity, not the
pipeline position, and rejecting later could only retract a chair whose OBJ and USD are already on
disk. Measurements taken directly off the exported meshes.
"""
from __future__ import annotations

import argparse
import base64
import collections
import contextlib
import importlib
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

AXES = ("X", "Y", "Z")


class FurnitureType:
    """The seam this pipeline is parameterised on: everything that differs between one furniture
    library and the next. `ChairType` (below) and `build_table_manifest.TableType` are the two
    implementations; the stages themselves know nothing about chairs or tables.

    WHY THIS IS A CLASS AND NOT A DICT OF NUMBERS. The desk plan's own premise for this extraction
    was "what is chair-specific is the numbers", and the measurement gate disproved it. Two of the
    chair pipeline's rules do not merely need different constants for a table, they need different
    behaviour:

      * There is NO FACING SIGNAL for a table. Measured with the chair gate's exact definition,
        chairs separate 170x against their own noise floor and tables 9x; only 2 of 15 usable
        tables clear the chair gate's `|asym| > 0.10` confidence cut, and the median asymmetry
        (0.020) is the magnitude chairs show on the axis where there is provably nothing. So
        `_facing_axis` is not re-tuned for tables, it is ABSENT -- a table has a long axis, not a
        front, and the long axis comes free from the bounding box.
      * The HEIGHT DATUM is a different quantity. A chair is scaled so its bounding box reaches a
        uid-seeded target; a table is scaled so its WORK SURFACE reaches a fixed 0.74 m, and on 2
        of the 20 usable meshes measured that plane is not the top of the bounding box (0.65 and
        0.775 of the height). Scaling a table by its bbox would put those work surfaces at 0.48 m
        and 0.57 m.

    A dict of thresholds cannot express either. Hence: scalars for the numbers, methods for the
    steps, and a subclass per furniture type so each override sits next to the measurement that
    justifies it.

    The up-axis is the one place the plan expected a fork and the code does not need one.
    `_up_axis` already trusts the file format's declared +Y for GLB/glTF and falls back to
    argmax(extents) only where no convention is declared -- a change the chair library made in
    response to this same desk gate. It is correct for both types as it stands, and forking it
    would produce two copies of one rule.
    """

    # --- identity ---------------------------------------------------------------------------
    #: Human, singular. Used in CLI help and messages only.
    name = "furniture"
    #: The manifest key this type's candidate list lives under.
    entries_key = "furniture"
    #: Module under scripts/simvla that reads/writes the manifest (see furniture_manifest).
    manifest_module = None
    #: LVIS category the CLI fetches from when none is given.
    default_category = None

    # --- curate -----------------------------------------------------------------------------
    #: Default criteria thresholds, recorded into manifest["thresholds"] on every curate.
    thresholds: dict = {}
    #: Ordered predicates, each (measurements, thresholds, accepted) -> reason or None. curate
    #: records the FIRST failure, so the order decides which bucket a rejection lands in.
    criteria: tuple = ()
    #: uid -> reason, from a human who looked at a render. Applied before any criterion.
    manual_rejects: dict = {}

    # --- fetch / select ---------------------------------------------------------------------
    #: Default seed string for the deep sample over an LVIS category (see _sampled_uids).
    sample_seed = None
    #: Largest share of a selection any one LVIS category may supply (see select(), rule 5).
    category_cap = None
    #: LVIS categories this library is FOR, or () for "every accepted category". Applied by
    #: select() (rule 0), not by curate. See select()'s docstring for why the room the furniture
    #: is destined for is a selection question and not a curation one.
    room_categories: tuple = ()
    #: uid -> reason, for a piece that is real, whole, correctly curated furniture and still does
    #: not belong in the room this library furnishes. Applied by select() alongside
    #: room_categories. DISTINCT FROM manual_rejects, which is "not furniture, or not one object"
    #: -- see select() rule 0 and MANUAL_REJECTS' own docstring, which explicitly refuses to hold
    #: "a chair that is real but unwanted".
    room_rejects: dict = {}

    # --- contact sheet ----------------------------------------------------------------------
    #: Half-width in metres of the shared world window every cell is framed to.
    contact_sheet_half_extent_m = 0.7

    # --- behaviour --------------------------------------------------------------------------

    def measure_extras(self, mesh, up_idx) -> dict:
        """Per-mesh measurements this type records beyond the shared ones (extents, normalised
        extents, faces, up axis, shape descriptor). Chairs add a facing axis; tables add the
        work-surface plane and the overhang ratio. Recorded whatever the verdict, so a later
        threshold change can re-decide from the manifest alone."""
        return {}

    def export_geometry(self, measurements, uid):
        """(export_height_m, [smaller, larger] footprint in metres) that `export` will actually
        produce, computed at curate time from measurements alone -- or None when this type cannot
        know it. Exists so a post-scale criterion can live in curate rather than retracting an
        entry whose OBJ and USD are already on disk."""
        return None

    def grid_scale(self, mesh) -> float:
        """The divisor that makes _shape_grid scale-invariant, measured on the stood-up mesh.

        Height for a chair (the grid box is x,y in [-0.5, 0.5] and z in [0, 1], and a chair is
        taller than it is wide, so height keeps it inside). A table is up to 3.4x wider than it is
        tall, so height-normalising one would clip two thirds of its width into the border cells
        and make every table's descriptor look alike -- see TableType.grid_scale."""
        return float(mesh.bounding_box.extents[2])

    def export_scale(self, entry, mesh):
        """(target_height_m, raw_datum_extent) for the stood-up, unscaled `mesh`. export divides
        the two to get one uniform scale factor."""
        raise NotImplementedError

    def export_extras(self, mesh) -> dict:
        """Fields recorded on the entry from the FINAL exported geometry (upright, rescaled,
        floor-aligned, centred), which is a different frame from the one curate measured."""
        return {}

    def refresh_export_extras(self, entry) -> None:
        """Re-derive export_extras for an entry `export` skips because its OBJ already exists.
        Only needed for fields two stages both write in different frames."""
        return None

#: Edge count of the coarse occupancy grid _shape_grid builds -- the descriptor the distinctness
#: criterion (_distinctness) compares chairs with. One cell is 1/12 of the chair's own height,
#: about 7 cm at a 0.85 m export height: coarser than a chair leg, so the descriptor stays a
#: silhouette rather than becoming a mesh diff.
#:
#: The floor on this number is measured, not guessed (2026-08-13-chair-distinctness-findings.md,
#: "The threshold, and why 0.98", resolution sweep table): at N=6 SIX pairs of visibly different
#: chairs collide at IoU 1.0000, while at every N from 8 to 28 exactly the two identical-geometry
#: pairs do. 12 sits above that floor with margin, at 11 ms per mesh against a curate stage that
#: already pays a GLB load per candidate.
#:
#: Changing this does NOT invalidate duplicate_iou_max: that threshold was deliberately set
#: against the worst case over the whole N=8..28 sweep, not against the band at N=12. It DOES
#: invalidate every shape_grid already recorded in a manifest -- _distinctness compares only
#: descriptors whose shape_grid_n matches, and silently declines to compare across a mismatch, so
#: a manifest curated at one N and re-curated at another loses duplicate detection against the
#: older entries until they are re-measured.
SHAPE_GRID_N = 12

#: Mean chair height (floor to the top of the backrest) the per-chair Gaussian draw (see
#: _sample_height) is centred on. Objaverse carries no units, so there is no "native" scale to
#: preserve -- this is a fixed centre inside a normal dining-chair's real-world height range (a
#: plain seat-height chair without a tall backrest runs closer to 0.75-0.85 m; one with a tall
#: backrest runs closer to 0.95-1.05 m), and inside the brief's own test band [0.75, 1.0] m.
TARGET_HEIGHT_M = 0.85

#: Standard deviation for the per-chair height draw. Real dining chair back-heights run roughly
#: 0.75-1.05 m -- a ~0.30 m real-world spread. Treating that as a roughly 3-sigma-each-side band
#: around TARGET_HEIGHT_M gives sigma = 0.30 / 6 = 0.05 m: tight enough that every export is
#: still unmistakably chair-sized, loose enough that twenty exports are visibly not identical
#: (measured over 2000 synthetic uids: mean 0.850 m, stdev 0.050 m as intended, ~7 cm between the
#: 25th/75th percentile draws). Chosen from furniture dimensions, not from the mesh corpus -- the
#: corpus has no units, which is why a constant was needed here in the first place.
HEIGHT_SIGMA_M = 0.05

#: Hard clip band for the sampled height. A Gaussian's tails are unbounded -- without this, a
#: rare draw could scale a chair to something absurd (0.4 m, 1.5 m). Set to exactly the band
#: scripts/simvla/test_chair_manifest.py has always asserted on export's output
#: (0.75 <= height <= 1.0), so clipping here is honoring a promise the tests already made, not
#: adding a new one.
HEIGHT_CLIP_MIN_M = 0.75
HEIGHT_CLIP_MAX_M = 1.0


def _sample_height(uid, *, mean=TARGET_HEIGHT_M, sigma=HEIGHT_SIGMA_M,
                    lo=HEIGHT_CLIP_MIN_M, hi=HEIGHT_CLIP_MAX_M):
    """Deterministic per-chair target height: draw once from N(mean, sigma) seeded from `uid`,
    then clip to [lo, hi]. Same uid -> same height, every run, in every process.

    Seeded from the uid STRING itself -- not from hash(uid), and not from a tuple wrapping it.
    Python's builtin hash() of a str is salted per-process by PYTHONHASHSEED (verified: hash("x")
    differs across two separate `python -c` invocations unless PYTHONHASHSEED is pinned), and
    random.Random((uid,)) -- a tuple -- falls through to that same salted hash() internally, so
    it is exactly as unstable; that is the trap this project already hit once. random.Random(uid)
    with a str/bytes argument instead seeds from a SHA-512 digest of the string's bytes, which
    involves no process-specific salt at all. Verified empirically: `random.Random(some_uid)
    .gauss(...)` was run as five separate `python -c` process invocations -- PYTHONHASHSEED
    unset, 0, 12345, and "random" (twice, to catch two different random salts) -- and all five
    produced the bit-identical float, while the same processes' hash(some_uid) differed across
    every PYTHONHASHSEED value. Without this, export's recorded height_m would describe only the
    most recent run, not the corpus.
    """
    rng = random.Random(uid)
    return min(max(rng.gauss(mean, sigma), lo), hi)


#: Default seed string for _sampled_uids, and the one the fifty-chair library was drawn with
#: (2026-08-13). A STRING, not an int, for the same reason _sample_height seeds from the uid
#: string: random.Random(str) seeds from a SHA-512 of the bytes with no per-process salt, so the
#: draw is reproducible across interpreters without pinning PYTHONHASHSEED.
SAMPLE_SEED = "fifty-chairs-2026-08-13"


def _sampled_uids(uids, category, seed=SAMPLE_SEED):
    """A reproducible deep-sampling order over an LVIS category's uid list.

    fetch() without a seed takes the HEAD of the category list, and on `chair` that head is one
    Objaverse furniture pack: the first 40 uids drawn that way produced a library that is roughly
    20 shell-back/cantilever chairs out of 38 accepted
    (2026-08-13-chair-distinctness-findings.md, "And the real archetype is much larger than six").
    Drawing the next batch the same way draws more of the same pack, which is why the fifty-chairs
    plan's second amendment calls variety a SOURCING problem rather than a filtering one.

    So: sort the pool (the annotation file's own ordering must not decide which chairs the library
    gets -- sorting makes the draw a function of the uid SET alone), then permute it with a
    seeded RNG. Uniform over the whole category, so the head has no privilege: on `chair`, the 40
    uids already fetched from the head occupy 40/453 = 8.8% of the pool and are expected to
    contribute that share of any batch, instead of all of it.

    Deterministic, and stable across processes: same seed + same category + same uid set -> same
    order, forever. Seeded per-category so that adding a category later does not re-order the
    draws of the ones already fetched.
    """
    pool = sorted(set(uids))
    random.Random(f"{seed}:{category}").shuffle(pool)
    return pool


def _redirect_objaverse_cache(objaverse):
    """Point objaverse's download cache at $OBJAVERSE_PATH when it is set. No-op otherwise.

    objaverse 0.1.7 hard-codes its cache to ~/.objaverse with no argument and no environment
    variable of its own -- BASE_PATH and _VERSIONED_PATH are module globals, read inside each
    function at call time, which is why assigning to them here works at all.

    This matters operationally rather than aesthetically: on this cluster /home is a 200 GB shared
    quota that was 98% full when the fifty-chair pool was drawn, with 4.8 GB free, and 167 GLBs
    at this corpus's ~4 MB average is most of that. A fetch that fills a shared home directory
    breaks more than this pipeline. /lustre has 575 TB free.

    Nothing about an already-fetched entry changes: fetch skips uids the manifest already has, so
    a moved cache never re-downloads, and each entry's recorded raw_path keeps pointing wherever
    it was downloaded to at the time. Mixed roots across a manifest are expected and fine.
    """
    base = os.environ.get("OBJAVERSE_PATH")
    if not base:
        return
    objaverse.BASE_PATH = base
    objaverse._VERSIONED_PATH = os.path.join(base, "hf-objaverse-v1")


def _library(spec):
    """The scripts/simvla module that reads and writes `spec`'s manifest.

    Imported lazily, and with the same sys.path insertion every stage used to do inline: this
    module lives in scripts/tools and its sibling manifest modules live in scripts/simvla, which is
    not a package.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    return importlib.import_module(spec.manifest_module)


def _entries(manifest, spec):
    """This furniture type's candidate list inside `manifest` -- "chairs", "tables", ..."""
    return manifest.get(spec.entries_key, [])


def fetch(manifest, category, count, *, download=None, uids=None, seed=None, spec=None):
    """Add up to `count` NEW candidates from an LVIS `category` to `manifest`. Returns `manifest`.

    Skips uids already present in manifest["chairs"] -- at whatever stage they've reached -- so
    a re-run only downloads what's new. That matters because a campaign grows past its first
    `count` over multiple runs, and Objaverse downloads are slow and rate-limited enough that
    re-fetching a uid this manifest already has would be a real cost, not just wasted work.

    `download` and `uids` are both injectable and default to the real Objaverse calls --
    objaverse.load_objects and objaverse.load_lvis_annotations()[category] -- imported LAZILY,
    here inside the function, so importing this module never requires the objaverse package, and
    a test can inject fakes for both and never touch the network.

    `seed`, when given, replaces "take the head of the category list" with a reproducible deep
    sample over the whole category -- see _sampled_uids for why the head is a bad draw. Default
    None keeps the original head-order behaviour, so a manifest built before this existed can
    still be extended exactly as it was.

    Adds only uid/category/raw_path to each new entry -- no verdict. Judging candidates is
    curate's job (Task 3): an entry `fetch` produces never carries "accepted".
    """
    spec = spec or CHAIR
    if uids is None or download is None:
        import objaverse
        _redirect_objaverse_cache(objaverse)
        if uids is None:
            uids = objaverse.load_lvis_annotations()[category]
        if download is None:
            download = objaverse.load_objects

    if seed is not None:
        uids = _sampled_uids(uids, category, seed)

    chairs = manifest.setdefault(spec.entries_key, [])
    known = {e["uid"] for e in chairs}
    new_uids = [uid for uid in uids if uid not in known][:count]
    if not new_uids:
        return manifest

    paths = download(new_uids, download_processes=1)
    for uid in new_uids:
        chairs.append({
            "uid": uid,
            "category": category,
            "raw_path": str(paths[uid]),
        })
    return manifest


def _shape_grid(mesh, up_idx, n=SHAPE_GRID_N, *, spec=None):
    """A coarse occupancy grid of `mesh` in the canonical chair pose: the shape descriptor the
    distinctness criterion compares. Returns an (n, n, n) bool array over the box
    x, y in [-0.5, 0.5], z in [0, 1].

    The pose is the one the `export` stage produces, so the descriptor describes the chair that
    actually ships: up-axis rotated onto +Z (_stand_up), uniform-scaled to height 1, centred in
    XY, base at z = 0. Scale-invariant by construction, which matters because Objaverse carries
    no units at all -- raw extents in this corpus span four orders of magnitude, so two uploads of
    one asset can differ by a factor of a thousand and still be the same chair. Height is what
    normalises, not the largest extent per axis: normalising each axis independently would erase
    proportion, which is half the shape signal. x and y stay inside [-0.5, 0.5] for free, since
    curate defines up_axis as argmax(extents).

    Occupancy comes from SUBDIVIDING each triangle until no edge exceeds half a cell and then
    marking each resulting vertex's cell -- deliberately not from sampling points on the surface.
    Both were measured (findings doc, "Determinism"): they agree to within 0.0545 max / 0.0084
    mean IoU over all 703 pairs, but sampling needs a seed, and a seeded number recorded into a
    manifest is the exact reproducibility trap _sample_height's docstring above describes at
    length. Subdivision has no RNG in it. Verified: the SHA-256 of all 40 candidates' packed grids
    is identical across three interpreter processes with PYTHONHASHSEED unset, 0 and "random".

    WHAT NORMALISES is `spec.grid_scale` -- height for a chair, exactly as this always did, and the
    largest extent for a table, which is wider than it is tall and would otherwise have two thirds
    of its width clipped into the border cells.
    """
    spec = spec or CHAIR
    mesh = mesh.copy()
    _stand_up(mesh, up_idx)
    height = float(spec.grid_scale(mesh))
    if height <= 0:
        return np.zeros((n, n, n), dtype=bool)
    mesh.apply_scale(1.0 / height)
    bounds = mesh.bounds
    mesh.apply_translation([
        -(bounds[0][0] + bounds[1][0]) / 2,
        -(bounds[0][1] + bounds[1][1]) / 2,
        -bounds[0][2],
    ])

    vertices, _ = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.5 / n)
    grid = np.zeros((n, n, n), dtype=bool)
    grid[
        np.clip(((vertices[:, 0] + 0.5) * n).astype(int), 0, n - 1),
        np.clip(((vertices[:, 1] + 0.5) * n).astype(int), 0, n - 1),
        np.clip((vertices[:, 2] * n).astype(int), 0, n - 1),
    ] = True
    return grid


def _encode_grid(grid) -> str:
    """A _shape_grid as base64 of its packed bits -- 288 characters at n = 12, small enough to
    record on every manifest entry so a later threshold change can re-decide distinctness from the
    manifest alone, without re-loading a single mesh (the same promise curate's docstring already
    makes for extents and faces)."""
    return base64.b64encode(np.packbits(grid).tobytes()).decode("ascii")


def _decode_grid(encoded: str, n: int):
    return np.unpackbits(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
                         count=n ** 3).astype(bool).reshape(n, n, n)


def _shape_iou(a_encoded: str, b_encoded: str, n: int) -> float:
    """Intersection-over-union of two _shape_grids, maximised over the 8 symmetries of the square
    in the XY plane (four 90-degree yaws x mirror).

    The maximisation is not optional: Objaverse uploads carry no shared facing convention, and
    _stand_up only fixes which axis is up, never the yaw about it -- without this a duplicate
    uploaded a quarter-turn round would score near zero. The group is limited to multiples of
    90 degrees because those are exactly the rotations that map this axis-aligned grid onto
    itself; a duplicate rotated by an arbitrary angle is NOT caught, and the findings doc measures
    how fast that falls off (5 degrees of yaw already drops an identical mesh from 1.0000 to
    0.8042, below the 0.98 threshold).
    """
    return _grid_iou(_decode_grid(a_encoded, n), _decode_grid(b_encoded, n))


def _grid_iou(a, b) -> float:
    """_shape_iou on already-decoded grids. Split out for select(), which scores every pair of a
    pool of ~200 chairs and would otherwise base64-decode the same descriptor 200 times over."""
    best = 0.0
    for k in range(4):
        rotated = np.rot90(a, k=k, axes=(0, 1))
        for variant in (rotated, rotated[::-1, :, :]):
            union = np.logical_or(variant, b).sum()
            if union:
                best = max(best, float(np.logical_and(variant, b).sum()) / float(union))
    return best


def _export_footprint(normalized_extents, up_idx, uid):
    """The two horizontal extents, in METRES, the `export` stage will actually produce for this
    chair -- returned smaller-first. Requires no mesh: it is exact arithmetic on what curate has
    already measured.

    Every existing normalized-extent criterion judges a chair BEFORE scaling. This one is about
    what ships, which is after. It can still be answered at curate time because the scale is known
    in closed form:

        exported_extent[i] = normalized_extents[i] / normalized_extents[up_axis]
                             * _sample_height(uid)                            (i != up_axis)

    -- because _stand_up rotates by a cardinal 90 degrees (or the identity), so axis-aligned
    extents merely permute; and because export scales UNIFORMLY, so every extent is multiplied by
    the same target_height / raw_height, and raw_height is the extent along up_axis.

    The division by normalized_extents[up_axis] is NOT redundant. It used to be: while up_axis was
    argmax(extents), extents.max() was the height by construction and normalized_extents[up_axis]
    was identically 1.0. Since _up_axis started trusting the file's declared Y-up, a mesh longer
    than it is tall normalises to less than 1.0 on its own up axis, and dropping the division would
    under-predict every such footprint -- by 39% on the worst chair in this pool.

    Verified, not assumed: predicted against all 38 exported OBJs read back off disk, max
    horizontal error 9.5e-09 m (2026-08-13-chair-export-width-findings.md, "The quantity, and why
    it can be known at curate time"). Ten nanometres, against a bound with 24 mm of margin.

    The height used is the chair's OWN uid-seeded draw, not the nominal TARGET_HEIGHT_M -- see
    _export_width for why, and for the coupling that creates.
    """
    height = _sample_height(uid)
    up_extent = normalized_extents[up_idx]
    return height, sorted(normalized_extents[i] / up_extent * height
                          for i in range(3) if i != up_idx)


#: File suffixes whose format defines +Y as up. glTF says so normatively ("the 3D scene uses a
#: right-handed coordinate system ... +Y is up"), and Objaverse is entirely GLB.
Y_UP_SUFFIXES = (".glb", ".gltf")


def _up_axis(mesh, raw_path):
    """Which axis of `mesh` is up: the FILE FORMAT's answer where it has one, measurement only as a
    fallback. Returns (up_idx, source).

    This replaced `argmax(extents)`, and the replacement is measured, not preferred. The desk gate
    rendered 60 meshes and checked both rules against what it could see: glTF Y-up was correct on
    20/20 of the sample it
    hand-labelled, and argmax picked Y on 0/20. This library then shipped the same defect in the
    other direction -- the contact sheet found 8 of 50 offered chairs exported lying down, every
    one of them a lounger or rocker whose longest axis is along the floor, which is precisely the
    case argmax cannot get right because it assumes furniture is tallest.

    argmax is kept ONLY for a source with no declared convention (a bare OBJ or STL), where a
    guess is all there is. It is not a vote and not a tie-break: where the format declares up, the
    declaration wins outright, because a mesh can be genuinely longer than it is tall and that is
    a fact about the object, not an error to correct.

    WHAT THIS QUIETLY REPAIRS BESIDES ORIENTATION. Under argmax the up axis WAS the longest axis by
    construction, so normalized_extents[up] was identically 1.0 and every horizontal was <= 1.0 --
    which means _aspect_ratio could never once observe "this object is longer than it is tall". It
    was structurally blind to loungers, beds, railings and billboards lying in the plane. With a
    file-declared up axis, a mesh longer than it is tall now normalises to a horizontal extent of
    1.0 and _aspect_ratio rejects it against the aspect_max it has always had. No new threshold:
    the criterion finally measures the quantity its findings line always described.
    """
    if str(raw_path).lower().endswith(Y_UP_SUFFIXES):
        return 1, "gltf-y-up"
    return int(np.argmax(mesh.bounding_box.extents)), "argmax-extents"


def _measure(raw_path, uid=None, *, spec=None):
    """Load one candidate and compute the fields curate judges and records.

    `uid` is optional only so that a caller with a bare path (a test, the findings docs' repro
    snippets) can still measure geometry. When it is given, the post-scale fields _export_width
    judges -- export_height_m and export_footprint_m -- are computed too; they depend on the uid
    because the export height is drawn per-chair from it. curate always passes it.

    Mirrors Task 1's measurement method exactly (findings doc, "Method", Step 3): trimesh.load()
    returns a Scene for effectively every GLB -- even a structurally single-part one -- so a
    Scene is always merged via to_geometry() (baking each sub-part's scene-graph transform into
    one Trimesh) BEFORE any extent is measured. This is the findings doc's proposed threshold #3,
    "multi-body handling: merge, not reject": a multi-body candidate's per-node offsets/rotations
    would make per-geometry extents meaningless, and 4 of the 5 multi-body scenes in the Task 1
    sample were good chairs once merged -- rejecting on multi-body-ness, or on the merge not
    being watertight (0/5 multi-body merges were), would have thrown those away for no reason
    tied to shape quality.

    Which EXTRA fields are recorded is `spec`'s decision, not this function's -- a chair records a
    facing axis and a table records its work-surface plane and overhang ratio. See FurnitureType.
    """
    spec = spec or CHAIR
    loaded = trimesh.load(raw_path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded

    extents = mesh.bounding_box.extents
    up_axis, up_axis_source = _up_axis(mesh, raw_path)
    # Still normalised by the LARGEST extent, not by the up extent. That keeps every existing
    # threshold measuring exactly what its findings line measured for an upright chair (whose
    # largest extent IS its height, so nothing moves), while giving a mesh that is longer than it
    # is tall a horizontal extent of 1.0 for _aspect_ratio to reject. Normalising by the up extent
    # instead would push those horizontals above 1.0 and silently rescale the whole band.
    normalized = extents / extents.max()

    measurements = {
        "extents": [float(v) for v in extents],
        "normalized_extents": [float(v) for v in normalized],
        "faces": int(len(mesh.faces)),
        "up_axis": AXES[up_axis],
        "up_axis_source": up_axis_source,
        # Splatted here, rather than update()d afterwards, so a chair entry's key order is
        # byte-for-byte what it has always been: facing_axis between up_axis_source and
        # shape_grid_n.
        **spec.measure_extras(mesh, up_axis),
        "shape_grid_n": SHAPE_GRID_N,
        "shape_grid": _encode_grid(_shape_grid(mesh, up_axis, spec=spec)),
    }
    if uid is not None:
        geometry = spec.export_geometry(measurements, uid)
        if geometry is not None:
            height, footprint = geometry
            measurements["export_height_m"] = float(height)
            measurements["export_footprint_m"] = [float(v) for v in footprint]
    return measurements


def _aspect_ratio(measurements, thresholds, accepted=()):
    """Reject if either horizontal (non-up-axis) normalized extent falls outside
    [thresholds["aspect_min"], thresholds["aspect_max"]].

    `accepted` is ignored here -- this is a per-mesh criterion. It is in the signature because
    CRITERIA is applied uniformly and one of its members (_distinctness) is pairwise; see the
    comment above CRITERIA.

    Findings doc, proposed threshold #1: bounds [0.20, 0.95] sit outside the entire observed
    cluster from 40 real chairs ([0.308, 0.811] width, [0.350, 0.788] depth), with margin for
    styles not present in that sample. The low bound catches poles/sticks; the high bound catches
    flat or near-cubic slabs.
    """
    lo, hi = thresholds["aspect_min"], thresholds["aspect_max"]
    normalized = measurements["normalized_extents"]
    up_idx = AXES.index(measurements["up_axis"])
    for i, value in enumerate(normalized):
        if i == up_idx:
            continue
        if value < lo or value > hi:
            return (
                f"aspect ratio: normalized extent {value:.3f} on axis {AXES[i]} "
                f"outside [{lo}, {hi}]"
            )
    return None


def _face_count(measurements, thresholds, accepted=()):
    """Reject if face count falls outside [thresholds["face_floor"], thresholds["face_ceiling"]].

    `accepted` is ignored -- per-mesh criterion, see _aspect_ratio.

    Findings doc, proposed threshold #2. Floor 100: the lowest observed face count in the 40-mesh
    sample (160) rendered as a clean, recognizable chair; 100 leaves a small margin below that
    observed working minimum. Ceiling 200,000 is explicitly a compute-cost guard, NOT a validated
    quality signal: two meshes with 390,167 and 49,885 faces were confirmed good chairs (just
    over-tessellated); only the single 782,996-face mesh in the sample was bad, and for reasons
    independent of its face count (wrong up-axis, 33 scene fragments, not a dining chair). A
    face-count rejection here means "too expensive to convert," not "not a chair."
    """
    n = measurements["faces"]
    floor, ceiling = thresholds["face_floor"], thresholds["face_ceiling"]
    if n < floor:
        return f"face count: {n} faces below floor {floor}"
    if n > ceiling:
        return f"face count: {n} faces above ceiling {ceiling}"
    return None


def _export_width(measurements, thresholds, accepted=()):
    """Reject if the chair's LARGER horizontal extent, after export's uniform height scaling,
    would be below thresholds["export_width_min_m"] metres -- i.e. it is not sittable furniture at
    the size it ships at, whatever it looks like as a model.

    `accepted` is ignored -- per-mesh criterion, see _aspect_ratio.

    This is the only criterion here that judges the chair AFTER scaling. That does not put it at a
    different pipeline stage: the scale is known in closed form at curate time (see
    _export_footprint), so "post-scale" describes the QUANTITY, not the position. It must be a
    curate criterion, because a verdict reached any later could only RETRACT an already-accepted
    chair -- stranding an OBJ and a USD on disk that something may already reference, which
    curate's ordering rule 4 refuses to do -- and because rejected_because/rejection_histogram are
    curate's channel and nothing else's.

    Read 2026-08-13-chair-export-width-findings.md before widening this, and in particular before
    assuming it means more than it does:

    * It bounds the LARGER horizontal extent, i.e. it fires only when a chair is below the bound on
      BOTH axes. Nothing bounds the smaller one: a hypothetical 0.20 x 0.55 m chair passes. No such
      chair exists in the 38 measured (both series break after the same two chairs), and adding a
      second bound nothing has measured is what this project's ledger forbids. Both numbers are
      recorded per entry so a later measurement can add one without re-loading a mesh.
    * It is a BOUNDING BOX, not a seat. For an armchair it includes the arms; for a cantilever
      chair it includes the base sled. A wide back over a narrow seat passes.

    THE HEIGHT IS THE CHAIR'S OWN DRAW, NOT THE NOMINAL 0.85 m. That is deliberate, and it is not a
    source of non-determinism: _sample_height seeds from the uid string via a SHA-512 with no
    per-process salt, so one uid has exactly one draw, forever (see its docstring, and the
    cross-process test that guards it). What it IS is a coupling -- changing TARGET_HEIGHT_M,
    HEIGHT_SIGMA_M or the clip band re-decides every width verdict, exactly as changing
    SHAPE_GRID_N invalidates every recorded shape_grid. The alternative was measured and is worse:
    the clip band is +-13% around the mean, which moves exported width by up to 0.046 m at this
    bound -- WIDER than the 0.040 m gap the bound sits in -- and five of the 38 accepted chairs
    have a verdict that would flip somewhere inside their own clip band. Judging the nominal height
    would systematically under-detect exactly the chairs at risk, the ones with a low draw. curate
    records export_height_m and export_footprint_m on every judged entry so any verdict can be
    audited, and re-decided, from the manifest alone.

    Fails OPEN when the entry carries no footprint (measured by a caller that passed no uid),
    rather than guessing a height -- same choice _distinctness makes for a missing descriptor.
    """
    footprint = measurements.get("export_footprint_m")
    if not footprint:
        return None

    minimum = thresholds["export_width_min_m"]
    narrower, wider = min(footprint), max(footprint)
    if wider < minimum:
        return (
            f"export width: {wider:.3f} m x {narrower:.3f} m footprint at its "
            f"{measurements['export_height_m']:.3f} m export height -- larger horizontal extent "
            f"below {minimum} m"
        )
    return None


def _distinctness(measurements, thresholds, accepted=()):
    """Reject if this candidate's shape descriptor matches an ALREADY-ACCEPTED chair's at an IoU
    above thresholds["duplicate_iou_max"] -- i.e. it is the same mesh re-uploaded to Objaverse
    under a second uid. Names the uid it duplicates, because "rejected" is not actionable and
    "too similar to <uid>" is: it tells a human which chair they still have.

    Read the findings doc (2026-08-13-chair-distinctness-findings.md) before widening this. In
    particular, read it before assuming it does more than it does:

    * This is a RE-UPLOAD detector, not a design-similarity detector. On the 38 accepted chairs it
      fires on exactly two pairs, and both are byte-identical geometry (same SHA-256 of canonical
      vertices, same face count, indistinguishable renders, different GLB bytes).
    * The six chairs objaverse_chairs.md calls near-identical "pod chairs" are NOT a measurable
      cluster: their 15 pairwise scores span 0.160-0.862, and 608 of the 688 pairs OUTSIDE that
      group score higher than its weakest member. There is no threshold that selects them. The
      findings doc says this plainly rather than splitting the continuum somewhere flattering.
    * So "0 rejected for distinctness" does not mean the library is varied. Roughly half of these
      38 are one Objaverse furniture pack. The fix for that is a wider LVIS category, and the
      thing that will show it is the contact sheet, not this number.

    `accepted` is the entries curate has accepted so far, and the ORDER OF JUDGING therefore
    decides which member of a duplicate pair survives -- unlike every other criterion here, whose
    verdict depends on nothing but its own mesh. curate() judges in uid order for exactly this
    reason; see its docstring for the whole rule. This function itself is order-independent: it
    reports the incumbent with the HIGHEST IoU, breaking an exact tie on the smallest uid, so the
    reason string is a function of the candidate and the accepted SET, not of iteration order.

    Incumbents whose shape_grid_n differs from this candidate's (a manifest curated under a
    different SHAPE_GRID_N, or before this criterion existed and so carrying no descriptor at all)
    are skipped rather than compared at a mismatched resolution. That fails OPEN -- a duplicate of
    such an entry is accepted, not rejected. Re-curating from scratch is what fixes it.
    """
    encoded = measurements.get("shape_grid")
    n = measurements.get("shape_grid_n")
    if not encoded or not n:
        return None

    limit = thresholds["duplicate_iou_max"]
    matches = []
    for other in accepted:
        if other.get("shape_grid_n") != n or not other.get("shape_grid"):
            continue
        value = _shape_iou(encoded, other["shape_grid"], n)
        if value > limit:
            matches.append((value, other["uid"]))
    if not matches:
        return None

    value, uid = min(matches, key=lambda match: (-match[0], match[1]))
    return f"distinctness: too similar to {uid} (shape IoU {value:.4f} above {limit})"


# One predicate per criterion, each returning a reason string or None. curate() applies them IN
# THIS ORDER and records only the first failure, so the rejection histogram stays meaningful --
# "31 rejected for aspect ratio, 12 for face count" is actionable; "43 rejected" is not. Does NOT
# include an up-axis or facing-axis criterion: the findings doc frames up-axis detection
# reliability (97.5%) and facing-axis confidence (90%) as properties of the MEASUREMENT itself,
# not as proposed reject thresholds with a number attached -- it explicitly says low facing
# confidence must NOT be read as "reject." Both are still measured and recorded on every entry
# (see _measure) for a human to read, just not judged on here.
#
# Every predicate takes (measurements, thresholds, accepted) even though only _distinctness reads
# the third argument, so curate() can apply them all through one call shape. _distinctness is LAST
# on purpose, and the order is part of the design, not incidental:
#   * A mesh that is both a duplicate AND over the face ceiling should be reported as face count.
#     That reason is a property of the mesh alone and stays true however the corpus changes; "too
#     similar to <uid>" is only true relative to what else was accepted that day.
#   * Only a mesh that already passed every per-mesh criterion is worth asking "is it distinct?"
#     about -- distinctness from a corpus of good chairs is the question, and a candidate that is
#     not a usable chair at all should never become the incumbent that shadows a later one.
#
# _export_width (2026-08-13) is per-mesh too, and goes THIRD: after the two pre-existing per-mesh
# criteria so that no existing rejection changes bucket -- the pole fixture box(0.1, 0.1, 2.0)
# fails both aspect ratio (normalized 0.05) and width (0.043 m exported), and "aspect ratio" is the
# better reason because it says the thing is not a chair at all rather than that it is a small one
# -- and before _distinctness for the same reason face count is, since a chair that will not ship
# at a sittable size must never become the incumbent that shadows a later, better one.
CRITERIA = [_aspect_ratio, _face_count, _export_width, _distinctness]

#: The single source of truth for curate's thresholds. From the chair-findings measurement notes,
#: "Proposed thresholds":
#: 1. aspect ratio bounds [0.20, 0.95] -- observed range across 40 real chairs was [0.308, 0.811]
#:    (width) / [0.350, 0.788] (depth); these bounds sit outside that whole cluster with margin.
#: 2. face count [100, 200_000] -- floor 100 sits just below the lowest observed working face
#:    count (160); ceiling 200_000 is a compute-cost guard, not a validated quality signal.
#: Before this constant existed, these four numbers lived only in a test fixture and in this
#: docstring's prose -- two places that could silently drift apart. curate() defaults to this
#: dict when no thresholds are passed, and the CLI's `curate`/`all` subcommands use it too, so
#: there is exactly one copy; test_chair_manifest.py's chairish_thresholds fixture imports it
#: rather than re-transcribing the numbers.
#:
#: 3. duplicate_iou_max 0.98 -- the distinctness criterion (_distinctness), from the
#:    chair-distinctness measurement notes, "The threshold, and why 0.98", quoted verbatim:
#:
#:      "The highest shape IoU measured between any two non-identical accepted chairs, at any
#:       grid resolution from N = 8 to N = 28, is 0.9737 (19b2c1893f08 / 9173450a6303, at N = 8).
#:       The two identical-geometry pairs sit at exactly 1.0000 at every resolution. The interval
#:       (0.9737, 1.0000) contains no measured pair at any resolution tested."
#:
#:    0.98 sits inside that empty interval, 0.0263 above the worst-case non-identical pair and
#:    0.02 below identity. Set against the WORST case over the whole resolution sweep rather than
#:    the band at the shipped SHAPE_GRID_N, because the band's width is not monotone in N and
#:    swings between 0.026 and 0.264 across the sweep -- choosing the resolution where the band
#:    happened to be widest would be manufacturing a gap, which the chair project's ledger
#:    forbids. Two things this number is NOT: it is not tuned to reach a chair count (the fifty is
#:    reached by fetching more, never by keeping duplicates), and it is not a "similar design"
#:    bound -- the findings doc shows the corpus is a continuum below 1.0 with no gap to cut at,
#:    so nothing lower than this would be measured, only chosen.
#:
#: 4. export_width_min_m 0.35 -- the post-scale width criterion (_export_width), from the
#:    chair-export-width measurement notes, "The threshold, and why 0.35", quoted verbatim:
#:
#:      "The two narrowest exported chairs of the 38 have larger horizontal extents of 0.301 m and
#:       0.334 m. The next-narrowest is 0.374 m. The interval (0.334, 0.374) contains no measured
#:       chair, and it is 2.1x wider than the largest gap anywhere else below 0.50 m in that
#:       series (0.019 m, at 0.411 -> 0.430)."
#:
#:    0.35 sits inside that empty interval, 0.016 m above the widest rejected chair and 0.024 m
#:    below the narrowest kept one -- placed slightly below the interval's midpoint (0.354) on
#:    purpose, because inside a gap the lower number is the more permissive one and a criterion
#:    that throws chairs away should err toward keeping. It rejects 2 of the 38 accepted, one of
#:    them in the offered twenty.
#:
#:    It is deliberately NOT the 0.40-0.55 m figure real dining chairs are quoted at, which the
#:    task brief names as the folk number to avoid. 0.40 falls in the middle of the dense band:
#:    it would reject four chairs (d2d5000b1e7a, 19b2c1893f08, f341308cee4a, 9173450a6303) that
#:    render as ordinary armchairs, and the quoted figure is about SEAT width while this measures
#:    an axis-aligned bounding box around a whole mesh -- not the same quantity, so it cannot be
#:    transplanted. The renders and the numbers break in the same place, and that place is 0.35.
DEFAULT_THRESHOLDS = {
    "aspect_min": 0.20,
    "aspect_max": 0.95,
    "face_floor": 100,
    "face_ceiling": 200_000,
    "duplicate_iou_max": 0.98,
    "export_width_min_m": 0.35,
}

#: Largest share of a selection any single LVIS category may supply (see select(), rule 5).
#:
#: 0.24 -- 12 chairs of 50 -- chosen from the measured harm, not from a balance ideal. The library
#: the contact sheet rejected had deck_chair supplying 13 of 50, a 26% share drawn from a 24-uid
#: list whose tail is junk, and 10 of those 13 were not chairs. 0.24 sits strictly below that
#: share. It is NOT the even split (six categories would be 16.7%): an even split is a claim that
#: the categories are equally good, and the measured accept yields say plainly that they are not --
#: chair 80%, folding_chair 75%, rocking_chair 70%, stool 68%, armchair 40%, deck_chair 5%.
#: Capping the best-yielding category to promote the worst is anti-correlated with quality, which
#: is the failure mode this whole task exists to undo, only pointed the other way.
#:
#: Two numbers checked and rejected, so the choice is visible rather than asserted:
#:   * 0.20 (10/50) forces the selection to exactly 10+10+10+10+9+1 = 50 out of this pool -- it
#:     binds with ZERO slack, so it would be doing the work of reaching the count, which this
#:     project's ledger forbids. It also drags in 10 armchairs, the worst-yielding category still
#:     in the pool, and its worst selected pair jumps to 0.6048 IoU.
#:   * 0.16 (8/50) cannot reach 50 at all from this pool: 41.
#:
#: What 0.24 costs, measured over the selection's 1225 pairs: mean shape IoU 0.2009 against 0.2047
#: uncapped (slightly BETTER -- the cap spends picks on other categories rather than deeper into
#: one), worst pair 0.5000 against 0.4465. So the cost is entirely in the worst pair, about 0.05
#: IoU, and it buys a largest-category share of 24% instead of 34%. Reachable with slack: the
#: capped ceiling for this pool is 12+12+12+12+9+1 = 58 against a target of 50.
DEFAULT_CATEGORY_CAP = 0.24

#: Chairs a human looked at and rejected, uid -> reason. Applied by curate() BEFORE its
#: already-judged skip, so a re-curate cannot resurrect one and a previously-accepted entry is
#: retracted rather than left standing.
#:
#: WHY THIS EXISTS AS DATA RATHER THAN A CRITERION. Every entry below was found by rendering the
#: library and looking at it (the contact sheet, 2026-08-13; task-6 report). Some are mechanically
#: detectable and now ARE detected -- the flat billboards by _planarity, the exploded multi-part
#: meshes by _components -- and they stay listed anyway, because a criterion that happens to catch
#: a mesh today is not a promise about the mesh, it is a promise about the threshold. The rest are
#: not detectable by any geometry at all: "a licence-text placard welded into the same mesh as the
#: chair" and "a wrought-iron railing section" are correct, well-formed, single-component,
#: chair-proportioned solids. No number separates them from furniture. A human pass is the only
#: gate that sees them, and its result has to outlive the next rebuild or the next selection picks
#: them straight back.
#:
#: THIS IS NOT A PLACE TO PUT A CHAIR THAT IS MERELY INCONVENIENT. Everything here is "not a chair,
#: or not one object". A chair that is real but unwanted (an office chair on castors, a deck chair)
#: is a decision about what the library is FOR, and belongs in a criterion or in the brief -- not
#: in a hand-maintained deny-list, which no future reader can audit against a rule.
MANUAL_REJECTS = {
    # --- not furniture at all: nothing in these meshes is a chair from any angle ---------------
    "40e2289362014f6196998cb4d9f67099": "not a chair: a flat billboard plane with a sketch on it",
    "12f1e1394b2241f7b05d2d9bac695cc3": "not a chair: a flat billboard plane and a thin post",
    "39059edc24d547f393c91856606e25da":
        "not a chair: a licence-text placard welded into the mesh, plus an inverted blob",
    "844c8bc92944471ca525e78dc97da253":
        "not a chair: a hinged panel -- hinge barrels and bolt bosses, a door or machine hatch",
    "7edddfba4b51475f9438cfd8043ded86":
        "not a chair: a wrought-iron railing or gate section with a scrolled bracket",
    "b306e280c35540f4bfa844631f596d07": "not a chair: a loft bed -- frame, side ladder, platform",
    "543c8e12a8ec41548eb182960df772d1": "not a chair: a large disc on a small stand",
    "c14060be9257436281dca96b55d118c4": "not a chair: a disc on a wall bracket, no legs",
    "25fb43458b864c5d8a3e792e9cd218c6":
        "not a chair: a boxy shell with no legs -- machine housing or a vehicle seat pan",
    # The blind spot the module docstring has named since Task 1, finally excluded by the only
    # gate that can see it. A box on a Z-shaped plinth: no seat/back split, no legs, no arms.
    "8a4a3a90bc104f11b82cedd9b4e5ab6b":
        "not a chair: a solid box on a Z-shaped plinth, no seat/back split",
    # --- not one object: broken, exploded, or several chairs in one file -----------------------
    "ba348ad8e81845279317ddc441d4a101":
        "not one chair: three separate folded chairs floating in a vertical column",
    "8ce51b2e9d774ba28aa7d768171a0def":
        "not one chair: a sling panel and a detached X-frame, disconnected",
    "796b03afe6604fa0891ebc06dc0e03f1":
        "not one chair: a skeletal frame -- two rails, a wireframe seat, no back",
}

#: The LVIS categories a KITCHEN library draws from. Applied by select() as rule 0, never by
#: curate: everything outside it is still a real, correctly-curated chair and keeps accepted=True.
#:
#: `chair`, `armchair`, `stool` and `folding_chair` stay -- a dining chair, a kitchen carver, a
#: counter stool at an island and a folding chair round a kitchen table are all things a kitchen
#: has. `rocking_chair` and `deck_chair` go: nothing MEASURABLE separates a rocking chair from a
#: dining chair (both are upright, whole, correctly-scaled seating that passes every criterion in
#: this file), which is exactly why this is an allow-list over the source category rather than a
#: criterion. It cost the library 12 of its fifty -- rocking_chair supplied 11 and deck_chair 1.
#:
#: THIS IS WHY DEFAULT_CATEGORY_CAP IS NOW UNSATISFIABLE, and the arithmetic is worth writing down
#: rather than discovering: a cap of 0.24 admits at most 24% of the target from any one category,
#: so FOUR categories can supply at most 96% of it. Fifty is unreachable from four categories at
#: this cap whatever the pool holds, and the measured ceiling on the pool that remains is 31
#: (12 chair + 12 stool at the cap, 6 folding_chair and 1 armchair exhausted below it). The cap has
#: NOT been touched to paper over that -- objaverse_chairs.md carries the measured alternatives.
KITCHEN_CATEGORIES = ("chair", "armchair", "stool", "folding_chair")

#: Chairs that are real, whole, correctly-curated furniture and still do not belong in a kitchen,
#: uid -> reason. Applied by select() as rule 0, alongside KITCHEN_CATEGORIES.
#:
#: WHY THIS IS NOT IN MANUAL_REJECTS. That dict's docstring refuses these outright -- "NOT a place
#: to put a chair that is merely inconvenient... A chair that is real but unwanted (an office chair
#: on castors, a deck chair) is a decision about what the library is FOR". It is right, and the
#: distinction is load-bearing in both directions: a `manual_rejects` entry is a claim that the
#: mesh is broken and stays true for every consumer, while everything below is a claim about the
#: ROOM and would be wrong in a library of lounge furniture. So they are separate dicts, the
#: verdicts land in different fields (`rejected_because` against `not_for_room`), and the
#: rejection histogram is not polluted with taste.
#:
#: EVERY UID BELOW WAS RENDERED AT FOUR AZIMUTHS AT FULL SIZE AND LOOKED AT, not read off a
#: contact sheet. Of the nineteen: six were proposed from the sheet and confirmed, five were found
#: by inspecting all 38 survivors, and eight were drawn by the backfill and caught before they
#: shipped. Four more were proposed from the sheet and OVERRULED -- three cantilever chairs whose
#: flat sled base reads as a rocker runner in a downsampled side-on silhouette (measured: min-z
#: within 5 mm of the floor across the whole base span, against 0.11-0.13 m for a real rocker), and
#: one compact tub carver read as a wingback.
KITCHEN_REJECTS = {
    # --- office seating ------------------------------------------------------------------------
    "cf0690389b9543caba4d011b0c7b3a7b":
        "not kitchen furniture: an office task chair -- five-star castor base, gas lift, "
        "contoured mesh back, armrests",
    # --- upholstered lounge / wingback armchairs -------------------------------------------------
    "a07501cd7f6c40fc9cf4cf438e41bac1":
        "not kitchen furniture: a wingback armchair -- forward-curving wings, scrolled arms, deep "
        "seat on an exposed stretcher frame",
    "b2fefa6f7af04c18966655d68a458974":
        "not kitchen furniture: a wingback armchair -- flat wing panels standing proud either side "
        "of a tall back, low seat, stub peg legs",
    "09ee59423b4b428f94f01fc5beccd227":
        "not kitchen furniture: an ornate wingback armchair -- arched crested back, padded scrolled "
        "arms, cabriole front legs and a carved apron",
    "e2fd51f8ce0141f1bf3bfa2d7b1bf5ef":
        "not kitchen furniture: a square-wing club armchair -- high enveloping back, thick rolled "
        "arms, deep seat, block feet; the widest chair the library offered",
    "c603a9922c6a4e77ab2306590f536ad6":
        "not kitchen furniture: a mid-century lounge chair -- low wide seat, back raked ~30 deg, "
        "thick wrap-around padded arms on splayed peg legs",
    "f5ca70f46c5549b8a754ee1abf42af68":
        "not kitchen furniture: a boxy upholstered club armchair -- deep seat cushion, full padded "
        "block arms, stub splayed legs",
    # --- garden / camping seating ----------------------------------------------------------------
    "a51e4acfdbb349c7876d7c37d2a0ee87":
        "not kitchen furniture: a slatted garden lounge armchair -- reclined slat back, thick "
        "square arms, four block feet (four discrete floor contacts, no castors, no column)",
    "7f558fcd893a4a78aaa6e91e41ce798f":
        "not kitchen furniture: a folding garden recliner -- multi-notch ratchet back hinge, sling "
        "fabric back and seat on a tubular frame",
    "3108e5f9977b409bb46d5b5dc7a3a452":
        "not kitchen furniture: a folding steamer/deck chair -- X-frame with the seat sling "
        "scooping to near floor level under a long raked slatted back",
    "4a673a28b08b461083c9a0270fc6e369":
        "not kitchen furniture: a folding camp/quad chair -- X-braced tubular frame with a fabric "
        "sling seat, fabric back and fabric armrests on four splayed feet",
    # --- the backfill wave, 2026-08-15 -----------------------------------------------------------
    # Six of the eleven chairs the first backfill drew, rejected on sight for the same reasons as
    # the eleven above. Five of the six are `armchair`, which is the measurement worth keeping:
    # 11 of the 12 accepted `armchair` entries in this manifest are lounge furniture. LVIS's
    # `armchair` is not "a chair with arms", it is "an upholstered easy chair", and a kitchen
    # library sourcing from it pays eleven hand verdicts for one usable carver.
    "eddd78c746734197a81613e10affed89":
        "not kitchen furniture: a boxy upholstered club armchair -- tall flat back panel, thick "
        "block arms, four thin splayed peg legs",
    "c6f69c82912b4961a6b5b7df2fa37747":
        "not kitchen furniture: a small upholstered tub/wing armchair with a fabric skirt on stub "
        "legs -- a nursing or accent chair",
    "7ab75054dea04187bca4da0917f501a3":
        "not kitchen furniture: a boxy club armchair -- deep square seat, thick block arms, arched "
        "back, stub legs",
    "91d8fea62ffb4ddd9dbc7798e09190cd":
        "not kitchen furniture: an ornate wingback armchair on cabriole legs",
    "b4ea7af8484c4c7e98704585b70002e9":
        "not kitchen furniture: a wingback armchair with rolled arms on cabriole legs",
    "cb9d503f2a274d799cd2cf2c60c3fae5":
        "not kitchen furniture: a pedestal chair -- a moulded shell seat with closed arms on a "
        "fluted column standing on a square plinth; a salon, barber or theatre chair",
    # The second backfill wave drew exactly one new chair, and it was the OTHER office chair --
    # the one objaverse_chairs.md names as "no longer selected" from the previous library. That is
    # the loop this list exists to close: without a durable verdict the replacement draw walks
    # straight back to the thing that was just removed.
    "b8c382798cdf473b86ed497a34770b35":
        "not kitchen furniture: an executive office chair -- five-star castor base, gas lift, "
        "padded high back, armrests",
    "0730fabb2c8341aaaf303351f2d644c7":
        "not kitchen furniture: a pedestal chair on a fluted column and plinth -- the same source "
        "pack as cb9d503f2a27, a salon/theatre chair",
}

#: Chairs the visual pass found LYING DOWN under the old argmax(extents) up-axis rule. They are
#: NOT manual rejects: the mesh is a real, whole piece of furniture and the defect was ours. The
#: up-axis fix (see _up_axis) is expected to stand them up, and the rebuild's contact sheet is what
#: decides whether it did -- listed here so that check is against a written-down set rather than
#: against memory.
LYING_DOWN_UNDER_ARGMAX = (
    "eebd97010dc947ab946b7e90d6397bfb", "47f6072b539f4292a9b7809e6fd5c8c6",
    "31d46740ef6a41b18f67808225cef468", "a44483bf93cd482ca92474620d405a1c",
    "24d8b2e005524aff82607e63e8eeb9ed", "395d619aae4440308b422ed8ecf08f8b",
    "897ce33a65d04bb69eb3d87d0742464f", "9d2c7a71d10442afb919acf853bbb12b",
)


def curate(manifest, *, thresholds=None, spec=None):
    """Judge every candidate that has a raw_path and no verdict yet. Returns manifest.

    For each such entry: loads and measures the mesh (_measure), runs CRITERIA in order against
    those measurements, and records the first failure's reason as `rejected_because` (accepted =
    False) or, if every criterion passes, accepted = True. The measurements themselves --
    extents, normalized_extents, faces, up_axis, facing_axis, plus the post-scale
    export_height_m/export_footprint_m -- are recorded on the entry regardless of verdict, so a
    later threshold change can re-decide every candidate by reading the manifest alone, without
    re-loading a single mesh.

    _measure is given the entry's uid, not just its raw_path, because one criterion (_export_width)
    judges the chair at the size it will SHIP at, and export height is drawn per-chair from the uid.
    That draw is fixed per uid and stable across processes, so this adds no non-determinism -- see
    _export_width for the coupling it does add.

    Entries without a raw_path (not yet fetched) are left untouched. Entries that already carry
    an "accepted" key are left untouched too -- curate does not re-judge or re-measure a
    candidate a previous run already decided, even if its raw_path no longer resolves.

    `thresholds` -- e.g. {"aspect_min": 0.20, "aspect_max": 0.95, "face_floor": 100,
    "face_ceiling": 200_000, "duplicate_iou_max": 0.98, "export_width_min_m": 0.35} -- defaults to
    DEFAULT_THRESHOLDS when omitted (or None), and is recorded into manifest["thresholds"] every
    call, so a saved manifest always carries the criteria that produced its verdicts.

    ORDER OF JUDGING, and why it is uid order
    -----------------------------------------
    _distinctness is pairwise: its verdict depends on what else is accepted, so which member of a
    duplicate pair survives is decided by which one is judged first. Left to the manifest's own
    list order that would be FETCH order -- which depends on how Objaverse happened to return the
    LVIS uid list and on how many batches have been run -- and the same 40 candidates fetched in
    two batches instead of one could keep a different chair. The rule instead:

    1. Unjudged candidates are judged in `uid` LEXICOGRAPHIC order, never list order.
    2. A candidate is compared only against entries ALREADY ACCEPTED: those carried in from a
       previous run, plus those accepted earlier in this same call. A rejected entry never shadows
       a later one -- a chair thrown out for face count must not take a good chair with it.
    3. So the survivor of a mutual-duplicate group is the member whose uid sorts FIRST, since it
       is judged first and becomes the incumbent for the rest.
    4. An entry a previous run already accepted always wins, whatever its uid, because rule 2 of
       curate's own idempotence (above) never re-judges a recorded verdict. Retracting a chair
       that has already been exported and converted would strand an OBJ and a USD on disk that
       something may already reference.
    5. The reason string names the incumbent with the highest IoU, tie broken on smallest uid --
       see _distinctness, which is itself order-independent.

    The entries themselves are mutated in place and the manifest's list order is NEVER changed, so
    nothing downstream that indexes into manifest["chairs"] is disturbed by this.

    Verified: shuffling manifest["chairs"] into different orders and curating each from scratch
    produces identical verdicts and identical rejected_because strings
    (test_chair_manifest.py::test_distinctness_verdicts_do_not_depend_on_manifest_order).
    """
    spec = spec or CHAIR
    if thresholds is None:
        thresholds = spec.thresholds
    manifest["thresholds"] = dict(thresholds)

    chairs = _entries(manifest, spec)

    # Hand verdicts first, and unconditionally. Two things this ordering buys that a criterion
    # cannot: a manual reject RETRACTS an entry a previous run accepted (every one of these was
    # accepted, selected and converted before a human saw it), and it is applied before the
    # already-judged skip, so re-curating a manifest cannot resurrect one. accepted_so_far is
    # rebuilt afterwards so a retracted chair also stops shadowing duplicates of itself.
    for entry in chairs:
        if entry.get("uid") in spec.manual_rejects:
            entry["accepted"] = False
            entry["manual_reject"] = True
            entry["rejected_because"] = f"manual reject: {spec.manual_rejects[entry['uid']]}"
    accepted_so_far = [e for e in chairs if e.get("accepted") is True]

    for entry in sorted(chairs, key=lambda e: e.get("uid", "")):
        if not entry.get("raw_path"):
            continue
        manual = entry.get("manual_reject") is True
        if "accepted" in entry and not manual:
            continue
        # A hand-rejected entry still gets MEASURED, once. Its verdict is already fixed and nothing
        # below can change it, but chair_manifest.py's contract is that a rejected entry keeps the
        # numbers it was judged on so a later reader can re-decide from the manifest alone without
        # re-fetching. Skipping measurement here would make the hand verdicts the only rejections
        # in the file with no evidence attached -- exactly the ones a sceptical reader most wants
        # to check. Guarded on "extents" so this costs one GLB load ever, not one per re-curate.
        if manual and "extents" in entry:
            continue
        try:
            measurements = _measure(entry["raw_path"], entry.get("uid"), spec=spec)
        except Exception as error:                          # noqa: BLE001 -- see below
            # A candidate whose GLB trimesh cannot turn into a mesh at all is a REJECT, not a
            # crash. This is not a criterion and has no threshold: it is the I/O boundary of a
            # stage that judges a couple of hundred arbitrary Objaverse uploads in one call, where
            # one malformed file would otherwise abort the batch and throw away every measurement
            # taken before it (the CLI saves the manifest once, at the end). The reason string
            # carries the exception so the bucket is diagnosable rather than mysterious, and the
            # entry keeps no measurements because none were taken.
            entry["accepted"] = False
            # A hand verdict outranks the I/O boundary: this entry was rejected because a human
            # looked at the mesh, and "unreadable" would replace that reason with a fact about
            # today's filesystem. The verdict is False either way; only the reason differs.
            if not manual:
                entry["rejected_because"] = f"unreadable: {type(error).__name__}: {error}"
            continue
        entry.update(measurements)
        if manual:
            continue        # measured for the audit trail; the hand verdict stands regardless

        reason = None
        for predicate in spec.criteria:
            reason = predicate(measurements, thresholds, accepted_so_far)
            if reason is not None:
                break

        entry["accepted"] = reason is None
        if reason is None:
            accepted_so_far.append(entry)
        else:
            entry["rejected_because"] = reason
    return manifest


def offered(manifest, *, spec=None) -> list:
    """The entries the library actually ships: accepted, and -- once select() has run -- selected.

    Before any selection exists this is exactly chair_manifest.accepted(manifest), which is what
    every stage meant by "accepted" before select() was added, so a manifest built without a
    selection behaves exactly as it did. After a selection exists, the expensive downstream stages
    (export, convert) restrict themselves to it: converting a chair that is not going to be
    offered costs one Isaac boot for nothing, and the fifty-chairs plan's second amendment moves
    conversion after selection for precisely that reason.

    "Any entry carries the key" is the test for whether a selection exists, not "any entry is
    True": select() writes selected=False on the chairs it passed over, so a pool where the
    selection happened to keep nothing is still a selection, not an unselected manifest.
    """
    chairs = _entries(manifest, spec or CHAIR)
    accepted_entries = [e for e in chairs if e.get("accepted") is True]
    if any("selected" in e for e in chairs):
        return [e for e in accepted_entries if e.get("selected") is True]
    return accepted_entries


#: Sentinel for select()'s category_cap, so that `None` keeps its own meaning ("no cap at all",
#: which one test asserts explicitly) while "not given" means "whatever this furniture type's
#: DEFAULT_CATEGORY_CAP is". A plain default of DEFAULT_CATEGORY_CAP would have hard-coded the
#: chair value into every call the table CLI makes.
_CAP_UNSET = object()


def select(manifest, target, *, category_cap=_CAP_UNSET, spec=None):
    """Choose `target` maximally-dissimilar chairs out of everything curate accepted, by greedy
    max-min shape IoU. Records selected (bool) on every accepted entry and selection_rank (0-based
    pick order) on the chosen ones, plus manifest["selection"] provenance. Returns manifest.

    WHY RANKING AND NOT A THRESHOLD. Task 1 measured all 703 pairs of the 38 accepted chairs at ten
    grid resolutions and found no gap anywhere below exact identity
    (2026-08-13-chair-distinctness-findings.md): the corpus is a continuum, so no similarity
    threshold can separate "different designs" from "the same design twice", and _distinctness is
    deliberately only a re-upload detector. Ranking needs no gap. "The fifty most mutually distinct
    of N accepted" is well defined whether or not the distribution has a knee, which is why the
    plan's amendment optimises over the measure here instead of reintroducing a cut in it.

    THE RULE, stated so it can be repeated:

    0. An accepted entry outside `spec.room_categories`, or named in `spec.room_rejects`, is not
       in the pool at all. It keeps `accepted: True` and every measurement, and is MARKED with
       `not_for_room` carrying the reason -- never deleted, so a later change of mind can read
       exactly what was excluded and why.

       WHY THIS SITS HERE AND NOT IN curate. `accepted` means "this mesh is a usable piece of
       furniture", which is a fact about the mesh and stays true whatever room it is destined
       for; "a rocking chair does not belong in a kitchen" is a fact about this library's
       purpose. Putting the second in a criterion would overload the first and make the
       rejection histogram unreadable. Putting it in `manual_rejects` is refused outright by
       that dict's own docstring ("NOT a place to put a chair that is merely inconvenient... A
       chair that is real but unwanted is a decision about what the library is FOR"). And it has
       to sit HERE rather than in a post-hoc filter over the fifty, because select() is also
       what BACKFILLS: excluding a rocking chair from the offered set but not from the pool
       means the replacement draw pulls in the next rocking chair.

    1. The pool is every entry surviving rule 0 and carrying a shape_grid at the manifest's own
       grid resolution. Entries WITHOUT a comparable descriptor cannot be ranked at all, so
       they are appended after the ranked ones in uid order -- fail open, the same choice
       _distinctness makes for a missing descriptor, rather than dropping a real chair because an
       older manifest predates the descriptor.
    2. The FIRST pick is the chair whose highest IoU against the whole pool is lowest: the corpus
       outlier. Greedy max-min is undefined on an empty selected set, and seeding it from "the
       first uid" would make the whole ordering an artefact of hex digits.
    3. Every pick after that is the candidate whose highest IoU against the ALREADY-SELECTED set is
       lowest -- the plan's rule verbatim.
    4. Ties break on the smallest uid, at every step including the first.
    5. No LVIS category may supply more than `category_cap` x target of the selection. A category
       at its cap is skipped over; the rule is otherwise unchanged. The first pick is exempt --
       it is chosen before any category has a count, and one chair cannot constitute domination.

    There is no RNG here at all. The result is a function of the accepted set's descriptors and
    nothing else -- not of manifest list order, not of fetch order, not of PYTHONHASHSEED. That is
    tested (test_selection_is_reproducible_and_independent_of_manifest_order).

    Selecting FEWER than target when the pool is smaller is not an error and not a reason to widen
    anything: the CLI reports the shortfall and the rejection histogram, exactly as `all` does, and
    what to do about it is the user's call. See the module's standing rule against relaxing a
    threshold to reach a count.
    """
    spec = spec or CHAIR
    if category_cap is _CAP_UNSET:
        category_cap = spec.category_cap
    chairs = _entries(manifest, spec)
    judged = sorted((e for e in chairs if e.get("accepted") is True), key=lambda e: e["uid"])

    # Rule 0. Marked on the entry rather than filtered silently: the manifest is the audit trail,
    # and "why is this accepted chair never offered?" has to be answerable from the file alone.
    # Rewritten from scratch on every select() -- an entry that leaves room_rejects, or a category
    # that is re-admitted, must lose the mark rather than keep a stale one.
    accepted_entries = []
    for entry in judged:
        reason = None
        if entry["uid"] in spec.room_rejects:
            reason = spec.room_rejects[entry["uid"]]
        elif (spec.room_categories and entry.get("category") is not None
                and entry.get("category") not in spec.room_categories):
            # `is not None` deliberately, and for the reason _capped gives a few lines below:
            # "uncategorised" is one bucket holding everything the manifest failed to label, not a
            # category. Excluding it would silently truncate a selection for a reason no user
            # could read -- and a manifest predating the `category` field would select nothing at
            # all. Fail open, the same choice the cap and _distinctness both make.
            reason = (f"category {entry.get('category')!r} is not one this library furnishes "
                      f"({', '.join(spec.room_categories)})")
        if reason is None:
            entry.pop("not_for_room", None)
            accepted_entries.append(entry)
        else:
            entry["not_for_room"] = reason

    def _rankable(entry):
        return bool(entry.get("shape_grid")) and entry.get("shape_grid_n") == SHAPE_GRID_N

    rankable = [e for e in accepted_entries if _rankable(e)]
    unrankable = [e for e in accepted_entries if not _rankable(e)]

    grids = {e["uid"]: _decode_grid(e["shape_grid"], SHAPE_GRID_N) for e in rankable}
    uids = sorted(grids)
    similarity = {uid: {} for uid in uids}
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            value = _grid_iou(grids[a], grids[b])
            similarity[a][b] = value
            similarity[b][a] = value

    category_of = {e["uid"]: e.get("category") for e in accepted_entries}
    cap = None if category_cap is None else max(1, int(np.ceil(category_cap * target)))
    taken = {}

    def _capped(uid):
        # An entry with no category is never capped. "Uncategorised" is not a category that can
        # dominate -- it is one bucket holding everything the manifest failed to label, and
        # counting it would silently truncate a selection for a reason no user could read.
        category = category_of.get(uid)
        return cap is not None and category is not None and taken.get(category, 0) >= cap

    order = []
    if uids:
        remaining = set(uids)
        # Rule 2: the corpus outlier opens the selection.
        first = min(uids, key=lambda uid: (max(similarity[uid].values(), default=0.0), uid))
        order.append(first)
        remaining.discard(first)
        taken[category_of.get(first)] = 1
        # Rule 3: highest-similarity-to-the-selected-set, minimised.
        worst = {uid: similarity[uid][first] for uid in remaining}
        while remaining and len(order) < target:
            # Rule 5: a category at its cap is skipped, not re-ranked. The eligible set is
            # recomputed every step because a category fills up mid-selection.
            eligible = [uid for uid in remaining if not _capped(uid)]
            if not eligible:
                break
            pick = min(eligible, key=lambda uid: (worst[uid], uid))
            order.append(pick)
            remaining.discard(pick)
            taken[category_of.get(pick)] = taken.get(category_of.get(pick), 0) + 1
            for uid in remaining:
                if similarity[uid][pick] > worst[uid]:
                    worst[uid] = similarity[uid][pick]

    chosen = {uid: rank for rank, uid in enumerate(order[:target])}
    for entry in unrankable:
        if len(chosen) < target:
            chosen[entry["uid"]] = len(chosen)

    # `judged`, not `accepted_entries`: an entry rule 0 excluded must be written selected=False
    # rather than keeping whatever a previous run left on it. Omitting it here is how a chair the
    # allow-list just dropped would go on being offered.
    for entry in judged:
        entry["selected"] = entry["uid"] in chosen
        if entry["selected"]:
            entry["selection_rank"] = chosen[entry["uid"]]
        else:
            entry.pop("selection_rank", None)

    excluded = [e for e in judged if e.get("not_for_room")]
    manifest["selection"] = {
        "rule": "greedy max-min shape IoU; first pick is the pool's lowest-max-IoU outlier; "
                "ties break on smallest uid"
                + ("" if cap is None else f"; no category may supply more than {cap}")
                + ("" if not spec.room_categories and not spec.room_rejects
                   else "; entries outside the room allow-list or hand-excluded from it are "
                        "marked not_for_room and never enter the pool"),
        "category_cap": category_cap,
        "category_cap_count": cap,
        "by_category": dict(sorted((k, v) for k, v in taken.items() if k is not None)),
        "room_categories": list(spec.room_categories),
        "room_excluded": len(excluded),
        "room_excluded_by_category": dict(sorted(collections.Counter(
            e.get("category") for e in excluded).items(), key=lambda kv: str(kv[0]))),
        "target": int(target),
        "selected": len(chosen),
        "pool": len(accepted_entries),
        "grid_n": SHAPE_GRID_N,
    }
    return manifest


def prune(manifest, *, remove=None, spec=None):
    """Delete the OBJ/USD artefacts of entries the library no longer offers, and drop the fields
    that point at them. Returns manifest.

    This exists because export() and convert() are idempotent by SKIPPING what already has an
    artefact, which is right for growing a library and wrong for re-deciding one. Two things
    re-decide it: re-curating from stripped verdicts (a chair that was accepted can become
    rejected -- both new criteria landed on 2026-08-13 cost the offered twenty one chair each),
    and select() (a chair can be accepted and still not make the fifty). Left alone, those chairs
    keep their usd_path, and scripts/simvla/kitchen_build.py offers exactly the chairs that have
    one -- so the library would silently ship more than it selected.

    Only artefact POINTERS are dropped (obj_path, usd_path, hull_count, has_collision). Every
    measurement stays -- extents, faces, shape_grid, export_footprint_m, the verdict, the reason --
    because the whole point of keeping a rejected entry is that a later threshold change can see
    what it excluded without re-fetching (chair_manifest.py's module docstring). A pruned chair can
    be brought back by export + convert alone; nothing about it has to be re-measured.

    `remove` is injectable (defaults to os.remove) so a test can assert exactly which files this
    would delete without touching a real library. A path that is already gone is not an error:
    prune is a reconciliation, and being asked to delete what is already deleted is the state it
    is trying to reach.
    """
    spec = spec or CHAIR
    if remove is None:
        remove = os.remove

    keep = {id(e) for e in offered(manifest, spec=spec)}
    for entry in _entries(manifest, spec):
        if id(entry) in keep:
            continue
        for field in ("obj_path", "usd_path"):
            path = entry.pop(field, None)
            if path:
                with contextlib.suppress(OSError):
                    remove(path)
        entry.pop("hull_count", None)
        entry.pop("has_collision", None)
    return manifest


def _stand_up(mesh, up_idx):
    """Rotate `mesh` in place so the axis curate measured as "up" (X=0, Y=1, Z=2) points along
    world +Z. Uses curate's own up_axis rather than re-deriving it from extents here, per the
    task brief: re-measuring after curate already did the work would let the two disagree.

    trimesh.geometry.align_vectors(source, target) returns the rotation that carries `source`
    onto `target` -- a proper rotation (determinant +1), not a reflection, so geometry is never
    mirrored. When up_idx is already 2 it returns the identity (source == target), which is why
    this is safe to call unconditionally instead of special-casing the Z-up case.
    """
    if up_idx == 2:
        return
    source = np.zeros(3)
    source[up_idx] = 1.0
    matrix = trimesh.geometry.align_vectors(source, np.array([0.0, 0.0, 1.0]))
    mesh.apply_transform(matrix)


def _facing_axis(mesh, up_idx):
    """Which horizontal axis (X or Y, given up_idx == 2) the chair faces, measured on THIS
    mesh's current geometry.

    Same method _measure uses for curate (findings doc, "Method", Step 3, "horizontal
    asymmetry"): split vertices at the bounding-box midpoint along up_idx; the horizontal axis
    whose above-midpoint mean leans furthest from centre (relative to its own extent) is the
    facing axis. Deliberately recomputed here rather than copied from the entry's curate-time
    value: curate measured facing_axis in the RAW mesh's frame, and _stand_up's rotation moves
    which world axis a given horizontal direction lands on -- curate's "Z" can become export's
    "X". chair_manifest.py's docstring calls this out explicitly: export's facing_axis is
    "(upright, rescaled)", i.e. in the exported OBJ's own frame, not curate's.
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extents = bounds[1] - bounds[0]
    above_mid = mesh.vertices[:, up_idx] > center[up_idx]
    horizontal_axes = [i for i in range(3) if i != up_idx]
    asymmetry = {}
    for i in horizontal_axes:
        mean_above = mesh.vertices[above_mid, i].mean() if np.any(above_mid) else center[i]
        asymmetry[i] = (mean_above - center[i]) / (extents[i] / 2)
    facing_idx = max(horizontal_axes, key=lambda i: abs(asymmetry[i]))
    return AXES[facing_idx]


def _refresh_facing_axis(entry):
    """Rewrite `entry["facing_axis"]` from the OBJ already on disk, in the exported frame.

    process=False so this measures the geometry as written, which is what export() itself measured
    on the in-memory mesh before writing it. trimesh's default processing merges duplicate
    vertices, which shifts the per-axis vertex means and can flip the answer on a chair whose two
    horizontal asymmetries are near-tied -- kitchen_build.FACING_CONFIDENCE_CUT exists for exactly
    those chairs, and this must not disagree with export() over which of the two it picked.

    Silently leaves the entry alone if the OBJ cannot be read: a missing OBJ is prune's business,
    not export's, and refusing to export a whole batch because one stale pointer dangles would be
    a worse failure than an unrefreshed field.
    """
    try:
        loaded = trimesh.load(entry["obj_path"], process=False, force="mesh")
    except Exception:                                       # noqa: BLE001 -- see docstring
        return
    entry["facing_axis"] = _facing_axis(loaded, 2)


def facing_direction(mesh) -> tuple:
    """The unit XY direction a chair faces: away from its backrest.

    The manifest's facing_axis records WHICH axis the chair is asymmetric on, measured above
    mid-height where the backrest is. It does not record which WAY -- and a chair turned 180
    degrees looks worse at a table than one not turned at all, so the sign has to come from
    the geometry rather than a coin flip.

    The backrest is the mass above seat height, offset to one side of the chair's own centre.
    A chair faces the opposite way. Returned normalised so callers can use it as a direction
    without rescaling.

    Measured before this was written (chair-placement Task 1 findings, "Chair-placement Task 1"):
    across 38 real accepted, exported chairs, the dominant axis of this same offset agreed with
    the recorded facing_axis 36/38 times overall, and 30/30 (100%) among the chairs where the
    signal was not near-zero -- both disagreements were in the weak-signal group, one of them a
    mesh independently flagged by eye as not really having a seat/back split. The signal is real,
    not assumed.
    """
    import numpy as np

    v = np.asarray(mesh.vertices, dtype=float)
    z_lo, z_hi = float(v[:, 2].min()), float(v[:, 2].max())
    upper = v[v[:, 2] > z_lo + 0.5 * (z_hi - z_lo)]
    if len(upper) == 0:
        return (1.0, 0.0)

    offset = upper[:, :2].mean(axis=0) - v[:, :2].mean(axis=0)
    norm = float(np.hypot(*offset))
    if norm < 1e-9:                      # symmetric: no backrest signal, pick a stable default
        return (1.0, 0.0)
    return (float(-offset[0] / norm), float(-offset[1] / norm))


def export(manifest, *, spec=None):
    """Stand every offered candidate upright, scale it to a life-sized chair, and sit it on the
    floor over the origin. Writes `<CHAIR_OBJ_DIR>/obj/<uid>.obj` and records obj_path,
    scale_applied, height_m and facing_axis on the entry. Also records
    manifest["material_group"] = "floor" (see below). Returns manifest.

    "Offered", not merely "accepted" -- see offered(). Without a selection those are the same set,
    which is what this stage always did; with one, obj/ holds exactly the chairs the library ships
    rather than every candidate that passed curate.

    No material is exported alongside the geometry (`mesh.export(..., include_texture=False)`).
    Two reasons, not one:

    1. A bug this function used to have: trimesh's OBJ exporter defaults to a FIXED filename,
       `material.mtl`, regardless of which chair is being exported. Every accepted candidate
       shares the same `obj_dir`, so every export() call after the first CLOBBERED the previous
       chair's `material.mtl` (and its texture PNGs, same fixed-name problem) in place -- by the
       time all accepted chairs had been exported, only the LAST one's material was still on disk;
       every other `.obj`'s `usemtl` directive resolved to a mismatched material. Geometry itself
       was never affected (embedded per-file, not shared) -- this was a materials-only bug, and
       trimesh's `include_texture` flag is what gates the whole materials/mtllib code path (see
       `trimesh.exchange.obj.export_obj`): `False` collects no materials at all, so no `mtllib`
       line and no `.mtl`/texture files are written, sidestepping the collision entirely rather
       than papering over it with a per-uid filename.
    2. Even a correctly-named per-uid material would be thrown away: the user decided chairs take
       the KITCHEN's own floor material at placement time -- the same `MATERIALS['floor']` pool of
       18 wood/tile MDLs `scripts/simvla/kitchen_build.py` already uses for the floor and, per an
       earlier decision recorded there, the table (`GEOMETRY2MATERIAL["/world/table/.*"] =
       'floor'`). A chair's own Objaverse-sourced material was therefore never going to reach the
       final scene, so there is nothing to preserve here -- geometry-only export is not a
       stopgap, it is the actual right shape of this data. `manifest["material_group"] = "floor"`
       records that decision so a later placement stage does not have to rediscover it by reading
       kitchen_build.py's table special-case and guessing whether the same logic applies to
       chairs.

    Objaverse carries no units and no reliable up-axis, so there is no "native" pose to preserve
    -- everything here is derived from curate's own measurements (up_axis) plus a per-chair
    target height drawn from a Gaussian centred on TARGET_HEIGHT_M (see _sample_height), seeded
    from the chair's own uid so the same chair gets the same height on every run.

    Order matters: rotate to Z-up FIRST (_stand_up), so the height used for scaling is measured
    along the axis that will actually be vertical after export -- scaling by the pre-rotation
    "tallest extent" would be correct here since up_axis IS the tallest extent (curate's
    definition), but rotating first keeps this correct even if a future up_axis source is no
    longer tied to the tallest-extent heuristic. Bottom-alignment happens LAST, after scaling,
    because scaling about the origin moves bounds[0][2] too -- translating first and then scaling
    would put the underside back below (or above) z=0.

    The bottom-alignment offset is read from the mesh's actual bounds AFTER scaling, not computed
    as height/2: this project's own kitchen table shipped with that exact height/2 mistake (an
    asset origin ~15 mm off-centre left it sunk into the floor) and no test caught it because
    nothing asserted z. -bounds[0][2] is exact regardless of where the origin sits.

    Entries that already carry obj_path are left untouched -- re-running export after export has
    already run must not re-export or move an existing OBJ out from under whatever now references
    it.

    WHAT "scale to life size" MEANS is `spec.export_scale`: a chair's whole bounding box reaches a
    uid-seeded target height, a table's WORK SURFACE reaches a fixed 0.74 m. The rest of this
    function -- stand up, scale uniformly, centre x/y, sit on the floor -- is the same for both.
    """
    spec = spec or CHAIR
    library = _library(spec)

    obj_dir = os.path.join(library.root(), "obj")
    os.makedirs(obj_dir, exist_ok=True)

    # Recorded unconditionally on every call, same as curate() does for manifest["thresholds"] --
    # a saved manifest should always carry the decision that produced its exported OBJs, not just
    # the first time export() happened to run.
    manifest["material_group"] = "floor"

    for entry in offered(manifest, spec=spec):
        if "obj_path" in entry:
            # Already exported, so the OBJ is left exactly where it is -- but facing_axis is NOT
            # left alone, because it is the one field two stages both write. curate measures it in
            # the RAW mesh's frame; export rewrites it in the EXPORTED frame, and _stand_up's
            # rotation moves which world axis a horizontal direction lands on, so curate's "Z" can
            # be export's "X". Re-curating a manifest whose OBJs already exist (which is exactly
            # what the fifty-chair rebuild did: verdicts stripped, all 207 candidates re-judged)
            # therefore overwrites this field with a raw-frame value that the skip above then
            # preserves forever. Measured on the three chairs that had survived from the previous
            # library: two came out carrying facing_axis "Z", which is not even a possible answer
            # in the exported frame, and one carried "X" for a chair whose exported asymmetry is
            # +0.91 on Y. Recomputing from the OBJ on disk costs one trimesh load per already-
            # exported chair and makes the field true by construction rather than by luck.
            spec.refresh_export_extras(entry)
            continue

        loaded = trimesh.load(entry["raw_path"], process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        mesh = mesh.copy()

        up_idx = entry["up_axis"] if isinstance(entry["up_axis"], int) else AXES.index(entry["up_axis"])
        _stand_up(mesh, up_idx)

        target_height, raw_height = spec.export_scale(entry, mesh)
        scale = target_height / raw_height
        mesh.apply_scale(scale)

        # Sit it on the floor AND put it over the origin. The z half is old; the x/y half is a
        # 2026-08-13 fix for a LIVE PLACEMENT BUG, not a tidy-up. Objaverse meshes carry whatever
        # origin their author left, and export used to preserve it: 31 of the 50 offered chairs sat
        # more than 5 cm from x=y=0 and ba348ad8e818... sat 9.95 m away. add_chair places a chair by
        # composing the seat-ring transform with the asset's own frame, so that offset passes
        # straight through -- measured end to end, the 9.95 m chair landed 9.953 m from the seat
        # the ring computed for it, outside the kitchen entirely. Nothing caught it because the
        # overlap gate only sees chairs that touch something, and a chair in the next room touches
        # nothing.
        #
        # Read from the post-scale bounds, like the z offset and for the same reason: scaling about
        # the origin moves the centre too. Horizontal BBOX centre rather than centre of mass -- the
        # ring reserves an axis-aligned footprint per seat (chair_footprint_m), so the quantity that
        # must be centred is the one the ring reserves, not where the mass happens to sit.
        bounds = mesh.bounds
        mesh.apply_translation([-(bounds[0][0] + bounds[1][0]) / 2.0,
                                -(bounds[0][1] + bounds[1][1]) / 2.0,
                                -float(bounds[0][2])])

        obj_path = os.path.join(obj_dir, f"{entry['uid']}.obj")
        # include_texture=False: no per-chair material is wanted (see the docstring above) --
        # this also happens to be what stops every export() call from clobbering the one shared
        # obj_dir/material.mtl that trimesh's default OBJ exporter would otherwise write.
        mesh.export(obj_path, include_texture=False)

        entry["obj_path"] = obj_path
        entry["scale_applied"] = float(scale)
        entry["height_m"] = float(mesh.bounding_box.extents[2])
        entry.update(spec.export_extras(mesh))
    return manifest


# --- convert ---------------------------------------------------------------------------------

#: scripts/tools/convert_mesh.py, resolved relative to THIS file rather than the caller's cwd --
#: convert may be invoked (via the CLI below) from anywhere.
_CONVERT_MESH_SCRIPT = str(Path(__file__).resolve().with_name("convert_mesh.py"))

#: scripts/tools/repair_chair_collision.py -- see repair_collisions()'s docstring for why this
#: exists as a SEPARATE tool rather than a change to convert_mesh.py or the vendored
#: isaaclab.sim.converters.mesh_converter.MeshConverter underneath it.
_REPAIR_COLLISION_SCRIPT = str(Path(__file__).resolve().with_name("repair_chair_collision.py"))

#: How many times _default_convert_runner will re-run convert_mesh.py for one mesh before giving
#: up. 3, because the failure it absorbs is a storage race, not a mesh defect: measured on one
#: chair with the identical command on the identical node, attempt 1 succeeded, attempt 2 died
#: with "Accessed invalid null prim", attempt 3 succeeded. See _default_convert_runner.
_CONVERT_ATTEMPTS = 3

#: Matches repair_chair_collision.py's "RESULT <uid> <True/False>" lines -- deliberately NOT
#: anchored at end-of-line, since repair_one() appends a trailing "  # repaired=..." comment
#: after the True/False that this regex must ignore rather than fail to match.
_REPAIR_RESULT_RE = re.compile(r"^RESULT\s+(\S+)\s+(True|False)\b", re.MULTILINE)

#: Best-effort match for a hull/convex-shape count in convert_mesh.py's stdout, in case a future
#: version of that script (or MeshConverter under it) ever reports one. As of this writing it does
#: not -- see _extract_hull_count's docstring for the investigation -- so this regex currently
#: never matches anything real; it exists so that IF the tool starts reporting a count, convert()
#: picks it up without a code change, rather than needing this file edited in lockstep with that
#: one.
_HULL_COUNT_RE = re.compile(r"(?:hull[_ ]count|number of convex (?:hulls|shapes))\s*[:=]\s*(\d+)", re.IGNORECASE)


def _extract_hull_count(result):
    """The physics cost of convex decomposition, if it can be known at all from a completed
    convert_mesh.py run. Returns an int, or None when it genuinely cannot be determined -- never a
    fabricated number.

    It cannot be determined today, from either of the two places the task brief allows:

    1. convert_mesh.py's stdout: it never prints a hull count. Confirmed by reading
       isaaclab.sim.converters.mesh_converter.MeshConverter._convert_asset -- the whole of what
       "convert" does with cfg.collision_approximation ("convexDecomposition") is set it as a
       MeshCollisionAPI.approximation ATTRIBUTE (a string) on the mesh prim. No decomposition
       happens at conversion time, so there is nothing to count yet.

    2. Reading the produced USD back: still nothing to read. The actual per-hull decomposition is
       PhysX cooking, which runs lazily the first time a RUNNING physics scene touches the
       collider -- confirmed by omni.physx's own binding for the query that would answer this,
       get_nb_convex_mesh_data(meshPath), whose docstring says outright: "Get number of convex
       mesh data for given prim path. (Does work only when simulation is running.)" A static USD
       on disk, opened by nothing more than a converter, has no hull count stored in it to read.
       Even opening the file at all would need the `pxr` module, which this checkout cannot
       import outside a booted Isaac Sim/Kit process in the first place (verified: no standalone
       `pxr` site-package exists anywhere under the env_isaaclab conda env; every copy of it lives
       bundled inside an individual Isaac Sim extension package, added to sys.path by AppLauncher
       at Kit boot).

    So getting a real hull count needs a physics scene actually PLAYING, not merely Isaac Sim
    booted (which convert_mesh.py already does, and convert() already pays for via the
    subprocess) -- a materially bigger step than "convert one mesh to USD," and out of scope for
    this task. _HULL_COUNT_RE above still checks stdout first, in case a future convert_mesh.py
    starts reporting one; when it does not, this returns None rather than pretending to know.
    """
    stdout = getattr(result, "stdout", "") or ""
    match = _HULL_COUNT_RE.search(stdout)
    return int(match.group(1)) if match else None


def _default_convert_runner(input_path, output_path):
    """Shell out to convert_mesh.py under THIS SAME interpreter (sys.executable) -- when convert()
    is itself running inside the env_isaaclab conda env (as it must: this module imports trimesh,
    which is only installed there), sys.executable already has `isaaclab` importable, exactly what
    convert_mesh.py needs.

    A subprocess, not an import: convert_mesh.py calls isaaclab.app.AppLauncher at module scope,
    which boots Isaac Sim/Kit the moment it runs. That cannot happen inside an already-running
    pytest process (which has not booted Kit itself), and Kit does not support being booted twice
    in one process -- the same reason fetch() shells out to (rather than imports) objaverse's own
    network calls in-process only when uninjected.

    --headless: this pipeline converts N accepted chairs in a batch, with no display expected to
    be attached.

    RETRIES, and what they are for. MeshConverter._convert_asset writes its USD and then re-opens
    that same path in the same process; on this cluster's Lustre-backed storage that reopen loses
    the race against its own write. The 2026-08-11 investigation caught the QUIET form of this --
    the reopened stage is missing the collision schema, which repair_collisions() then authors
    post-hoc. The fifty-chair batch caught the LOUD form: the reopened stage has no
    /<basename>/geometry prim at all, and the converter dies with "RuntimeError: Accessed invalid
    null prim" at mesh_converter.py:121. Measured on one chair, same command, same node: attempt 1
    succeeded, attempt 2 crashed, attempt 3 succeeded. It is the same race, it is not a property
    of the mesh, and a batch of fifty that gives up on the first instance converts nothing.

    So: retry, up to _CONVERT_ATTEMPTS times, and raise the last failure if every attempt loses the
    race. A retry is cheap relative to the alternative (one lost Isaac boot against a whole batch)
    and is not papering over a mesh problem -- a mesh the converter genuinely cannot handle fails
    all three times, and then this raises exactly as it did before.
    """
    last = None
    for _ in range(_CONVERT_ATTEMPTS):
        try:
            return subprocess.run(
                [sys.executable, _CONVERT_MESH_SCRIPT, input_path, output_path, "--headless"],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as error:
            last = error
    raise last


def convert(manifest, *, runner=None, limit=None, spec=None):
    """Convert every accepted candidate's exported OBJ into a USD with a collision mesh, via
    scripts/tools/convert_mesh.py (--collision-approximation defaults to convexDecomposition --
    the known hard case for a chair: thin legs and slatted backs either explode the hull count or
    make the legs vanish, which is exactly why the brief wants hull_count visible per entry before
    a whole corpus gets generated on top of it). Writes <CHAIR_OBJ_DIR>/usd/<uid>.usd. Records
    usd_path and hull_count (see _extract_hull_count for why hull_count is usually None today).
    Returns manifest.

    `runner(input_path, output_path)` defaults to _default_convert_runner (a real subprocess) and
    is injectable so a test can drive convert() without ever booting Isaac Sim -- exactly how
    fetch takes an injectable `download`.

    Only entries the library OFFERS (see offered(): accepted, and selected once select() has run),
    already exported (carry obj_path), and not yet converted (carry no usd_path) are touched:
      * not offered -- covers a rejected entry (accepted False), one curate hasn't judged yet, and
        one select() passed over. convert must never shell out for any of them: this is the
        pipeline's single most expensive step, one Isaac boot per mesh, which is exactly why the
        fifty-chairs plan moved it after selection.
      * no obj_path yet -- export hasn't run for this entry; there is nothing to convert.
      * usd_path already present -- convert already ran for this entry; re-running convert must
        not redo the (expensive, Isaac-booting) work, mirroring export's own obj_path guard.

    `limit`, when given, converts at most that many entries per call and leaves the rest for the
    next one. This stage is the only one in the pipeline whose runtime is minutes-per-item rather
    than milliseconds -- one Isaac Sim boot each, ~40 s measured -- so a fifty-chair batch is over
    half an hour in a single call, and the CLI saves the manifest once, at the end. Anything that
    interrupts that call (a wall-clock cap, a node eviction, a Ctrl-C) throws away every boot it
    already paid for. Converting in bounded chunks makes the manifest a checkpoint: the usd_path
    guard above means the next call resumes exactly where this one stopped, with no argument to
    keep in sync.
    """
    spec = spec or CHAIR
    if runner is None:
        runner = _default_convert_runner

    usd_dir = os.path.join(_library(spec).root(), "usd")
    os.makedirs(usd_dir, exist_ok=True)

    converted = 0
    for entry in offered(manifest, spec=spec):
        if "obj_path" not in entry or "usd_path" in entry:
            continue
        if limit is not None and converted >= limit:
            break

        usd_path = os.path.join(usd_dir, f"{entry['uid']}.usd")
        result = runner(entry["obj_path"], usd_path)

        entry["usd_path"] = usd_path
        entry["hull_count"] = _extract_hull_count(result)
        converted += 1
    return manifest


# --- repair_collisions ------------------------------------------------------------------------
#
# Bugfix (2026-08-11 findings, "17 of 20 converted chairs have no collision geometry"): convert()
# above faithfully calls the vendored converter as designed, but MeshConverter._convert_asset
# (source/isaaclab/isaaclab/sim/converters/mesh_converter.py, also vendored -- not edited by this
# fix, same as convert_mesh.py itself) has a race between writing its converted USD and
# immediately re-opening that same path in the same process to author collision on it. See
# scripts/tools/repair_chair_collision.py's module docstring for the full investigation: face
# count, watertightness, vertex count and aspect ratio do not separate the chairs that came out
# WITH a collider from the ones that didn't, and re-converting a chair that previously succeeded,
# fresh, to a brand new output directory, reproducibly failed 5/5 times -- so this is not a
# property of any specific mesh, and not something a differently-chosen conversion flag fixes.


def _default_repair_collision_runner(usd_paths: dict):
    """Shell out to repair_chair_collision.py, once, for the WHOLE batch -- one Isaac boot to
    repair (or merely verify) every uid in `usd_paths`, not one boot per chair. Mirrors
    _default_convert_runner's reasoning for why this must be a subprocess: repair_chair_collision.py
    boots Isaac Sim at module scope, which cannot happen inside this (or any pytest) process.

    Passes --results-file (a throwaway tempfile) rather than relying on the subprocess's captured
    stdout: empirically, stdout captured via subprocess.run(capture_output=True) from a script
    that boots Isaac Sim/Kit reliably loses everything printed AFTER Kit finishes booting -- the
    exact same symptom hit and worked around during this investigation's own diagnostics (a plain
    print()-based version of this runner produced zero parseable output across 20 real chairs,
    while the diagnostic that instead wrote to a flushed file captured every line). The file's
    contents are read back and returned as `result.stdout` so _parse_repair_results doesn't need
    to know or care which channel the text came from.
    """
    targets = [f"{uid}={path}" for uid, path in usd_paths.items()]
    with tempfile.NamedTemporaryFile(
        mode="r", suffix=".txt", prefix="repair_chair_collision_", delete=False
    ) as f:
        results_file = f.name
    try:
        proc = subprocess.run(
            [sys.executable, _REPAIR_COLLISION_SCRIPT, *targets,
             "--results-file", results_file, "--headless"],
            capture_output=True, text=True, check=True,
        )
        with open(results_file) as f:
            file_contents = f.read()
    finally:
        with contextlib.suppress(OSError):
            os.remove(results_file)
    # Prefer the file's contents (reliable) but fall back to whatever stdout did capture, so a
    # result is never thrown away if the file somehow came back empty.
    proc.stdout = file_contents or proc.stdout
    return proc


def _parse_repair_results(result) -> dict:
    """{uid: bool} parsed from repair_chair_collision.py's "RESULT <uid> <True/False>" stdout
    lines. A uid missing from the returned dict (stdout didn't contain a RESULT line for it, e.g.
    the subprocess crashed before reaching it) is left for the caller to notice by its absence,
    rather than this function guessing True or False on its behalf.
    """
    stdout = getattr(result, "stdout", "") or ""
    return {uid: (value == "True") for uid, value in _REPAIR_RESULT_RE.findall(stdout)}


def repair_collisions(manifest, *, runner=None, spec=None):
    """For every entry that has been converted (carries usd_path), ensure its USD's mesh actually
    carries collision geometry -- repairing it directly (see repair_chair_collision.py) if
    convert()'s own MeshConverter call silently dropped it. Records a per-entry `has_collision`
    bool. Returns manifest.

    `runner(usd_paths: dict[uid, path])` defaults to _default_repair_collision_runner (a real,
    single subprocess covering the whole batch) and is injectable exactly like convert()'s
    `runner`, so a test can drive this without booting Isaac.

    Idempotent and safe to call on every entry every time: repair_chair_collision.py itself skips
    any mesh that already carries UsdPhysics.MeshCollisionAPI (see its repair_one()), so
    re-running this on an already-repaired (or always-fine) chair is a cheap no-op check, not a
    redo. Unlike export()/convert()'s obj_path/usd_path guards, this does NOT skip entries that
    already carry `has_collision` -- verifying is exactly what this function is for, and the cost
    is one shared Isaac boot for the whole batch, not one per entry.
    """
    if runner is None:
        runner = _default_repair_collision_runner

    entries = [e for e in _entries(manifest, spec or CHAIR) if e.get("usd_path")]
    if not entries:
        return manifest

    usd_paths = {e["uid"]: e["usd_path"] for e in entries}
    result = runner(usd_paths)
    results = _parse_repair_results(result)

    for entry in entries:
        if entry["uid"] in results:
            entry["has_collision"] = results[entry["uid"]]
    return manifest


# --- the contact sheet --------------------------------------------------------------------------
#
# The only gate on this library that is not geometric. Every criterion above judges numbers, and
# two measured failures say that is not enough:
#
#   * _aspect_ratio and _face_count pass an abstract pedestal stand that is not furniture
#     (0730fabb2c8341aaaf303351f2d644c7 in the first library, 8a4a3a90bc104f11b82cedd9b4e5ab6b in
#     this one) -- and select()'s greedy max-min diversity actively PREFERS such a thing, because
#     "unlike every other candidate" is exactly what a non-chair is. Selecting for distinctness
#     selects for weirdness.
#   * An LVIS label can simply be wrong. The desk gate's 60-mesh sample contained two staircases
#     and an escalator labelled `desk`, visible only once a ground plane that dominated their bounds
#     was stripped. No threshold catches that; only looking does.
#
# So this stage renders, and a human reads what it renders. It is a stage rather than a scratch
# script because the library is rebuilt per batch and a sheet that goes stale is worse than none.

#: Half-width, in metres, of the world window every cell is framed to. Every cell uses the SAME
#: window rather than fitting each mesh, so relative scale is visible: a chair rendered small IS
#: small. 0.7 m covers the widest export with margin -- the offered fifty run to 0.906 m across,
#: whose 35-degree diagonal silhouette is ~0.64 m half-width -- and _flat_shade reports any mesh
#: that overflows it instead of silently cropping.
CONTACT_SHEET_HALF_EXTENT_M = 0.7

#: Two viewpoints per chair, not one, and 90 degrees apart. Both earlier visual passes used two
#: for a measured reason: a pedestal stand reads as a chair from one azimuth, and a mesh with a
#: welded-in floor plane or backdrop can hide it edge-on.
CONTACT_SHEET_AZIMUTHS = (35.0, 125.0)
CONTACT_SHEET_ELEV = 18.0

#: How far below world z=0 the window reaches, as a fraction of the half-extent. See _flat_shade:
#: a floor-standing chair's near footprint corner projects BELOW z=0 by sin(elev) x its half-
#: diagonal, so a window flush with the floor would clip every wide chair's front foot.
CONTACT_SHEET_FLOOR_MARGIN = 0.3

#: Keep at most this many faces, the BIGGEST ones. A random subsample turns a dense mesh into
#: swiss cheese; the largest faces preserve slabs and silhouettes, which is all this rasteriser
#: shows anyway. The offered fifty top out at 112k faces, so this does not bite today -- it is
#: here because the desk sample hit 1.07 M and the next chair batch may too.
CONTACT_SHEET_FACE_CAP = 120_000

#: Report a chair whose exported horizontal centre is further than this from the origin. 5 cm is
#: comfortably below any real chair's half-width, so anything over it is an asset that was never
#: centred rather than a mesh whose origin sits slightly off its seat.
CONTACT_SHEET_OFF_CENTRE_REPORT_M = 0.05

CONTACT_SHEET_NAME = "contact_sheet.png"


# --- the chair furniture type -------------------------------------------------------------------
#
# Assembled here, at the end of the constants, because it names things defined throughout this
# module -- DEFAULT_THRESHOLDS, CRITERIA, MANUAL_REJECTS, SAMPLE_SEED, DEFAULT_CATEGORY_CAP,
# CONTACT_SHEET_HALF_EXTENT_M. Every one of those constants keeps its module-level name: they are
# imported by name in tests and quoted by uid in findings docs, and this extraction is not the
# place to move them.


class ChairType(FurnitureType):
    """The chair library, as a FurnitureType. Every method below is the code `curate`, `export`
    and `_measure` used to run inline; nothing about a chair's treatment changed."""

    name = "chair"
    entries_key = "chairs"
    manifest_module = "chair_manifest"
    default_category = "chair"

    thresholds = DEFAULT_THRESHOLDS
    criteria = tuple(CRITERIA)
    manual_rejects = MANUAL_REJECTS

    sample_seed = SAMPLE_SEED
    category_cap = DEFAULT_CATEGORY_CAP
    room_categories = KITCHEN_CATEGORIES
    room_rejects = KITCHEN_REJECTS

    contact_sheet_half_extent_m = CONTACT_SHEET_HALF_EXTENT_M

    def measure_extras(self, mesh, up_idx) -> dict:
        """The facing axis, in the RAW mesh's frame -- recorded, never judged on. The findings doc
        found it self-consistent on 40/40 real chairs but "confidently" (|value| > 0.10) so on only
        36/40; the other 4 are real wrap-around/barrel-back chairs, not detection failures, so low
        confidence must not read as "reject."

        A table records nothing here -- see FurnitureType's docstring for the 170x-against-9x
        measurement that decided that.
        """
        return {"facing_axis": _facing_axis(mesh, up_idx)}

    def export_geometry(self, measurements, uid):
        """The exported height and footprint, known in closed form at curate time because export
        scales uniformly to a uid-seeded height. _export_width judges the result."""
        return _export_footprint(measurements["normalized_extents"],
                                 AXES.index(measurements["up_axis"]), uid)

    def export_scale(self, entry, mesh):
        """A chair's whole bounding box reaches its own uid-seeded target height."""
        return _sample_height(entry["uid"]), float(mesh.bounding_box.extents[2])

    def export_extras(self, mesh) -> dict:
        """facing_axis again, in the EXPORTED frame this time: _stand_up's rotation moves which
        world axis a horizontal direction lands on, so curate's "Z" can be export's "X"."""
        return {"facing_axis": _facing_axis(mesh, 2)}

    def refresh_export_extras(self, entry) -> None:
        """Rewrite facing_axis from the OBJ already on disk -- the one field two stages both write,
        so a re-curate of an already-exported manifest must not leave a raw-frame value behind."""
        _refresh_facing_axis(entry)


CHAIR = ChairType()


def _agg_pyplot():
    """matplotlib on the Agg backend, imported lazily.

    Lazily because every other stage in this module runs without matplotlib, and importing pyplot
    costs ~1 s; Agg because this node has no display and no Xvfb (nor pyrender), which is the whole
    reason _flat_shade exists instead of `scene.save_image()`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    return plt, PolyCollection


def _horizontal_centre(mesh):
    """[cx, cy, 0]: the mesh's horizontal bounding-box centre, with z left alone.

    z is left alone deliberately -- export() bottom-aligns it to the floor, and that IS the datum
    the sheet frames against. Only x and y are arbitrary.
    """
    bounds = np.asarray(mesh.bounds)
    return np.array([(bounds[0][0] + bounds[1][0]) / 2.0,
                     (bounds[0][1] + bounds[1][1]) / 2.0,
                     0.0])


def _flat_shade(ax, mesh, *, azim, elev=CONTACT_SHEET_ELEV,
                half_extent=CONTACT_SHEET_HALF_EXTENT_M):
    """Draw `mesh` into `ax` as a flat-shaded raster, and return True if it fitted the window.

    A from-scratch painter's-algorithm rasteriser, first written for the chairs project's Task 1
    and reused by the desk gate: project every triangle onto a camera basis, shade each by
    |n . light| so faces at different angles separate, depth-sort back-to-front, and hand the whole
    lot to one matplotlib PolyCollection. No OpenGL, so no display and no new dependency.

    What it CANNOT show, stated so nobody reads more into a sheet than is in it: it ignores UVs,
    materials and vertex colours entirely. A mesh can be shape-correct and texture-broken and look
    perfect here. It shows silhouette, part layout and gross proportion -- which is precisely what
    the non-chair failures above are made of.

    The window is fixed in WORLD metres (see CONTACT_SHEET_HALF_EXTENT_M) rather than fitted to the
    mesh: exports are life-sized and sit on z=0, so a shared window makes "this one is doll-sized"
    or "this one is enormous" readable directly off the sheet. Vertically the window starts just
    below the floor so a chair that does not reach it, or sinks through it, is visible too.

    SCALE is shared; POSITION is not. The mesh is slid so its horizontal bounding-box centre is at
    the origin before projecting, and only then framed. That is not cosmetic: export() bottom-
    aligns z but never centres x/y, so an exported chair sits wherever its source GLB happened to
    put it -- measured on the offered fifty, 31 are more than 5 cm off-origin and one
    (ba348ad8e81845279317ddc441d4a101) is 9.95 m away. Framing on world x=y=0 would render 31 blank
    cells and a sheet nobody could read, while telling you nothing you could not get from the
    manifest. contact_sheet() records the offsets instead, so the defect is reported as a number
    rather than as a missing picture.
    """
    _, PolyCollection = _agg_pyplot()

    faces = mesh.faces
    if len(faces) > CONTACT_SHEET_FACE_CAP:
        faces = faces[np.argsort(-mesh.area_faces)[:CONTACT_SHEET_FACE_CAP]]

    a, e = np.radians(azim), np.radians(elev)
    forward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    cam_up = np.cross(forward, right)

    tri = (np.asarray(mesh.vertices) - _horizontal_centre(mesh))[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0          # degenerate triangles: shade them, do not divide by zero
    normals /= lengths

    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.abs(normals @ light)

    order = np.argsort(-(tri.mean(axis=1) @ forward))     # far first: painter's algorithm
    proj = np.stack([tri @ right, tri @ cam_up], axis=-1)[order]
    colors = np.clip(shade[order][:, None] * np.array([[0.55, 0.62, 0.72]]), 0, 1)
    ax.add_collection(PolyCollection(proj, facecolors=colors, edgecolors="none",
                                     antialiased=False))

    # World z=0 lands a fixed fraction up every cell, so the exports all stand on the same line.
    # The window drops BELOW the floor by CONTACT_SHEET_FLOOR_MARGIN because a horizontal plane
    # does not project to a horizontal line: at a non-zero elevation the near corner of a chair's
    # footprint lands below its far corner, by sin(elev) x the footprint's half-diagonal -- ~0.2 m
    # for the widest export. Without that margin every wide chair would report an overflow it does
    # not have.
    bottom = -CONTACT_SHEET_FLOOR_MARGIN * half_extent
    top = (2.0 - CONTACT_SHEET_FLOOR_MARGIN) * half_extent
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")

    xs, ys = proj[..., 0], proj[..., 1]
    return bool(xs.min() >= -half_extent and xs.max() <= half_extent
                and ys.min() >= bottom and ys.max() <= top)


def contact_sheet(manifest, *, out_path=None, columns=10, dpi=100,
                  azimuths=CONTACT_SHEET_AZIMUTHS, spec=None):
    """Render one image grid of every OFFERED chair, each from len(azimuths) viewpoints and
    labelled with a short uid and its selection_rank. Returns a report dict:

        {"path": ..., "drawn": [uid, ...], "failed": {uid: reason},
         "overflowed": [uid, ...], "off_centre_m": {uid: metres}}

    `drawn` is every offered uid that got a cell, in the order they appear on the sheet, and the
    function RAISES rather than returning if that set is not exactly offered(manifest). A sheet
    with rows missing is worse than no sheet at all: it will be skimmed and trusted, and the chair
    it omitted is the one nobody ever looks at. The obvious way to write this loop --
    `zip(np.ravel(axes), uids)`, which is what both scratch versions of this renderer did -- fails
    exactly that way when the grid is one cell too small: zip stops, silently, and the sheet looks
    complete. Hence the explicit row count and the check.

    A chair whose mesh will not load gets a cell reading FAILED rather than no cell, for the same
    reason, and is listed in `failed`. One unreadable OBJ must not cost the other forty-nine their
    inspection.

    "Offered" and not "accepted": offered() is the set the library actually ships (accepted, and
    selected once a selection exists) -- the 50 carrying a usd_path, not the 161 that passed
    curate. Rendering the pool would bury the shipped chairs in it.

    The OBJ is rendered, not the USD: it is the exported, upright, life-sized, floor-aligned
    geometry the USD was converted FROM, and reading it needs trimesh rather than an Isaac boot.
    That means this sheet cannot catch a defect introduced by conversion itself -- collision is
    what conversion gets wrong here, and repair_collisions is what checks it.

    `off_centre_m` is how far each chair's horizontal bounding-box centre sits from x=y=0, for
    every chair over CONTACT_SHEET_OFF_CENTRE_REPORT_M. Rendering measured this by accident and it
    is reported rather than swallowed: export() never centres x/y, so 31 of the offered fifty are
    off-origin and one is 9.95 m out. See _flat_shade for how the sheet frames around it.

    The world window is `spec.contact_sheet_half_extent_m`, because "a chair rendered small IS
    small" only reads if the window is sized to the library: a table library whose longest
    permitted export is 2.00 m would be cropped in every cell of a 0.7 m chair window.
    """
    spec = spec or CHAIR
    plt, _ = _agg_pyplot()

    entries = sorted(offered(manifest, spec=spec),
                     key=lambda e: (e.get("selection_rank") is None,
                                    e.get("selection_rank", 0), e["uid"]))
    if not entries:
        raise ValueError(f"no offered {spec.name}s to render -- run curate (and select) first")

    out_path = out_path or os.path.join(_library(spec).root(), CONTACT_SHEET_NAME)
    columns = max(1, min(columns, len(entries)))
    blocks = (len(entries) + columns - 1) // columns      # explicit: never let zip() truncate
    rows = blocks * len(azimuths)

    fig, axes = plt.subplots(rows, columns, squeeze=False, facecolor="white",
                             figsize=(2.3 * columns, 2.5 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")

    report = {"path": out_path, "drawn": [], "failed": {}, "overflowed": [], "off_centre_m": {}}

    for index, entry in enumerate(entries):
        block, column = divmod(index, columns)
        uid = entry["uid"]
        cells = [axes[block * len(azimuths) + k][column] for k in range(len(azimuths))]

        try:
            mesh = trimesh.load(entry["obj_path"], process=False, force="mesh")
        except Exception as exc:                                          # noqa: BLE001
            report["failed"][uid] = f"{type(exc).__name__}: {exc}"
            cells[0].text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=11,
                          color="crimson", transform=cells[0].transAxes)
        else:
            offset = float(np.linalg.norm(_horizontal_centre(mesh)))
            if offset > CONTACT_SHEET_OFF_CENTRE_REPORT_M:
                report["off_centre_m"][uid] = offset
            for cell, azim in zip(cells, azimuths):
                if (not _flat_shade(cell, mesh, azim=azim,
                                    half_extent=spec.contact_sheet_half_extent_m)
                        and uid not in report["overflowed"]):
                    report["overflowed"].append(uid)

        rank = entry.get("selection_rank")
        footprint = entry.get("export_footprint_m") or [float("nan"), float("nan")]
        cells[0].set_title(
            f"#{'--' if rank is None else rank}  {uid[:12]}\n"
            f"{max(footprint):.2f} x {min(footprint):.2f} m, h {entry.get('height_m', float('nan')):.2f}\n"
            f"{entry.get('category', '?')}  {entry.get('faces', '?')}f",
            fontsize=7)
        report["drawn"].append(uid)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    missing = {e["uid"] for e in entries} - set(report["drawn"])
    if missing:
        raise RuntimeError(
            f"contact sheet is missing {len(missing)} offered {spec.name}(s): {sorted(missing)}")
    return report


# --- the CLI -----------------------------------------------------------------------------------
#
# Task 4's brief did not leave anything to invoke the pipeline from a shell -- Task 6's "run the
# pipeline to twenty accepted" needs exactly that, so it is added here alongside convert().


def rejection_histogram(manifest, *, spec=None) -> dict:
    """Count of rejected entries per rejection reason, e.g. {"aspect ratio": 12, "face count": 5}.

    Grouped by the CRITERION NAME -- the phrase before ":" in the strings _aspect_ratio and
    _face_count produce ("aspect ratio: normalized extent 0.05 on axis X outside [0.2, 0.95]" ->
    "aspect ratio") -- not the full per-entry reason text, which embeds entry-specific numbers and
    would make every rejection its own bucket ("43 rejected" instead of "31 for aspect ratio, 12
    for face count," the exact distinction CRITERIA's own module comment calls out). This is what
    the CLI prints when `all` stops short of its target, so a human has the count-per-reason in
    front of them before deciding whether to widen anything.
    """
    histogram: dict[str, int] = {}
    for entry in _entries(manifest, spec or CHAIR):
        if entry.get("accepted") is False:
            reason = entry.get("rejected_because", "unknown")
            key = reason.split(":", 1)[0].strip()
            histogram[key] = histogram.get(key, 0) + 1
    return histogram


def run_all(manifest, category, *, target=20, batch_size=20, thresholds=None,
            download=None, uids=None, runner=None, repair_runner=None, spec=None):
    """Run fetch -> curate -> export -> convert -> repair_collisions in a loop, growing the
    manifest by up to `batch_size` NEW candidates per round, until EITHER `target` entries are
    accepted OR a fetch call adds no new uids at all -- i.e. `category`'s whole candidate pool
    (fetch's `uids`, real or injected) has been consumed with fewer than `target` accepted.
    Returns manifest.

    Deliberately does NOT widen `thresholds` to reach `target` when the pool runs dry -- "the twenty
    rule": that is the user's decision to make with the rejection histogram (see
    rejection_histogram) in front of them, not something this function decides on its own. A
    caller that wants more candidates widens the CATEGORY instead (a separate run_all call with a
    different category, e.g. "armchair" after "chair" -- Task 6's own plan) or simply accepts
    fewer than target.

    `thresholds` defaults to DEFAULT_THRESHOLDS, same as curate() itself. `repair_runner` is
    passed through to repair_collisions() exactly as `runner` is passed through to convert().
    """
    spec = spec or CHAIR
    if thresholds is None:
        thresholds = spec.thresholds

    library = _library(spec)

    while True:
        before = len(_entries(manifest, spec))
        manifest = fetch(manifest, category, batch_size, download=download, uids=uids, spec=spec)
        after = len(_entries(manifest, spec))

        manifest = curate(manifest, thresholds=thresholds, spec=spec)
        manifest = export(manifest, spec=spec)
        manifest = convert(manifest, runner=runner, spec=spec)
        manifest = repair_collisions(manifest, runner=repair_runner, spec=spec)

        if len(library.accepted(manifest)) >= target:
            break
        if after == before:
            # fetch added nothing new: category's candidate pool is exhausted.
            break
    return manifest


def _build_arg_parser(spec=None):
    spec = spec or CHAIR
    parser = argparse.ArgumentParser(
        description=(f"Build the {spec.name} manifest under ${spec.manifest_module.upper()}"
                     f"'s root: fetch, curate, select, export, convert."),
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p_fetch = sub.add_parser("fetch", help="Download new candidates from an LVIS category.")
    p_fetch.add_argument("--category", default=spec.default_category)
    p_fetch.add_argument("--count", type=int, default=40)
    p_fetch.add_argument(
        "--seed", default=None,
        help=("Sample the category deep with this seed instead of taking the head of its uid "
              f"list (see _sampled_uids). This library used {spec.sample_seed!r}."),
    )

    sub.add_parser("curate", help="Judge fetched candidates against this type's thresholds.")

    p_select = sub.add_parser(
        "select",
        help=(f"Pick the --target most mutually distinct accepted {spec.name}s by greedy max-min "
              "shape IoU. Run before export/convert: those stages then touch only the selection."),
    )
    p_select.add_argument("--target", type=int, default=50)

    sub.add_parser(
        "prune",
        help=(f"Delete the OBJ/USD artefacts of entries the library no longer offers -- {spec.name}s"
              " a re-curate rejected, or select() passed over -- so usd/ matches the selection."),
    )

    sub.add_parser("export", help="Stand offered candidates upright and write life-sized OBJs.")
    p_convert = sub.add_parser(
        "convert", help="Convert exported OBJs to USD with a convexDecomposition collider.")
    p_convert.add_argument(
        "--limit", type=int, default=None,
        help=(f"Convert at most this many {spec.name}s, then save and stop. Each conversion boots "
              "Isaac Sim (~40 s), and the manifest is saved once per invocation -- so chunking a "
              "long batch checkpoints it. Re-run to continue; converted entries are skipped."),
    )
    sub.add_parser(
        "repair-collision",
        help=(
            "Verify (and repair if needed) that every converted USD's mesh actually carries "
            "collision geometry -- works around a race in the vendored MeshConverter, see "
            "repair_collisions()'s docstring."
        ),
    )

    p_sheet = sub.add_parser(
        "contact-sheet",
        help=(f"Render one labelled image grid of every offered {spec.name}, two viewpoints each, "
              "for a human to actually look at. The geometric criteria cannot tell furniture from "
              "a sculpture and select() prefers outliers -- this is the only gate that can."),
    )
    p_sheet.add_argument(
        "--out", default=None,
        help=f"Output PNG. Default: <asset root>/{CONTACT_SHEET_NAME} (never in git).")
    p_sheet.add_argument("--columns", type=int, default=10,
                         help=f"{spec.name.capitalize()}s per row. Default 10.")
    p_sheet.add_argument("--dpi", type=int, default=100)

    p_all = sub.add_parser(
        "all",
        help="Run fetch -> curate -> export -> convert until --target are accepted or the pool is exhausted.",
    )
    p_all.add_argument("--category", default=spec.default_category)
    p_all.add_argument("--target", type=int, default=20)
    p_all.add_argument("--batch-size", type=int, default=20)

    return parser


def main(argv=None, *, download=None, uids=None, runner=None, repair_runner=None, spec=None):
    """CLI entry point. `download`, `uids`, `runner`, and `repair_runner` are the same injection
    points fetch(), convert(), and repair_collisions() already take -- exposed here so a test can
    drive the CLI's argument wiring (argv in, manifest out) without ever touching the network or
    booting Isaac. Loads the manifest, runs the requested stage, saves it back, prints a short
    report, and returns the manifest.

    `spec` selects the furniture type; build_table_manifest.py's own main() is this function with
    spec=TABLE, so the two CLIs cannot drift apart.
    """
    spec = spec or CHAIR
    args = _build_arg_parser(spec).parse_args(argv)

    library = _library(spec)
    manifest = library.load()

    if args.stage == "fetch":
        manifest = fetch(manifest, args.category, args.count, download=download, uids=uids,
                         seed=args.seed, spec=spec)
    elif args.stage == "curate":
        manifest = curate(manifest, thresholds=spec.thresholds, spec=spec)
    elif args.stage == "select":
        manifest = select(manifest, args.target, spec=spec)
    elif args.stage == "prune":
        before = sum(1 for c in _entries(manifest, spec)
                     for f in ("obj_path", "usd_path") if c.get(f))
        manifest = prune(manifest, spec=spec)
        after = sum(1 for c in _entries(manifest, spec)
                    for f in ("obj_path", "usd_path") if c.get(f))
        print(f"pruned {before - after} artefact(s) belonging to {spec.name}s the library no "
              "longer offers")
    elif args.stage == "export":
        manifest = export(manifest, spec=spec)
    elif args.stage == "convert":
        manifest = convert(manifest, runner=runner, limit=args.limit, spec=spec)
    elif args.stage == "repair-collision":
        manifest = repair_collisions(manifest, runner=repair_runner, spec=spec)
    elif args.stage == "contact-sheet":
        report = contact_sheet(manifest, out_path=args.out, columns=args.columns, dpi=args.dpi,
                               spec=spec)
        print(f"contact sheet: {len(report['drawn'])} offered {spec.name}(s), "
              f"{len(CONTACT_SHEET_AZIMUTHS)} viewpoints each -> {report['path']}")
        if report["failed"]:
            print(f"{len(report['failed'])} {spec.name}(s) drew a FAILED cell rather than no cell:")
            for uid, reason in sorted(report["failed"].items()):
                print(f"  {uid}: {reason}")
        if report["overflowed"]:
            print(f"{len(report['overflowed'])} {spec.name}(s) overflowed the shared "
                  f"{spec.contact_sheet_half_extent_m} m window and are cropped: "
                  f"{', '.join(report['overflowed'])}")
        if report["off_centre_m"]:
            worst = sorted(report["off_centre_m"].items(), key=lambda kv: -kv[1])
            print(f"{len(worst)} {spec.name}(s) are exported off-origin horizontally; "
                  "furthest five:")
            for uid, metres in worst[:5]:
                print(f"  {uid}: {metres:.3f} m")
    elif args.stage == "all":
        manifest = run_all(
            manifest, args.category, target=args.target, batch_size=args.batch_size,
            download=download, uids=uids, runner=runner, repair_runner=repair_runner, spec=spec,
        )

    library.save(manifest)

    chairs = _entries(manifest, spec)
    n_accepted = len(library.accepted(manifest))
    print(f"{spec.entries_key}: {len(chairs)} total, {n_accepted} accepted")

    if manifest.get("selection"):
        selection = manifest["selection"]
        print(f"selection: {selection['selected']}/{selection['target']} chosen from a pool of "
              f"{selection['pool']} accepted, by {selection['rule']}")
        if selection["selected"] < selection["target"]:
            # WHICH constraint stopped it, measured rather than assumed. This used to say "the
            # accepted pool is smaller than the target" unconditionally, and the room allow-list
            # made that untrue: a cap of c admits at most ceil(c x target) per category, so with
            # k categories in the pool the ceiling is a function of the CAP, and it can bind hard
            # while the pool is twice the target. Telling a user to fetch more candidates when
            # more candidates cannot possibly help is worse than saying nothing.
            cap_count = selection.get("category_cap_count")
            per_category = collections.Counter(
                e.get("category") for e in _entries(manifest, spec)
                if e.get("accepted") is True and not e.get("not_for_room"))
            ceiling = (sum(min(cap_count, n) for n in per_category.values())
                       if cap_count else selection["pool"])
            if cap_count and ceiling < selection["target"]:
                print(
                    f"stopped at {selection['selected']}/{selection['target']} selected -- the "
                    f"CATEGORY CAP binds, not the pool. {selection['pool']} accepted are in the "
                    f"pool, but no category may supply more than {cap_count}, and the "
                    f"{len(per_category)} categories present can supply at most {ceiling} between "
                    f"them ({', '.join(f'{k}:{min(cap_count, n)}' for k, n in sorted(per_category.items()))}). "
                    "Fetching more candidates in these categories cannot help. Not widening the "
                    "cap to reach the target -- that is your call."
                )
            else:
                print(
                    f"stopped at {selection['selected']}/{selection['target']} selected -- the "
                    "accepted pool is smaller than the target. Not widening a threshold to reach "
                    "it -- that is your call. Fetch more candidates instead."
                )
            histogram = rejection_histogram(manifest, spec=spec)
            if histogram:
                print("rejections by reason:")
                for reason, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
                    print(f"  {reason}: {count}")

    converted = [c for c in chairs if c.get("usd_path")]
    if converted:
        n_collision = sum(1 for c in converted if c.get("has_collision") is True)
        print(f"collision: {n_collision}/{len(converted)} converted {spec.name}s "
              "carry collision geometry")

    if args.stage == "all" and n_accepted < args.target:
        print(
            f"stopped at {n_accepted}/{args.target} accepted -- category "
            f"'{args.category}' candidate pool exhausted ({len(chairs)} candidates fetched). "
            "Not widening a threshold to reach target -- that is your call."
        )
        histogram = rejection_histogram(manifest, spec=spec)
        if histogram:
            print("rejections by reason:")
            for reason, count in sorted(histogram.items(), key=lambda kv: -kv[1]):
                print(f"  {reason}: {count}")

    return manifest


if __name__ == "__main__":
    main()
