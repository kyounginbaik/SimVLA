#!/usr/bin/env python3
"""Pipeline that builds the Objaverse TABLE library: fetch -> curate -> select -> export ->
convert -> repair-collision -> contact-sheet, writing $TABLE_OBJ_DIR/table_manifest.json.

This file is a FurnitureType and a CLI. Every stage lives in build_chair_manifest.py, which is the
shared engine (see its module docstring for why it kept that name), and the pipeline is driven from
here by passing `spec=TABLE`. Copying the engine would have forked the MeshConverter collision
workaround, the convert retry, the cross-process-stable uid seeding, the exact-duplicate detector,
the off-origin export fix and the contact sheet -- every one of them expensive to get right and
none of them about chairs.

WHAT A TABLE NEEDS THAT A CHAIR DOES NOT, and vice versa. All of it measured first, over 60
downloaded meshes every one of which was rendered and hand-labelled:

  1. NO FACING AXIS. Chairs separate 170x against their own noise floor on the asymmetry measure;
     tables separate 9x, and only 2 of 15 usable tables clear the chair gate's own 0.10 confidence
     cut. `_facing_axis` and `facing_direction` are therefore not ported, not re-tuned -- absent.
     A table has a long axis, not a front, and the long axis comes free from the bounding box.
  2. A WORK-SURFACE PLANE, which is both the up-axis validator and the criterion that catches the
     whole `desk` failure mode. See _work_surface.
  3. A DIFFERENT HEIGHT MODEL. Fixed 0.74 m to the work surface, not a per-uid Gaussian draw to the
     top of the bounding box. See TABLE_SURFACE_HEIGHT_M.
  4. A FOOTPRINT BAND, top and bottom. Scaling by height does not control plan area, and at a
     0.74 m surface these meshes reach 2.59 m long against a procedural maximum of 1.80 m.
  5. NO ASPECT-RATIO BAND. Usable h/L spans [0.285, 0.893] and rejects span [0.204, 1.145]; 34 of
     40 rejects sit inside the usable band. A band tight enough to matter would cost real tables.
  6. A DIFFERENT FACE-COUNT BAND. The chair floor of 100 rejects two good tables (60 and 92 faces)
     and the chair ceiling of 200k rejects two more, one of them among the best.
  7. A DIFFERENT SHAPE-DESCRIPTOR NORMALISATION. See TableType.grid_scale.

The up axis is the one place the plan predicted a fork and none is needed: build_chair_manifest's
`_up_axis` already trusts glTF's declared +Y and falls back to argmax(extents) only for a format
with no convention. That is exactly this gate's recommendation, so it is shared, not duplicated.

WHERE THE SOURCE CATEGORIES COME FROM. The gate's hand-labelled yields: `table` 94% (15/16),
`desk` 17% (4/24), `dining_table` 5% (1/20). `dining_table` is not merely poor but disqualifying --
15 of its 20 are dining SETS carrying their own 1-8 chairs, and the wizard places its own chairs on
a ring derived from the table's bounds, so such a table would arrive surrounded by a second,
misaligned set. It is not in DEFAULT_CATEGORIES at any position.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_chair_manifest as pipeline                                       # noqa: E402
from build_chair_manifest import AXES, FurnitureType                          # noqa: E402

# --- the work-surface plane ---------------------------------------------------------------------
#
# One measurement answers three questions at once, which is why it is the centre of this file:
# is Y really up, is anything sitting above the tabletop, and does anything stick out past it.
# The findings doc's definition, quoted: "the highest horizontal band (2.5 % of height, face
# normals within 20 deg of +-Y) whose projected area covers >= 50 % of the footprint, expressed as
# a fraction of total height", and overhang = "full horizontal bounding-box area divided by that
# plane's own bounding-box area".

#: Thickness of the band, as a fraction of the mesh's height along the up axis. A tabletop slab is
#: 2-5 cm on a 0.74 m table, so 2.5% (1.85 cm) selects the top surface without swallowing the rails
#: below it.
WORK_SURFACE_BAND_FRAC = 0.025

#: How far from vertical a face normal may lean and still count as part of a horizontal plane.
WORK_SURFACE_NORMAL_TOL_DEG = 20.0

#: Fraction of the mesh's own horizontal bounding-box area the band must cover to BE a work
#: surface. Half: a tabletop covers essentially all of its own footprint, while a dining set's
#: tabletop covers only 33-44% of a bounding box enlarged by the chairs around it -- measured
#: directly on four of the eight ornate sets in the gate's sample. That gap is why 0.5 separates.
WORK_SURFACE_COVERAGE = 0.5


def _work_surface(mesh, up_idx):
    """{"work_surface_frac", "overhang"} for `mesh`, or None when it has no work surface at all.

    COVERAGE IS MEASURED PER FACING, NOT SUMMED, and that is not a detail. Summing |n . up| * area
    over every horizontal face in the band double-counts a slab thin enough that its top and its
    underside both land in the same 2.5% band, so a plane covering a quarter of the footprint reads
    as covering half. Measured consequence on the gate's own sample: a summed measure finds a
    "work surface" halfway up a staircase (`7aa2f0637fde`, frac 0.50 against the gate's 0.02) and
    on a plank-floored workshop scene (`cc7460534919`, 0.41 against 0.02). Taking the larger of the
    upward-facing and downward-facing totals costs nothing, cannot double-count a closed slab, and
    -- unlike simply keeping upward normals -- still works on a mesh whose winding is inverted.

    Reproduction against the gate: this function was run over all 60 of the gate's meshes and
    agrees with its published per-mesh table on 58/60 for work_surface_frac and 56/60 for overhang
    (2 decimal places), and reproduces its headline exactly on 59 of 60 -- keeps 18/20 hand-labelled
    usable, admits 1 of 40 rejects. The one disagreement is `397ad6222cf6`, a dining SET whose
    overhang this function reads as 1.00 and the gate as 1.50; it is in `dining_table`, the category
    this pipeline does not source from. Stated rather than hidden: the gate's "0/40 false accepts"
    is a fitted number and this is a 1/40 reproduction of it.
    """
    bounds = np.asarray(mesh.bounds)
    extents = bounds[1] - bounds[0]
    height = float(extents[up_idx])
    horizontal = [i for i in range(3) if i != up_idx]
    footprint_area = float(extents[horizontal[0]] * extents[horizontal[1]])
    if height <= 0 or footprint_area <= 0 or len(mesh.faces) == 0:
        return None

    dot = np.asarray(mesh.face_normals)[:, up_idx]
    flat = np.flatnonzero(np.abs(dot) >= math.cos(math.radians(WORK_SURFACE_NORMAL_TOL_DEG)))
    if len(flat) == 0:
        return None

    centre = np.asarray(mesh.triangles_center)[flat, up_idx]
    order = np.argsort(centre)
    flat, centre = flat[order], centre[order]
    projected = np.asarray(mesh.area_faces)[flat] * np.abs(dot[flat])

    # Sliding band [c - band, c] for every candidate face height c, by cumulative sum: O(n log n)
    # rather than the O(n^2) the obvious loop would cost on a 1.3 M-face candidate.
    lower = np.searchsorted(centre, centre - WORK_SURFACE_BAND_FRAC * height, side="left")
    upper = np.arange(len(centre)) + 1
    up_cum = np.concatenate([[0.0], np.cumsum(projected * (dot[flat] > 0))])
    down_cum = np.concatenate([[0.0], np.cumsum(projected * (dot[flat] < 0))])
    covered = np.maximum(up_cum[upper] - up_cum[lower], down_cum[upper] - down_cum[lower])

    qualifying = np.flatnonzero(covered >= WORK_SURFACE_COVERAGE * footprint_area)
    if len(qualifying) == 0:
        return None
    top = int(qualifying[-1])                      # the HIGHEST band that qualifies

    plane_faces = flat[lower[top]:top + 1]
    plane_vertices = mesh.vertices[mesh.faces[plane_faces].reshape(-1)][:, horizontal]
    plane_extents = plane_vertices.max(axis=0) - plane_vertices.min(axis=0)
    plane_area = float(plane_extents[0] * plane_extents[1])
    return {
        "work_surface_frac": (float(centre[top]) - float(bounds[0][up_idx])) / height,
        "overhang": footprint_area / plane_area if plane_area > 0 else float("inf"),
    }


# --- the height constant ------------------------------------------------------------------------

#: Height of the WORK SURFACE above the floor, in metres, that every exported table is scaled to.
#:
#: Objaverse carries no units at all -- raw max extents across the gate's 20 usable meshes span
#: 1780x -- so a constant has to come from somewhere. It is not invented here: kitchen_build's
#: TABLE_VARIANTS sets height=0.74 on NINE of its twelve procedural TableAsset variants, and
#: TableAsset's height spans the full bounding box with the surface on top, so the two numbers are
#: directly comparable. These meshes land in the same gallery, beside those twelve, and are seated
#: by the same chair ring; a different number would make them visibly wrong and would silently
#: change the ring geometry. It is also the real-world standard (EN 527 / ANSI-BIFMA fixed-height
#: desks 0.72-0.75 m; dining tables 0.73-0.76 m).
#:
#: THE CHAIRS' 0.85 m IS WRONG HERE IN A SPECIFIC WAY. That figure is measured to the top of a
#: BACKREST. Applied to a table it would put the work surface 11 cm above standard, above a seated
#: adult's elbow, and higher than the backrest of every chair in the library -- so no chair could
#: tuck under. The chair library exports at 0.75-0.96 m to the backrest top, whose seat sits about
#: 0.30-0.40 m lower, which is what gives 0.74 m its 0.28-0.30 m of seat-to-surface clearance.
#:
#: AND THERE IS NO PER-UID GAUSSIAN DRAW. Chairs sample N(0.85, 0.05) because real chairs genuinely
#: vary in back height. Real work surfaces do not -- a desk is 0.74 m because a person sits at it --
#: and a fixed constant is what keeps mesh tables consistent with the nine procedural variants that
#: are all exactly 0.74. This is a deliberate non-port, not an oversight.
TABLE_SURFACE_HEIGHT_M = 0.74


# --- criteria -------------------------------------------------------------------------------

def _no_work_surface(measurements, thresholds, accepted=()):
    """Reject a mesh with no horizontal plane covering >= WORK_SURFACE_COVERAGE of its own
    footprint anywhere along the declared up axis.

    This is the UP-AXIS VALIDATOR, which is why it is first. The gate's recommendation was "trust
    glTF Y-up; validate, never detect": taking each axis in turn as the candidate up, a qualifying
    plane exists along Y on 20/20 usable meshes against X 3/20 and Z 2/20, so a mesh that has none
    along Y is telling you it is not a table standing the right way up, whatever its label says.
    On its own it catches 9 of the gate's 40 rejects, including all eight ornate dining sets.

    It also makes this pipeline structurally incapable of the blind spot the chair pipeline shipped
    with for its whole life: while `up` was argmax(extents), `normalized_extents[up]` was
    identically 1.0 and `_aspect_ratio` could never reject anything for being longer than it is
    tall. Nothing here is 1.0 by construction. work_surface_frac ranges over [0.02, 1.00] on the
    gate's 60 and overhang over [1.00, 2.74]; a flat billboard lying in the horizontal plane has
    zero height and is rejected here, and one standing vertically has no horizontal faces and is
    rejected here too.
    """
    if measurements.get("work_surface_frac") is None:
        return ("no work surface: no horizontal plane covers "
                f"{WORK_SURFACE_COVERAGE:.0%} of the footprint anywhere along "
                f"{measurements.get('up_axis', '?')}")
    return None


def _work_surface_at_top(measurements, thresholds, accepted=()):
    """Reject when the work-surface plane is not near the top of the mesh.

    Findings doc, proposed threshold #2: usable meshes span [0.65, 1.00] with 18 of 20 at exactly
    1.00 -- for a bare table the work surface IS the top of the bounding box -- against a reject
    range of [0.025, 1.00] with a median of 0.625. At 0.85 this keeps 18/20 usable and admits 3/40
    rejects on its own.

    This is the criterion that catches the whole `desk` failure mode. A monitor, hutch, partition,
    overhead shelf or lamp above the work surface pushes the fraction down; so do a dining set's
    chair backs. Its cost is stated in the findings doc rather than hidden: it drops
    `e8d10d0d2356` (0.65, a glass desk with trinkets on top) and `810764ab68f2` (0.775, the only
    usable `dining_table` in the whole sample).
    """
    value = measurements.get("work_surface_frac")
    minimum = thresholds["work_surface_frac_min"]
    if value is not None and value < minimum:
        return (f"work surface: plane at {value:.2f} of the height, below {minimum} -- something "
                "sits above the work surface")
    return None


def _overhang(measurements, thresholds, accepted=()):
    """Reject when the mesh is wider than its own work surface.

    Findings doc, proposed threshold #3: full horizontal bounding-box area divided by the work
    surface's own bounding-box area. A bare table is 1.0 by construction; usable meshes span
    [1.000, 1.175] with 16 of 20 at exactly 1.000, against a reject range of [1.000, 2.74].

    THE HONEST CAVEAT, from the findings doc, because "100% precision" would overstate it: this and
    _work_surface_at_top are NOT independent evidence. Both key off one geometric fact -- is
    anything in this mesh outside or above the tabletop -- which is exactly what a welded-in chair,
    monitor or partition is. That one fact separates the gate's sample cleanly; two criteria
    agreeing about it is not two confirmations.
    """
    value = measurements.get("overhang")
    limit = thresholds["overhang_max"]
    if value is not None and value > limit:
        return (f"overhang: bounding box is {value:.2f}x the work surface's own, above {limit} -- "
                "something sticks out past the table")
    return None


def _footprint(measurements, thresholds, accepted=()):
    """Reject when the table's LONGER horizontal extent, after export's uniform scaling to a 0.74 m
    work surface, falls outside [footprint_min_m, footprint_max_m] metres.

    Scaling by height does not control plan area, and this is the same defect the chair library
    shipped with pointed the other way ("some chairs are too narrow to be furniture"). Scaled to a
    0.74 m surface the gate's 20 usable meshes imply longer horizontals of 0.83 to 2.59 m, against
    twelve procedural variants that span 0.70-1.80 m wide. The worst offender, `9e19bc36beef`, is
    an ornate COFFEE table (h/L 0.29) that becomes a 2.59 x 1.21 m banquet table when its top is
    raised to desk height -- the `table` LVIS category mixes coffee tables with dining tables and
    one height constant cannot serve both.

    [0.70, 2.00] is the findings doc's proposal: the low bound is the procedural minimum and sits
    below the observed usable minimum of 0.83 m so it rejects none of them, and the high bound
    rejects 3 of the 20 (2.09, 2.22 and 2.59 m) at 0.20 m of margin above the procedural maximum.
    It must be decided BEFORE the chair ring is swept across these tables, not discovered during
    it: a 2.6 m table is exactly the case the ring was never verified against.

    Fails OPEN when the entry carries no footprint -- measured by a caller that passed no uid, or a
    mesh with no work-surface plane to scale from, which _no_work_surface has already rejected.
    """
    footprint = measurements.get("export_footprint_m")
    if not footprint:
        return None
    longer, shorter = max(footprint), min(footprint)
    low, high = thresholds["footprint_min_m"], thresholds["footprint_max_m"]
    if longer < low or longer > high:
        return (f"footprint: {longer:.2f} m x {shorter:.2f} m at a "
                f"{TABLE_SURFACE_HEIGHT_M} m work surface -- longer horizontal extent outside "
                f"[{low}, {high}] m")
    return None


#: Applied in order; curate records only the first failure, so the order decides which bucket a
#: rejection lands in and the histogram stays actionable.
#:
#: 1-3 are the three facts that say whether this mesh is a standalone table at all, cheapest and
#: most structural first: no plane means the up axis is wrong or it is not furniture; a low plane
#: means something is above the work surface; a high overhang means something is beside it.
#: 4 is a conversion-cost guard, not a quality signal (see build_chair_manifest._face_count).
#: 5 judges the size it will actually ship at.
#: 6 is LAST for the reason the chair pipeline gives: every reason above is a property of the mesh
#: alone and stays true however the corpus changes, while "too similar to <uid>" is only true
#: relative to what else was accepted that day -- and a mesh that is not a usable table must never
#: become the incumbent that shadows a later, better one.
CRITERIA = [_no_work_surface, _work_surface_at_top, _overhang,
            pipeline._face_count, _footprint, pipeline._distinctness]

#: The single source of truth for curate's table thresholds. Every number is the findings doc's
#: proposal with the measurement it rests on; none was chosen to reach a count.
#:
#: work_surface_frac_min 0.85 -- usable [0.65, 1.00], 18/20 at exactly 1.00; reject median 0.625.
#: overhang_max          1.25 -- usable [1.000, 1.175], 16/20 at exactly 1.000; reject max 2.74.
#: face_floor              50 -- the lowest usable face count is 60 (`b874f4ffbafd`, which rendered
#:                               as a perfectly clean table), with 92 and 108 close behind. This is
#:                               HALF the chair pipeline's floor of 100, which would reject two
#:                               good tables here.
#: face_ceiling       700_000 -- the largest usable is 602,377 (`810764ab68f2`). THREE TIMES the
#:                               chair ceiling of 200,000, which would reject two usable tables,
#:                               one of them among the best (`cccefbf8c735`, 487k). A cost guard,
#:                               not a quality signal: reject median 11,500 against usable median
#:                               1,107 -- if anything the rejects are heavier.
#: footprint_min_m       0.70 -- the procedural minimum width; observed usable minimum is 0.83 m,
#:                               so this rejects none of them.
#: footprint_max_m       2.00 -- 0.20 m above the procedural maximum of 1.80; rejects 3 of 20.
#: duplicate_iou_max     0.98 -- inherited unchanged from the chair pipeline, and inherited as what
#:                               it is there: an EXACT RE-UPLOAD detector, nothing weaker. The
#:                               empty interval that justifies it (0.9737 to 1.0000) was measured
#:                               on chairs, not on tables, so this number is not re-derived here --
#:                               identical geometry still scores 1.0000 whatever the corpus, which
#:                               is the only claim being made. Whether this table corpus has a
#:                               near-duplicate continuum is the contact sheet's question.
DEFAULT_THRESHOLDS = {
    "work_surface_frac_min": 0.85,
    "overhang_max": 1.25,
    "face_floor": 50,
    "face_ceiling": 700_000,
    "footprint_min_m": 0.70,
    "footprint_max_m": 2.00,
    "duplicate_iou_max": 0.98,
}

#: Seed for the deep sample over each LVIS category (build_chair_manifest._sampled_uids). A STRING,
#: not an int: random.Random(str) seeds from a SHA-512 of the bytes with no per-process salt, so the
#: draw reproduces across interpreters without pinning PYTHONHASHSEED.
SAMPLE_SEED = "objaverse-tables-2026-08-13"

#: LVIS categories to draw from, in the order a campaign should exhaust them. The order is measured
#: accept yield over each category's WHOLE pool, run 2026-08-13, not the gate's samples:
#:
#:   table          101 candidates -> 68 accepted (67%).  The gate hand-labelled 15/16 usable.
#:   desk            76 candidates -> 12 accepted (16%).  The gate measured 4/24 = 17% on a sample;
#:                                                        the full pool agrees almost exactly.
#:   kitchen_table   48 candidates ->  9 accepted (19%).  Never sampled by the gate. 27 of its 48
#:                                                        fail on work surface -- it is kitchen
#:                                                        SCENES, the `desk` failure mode again.
#:   coffee_table    51 candidates -> not fetched. Listed last because a coffee table's proportions
#:                                                        are exactly what _footprint rejects once
#:                                                        its top is raised to 0.74 m.
#:
#: `kitchen_table` is demoted below `desk` on a second, worse measurement that only the contact
#: sheet could make: of the 7 `kitchen_table` entries that reached the first sheet of fifty, FIVE
#: were architectural models -- multi-storey structural frames and a room shell (see
#: MANUAL_REJECTS). Its 19% accept rate overstates it. Fetch from it only with a visual pass ready.
#:
#: `dining_table` is absent on purpose and must not be added: 15 of its 20 sampled meshes are dining
#: SETS with their own 1-8 chairs welded in, which would collide with the wizard's own chair ring.
DEFAULT_CATEGORIES = ("table", "desk", "kitchen_table", "coffee_table")

#: NO CATEGORY CAP, and that is a decision from the same measurement the chair cap came from.
#:
#: The chair library caps a single LVIS category at 24% of a selection because `deck_chair` supplied
#: 13 of 50 from a 24-uid list whose tail was junk. The argument recorded there is that a cap is
#: category-BLIND -- it cannot prefer good categories, so it must be set loose enough not to trade
#: quality for balance. Here the yields are far more lopsided than the chair pool's ever were: 94%
#: for `table` against 17% for `desk` and 5% for the category this project is named after. A cap
#: that pushed the selection away from `table` and toward `desk` would import office scenes,
#: monitors and cubicle partitions to buy a balance nobody asked for. Diversity within `table` is
#: select()'s greedy max-min job, which needs no cap, and the thing that will show whether it worked
#: is the contact sheet.
DEFAULT_CATEGORY_CAP = None

#: Tables a human looked at on the contact sheet and rejected, uid -> reason. Same contract as the
#: chair library's MANUAL_REJECTS: applied by curate BEFORE its already-judged skip, so a re-curate
#: cannot resurrect one and a previously-accepted entry is retracted rather than left standing.
#:
#: This exists as DATA rather than as a criterion because no number separates its members from
#: furniture. The chair library's list holds a licence-text placard, a wrought-iron railing and a
#: loft bed -- correct, well-formed, chair-proportioned solids. The desk gate's own 60-mesh sample
#: contained two staircases and an escalator labelled `desk`, visible only after stripping a ground
#: plane that dominated their bounds. A human pass is the only gate that sees these, and its result
#: has to outlive the next rebuild or the next selection picks them straight back.
#:
#: NOT a place for a table that is merely inconvenient. Everything here is "not a table, or not one
#: object".
#:
#: WHAT THE FIRST SHEET FOUND, and why it is the same finding the desk gate made about `desk`.
#: Five of the first fifty are architectural models: multi-storey structural frames and a room
#: shell, every one of them from `kitchen_table`, and every one of them passing every geometric
#: criterion with work_surface_frac 0.97-1.00 and overhang 1.00-1.06. They pass BECAUSE a
#: building's top floor slab genuinely is a horizontal plane covering its own footprint with
#: nothing above it or outside it -- which is the criterion's definition of a work surface. No
#: threshold reaches them. This is the `desk` staircases again, in a category the gate never
#: sampled.
MANUAL_REJECTS = {
    #: Found 2026-08-15 by rendering the offered fifty at four azimuths each. Not a taste verdict:
    #: `fcf21ed719f1` carries THREE flat, zero-thickness glyph components (99, 50 and 6 faces) all
    #: at z = 0.497, standing in one vertical plane beside the table -- an extruded "Text" label
    #: welded into the same mesh, the chair library's `39059edc24d5` failure mode exactly. They are
    #: also what its overhang of 1.187 is measuring, so objaverse_tables.md's "one table wears a
    #: tablecloth" was reading the wrong uid: the table that actually wears a cloth is
    #: `63e4d8faac73`, which is still offered.
    "fcf21ed719f14eadb1a1b22c914865d1":
        "not one table: flat zero-thickness text glyphs welded into the mesh beside the table -- "
        "a watermark or licence label, and the reason its overhang reads 1.19",
    #: Drawn by the 2026-08-15 backfill and rejected on sight. Measured rather than eyeballed:
    #: every support component sits at x = +0.37..+0.40 under a top spanning x = -0.55..+0.57, so
    #: half the top is over nothing, and two thin rods lean out past the footprint entirely. No
    #: criterion here looks below the work surface, which is why this needs a hand verdict.
    "6147f2acab3c44c09e7883869d2c360a":
        "not one table: the entire base sits under one end of the top (all support components at "
        "x ~ +0.38 under a top spanning -0.55..+0.57) plus two loose slanted rods",
    "4472511a6f4b4fb980c5c014faf20dea":
        "not a table: a multi-storey structural frame -- columns, beams, several floor slabs and "
        "base plates",
    "efe3e41775a646c2bc3109923fec9cf8":
        "not a table: a structural frame -- a forest of columns carrying floor slabs",
    "cc1e1a66e7bc431487a05388c24b1449":
        "not a table: a hollow open-topped box -- a floor and three walls, a room shell",
    "fed9507fe93349f0b330acec71f4fe51":
        "not a table: a multi-level structural frame with columns of two lengths",
    "cfe6466ff2474104bfde9ec8ec9a8b71":
        "not a table: a structural frame in two separate column-and-slab blocks",
}

#: The LVIS categories a KITCHEN library draws from. Applied by select() as rule 0, never by
#: curate. `desk` goes -- an office desk is not a kitchen table, and this is the same category
#: whose whole failure mode (monitors, partitions, swivel chairs welded in) already costs six
#: downloads per usable asset. `table` and `kitchen_table` stay. It cost the library 6 of its
#: fifty, all `desk`.
#:
#: Unlike the chair library, this costs the count NOTHING: DEFAULT_CATEGORY_CAP is None here, so
#: two categories can supply a whole selection, and `table` alone accepts 68 against a target of
#: 50.
KITCHEN_CATEGORIES = ("table", "kitchen_table")

#: Tables that are real, whole, correctly-curated furniture and still do not belong in a kitchen,
#: uid -> reason. Applied by select() as rule 0, alongside KITCHEN_CATEGORIES. See
#: build_chair_manifest.KITCHEN_REJECTS for why this is a separate dict from MANUAL_REJECTS.
#:
#: Every uid below was rendered at four azimuths and looked at. Four more were proposed from the
#: contact sheet and OVERRULED (see objaverse_tables.md, "Why these fifty are kitchen tables") --
#: among them a round pedestal dining table and a 2.00 x 1.20 m farmhouse trestle, both of which
#: are exactly what a kitchen wants.
KITCHEN_REJECTS = {
    "8be12b142f624dac97ebf0da44f12eb9":
        "not kitchen furniture: a hall console -- 0.93 x 0.27 m, too shallow to eat at and too "
        "narrow to seat anyone; the smallest thing in the library",
    "0c1ec5c3e561484686b251fec88189f8":
        "not kitchen furniture: a height-adjustable office desk -- two T-foot columns with base "
        "plates and a cable tray under the rear edge (labelled kitchen_table by LVIS; it is not)",
    "b5337ed2243e4ef0a0d669efd4c02891":
        "not kitchen furniture: two A-frame sawhorse trestles with loose planks laid across them",
    # --- the backfill wave, 2026-08-15 -----------------------------------------------------------
    "8f964a73dba945eca40abead9e4e12e5":
        "not kitchen furniture: a double-bay workbench/staging frame -- THREE full-footprint decks "
        "stacked at z = 0.643, 0.701 and 0.740 m, so its 'work surface' is the top shelf of a "
        "shelving unit",
    "0fd1d0546f8e422b97a4fd5cfe959d2e":
        "not kitchen furniture: a camping/field table -- recessed rectangular wells and catch tabs "
        "cut into the top, on four thin splayed rod legs; nothing sits flat on it",
    "625e45f9aca946ee8307ae335509822e":
        "not kitchen furniture: a trough/planter on an X-trestle base -- walls rise 0.10 m above "
        "the 0.74 m plane round the whole 1.02 x 1.47 m footprint, so an object placed on "
        "table/top lands inside a box",
    #: NOT a taste verdict, and stated as such rather than hidden among them. This is the table
    #: objaverse_tables.md already names in test_table_variants.TABLES_WITH_NO_WORK_SURFACE: its
    #: top is 1664 separate up-facing triangles at z = 0.721 that trimesh never merges into a facet
    #: above label_support's 0.01 m^2 floor, so "On the table" places nothing on it. It renders as
    #: a perfectly ordinary four-leg table. The principled home for it is a MEASURED criterion
    #: ("no mergeable support facet"), not a hand list; that criterion does not exist yet, and
    #: shipping a table the wizard cannot put anything on is worse than recording the debt here.
    "db9344bf45ca434fbdf4dcefab54e5c2":
        "unusable in a kitchen: no mergeable support surface -- its top is 1664 unmerged triangles "
        "below label_support's facet floor, so nothing can be placed on it",
}

#: Half-width in metres of the shared world window the contact sheet frames every cell to.
#:
#: Derived from footprint_max_m, not from the corpus. The chair constant (0.7 m) was fitted to the
#: widest chair then offered, and had to be revisited when the library changed; a table library has
#: a hard upper bound on plan size, so the window can be sized to the BOUND once. The silhouette's
#: half-width at the sheet's azimuth is half the plan diagonal, worst case for a square table at the
#: bound: hypot(2.00, 2.00) / 2 = 1.414 m. 1.45 covers it with margin.
#:
#: Measured, not assumed: at 1.25 m one of the fifty overflowed and was silently cropped --
#: `c801881f25e3`, a 1.91 x 1.91 m square table whose half-diagonal is 1.351 m. contact_sheet
#: reported it rather than hiding it, which is what that report is for.
#:
#: The cost of a window this wide is that a 0.74 m table occupies about a third of the cell height.
#: The alternative -- fitting each cell to its own mesh -- normalises scale away, and "this one is
#: enormous" is one of the things the sheet exists to show.
CONTACT_SHEET_HALF_EXTENT_M = 1.45


class TableType(FurnitureType):
    """The table library, as a FurnitureType."""

    name = "table"
    entries_key = "tables"
    manifest_module = "table_manifest"
    default_category = DEFAULT_CATEGORIES[0]

    thresholds = DEFAULT_THRESHOLDS
    criteria = tuple(CRITERIA)
    manual_rejects = MANUAL_REJECTS

    sample_seed = SAMPLE_SEED
    category_cap = DEFAULT_CATEGORY_CAP
    room_categories = KITCHEN_CATEGORIES
    room_rejects = KITCHEN_REJECTS

    contact_sheet_half_extent_m = CONTACT_SHEET_HALF_EXTENT_M

    def measure_extras(self, mesh, up_idx) -> dict:
        """The work-surface plane and the overhang ratio, recorded whatever the verdict so a later
        threshold change can re-decide from the manifest alone. None when the mesh has no plane at
        all, which is itself the recorded evidence for _no_work_surface's rejection.

        No facing axis: see this module's docstring.
        """
        return _work_surface(mesh, up_idx) or {"work_surface_frac": None, "overhang": None}

    def grid_scale(self, mesh) -> float:
        """The LARGEST extent, where a chair uses its height.

        _shape_grid's box is x, y in [-0.5, 0.5] and z in [0, 1], and its own docstring states the
        invariant that keeps a mesh inside it: "x and y stay inside [-0.5, 0.5] for free, since
        curate defines up_axis as argmax(extents)". A table's height/longer-footprint ratio runs
        0.285 to 0.893, so height-normalising one puts most of its width outside the grid, where
        np.clip stacks it into the border columns -- the descriptor would then describe a central
        slice of the table rather than the table. Dividing by the largest extent restores the
        invariant exactly.

        What that costs and what it does NOT buy, stated because it is easy to overclaim: the
        clipped descriptor is not simply "less discriminating". Measured on a four-leg and a
        pedestal table of the same overall size, the clipped pair scores IoU 0.32 and the unclipped
        pair 0.58 -- clipping CHANGES the answer rather than compressing it, which is precisely why
        a descriptor built on it cannot be reasoned about. The claim here is the invariant, not a
        direction.

        Chairs keep their own rule and their recorded descriptors stay valid; nothing here changes
        that. Proportion is preserved either way, since this is one uniform divisor and not a
        per-axis normalisation.
        """
        return float(max(mesh.bounding_box.extents))

    def export_geometry(self, measurements, uid):
        """(exported bounding-box height, [smaller, larger] horizontal extents) in metres, known in
        closed form at curate time so _footprint can judge the shipped size without exporting.

        export scales UNIFORMLY by TABLE_SURFACE_HEIGHT_M / (raw height x work_surface_frac), and
        _stand_up rotates by a cardinal 90 degrees so axis-aligned extents merely permute. Hence

            exported[i] = normalized[i] / normalized[up] x TABLE_SURFACE_HEIGHT_M / ws_frac

        and the exported bounding-box height is TABLE_SURFACE_HEIGHT_M / ws_frac -- equal to the
        work-surface height itself on the 18 of 20 usable meshes whose top IS the work surface.
        """
        frac = measurements.get("work_surface_frac")
        if not frac:
            return None
        normalized = measurements["normalized_extents"]
        up_idx = AXES.index(measurements["up_axis"])
        height = TABLE_SURFACE_HEIGHT_M / frac
        return height, sorted(normalized[i] / normalized[up_idx] * height
                              for i in range(3) if i != up_idx)

    def export_scale(self, entry, mesh):
        """Scale so the WORK SURFACE lands at 0.74 m, not the top of the bounding box.

        For the 18 of 20 usable meshes whose work_surface_frac is 1.00 these coincide. For the two
        that differ (0.65 and 0.775) a bounding-box scale would put the work surface at 0.48 m and
        0.57 m -- a doll's table under a real chair. Reading the fraction off the entry rather than
        re-measuring the stood-up mesh keeps export in exact agreement with what curate predicted
        and _footprint judged.
        """
        frac = entry.get("work_surface_frac")
        if not frac:
            raise ValueError(
                f"{entry.get('uid')}: no work_surface_frac recorded -- curate must run first, and "
                "a table with no work surface should never have been accepted")
        return TABLE_SURFACE_HEIGHT_M, float(mesh.bounding_box.extents[2]) * float(frac)

    def export_extras(self, mesh) -> dict:
        """Where the work surface actually ended up, MEASURED on the exported geometry rather than
        restated from the constant -- so the field is a check on export's arithmetic instead of a
        promise about it. Falls back to the nominal only if the plane cannot be found in the
        exported frame, which would itself be worth noticing."""
        found = _work_surface(mesh, 2)
        if found is None:
            return {"surface_height_m": TABLE_SURFACE_HEIGHT_M}
        return {"surface_height_m": float(found["work_surface_frac"]
                                          * mesh.bounding_box.extents[2])}


TABLE = TableType()


def main(argv=None, **injected):
    """The table CLI: build_chair_manifest.main with spec=TABLE, so the two cannot drift apart."""
    return pipeline.main(argv, spec=TABLE, **injected)


if __name__ == "__main__":
    main()
