"""SimVLA: the kitchen generator's UI, served on the port the SSH tunnel already reaches.

ONE long-lived server for a whole run. `GET /` is always the CURRENT step's page, and every page
polls `GET /step` and reloads itself when the id changes -- so the run drives the browser, the URL
never changes, and one tunnel covers setup, preview, progress, prim picking and composing.

The Python side stays a linear director that blocks on a threading.Event until the browser answers.
That is not a style choice: Omniverse's USD context is not thread-safe, so the main thread must
remain the only one that touches USD. Handlers here only read and write wizard state.

Imports task_composer, emit_job and stdlib -- no isaaclab, omni or pxr -- so this module is
importable, and therefore testable, without booting Omniverse. render_edit_page adds trimesh, and
imports it INSIDE the function: the edit step is a 3D page now (one trimesh.Scene through the same
viewer template the galleries use), and paying that import only when a page actually draws keeps
the pre-boot sequence below as short as it was.

THE PRE-BOOT INVARIANT: nothing may reach scene_synthesizer before AppLauncher has run -- not at
import time, and not at CALL time either.

Why it matters. The generator constructs this server BEFORE it boots Omniverse, so the URL and the
ssh -L line are on screen while Isaac loads (~16s in practice). scene_synthesizer's usd_import and
usd_export modules soft-import pxr (`try: from pxr import ... except ImportError: warn`), and pxr
does not exist until AppLauncher has run. Reaching scene_synthesizer inside that window therefore
leaves those two modules bound to NOTHING, permanently -- sys.modules already has them, so the
generator's own post-boot import of them is a no-op -- and the run's first
`kitchen.export(file_type="usd")` dies with `NameError: name 'Usd' is not defined`. That is at the
commit: after setup, after the build, after the preview, at the first byte written.

Both halves are enforced:

  * IMPORT time -- kitchen_build is imported lazily, inside form_payload and validate_setup, and
    task_composer does the same for scene_spec.SUPPORT_SURFACES (a PEP 562 hook that derives itself
    from kitchen_build, so merely naming it in an import list pulls the whole chain in).
    test_the_generators_pre_boot_sequence_does_not_reach_scene_synthesizer walks the generator's
    real pre-boot sequence in a clean subprocess and holds this line.

  * CALL time -- laziness alone just moves the hazard, because this server is LIVE for that whole
    window and its handler threads can call those functions. /scene_defaults reaches
    SUPPORT_SURFACES; /template reaches validate_scene; neither has a step guard of its own, and a
    composer tab left open on 8777 from a previous run refetches /scene_defaults as soon as this
    server answers. Handler._sealed() therefore refuses everything but the boot page's own four
    paths with a 503 until the director publishes its first real step -- which happens strictly
    after AppLauncher. test_the_boot_step_refuses_everything_that_could_touch_scene_synthesizer
    holds this half.
"""

from __future__ import annotations

import glob
import json
import os
import random
import threading
import time
import traceback
from urllib.parse import urlparse

import emit_job
from kitchen_preview import DEFAULT_ROBOT, ROBOTS, SCALE_NODE_PREFIX, THEME_CSS, script_json
from task_composer import ComposerServer, save_composed_template


class SetupError(ValueError):
    """The setup form posted something the generator cannot build.

    The message is shown inline on the form, so it names the offending row and says why.
    """


def ensure_hf_user() -> str | None:
    """Give HF_USER a value if it has none. Returns what it defaulted to, or None if it was set.

    Nothing in the generator uploads anything, yet an unset HF_USER kills the run — at the LAST
    step, and only there. Composing a task imports isaaclab.simvla.utils, whose package __init__
    reaches isaaclab.utils.datasets, where hdf5_dataset_file_handler declares
    `repo_prefix: str = os.environ['HF_USER']` as a DEFAULT ARGUMENT. Default arguments are
    evaluated at import, so the missing variable is a KeyError during that import rather than at
    any upload — and by then the kitchen is committed, its twelve rotations are exported and their
    envs are registered. The whole authoring session is lost to a variable that could have been
    checked before Isaac Sim was even booted.

    Defaulting rather than refusing, and to $USER, because emit_job already writes exactly that
    fallback into the sbatch script the composer's *Generate goals* button submits
    (`os.environ.get("HF_USER") or os.environ.get("USER")`). Inventing a second rule here would let
    a run and the job it emits disagree about who they are. The caller says out loud what it picked,
    so a real HuggingFace username can be exported before anything is pushed.
    """
    if os.environ.get("HF_USER"):
        return None
    os.environ["HF_USER"] = os.environ.get("USER") or "simvla"
    return os.environ["HF_USER"]


def existing_kitchen_numbers(kitchen_dir: str) -> list[int]:
    """Kitchen numbers whose base USD is already on disk — Accept would overwrite them.

    Only `kitchen_<N>.usd` counts; `kitchen_<N>_<i>.usd` is one of that kitchen's twelve rotations.
    """
    numbers = set()
    for path in glob.glob(os.path.join(kitchen_dir, "kitchen_*.usd")):
        stem = os.path.basename(path)[len("kitchen_"):-len(".usd")]
        if stem.isdigit():
            numbers.add(int(stem))
    return sorted(numbers)


#: The object menu's two headings. Which one a type sits under is the only thing on the form that
#: says whether it will grasp with real BODex data.
#:
#: There is no bbox-center fallback to fall back to: kitchen_scene_generator.py stamps BODex_path on
#: EVERY placed object (rotate_and_register_envs), so plan_arm_grasp's CASE 1 (real graspdata) is
#: what always fires; its CASE 2 bbox-center branch is dead code, reachable only if BODex_path were
#: ever absent. Placing a type with no grasp data instead runs load_grasp_file -> None ->
#: `raise FileNotFoundError(f"No grasp base path found for {bodex_attr.Get()}")` deep inside data
#: generation, after the kitchen is committed and the goals are generated.
BODEX_GROUP = "Grasps with BODex"
NO_GRASP_GROUP = "No grasp data — grasping will fail"


def object_menu(manifest=None) -> list[dict]:
    """[{type, label, group}] for the setup page's object menu, in registry order.

    The form is the last place a wrong pick is cheap. A type with no grasp data looks exactly like
    any other row there, and the difference only surfaces much later — as a FileNotFoundError deep
    inside data generation, when plan_arm_grasp's real-graspdata path (which every placed object
    takes; there is no bbox-center fallback to catch it) finds no grasp file for the mesh — by which
    point the kitchen is committed and the goals are generated. So what the manifest measured is put
    on the options themselves.

    `type` is what the row carries and what validate_setup checks; `label` is display only, so
    annotating an option cannot change what it means.
    """
    from grasp_manifest import strategy_for_type              # stdlib-only; see module doc
    from scene_spec import CLUTTER, GRASPABLE_TYPES, NO_GRASPDATA

    menu = []
    for obj_type in list(GRASPABLE_TYPES) + list(CLUTTER):
        measured = strategy_for_type(manifest, obj_type)
        if measured == "unknown":
            # No manifest (or a type absent from it): fall back to the declared lists.
            measured = "none" if obj_type in NO_GRASPDATA else "single"
        label = obj_type
        if measured == "partial":
            label = f"{obj_type} — some meshes lack grasps"
        elif measured == "none":
            label = f"{obj_type} — no grasp data"
        menu.append({
            "type": obj_type,
            "label": label,
            "group": NO_GRASP_GROUP if measured == "none" else BODEX_GROUP,
        })
    return menu


#: The most chairs one kitchen may carry.
#:
#: The cap lives HERE and deliberately not in kitchen_build.build_kitchen: the builder places what
#: it is given, and a scripted call that wants eight chairs is not a geometry error. Six is a
#: judgement about what is reasonable to author, so it belongs on the authoring side.
#:
#: It is also the number kitchen_build lays its DEFAULT seating out for (DEFAULT_CHAIR_ROW /
#: _seats_around_table), so a kitchen at the cap is exactly the arrangement that was measured
#: against the placement gate. Raising it here without raising that would silently start handing
#: the seventh chair a seat that was never checked.
#:
#: IT IS 8, AND DEFAULT_CHAIR_ROW IS 8 TOO. Both were required; the measurement that settled it
#: (2026-08-13, the fifty-chair registry, island / square_small, eight chairs at their own
#: defaults, the wizard's own build-then-auto-face sequence):
#:
#:   MAX_CHAIRS = 8, DEFAULT_CHAIR_ROW = 6   16 of 50 chairs clean -- 34 REJECTED
#:   MAX_CHAIRS = 8, DEFAULT_CHAIR_ROW = 8   50 of 50 clean, worst depth 0.0 mm
#:
#: add_chair draws chair i's seat from a max(DEFAULT_CHAIR_ROW, i + 1)-seat ring, so at eight
#: chairs 0-5, 6 and 7 come off three rings of three different radii: measured on square_small
#: they land 0.58 m and 0.63 m apart against the 1.118 m their own slot rule reserves, and the
#: gate reads "chair 5 overlaps chair 6". Raising this alone rejects two thirds of the library at
#: eight, which is worse than rejecting all of it -- it looks like it works. So if this is ever
#: raised again, DEFAULT_CHAIR_ROW in kitchen_build.py moves with it, and the sweep is re-run
#: across EVERY chair -- not the widest, which is the one with enough slack to hide the fault.
#:
#: AND THE RING GETS BIG. Off square_small's 0.70 m edge, seat centres stand 0.61-1.47 m out at
#: six (narrowest to widest offered chair) and 1.18-2.80 m at eight. Eight chairs at a small table
#: is geometrically valid and reads as a ring of chairs in a room, not as seating at a table; the
#: earlier sweep called 1.35 m "no longer seating at a table under any interpretation". That is a
#: look cost, not a gate failure -- every one of those 50 passes.
MAX_CHAIRS = 8


def _chair_size_line(variant) -> str:
    """A chair's DATA LINE: the short uid, then what was measured off it. "" if there is neither.

    Shown under the chair's name in its row, and the name is now only a number ("Chair 7"), so the
    uid leading this line is not decoration. kitchen_build.FURNITURE_NUMBERING has the full story:
    the number is a position in a registry that is rebuilt from a manifest, so an author who wrote
    down "Chair 7" and came back after a rebuild would have a different chair; the uid is the
    identity, and it is what the wizard's own state and the exported scene carry.

    Otherwise presentation only, and deliberately only what the registry ACTUALLY holds: a chair's
    height and the plan-view footprint the seat ring reserves for it (chair_footprint_m's diagonal,
    which is why it is not written as w x d -- the registry does not keep the two extents
    separately once the diagonal is taken). A manifest written before build_chair_manifest recorded
    export_footprint_m has neither, and that row shows the uid alone rather than a number nobody
    measured.
    """
    if variant is None:
        return ""
    parts = []
    if getattr(variant, "detail", None):
        parts.append(str(variant.detail))
    if getattr(variant, "height_m", None):
        parts.append(f"h {float(variant.height_m):.2f} m")
    if getattr(variant, "footprint_m", None):
        parts.append(f"footprint {float(variant.footprint_m):.2f} m")
    return " · ".join(parts)


def _chair_edit_line(scale: float, material) -> str:
    """A chair row's EDIT line: what the author changed, and nothing when they changed nothing.

    Blank for an unedited chair on purpose -- "1.00× · floor" on every row of a kitchen nobody has
    edited is noise that makes the one row that IS edited harder to find, which is the opposite of
    why the line exists.
    """
    parts = []
    if float(scale) != 1.0:
        parts.append(f"{float(scale):.2f}×")
    if material:
        parts.append(str(material))
    return " · ".join(parts)


def _chair_rows(chairs, labels, sizes=None) -> list[dict]:
    """One row per DISTINCT chair, with how many of it are placed. Presentation only.

    The state stays a FLAT list of chairs with repeats -- that is what build_kitchen consumes, and
    what numbers them chair_0, chair_1, ... Grouping it into [(spec, count)] in the state itself
    would change every consumer (validate_setup, run_wizard, both builds) to spare the form a
    loop, so the grouping lives here, on the way to the page, and nowhere else.

    "DISTINCT" IS THE WHOLE SPEC NOW, not the uid. It was the uid while every chair of a design was
    interchangeable; per-piece scale and material end that, and a row keyed by uid alone would show
    one count and one stepper for two chairs that are no longer the same object -- and its Edit
    button would have no way to say which of them it meant. Keying on (uid, scale, material) makes
    a row exactly "the chairs that are identical", so + duplicates one of them, - takes one away,
    and Edit re-specs all of them, all of which are true statements about a row's contents. It
    degenerates to the old grouping exactly when nothing has been edited, which is every kitchen
    built before this existed.

    Each row carries `indices`, the positions in the flat list its chairs occupy. That is what
    makes Edit able to change some chairs of a uid and not others: the director rewrites those
    positions and leaves the rest, so an edited chair simply falls out of this row into its own.

    First-appearance order, which a plain dict gives for free (insertion-ordered since 3.7). It is
    the author's own order and it is the only one that holds still: sorting by count moves a row
    the moment its + is pressed, which is to say the row moves out from under the cursor that just
    pressed it, and sorting by label (or by registry order) throws away the order they picked in.

    `can_add` is the same for every row because the cap counts CHAIRS, not designs -- six of one
    chair is at the cap exactly as six different ones are. It is sent per row anyway so the page
    can disable the button that is actually refused, rather than reason about the cap in its own JS
    (the same reason max_chairs is sent at all).

    A uid the registry no longer offers falls back to showing itself, mirroring what the page did
    with its own label lookup: a checkout that loses /lustre mid-run should still show what it is
    holding rather than a row of blanks. `sizes` is optional and defaults to blank for the same
    reason -- the row is still a row without its measurement.
    """
    sizes = sizes or {}
    grouped: dict[tuple, list[int]] = {}
    for index, chair in enumerate(chairs or ()):
        key = (chair["uid"], float(chair.get("scale", 1.0)), chair.get("material"))
        grouped.setdefault(key, []).append(index)
    can_add = sum(len(idx) for idx in grouped.values()) < MAX_CHAIRS
    return [
        {"uid": uid, "label": labels.get(uid, uid), "size": sizes.get(uid, ""),
         "scale": scale, "material": material, "edit": _chair_edit_line(scale, material),
         "indices": indices, "count": len(indices), "can_add": can_add}
        for (uid, scale, material), indices in grouped.items()
    ]


def chair_row_indices(chairs, row: int) -> list[int]:
    """Which chairs, by position in the flat list, the setup form's row `row` is showing.

    The one thing the director needs out of the grouping, exported so the grouping rule itself
    stays in this module: +, − and Edit… all name a row, and every one of them then has to act on
    the instances behind it. [] for a row that does not exist, which is the same "nothing to do"
    answer _remove_chair already gives for a chair it cannot find.
    """
    rows = _chair_rows(chairs, {})
    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < len(rows):
        return []
    return list(rows[row]["indices"])


def form_payload(kitchen_dir: str, state=None) -> dict:
    """Everything the setup page needs to render itself.

    Locations are per kitchen type — most of the ~21 placements are type-specific, which is why the
    Tk menu rebuilt itself on every kitchen change — so the page gets a map and switches locally.

    has_table is read off `state`, not off the kitchen type: a table is chosen and placed by the
    user on any layout (see kitchen_build.available_locations), so the SAME boolean applies to
    every entry in `locations` below. Without a table in state, "On the table" is left out of
    every type's list — there is nothing there to place an object on yet.
    """
    from kitchen_build import (                                          # see module doc
        ALL_TABLE_VARIANTS, CHAIR_VARIANTS, FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES,
        KITCHEN_BUILDERS, available_locations,
    )
    from grasp_manifest import load as load_grasp_manifest
    from grasp_manifest import stale_warning

    manifest = load_grasp_manifest()
    warning = stale_warning(manifest)
    if warning:
        print(warning)

    has_table = bool(state and state.get("table"))
    chair_labels = {v.uid: v.label for v in CHAIR_VARIANTS}
    chair_sizes = {v.uid: _chair_size_line(v) for v in CHAIR_VARIANTS}
    return {
        "kitchen_types": sorted(KITCHEN_BUILDERS),
        "object_types": object_menu(manifest),
        "locations": {
            name: [
                {"loc": p.loc, "label": p.label, "group": p.group}
                for p in available_locations(name, has_table=has_table)
            ]
            for name in sorted(KITCHEN_BUILDERS)
        },
        # {variant key -> display label}, so the form can show what is currently chosen without
        # importing the registry into the page's own JS.
        #
        # ALL_TABLE_VARIANTS, so a mesh table the author picked in the gallery has a label here
        # too. Keyed by the same key add_table resolves, procedural and Objaverse alike; with
        # only the twelve procedural keys, showTable() looked the key up, found nothing, and the
        # form said "no table" while the state carried one.
        "table_variants": {v.key: v.label for v in ALL_TABLE_VARIANTS},
        # The table's DATA LINE, keyed the same way. "Table 7" says which position in the registry
        # and nothing about the table, so this is what a procedural variant's old display label
        # became ("Long dining") and what identifies an Objaverse one ("uid e7cc55"). Sent
        # separately rather than folded into the label above so that the label stays the label --
        # it is what the run summary ellipsises and what the gallery answers with.
        "table_details": {v.key: (v.detail or "") for v in ALL_TABLE_VARIANTS},
        # The same idea for chairs, keyed by uid because that is what the state carries. EMPTY on a
        # checkout with no chair library: CHAIR_VARIANTS is derived from the manifest on /lustre and
        # is () when it is absent, which the page reads as "there are no chairs to offer" rather
        # than as an error.
        "chair_variants": chair_labels,
        # The chairs already placed, grouped one row per distinct chair -- what the form draws its
        # +/- rows from. Grouped HERE rather than in the page's JS so that the grouping is real
        # behaviour with a real test, and so the rows the author sees are the server's reading of
        # the very list the form posts back.
        "chair_rows": _chair_rows(
            state.get("chairs") if state else None, chair_labels, chair_sizes
        ),
        # The cap, sent to the page rather than written into its JS, so the button that stops
        # offering the picker and the check that refuses a seventh chair cannot drift apart.
        "max_chairs": MAX_CHAIRS,
        # The two edit menus, sent for the same reason the cap is: the edit STEP offers exactly
        # these and validate_setup refuses anything else, so the form must not carry a third copy
        # that can drift from either. The form does not edit anything itself -- Edit… is a submit
        # that opens the edit step -- but it does have to SHOW what is currently chosen.
        "scales": list(FURNITURE_SCALES),
        "material_groups": list(FURNITURE_MATERIAL_GROUPS),
        # The table's own edit, for its line under the Table heading. Read off `state` like
        # everything else there; absent means an unedited table (or none at all).
        "table_edit": _chair_edit_line(
            (state or {}).get("table_scale") or 1.0, (state or {}).get("table_material")
        ),
        "existing": existing_kitchen_numbers(kitchen_dir),
        # What the form was showing when it left for a gallery, so it can come back the same. Empty
        # on a fresh form.
        "state": dict(state) if state else {},
    }


# A kitchen number is a small index -- the largest committed one today is three digits. CPython
# itself refuses to convert a numeral past ~4300 digits into an int at all (a denial-of-service
# guard baked into the interpreter); this cap is far below that, so int() below never gets the
# chance to raise, and the browser gets a reason instead of a dropped connection.
_MAX_KITCHEN_NUM_DIGITS = 9


def _kitchen_number(raw: str) -> int:
    """The kitchen number, checked. Only the actions that write need one."""
    if not raw.isdigit():
        raise SetupError("Kitchen number must be a non-negative integer.")
    if len(raw) > _MAX_KITCHEN_NUM_DIGITS:
        raise SetupError(f"Kitchen number has too many digits ({len(raw)}); that cannot be right.")
    return int(raw)


def _validate_rows(raw_objects, name: str, mesh_files, *, has_table: bool = False) -> list[dict]:
    """The object rows, checked against this kitchen type, in build_kitchen's own spec shape.

    Shared by every action that carries rows, so browsing a gallery refuses a nonsense row up
    front rather than after the ~27 seconds it takes to build the kitchen tiles.

    has_table gates "On the table" (loc 42) exactly as it gates the form's own location menu —
    see available_locations. Without it, a stale row aimed at a table the user has since given
    up would be silently legal here while invisible on the page that produced it.
    """
    from kitchen_build import OBJECT_TYPES, available_locations, resolve_mesh

    if raw_objects is None:
        raw_objects = []
    elif not isinstance(raw_objects, list):
        raise SetupError(f"'objects' must be a list of rows, not {type(raw_objects).__name__}.")

    legal = {p.loc for p in available_locations(name, has_table=has_table)}
    counts: dict[str, int] = {}
    objects = []
    for index, row in enumerate(raw_objects, start=1):
        if not isinstance(row, dict):
            raise SetupError(f"Row {index}: expected an object, got {type(row).__name__}.")
        obj_type = row.get("type")
        if obj_type not in OBJECT_TYPES:
            raise SetupError(f"Row {index}: unknown object type {obj_type!r}.")

        try:
            loc = int(row.get("loc"))
        except (TypeError, ValueError):
            raise SetupError(f"Row {index}: no placement location chosen.") from None
        if loc not in legal:
            pretty = name.replace("_", " ")
            raise SetupError(f"Row {index}: that location is not available on a {pretty} kitchen.")

        mesh = row.get("mesh") or None
        if mesh is not None and mesh not in mesh_files:
            raise SetupError(f"Row {index}: {mesh} is not in the BODex dataset.")
        if mesh is None:
            try:
                resolve_mesh(obj_type, mesh_files)   # fail now, not twenty minutes into a build
            except ValueError as exc:
                raise SetupError(f"Row {index}: {exc}") from exc

        counts[obj_type] = counts.get(obj_type, 0) + 1
        objects.append({
            "obj_n": f"{obj_type}{counts[obj_type] - 1}",
            "type": obj_type,
            "loc": loc,
            "mesh": mesh,
        })
    return objects


def _validate_scale(raw, what: str) -> float:
    """A posted size scale, checked against the exact steps the wizard offers.

    Not merely "inside the range": the edit page's slider moves in exactly these steps and shows
    the current value as one of them, so a scale of 1.07 would be a state the author could see the
    effect of and never drag back to. kitchen_build.check_furniture_scale / check_chair_scale
    enforce the bound where the geometry is built; this enforces the MENU, which is the authoring
    side's business.

    The menu is FURNITURE_SCALES for every piece. It used to take a per-piece list, because a few
    narrow chairs were refused the bottom of the range; that width floor is advisory now -- it is
    marked on the edit page's slider and never withheld, see kitchen_build.chair_width_floor_scale
    -- so there is one menu again and this takes no list.

    Absent is 1.0 -- an unedited piece -- so a form (or a script) that says nothing about scale
    still builds exactly what it built before this existed.
    """
    from kitchen_build import FURNITURE_SCALES                # ^ lazy; see module doc.

    offered = FURNITURE_SCALES
    if raw is None:
        return 1.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SetupError(f"{what}: scale must be a number, not {type(raw).__name__}.")
    for scale in offered:
        # Compared with a tolerance, not ==: these arrive as JSON floats and 0.85 does not
        # survive every JSON round trip bit-identically. Half a step is the widest tolerance
        # that still cannot confuse two neighbouring offers.
        if abs(float(raw) - scale) < 1e-6:
            return scale
    raise SetupError(
        f"{what}: {float(raw):g} is not one of the offered scales "
        f"({', '.join(f'{s:g}' for s in offered)})."
    )


def _validate_material(raw, what: str):
    """A posted material group, checked against the offered groups. None means "no choice".

    None rather than the default group's name, deliberately: "the author chose nothing" and "the
    author chose floor" produce the same picture but not the same data, and only the first should
    make furniture_material_overrides emit an override binding at all.
    """
    from kitchen_build import FURNITURE_MATERIAL_GROUPS       # ^ lazy; see module doc.

    if raw is None:
        return None
    if raw not in FURNITURE_MATERIAL_GROUPS:
        raise SetupError(f"{what}: unknown material {raw!r}.")
    return raw


def _validate_robot(raw) -> str:
    """The session's robot, as one of kitchen_preview.ROBOTS or "" for none.

    Chosen once, on the first step of the run, and carried on the setup state from there -- so it
    survives every trip out to a gallery and back, and every later comparison is drawn against the
    same robot. Checked here for the same reason the table variant is: the browser can post
    anything, and a name no robot answers to would raise inside the edit page's own render
    (render_edit_page refuses it) rather than being refused on the way in.

    MISSING IS DEFAULT_ROBOT, not "": a form posted before this field existed -- or by anything
    that does not know about it -- has to keep drawing the comparison it was drawing, and that is
    the robot the edit step opened on before the choice existed. An explicit "" is No robot, which
    is a state the preview already has and a choice this field must be able to carry.
    """
    if raw is None:
        return DEFAULT_ROBOT
    if not isinstance(raw, str) or (raw and raw not in ROBOTS):
        raise SetupError(f"Unknown robot {raw!r}.")
    return raw


def _validate_chairs(raw_chairs) -> list[dict]:
    """The chairs the form posted -- ONE DICT PER INSTANCE -- checked against the offered registry,
    the offered scales and the cap.

    The shape is kitchen_build.chair_spec's, minus `transform`, which the form has no way to set
    (chairs are dragged in the preview, not typed into the setup page) and which therefore must
    not be accepted from a browser: a posted 4x4 would be a world pose nothing had validated.

    A bare uid string is refused by name. That is what a call site missed in the migration off the
    flat uid list looks like, and refusing it here is the difference between an error and a chair
    that silently ignores the author's edit.

    Shared by every action that carries chairs, for the same reason _validate_rows is: leaving for
    a gallery must refuse a nonsense list up front rather than after a build.

    CHAIR_VARIANT_BY_UID is imported lazily, like every other kitchen_build name in this module --
    see the module docstring's pre-boot invariant. It is also why the registry is read at CALL time
    rather than captured at import: it is derived from the manifest on /lustre, and a checkout
    without one has an EMPTY registry. That state refuses every uid, which is correct (there is no
    such chair to place), while still accepting an empty list -- so a kitchen with no chairs builds
    exactly as it always did.

    The cap is checked before the uids so that seven real chairs are refused for being seven,
    which is the reason the author needs to see.
    """
    from kitchen_build import CHAIR_VARIANT_BY_UID                     # ^ lazy; see module doc.

    if raw_chairs is None:
        return []
    if not isinstance(raw_chairs, list):
        raise SetupError(
            f"'chairs' must be a list of chairs, not {type(raw_chairs).__name__}."
        )
    if len(raw_chairs) > MAX_CHAIRS:
        raise SetupError(
            f"At most {MAX_CHAIRS} chairs per kitchen; that is {len(raw_chairs)}."
        )
    chairs = []
    for index, raw in enumerate(raw_chairs, start=1):
        what = f"Chair {index}"
        if isinstance(raw, str):
            raise SetupError(
                f"{what}: a chair is now {{'uid': ..., 'scale': ..., 'material': ...}}, not a "
                f"bare uid -- two chairs of the same design can differ."
            )
        if not isinstance(raw, dict):
            raise SetupError(f"{what}: expected a chair, got {type(raw).__name__}.")
        unknown = set(raw) - {"uid", "scale", "material"}
        if unknown:
            raise SetupError(f"{what}: unknown field(s) {', '.join(sorted(unknown))}.")
        uid = raw.get("uid")
        if not isinstance(uid, str) or uid not in CHAIR_VARIANT_BY_UID:
            raise SetupError(f"{what}: unknown chair {uid!r}.")
        chairs.append({
            "uid": uid,
            # The whole menu, for every chair. The width floor a narrow chair crosses on the way
            # down is marked on the edit page's slider and never refused -- see
            # kitchen_build.chair_width_floor_scale.
            "scale": _validate_scale(raw.get("scale"), what),
            "material": _validate_material(raw.get("material"), what),
        })
    return chairs


def validate_setup(payload: dict, mesh_files) -> dict:
    """Parse and check a posted setup form; raise SetupError with a displayable reason.

    Server-side because the browser can post anything. Returns objects in exactly the shape
    build_kitchen() consumes, including the "mesh" field (None = pick a random matching mesh at
    build time, today's behaviour).

    Every failure here must come back as SetupError: _handle_setup only knows how to turn that one
    exception type into an HTTP response, and a POST that gets no response at all shows the user
    nothing -- not even a rejection, just a dead-looking form.
    """
    from kitchen_build import KITCHEN_BUILDERS, TABLE_VARIANT_BY_KEY   # ^ lazy; see module doc.

    action = payload.get("action")
    if action not in ("generate", "compose_existing", "pick_kitchen", "pick_mesh", "pick_table",
                      "pick_chair", "add_chair", "remove_chair", "edit_table", "edit_chair"):
        raise SetupError(f"unknown action {action!r}")

    raw_number = str(payload.get("kitchen_num", "")).strip()

    if action == "compose_existing":
        return {
            "action": action,
            "kitchen_num": _kitchen_number(raw_number),
            "kitchen_name": None,
            "seed": None,
            "objects": [],
        }

    name = payload.get("kitchen_name") or "random"
    if name == "random":
        name = random.choice(sorted(KITCHEN_BUILDERS))
    if not isinstance(name, str) or name not in KITCHEN_BUILDERS:
        raise SetupError(f"Unknown kitchen type {name!r}.")

    seed = payload.get("seed")
    # bool is an int subclass in Python -- without this, {"seed": true} would silently pass as
    # isinstance(seed, int) and build with seed 1. Same guard, same voice, as the row check below.
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise SetupError(f"Seed must be a whole number or absent, not {type(seed).__name__}.")

    # None means no table -- there is no separate checkbox; you decline one by cancelling the
    # picker. A non-None value has to be a real variant key, or the wizard would carry a choice
    # build_kitchen.add_table cannot honour.
    table = payload.get("table")
    if table is not None and table not in TABLE_VARIANT_BY_KEY:
        raise SetupError(f"Unknown table variant {table!r}.")

    # The table's own edit, as two sibling fields rather than a dict in `table`. A kitchen has ONE
    # table, so unlike the chairs there is no second instance for these to have to be attached to,
    # and `table` stays the nullable identity every consumer already reads (`bool(state["table"])`
    # is "is there a table", the gallery answers with a key, add_table resolves one). Both are
    # meaningless without a table and are checked anyway: a stale scale left behind by a Cancel is
    # a value the next build must not silently pick up.
    table_scale = _validate_scale(payload.get("table_scale"), "Table")
    table_material = _validate_material(payload.get("table_material"), "Table")

    # One dict per placed chair, in the order they were picked -- the same order build_kitchen
    # numbers them chair_0, chair_1, ... Absent or empty means a kitchen with no chairs.
    chairs = _validate_chairs(payload.get("chairs"))

    # The session's robot, picked on the first step of the run and echoed by the form on every
    # submit. It is not part of the kitchen -- nothing is built or committed from it -- but it
    # rides on the same state for the same reason the table's scale does: every step that draws a
    # size comparison is reached THROUGH that state, and a value the director carried separately
    # would have to be threaded into each of them by hand.
    robot = _validate_robot(payload.get("robot"))

    objects = _validate_rows(payload.get("objects"), name, mesh_files, has_table=table is not None)

    if action in ("pick_kitchen", "pick_mesh", "pick_table", "pick_chair",
                  "add_chair", "remove_chair", "edit_table", "edit_chair"):
        # Deliberately NOT checking the kitchen number: you should be able to look at kitchens
        # before deciding what to number the result. generate still checks it.
        state = {
            "kitchen_num": raw_number,
            "kitchen_name": name,
            "seed": seed,
            "objects": [{"type": o["type"], "loc": o["loc"], "mesh": o["mesh"]} for o in objects],
            "table": table,
            "table_scale": table_scale,
            "table_material": table_material,
            "chairs": chairs,
            "robot": robot,
        }
        row = None
        if action == "pick_mesh":
            row = payload.get("row")
            if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < len(objects):
                raise SetupError(f"Cannot choose a mesh for row {row!r}: no such row.")
        # +, - and Edit name the chair ROW they were pressed on, BY INDEX -- the exact counterpart
        # of pick_mesh's row check, refused with the same voice.
        #
        # An index into the GROUPED rows, not a uid, and that is the shape change per-piece edits
        # force. A row used to be a uid because every chair of a uid was the same chair; now a row
        # is one distinct (uid, scale, material) and there can be two rows of the same uid, so a
        # uid no longer names one of them. The index is well defined on both sides because
        # _chair_rows is a pure function of `chairs` and `chairs` is exactly what the form posts
        # back -- the server regroups the same list it grouped when it drew the page. An empty
        # list refuses all three actions, which is right: there is no row to press.
        chair_row = None
        if action in ("add_chair", "remove_chair", "edit_chair"):
            chair_row = payload.get("chair_row")
            rows = _chair_rows(chairs, {})
            if (not isinstance(chair_row, int) or isinstance(chair_row, bool)
                    or not 0 <= chair_row < len(rows)):
                raise SetupError(f"Cannot change chair row {chair_row!r}: no such row.")
        if action == "edit_table" and table is None:
            raise SetupError("There is no table to edit.")
        return {"action": action, "state": state, "row": row, "chair_row": chair_row}

    return {
        "action": action,
        "kitchen_num": _kitchen_number(raw_number),
        "kitchen_name": name,
        "seed": seed,
        "objects": objects,
        "table": table,
        "table_scale": table_scale,
        "table_material": table_material,
        "chairs": chairs,
        "robot": robot,
    }


# How often every page polls /step and reloads when it changes (see _STEP_SCRIPT below). finish()
# has to sleep for longer than this -- with margin, since a poll can be mid-wait right when "done"
# publishes -- or the run (often the whole process) can end before the browser ever notices the new
# step and lands on the closing page. Both numbers come from this one constant so that changing the
# poll interval cannot silently erode finish()'s margin without anyone changing it on purpose.
_POLL_INTERVAL_MS = 1000
_FINISH_MARGIN_S = 1.2 * _POLL_INTERVAL_MS / 1000

# Injected into EVERY page the wizard publishes, including the trimesh 3D pages. It gives the page
# its step id (what /answer is checked against), a POST helper that stamps it, and the poller that
# reloads when the run moves on.
_STEP_SCRIPT = """
<script>
window.SIMVLA_STEP = __STEP__;
window.simvlaPost = function (path, body) {
  body = body || {};
  body.step = window.SIMVLA_STEP;
  return fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  }).then(function (r) {
    return r.json().catch(function () { return {}; }).then(function (j) {
      return {status: r.status, body: j};
    });
  });
};
(function () {
  setInterval(function () {
    fetch('/step', {cache: 'no-store'})
      .then(function (r) { return r.json(); })
      .then(function (s) { if (s.id !== window.SIMVLA_STEP) { location.reload(); } })
      .catch(function () {});   // the run has ended and the server is gone; nothing to do
  }, __POLL_MS__);
})();
</script>
"""


#: The run, in the order it happens. Every published page shows this strip with its own phase lit,
#: because until now no page said what came after it -- you could not tell from the preview whether
#: accepting it was the last thing or the middle of five.
PHASES = (
    ("setup", "set up"),
    ("pick", "pick"),
    ("preview", "preview"),
    ("build", "build"),
    ("compose", "compose"),
)

#: Pinned top-LEFT of the viewport rather than folded into the wizard's own header bar, which is
#: where the approved mockup drew it. It has to be: this strip is injected by inject_step_script
#: into EVERY page the run publishes -- including the trimesh 3D pages, which have no header bar of
#: this module's to put it in. One position that works on all five surfaces beats a bar that exists
#: on two of them. Left, not right, because #simvla-panel and #gal are both fixed top-RIGHT.
_STRIP_CSS = """
<style>
#simvla-phases{position:fixed;top:0;left:0;z-index:10000;display:flex;gap:0;
  font:10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;
  background:var(--scrim);border-right:1px solid var(--edge);
  border-bottom:1px solid var(--edge)}
#simvla-phases span{padding:7px 11px;color:var(--muted)}
#simvla-phases span+span{border-left:1px solid var(--edge)}
/* Brass on ink for the phase the run is IN -- the same one accent the current selection and the
   primary action get, and nothing else on the strip takes it. The phases already behind you read
   in plain text; the ones ahead stay muted. */
#simvla-phases span.now{color:var(--accent);background:var(--bg);font-weight:600;
  box-shadow:inset 0 -2px 0 var(--accent)}
#simvla-phases span.done{color:var(--fg-2)}
</style>
"""


def _phase_strip(phase) -> str:
    """The run's phases as a strip, with `phase` lit and everything before it marked done."""
    if phase is None:
        return ""
    names = [key for key, _ in PHASES]
    at = names.index(phase) if phase in names else -1
    cells = []
    for index, (key, label) in enumerate(PHASES):
        state = "now" if index == at else ("done" if at > index else "")
        cells.append(f'<span class="{state}">{label}</span>')
    return _STRIP_CSS + '<div id="simvla-phases">' + "".join(cells) + "</div>"


def inject_step_script(html: str, step_id: int, phase=None) -> str:
    """Give a page its step id, the POST helper, the reload poller and the phase strip.

    Every page the wizard serves goes through here, including the 3D pages built by other
    modules — which is exactly why the strip is injected here rather than in each renderer.
    """
    script = (
        _STEP_SCRIPT.replace("__STEP__", str(int(step_id)))
        .replace("__POLL_MS__", str(_POLL_INTERVAL_MS))
    )
    addition = _phase_strip(phase) + script
    if "</body>" in html:
        return html.replace("</body>", addition + "\n</body>", 1)
    return html + addition


#: The wizard's own pages, as one card on the ink ground: a header bar, then whatever the step put
#: in __BODY__.
#:
#: Two faces, split by job (see THEME_CSS): --ui carries every label, heading and sentence; --mono is
#: kept for DATA -- kitchen numbers, seeds, uids, mesh paths, measured sizes, counts. The old shell
#: set `font:13px/1.55 var(--mono)` on <body> and everything inherited it, so a paragraph of prose
#: and a mesh path were typeset identically and neither read as what it was.
#:
#: Top padding, not margin, and 44px of it: the phase strip is position:fixed at the top-left of the
#: VIEWPORT (see _STRIP_CSS), injected after this page is rendered, so the card has to leave room
#: for something it cannot see.
_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>SimVLA kitchen generator</title>
<style>
__THEME__
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:44px 22px 80px;
  font:13.5px/1.55 var(--ui);-webkit-font-smoothing:antialiased}
.page{max-width:1080px;margin:0 auto}
.app{border:1px solid var(--edge);background:var(--panel)}
.bar{display:flex;align-items:center;gap:16px;padding:13px 18px;border-bottom:1px solid var(--edge);
  background:var(--panel-2)}
.mark{display:flex;align-items:baseline;gap:9px}
.mark b{font-size:15px;font-weight:640;letter-spacing:-.01em}
.mark span{font-family:var(--mono);font-size:10.5px;color:var(--muted);letter-spacing:.08em}

/* The form's two columns: the work on the left, the run summary and its primary action on the
   right. Every other page publishes a single `.main.solo`. */
.body{display:grid;grid-template-columns:1fr 300px}
.main{padding:22px 20px;min-width:0}
.main.solo{padding:24px 22px}
.side{border-left:1px solid var(--edge);padding:22px 18px;background:var(--panel-2);min-width:0}
@media(max-width:860px){
  .body{grid-template-columns:1fr}
  .side{border-left:0;border-top:1px solid var(--edge)}
}

h1{font-size:19px;margin:0 0 5px;font-weight:620;letter-spacing:-.015em}
/* Section names, not numbers -- and set as chrome (mono, micro, tracked) so they name the work
   without competing with it. */
h2{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  margin:0 0 3px;font-family:var(--mono)}
h3{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  margin:0 0 9px;font-family:var(--mono)}
.sub{color:var(--fg-2);font-size:13.5px;max-width:70ch}
.hint{font-size:12.5px;color:var(--muted);margin:0 0 13px;max-width:70ch}
.note{color:var(--muted);font-size:12px;margin-top:10px;line-height:1.5;max-width:70ch}
.err{color:var(--err);font-size:12.5px;margin:10px 0;min-height:18px}
.warn{color:var(--warn);font-size:12px;min-height:16px}
.data{font-family:var(--mono);font-size:11.5px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.sect+.sect{margin-top:30px;padding-top:26px;border-top:1px solid var(--edge)}

/* ---- controls ---- */
label{display:block;margin:6px 0}
.fld{display:flex;flex-direction:column;gap:5px}
.fld>span{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
  color:var(--muted);font-weight:600}
input,select{font:inherit;font-size:13px;background:var(--bg);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:6px 9px;max-width:100%}
input{font-family:var(--mono);font-variant-numeric:tabular-nums}
button{font:inherit;font-size:12.5px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:6px 13px;cursor:pointer}
button:hover:enabled{background:var(--edge);color:var(--accent)}
button:disabled{opacity:.35;cursor:default}
input:focus,select:focus,button:focus-visible{outline:1px solid var(--accent);outline-offset:1px}
/* Brass, spent once: the primary action and the current selection, and nothing else on the page. */
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);
  font-weight:640;padding:9px 18px;font-size:13px}
button.primary:hover:enabled{background:var(--accent-lift);color:var(--accent-fg)}
button.ghost{background:transparent;color:var(--muted);border-color:var(--edge)}
button.ghost:hover:enabled{background:transparent;color:var(--accent);border-color:var(--edge-bright)}
.side button.primary,.side button.ghost{width:100%}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}

/* ---- choice cards (the room picker) ---- */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:9px}
.card{border:1px solid var(--edge);background:var(--bg);padding:0;cursor:pointer;text-align:left;
  font:inherit;color:inherit;position:relative;transition:border-color .12s}
.card:hover:enabled{border-color:var(--edge-bright);background:var(--bg);color:inherit}
.card.on{border-color:var(--accent)}
.card.on::after{content:"";position:absolute;inset:0;background:var(--accent-wash);
  pointer-events:none}
.card .thumb{aspect-ratio:4/3;display:block;width:100%;background:var(--panel-2)}
.card .cap{padding:7px 9px 8px;border-top:1px solid var(--edge)}
.card .cap b{display:block;font-size:12.5px;font-weight:560;letter-spacing:-.005em}
.card.on .cap b{color:var(--accent)}
/* Plan-view glyphs: schematic, drawn from tokens rather than literal hexes so a palette change
   carries them. `.hi` is the layout's defining feature (the island, the peninsula), and it is the
   one part that takes brass -- only on the card that is currently chosen. */
.thumb .g{fill:var(--panel-2)}
.thumb .w{fill:none;stroke:var(--muted);stroke-width:1.2}
.thumb .u{fill:var(--raise);stroke:var(--muted);stroke-width:1}
.thumb .hi{fill:var(--edge);stroke:var(--edge-bright);stroke-width:1}
.card.on .thumb .hi{stroke:var(--accent)}

/* ---- picked chairs: name, measured size, a bordered stepper, a remove ---- */
.picked{display:flex;flex-direction:column;gap:7px;margin-top:12px}
.pk{display:flex;align-items:center;gap:11px;border:1px solid var(--edge);background:var(--bg);
  padding:7px 9px}
.pk .nm{min-width:0;flex:1}
.pk .nm b{display:block;font-size:12.5px;font-weight:540;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.pk .nm span{font-family:var(--mono);font-size:10px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.step{display:flex;align-items:center;border:1px solid var(--edge-bright);flex:none}
.step button{width:28px;height:24px;background:transparent;border:0;color:var(--fg-2);
  font-size:14px;line-height:1;padding:0}
.step button:hover:enabled{background:var(--raise);color:var(--accent)}
.step .n{width:28px;text-align:center;font-family:var(--mono);font-size:12px;line-height:24px;
  border-left:1px solid var(--edge-bright);border-right:1px solid var(--edge-bright);
  font-variant-numeric:tabular-nums}
button.x{background:transparent;border:0;color:var(--muted);font-size:15px;padding:0 4px;
  line-height:1;flex:none}
button.x:hover:enabled{background:transparent;color:var(--warn)}

/* ---- object rows ---- */
table{width:100%;border-collapse:collapse;margin-top:10px}
th{text-align:left;font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);padding:0 9px 6px;font-weight:600}
td{text-align:left;padding:7px 9px;border-top:1px solid var(--edge);font-size:12.5px;
  vertical-align:middle}
td.m{font-family:var(--mono);font-size:11px;color:var(--fg-2);font-variant-numeric:tabular-nums}
td.r{text-align:right}

/* ---- the run summary ---- */
.sum{font-family:var(--mono);font-size:11.5px}
.sum .r{display:flex;justify-content:space-between;gap:10px;padding:6px 0;
  border-bottom:1px solid var(--edge)}
.sum .r span{color:var(--muted);flex:none}
/* min-width:0 so the value can actually ellipsise: a flex item will not shrink below its content
   without it, and a long Objaverse table label would push the field name off the panel instead. */
.sum .r b{font-weight:520;text-align:right;font-variant-numeric:tabular-nums;min-width:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.warnbox{margin-top:16px;border:1px solid var(--edge);border-left:2px solid var(--steel);
  padding:9px 11px;font-size:11.5px;color:var(--fg-2);line-height:1.5}
.foot{margin-top:16px;font-size:11px;color:var(--muted);line-height:1.5}

/* ---- progress, and the >4-option list ---- */
.meter{background:var(--bg);border:1px solid var(--edge-bright);height:10px;overflow:hidden;
  margin-top:16px}
.meter>div{background:var(--accent);height:100%;width:0;transition:width .25s linear}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.list{max-height:340px;overflow:auto;border:1px solid var(--edge);background:var(--bg);
  margin-top:14px}
.list label{margin:0;padding:7px 11px;display:block;cursor:pointer;font-size:12.5px;
  font-family:var(--mono);border-top:1px solid var(--edge)}
.list label:first-child{border-top:0}
.list label:hover{background:var(--panel-2)}

/* The kitchen-type <select> is kept as the value the rest of the form reads and posts, but the
   room CARDS above it are what you actually operate -- so it is mirrored out of sight rather than
   shown twice. tabindex=-1 + aria-hidden so it is not a second, invisible tab stop. */
.mirror{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0 0 0 0);white-space:nowrap;border:0}
</style></head><body>
<div class="page"><div class="app">
<div class="bar"><div class="mark"><b>Kitchen studio</b><span>SIMVLA</span></div></div>
__BODY__
</div></div>
</body></html>
"""


def _page(body: str) -> str:
    """A wizard page: the shared theme, then this step's body, in the common shell."""
    return _SHELL.replace("__THEME__", THEME_CSS).replace("__BODY__", body)


_FORM_BODY = """
<div class="body">
<div class="main">

  <div class="sect">
    <h2>Room</h2>
    <p class="hint">Five procedural layouts. Walls and materials are randomised per seed.</p>
    <div id="rooms" class="cards"></div>
    <select id="ktype" class="mirror" tabindex="-1" aria-hidden="true"></select>
    <div class="row">
      <button id="browse">Browse layouts…</button>
      <button id="roll" class="ghost">Roll a random one</button>
      <span class="data" id="seedline"></span>
    </div>
  </div>

  <div class="sect">
    <h2>Table</h2>
    <p class="hint">Procedural and Objaverse both, every one measured and inspected. A kitchen
      without a table is a valid kitchen — cancel the picker to decline one.</p>
    <div class="row">
      <button id="tablebtn">Choose a table…</button>
      <button id="tableedit" class="ghost">Edit…</button>
      <span class="data" id="tableline"></span>
    </div>
  </div>

  <div class="sect">
    <h2>Chairs</h2>
    <p class="hint">Add copies with − and +. Edit… resizes a chair or changes its
      material. Chairs turn to face the table, and every seat is checked against the open pose of
      each door and drawer.</p>
    <div id="chairrows" class="picked"></div>
    <div class="row">
      <button id="chairbtn">Add a chair…</button>
      <span class="data" id="chairline"></span>
    </div>
  </div>

  <div class="sect">
    <h2>Objects</h2>
    <p class="hint">Placed on a support surface and checked for clearance before anything is
      written.</p>
    <table id="rows"></table>
    <div class="row">
      <select id="otype"></select>
      <select id="oloc"></select>
      <button id="add">Add object</button>
      <button id="clear" class="ghost">Clear all</button>
    </div>
  </div>

</div>

<div class="side">
  <h3>This run</h3>
  <div class="sum">
    <div class="r"><span>number</span><b id="sum-num">—</b></div>
    <div class="r"><span>room</span><b id="sum-room">—</b></div>
    <div class="r"><span>seed</span><b id="sum-seed">—</b></div>
    <div class="r"><span>table</span><b id="sum-table">—</b></div>
    <div class="r"><span>chairs</span><b id="sum-chairs">0</b></div>
    <div class="r"><span>objects</span><b id="sum-objects">0</b></div>
  </div>

  <label class="fld" style="margin-top:18px"><span>Kitchen number</span>
    <input id="num" type="text" inputmode="numeric" autocomplete="off"></label>
  <div id="numwarn" class="warn"></div>

  <div id="err" class="err"></div>
  <div class="row"><button id="generate" class="primary">Generate scene</button></div>
  <div class="row"><button id="compose" class="ghost">Compose on existing…</button></div>

  <div class="warnbox">Nothing is written to disk until you accept the 3D preview.</div>
  <div class="foot">Compose on existing loads that kitchen back from its USD and opens the task
    composer on it. It writes no scene.</div>
</div>
</div>

<script>
(function () {
  var DATA = __FORM_PAYLOAD__;
  var rows = [];
  var seed = null;
  var table = null;   // a variant key, or null for "no table" -- there is no separate checkbox
  // The table's own edit. Two plain values beside `table` rather than a dict in it, because a
  // kitchen has exactly one table and `table` stays the identity everything else here reads.
  var tableScale = 1;
  var tableMaterial = null;
  // The chairs, ONE OBJECT PER PLACED CHAIR, in the order they were picked:
  // {uid, scale, material}. Not a list of uids -- two chairs of the same design can now differ,
  // so the instance carries its own properties. The picker APPENDS to this (that is what makes
  // the flow add-another), and every change to it goes through the server.
  var chairs = [];
  // The session's robot, chosen on the first step of the run. This form does not offer it and
  // does not show it -- it is not part of the kitchen -- but every submit has to carry it, or the
  // choice is lost the first time the author touches this page. Read OUTSIDE the restore block
  // below on purpose: on the very first form there is no kitchen_name yet, which is exactly the
  // submit that would otherwise drop it.
  var robot = (DATA.state && DATA.state.robot !== undefined) ? DATA.state.robot : null;
  var ktype = document.getElementById('ktype');
  var otype = document.getElementById('otype');
  var oloc = document.getElementById('oloc');
  var err = document.getElementById('err');

  // Plan-view glyphs for the layouts kitchen_build offers: worktop runs as lines, units as blocks,
  // and `.hi` for whatever gives the layout its name (the island, the peninsula) -- which is the
  // one part of a glyph that takes the accent, and only on the card that is currently chosen.
  //
  // SCHEMATIC, NOT RENDERS. Real plan-view thumbnails would need a per-asset render stage that does
  // not exist; these are five hand-drawn diagrams that cost nothing and are honest about being
  // diagrams. A kitchen type with no entry here (a sixth added to KITCHEN_BUILDERS) still gets a
  // card, with an empty plate rather than a broken one.
  var GLYPHS = {
    island: '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/>'
      + '<path class="w" d="M18 62h84M18 62V34h84v28"/>'
      + '<rect class="u" x="30" y="44" width="26" height="14"/>'
      + '<rect class="u" x="66" y="44" width="24" height="14"/>'
      + '<rect class="hi" x="48" y="66" width="30" height="10"/></svg>',
    l_shaped: '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/>'
      + '<path class="w" d="M20 66V30h70M20 66h18"/>'
      + '<rect class="u" x="24" y="36" width="14" height="24"/>'
      + '<rect class="u" x="46" y="34" width="30" height="12"/></svg>',
    peninsula: '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/>'
      + '<path class="w" d="M18 32h80M18 32v34h30"/>'
      + '<rect class="u" x="24" y="38" width="26" height="12"/>'
      + '<rect class="u" x="62" y="38" width="26" height="12"/>'
      + '<rect class="hi" x="52" y="56" width="12" height="26"/></svg>',
    single_wall: '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/>'
      + '<path class="w" d="M16 40h88"/>'
      + '<rect class="u" x="24" y="44" width="26" height="14"/>'
      + '<rect class="u" x="58" y="44" width="30" height="14"/></svg>',
    u_shaped: '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/>'
      + '<path class="w" d="M22 68V30h64v38"/>'
      + '<rect class="u" x="28" y="36" width="16" height="14"/>'
      + '<rect class="u" x="56" y="36" width="24" height="14"/>'
      + '<rect class="u" x="28" y="56" width="16" height="10"/></svg>'
  };
  var EMPTY_GLYPH =
    '<svg class="thumb" viewBox="0 0 120 90"><rect class="g" width="120" height="90"/></svg>';
  // Display only. The VALUE posted is always the bare type name, so a prettier label here can
  // never change which builder runs; an unlisted type falls back to its own name.
  var ROOM_LABELS = {island: 'Island', l_shaped: 'L-shaped', peninsula: 'Peninsula',
                     single_wall: 'Single wall', u_shaped: 'U-shaped'};

  function roomLabel(name) {
    return ROOM_LABELS[name] || String(name).replace(/_/g, ' ');
  }

  function opt(value, label) {
    var o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    return o;
  }

  ktype.appendChild(opt('random', 'Random'));
  DATA.kitchen_types.forEach(function (t) { ktype.appendChild(opt(t, t.replace(/_/g, ' '))); });
  // Grouped like the placement menu below, and for a stronger reason: the heading is the only thing
  // here that says whether a type grasps with real BODex data. The option's VALUE stays the bare
  // type name, so an annotated label is still submitted as itself.
  var otypeGroups = {};
  DATA.object_types.forEach(function (t) {
    if (!otypeGroups[t.group]) {
      otypeGroups[t.group] = document.createElement('optgroup');
      otypeGroups[t.group].label = t.group;
      otype.appendChild(otypeGroups[t.group]);
    }
    otypeGroups[t.group].appendChild(opt(t.type, t.label));
  });
  ktype.value = DATA.kitchen_types[0];

  // Coming back from a gallery, the form is re-rendered from scratch: everything the user had
  // typed arrives in DATA.state and is put back here.
  if (DATA.state && DATA.state.kitchen_name) {
    document.getElementById('num').value = DATA.state.kitchen_num || '';
    checkNumber();      // .value set programmatically fires no 'input' event -- run the check by hand
    ktype.value = DATA.state.kitchen_name;
    seed = (DATA.state.seed === undefined) ? null : DATA.state.seed;
    table = (DATA.state.table === undefined) ? null : DATA.state.table;
    tableScale = DATA.state.table_scale || 1;
    tableMaterial = (DATA.state.table_material === undefined) ? null : DATA.state.table_material;
    chairs = (DATA.state.chairs || []).slice();
    (DATA.state.objects || []).forEach(function (o) {
      rows.push({type: o.type, loc: o.loc, locLabel: labelFor(o.loc), mesh: o.mesh});
    });
  }

  // Locations are per kitchen type, so 'Random' cannot show a location list. Picking one here the
  // moment it is chosen keeps the menu honest and still gives you the old Change-Kitchen-Type roll.
  function currentType() {
    if (ktype.value === 'random') {
      ktype.value = DATA.kitchen_types[Math.floor(Math.random() * DATA.kitchen_types.length)];
    }
    return ktype.value;
  }

  function labelFor(loc) {
    var found = '';
    (DATA.locations[ktype.value] || []).forEach(function (p) {
      if (p.loc === loc) { found = p.label; }
    });
    return found;
  }

  // The one path a kitchen-type change takes, whichever control started it: a room card, the Roll
  // button, or the mirrored <select>'s own change event. Every one of them has to do the same three
  // things, and the seed reset is the one that matters -- a layout picked in the gallery is a seed
  // for a SPECIFIC type and means nothing on another.
  function chooseType(name) {
    ktype.value = name;
    seed = null;          // a different type means the picked layout no longer applies
    showSeed();
    refreshLocations();   // resolves 'random' into a real type, so lighting a card comes after it
    showRooms();
    updateRun();
  }

  // One card per kitchen type. These ARE the type control; the <select> is the value they write to
  // and what submit() posts, kept in the DOM (mirrored out of sight) so the rest of the form goes
  // on reading ktype.value exactly as it did.
  function buildRooms() {
    var box = document.getElementById('rooms');
    box.innerHTML = '';
    DATA.kitchen_types.forEach(function (name) {
      var card = document.createElement('button');
      card.type = 'button';
      card.className = 'card';
      card.setAttribute('data-type', name);
      card.innerHTML = GLYPHS[name] || EMPTY_GLYPH;   // a literal from this file, never page data
      var cap = document.createElement('div');
      cap.className = 'cap';
      var name_el = document.createElement('b');
      name_el.textContent = roomLabel(name);
      cap.appendChild(name_el);
      card.appendChild(cap);
      card.addEventListener('click', function () { chooseType(name); });
      box.appendChild(card);
    });
  }

  function showRooms() {
    var cards = document.getElementById('rooms').children;
    for (var i = 0; i < cards.length; i++) {
      var on = (cards[i].getAttribute('data-type') === ktype.value);
      cards[i].classList.toggle('on', on);
      cards[i].setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  // The run summary, kept in step with the form as it is edited. Every field is read from the SAME
  // variables submit() posts, so it cannot end up describing a run other than the one about to be
  // built -- which is the whole point of showing it.
  function updateRun() {
    setRun('sum-num', document.getElementById('num').value.trim() || '—');
    setRun('sum-room', roomLabel(ktype.value));
    setRun('sum-seed', (seed === null) ? 'random' : String(seed));
    setRun('sum-table', tableName(table) || 'none');
    setRun('sum-chairs', String(chairs.length));
    setRun('sum-objects', String(rows.length));
  }

  // title as well as text: the summary's value column ellipsises, and a truncated table name with
  // no way to read it in full is worse than no summary row.
  function setRun(id, text) {
    var el = document.getElementById(id);
    el.textContent = text;
    el.title = text;
  }

  function showSeed() {
    document.getElementById('seedline').textContent =
      (seed === null) ? 'random layout' : ('layout #' + seed + ' — the one you picked');
    updateRun();
  }

  // "Table 7 · Long dining". The label alone is a position in a registry and says nothing about
  // the shape, so the data line follows it here as well as in the gallery -- see
  // kitchen_build.FURNITURE_NUMBERING.
  function tableName(key) {
    var label = key && DATA.table_variants ? DATA.table_variants[key] : null;
    if (!label) { return null; }
    var detail = DATA.table_details ? DATA.table_details[key] : '';
    return detail ? (label + ' · ' + detail) : label;
  }

  function showTable() {
    var name = tableName(table);
    // DATA.table_edit is the SERVER's reading of the same two values this form posts back (see
    // kitchen_wizard._chair_edit_line), so an edited table says so here rather than looking
    // identical to an unedited one.
    var edit = DATA.table_edit || '';
    document.getElementById('tableline').textContent =
      name ? ('table: ' + name + (edit ? ' · ' + edit : '')) : 'no table';
    // Nothing to edit without a table, and the server refuses edit_table in that state anyway --
    // the same defence in depth the chair picker's cap check is.
    document.getElementById('tableedit').disabled = !table;
    updateRun();
  }

  // The chair line, one row per DISTINCT chair with its count and its +/- buttons, and whether
  // the picker is still on offer.
  //
  // The picker is hidden at the cap rather than left live: the server refuses a seventh chair, and
  // a button that always comes back rejected is worse than no button. It is hidden with nothing to
  // offer too -- DATA.chair_variants is empty on a checkout with no chair library.
  //
  // The rows come from DATA.chair_rows -- the SERVER's grouping of the same list this form posts
  // back (see kitchen_wizard._chair_rows) -- and +/- are submits, not local edits, because the
  // cap refusal has to be able to say why it refused. So this function never has to regroup:
  // every change goes through the server and comes back as a freshly rendered form.
  function showChairs() {
    var offered = Object.keys(DATA.chair_variants || {}).length;
    var max = DATA.max_chairs || 0;
    var btn = document.getElementById('chairbtn');
    btn.disabled = (offered === 0) || (chairs.length >= max);
    document.getElementById('chairline').textContent =
      (offered === 0) ? 'no chair library on this machine'
        : (chairs.length ? (chairs.length + ' of ' + max + ' chairs') : 'no chairs');

    var list = document.getElementById('chairrows');
    list.innerHTML = '';
    // A row is now one DISTINCT chair -- same uid, same scale, same material -- so its index in
    // this list is what +, − and Edit… name. Not the uid: two rows can share a uid once one of
    // them is edited, and a uid would then name both. See kitchen_wizard._chair_rows.
    (DATA.chair_rows || []).forEach(function (row, rowIndex) {
      var pk = document.createElement('div');
      pk.className = 'pk';

      var nm = document.createElement('div');
      nm.className = 'nm';
      var label = document.createElement('b');
      label.textContent = row.label;
      // What was measured off this chair's own exported mesh (kitchen_wizard.form_payload reads it
      // straight from the registry), then what the author changed about it. Both blank-tolerant:
      // an unmeasured chair and an unedited one each show nothing rather than a made-up number.
      var size = document.createElement('span');
      size.textContent = row.edit ? ((row.size ? row.size + ' · ' : '') + row.edit)
                                  : (row.size || '');
      nm.appendChild(label);
      nm.appendChild(size);
      pk.appendChild(nm);

      // The size scale and the material of the chairs in THIS row. They are identical by
      // construction, so this edits all of them; a chair edited to something else falls out of
      // this row into one of its own.
      var edit = document.createElement('button');
      edit.textContent = 'Edit…';
      edit.className = 'ghost';
      edit.title = 'size and material';
      edit.addEventListener('click', function () {
        submit('edit_chair', undefined, rowIndex);
      });
      pk.appendChild(edit);

      var step = document.createElement('div');
      step.className = 'step';
      // One fewer of THIS chair. At a count of one it takes the row away with it, which is what
      // the old per-instance Remove did too.
      var minus = document.createElement('button');
      minus.textContent = '−';
      minus.title = 'one fewer of this chair';
      minus.addEventListener('click', function () {
        submit('remove_chair', undefined, rowIndex);
      });
      step.appendChild(minus);
      var count = document.createElement('div');
      count.className = 'n';
      count.textContent = String(row.count);
      step.appendChild(count);
      // One more of the SAME chair -- the same design AND the same edit -- without another trip
      // through the gallery, which is the whole point of the row. Disabled at the cap for the same
      // reason the picker is, and the server refuses it there anyway (this form is not the only
      // thing that can POST).
      var plus = document.createElement('button');
      plus.textContent = '+';
      plus.disabled = !row.can_add;
      plus.title = row.can_add ? 'one more of this chair' : ('at most ' + max + ' chairs');
      plus.addEventListener('click', function () {
        submit('add_chair', undefined, rowIndex);
      });
      step.appendChild(plus);
      pk.appendChild(step);

      list.appendChild(pk);
    });
    updateRun();
  }

  function refreshLocations() {
    var name = currentType();
    oloc.innerHTML = '';
    var groups = {};
    (DATA.locations[name] || []).forEach(function (p) {
      if (!groups[p.group]) {
        groups[p.group] = document.createElement('optgroup');
        groups[p.group].label = p.group;
        oloc.appendChild(groups[p.group]);
      }
      groups[p.group].appendChild(opt(String(p.loc), p.label));
    });
    // A location legal on the old type may not exist on the new one; drop those rows rather than
    // let the server reject the whole form later.
    var legal = {};
    (DATA.locations[name] || []).forEach(function (p) { legal[p.loc] = p.label; });
    rows = rows.filter(function (r) { return legal[r.loc] !== undefined; });
    renderRows();
  }

  function renderRows() {
    var table = document.getElementById('rows');
    table.innerHTML = '';
    updateRun();
    if (!rows.length) { return; }
    var head = table.insertRow();
    ['Object', 'Placement', 'Mesh', '', ''].forEach(function (t) {
      var th = document.createElement('th');
      th.textContent = t;
      head.appendChild(th);
    });
    rows.forEach(function (r, i) {
      var tr = table.insertRow();
      tr.insertCell().textContent = r.type;
      tr.insertCell().textContent = r.locLabel;
      var mesh = tr.insertCell();
      mesh.className = 'm';                  // a uid, read character by character: mono
      mesh.textContent = r.mesh ? shortMesh(r.mesh) : 'random';
      mesh.title = r.mesh || 'a random matching mesh, chosen at build time';
      var choose = document.createElement('button');
      choose.textContent = 'Choose…';
      choose.className = 'ghost';
      choose.addEventListener('click', function () { submit('pick_mesh', i); });
      tr.insertCell().appendChild(choose);
      // A local splice, not a submit: this row exists only in the page until something is posted.
      // (That is why an object row can carry an × and a CHAIR row cannot -- a chair count only
      // changes by asking the server, which has to be free to refuse it.)
      var remove = document.createElement('button');
      remove.textContent = '×';
      remove.className = 'x';
      remove.title = 'remove this object';
      remove.addEventListener('click', function () { rows.splice(i, 1); renderRows(); });
      var cell = tr.insertCell();
      cell.className = 'r';
      cell.appendChild(remove);
    });
  }

  // '<root>/use_data/core_mug_1038e4ea/mesh/simplified.obj' -> 'core_mug_1038e4ea'. Every mesh is
  // called simplified.obj, so the filename identifies nothing.
  //
  // Empty segments are dropped before counting: an absolute path's leading '/' otherwise produces
  // a leading '' that throws the "-3" index off by one -- '/data/mug.obj'.split('/') is
  // ['', 'data', 'mug.obj'], length 3, and parts[length - 3] was '', a blank cell where a name
  // belongs. A path with fewer than three real segments falls back to showing itself in full,
  // which is never blank.
  function shortMesh(path) {
    var parts = String(path).split('/').filter(function (p) { return p !== ''; });
    return parts.length >= 3 ? parts[parts.length - 3] : String(path);
  }

  function checkNumber() {
    var n = parseInt(document.getElementById('num').value.trim(), 10);
    var warn = document.getElementById('numwarn');
    warn.textContent = (DATA.existing.indexOf(n) >= 0)
      ? ('kitchen ' + n + ' already exists — Accept & Generate will overwrite it')
      : '';
    updateRun();
  }

  function submit(action, row, chairRow) {
    err.textContent = '';
    var body = {
      action: action,
      kitchen_num: document.getElementById('num').value.trim(),
      kitchen_name: ktype.value,
      seed: seed,
      table: table,
      table_scale: tableScale,
      table_material: tableMaterial,
      chairs: chairs,
      robot: robot,
      objects: rows.map(function (r) { return {type: r.type, loc: r.loc, mesh: r.mesh}; })
    };
    if (row !== undefined) { body.row = row; }
    // Which chair row +, − or Edit… was pressed on, by index. Sent only by those three actions,
    // so every other submit posts exactly the body it always did.
    if (chairRow !== undefined) { body.chair_row = chairRow; }
    window.simvlaPost('/setup', body).then(function (r) {
      // On 200 the director advances the run — to a gallery, or into the build — and the poller
      // reloads this page for us.
      if (r.status !== 200) {
        err.textContent = r.body.reason || ('rejected (' + r.status + ')');
      }
    });
  }

  document.getElementById('add').addEventListener('click', function () {
    if (!oloc.value) { err.textContent = 'This kitchen type has no placement locations.'; return; }
    rows.push({
      type: otype.value,
      loc: parseInt(oloc.value, 10),
      locLabel: oloc.options[oloc.selectedIndex].textContent.trim(),
      mesh: null
    });
    renderRows();
  });
  document.getElementById('clear').addEventListener('click', function () { rows = []; renderRows(); });
  document.getElementById('num').addEventListener('input', checkNumber);
  // Still bound, even though the <select> is mirrored out of sight: it remains the value of record,
  // and anything that sets it and dispatches 'change' must take the same path a card click does.
  ktype.addEventListener('change', function () { chooseType(ktype.value); });
  // The Random option the <select> has always carried, as its own control now that the cards show
  // the five real layouts. currentType() rolls it into one of them the moment it is chosen, which
  // is exactly what picking "Random" in the old menu did.
  document.getElementById('roll').addEventListener('click', function () { chooseType('random'); });
  document.getElementById('generate').addEventListener('click', function () { submit('generate'); });
  document.getElementById('compose').addEventListener('click', function () {
    submit('compose_existing');
  });
  document.getElementById('browse').addEventListener('click', function () {
    submit('pick_kitchen');
  });
  document.getElementById('tablebtn').addEventListener('click', function () {
    submit('pick_table');
  });
  document.getElementById('tableedit').addEventListener('click', function () {
    submit('edit_table');
  });
  document.getElementById('chairbtn').addEventListener('click', function () {
    submit('pick_chair');
  });

  buildRooms();
  refreshLocations();
  showRooms();
  showSeed();
  showTable();
  showChairs();
})();
</script>
"""


def render_form_page(payload: dict) -> str:
    """The setup page: kitchen number + type, object rows, and the two ways out of this step."""
    return _page(_FORM_BODY.replace("__FORM_PAYLOAD__", script_json(payload)))


_CHOICE_BODY = """
<div class="main solo">
<h1>__TITLE__</h1>
<div class="sub">__TEXT__</div>
<div id="opts" class="opts"></div>
<div id="err" class="err"></div>
</div>
<script>
(function () {
  var OPTIONS = __OPTIONS__;
  var box = document.getElementById('opts');
  OPTIONS.forEach(function (o, i) {
    var b = document.createElement('button');
    b.textContent = o.label;
    if (i === 0) { b.className = 'primary'; }
    b.addEventListener('click', function () {
      box.querySelectorAll('button').forEach(function (x) { x.disabled = true; });
      window.simvlaPost('/answer', {id: o.id}).then(function (r) {
        // On 200 the director moves on and the poller reloads us.
        if (r.status !== 200) {
          document.getElementById('err').textContent =
            r.body.reason || ('rejected (' + r.status + ')');
          box.querySelectorAll('button').forEach(function (x) { x.disabled = false; });
        }
      });
    });
    box.appendChild(b);
  });
})();
</script>
"""

_LIST_CHOICE_BODY = """
<div class="main solo">
<h1>__TITLE__</h1>
<div class="sub">__TEXT__</div>
<div id="list" class="list"></div>
<div class="opts">
  <button id="ok" class="primary">OK</button>
</div>
<div id="err" class="err"></div>
</div>
<script>
(function () {
  var OPTIONS = __OPTIONS__;   // the radio choices -- the cancel option, if any, is never in here
  var CANCEL = __CANCEL__;     // {id, label}, or null when this step offers no cancellation
  var list = document.getElementById('list');
  OPTIONS.forEach(function (o, i) {
    var label = document.createElement('label');
    var radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'pick';
    radio.value = o.id;
    if (i === 0) { radio.checked = true; }
    label.appendChild(radio);
    label.appendChild(document.createTextNode(' ' + o.label));
    list.appendChild(label);
  });

  function answer(id) {
    window.simvlaPost('/answer', {id: id}).then(function (r) {
      if (r.status !== 200) {
        document.getElementById('err').textContent =
          r.body.reason || ('rejected (' + r.status + ')');
      }
    });
  }
  document.getElementById('ok').addEventListener('click', function () {
    var picked = document.querySelector('input[name="pick"]:checked');
    if (!picked) { document.getElementById('err').textContent = 'Pick one first.'; return; }
    answer(picked.value);
  });
  // Only ever added when the caller's own options included one with id "" -- so the button on
  // screen can never answer an id publish() did not allow, and a step that offers no cancellation
  // gets no such button at all.
  if (CANCEL) {
    var cancelBtn = document.createElement('button');
    cancelBtn.id = 'cancel';
    cancelBtn.textContent = CANCEL.label;
    cancelBtn.addEventListener('click', function () { answer(CANCEL.id); });
    document.querySelector('.opts').appendChild(cancelBtn);
  }
})();
</script>
"""

_EDIT_BODY = """
<style>
__THEME__
/* Wider than kitchen_gallery's fixed 280px and narrower than the preview's clamp: this panel
   carries a slider that has to be draggable with the pointer AND a live measurement line under
   it ("1.30x  ·  2.34 x 1.11 m"), which truncates at 280. It does not carry the preview's long
   joint paths, so it does not need 460. */
#edit{position:fixed;top:0;right:0;width:clamp(300px, 27vw, 380px);max-height:100vh;
  overflow-y:auto;background:var(--panel);color:var(--fg);box-sizing:border-box;padding:16px 18px;
  z-index:9999;font:12.5px/1.55 var(--ui);border-left:1px solid var(--edge)}
#edit h2{font-size:14px;margin:0 0 3px;font-weight:620;letter-spacing:-.01em}
#edit h3{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin:18px 0 7px;font-weight:600;font-family:var(--mono)}
#edit .sub{color:var(--fg-2);font-size:11.5px}
#edit .note{color:var(--muted);font-size:11px;line-height:1.5;margin-top:7px}
#edit .data{font-family:var(--mono);font-size:12px;color:var(--fg);
  font-variant-numeric:tabular-nums;margin-top:6px}
#edit input[type=range]{width:100%;margin:2px 0 0}
/* The measured band, drawn as a bar under the slider rather than as a sentence: the point is
   WHERE on the travel it ends, and a sentence cannot say that. */
#edit .band{position:relative;height:5px;background:var(--edge);margin:3px 0 2px}
#edit .band i{position:absolute;top:0;bottom:0;background:var(--warn-dim);
  border-right:1px solid var(--warn)}
#edit .ends{display:flex;justify-content:space-between;color:var(--muted);
  font-family:var(--mono);font-size:10px}
#edit .row{display:flex;gap:7px;margin:14px 0 0;align-items:center}
#edit select{font:inherit;font-size:12px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:5px 8px;width:100%}
#edit .check{display:flex;gap:7px;align-items:center;cursor:pointer}
#edit .check input:disabled{cursor:default}
#edit .swatchrow{display:flex;gap:8px;align-items:center;margin-top:7px}
#edit .chip{width:26px;height:26px;border:1px solid var(--edge-bright);flex:0 0 auto}
#edit button{font:inherit;font-size:12px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:5px 10px;cursor:pointer}
#edit button:hover:enabled{background:var(--edge);color:var(--accent)}
#edit button:focus-visible,#edit input:focus-visible,#edit select:focus-visible{
  outline:1px solid var(--accent);outline-offset:1px}
#edit #accept{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);
  font-weight:640}
#edit #accept:hover:enabled{background:var(--accent-lift);color:var(--accent-fg)}
#edit .err{color:var(--err);font-size:11.5px;margin-top:9px;min-height:15px}
</style>
<div id="edit">
  <h2 id="edit-title"></h2>
  <div class="sub" id="edit-text"></div>

  <h3>Size</h3>
  <input type="range" id="scale-slider" min="0" max="1" step="1" value="0">
  <div class="band" id="band"></div>
  <div class="ends"><span id="scale-lo"></span><span id="scale-hi"></span></div>
  <div class="data" id="scaleline"></div>
  <div class="note" id="bandnote"></div>
  <div class="note">Width, depth and height together. A table's work surface is 0.74 m as exported
    and the chairs are drawn to tuck under it, so resizing one and not the other breaks that
    pairing — watch the height above. The view resizes as you drag; only the value you accept
    is sent.</div>
  <div class="note" id="editnote"></div>

  <h3>Material</h3>
  <select id="mat"></select>
  <div class="swatchrow"><span class="chip" id="swatch"></span>
    <span class="note" style="margin:0">Approximate. The real material is an Omniverse MDL, bound
      on the USD stage at commit — it does not exist in this preview at all, so this is one
      representative colour for the group, not the surface you will get.</span></div>

  <h3>Scale reference</h3>
  <select id="robot"></select>
  <div class="note" id="figurenote"></div>

  <div id="tablesection">
    <h3>The table it goes with</h3>
    <label class="check"><input type="checkbox" id="showtable"> <span id="tablelabel"></span></label>
    <div class="note" id="tablenote"></div>
  </div>

  <div class="row">
    <button id="accept">Use it</button>
    <button id="cancel">Cancel</button>
  </div>
  <div class="err" id="err"></div>
</div>
<script>
(function () {
  var DATA = __EDIT_PAYLOAD__;
  var index = 0;
  var material = DATA.material;
  var err = document.getElementById('err');

  DATA.scales.forEach(function (s, i) {
    if (Math.abs(s - DATA.scale) < 1e-6) { index = i; }
  });

  function scaleNow() { return DATA.scales[index]; }

  // The answer id, and the ONE place its shape is written. The director splits on the first colon
  // (a scale can never contain one), and every combination this page can produce is in the step's
  // own option list -- which is what makes it impossible for a click here to be refused with a 400.
  // The slider's value is an INDEX into DATA.scales, so it cannot produce a scale that is not in
  // that list however it is dragged.
  function answerId() {
    return scaleNow().toFixed(2) + ':' + material;
  }

  document.getElementById('edit-title').textContent = DATA.title;
  document.getElementById('edit-text').textContent = DATA.text;
  document.getElementById('editnote').textContent = DATA.note || '';

  var slider = document.getElementById('scale-slider');
  slider.min = 0;
  slider.max = DATA.scales.length - 1;
  slider.value = index;
  document.getElementById('scale-lo').textContent = DATA.scales[0].toFixed(2) + '×';
  document.getElementById('scale-hi').textContent =
    DATA.scales[DATA.scales.length - 1].toFixed(2) + '×';

  // The band: everything BELOW the measured floor is shaded, so the slider says where the
  // measured-safe region ends instead of pretending the whole travel is equivalent. No floor
  // (a table, or a chair the manifest never measured) means no shading and no note.
  if (DATA.measured_floor !== null && DATA.measured_floor !== undefined) {
    var lo = DATA.scales[0];
    var hi = DATA.scales[DATA.scales.length - 1];
    var mark = 100 * (DATA.measured_floor - lo) / (hi - lo);
    var shade = document.createElement('i');
    shade.style.left = '0';
    shade.style.width = Math.max(0, Math.min(100, mark)) + '%';
    document.getElementById('band').appendChild(shade);
    document.getElementById('bandnote').textContent = DATA.floor_note;
  }

  // ---- the 3D: the piece resizes as the slider moves ----------------------------------------
  // `scene` and `render` are globals from trimesh's bundled viewer, the same ones the preview and
  // the galleries reach for. The piece was added as ONE node with an identity transform, centred
  // in plan and standing on z = 0, so scaling it about its own origin resizes it in place -- it
  // neither lifts off the floor nor sinks through it, and it does not drift sideways.
  var piece = null;
  var pieceMeshes = [];
  // The chosen table, drawn beside the piece and hidden by the toggle. Client-side, unlike the
  // robot: it is a few hundred KB, so it is embedded once and shown or hidden without a round
  // trip, and nothing about the choice is being made here -- the table was chosen on the form.
  var tableNode = null;

  function findPiece() {
    if (typeof scene === 'undefined' || !scene) { return false; }
    // 'world' is trimesh's own root node, and waiting for IT rather than for the piece is what
    // makes this poll end on a page that has no piece to find -- piece=None is a supported state
    // (a chair whose OBJ has gone missing still gets its material), and gating on the piece left
    // such a page spinning for the full ~20 s before it drew its dimension labels. The global
    // `scene` is not that test: the template creates it before the GLB has decoded.
    if (!scene.getObjectByName('world')) { return false; }
    if (DATA.table) { tableNode = scene.getObjectByName(DATA.table.node); }
    piece = scene.getObjectByName(DATA.node);
    if (piece) { piece.traverse(function (o) { if (o.isMesh) { pieceMeshes.push(o); } }); }
    return true;
  }

  // The ring reflows as the chair grows, so the table stands further off a bigger chair -- which
  // is what the build does (kitchen_build._seats_around_table). The offsets are ENUMERATED per
  // slider step by the director, so this reads the ring's own answer rather than re-deriving a
  // placement rule the browser would then own a second copy of.
  function seatTable() {
    if (!tableNode || !DATA.table.offsets_m.length) { return; }
    var y = DATA.table.offsets_m[Math.min(index, DATA.table.offsets_m.length - 1)];
    tableNode.position.y = y;
    tableNode.updateMatrixWorld(true);
  }

__SPRITE_LABEL__

  // The dimension lines' text. Geometry cannot carry it -- a trimesh scene has no text primitive --
  // so it is rasterised here, by kitchen_preview.SPRITE_LABEL_JS, the same helper the gallery
  // labels its tiles with and the preview labels its own rules with. Always on here: this page
  // exists to judge a size, so the reference has no toggle to be off.
  //
  // Parented to the gltf 'world' node, which carries trimesh's scene frame, so `at` is in the same
  // metres as the piece.
  function buildScaleLabels() {
    if (typeof scene === 'undefined' || !scene) { return; }
    var root = scene.getObjectByName('world') || scene;
    (DATA.scale_labels || []).forEach(function (item) {
      var sprite = simvlaLabelSprite(item.text, item.width);
      sprite.position.set(item.at[0], item.at[1], item.at[2]);
      root.add(sprite);
    });
  }

  function applyScale() {
    if (piece) {
      // All three axes, height included -- see kitchen_build.FURNITURE_SCALE_IS_UNIFORM. The piece
      // stands on z = 0 with its origin at its own underside, so scaling z about that origin grows
      // it upward off the floor rather than sinking it, which is what the commit will build.
      piece.scale.set(scaleNow(), scaleNow(), scaleNow());
      piece.updateMatrixWorld(true);
    }
    seatTable();
    if (typeof render === 'function') { render(); }
  }

  function applyMaterial() {
    var hex = DATA.swatches[material];
    if (!hex) { return; }
    pieceMeshes.forEach(function (m) {
      if (m.material && m.material.color) {
        m.material = m.material.clone();
        m.material.color.set(hex);
        m.material.needsUpdate = true;
      }
    });
    if (typeof render === 'function') { render(); }
  }

  function showScale() {
    var s = scaleNow();
    var size = DATA.size || [];
    var line = s.toFixed(2) + '×';
    // All three, because all three move now -- a panel quoting only the plan would hide the one
    // dimension this change made interesting. See kitchen_build.FURNITURE_SCALE_IS_UNIFORM.
    if (size.length >= 3) {
      line += '  ·  ' + (size[0] * s).toFixed(2) + ' × ' + (size[1] * s).toFixed(2)
            + ' × ' + (size[2] * s).toFixed(2) + ' m';
    }
    if (Math.abs(s - 1) < 1e-6) { line += '  ·  as exported'; }
    document.getElementById('scaleline').textContent = line;
  }

  slider.addEventListener('input', function () {
    index = Number(this.value);
    showScale();
    applyScale();
  });

  var mat = document.getElementById('mat');
  DATA.materials.forEach(function (g) {
    var o = document.createElement('option');
    o.value = g;
    o.textContent = g;
    mat.appendChild(o);
  });
  mat.value = material;
  mat.addEventListener('change', function () {
    material = mat.value;
    document.getElementById('swatch').style.background = DATA.swatches[material] || '#888';
    applyMaterial();
  });
  document.getElementById('swatch').style.background = DATA.swatches[material] || '#888';

  // ---- the scale reference, chosen SERVER-SIDE ----------------------------------------------
  // Filled in JS from DATA.robot_choices exactly as the material menu above is filled from
  // DATA.materials, and 'No robot' is prepended here for the same reason the preview's selector
  // prepends it: it is a state, not a robot, so it belongs to the control and not to the registry.
  //
  // The choice is server-side because each robot is 5-13 MB of GLB at source resolution -- a
  // client-side switch would have to embed all three and charge every author 25 MB for a control
  // most of them never touch. So a pick POSTs it back and the director re-renders this page,
  // which is the same trade kitchen_preview's selector makes through the same `robot:` answer.
  //
  // AND IT CARRIES answerId() WITH IT, which is the whole difference from the preview's pick. The
  // slider and the material menu are client state that nothing has posted yet; a re-render from
  // the values this step opened on would silently throw away whatever was dialled in. answerId()
  // is exactly those two, and the director re-publishes the page from them -- so the robot
  // changes and nothing else does.
  var robots = document.getElementById('robot');
  [{id: '', label: 'No robot'}].concat(DATA.robot_choices).forEach(function (c) {
    var o = document.createElement('option');
    o.value = c.id;
    o.textContent = c.height_m ? c.label + ' · ' + c.height_m.toFixed(2) + ' m' : c.label;
    robots.appendChild(o);
  });
  robots.value = DATA.robot;
  robots.addEventListener('change', function () {
    robots.disabled = true;
    // Re-enabled on BOTH outcomes, so a rejected or unreachable POST cannot leave the control
    // stuck: on 200 the poller reloads this page anyway, and on anything else answer() has already
    // written the reason into #err.
    var live = function () { robots.disabled = false; };
    answer('robot:' + robots.value + ':' + answerId()).then(live, live);
  });

  var standing = DATA.robot_choices.filter(function (c) { return c.id === DATA.robot; })[0];
  document.getElementById('figurenote').textContent = standing
    ? 'Beside the piece: ' + standing.label + ', with a dimension line at its measured standing '
      + 'height of ' + standing.height_m.toFixed(3) + ' m. Both are drawn on this page only — '
      + 'neither is ever placed, checked for clashes or exported. Picking another re-renders this '
      + 'page, because only the robot you choose is sent with it; the size and material you '
      + 'have set come with it.'
    : 'No robot beside the piece, and none is sent with the page. Pick one to judge the size '
      + 'against something person-sized; the size and material you have set come with it.';

  // ---- the table this piece goes with -------------------------------------------------------
  // A chair's size only means something against the table it will stand at, so the chosen table --
  // at the scale and in the material the author set for it -- is drawn at the seat the build
  // would put this chair in. The toggle is here because the size of the piece ITSELF is easier to
  // read with nothing else in the view, and both readings are wanted.
  var showtable = document.getElementById('showtable');
  var tablelabel = document.getElementById('tablelabel');
  if (DATA.table) {
    showtable.checked = true;
    tablelabel.textContent = 'Show ' + DATA.table.label;
    document.getElementById('tablenote').textContent =
      (DATA.table.detail ? DATA.table.detail + '  ·  ' : '')
      + DATA.table.size.map(function (v) { return v.toFixed(2); }).join(' × ') + ' m, at the '
      + 'scale and material you set for it. It stands off this piece by the same seating ring the '
      + 'build uses, so it moves back as you enlarge the chair. The chair is drawn in the '
      + 'orientation it was exported in; the build turns it to face the table.';
  } else {
    showtable.checked = false;
    showtable.disabled = true;
    tablelabel.textContent = 'No table chosen';
    document.getElementById('tablenote').textContent = DATA.table_prompt;
    // THE WHOLE SECTION GOES when the caller did not ask for one. "The table it goes with" is a
    // question about a chair; on the TABLE's own edit step it is not a question at all, and a
    // disabled toggle reading "no table chosen" would be nonsense on the page where the author is
    // editing the very table they chose.
    if (!DATA.table_prompt) {
      document.getElementById('tablesection').style.display = 'none';
    }
  }
  showtable.addEventListener('change', function () {
    if (!tableNode) { return; }
    tableNode.visible = showtable.checked;
    if (typeof render === 'function') { render(); }
  });

  function answer(id) {
    return window.simvlaPost('/answer', {id: id}).then(function (r) {
      // On 200 the director moves on and the poller reloads us.
      if (r.status !== 200) {
        err.textContent = (r.body && r.body.reason) || ('rejected (' + r.status + ')');
      }
    });
  }
  document.getElementById('accept').addEventListener('click', function () { answer(answerId()); });
  // "" is this step's cancel id, exactly as the galleries use -- the director returns the state
  // untouched, so a piece that was never edited keeps its defaults.
  document.getElementById('cancel').addEventListener('click', function () { answer(''); });

  showScale();

  // The template decodes the GLB asynchronously, so the piece does not exist yet. Same poll the
  // preview panel uses; the panel above is live from the first paint either way, because the
  // slider's value and the answer it posts do not depend on the 3D at all.
  var tries = 0;
  var timer = setInterval(function () {
    if (++tries > 400 || findPiece()) {      // ~20s
      clearInterval(timer);
      applyScale();
      applyMaterial();
      buildScaleLabels();
      if (typeof render === 'function') { render(); }
    }
  }, 50);
})();
</script>
"""


_PROGRESS_BODY = """
<div class="main solo">
<h1>__TITLE__</h1>
<div class="sub">__NOTICE__</div>
<div class="meter"><div id="fill"></div></div>
<div id="detail" class="data" style="margin-top:9px">starting…</div>
<div class="note">This runs on the generator's main thread — Omniverse's USD context is not
thread-safe — so the page only watches.</div>
</div>
<script>
(function () {
  setInterval(function () {
    fetch('/progress', {cache: 'no-store'})
      .then(function (r) { return r.json(); })
      .then(function (p) {
        if (!p.total) { return; }
        document.getElementById('fill').style.width =
          Math.round(100 * p.done / p.total) + '%';
        document.getElementById('detail').textContent = p.label;
      })
      .catch(function () {});
  }, 500);
})();
</script>
"""

_DONE_BODY = """
<div class="main solo">
<h1>__TITLE__</h1>
<div class="sub">__TEXT__</div>
<div class="note">This run is finished — you can close this tab.</div>
</div>
"""


def _escape(text: str) -> str:
    """Text going into the page's HTML. Prim paths and file paths are not markup."""
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def render_choice_page(*, title: str, text: str, options) -> str:
    """A question with N buttons: the prim picker, the compose offer, and every notice.

    More than four options renders as a scrolling radio list with OK — the prim picker's shape —
    because a row of buttons stops being readable there. An option with id "" is not a radio choice:
    it is this step's opt-in to a Cancel button. In that layout it is pulled out of the radio list
    and rendered as the button instead, answering with the exact id `ask()` published for it; a step
    that does not include one gets no Cancel button at all. That keeps every button and radio on
    screen tied to an id the step actually published, so nothing shown can ever be refused with a
    400 -- the four-or-fewer layout already has this property for free, since it renders exactly
    the options it was given and nothing more.
    """
    options = [{"id": str(o["id"]), "label": str(o["label"])} for o in options]
    if len(options) > 4:
        list_options = [o for o in options if o["id"] != ""]
        cancel = next((o for o in options if o["id"] == ""), None)
        return _page(
            _LIST_CHOICE_BODY.replace("__TITLE__", _escape(title))
                .replace("__TEXT__", _escape(text))
                .replace("__OPTIONS__", script_json(list_options))
                .replace("__CANCEL__", script_json(cancel))
        )
    return _page(
        _CHOICE_BODY.replace("__TITLE__", _escape(title))
            .replace("__TEXT__", _escape(text))
            .replace("__OPTIONS__", script_json(options))
    )


#: The gap left between two robots standing side by side on the robot step, in metres, measured
#: between their slots -- and a slot is the wider of the assembly's plan footprint (the robot AND
#: its dimension line, see kitchen_preview.scale_reference_extents) and the circle the robot sweeps
#: as it turns. Wide enough that nothing on this page passes through anything else's, narrow enough
#: that all three fit in one framed view.
#:
#: A robot CAN still sweep across ITS OWN dimension line, and does: the line stands 5 cm clear of
#: the robot's standing footprint (kitchen_preview._RULE_GAP_M), while turning takes the robot out
#: to its half-diagonal, which is 10-15 cm further for two of the three. Left alone deliberately --
#: moving the rule out for this page would make the same assembly measure differently here than on
#: the preview and the edit page, to fix a thin ribbon being crossed by an arm on a page whose
#: question is how TALL the robot is.
_ROBOT_TILE_GAP_M = 0.55

#: How fast the robots turn on that page, in radians per second. A quarter turn takes ~12 s: slow
#: enough to read as a considered turntable rather than a spin, fast enough that a glance shows it
#: is moving. Payload data rather than a JS literal so the one number is here with the rest of the
#: page's geometry.
_ROBOT_SPIN_RAD_PER_S = 0.13


def robot_answer_ids(robots=None) -> list[str]:
    """Every answer the robot step can produce: one per robot, plus its Cancel.

    The galleries' rule (kitchen_gallery.gallery_options), applied to a page with no paging: the
    server refuses an id the step did not publish, so anything clickable must be in here. "" is
    Cancel, which the director reads as "keep the robot the session already has".
    """
    return [str(name) for name in (ROBOTS if robots is None else robots)] + [""]


def render_robot_page(*, title: str, text: str, robots=None, chosen: str = "") -> str:
    """The robot step: the three robots side by side, at TRUE SIZE, slowly turning.

    THE FIRST STEP OF THE RUN, before the setup form, because the robot is what every later size
    comparison is drawn against -- the edit step's figure, the preview's scale reference -- and
    picking it per page made each of those a separate decision about the same thing.

    Modelled on the galleries: one trimesh.Scene through trimesh's viewer template with a panel
    injected (kitchen_preview.scene_page), one answer id per tile, "" for Cancel. Three tiles, so
    there is no paging and no Prev/Next.

    NOT kitchen_gallery.kitchen_tiles, which is the one thing here that looks like reuse and is
    not: that helper normalises every tile to the same size and repaints it in one neutral tile
    grey. Both are exactly wrong for this page -- the whole question is which robot is how TALL
    (1.19 m against 1.61 m), and the livery is most of what makes a robot recognisable at a glance
    (see kitchen_preview.robot_mesh). So each robot is stood in the scene by the same
    add_scale_reference the preview and the edit page use, dimension line and all.

    THE PICK IS MADE IN THE PANEL, not by clicking the view. A click in a 3D view that commits is
    the misfire kitchen_gallery's own panel documents (a drag that ends a pixel from where it
    started is a click), and the fix there -- click selects, a second button commits -- buys
    nothing on a page with three named options. The view is here to be looked at.

    ALL THREE ROBOTS ARE EMBEDDED, which is the one place this page departs from the rule the
    preview and the edit page follow (one robot per page, chosen server-side, because each is
    5-13 MB of GLB). It cannot follow it: the choice being made IS which robot, and a page that
    showed one at a time would be asking the question and hiding the answer. The cost is measured
    in test_the_robot_step_shows_every_robot_at_its_own_height, and it is paid once per run.
    """
    import trimesh                                            # ^ see render_edit_page

    import math

    from kitchen_preview import (
        SPRITE_LABEL_JS,
        add_scale_reference,
        robot_mesh,
        robot_node,
        sanitize_three_name,
        scale_reference_extents,
        scene_page,
    )

    names = [str(n) for n in (ROBOTS if robots is None else robots)]
    unknown = [n for n in names if n not in ROBOTS]
    if unknown:
        raise ValueError(f"{unknown} are not in {sorted(ROBOTS)}")

    scene = trimesh.Scene()
    tiles = []
    labels = []
    # Left to right in registry order, each robot standing on the same y so the three dimension
    # lines can be read against each other.
    #
    # THE SLOT IS THE TURNING CIRCLE, not the standing footprint, because these robots rotate. A
    # robot spins about its own plan centre, so the x it can reach while turning is its plan
    # HALF-DIAGONAL (0.38-0.45 m for the three), which is 4-15 cm wider than the half-span it
    # stands in. Laying the row out on the static footprint would have each robot pass through its
    # neighbour's dimension line twice a revolution.
    #
    # `at` is the ASSEMBLY's plan centre (add_scale_reference says so, and the assembly is the
    # robot plus its rule), so the walk is kept in the robot's own frame and converted at the last
    # step: `origin` is where the robot stands, and `at` is that shifted by the assembly's offset.
    walked = 0.0
    for name in names:
        x_lo, x_hi, _y_lo, _y_hi = scale_reference_extents(name)
        mesh = robot_mesh(name)
        turn = 0.0 if mesh is None else float(
            math.hypot(mesh.extents[0], mesh.extents[1]) / 2.0
        )
        lo, hi = min(x_lo, -turn), max(x_hi, turn)
        origin = walked - lo
        walked = origin + hi + _ROBOT_TILE_GAP_M
        _nodes, drawn = add_scale_reference(
            scene, at=(origin + (x_lo + x_hi) / 2.0, 0.0), robot=name
        )
        labels += drawn
        tiles.append({
            "id": name,
            "label": ROBOTS[name]["label"],
            "height_m": float(ROBOTS[name]["height_m"]),
            # The FIGURE's node, which is what turns. The dimension line beside it is a ruler and
            # stays put; a rule that swung round with the robot would stop measuring anything.
            # Carried even on a checkout with no cached GLB for this robot, where nothing was
            # drawn under it: the browser looks the name up and skips what is not there, and the
            # tile is still a real choice because the rule beside it still is.
            "node": sanitize_three_name(robot_node(name)),
        })

    payload = {
        "title": str(title),
        "text": str(text),
        "tiles": tiles,
        "chosen": str(chosen),
        "spin_rad_per_s": _ROBOT_SPIN_RAD_PER_S,
        # The dimension lines' text, drawn as canvas sprites: a trimesh scene has no text
        # primitive. Same mechanism, same helper, as the preview's and the edit page's.
        "scale_labels": labels,
    }
    panel = (
        _ROBOT_BODY.replace("__THEME__", THEME_CSS)
        .replace("__SPRITE_LABEL__", SPRITE_LABEL_JS)
        .replace("__ROBOT_PAYLOAD__", script_json(payload))
    )
    # An install with no cached robot GLB at all still has three dimension lines to draw, so the
    # scene is never empty here and needs no equivalent of render_edit_page's bare-panel fallback.
    return scene_page(scene, panel)


_ROBOT_BODY = """
<style>
__THEME__
#robotpick{position:fixed;top:0;right:0;width:clamp(300px, 27vw, 380px);max-height:100vh;
  overflow-y:auto;background:var(--panel);color:var(--fg);box-sizing:border-box;padding:16px 18px;
  z-index:9999;font:12.5px/1.55 var(--ui);border-left:1px solid var(--edge)}
#robotpick h2{font-size:14px;margin:0 0 3px;font-weight:620;letter-spacing:-.01em}
#robotpick h3{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin:18px 0 7px;font-weight:600;font-family:var(--mono)}
#robotpick .sub{color:var(--fg-2);font-size:11.5px}
#robotpick .note{color:var(--muted);font-size:11px;line-height:1.5;margin-top:7px}
#robotpick .tiles{display:flex;flex-direction:column;gap:3px;margin-top:7px}
#robotpick button{font:inherit;font-size:12px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:6px 10px;cursor:pointer;
  text-align:left;width:100%}
#robotpick button:hover:enabled{background:var(--edge);color:var(--accent)}
#robotpick button:disabled{opacity:.4;cursor:default}
#robotpick button:focus-visible{outline:1px solid var(--accent);outline-offset:1px}
/* Brass on the robot the session already carries, so Cancel is visibly "keep this one". */
#robotpick button.on{border-color:var(--accent);color:var(--accent);background:var(--accent-wash)}
#robotpick .h{font-family:var(--mono);font-size:11px;color:var(--muted);
  font-variant-numeric:tabular-nums}
#robotpick .row{display:flex;gap:7px;margin:14px 0 0;align-items:center}
#robotpick .row button{width:auto}
#robotpick .err{color:var(--err);font-size:11.5px;margin-top:9px;min-height:15px}
</style>
<div id="robotpick">
  <h2 id="robot-title"></h2>
  <div class="sub" id="robot-text"></div>

  <h3>Robots</h3>
  <div class="tiles" id="robot-tiles"></div>
  <div class="note">Each one is drawn at its measured standing height, beside a dimension line at
    that height. Every later comparison in this run — the furniture edit step, the kitchen preview —
    is drawn against the one you pick here.</div>

  <div class="row">
    <button id="robot-cancel">Cancel</button>
  </div>
  <div class="note" id="robot-keep"></div>
  <div class="err" id="robot-err"></div>
</div>
<script>
(function () {
  var DATA = __ROBOT_PAYLOAD__;
  var err = document.getElementById('robot-err');
  var answering = false;

  document.getElementById('robot-title').textContent = DATA.title;
  document.getElementById('robot-text').textContent = DATA.text;

  function answer(id) {
    if (answering) { return; }
    answering = true;
    document.querySelectorAll('#robotpick button').forEach(function (b) { b.disabled = true; });
    window.simvlaPost('/answer', {id: id}).then(function (r) {
      // On 200 the director moves on and the step poller reloads this tab.
      if (r.status !== 200) {
        err.textContent = (r.body && r.body.reason) || ('rejected (' + r.status + ')');
        answering = false;
        document.querySelectorAll('#robotpick button').forEach(function (b) {
          b.disabled = false;
        });
      }
    });
  }

  // The list COMMITS, exactly as kitchen_gallery's tile list does ("a named row you aimed at, so
  // it commits directly"). There is no select-then-use here because there is no way to select by
  // clicking the view -- see render_robot_page's docstring.
  var list = document.getElementById('robot-tiles');
  DATA.tiles.forEach(function (t) {
    var b = document.createElement('button');
    b.appendChild(document.createTextNode(t.label + '  '));
    var h = document.createElement('span');
    h.className = 'h';
    h.textContent = t.height_m.toFixed(2) + ' m';
    b.appendChild(h);
    if (t.id === DATA.chosen) { b.className = 'on'; }
    b.addEventListener('click', function () { answer(t.id); });
    list.appendChild(b);
  });

  var already = DATA.tiles.filter(function (t) { return t.id === DATA.chosen; })[0];
  document.getElementById('robot-keep').textContent = DATA.chosen
    ? 'Cancel keeps ' + ((already && already.label) || DATA.chosen)
      + ', which is what this run starts with.'
    : 'Cancel leaves the run with no robot to size anything against.';
  document.getElementById('robot-cancel').addEventListener('click', function () { answer(''); });

__SPRITE_LABEL__

  function buildScaleLabels() {
    if (typeof scene === 'undefined' || !scene) { return; }
    var root = scene.getObjectByName('world') || scene;
    (DATA.scale_labels || []).forEach(function (item) {
      var sprite = simvlaLabelSprite(item.text, item.width);
      sprite.position.set(item.at[0], item.at[1], item.at[2]);
      root.add(sprite);
    });
  }

  // ---- the turntable ------------------------------------------------------------------------
  // A rotation about each robot's OWN up axis, applied to the node add_scale_reference put it
  // under. That node's origin is the robot's standing point (the reference GLB is centred in
  // plan by its builder), so this is a turn in place rather than an orbit, and the dimension line
  // beside it is a separate node that does not move.
  //
  // Its own requestAnimationFrame loop, NOT the viewer template's: trimesh's bundled `animate()`
  // only calls controls.update() and never renders -- render() is bound to the controls' change
  // event -- so a rotation driven from it would move nothing on screen.
  //
  // REDUCED MOTION IS RESPECTED: with the preference set nothing starts, the robots stand still,
  // and the page is otherwise identical. The pick does not depend on the movement.
  var reduced = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function turntable() {
    if (typeof scene === 'undefined' || !scene) { return false; }
    // 'world' is trimesh's own root node -- the honest test that the GLB has decoded, and the
    // only one available on a checkout with no cached robot assets, where there is nothing to
    // turn but there are still three dimension lines that need their labels.
    if (!scene.getObjectByName('world')) { return false; }
    var turning = [];
    DATA.tiles.forEach(function (t) {
      var node = t.node && scene.getObjectByName(t.node);
      if (node) { turning.push(node); }
    });
    buildScaleLabels();
    if (reduced || !turning.length) {
      if (typeof render === 'function') { render(); }
      return true;
    }
    var started = null;
    function frame(now) {
      if (started === null) { started = now; }
      var angle = ((now - started) / 1000) * DATA.spin_rad_per_s;
      turning.forEach(function (node) {
        node.rotation.z = angle;
        node.updateMatrixWorld(true);
      });
      if (typeof render === 'function') { render(); }
      window.requestAnimationFrame(frame);
    }
    window.requestAnimationFrame(frame);
    return true;
  }

  // The template decodes the GLB asynchronously, so the nodes do not exist yet -- the same poll
  // the edit panel and the preview use.
  var tries = 0;
  var timer = setInterval(function () {
    if (++tries > 400 || turntable()) {      // ~20s
      clearInterval(timer);
    }
  }, 50);
})();
</script>
"""


#: What a "re-render this page for that robot" answer starts with, on the preview and on the edit
#: page alike. ONE prefix for the one mechanism: kitchen_scene_generator's preview loop and its
#: _edit_furniture both recognise a pick by it, and the preview's own option list is built from it
#: (kitchen_scene_generator._ROBOT_PREFIX). Written here because this is where the edit page's
#: whitelist is enumerated.
ROBOT_ANSWER_PREFIX = "robot:"


def edit_answer_ids(scales, materials) -> list[str]:
    """Every answer the edit page can produce, plus its Cancel. The step's option whitelist.

    ENUMERATED, which is why the page's slider is a slider OVER THESE STEPS and not a continuous
    one. The server accepts an answer only if its id is in the list the step published
    (WizardServer._handle_answer), so a control whose value cannot be enumerated would have to be
    waved through unchecked and validated afterwards by the director -- which is the "the POST
    returned 200 and nothing happened" failure this codebase keeps out of every other step (see
    render_choice_page's docstring for the same rule applied to Cancel). The slider's own value is
    an index into `scales`, so no amount of dragging can produce a value that is not in this list.

    "<scale to 2dp>:<group>", and "" for Cancel. The scale leads because it can never contain a
    colon while a group name is only promised not to; the director splits on the FIRST one.

    AND THE SAME PAIR AGAIN BEHIND EVERY ROBOT -- "robot:<name>:<scale>:<group>", plus "robot::…"
    for No robot. That is the whole reason a robot pick on this page cannot lose the author's
    work: the scale and the material are CLIENT-side (a slider and a menu, neither posted until
    Accept) while the robot is chosen SERVER-side, so a pick that carried only the robot would
    re-render the page at the values the step opened on and silently undo the drag. The preview
    has no such pair to lose -- its client state is dragged placements, which ride their own
    /placements POST -- which is why its pick is the bare prefix and this one is not.

    The cross product is the price of the enumeration rule above, and it is only a list of strings
    held by the server: /step reports {"id", "kind"} and never the options, so none of it is
    carried to the browser.
    """
    picks = [f"{float(s):.2f}:{g}" for s in scales for g in materials]
    return picks + [
        f"{ROBOT_ANSWER_PREFIX}{robot}:{pick}"
        for robot in ("", *ROBOTS)
        for pick in picks
    ] + [""]


#: The scene-graph node the piece of furniture is drawn under on the edit page. Chosen to survive
#: THREE.PropertyBinding.sanitizeNodeName unchanged (no '[', ']', '.', ':' or '/'), so the name the
#: payload carries is the name the browser's getObjectByName finds. One node, added
#: with an IDENTITY transform, so the browser's `piece.scale.set(s, s, 1)` is exactly a plan-view
#: resize about the piece's own centre -- no drift sideways, no lift off the floor, and no shear
#: from a rotation baked into a parent.
_EDIT_PIECE_NODE = "piece"

#: How far to the left of the piece the scale reference stands, in metres, measured from the
#: piece's widest possible plan half-width (its extent at the LARGEST offered scale). Constant
#: rather than proportional so the gap does not open up as the slider is dragged; measured off the
#: largest scale so the piece cannot grow into the reference at the top of the range.
#:
#: The gap is to the reference's RIGHT EDGE, not to the robot's midline. The reference is a row --
#: the robot, then its dimension line -- which reaches 0.41-0.47 m to the robot's right depending on
#: which robot it is, so anchoring the robot here would stand the dimension line inside the table.
#: See kitchen_preview.scale_reference_extents, which is measured off the meshes themselves.
_EDIT_FIGURE_GAP_M = 0.55

#: The scene-graph node the CHOSEN TABLE is drawn under when a chair is being edited against it.
#:
#: kitchen_preview.SCALE_NODE_PREFIX, deliberately: this table is a ruler, exactly as the robot
#: beside it is. Every containment check in this project is written as "no node whose name starts
#: with that prefix reaches `objects` / the placement gate / the USD export / _label_supports", so
#: naming it this way enrols it in all four without a second mechanism to keep in step. Nothing
#: here can reach a committed kitchen in the first place -- this page builds its own scene from
#: nothing and only ever renders it -- but the page that drew the robot into a caller's scene was
#: one refactor away from doing exactly that, and the prefix is what makes the next one visible.
_EDIT_TABLE_NODE = SCALE_NODE_PREFIX + "table"


def render_edit_page(*, title: str, text: str, scales, scale: float, materials,
                     material: str, piece=None, swatches=None, measured_floor=None,
                     floor_note: str = "", note: str = "", robot: str = DEFAULT_ROBOT,
                     table=None, table_prompt: str = "") -> str:
    """The edit step: a size scale and a material group for ONE piece of furniture, in 3D.

    Modelled on the galleries: one trimesh.Scene through trimesh's viewer template with a panel
    injected (kitchen_preview.scene_page), its Cancel answers "" and leaves the state untouched, it
    carries the same phase. It is a 3D page because the thing being chosen is a SIZE -- a row of
    buttons reading "1.30×" says nothing about whether the chair is then too big for the room, and
    the author has to look at it.

    piece    -- the furniture as a trimesh.Scene (or Trimesh), at its EXPORTED size. Flattened to
                one mesh, centred in plan, stood on z = 0 and added under _EDIT_PIECE_NODE, which
                is the node the browser scales. None renders the page with the reference alone, which
                is what a caller with no geometry to show gets rather than an exception.
    swatches -- {material group -> '#rrggbb'}, kitchen_build.FURNITURE_MATERIAL_SWATCHES. An
                APPROXIMATION and the page says so: MDL materials exist only on the USD stage.
    measured_floor -- the scale below which this piece stops being something the project has
                measured as acceptable (kitchen_build.chair_width_floor_scale), or None. Marked on
                the slider; never enforced.
    floor_note -- why that mark is there, in the author's words.
    note     -- anything else true of THIS piece that the picture does not say.
    robot    -- which of kitchen_preview.ROBOTS stands beside the piece, or "" for none. The page
                always offers the selector -- this page IS the comparison, so it opens on
                DEFAULT_ROBOT rather than on none the way the preview does -- and picking from it
                answers `robot:<name>:<scale>:<group>`, which the director re-renders this page
                from. THE SAME SERVER-SIDE MECHANISM THE PREVIEW'S SELECTOR USES, for the same
                reason (see kitchen_preview.render_page's `scale_reference`): each robot is
                5-13 MB of GLB, so a client-side switch would have to embed all three. No robot is
                offered because it is the state that costs nothing -- an author who only wants to
                see the piece pays none of those megabytes.
    table    -- THE TABLE THIS PIECE GOES WITH, or None. A chair's size only means something
                against one, so the chair's edit page draws the table the author chose -- at the
                scale and in the material they set, because a default-sized table would answer a
                question nobody asked. A dict, not a scene, because the placement needs numbers
                with it:
                  scene      -- the table as a trimesh.Scene, at the size it will be BUILT at
                                (kitchen_build.TableVariant.build takes the scale, so a procedural
                                table is rebuilt rather than stretched -- see _edit_scene_for_table)
                  label      -- what to call it in the panel
                  detail     -- its data line ("Long dining", "uid e7cc55"), or ""
                  material   -- which swatch to tint it, keyed into `swatches`
                  offsets_m  -- one y offset per entry of `scales`, in the SAME order: how far the
                                table's plan centre stands from a chair seated at its near edge,
                                for a chair at that scale. Computed by the director from
                                kitchen_build's own seating ring, and enumerated for the same
                                reason the slider's own values are -- the chair resizes in the
                                browser, the ring reflows with it, and the browser must not have
                                to re-derive a placement rule to keep up.
                The table is a TOGGLE, on by default, and drawn CLIENT-side: unlike a robot it is
                a few hundred KB, so it can be embedded once and hidden, and hiding it costs no
                round trip.
    table_prompt -- what to say INSTEAD, when this piece has a table section but no table to put
                in it ("pick one on the setup form"). Empty means the piece has no such section at
                all and the whole block is hidden -- which is the TABLE's own edit page, where
                "the table it goes with" is not a question anyone is asking.

    `scale` and `material` are what the piece currently has, so re-opening this on an
    already-edited piece shows the edit rather than the defaults -- without that, Edit… would be
    indistinguishable from "reset".

    ONE SLIDER FOR ALL THREE AXES -- there is no separate height control, because the scale is
    uniform: see kitchen_build.FURNITURE_SCALE_IS_UNIFORM. The page's own hint text says what that
    costs (the 0.74 m work surface the chairs are drawn against stops being 0.74 m), rather than
    leaving it in a comment here.

    THE SCENE IS BUILT HERE, from a COPY of `piece`. The table gallery caches its scenes for the
    whole run, so adding the reference to the scene it was handed would put a robot in the table
    gallery's tiles -- and one more of it on every visit. trimesh is imported inside the function:
    this module must stay importable before Omniverse boots (see the module docstring), and the
    import is only paid on a page that actually draws.
    """
    import trimesh                                            # ^ see the docstring

    from kitchen_preview import (
        SPRITE_LABEL_JS,
        add_scale_reference,
        scale_reference_extents,
        scene_page,
    )

    if robot and robot not in ROBOTS:
        raise ValueError(f"{robot!r} is not one of {sorted(ROBOTS)}")
    scales = [float(s) for s in scales]
    swatches = dict(swatches or {})
    scene = trimesh.Scene()
    size = []
    if piece is not None:
        mesh = piece.to_geometry() if isinstance(piece, trimesh.Scene) else piece.copy()
        low, high = mesh.bounds
        # Centred in plan and standing on the floor: the browser scales this node about its own
        # origin, and an off-centre piece would slide across the view as the slider moved.
        mesh.apply_translation([
            -(float(low[0]) + float(high[0])) / 2.0,
            -(float(low[1]) + float(high[1])) / 2.0,
            -float(low[2]),
        ])
        size = [float(v) for v in mesh.extents]
        # A material of its own, in the currently chosen group's colour. Without ANY material a
        # mesh exports to glTF carrying none and three.js falls back to a near-black default --
        # the trap kitchen_gallery._tile_material documents -- so this is what makes the piece
        # visible at all, as well as what makes the first paint already show the right tint.
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="piece", baseColorFactor=list(_hex_to_rgba(swatches.get(material))),
                metallicFactor=0.0, roughnessFactor=0.75,
            )
        )
        scene.add_geometry(mesh, node_name=_EDIT_PIECE_NODE, geom_name=_EDIT_PIECE_NODE)

    # THE TABLE THE PIECE GOES WITH, seated as the build would seat it: the chair stays where it
    # is -- at the origin, in the orientation it was exported in, which is what the slider resizes
    # about -- and the table stands off it in +y by the ring's own answer for a chair this wide
    # (`offsets_m`). The camera looks from -y, so the reader sees the chair at the table's near
    # edge with the work surface behind it, which is the pairing the panel's note is about.
    #
    # THE SEAT'S YAW IS DELIBERATELY NOT APPLIED. kitchen_build._seats_around_table turns each
    # seat to face the table on the assumption that a chair's own +x is its front, and
    # face_chair_toward then CORRECTS that for any chair whose facing was measured confidently --
    # so the true placed orientation is not knowable from the uid alone. Drawing the chair
    # unrotated keeps this view identical to the chair-alone one, and the panel says the build
    # turns it to face the table.
    table_view = None
    if table and table.get("scene") is not None:
        offsets = [float(v) for v in (table.get("offsets_m") or [])]
        # The offset for the scale this page opened at, so a page whose JS never ran still stands
        # the table in the right place. The browser re-reads the list on every slider step.
        at = 0
        for index, value in enumerate(scales):
            if abs(value - float(scale)) < abs(scales[at] - float(scale)):
                at = index
        source = table["scene"]
        top = source.to_geometry() if isinstance(source, trimesh.Scene) else source.copy()
        low, high = top.bounds
        top.apply_translation([
            -(float(low[0]) + float(high[0])) / 2.0,
            -(float(low[1]) + float(high[1])) / 2.0,
            -float(low[2]),
        ])
        top.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="table", baseColorFactor=list(_hex_to_rgba(swatches.get(table.get("material")))),
                metallicFactor=0.0, roughnessFactor=0.75,
            )
        )
        scene.add_geometry(
            top, node_name=_EDIT_TABLE_NODE, geom_name=_EDIT_TABLE_NODE,
            transform=trimesh.transformations.translation_matrix(
                [0.0, offsets[at] if at < len(offsets) else 0.0, 0.0]
            ),
        )
        table_view = {
            "node": _EDIT_TABLE_NODE,
            "label": str(table.get("label") or "the table"),
            "detail": str(table.get("detail") or ""),
            "size": [float(v) for v in top.extents],
            "offsets_m": offsets,
        }

    widest = (size[0] if size else 0.0) * (max(scales) if scales else 1.0)
    # ONE ROBOT PER PAGE, the one that was chosen -- nothing is drawn and nothing is embedded for
    # No robot, which is what makes that option 8-18 MB lighter rather than merely tidier.
    #
    # The reference is placed by its own plan CENTRE, so shift by half its span to land its right
    # edge exactly _EDIT_FIGURE_GAP_M clear of the piece at the biggest scale it can be dragged to.
    # Measured off the robot being drawn, not off DEFAULT_ROBOT: the three assemblies are
    # 0.58-0.71 m across, and anchoring them all on one of those spans would stand a wider one's
    # dimension line inside the table.
    scale_labels = []
    if robot:
        x_lo, x_hi, _y_lo, _y_hi = scale_reference_extents(robot)
        right_edge = -(widest / 2.0 + _EDIT_FIGURE_GAP_M)
        _nodes, scale_labels = add_scale_reference(
            scene, at=(right_edge - (x_hi - x_lo) / 2.0, 0.0), robot=robot
        )

    payload = {
        "title": str(title),
        "text": str(text),
        "note": str(note),
        "scales": scales,
        "scale": float(scale),
        "materials": list(materials),
        "material": material,
        "swatches": swatches,
        "measured_floor": None if measured_floor is None else float(measured_floor),
        "floor_note": str(floor_note),
        # The three.js name of the node the slider scales, and the piece's real plan size at 1.00x
        # so the panel can print metres rather than only a multiplier.
        "node": _EDIT_PIECE_NODE,
        "size": size,
        # The selector: which robot is standing there now ("" for none), and every one that can be
        # picked with the height its dimension line is drawn at -- so the panel quotes the geometry
        # beside it rather than restating it. The same two payload keys, carrying the same shape,
        # as kitchen_preview.render_page's; unlike there this list is never empty, because this
        # page always offers the choice.
        "robot": robot,
        "robot_choices": [{"id": name, "label": spec["label"], "height_m": spec["height_m"]}
                          for name, spec in ROBOTS.items()],
        # The text beside the dimension lines, which cannot be geometry: a trimesh scene has no
        # text primitive. Drawn in the browser as canvas sprites at these scene-frame positions.
        "scale_labels": scale_labels,
        # The table drawn beside the piece, or null -- which the panel reads as "no table chosen
        # yet" and says so on a disabled toggle, rather than leaving a control that does nothing.
        # It carries its own node name, so the toggle and the slider's re-seating reach exactly
        # the geometry this render added and nothing else.
        "table": table_view,
        # What to say when there is a table section and no table for it. "" hides the section --
        # see the docstring.
        "table_prompt": str(table_prompt),
    }
    panel = (
        _EDIT_BODY.replace("__THEME__", THEME_CSS)
        .replace("__SPRITE_LABEL__", SPRITE_LABEL_JS)
        .replace("__EDIT_PAYLOAD__", script_json(payload))
    )
    # NOTHING TO DRAW is now reachable and must not be an exception: trimesh refuses to export an
    # empty scene ("Can't export empty scenes!"), and this page is empty when a piece whose
    # geometry could not be loaded (piece=None, a supported state -- see the docstring) meets No
    # robot. The panel is the step; it goes into the ordinary wizard shell instead of the viewer
    # one, and every control on it stays live because its JS already guards each reach for the 3D
    # globals (`typeof scene === 'undefined'`). Picking a robot draws something again.
    if not scene.geometry:
        return _page(panel)
    return scene_page(scene, panel)


def _hex_to_rgba(colour, default=(198, 196, 190)):
    """'#rrggbb' as the 0-255 RGBA list trimesh's PBRMaterial wants. `default` for anything else.

    A group with no swatch (an older kitchen_build, a caller that passed none) gets the galleries'
    neutral tile grey rather than an exception: the page's job is to show the SHAPE, and a missing
    colour must not cost the whole preview.
    """
    text = str(colour or "").lstrip("#")
    if len(text) != 6:
        return list(default) + [255]
    try:
        return [int(text[i:i + 2], 16) for i in (0, 2, 4)] + [255]
    except ValueError:
        return list(default) + [255]


def render_progress_page(*, title: str, notice: str = "") -> str:
    return _page(
        _PROGRESS_BODY.replace("__TITLE__", _escape(title)).replace("__NOTICE__", _escape(notice))
    )


def render_done_page(*, title: str, text: str) -> str:
    return _page(
        _DONE_BODY.replace("__TITLE__", _escape(title)).replace("__TEXT__", _escape(text))
    )


def goal_generation_blocked(kitchen_num, kitchen_dir: str):
    """Why goals cannot be generated for this kitchen, or None when they can.

    Checked BEFORE offering, never after: asking "generate goals now?" and then refusing would make
    the author commit to a decision this code already knew it could not honour.

    Returns `(reason, tag)`, never a bare string: the two block reasons call for different advice. A
    missing kitchen number still leaves `task_emit` itself usable (against a different kitchen), but
    missing rotation USDs mean `task_emit` would fail for the exact reason just given, so
    recommending it would be misleading.
    """
    if kitchen_num is None:
        return "this kitchen has no number to emit for.", "no_kitchen_num"
    if not emit_job.kitchen_is_on_disk(kitchen_dir, kitchen_num):
        return (
            f"kitchen {kitchen_num:02d} has no rotation USDs on disk "
            f"(kitchen_{kitchen_num:02d}_00.usd … _11.usd), which task_emit loads "
            f"one per rotation."
        ), "missing_rotations"
    return None


class WizardServer(ComposerServer):
    """Serves whichever step the director has published, and carries the browser's answer back.

    Everything the handler threads read is initialised BEFORE super().__init__(), which starts the
    serving thread: a request that arrives in that window would otherwise hit a half-built object.
    """

    def __init__(self, port: int | None = None):
        self._step = {"id": 0, "kind": "boot", "options": []}
        self._answer = None
        self._answered = threading.Event()
        self._progress = {"label": "", "done": 0, "total": 0}
        #: Which of PHASES the run has reached; None until the director says.
        self._phase = None
        self._mesh_files: list[str] = []
        self._save_ctx = {"kitchen_num": None, "kitchen_dir": ""}
        # The boot page carries the step script too. The generator starts this server BEFORE
        # Omniverse (~1 min) precisely so the URL is on screen while it loads, which means a tab is
        # very likely opened on this page: without the poller it would sit here until someone
        # refreshed by hand, instead of reloading itself onto the first real step.
        #
        # It goes through _page() like every other step, which it did not used to: a bare
        # <html><body> carries no theme, so the page a run OPENS on -- and sits on for the whole
        # minute Omniverse takes to boot, making it the longest-lived page in the run -- was an
        # unstyled white flash before every other surface came up dark.
        super().__init__(
            inject_step_script(
                _page(
                    '<div class="main solo"><h1>Starting…</h1>'
                    '<div class="sub">Booting Omniverse. This page moves on by itself.</div>'
                    "</div>"
                ),
                self._step["id"],
            ),
            port=port,
        )

    # ---- state the pages read ----

    @property
    def step(self) -> dict:
        """{"id", "kind"} — what /step reports and what an answer is checked against."""
        with self._lock:
            return {"id": self._step["id"], "kind": self._step["kind"]}

    @property
    def progress(self) -> dict:
        with self._lock:
            return dict(self._progress)

    def set_phase(self, phase) -> None:
        """Move the run to a phase without publishing a page; the next page picks it up."""
        with self._lock:
            self._phase = phase

    def set_progress(self, label: str, done: int, total: int) -> None:
        with self._lock:
            self._progress = {"label": label, "done": int(done), "total": int(total)}

    def set_setup_context(self, *, mesh_files) -> None:
        """The BODex meshes /setup validates an object row's mesh against.

        Set by the director once the dataset is loaded — after the server is already up, because the
        server starts before Omniverse boots so the URL is on screen while it loads. The kitchen
        directory isn't carried here: form_payload(kitchen_dir) takes it as a plain argument, and
        nothing in this server needs it once the page has been rendered.
        """
        with self._lock:
            self._mesh_files = list(mesh_files)

    def set_save_context(self, *, kitchen_num, kitchen_dir: str) -> None:
        """Which kitchen /save_task's goal-generation offer is about."""
        with self._lock:
            self._save_ctx = {"kitchen_num": kitchen_num, "kitchen_dir": kitchen_dir}

    def set_grasp_thumbs(self, thumbs: dict) -> None:
        """The {object key -> thumbnails folder} map /grasps/* serves.

        ComposerServer takes these at construction, because it was built once per composer session.
        This server outlives every session, so the composer step sets them as it opens.
        """
        with self._lock:
            self._grasp_thumbs = dict(thumbs or {})

    # ---- the director's side ----

    def publish(self, kind: str, html: str, *, options=(), phase=None) -> int:
        """Make `html` the current page. Returns the new step id.

        `options` are the answer ids this step accepts; anything else comes back 400.
        `phase` is which of PHASES the run has reached. It STICKS: pass it when the run moves on,
        and every page published after that inherits it. Otherwise the strip would vanish on every
        question and progress page — the pages where "what happens next?" is asked most.
        """
        with self._lock:
            if phase is not None:
                self._phase = phase
            step_id = self._step["id"] + 1
            self._step = {"id": step_id, "kind": kind, "options": [str(o) for o in options]}
            self._page = inject_step_script(html, step_id, self._phase).encode("utf-8")
            self._answer = None
            if kind == "scene":
                # A fresh server per round used to clear these; one long-lived server must do it
                # explicitly, or a re-roll inherits the previous round's drags.
                self._placements = {}
                self._template = None
            self._answered.clear()
        return step_id

    def wait_answer(self) -> dict:
        """Block until the browser answers the current step.

        Waits in short slices rather than one indefinite wait so a KeyboardInterrupt on the main
        thread is delivered promptly instead of at the end of the run.

        UNDER THE REAL GENERATOR THAT NEVER HAPPENS, and this docstring used to claim it did.
        SimulationApp installs its own SIGINT handler at boot -- after
        AppLauncher({"headless": True}).app, signal.getsignal(signal.SIGINT) is
        `SimulationApp.__init__.<locals>.signal_handler`, not Python's default_int_handler -- so
        Ctrl-C tears the process down inside Omniverse and no KeyboardInterrupt is ever raised on
        this thread. Measured: SIGINT during this wait killed the server within 0.2s, published no
        closing page, and lost the generator's own "[Main] Interrupted." on the way out. See
        kitchen_scene_generator.__main__ for the matching note.

        The slicing is not dead code, though: this server is constructed with no Omniverse at all by
        every test in test_kitchen_wizard.py, and by anything else that drives it from a plain
        python. There the default handler is intact and the slices are what make Ctrl-C prompt.
        """
        while not self._answered.wait(0.2):
            pass
        with self._lock:
            return dict(self._answer or {})

    def ask(self, title: str, text: str, options) -> str:
        """Publish a question and block until one of `options` is clicked. Returns its id."""
        page = render_choice_page(title=title, text=text, options=options)
        self.publish("choice", page, options=[str(o["id"]) for o in options])
        return self.wait_answer().get("id", "")

    def notice(self, title: str, text: str, label: str = "OK") -> None:
        """A message with one button — what messagebox.showinfo/showerror used to be."""
        self.ask(title, text, [{"id": "ok", "label": label}])

    def show_progress(self, title: str, notice: str = "") -> int:
        """Publish the progress page. Nothing to wait for: the page polls /progress."""
        return self.publish("progress", render_progress_page(title=title, notice=notice))

    def finish(self, title: str, text: str) -> None:
        """Publish the closing page and give the poller enough time to land on it."""
        self.publish("done", render_done_page(title=title, text=text))
        time.sleep(_FINISH_MARGIN_S)

    def _record_answer(self, payload: dict) -> None:
        with self._lock:
            self._answer = dict(payload)
            self._answered.set()

    # ---- HTTP ----

    def _make_handler(self):
        server = self
        base = ComposerServer._make_handler(self)

        class Handler(base):
            def _sealed(self, path: str) -> bool:
                """True (and a 503 already written) if this path must not run yet.

                Default-DENY for the whole boot step: only the four paths the boot page itself
                needs are served, and everything else -- including every endpoint a future
                ComposerServer might add -- is refused until the director publishes its first real
                step, which happens strictly after AppLauncher has run.

                This is the enforcement half of the pre-boot invariant in this module's docstring.
                Making the kitchen_build imports lazy only moved the hazard from import time to
                call time: /scene_defaults reaches SUPPORT_SURFACES -> kitchen_build ->
                scene_synthesizer, and /template reaches validate_scene, both with no step guard of
                their own. A composer tab left open on 8777 from a previous run refetches
                /scene_defaults the moment this server answers -- roughly sixteen seconds before
                Isaac finishes booting -- and half-initialises scene_synthesizer exactly as the
                top-level import used to, for a NameError twenty minutes later at the commit.

                A whitelist rather than a blacklist on purpose: the endpoints that are SAFE here are
                a short, closed list this module owns, while the endpoints that are dangerous are
                spread across two base classes and grow.
                """
                if server.step["kind"] != "boot":
                    return False
                if self.command == "GET" and path in ("/", "/index.html", "/ping", "/step"):
                    return False
                self._write_json(503, {
                    "ok": False,
                    "reason": "the generator is still booting Omniverse; this page moves on by "
                              "itself when the run starts",
                })
                return True

            def do_GET(self):
                path = urlparse(self.path).path
                if self._sealed(path):
                    return
                if path == "/step":
                    self._write_json(200, server.step)
                    return
                if path == "/progress":
                    self._write_json(200, server.progress)
                    return
                base.do_GET(self)

            def do_POST(self):
                path = urlparse(self.path).path
                if self._sealed(path):
                    return
                if path == "/setup":
                    payload = self._read_step_json()
                    if payload is not None:
                        self._handle_setup(payload)
                    return
                if path in ("/save_task", "/emit_goals"):
                    payload = self._read_step_json()
                    if payload is None:
                        return
                    if path == "/save_task":
                        self._handle_save_task()
                    else:
                        self._handle_emit_goals(payload)
                    return
                if path == "/answer":
                    payload = self._read_step_json()
                    if payload is not None:
                        self._handle_answer(payload)
                    return
                base.do_POST(self)

            def _read_step_json(self):
                """The POST body, or None when it was malformed or came from a stale page.

                Writes the error response itself, so a caller that gets None just returns.
                """
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, TypeError) as exc:
                    self._write_json(400, {"ok": False, "reason": f"malformed request: {exc}"})
                    return None
                if not isinstance(payload, dict):
                    self._write_json(400, {"ok": False, "reason": "expected a JSON object"})
                    return None
                if payload.get("step") != server.step["id"]:
                    # The guard the un-closeable Tk modal used to provide: a stale tab must not be
                    # able to answer a step the run has already left.
                    self._write_json(409, {
                        "ok": False,
                        "reason": "this page is out of date; it will reload",
                    })
                    return None
                return payload

            def _handle_setup(self, payload):
                with server._lock:
                    mesh_files = list(server._mesh_files)
                try:
                    parsed = validate_setup(payload, mesh_files)
                except SetupError as exc:
                    # The reason travels in the body so the form shows it inline and the run stays
                    # on this step — a rejected form must not advance anything.
                    self._write_json(400, {"ok": False, "reason": str(exc)})
                    return
                except Exception as exc:
                    # Invariant: a POST to this server always gets an answer. validate_setup should
                    # only ever raise SetupError; if some input shape we did not anticipate slips
                    # past it anyway, this is the backstop -- without it the exception propagates out
                    # of the handler thread, socketserver closes the connection with no response ever
                    # sent, and the page shows the user nothing at all (not even a rejection).
                    traceback.print_exc()
                    self._write_json(500, {"ok": False, "reason": f"internal error: {exc}"})
                    return
                server._record_answer(parsed)
                self._write_json(200, {"ok": True})

            def _handle_answer(self, payload):
                with server._lock:
                    allowed = list(server._step["options"])
                chosen = payload.get("id")
                if chosen not in allowed:
                    self._write_json(400, {
                        "ok": False,
                        "reason": f"{chosen!r} is not one of this step's options",
                    })
                    return
                server._record_answer(payload)
                self._write_json(200, {"ok": True})

            def _handle_save_task(self):
                template = server.template
                if template is None:
                    self._write_json(400, {"ok": False, "reason": (
                        "Compose the task in the browser and click Save there first — that "
                        "validates it and hands it back here."
                    )})
                    return
                try:
                    out = save_composed_template(template)

                    with server._lock:
                        ctx = dict(server._save_ctx)
                    blocked = goal_generation_blocked(ctx["kitchen_num"], ctx["kitchen_dir"])
                    if blocked:
                        reason, tag = blocked
                        if tag == "missing_rotations":
                            # task_emit would fail for this exact reason, so pointing at it here
                            # would be misleading -- the real next step is to create the rotations.
                            advice = (
                                'Generate the scene first: "Accept & Generate" on this kitchen '
                                "writes the rotation USDs, after which Save Task can offer to "
                                "generate goals."
                            )
                        else:
                            advice = (
                                "Emit it later with: ./isaaclab.sh -p scripts/simvla/task_emit.py "
                                f"--template {out} --kitchens <N> --out <dir>"
                            )
                        self._write_json(200, {"ok": True, "message": (
                            f"Task '{template.name}' written to {out}. "
                            f"Goals cannot be generated from here: {reason} {advice}"
                        )})
                        return

                    self._write_json(200, {
                        "ok": True,
                        "message": f"Task '{template.name}' written to {out}.",
                        "confirm": {
                            "text": (
                                f"Also generate goals for kitchen {ctx['kitchen_num']:02d} now? "
                                f"(submits a Slurm job)"
                            ),
                            "path": "/emit_goals",
                            "body": {"template": str(out)},
                        },
                    })
                except ValueError as exc:
                    # save_composed_template's one documented failure -- an unnamed task -- is the
                    # author's to fix (name it in the composer), so it is a 400 with the reason
                    # shown inline, same as _handle_setup's SetupError branch.
                    self._write_json(400, {"ok": False, "reason": str(exc)})
                except Exception as exc:
                    # Invariant carried over from _handle_setup: a POST to this server always gets
                    # an answer. save_composed_template also does real filesystem I/O (mkdir +
                    # write_text), which can raise OSError (permissions, a full disk) -- ValueError
                    # does not cover that, so this is the backstop for everything else; without it
                    # the exception propagates out of the handler thread, socketserver closes the
                    # connection with no response ever sent, and the page shows nothing at all.
                    traceback.print_exc()
                    self._write_json(500, {"ok": False, "reason": f"internal error: {exc}"})

            def _handle_emit_goals(self, payload):
                with server._lock:
                    ctx = dict(server._save_ctx)

                # Re-checked here, not trusted from /save_task's earlier check: /emit_goals is
                # directly POST-able, and the confirm round-trip is asynchronous -- the author can
                # click "Yes" any time after the offer -- so the save context this POST actually
                # arrives with can differ from what /save_task saw. Without this, a POST arriving
                # with set_save_context never called (kitchen_num still None) or with rotations
                # since deleted would build kitchens="None" and submit a real Slurm job that only
                # fails deep inside task_emit.py, which is exactly what goal_generation_blocked's
                # own "checked before offering, never after" contract is meant to prevent -- that
                # contract has to hold for the action too, not just the offer.
                blocked = goal_generation_blocked(ctx["kitchen_num"], ctx["kitchen_dir"])
                if blocked:
                    reason, _tag = blocked
                    self._write_json(400, {"ok": False, "reason": f"Cannot generate goals: {reason}"})
                    return

                # Not a security boundary -- this server only ever listens on loopback -- but a
                # wrong or missing template path here would otherwise only surface as a Slurm
                # failure minutes later. Reject it up front instead.
                template_path = payload.get("template")
                if (not isinstance(template_path, str) or not template_path
                        or not os.path.isfile(template_path)):
                    self._write_json(400, {"ok": False, "reason": (
                        f"No such template file: {template_path!r}. Save the task again before "
                        "generating goals."
                    )})
                    return

                try:
                    out_dir = emit_job.resolve_out_dir()
                    script = emit_job.build_emit_sbatch(
                        template_path=template_path,
                        kitchens=str(ctx["kitchen_num"]),
                        out_dir=out_dir,
                        repo_root=os.environ.get("SIMVLA_REPO_ROOT", ""),
                        **emit_job.env_defaults(),
                    )
                    job_id = emit_job.submit_emit_job(script)
                except FileNotFoundError:
                    self._write_json(400, {"ok": False, "reason": (
                        "Could not find `sbatch` — this is not a Slurm login node. Run it on a GPU "
                        "node instead: ./isaaclab.sh -p scripts/simvla/task_emit.py --template "
                        f"{template_path} --kitchens {ctx['kitchen_num']} "
                        f"--out {emit_job.resolve_out_dir()}"
                    )})
                    return
                except RuntimeError as exc:
                    self._write_json(400, {"ok": False, "reason": str(exc)})
                    return
                except Exception as exc:
                    # Same invariant as _handle_setup and _handle_save_task above:
                    # resolve_out_dir/env_defaults/build_emit_sbatch used to run with no try around
                    # them at all, so any exception there dropped the connection with no response
                    # ever sent. FileNotFoundError and RuntimeError above are submit_emit_job's two
                    # documented failures; this is the backstop for everything else.
                    traceback.print_exc()
                    self._write_json(500, {"ok": False, "reason": f"internal error: {exc}"})
                    return

                self._write_json(200, {"ok": True, "message": (
                    f"Slurm job {job_id} is generating goals for kitchen {ctx['kitchen_num']} "
                    f"(all 12 rotations) into {out_dir}. Watch it: squeue -j {job_id}; "
                    f"log: {out_dir}/compose_emit_{job_id}.log"
                )})

        return Handler
