#!/usr/bin/env python3
"""Measure every BODex-grasped mesh and write SimVLA's grasp manifest.

Grasp capability is a fact about a MESH, read from the files on disk — not a membership in a
hardcoded type list. This walks <BODEX_OBJ_DIR>/graspdata_final/sim_parallel, measures the contact
width of every grasp at the size the object is actually placed at, applies the escalation ladder,
and writes <BODEX_OBJ_DIR>/grasp_manifest.json.

What "contact width" means: a BODex grasp records the two points where the parallel jaws touch the
object (`contact_point`, shape (1, N, 1, 2, 3)). Their separation is what the gripper must span —
NOT the object's bounding box, which is why a 14 cm plate grasped by its rim is a one-gripper job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Gripper finger separation, fully open. 0.080, and it is a fact about the asset rather than a
# tuning knob: in Robots/anubis_simvla.usd the gripper1_joint / gripper1R_joint prismatic pair each
# carry limits [0.0, 0.04] on axis X with localPos0 == localPos1 == (0,0,0), so the two pads share
# an origin and the widest gap they can present is 2 x 0.04. (anubis_wheels.py:61-64 duly opens all
# four finger joints to 0.04.)
#
# It read 0.09 before, cited to simvla_video.py:1810 -- a line that no longer says anything about
# the gripper. The only surviving mention is the SIMVLA_DBG `fingsep` trace, whose "open~0.09" is
# the same stale guess. The 1 cm of imaginary aperture is not free: it passed a mug whose cup body
# is 0.097 across as graspable, and the jaws cannot close around that at all -- they only nudge it.
APERTURE_M = 0.080
MARGIN_M = 0.005        # safety band so an at-limit grasp is not called feasible
MIN_CANDIDATES = 30     # grasps that must fit before an object counts as single-gripper
SHRINK_FLOOR = 0.7      # never shrink an object below this fraction of its real size


def contact_widths(grasp: dict, scale: float) -> list[float]:
    """Distance between the two contact points of each grasp, at placed scale, in metres."""
    cp = np.asarray(grasp["contact_point"])
    if cp.ndim == 5:                       # (1, N, 1, 2, 3) — the on-disk shape
        cp = cp[0, :, 0, :, :]
    elif cp.ndim != 3:                     # neither on-disk nor the already-squeezed (N, 2, 3)
        raise ValueError(f"unexpected contact_point shape {cp.shape}")
    return [float(w) for w in np.linalg.norm(cp[:, 0, :] - cp[:, 1, :], axis=1) * scale]


def classify(widths_all, widths_selected, *, limit_m: float, min_candidates: int,
             shrink_floor: float) -> dict:
    """Apply the ladder to one mesh's measured widths.

    Capability is judged on the WHOLE grasp pool. An earlier revision judged on the selection,
    reasoning that plan_arm_grasp only draws from selected_indices.json -- true, but it made a
    hand-curated selection of the five best grasps indistinguishable from an object the gripper
    cannot close on, and duly called three bowls and two mugs bimanual. The selection is reported
    instead: `thin_selection` says the planner will be drawing from a narrow pool, which is fixed by
    re-picking (import_bodex_grasps.select_grasps_by_quality), not by reaching for a second gripper.
    """
    fitting_all = sum(1 for w in widths_all if w <= limit_m)
    fitting_selected = (None if widths_selected is None
                        else sum(1 for w in widths_selected if w <= limit_m))
    out = {
        "grasps": len(widths_all),
        "fitting": fitting_all,
        "selected": None if widths_selected is None else len(widths_selected),
        "fitting_selected": fitting_selected,
        "thin_selection": (None if fitting_selected is None
                           else fitting_selected < min_candidates),
        "median_width_m": (round(float(np.median(widths_all)), 5) if widths_all else None),
        "scale_override": None,
    }
    if not widths_all:
        out["strategy"] = "none"
        out["reason"] = "no grasps"
        return out

    if fitting_all >= min_candidates:
        out["strategy"] = "single"
        return out

    # Tier 2: the scale that brings the min_candidates-th narrowest grasp inside the span.
    if len(widths_all) < min_candidates:
        out["strategy"] = "bimanual"
        out["reason"] = f"only {len(widths_all)} grasps; no scale reaches {min_candidates}"
        return out
    nth = sorted(widths_all)[min_candidates - 1]
    scale = limit_m / nth
    if scale >= shrink_floor:
        out["strategy"] = "single"
        out["scale_override"] = scale
        return out
    out["strategy"] = "bimanual"
    out["reason"] = f"would need scale {scale:.2f}, below the {shrink_floor} floor"
    return out


def default_bodex_obj_dir() -> str:
    """$BODEX_OBJ_DIR if set, else grasp_manifest's own default.

    The one place this value is computed: import_bodex_grasps.py's --bodex-obj-dir default and
    grasp_manifest._root() both resolve to the SAME value via this function / DEFAULT_BODEX_OBJ_DIR,
    so a machine with BODEX_OBJ_DIR set can't have the importer writing grasps into one tree while
    every reader (this tool included) looks at another.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    from grasp_manifest import DEFAULT_BODEX_OBJ_DIR
    return os.environ.get("BODEX_OBJ_DIR", DEFAULT_BODEX_OBJ_DIR)


def _categories():
    """type -> mesh-directory prefixes, from scene_spec."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    from scene_spec import CATEGORIES
    return CATEGORIES


def type_of(obj_name: str):
    """The declared type an object directory belongs to, or None.

    Matched on the CATEGORY prefix, the same rule kitchen_build.matching_meshes uses to decide what
    a row can spawn. They were two different rules before -- a longest-substring match here and a
    plain substring match there -- and they disagreed: `can` could spawn eight `sem_Candle_*` meshes
    that this function filed under `candle`, so the manifest described a different object than the
    one the scene placed.
    """
    for t, prefixes in _categories().items():
        if obj_name.startswith(prefixes):
            return t
    return None


def placed_scale(obj_type) -> float:
    """placed/mesh size ratio — the same factor plan_arm_grasp applies to contact offsets.

    1.0 for graspable types, whose REAL_HEIGHT scaling is gated to CLUTTER.
    """
    if obj_type is None:
        return 1.0
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    from object_dims import OBJECT_BASE_DIMS, _up_axis_index, real_placement_dims
    base = OBJECT_BASE_DIMS.get(obj_type)
    if not base:
        return 1.0
    up = _up_axis_index(base)
    return float(real_placement_dims(obj_type, base)[up] / base[up]) if base[up] else 1.0


def _selected_indices(floating: Path):
    """The (grasp index, hand) pairs the composer (or import_bodex_grasps) chose, or None.

    The HAND is kept, not discarded. Left and right are separate syntheses with their own contact
    points, so grasp 7 of the left hand and grasp 7 of the right are different grasps of different
    widths -- and import_bodex_grasps._split_selection deliberately gives each hand a DISJOINT set
    of indices. Dropping the hand and applying every index to both would measure grasps nobody
    selected, and would double the count while doing it.
    """
    f = floating / "scale010_grasp_segments_thumbnails" / "selected_indices.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text()).get("selected_indices", [])
    except json.JSONDecodeError:
        # Falls back to "no selection" either way, but this file EXISTS and is unreadable --
        # distinct from never having been written, and worth a name in the build log (these files
        # get hand-edited weeks after synthesis, so a corrupt one is a live risk, not a hypothetical).
        print(f"warning: {f} exists but is not valid JSON; treating as no selection",
              file=sys.stderr)
        return None
    return [(int(i), str(hand)) for i, hand in (tuple(x) for x in raw)] or None


def measure_object(floating: Path, obj_type) -> dict:
    """Measure one object's grasps and classify it."""
    scale = placed_scale(obj_type)
    widths_all, per_hand = [], {}
    for hand in ("left", "right"):
        npy = floating / f"scale010_grasp_{hand}.npy"
        if not npy.exists():
            continue
        grasp = np.load(npy, allow_pickle=True).item()
        w = contact_widths(grasp, scale)
        per_hand[hand] = w
        widths_all += w
    if not widths_all:
        return {"type": obj_type, "placed_scale": round(scale, 4), "strategy": "none",
                "reason": "no grasp files", "grasps": 0, "fitting": 0,
                "selected": None, "fitting_selected": None, "thin_selection": None,
                "median_width_m": None, "scale_override": None}

    pairs = _selected_indices(floating)
    widths_selected = None
    if pairs is not None:
        widths_selected = [per_hand[hand][i] for i, hand in pairs
                           if hand in per_hand and i < len(per_hand[hand])]

    out = classify(widths_all, widths_selected, limit_m=APERTURE_M - MARGIN_M,
                   min_candidates=MIN_CANDIDATES, shrink_floor=SHRINK_FLOOR)
    out["type"] = obj_type
    out["placed_scale"] = round(scale, 4)
    return out


def _object_names(root: Path) -> list[str]:
    """Every object the manifest should describe: placeable meshes UNION objects with grasp data.

    Not graspdata_final alone. A mesh with no grasp directory -- nutella and two toaster ovens on
    this dataset -- would then be absent rather than recorded as 'none', and absent reads as
    'unknown' downstream: 'toaster' would report single instead of partial, and nutella would come
    out right only by falling through to the hardcoded list this manifest exists to retire.
    """
    names = set()
    for sub in (root / "use_data", root / "graspdata_final" / "sim_parallel"):
        if sub.is_dir():
            names |= {p.name for p in sub.iterdir() if p.is_dir()}
    # Withdrawn meshes are dropped here rather than left in with their grasp facts intact: every
    # consumer reads this file to answer questions about objects a scene can actually place, and a
    # mesh nothing can place would still drag its type's verdict around.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
    from scene_spec import MESH_EXCLUDE
    return sorted(names - set(MESH_EXCLUDE))


#: Prefix on a mesh's "reason" when measure_object raised instead of returning -- the marker main()
#: uses to tell "genuinely no grasp files" apart from "a grasp file exists but would not read", so
#: a partial build's exit code can say so.
UNREADABLE_REASON_PREFIX = "unreadable grasp file: "


def build(bodex_obj_dir: str) -> dict:
    """The whole manifest for a dataset root.

    One corrupt .npy costs that object, not the whole build: a Lustre walk over ~361 objects should
    not abort on file 200 the way kitchen_gallery.mesh_tiles already refuses to let one corrupt mesh
    cost the whole gallery. measure_object's read is therefore wrapped per object; a failure is
    recorded as strategy 'none' with the exception named in "reason", and main() exits non-zero so a
    partial build is never mistaken for a clean one.
    """
    root = Path(bodex_obj_dir)
    gd = root / "graspdata_final" / "sim_parallel"
    meshes = {}
    for name in _object_names(root):
        obj_type = type_of(name)
        try:
            # measure_object handles a floating dir that does not exist: no npy files -> 'none'.
            meshes[name] = measure_object(gd / name / "floating", obj_type)
        except Exception as e:
            meshes[name] = {
                "type": obj_type, "placed_scale": None, "strategy": "none",
                "reason": f"{UNREADABLE_REASON_PREFIX}{e}", "grasps": None, "fitting": None,
                "selected": None, "fitting_selected": None, "thin_selection": None,
                "median_width_m": None, "scale_override": None,
            }
    return {
        "generated_from": str(root),
        "graspdata_mtime": int(gd.stat().st_mtime) if gd.is_dir() else 0,
        "aperture_m": APERTURE_M,
        "margin_m": MARGIN_M,
        "min_candidates": MIN_CANDIDATES,
        "shrink_floor": SHRINK_FLOOR,
        "meshes": meshes,
    }


def write(bodex_obj_dir: str, *, allow_empty: bool = False) -> Path:
    """Build the manifest and write it beside the data. Returns the path written.

    Refuses to write a 0-mesh manifest over whatever is already there unless allow_empty: pointing
    --bodex-obj-dir at an existing-but-wrong directory should not silently erase a good manifest.
    """
    manifest = build(bodex_obj_dir)
    if not manifest["meshes"] and not allow_empty:
        raise ValueError(
            f"refusing to write a 0-mesh manifest for {bodex_obj_dir} -- this looks like the wrong "
            f"--bodex-obj-dir, not an empty dataset. Pass allow_empty=True / --allow-empty if it "
            f"really is empty."
        )
    out = Path(bodex_obj_dir) / "grasp_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodex-obj-dir", default=default_bodex_obj_dir())
    ap.add_argument("--allow-empty", action="store_true",
                    help="write the manifest even if it describes zero meshes (default: refuse, "
                         "since that is almost always the wrong --bodex-obj-dir)")
    args = ap.parse_args()
    try:
        path = write(args.bodex_obj_dir, allow_empty=args.allow_empty)
    except ValueError as e:
        raise SystemExit(str(e))
    manifest = json.loads(path.read_text())
    tally = {}
    failed = []
    thin_selection = 0
    for name, entry in manifest["meshes"].items():
        tally[entry["strategy"]] = tally.get(entry["strategy"], 0) + 1
        if entry.get("thin_selection"):
            thin_selection += 1
        if str(entry.get("reason", "")).startswith(UNREADABLE_REASON_PREFIX):
            failed.append(name)
    print(f"wrote {path}  ({len(manifest['meshes'])} meshes: "
          + ", ".join(f"{n} {k}" for k, n in sorted(tally.items())) + ")")
    # Tiers 2/3 (scale_override, bimanual) and thin_selection are recorded but empty on today's
    # dataset -- naming their counts here is what makes the day they stop being empty visible.
    print(f"  bimanual (tier-3 tripwire): {tally.get('bimanual', 0)}; "
          f"thin_selection: {thin_selection}")
    if failed:
        print(f"error: {len(failed)} object(s) had unreadable grasp files, recorded as 'none': "
              + ", ".join(sorted(failed)), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
