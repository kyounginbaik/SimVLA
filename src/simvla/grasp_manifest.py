"""Read side of the grasp manifest: what BODex data actually exists, per mesh.

Written by scripts/tools/build_grasp_manifest.py and stored beside the dataset, because it
describes one install's data rather than the repo. stdlib only -- no numpy, no isaaclab -- so the
setup form, the strategy chooser and the campaign can all consult it without dragging in the
measurement stack.

Every query takes the loaded manifest AS AN ARGUMENT rather than loading it itself. That is what
keeps grasp_strategy pure (object_dims + scene_spec only, as its docstring promises) and stops a
page handler from touching the filesystem on every render.
"""
from __future__ import annotations

import json
import os

MANIFEST_NAME = "grasp_manifest.json"
DEFAULT_BODEX_OBJ_DIR = "BODex_obj"


def _root(bodex_obj_dir=None) -> str:
    return bodex_obj_dir or os.environ.get("BODEX_OBJ_DIR") or DEFAULT_BODEX_OBJ_DIR


def manifest_path(bodex_obj_dir=None) -> str:
    return os.path.join(_root(bodex_obj_dir), MANIFEST_NAME)


def load(bodex_obj_dir=None):
    """The manifest, or None when there isn't a readable one.

    None is a supported state, not an error: a fresh clone has no dataset, and every consumer
    falls back to its legacy behaviour rather than refusing to run.
    """
    try:
        with open(manifest_path(bodex_obj_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def object_name(mesh_path: str) -> str:
    """'<root>/use_data/core_mug_1038e4/mesh/simplified.obj' -> 'core_mug_1038e4'.

    Deliberately the SAME derivation as isaaclab.simvla.utils.load_grasp_file
    (`input_path.split("/")[-3]`): if these two ever disagree, the manifest describes one object
    and the planner loads another.
    """
    parts = str(mesh_path).split("/")
    return parts[-3] if len(parts) >= 3 else str(mesh_path)


def strategy_for_mesh(manifest, mesh_path: str) -> str:
    """'single' | 'bimanual' | 'none' | 'unknown' for one mesh."""
    if not manifest:
        return "unknown"
    entry = manifest.get("meshes", {}).get(object_name(mesh_path))
    return entry.get("strategy", "unknown") if entry else "unknown"


def strategy_for_type(manifest, obj_type: str) -> str:
    """The type's strategy: the one its meshes agree on, or 'partial' when they disagree.

    'partial' is a real state, not a defensive branch -- toaster ships 4 meshes of which 2 have
    grasps -- and a menu that hid it would promise a grasp the random mesh may not deliver.
    """
    if not manifest:
        return "unknown"
    found = {e.get("strategy", "unknown")
             for e in manifest.get("meshes", {}).values() if e.get("type") == obj_type}
    if not found:
        return "unknown"
    if len(found) == 1:
        return found.pop()
    return "partial"


def _graspdata_dir(manifest) -> str:
    return os.path.join(manifest.get("generated_from", ""), "graspdata_final", "sim_parallel")


def is_stale(manifest) -> bool:
    """True when graspdata_final has changed since the manifest was written.

    The one real failure mode of an offline manifest, so it is checkable rather than assumed.
    A dataset that is not present at all is not stale -- it is absent, which load() already models.

    (OSError, TypeError, ValueError): a vanished graspdata dir raises OSError; a hand-edited,
    non-numeric graspdata_mtime (None, "unknown", ...) raises TypeError or ValueError out of the
    int() calls -- neither should crash a caller that is merely asking "is this stale?".
    """
    if not manifest:
        return False
    try:
        return int(os.stat(_graspdata_dir(manifest)).st_mtime) > int(manifest.get("graspdata_mtime", 0))
    except (OSError, TypeError, ValueError):
        return False


def stale_warning(manifest) -> str | None:
    """A loud, human-readable warning when the manifest is older than graspdata_final, else None.

    Names both timestamps and the fix, per the design's error-handling table ("Manifest older than
    graspdata_final -> Loud warning naming both timestamps. This is the design's only real failure
    mode."). Callers just `print()` this -- kept as a plain string, not a logger, so kitchen_wizard
    (stdlib-only) can use it too.
    """
    if not is_stale(manifest):
        return None
    try:
        live = int(os.stat(_graspdata_dir(manifest)).st_mtime)
    except OSError:
        live = "unknown"
    return (
        f"WARNING: {manifest.get('generated_from', '?')}/grasp_manifest.json is stale -- "
        f"graspdata_final changed after the manifest was built "
        f"(manifest graspdata_mtime={manifest.get('graspdata_mtime')}, live mtime={live}). "
        f"Grasp strategies may be wrong. Fix: python scripts/tools/build_grasp_manifest.py"
    )
