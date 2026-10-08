"""Read/write side of the chair manifest: which Objaverse chair candidates exist, and where each
one stands in the pipeline.

Written by scripts/tools/build_chair_manifest.py and stored beside the dataset, because it
describes one install's data rather than the repo. stdlib only -- no objaverse, no trimesh, no
isaaclab -- so anything that just wants to know "which chairs are accepted?" can ask without
dragging in the fetch/curate/export/convert stack that produced the answer.

The generic half of this module now lives in furniture_manifest.Library, which table_manifest binds
the same way: the two differ only in an environment variable, a default directory, a filename and
the key their entries sit under. This module keeps its own name and its own module-level functions
because scripts/simvla/kitchen_build.py and scripts/simvla/test_chair_manifest.py both import it by
name and call `chair_manifest.load()` / `.accepted()` / `.root()` directly -- the shared class is an
implementation detail behind an unchanged surface.

Every chair entry ACCUMULATES fields as it passes through the pipeline's stages, rather than being
replaced at each stage:
  fetch    (Task 2) -- uid, category, raw_path
  curate   (Task 3) -- accepted, rejected_because (when False), extents, normalized_extents,
                        faces, up_axis, facing_axis
  export   (Task 4) -- obj_path, scale_applied, height_m, facing_axis (upright, rescaled)
  convert  (Task 5) -- usd_path, hull_count
  repair_collisions (bugfix, 2026-08-11 findings) -- has_collision: a per-entry bool recording
                        whether the converted USD's mesh actually carries UsdPhysics collision
                        geometry, since MeshConverter's own conversion silently drops it for most
                        chairs (see build_chair_manifest.repair_collisions's docstring).

One field is manifest-LEVEL, not per-entry, because it describes a single decision that applies
across every chair rather than something measured per candidate:
  manifest["thresholds"]     -- curate's criteria (Task 3), same on every accepted/rejected entry.
  manifest["material_group"] -- export (bugfix, 2026-08-11 findings) records "floor": chairs carry
                        no material of their own (Objaverse's own textures are dropped at export
                        time) and are meant to take scripts/simvla/kitchen_build.py's
                        MATERIALS['floor'] pool at placement time, the same wood/tile group the
                        kitchen's table already uses. Recorded here so a later placement stage
                        does not have to rediscover that decision.

A rejected entry is KEPT, never deleted. The point of curate recording "accepted": False plus a
"rejected_because" reason, rather than dropping the candidate, is so a later threshold change can
see exactly what it excluded and why -- without re-fetching or re-measuring anything.
"""
from __future__ import annotations

from . import furniture_manifest

MANIFEST_NAME = "chair_manifest.json"
DEFAULT_CHAIR_OBJ_DIR = "objaverse_chairs"

LIBRARY = furniture_manifest.Library(
    name="chair",
    entries_key="chairs",
    manifest_name=MANIFEST_NAME,
    env_var="CHAIR_OBJ_DIR",
    default_dir=DEFAULT_CHAIR_OBJ_DIR,
)


def root(chair_obj_dir=None) -> str:
    """The asset root: an explicit argument, else $CHAIR_OBJ_DIR, else the /lustre default."""
    return LIBRARY.root(chair_obj_dir)


def manifest_path(chair_obj_dir=None) -> str:
    return LIBRARY.manifest_path(chair_obj_dir)


def _empty_skeleton() -> dict:
    return LIBRARY.empty_skeleton()


def load(chair_obj_dir=None) -> dict:
    """The manifest, or an empty skeleton when there isn't a readable one yet."""
    return LIBRARY.load(chair_obj_dir)


def accepted(manifest) -> list[dict]:
    """Entries curate has judged and accepted. Excludes rejects and not-yet-judged candidates."""
    return LIBRARY.accepted(manifest)


def save(manifest, chair_obj_dir=None) -> None:
    """Write the manifest back beside the dataset, creating the root directory if needed."""
    LIBRARY.save(manifest, chair_obj_dir)
