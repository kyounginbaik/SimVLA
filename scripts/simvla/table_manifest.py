"""Read/write side of the table manifest: which Objaverse table candidates exist, and where each
one stands in the pipeline.

The chair library's sibling, and deliberately its mirror image -- same shape, same stages, same
"a rejected entry is kept with the numbers it was judged on" contract. Written by
scripts/tools/build_table_manifest.py and stored beside the dataset under $TABLE_OBJ_DIR.

Every table entry ACCUMULATES fields as it passes through the pipeline's stages:
  fetch    -- uid, category, raw_path
  curate   -- accepted, rejected_because (when False), extents, normalized_extents, faces,
              up_axis, up_axis_source, work_surface_frac, overhang, shape_grid(+_n),
              export_height_m, export_footprint_m
  select   -- selected, selection_rank
  export   -- obj_path, scale_applied, height_m, surface_height_m
  convert  -- usd_path, hull_count
  repair_collisions -- has_collision

Two fields a chair entry carries and a table entry deliberately does NOT:

  * facing_axis / facing_direction. The desk gate measured the chair pipeline's own asymmetry
    signal on 60 table meshes ("Does a facing signal exist? For a table, no."): chairs separate 170x against their noise
    floor, tables 9x, and only 2 of 15 usable tables clear the chair gate's own confidence cut.
    A table has a long axis, not a front, and the long axis comes free from the bounding box.
  * height_m as the thing that was scaled. A chair is scaled so its bounding box reaches a
    uid-seeded target height; a table is scaled so its WORK SURFACE reaches 0.74 m, which is not
    the top of the bounding box on every mesh. surface_height_m records the datum, height_m the
    resulting overall height.

manifest["thresholds"] and manifest["material_group"] are manifest-level, exactly as for chairs.
"""
from __future__ import annotations

import furniture_manifest

MANIFEST_NAME = "table_manifest.json"
DEFAULT_TABLE_OBJ_DIR = "objaverse_tables"

LIBRARY = furniture_manifest.Library(
    name="table",
    entries_key="tables",
    manifest_name=MANIFEST_NAME,
    env_var="TABLE_OBJ_DIR",
    default_dir=DEFAULT_TABLE_OBJ_DIR,
)


def root(table_obj_dir=None) -> str:
    """The asset root: an explicit argument, else $TABLE_OBJ_DIR, else the /lustre default."""
    return LIBRARY.root(table_obj_dir)


def manifest_path(table_obj_dir=None) -> str:
    return LIBRARY.manifest_path(table_obj_dir)


def _empty_skeleton() -> dict:
    return LIBRARY.empty_skeleton()


def load(table_obj_dir=None) -> dict:
    """The manifest, or an empty skeleton when there isn't a readable one yet."""
    return LIBRARY.load(table_obj_dir)


def accepted(manifest) -> list[dict]:
    """Entries curate has judged and accepted. Excludes rejects and not-yet-judged candidates."""
    return LIBRARY.accepted(manifest)


def save(manifest, table_obj_dir=None) -> None:
    """Write the manifest back beside the dataset, creating the root directory if needed."""
    LIBRARY.save(manifest, table_obj_dir)
