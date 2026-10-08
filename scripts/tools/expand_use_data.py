#!/usr/bin/env python3
"""Expand the scene's object pool from an already-downloaded BODex bundle.

`use_data/` is the curated 108 MB subset the kitchen scene generator reads (via the scene-synth
config's `*/mesh/simplified.obj` globber). The full bundle also ships `processed_data/` (every
object's mesh) and `graspdata_final/` (BODex grasps for a subset). This tool symlinks more objects
INTO `use_data/` so they become placeable, without re-downloading anything:

  --graspable   every object that has BODex grasp data but isn't in use_data yet (safe: same
                processing as the existing mugs/bowls, and it grasps with the real graspdata logic).
  --clutter     a curated, kitchen-relevant allow-list of categories that have meshes but NO grasp
                data (jar, plate, teapot, cereal box, milk carton). Placeable; grasping them falls
                back to a bbox-center grasp. Their in-scene ORIENTATION is not guaranteed to match the
                kitchen up-axis — preview a scene before relying on them.

Symlinks only (nothing copied, nothing downloaded), idempotent, and reversible with --undo. The
matching CODE side (registering the type names) lives in scene_spec.py / kitchen_build.py; a clutter
type whose meshes aren't linked here will simply raise "no BODex mesh matched" until you run this.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# The clutter allow-list is defined ONCE, in scene_spec.CLUTTER (which also registers the types), so
# this tool links exactly the categories the code knows about — they cannot drift. scene_spec is
# import-light (no numpy/omni), so pulling it in here stays cheap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simvla"))
from scene_spec import CLUTTER   # noqa: E402  {type: {"category": prefix, "link": n, ...}}

_HASH = re.compile(r"_[0-9a-f]{6,}.*$")


def _base_dir() -> Path:
    d = os.environ.get("BODEX_OBJ_DIR")
    if not d:
        raise SystemExit("Set BODEX_OBJ_DIR to your BODex bundle (the dir holding use_data/).")
    return Path(d)


def _category(obj_name: str) -> str:
    return _HASH.sub("", obj_name)


def _link(src: Path, dst: Path, dry: bool) -> bool:
    """Symlink src->dst if dst is absent. Returns True if it made (or would make) a link."""
    if dst.exists() or dst.is_symlink():
        return False
    if not (src / "mesh" / "simplified.obj").exists():
        return False
    if not dry:
        dst.symlink_to(src)
    return True


def expand(base: Path, *, graspable: bool, clutter: bool, dry: bool) -> None:
    use = base / "use_data"
    processed = base / "processed_data"
    graspdir = base / "graspdata_final" / "sim_parallel"
    present = {p.name for p in use.iterdir() if p.is_dir() or p.is_symlink()}

    if graspable:
        wanted = {p.name for p in graspdir.iterdir()} if graspdir.is_dir() else set()
        made = 0
        for obj in sorted(wanted - present):
            if _link(processed / obj, use / obj, dry):
                made += 1
        print(f"[graspable] linked {made} grasp-backed objects into use_data"
              f"{' (dry-run)' if dry else ''}")

    if clutter:
        for tname, spec in CLUTTER.items():
            prefix, cap = spec["category"], spec["link"]
            cands = sorted(p.name for p in processed.iterdir()
                           if p.name.startswith(prefix) and p.name not in present)
            made = 0
            for obj in cands[:cap]:
                if _link(processed / obj, use / obj, dry):
                    made += 1
            print(f"[clutter]  {tname:16} ({prefix}) linked {made}/{min(cap, len(cands))}"
                  f"{' (dry-run)' if dry else ''}")


def undo(base: Path, dry: bool) -> None:
    """Remove symlinks this tool added (real directories from the original download are left alone)."""
    use = base / "use_data"
    removed = 0
    for p in sorted(use.iterdir()):
        if p.is_symlink():
            if not dry:
                p.unlink()
            removed += 1
    print(f"[undo] removed {removed} symlinks{' (dry-run)' if dry else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graspable", action="store_true", help="link all grasp-backed objects")
    ap.add_argument("--clutter", action="store_true", help="link the curated clutter allow-list")
    ap.add_argument("--all", action="store_true", help="both --graspable and --clutter")
    ap.add_argument("--undo", action="store_true", help="remove symlinks this tool added")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, link nothing")
    args = ap.parse_args()

    base = _base_dir()
    if args.undo:
        undo(base, args.dry_run)
        return
    graspable = args.graspable or args.all
    clutter = args.clutter or args.all
    if not (graspable or clutter):
        ap.error("choose --graspable, --clutter, --all, or --undo")
    expand(base, graspable=graspable, clutter=clutter, dry=args.dry_run)


if __name__ == "__main__":
    main()
