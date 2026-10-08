#!/usr/bin/env python3
"""Repair a converted USD whose collision geometry MeshConverter silently dropped.

## Why this exists

`scripts/tools/convert_mesh.py` (an IsaacLab-provided, vendored tool -- not touched by this fix)
wraps `isaaclab.sim.converters.mesh_converter.MeshConverter`. Investigating why most chair
conversions came out with no `UsdPhysics.CollisionAPI` / `UsdPhysics.MeshCollisionAPI` at all
(3/20 had it, 17/20 didn't, each USD having exactly one Mesh prim) found the root cause one layer
deeper, inside `MeshConverter._convert_asset`
(source/isaaclab/isaaclab/sim/converters/mesh_converter.py) -- also vendored, so also not edited
here, per the same instruction that rules out touching convert_mesh.py.

`_convert_asset` does this, in order, inside ONE process:
  1. runs `omni.kit.asset_converter` to write a RAW converted mesh to `self.usd_path`
  2. builds an in-memory stage that REFERENCES that same `self.usd_path`
  3. `Export()`s (flattens) that in-memory stage back OUT to the SAME `self.usd_path`,
     overwriting what step 1 wrote
  4. immediately re-`Usd.Stage.Open()`s + `.Reload()`s that same path
  5. walks `geom_prim.GetChildren()` looking for the Mesh prim, and applies
     `UsdPhysics.MeshCollisionAPI` / `CollisionAPI` to whichever one it finds

Steps 3->4 write and then, in the same process, almost immediately re-read the SAME file path.
This is not a property of any specific mesh's geometry -- measured face count, watertightness,
vertex count and aspect ratio do not separate the chairs that got a collider from the ones that
didn't, and re-converting a chair that had previously come out WITH a collider, fresh, to a brand
new output directory, reproducibly comes out WITHOUT one (0/5 across five repeated attempts in
the investigation that found this). A crash was also reproduced at the exact `GetChildren()` call
in step 5 (`RuntimeError: Accessed invalid null prim`) when the same sequence ran back-to-back in
an already-warmed-up Kit session. Both symptoms point at the same place: step 4's reopen racing
step 3's write-to-the-same-path, most likely against this cluster's Lustre-backed storage under
load, not at anything convert_mesh.py's CALLER (this project's code) controls or can retry its
way past reliably from the outside.

## What this script does instead

The mesh GEOMETRY itself is never wrong -- every converted chair, working or not, has the
identical `/`<basename>`/geometry/mesh` topology once conversion finishes and the process exits;
only the collision schema authoring is missing. So rather than trying to win MeshConverter's
internal race, this script runs strictly AFTER convert_mesh.py has already finished and exited: it
opens each ALREADY-WRITTEN, ALREADY-CLOSED usd file in a FRESH process, walks every Mesh prim, and
if `UsdPhysics.MeshCollisionAPI` (or `CollisionAPI`) is missing, authors it directly -- via the
exact same calls MeshConverter itself makes (`UsdPhysics.MeshCollisionAPI.Apply` +
`isaaclab.sim.schemas.schemas.define_collision_properties`) -- and re-saves. There is no
export/reference/reopen-the-same-path dance here, just open -> mutate -> save, so there is no
place for the race to occur. Verified empirically: repairing a known-broken chair this way and
then reopening the result in ANOTHER fresh process confirms the collision persisted.

Boots Isaac Sim once (like convert_mesh.py) and repairs every USD passed on the command line in
that one process/boot -- not one boot per chair -- since this is a batch fix-up pass, not a
per-chair conversion step.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Ensure every given USD's mesh prim(s) carry collision.")
parser.add_argument(
    "targets",
    nargs="+",
    help="One or more uid=path.usd pairs (uid is an arbitrary label echoed back in the RESULT line).",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "meshSimplification"],
    help="The approximation to author when a mesh is missing one. Defaults to convexDecomposition,"
    " matching convert_mesh.py's own default.",
)
parser.add_argument(
    "--results-file",
    type=str,
    default=None,
    help=(
        "If given, RESULT lines are ALSO written here (one per target, flushed immediately), not"
        " just printed to stdout. Needed because stdout captured via subprocess.run from a script"
        " that boots Isaac Sim has been observed, empirically, to lose everything printed after"
        " Kit finishes booting -- reproduced identically by an earlier diagnostic in this same"
        " investigation (scripts/tools/build_chair_manifest.py callers rely on this file, not"
        " stdout, for exactly that reason -- see _default_repair_collision_runner)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.schemas import schemas, schemas_cfg  # noqa: E402


def _emit(line, results_fh):
    """Print AND (if given) write+flush to the results file. See --results-file's help for why
    the file, not captured stdout, is the reliable channel here."""
    print(line)
    if results_fh is not None:
        results_fh.write(line + "\n")
        results_fh.flush()


def repair_one(uid, usd_path, collision_approximation, results_fh=None):
    """Open usd_path, author collision on every Mesh prim missing it, save if anything changed.

    Returns True if every Mesh prim in the stage carries collision by the time this returns
    (whether it already did, or was just repaired), False if the stage has no Mesh prim at all
    (nothing to report success or failure about) or a mesh still lacks collision after the
    repair attempt (would indicate this workaround itself failed -- not expected, but not
    fabricated as a pass either).
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        _emit(f"RESULT {uid} False  # could not open {usd_path}", results_fh)
        return False

    collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
    mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    if not mesh_prims:
        _emit(f"RESULT {uid} False  # no Mesh prim found in {usd_path}", results_fh)
        return False

    changed = False
    for mesh_prim in mesh_prims:
        if mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        schemas.define_collision_properties(
            prim_path=mesh_prim.GetPath(), cfg=collision_props, stage=stage
        )
        changed = True

    if changed:
        stage.Save()

    # Re-check from the same stage object after Save -- confirms the API is actually attached to
    # the prim we just touched, not merely that Apply() didn't raise.
    ok = all(p.HasAPI(UsdPhysics.MeshCollisionAPI) for p in mesh_prims)
    _emit(f"RESULT {uid} {ok}  # repaired={changed}", results_fh)
    return ok


def main():
    results_fh = open(args_cli.results_file, "w", buffering=1) if args_cli.results_file else None
    try:
        results = {}
        for target in args_cli.targets:
            uid, _, usd_path = target.partition("=")
            results[uid] = repair_one(uid, usd_path, args_cli.collision_approximation, results_fh)

        n_ok = sum(1 for v in results.values() if v)
        _emit(f"SUMMARY {n_ok}/{len(results)} have collision", results_fh)
    finally:
        if results_fh is not None:
            results_fh.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
