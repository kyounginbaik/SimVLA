"""Repair the unphysical rigid-body inertias in the AI Worker (FFW_SG2) USD.

WHAT WAS WRONG. `base_link` shipped with a diagonal inertia of
(17254.7, 39837.5, 28461.4) kg m^2 against a mass of 35.99 kg. That is a radius of gyration of
22 to 33 METRES on a robot 0.68 m across, and it is not a unit slip -- the three ratios against
a uniform box of the link's own geometry are 7033, 15639 and 12550, which is not one factor.
Every other link in the asset is sane: the drive wheels read 0.007 kg m^2 for 3.8 kg.

WHAT IT COST. The base could not turn. `base_revolute_z_joint` carries a 100 N m cap in the USD,
so against Izz = 28461 the angular acceleration is 0.0035 rad/s^2 -- immobile in any episode.
Measured on kitchen 1215 with SIMVLA_NAV_TRACE: the nav commanded 0.463 rad/s of yaw for 425
frames and the base achieved exactly 0.000, never leaving its spawn pose, so all 16 envs sat on
goal step 0 forever. The failure looks like a nav bug and is not one; the actuator, the action
term and the joint drives were all measured correct first.

WHAT THIS DOES. For every rigid body whose implied radius of gyration exceeds the whole robot's
own bounding box -- the test for "this cannot describe this object" -- replace the diagonal
inertia with that of a uniform box of the link's own mass and its own visible extents. A uniform
box is an approximation, but it is the right order of magnitude and it is derived from the asset
rather than invented; the alternative on offer was a number four orders out.

IDEMPOTENT: a link already inside the bound is left untouched, so re-running is a no-op and the
script can be re-run after an asset refresh to check.

    conda run --no-capture-output -n env_isaaclab python scripts/tools/fix_aiworker_base_inertia.py
    conda run --no-capture-output -n env_isaaclab python scripts/tools/fix_aiworker_base_inertia.py --check

Re-execs itself with Isaac's vendored pxr; no Kit boot, no GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_REEXEC = "AIWORKER_INERTIA_REEXEC"


def reexec_with_pxr() -> None:
    if importlib.util.find_spec("pxr") is not None:
        return
    if os.environ.get(_REEXEC):
        raise SystemExit("pxr still not importable after re-exec")
    spec = importlib.util.find_spec("isaacsim")
    libs = next(
        (c for root in (spec.submodule_search_locations if spec else [])
         for c in sorted(Path(root).glob("extscache/omni.usd.libs-*"))
         if (c / "pxr").is_dir() and (c / "bin").is_dir()),
        None,
    )
    if libs is None:
        raise SystemExit("no isaacsim in this environment; run under conda -n env_isaaclab")
    env = dict(os.environ)
    env[_REEXEC] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(libs), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"), env.get("LD_LIBRARY_PATH", "")])
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


reexec_with_pxr()

import numpy as np                                          # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics                 # noqa: E402

REPO = Path(__file__).resolve().parents[2]
USD = REPO / "source/isaaclab_assets/data/Robots/MM/aiworker/ffw_sg2.usd"
ROOT = "/ffw_sg2_follower"

#: A link whose radius of gyration exceeds this many times the robot's own longest dimension is
#: describing something that is not this robot. 1.0 would be the strict bound; 2.0 leaves room
#: for a genuinely awkward mass distribution without admitting a 22 m one.
RG_TOLERANCE = 2.0


def visible_extents(stage, xcache, prim):
    """The link's own visible geometry, as extents in its own frame."""
    m = xcache.GetLocalToWorldTransform(prim)
    rot = np.array([[m[0][0], m[1][0], m[2][0]],
                    [m[0][1], m[1][1], m[2][1]],
                    [m[0][2], m[1][2], m[2][2]]])
    trans = np.array([m[3][0], m[3][1], m[3][2]])
    pts = []
    # TraverseInstanceProxies: the stage is make_instanceable, so a plain PrimRange sees nothing.
    for d in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if d.GetTypeName() != "Mesh":
            continue
        im = UsdGeom.Imageable(d)
        if im.ComputeVisibility() == "invisible" or im.ComputePurpose() not in ("default", "render"):
            continue
        v = UsdGeom.Mesh(d).GetPointsAttr().Get()
        if not v:
            continue
        mm = xcache.GetLocalToWorldTransform(d)
        r = np.array([[mm[0][0], mm[0][1], mm[0][2]],
                      [mm[1][0], mm[1][1], mm[1][2]],
                      [mm[2][0], mm[2][1], mm[2][2]]])
        t = np.array([mm[3][0], mm[3][1], mm[3][2]])
        pts.append((np.array([[q[0], q[1], q[2]] for q in v]) @ r + t - trans) @ rot)
    if not pts:
        return None
    P = np.vstack(pts)
    return P.max(axis=0) - P.min(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(prog="fix_aiworker_base_inertia.py", description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report without writing; exit 1 if any link is still unphysical")
    args = ap.parse_args()

    if not USD.is_file():
        raise SystemExit(f"missing {USD}")
    stage = Usd.Stage.Open(str(USD))
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    span = 0.0
    for p in bodies:
        e = visible_extents(stage, xcache, p)
        if e is not None:
            span = max(span, float(e.max()))
    limit = span * RG_TOLERANCE
    print(f"[inertia] {len(bodies)} rigid bodies; robot's longest link dimension {span:.3f} m, "
          f"so a radius of gyration over {limit:.3f} m is unphysical", flush=True)

    solver_bad = 0
    # THE ARTICULATION ROOT'S OWN AUTHORED SOLVER COUNTS. The USD authors
    # solverPositionIterationCount=32, and 8 is already enough to kill this robot's two
    # PRISMATIC base joints stone dead (measured: 0.0% of a commanded 0.20 m/s at 8, 88% at 4 --
    # the joints act through 1e-7 kg dummy links carrying 94 kg, and more position iterations
    # resolve that ill-conditioned chain against a velocity drive with stiffness ~0).
    # AIWORKER_CFG asks for 4, but a config that fails to apply leaves 32 in force and the
    # failure is silent, so the asset carries the safe value too.
    # Traverse the STAGE, not `bodies`: the articulation root here is root_joint, a JOINT, and
    # joints are not rigid bodies, so it never appears in that list.
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        for attr, want in (("physxArticulation:solverPositionIterationCount", 4),
                           ("physxArticulation:solverVelocityIterationCount", 0)):
            a = p.GetAttribute(attr)
            if a and a.Get() != want:
                print(f"[inertia] {p.GetName()}: {attr.split(':')[-1]} {a.Get()} -> {want}",
                      flush=True)
                solver_bad += 1
                if not args.check:
                    a.Set(want)
                    print("[inertia]   WRITTEN", flush=True)

    bad = 0
    for p in bodies:
        if not p.HasAPI(UsdPhysics.MassAPI):
            continue
        api = UsdPhysics.MassAPI(p)
        mass = api.GetMassAttr().Get()
        inertia = api.GetDiagonalInertiaAttr().Get()
        if not mass or inertia is None:
            continue
        # THE MASSLESS DUMMY LINKS ARE NOT A DEFECT. world / base_x / base_y carry mass 1e-7 kg
        # and inertia 1e-4 as numerical placeholders -- the ratio makes the radius of gyration
        # read 31.6 m, but there is no body there to have one. Anubis's USD carries byte-identical
        # values, which is what says they are intended rather than broken.
        if mass < 1e-3:
            continue
        I = np.array([float(v) for v in inertia])
        rg = np.sqrt(np.maximum(I, 0.0) / mass)
        if rg.max() <= limit:
            continue
        bad += 1
        ext = visible_extents(stage, xcache, p)
        if ext is None:
            print(f"[inertia] {p.GetName()}: rg={rg.max():.1f} m but no visible geometry to "
                  f"derive a replacement from -- LEFT ALONE", flush=True)
            continue
        x, y, z = ext
        box = np.array([mass * (y * y + z * z) / 12.0,
                        mass * (x * x + z * z) / 12.0,
                        mass * (x * x + y * y) / 12.0])
        print(f"[inertia] {p.GetName()}: mass {mass:.3f} kg, "
              f"inertia {tuple(round(float(v), 1) for v in I)} -> rg up to {rg.max():.1f} m",
              flush=True)
        print(f"[inertia]   extents {np.round(ext, 4).tolist()} -> uniform box "
              f"{tuple(round(float(v), 4) for v in box)}", flush=True)
        if not args.check:
            api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in box]))
            print(f"[inertia]   WRITTEN", flush=True)

    if bad == 0:
        print("[inertia] every rigid body is physical; nothing to do", flush=True)
    # THE SOLVER COUNTS MUST BE PART OF THIS DECISION. They were not: the early return below
    # fired whenever no INERTIA was bad, which discarded the solver writes without saving and
    # let --check exit 0 on a USD still authoring 32 position iterations. The write printed
    # "WRITTEN" and nothing reached disk.
    if bad == 0 and solver_bad == 0:
        return 0
    if args.check:
        if bad:
            print(f"[inertia] {bad} unphysical bodies remain (--check made no changes)",
                  flush=True)
        if solver_bad:
            print(f"[inertia] {solver_bad} solver attributes wrong (--check made no changes)",
                  flush=True)
        return 1
    stage.GetRootLayer().Save()
    print(f"[inertia] repaired {bad} bodies, {solver_bad} solver attributes; saved {USD}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
