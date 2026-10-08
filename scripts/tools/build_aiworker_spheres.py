"""Author cuRobo collision spheres for the AI Worker arms, MEASURED off the USD.

Hand-written spheres are how a planner ends up confidently swinging an arm through a counter:
too small and it clips, too large and every candidate grasp is rejected as unreachable. This
fits a short chain of spheres to each link's OWN visible geometry, expressed in that link's own
frame, which is the frame cuRobo wants them in.

    conda run --no-capture-output -n env_isaaclab python scripts/tools/build_aiworker_spheres.py

Writes configs/curobo/robot/spheres/aiworker_spheres_{right,left}.yml.

Re-execs itself with Isaac's vendored pxr when pxr is not importable, which is the normal case
-- no Kit boot, no GPU, no Omniverse app. Same trick as scripts/tools/build_robot_reference.py.

WHY THE GEOMETRY NEEDS TraverseInstanceProxies: this stage was converted with
make_instanceable (see MM/aiworker/config.yaml), so a plain Usd.PrimRange walks straight past
the meshes and reports an armless robot. That is not a hypothetical -- it is what the first
run of this probe did.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REEXEC = "AIWORKER_SPHERES_REEXEC"


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
    sys.stdout.flush()          # execve does not flush; without this the line is lost on a pipe
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


reexec_with_pxr()

import numpy as np                                    # noqa: E402
from pxr import Usd, UsdGeom                          # noqa: E402
from collision_sphere_fit import fit_box_grid, fit_convex_grid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
USD = Path(os.environ.get("SIMVLA_ASSETS_DIR", REPO / "source/isaaclab_assets/data")) / "Robots/MM/aiworker/ffw_sg2.usd"
ROOT = "/ffw_sg2_follower"
OUT = REPO / "configs/curobo/robot/spheres"

#: One entry per collision link, with how many spheres to string along its longest axis. The
#: arm links are slender, so a short chain along the principal axis covers each one tightly.
#: The gripper parts get two apiece because they are what actually reaches into a shelf, and a
#: single sphere over a finger is either a sphere that misses the tip or one that swallows the
#: gap between the jaws -- and a sphere in the gap makes every grasp look like a collision.
LINKS = {
    "arm_{s}_link1": 3, "arm_{s}_link2": 3, "arm_{s}_link3": 3, "arm_{s}_link4": 3,
    "arm_{s}_link5": 2, "arm_{s}_link6": 2, "arm_{s}_link7": 3,
    "gripper_{s}_rh_p12_rn_base": 2,
    "gripper_{s}_rh_p12_rn_r1": 2, "gripper_{s}_rh_p12_rn_r2": 2,
    "gripper_{s}_rh_p12_rn_l1": 2, "gripper_{s}_rh_p12_rn_l2": 2,
}


def link_points(stage, xcache, path):
    """Every visible vertex under `path`, expressed in that link's OWN frame."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = xcache.GetLocalToWorldTransform(prim)
    rot = np.array([[m[0][0], m[1][0], m[2][0]],
                    [m[0][1], m[1][1], m[2][1]],
                    [m[0][2], m[1][2], m[2][2]]])
    trans = np.array([m[3][0], m[3][1], m[3][2]])
    out = []
    for d in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if d.GetTypeName() != "Mesh":
            continue
        im = UsdGeom.Imageable(d)
        if im.ComputeVisibility() == "invisible":
            continue
        if im.ComputePurpose() not in ("default", "render"):
            continue                        # the collision hulls are `guide`, and far coarser
        pts = UsdGeom.Mesh(d).GetPointsAttr().Get()
        if not pts:
            continue
        mm = xcache.GetLocalToWorldTransform(d)
        r = np.array([[mm[0][0], mm[0][1], mm[0][2]],
                      [mm[1][0], mm[1][1], mm[1][2]],
                      [mm[2][0], mm[2][1], mm[2][2]]])
        t = np.array([mm[3][0], mm[3][1], mm[3][2]])
        world = np.array([[p[0], p[1], p[2]] for p in pts]) @ r + t
        out.append((world - trans) @ rot)
    return np.vstack(out) if out else None


def fit_chain(pts, n):
    """`n` spheres strung along the cloud's principal axis, each big enough to hold the slice
    it owns. Returns [(centre_xyz, radius), ...]."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    t = centred @ axis
    edges = np.linspace(t.min(), t.max(), n + 1)
    spheres = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        sel = (t >= lo) & (t <= hi) if i == n - 1 else (t >= lo) & (t < hi)
        slab = pts[sel]
        if len(slab) < 4:
            continue
        mid = centroid + axis * ((lo + hi) / 2.0)
        radius = float(np.linalg.norm(slab - mid, axis=1).max())
        spheres.append((mid, radius))
    return spheres


def main() -> None:
    if not USD.is_file():
        raise SystemExit(f"missing {USD}")
    stage = Usd.Stage.Open(str(USD))
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    OUT.mkdir(parents=True, exist_ok=True)
    for side, letter in (("right", "r"), ("left", "l")):
        lines = [
            "# GENERATED by scripts/tools/build_aiworker_spheres.py -- do not hand-edit.",
            "# Fitted to the visible geometry of Robots/MM/aiworker/ffw_sg2.usd, each sphere",
            "# expressed in its own link's frame, which is what cuRobo expects.",
            "collision_spheres:",
        ]
        total = 0
        for tmpl, n in LINKS.items():
            name = tmpl.format(s=letter)
            pts = link_points(stage, xcache, f"{ROOT}/{name}")
            if pts is None or len(pts) < 8:
                print(f"  [spheres] {name}: no geometry, skipped", flush=True)
                continue
            # Link7 includes the wrist camera bracket. Its old 77 mm spheres
            # penetrated the counter by 25 mm at a horizontal mug grasp even when
            # the physical link was above it. Cover the full box with smaller cells,
            # not smaller radii on the existing centers (which would miss geometry).
            if name.endswith("link7"):
                fitted = fit_box_grid(pts)
            elif name.startswith("gripper_"):
                # The former 36–49 mm palm/linkage spheres filled the open
                # grasp volume. Smaller conservative cells preserve that gap
                # without omitting any part of the link's bounding box.
                fitted = fit_convex_grid(pts, max_cell_m=0.015)
            else:
                fitted = fit_chain(pts, n)
            if not fitted:
                print(f"  [spheres] {name}: chain fit produced nothing, skipped", flush=True)
                continue
            lines.append(f"  {name}:")
            for centre, radius in fitted:
                lines.append(f"    - center: [{centre[0]:.5f}, {centre[1]:.5f}, {centre[2]:.5f}]")
                lines.append(f"      radius: {radius:.5f}")
                total += 1
        path = OUT / f"aiworker_spheres_{side}.yml"
        path.write_text("\n".join(lines) + "\n")
        print(f"[spheres] wrote {path} ({total} spheres)", flush=True)


if __name__ == "__main__":
    main()
