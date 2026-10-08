"""Do cuRobo's URDF and the simulated USD agree on where ee_link1 is?

WHY. simvla_gen plans with cuRobo against ai_worker_min/.../ffw_sg2_follower.urdf and then
CHECKS the result against the USD's `ee_link1` body pose, resetting the episode when the two
differ by more than 5 cm. If the two descriptions disagree about that frame, every plan lands a
fixed distance from where the checker looks and EVERY episode fails identically -- which is what
a run of 16 envs did, 15 resets and not one success.

This compares forward kinematics of the right-arm chain base_link -> ee_link1 computed two ways,
at the same joint values:

  * from the URDF, by composing origin transforms and joint rotations (plain numpy)
  * from the USD, by composing the stage's own xform ops and joint frames

A constant offset means the descriptions disagree -- a planner problem, fixed in the URDF.
Agreement means the frames are fine and the miss is tracking: gains, timing, or contact.

    conda run -n env_isaaclab python scripts/tools/check_aiworker_urdf_vs_usd.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_R = "_UV_REEXEC"
if importlib.util.find_spec("pxr") is None:
    if os.environ.get(_R):
        raise SystemExit("pxr not importable after re-exec")
    spec = importlib.util.find_spec("isaacsim")
    libs = next((c for root in (spec.submodule_search_locations if spec else [])
                 for c in sorted(Path(root).glob("extscache/omni.usd.libs-*"))
                 if (c / "pxr").is_dir() and (c / "bin").is_dir()), None)
    if libs is None:
        raise SystemExit("no isaacsim here; run under conda -n env_isaaclab")
    env = dict(os.environ); env[_R] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(libs), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"),
                                              env.get("LD_LIBRARY_PATH", "")])
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np                                                    # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf                          # noqa: E402

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf"
USD = REPO / "source/isaaclab_assets/data/Robots/MM/aiworker/ffw_sg2.usd"

#: C2, the pose the robot spawns in, right arm only. From AIWORKER_CFG.
C2 = {
    "arm_r_joint1": 0.7028987407684326, "arm_r_joint2": -0.349675714969635,
    "arm_r_joint3": -0.5260835289955139, "arm_r_joint4": -1.8550262451171875,
    "arm_r_joint5": -0.3125845491886139, "arm_r_joint6": 0.015327823348343372,
    "arm_r_joint7": 0.9495005011558533, "lift_joint": 0.0,
}


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,               cp * cr]])


def axis_rot(axis, q):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def T(R, p):
    m = np.eye(4); m[:3, :3] = R; m[:3, 3] = p
    return m


def urdf_fk(target="ee_link1", joints=C2, root="base_link", urdf_path=URDF):
    tree = ET.parse(urdf_path)
    xml_root = tree.getroot()
    by_child, info = {}, {}
    for j in xml_root.findall("joint"):
        child = j.find("child").get("link")
        parent = j.find("parent").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
        r_, p_, y_ = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
        ax = j.find("axis")
        axis = [float(v) for v in (ax.get("xyz").split() if ax is not None else [1, 0, 0])]
        by_child[child] = parent
        info[child] = (j.get("name"), j.get("type"), np.array(xyz), rpy(r_, p_, y_), axis)
    chain = []
    cur = target
    while cur in by_child and cur != root:
        chain.append(cur)
        cur = by_child[cur]
    chain.reverse()
    m = np.eye(4)
    for link in chain:
        name, jtype, xyz, R0, axis = info[link]
        q = float(joints.get(name, 0.0))
        m = m @ T(R0, xyz)
        if jtype in ("revolute", "continuous"):
            m = m @ T(axis_rot(axis, q), np.zeros(3))
        elif jtype == "prismatic":
            m = m @ T(np.eye(3), np.asarray(axis, float) * q)
    return m, chain


def usd_fk(target="ee_link1", joints=C2, root="base_link", usd_path=USD):
    """Compose the stage's joint frames. Joint local poses are authored on the joint prim as
    localPos0/localRot0 (parent side) and localPos1/localRot1 (child side)."""
    stage = Usd.Stage.Open(str(usd_path))
    by_child, info = {}, {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        parent, child = b0[0].name, b1[0].name
        p0 = np.array(j.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0))
        q0 = j.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        p1 = np.array(j.GetLocalPos1Attr().Get() or Gf.Vec3f(0, 0, 0))
        q1 = j.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0)

        def qm(qq):
            w = qq.GetReal(); x, y, z = qq.GetImaginary()
            return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

        axis_tok = prim.GetAttribute("physics:axis").Get() if prim.GetAttribute("physics:axis") else "X"
        axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}.get(str(axis_tok), [1, 0, 0])
        jtype = ("prismatic" if prim.IsA(UsdPhysics.PrismaticJoint)
                 else "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "fixed")
        by_child[child] = parent
        info[child] = (prim.GetName(), jtype, T(qm(q0), p0), T(qm(q1), p1), axis)

    chain, cur = [], target
    while cur in by_child and cur != root:
        chain.append(cur)
        cur = by_child[cur]
    chain.reverse()
    if cur != root:
        raise SystemExit(f"USD walk from {target} stopped at {cur!r}, never reaching {root!r}")
    m = np.eye(4)
    for link in chain:
        name, jtype, A0, A1, axis = info[link]
        q = float(joints.get(name, 0.0))
        m = m @ A0
        if jtype == "revolute":
            m = m @ T(axis_rot(axis, q), np.zeros(3))
        elif jtype == "prismatic":
            m = m @ T(np.eye(3), np.asarray(axis, float) * q)
        m = m @ np.linalg.inv(A1)
    return m, chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=URDF,
                        help=f"Generated FFW_SG2 planning URDF (default: {URDF})")
    parser.add_argument("--usd", type=Path, default=USD,
                        help=f"FFW_SG2 simulation USD (default: {USD})")
    args = parser.parse_args(argv)
    bad = 0
    for label, joints in (("all joints ZERO", {}), ("C2 (the spawn pose)", C2)):
        u, uchain = urdf_fk(joints=joints, urdf_path=args.urdf)
        s, schain = usd_fk(joints=joints, usd_path=args.usd)
        d = u[:3, 3] - s[:3, 3]
        n = float(np.linalg.norm(d))
        print(f"[fk] {label}")
        print(f"[fk]   URDF chain {' -> '.join(uchain)}")
        print(f"[fk]   USD  chain {' -> '.join(schain)}")
        print(f"[fk]   URDF ee_link1 ({u[0,3]:+.5f}, {u[1,3]:+.5f}, {u[2,3]:+.5f})")
        print(f"[fk]   USD  ee_link1 ({s[0,3]:+.5f}, {s[1,3]:+.5f}, {s[2,3]:+.5f})")
        print(f"[fk]   DELTA ({d[0]:+.5f}, {d[1]:+.5f}, {d[2]:+.5f})  |d| = {n:.5f} m"
              f"   {'<== EXCEEDS the 0.05 m reach gate' if n > 0.05 else 'ok'}")
        bad += n > 0.05
    # Wrist FK alone cannot detect a jaw model with correct centres but rotated
    # collision meshes. Compare all ten hand frames at the authored open pose.
    stage = Usd.Stage.Open(str(args.usd))
    cache = UsdGeom.XformCache()
    prims = {p.GetName(): p for p in stage.Traverse()}
    for side in ("l", "r"):
        arm_root = f"arm_{side}_link7"
        parent_inverse = cache.GetLocalToWorldTransform(prims[arm_root]).GetInverse()
        for suffix in ("base", "r1", "l1", "r2", "l2"):
            name = f"gripper_{side}_rh_p12_rn_{suffix}"
            measured = np.array(cache.GetLocalToWorldTransform(prims[name]) * parent_inverse).T
            predicted, _ = urdf_fk(target=name, joints={}, root=arm_root, urdf_path=args.urdf)
            pos_error = float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3]))
            cosine = (np.trace(measured[:3, :3].T @ predicted[:3, :3]) - 1) / 2
            rot_error = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            mismatch = pos_error > 1e-5 or rot_error > .01
            bad += mismatch
            print(f"[hand-fk] {name}: {pos_error:.8f} m / {rot_error:.6f} deg "
                  f"{'MISMATCH' if mismatch else 'ok'}")
    print(f"[fk] VERDICT: the two descriptions {'DISAGREE' if bad else 'agree'} about wrist/hand frames")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
