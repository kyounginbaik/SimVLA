"""Where is the PINCH POINT relative to ee_link1, and which way do the jaws travel?

WHY. The grasp target is authored as a pose for `ee_link1`, and the executor drives ee_link1 to
it. But the thing that has to end up around the bottle is the gap between the two jaw pads. If the
pinch point sits some distance from ee_link1's origin, then putting ee_link1 on the bottle's axis
puts the PADS somewhere else, and the jaws close on air -- which is what every run has done: the
hand rises 0.25-0.35 m and the bottle rises 0.0000.

This measures, off the USD and at the C2 spawn pose:
  * the pad midpoint expressed in ee_link1's own frame -- the offset the target needs
  * the jaw travel direction in that frame -- which axis the gap opens along, i.e. what the
    tool-frame quaternion has to align with the object

    conda run -n env_isaaclab python scripts/tools/measure_aiworker_pinch.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_R = "_PINCH_REEXEC"
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
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"), env.get("LD_LIBRARY_PATH", "")])
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np                                                    # noqa: E402
from pxr import Usd, UsdPhysics, Gf                                    # noqa: E402

USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source/isaaclab_assets/data/Robots/MM/aiworker/ffw_sg2.usd",
)

C2 = {
    "arm_r_joint1": 0.7028987407684326, "arm_r_joint2": -0.349675714969635,
    "arm_r_joint3": -0.5260835289955139, "arm_r_joint4": -1.8550262451171875,
    "arm_r_joint5": -0.3125845491886139, "arm_r_joint6": 0.015327823348343372,
    "arm_r_joint7": 0.9495005011558533, "lift_joint": 0.0,
}


def T(R_, p):
    m = np.eye(4); m[:3, :3] = R_; m[:3, 3] = p
    return m


def qm(qq):
    w = qq.GetReal(); x, y, z = qq.GetImaginary()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_rot(axis, q):
    a = np.asarray(axis, float); n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def build(stage):
    by_child, info = {}, {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        p0 = np.array(j.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0))
        q0 = j.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        p1 = np.array(j.GetLocalPos1Attr().Get() or Gf.Vec3f(0, 0, 0))
        q1 = j.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        at = prim.GetAttribute("physics:axis")
        axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}.get(
            str(at.Get()) if at else "X", [1, 0, 0])
        jtype = ("prismatic" if prim.IsA(UsdPhysics.PrismaticJoint)
                 else "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "fixed")
        by_child[b1[0].name] = b0[0].name
        info[b1[0].name] = (prim.GetName(), jtype, T(qm(q0), p0), T(qm(q1), p1), axis)
    return by_child, info


def fk(by_child, info, target, joints, root="base_link"):
    chain, cur = [], target
    while cur in by_child and cur != root:
        chain.append(cur); cur = by_child[cur]
    chain.reverse()
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
    return m


def main() -> int:
    stage = Usd.Stage.Open(USD)
    by_child, info = build(stage)

    ee = fk(by_child, info, "ee_link1", C2)
    r2 = fk(by_child, info, "gripper_r_rh_p12_rn_r2", C2)
    l2 = fk(by_child, info, "gripper_r_rh_p12_rn_l2", C2)

    mid_w = 0.5 * (r2[:3, 3] + l2[:3, 3])
    gap_w = r2[:3, 3] - l2[:3, 3]

    # Into ee_link1's own frame.
    Rt = ee[:3, :3].T
    mid_e = Rt @ (mid_w - ee[:3, 3])
    gap_e = Rt @ gap_w

    print(f"[pinch] ee_link1 (base frame)      ({ee[0,3]:+.4f}, {ee[1,3]:+.4f}, {ee[2,3]:+.4f})")
    print(f"[pinch] right pad                  ({r2[0,3]:+.4f}, {r2[1,3]:+.4f}, {r2[2,3]:+.4f})")
    print(f"[pinch] left pad                   ({l2[0,3]:+.4f}, {l2[1,3]:+.4f}, {l2[2,3]:+.4f})")
    print()
    print(f"[pinch] PAD MIDPOINT in ee_link1 frame  ({mid_e[0]:+.4f}, {mid_e[1]:+.4f}, "
          f"{mid_e[2]:+.4f})   |offset| = {np.linalg.norm(mid_e):.4f} m")
    print(f"[pinch] JAW GAP VECTOR in ee_link1 frame ({gap_e[0]:+.4f}, {gap_e[1]:+.4f}, "
          f"{gap_e[2]:+.4f})   width = {np.linalg.norm(gap_e):.4f} m")
    ax = "XYZ"[int(np.argmax(np.abs(gap_e)))]
    print(f"[pinch] the jaws open along ee_link1's {'+-'[gap_e[np.argmax(np.abs(gap_e))] < 0]}{ax} axis")
    print()
    if np.linalg.norm(mid_e) > 0.02:
        print(f"[pinch] VERDICT: the pinch point is {np.linalg.norm(mid_e):.4f} m from ee_link1. "
              f"A target authored FOR ee_link1 puts the pads that far off the object.")
        return 1
    print("[pinch] VERDICT: the pinch point is essentially at ee_link1; the offset is not the bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
