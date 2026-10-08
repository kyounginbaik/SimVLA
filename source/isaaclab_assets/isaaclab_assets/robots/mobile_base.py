"""Spawn virtual planar bases without an artificial one-revolution hard stop."""


def spawn_mobile_base_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Usd, UsdPhysics
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import find_matching_prims

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # These joints model the pose of the entire mobile platform, not a physical
    # rotary actuator with a cable/hard stop. In particular the public RBY1 USD
    # authors +/-359.989 degrees here, which stalls a second turn exactly at 2pi.
    # Edit only the in-memory stage, preserving downloaded asset bytes/hashes.
    count = 0
    for robot_prim in find_matching_prims(prim_path):
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() != "base_revolute_z_joint":
                continue
            joint = UsdPhysics.RevoluteJoint(prim)
            if not joint:
                raise ValueError(f"Expected virtual base yaw to be revolute: {prim.GetPath()}")
            joint.CreateLowerLimitAttr(float("-inf"))
            joint.CreateUpperLimitAttr(float("inf"))
            count += 1
    if not count:
        raise ValueError(f"Missing virtual base_revolute_z_joint under {prim_path}")
    print(f"[mobile-base] continuous virtual yaw configured for {count} robot(s)", flush=True)
    return root
