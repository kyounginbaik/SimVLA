"""Filter the overlapping head/torso housing pair without disabling arm contacts."""


def filter_head_torso(root):
    from pxr import UsdPhysics

    head = root.GetChild("head_link2")
    torso = root.GetChild("arm_base_link")
    if not head.IsValid() or not torso.IsValid():
        raise ValueError(f"Missing AI Worker head/torso links under {root.GetPath()}")
    UsdPhysics.FilteredPairsAPI.Apply(head).CreateFilteredPairsRel().AddTarget(torso.GetPath())


def spawn_aiworker_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import find_matching_prims

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # The head_link2 housing overlaps its grandparent arm_base_link collision
    # hull. Parent-child filtering does not cover this pair. It deflects the
    # head at rest and jitters during base turns (controlled probes, 2026-10-06).
    # Author only on the live stage: public asset bytes stay hash-verifiable.
    roots = find_matching_prims(prim_path)
    if not roots:
        raise ValueError(f"No AI Worker instances under {prim_path}")
    for robot in roots:
        filter_head_torso(robot)
    print(f"[aiworker] head/torso housing filter applied to {len(roots)} robot(s)", flush=True)
    return root
