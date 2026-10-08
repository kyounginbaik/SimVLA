"""Stage-local sink cavity repair; downloaded USDs are never modified."""
from __future__ import annotations

import os
import copy


def validate_sink_goal_environment(metadata):
    """Never execute a cavity diagnostic under the legacy collision/gate recipe."""
    if "diagnostic_sink_cavity" not in metadata:
        return
    for name in ("SIMVLA_SINK_CAVITY_COLLISIONS", "SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"):
        if os.environ.get(name, "0") != "1":
            raise ValueError(f"Sink cavity diagnostic requires {name}=1")


def kitchen813_interior_spec(spec):
    """Replace the legacy proximity leaf, preserving home and script conditions.

    This measured region is kitchen 813/00 only. The cabinet-root XY disk is
    inside the basin bounds; the root-height band excludes countertop and floor.
    This is root placement, not full object-mesh containment.
    """
    result = copy.deepcopy(spec)
    if not isinstance(result, dict) or not isinstance(result.get("all"), list):
        raise ValueError("Expected composed sink all-of success predicate")
    leaves = result["all"]
    candidates = [leaf for leaf in leaves if "obj_near_prim" in leaf]
    if (len(candidates) != 1
            or candidates[0]["obj_near_prim"].get("target_role") != "sink_cabinet"
            or candidates[0]["obj_near_prim"].get("role") != "mug0"):
        raise ValueError("Expected kitchen-813 mug/sink proximity predicate")
    leaves.remove(candidates[0])
    leaves[:] = [leaf for leaf in leaves if leaf != {"obj_z": {"role": "mug0", "lo": .73, "hi": .83}}]
    leaves.extend([
        {"obj_near_prim": {"role": "mug0", "target_role": "sink_cabinet",
                           "anchor": "body", "radius": .12, "xy_only": True}},
        {"obj_z": {"role": "mug0", "lo": .73, "hi": .83}},
    ])
    return result


def repair_sink_collisions(stage, kitchen_root):
    """Retain physical shape colliders, removing redundant aggregate Xform hulls.

    Both changes are necessary: mesh decomposition alone leaves a parent hull
    across the basin; removing only the parent leaves convex child hulls.
    Only named sink cabinetry is touched. Articulation and rigid-body APIs,
    analytic box colliders, doors and physical materials remain intact.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(str(kitchen_root))
    if not root:
        raise ValueError(f"Kitchen prim does not exist: {kitchen_root}")
    changes = {"decomposed_meshes": [], "removed_xform_collision_apis": []}
    for cabinet in Usd.PrimRange(root):
        if cabinet.GetName() != "sink_cabinet":
            continue
        corpus = cabinet.GetChild("corpus")
        if not corpus:
            raise ValueError(f"Sink cabinet has no corpus: {cabinet.GetPath()}")
        for name in ("sink", "sink_countertop"):
            mesh = corpus.GetChild(name)
            if not mesh or not mesh.IsA(UsdGeom.Mesh) or not mesh.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f"Expected a collidable sink mesh: {corpus.GetPath()}/{name}")
        for prim in Usd.PrimRange(cabinet):
            if prim.IsA(UsdGeom.Xform) and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                changes["removed_xform_collision_apis"].append(str(prim.GetPath()))
        for name in ("sink", "sink_countertop"):
            prim = corpus.GetChild(name)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
            changes["decomposed_meshes"].append(str(prim.GetPath()))
    return changes


def configure_sink_collisions(cfg, value=None, interior_gate=None):
    """Opt in before scene construction, identically in collection and replay."""
    value = os.environ.get("SIMVLA_SINK_CAVITY_COLLISIONS", "0") if value is None else value
    interior_gate = (os.environ.get("SIMVLA_KITCHEN813_SINK_INTERIOR_GATE", "0")
                     if interior_gate is None else interior_gate)
    if value not in ("0", "1"):
        raise ValueError("SIMVLA_SINK_CAVITY_COLLISIONS must be 0 or 1")
    if interior_gate not in ("0", "1"):
        raise ValueError("SIMVLA_KITCHEN813_SINK_INTERIOR_GATE must be 0 or 1")
    if interior_gate == "1":
        if value != "1" or not cfg.scene.kitchen.spawn.usd_path.endswith("/kitchen_813_00.usd"):
            raise ValueError("Interior gate requires repaired kitchen 813 rotation 00")
        cfg.terminations.success.params["spec"] = kitchen813_interior_spec(cfg.terminations.success.params["spec"])
    if value == "0":
        return
    spawn = cfg.scene.kitchen.spawn
    original = spawn.func
    if getattr(original, "_simvla_sink_cavity", False):
        return

    def spawn_with_cavity(*args, **kwargs):
        import omni.usd
        import json
        result = original(*args, **kwargs)
        # Isaac's spawn function may have cloned multiple environment roots.
        stage = omni.usd.get_context().get_stage()
        roots = [prim.GetPath() for prim in stage.Traverse()
                 if prim.GetName() == "Kitchen" and str(prim.GetPath()).startswith("/World/envs/")]
        if not roots:
            raise RuntimeError("Sink repair found no spawned Kitchen roots")
        for root in roots:
            changes = repair_sink_collisions(stage, root)
            print("[sink-cavity] " + json.dumps(changes), flush=True)
        return result

    spawn_with_cavity._simvla_sink_cavity = True
    spawn.func = spawn_with_cavity
