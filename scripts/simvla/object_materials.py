"""Per-instance materials for the OBJECTS a kitchen places, as opposed to its fixtures.

WHY THIS IS ITS OWN MODULE, AND WHY IT IMPORTS NOTHING HEAVY.

The obvious home for this was goal_generator, next to MATERIALS and create_preview_surface_material.
That is what was tried, and it broke every kitchen build twice:

  * imported at MODULE scope in kitchen_scene_generator, it pulls in goal_generator, which imports
    Isaac at module scope -- so Isaac loads BEFORE AppLauncher runs and the build dies at ~10 s
    with "omni.kit.numpy.common: initialization failed";
  * moved to a LAZY import inside apply_materials, it instead imports goal_generator from INSIDE a
    running Isaac app, and the build hangs indefinitely at commit_kitchen having printed
    "placement gate clean".

Both failures present identically from the outside: the SLURM job sits in RUNNING for its whole
walltime while producing nothing. So this module depends on `pxr` and the standard library only,
and can be imported from anywhere at any time.

WHAT IT IS FOR. GEOMETRY2MATERIAL covers cabinets, walls, floors, countertops, sinks and handles,
but NOT the objects placed on them: a mug or an apple renders with whatever material its BODex mesh
carries, identical in every kitchen ever generated. For sim-to-real that is backwards -- the one
thing a policy must locate is the one thing that never varies.
"""
from __future__ import annotations

import random

#: Per-object-type colour distribution, keyed by the type scene_spec uses.
#:
#: An apple is "red, but not the SAME red twice". The jitter is deliberately ASYMMETRIC: red swings
#: widest so an apple can read orange-ish or deep crimson, while green and blue stay tight so it
#: never washes out to pink-grey. Roughness varies proportionally more than colour because specular
#: highlight is what a camera keys on at 320x240, which is the resolution the datasets are recorded
#: at.
OBJECT_MATERIALS: dict[str, dict] = {
    "apple": {
        "base": (0.72, 0.09, 0.07),
        "jitter": (0.14, 0.07, 0.05),
        "metallic": (0.0, 0.05),
        "roughness": (0.28, 0.55),
    },
}


def _preview_surface(stage, mat_path, base_color, metallic, roughness):
    """A UsdPreviewSurface material. Duplicated from goal_generator rather than imported, because
    importing that module is exactly what this file exists to avoid.

    pxr is imported HERE, not at module scope: it only exists inside Isaac's bundled Python, and a
    module-level import would make this file unimportable on a laptop -- which would defeat the
    point of sample_color being testable without a GPU."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def sample_color(obj_type: str, seed=None):
    """The colour this instance would get, without touching a stage. Exists so the distribution is
    testable on a laptop -- no USD stage, no Isaac, no GPU."""
    spec = OBJECT_MATERIALS.get(obj_type)
    if spec is None:
        return None
    rng = random.Random(seed)
    base, jit = spec["base"], spec["jitter"]
    color = tuple(max(0.0, min(1.0, base[i] + rng.uniform(-jit[i], jit[i]))) for i in range(3))
    return color, rng.uniform(*spec["metallic"]), rng.uniform(*spec["roughness"])


def create_random_object_material(stage, mat_path, obj_type, seed=None):
    """A per-instance material for a placed object.

    Returns None for a type with no entry, which is the signal to leave the mesh's own material
    alone -- the behaviour every object had before this existed, and still has today for
    everything except the apple.
    """
    sampled = sample_color(obj_type, seed)
    if sampled is None:
        return None
    color, metallic, roughness = sampled
    return _preview_surface(stage, mat_path, color, metallic, roughness)
