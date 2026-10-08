"""Pure object size tables + real-world scaling. Stdlib only — importing this boots neither
Omniverse nor numpy, so it is unit-testable and safe to import from the composer core.

Extracted from kitchen_build.py (which re-imports these) so the campaign's width heuristic can
read placed sizes without the build stack. Behavior is identical to the pre-extraction code; the
one numpy call (np.argmax) is replaced by max(range(3), key=...).
"""
from __future__ import annotations

import os

from .scene_spec import GRASPABLE_TYPES, CLUTTER  # noqa: F401

OBJECT_TYPES = list(GRASPABLE_TYPES) + list(CLUTTER)

#: Base (width, depth, height) in metres for each placeable object, before the per-instance size
#: jitter. Graspable dims are set explicitly here (bottle/bowl/mug are goal_generator.py's historical
#: values; apple/sodacan/nutella real-world estimates; fruit measured from its BODex mesh). Clutter
#: dims come from scene_spec.CLUTTER (mesh extents at grasp scale) and are merged in below, so a new
#: clutter object is one entry in CLUTTER, not another line here.
OBJECT_BASE_DIMS: dict[str, tuple[float, float, float]] = {
    "bottle": (0.09, 0.09, 0.10),
    "bowl": (0.08, 0.08, 0.10),
    "mug": (0.06, 0.06, 0.08),        # was None; canonicalized (spec: retire profile mug_height)
    "apple": (0.07, 0.07, 0.07),
    "sodacan": (0.066, 0.066, 0.122),
    "nutella": (0.075, 0.075, 0.10),
    "fruit": (0.108, 0.108, 0.133),   # measured from the sem_Fruit BODex mesh at its grasp scale
}
OBJECT_BASE_DIMS.update({t: c["dims"] for t, c in CLUTTER.items()})


#: Real-world HEIGHT (the up-axis dimension), in metres, per object category. The BODex mesh scale
#: (scale010) is arbitrary, so the raw OBJECT_BASE_DIMS make objects the wrong physical size (a 14.8cm
#: "cup"). Scaling each mesh UNIFORMLY so its up-axis dimension matches this real height gives a
#: realistic absolute size while preserving the mesh's natural aspect (no per-object squishing / no
#: distortion). Grasps adapt automatically — plan_arm_grasp scales the contact offset by placed/mesh.
#: Add or tweak a line here to resize a category; objects with no entry keep their raw mesh dims.
#: CLUTTER (BODex-imported) surface-of-revolution objects only, whose up-axis IS the height — the
#: scaling below keys off that axis, and plan_arm_grasp's offset scaling is gated to CLUTTER. Flat
#: discs and boxes (cookie/plate/cerealbox) would blow up if scaled by their thin axis, and the
#: original graspables (mug/sodacan/...) carry their own grasp data at their own scale, so both are
#: intentionally omitted and keep their raw dims.
REAL_HEIGHT: dict[str, float] = {
    "can": 0.12, "jar": 0.11, "vase": 0.11, "candle": 0.14, "milkcarton": 0.19,
}


def _up_axis_index(dims) -> int:
    """Index of the up (symmetry) axis: the extent that differs from the other two (a surface of
    revolution has a circular cross-section, so two extents are ~equal). Falls back to the tallest
    axis for box-like objects."""
    dx, dy, dz = dims

    def close(a, b):
        return a > 0 and b > 0 and abs(a - b) / max(a, b) < 0.12

    if close(dx, dy):
        return 2
    if close(dx, dz):
        return 1
    if close(dy, dz):
        return 0
    return max(range(3), key=lambda i: dims[i])   # was np.argmax(dims)


def real_placement_dims(obj_type: str, fallback):
    """Absolute placement (w, d, h) that scales the mesh uniformly so its up-axis dimension equals the
    object's real-world height (REAL_HEIGHT), preserving its natural aspect. Returns `fallback` (the
    caller's mesh dims) unchanged when the type has no real height defined."""
    real_h = REAL_HEIGHT.get(obj_type)
    mesh = OBJECT_BASE_DIMS.get(obj_type)
    if real_h is None or not mesh:
        return list(fallback)
    up = _up_axis_index(mesh)
    if mesh[up] <= 0:
        return list(fallback)
    s = real_h / mesh[up]
    dims = [d * s for d in mesh]
    # Optional stability cap: a tall (top-heavy) object on a narrow base tips when the gripper brushes
    # it. SIMVLA_MAX_ASPECT caps height/width — trading a little realism for a graspable object. Unset
    # (the committed default) leaves the true real-world proportions.
    _cap = os.environ.get("SIMVLA_MAX_ASPECT")
    if _cap:
        w = max(dims[i] for i in range(3) if i != up)
        if w > 0 and dims[up] / w > float(_cap):
            dims[up] = w * float(_cap)
    return dims
