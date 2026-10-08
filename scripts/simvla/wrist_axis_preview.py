"""Diagnostic camera-local rotations; never changes the production camera config."""
import copy
import math


def local_quarter_turn(quaternion, axis):
    """Compose wxyz mount orientation with +90 degrees about a camera-local axis."""
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be x, y, or z")
    w, x, y, z = quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("invalid camera quaternion")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    c = math.sqrt(.5)
    a, b, d = (c if axis == name else 0. for name in ("x", "y", "z"))
    return (w*c-x*a-y*b-z*d, w*a+x*c+y*d-z*b,
            w*b-x*d+y*c+z*a, w*d+x*b-y*a+z*c)


def add_wrist_axis_previews(scene):
    rotations = {}
    for side in ("left", "right"):
        source = getattr(scene, f"wrist_{side}")
        for axis in ("x", "y", "z"):
            camera = copy.deepcopy(source)
            name = f"wrist_{side}_{axis}90"
            camera.prim_path = source.prim_path + f"_{axis}90"
            camera.offset.rot = local_quarter_turn(source.offset.rot, axis)
            setattr(scene, name, camera)
            rotations[name] = list(camera.offset.rot)
    return rotations
