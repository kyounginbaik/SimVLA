"""SimVLA: move BODex grasp poses into the frame the object is actually placed in.

A BODex grasp is NOT expressed in the mesh's raw file frame. Every grasp .npy records the pose the
object was posed at during synthesis, in `world_cfg[0]["mesh"][<key>]["pose"]` (curobo, wxyz), and
across this dataset it takes exactly two values: a +90 degree turn about X for the core_*/ddg_*
meshes (standing them on +Y) and identity for the sem_* ones (standing them on +Z).

The placer stands the object on mesh_orientation.resolved_up instead. When those two agree -- 347 of
the 357 grasped meshes -- a grasp needs only the kitchen variant's Z yaw, which is what
plan_arm_grasp has always applied. When they disagree the grasp is rotated wrong about the object,
and no amount of yaw fixes it: a plate hand-labelled (0,0,-1) was synthesised at (0,0,1), so its
grasps arrive 180 degrees out.

So the correction is P @ S^-1: out of the synthesis frame, into the placed one. It is exactly the
identity whenever the two frames agree, which is why this can be applied unconditionally rather than
to a list of known-bad meshes -- the list would go stale the next time a label changes.

Pure numpy/scipy: no Omniverse, no USD, so it is testable on a login node.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

#: What a mesh whose grasp file records no usable pose falls back to. Identity means "assume the
#: grasps are already in the placed frame", i.e. exactly the behaviour before this module existed --
#: a malformed file degrades to the old result instead of raising in the middle of goal generation.
IDENTITY = np.eye(3)


def rotation_to_z(up) -> np.ndarray:
    """3x3 rotation taking `up` (an axis-aligned unit vector) onto +Z, by the shortest turn.

    The same rotation kitchen_gallery._stood_up draws with and scene_synthesizer places with, so a
    grasp corrected through here lands on the object as the picker drew it.
    """
    up = np.asarray(up, dtype=float)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    if not np.any(axis):
        if up[2] > 0:
            return IDENTITY.copy()
        axis = np.array([1.0, 0.0, 0.0])        # 180 degrees: any perpendicular axis will do
    angle = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(angle * axis).as_matrix()


def synthesis_rotation(grasp_data) -> np.ndarray:
    """The rotation BODex applied to the mesh before synthesising these grasps.

    Returns IDENTITY when the file does not record one, rather than guessing: see IDENTITY.
    """
    try:
        meshes = grasp_data["world_cfg"][0]["mesh"]
        pose = next(iter(meshes.values()))["pose"]
        # Split rather than np.fromstring: the latter warns (and one day will raise) on text it
        # cannot fully consume, and this has to tolerate junk quietly -- see IDENTITY.
        pose = (np.array([float(v) for v in str(pose).strip("[]").split()])
                if isinstance(pose, str) else np.asarray(pose, dtype=float))
        if pose.shape[-1] < 7:
            return IDENTITY.copy()
        w, x, y, z = pose[3], pose[4], pose[5], pose[6]      # curobo stores wxyz
        norm = float(np.linalg.norm([w, x, y, z]))
        if not np.isfinite(norm) or norm == 0.0:
            return IDENTITY.copy()
        return Rotation.from_quat([x / norm, y / norm, z / norm, w / norm]).as_matrix()
    except (KeyError, IndexError, TypeError, ValueError, StopIteration):
        return IDENTITY.copy()


def object_type_from_prim_name(name):
    """"bottle0" -> "bottle". None when nothing is left to go on.

    The placed up depends on the object TYPE for apple and sodacan (mesh_orientation.TYPE_UP), and
    by the time a grasp is planned the only record of it is the prim's name -- build_kitchen names
    each placed object <type><n>. Getting this wrong matters in exactly one direction: an apple read
    as typeless would take the core_* fallback and earn a spurious 90 degree correction, when its
    frames actually agree.
    """
    if not name:
        return None
    return name.rstrip("0123456789") or None


def correction(placed_up, grasp_data) -> np.ndarray:
    """P @ S^-1 -- the rotation carrying a grasp from its synthesis frame to the placed one."""
    return rotation_to_z(placed_up) @ synthesis_rotation(grasp_data).T


def is_identity(matrix, tol: float = 1e-6) -> bool:
    """Whether a correction is a no-op, i.e. the object is placed as it was synthesised."""
    return bool(np.allclose(matrix, IDENTITY, atol=tol))


def apply(matrix, xyz, quat_wxyz):
    """Rotate grasp positions and orientations by `matrix`. Returns (xyz, quat_wxyz).

    Positions turn about the object's own origin, which is where BODex expressed them and where
    plan_arm_grasp re-anchors them, so no translation is involved. Orientations are PRE-multiplied:
    the gripper turns with the object, not in its own frame.
    """
    xyz = np.asarray(xyz, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    if is_identity(matrix):
        return xyz, quat_wxyz

    turned_xyz = xyz @ np.asarray(matrix, dtype=float).T

    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    turned = Rotation.from_matrix(matrix) * Rotation.from_quat(quat_xyzw)
    out_xyzw = turned.as_quat()
    return turned_xyz, np.hstack([out_xyzw[:, 3:4], out_xyzw[:, 0:3]])
