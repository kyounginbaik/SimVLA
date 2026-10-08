"""Shared rotation/orientation helpers used by SimVLA scripts and env code.

Consolidates conversions that were previously copy-pasted across
``scripts/simvla/real2sim.py`` and the Real2Sim termination module.
Kept dependency-light (only torch / numpy / scipy.spatial.transform)
so it can be imported from both script-side and env-side code without
pulling Isaac Sim into the import graph.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from isaaclab.utils.math import quat_inv, quat_mul


def sixd_to_quat_wxyz(
    rot6d: torch.Tensor,
    eps: float = 1e-8,
    project_svd: bool = True,
) -> torch.Tensor:
    """6D rotation representation → unit quaternion (w, x, y, z).

    ``rot6d`` is interpreted as ``[c1; c2]`` (first two columns of the
    rotation matrix, flattened). Returns the closest right-handed
    rotation as a wxyz quaternion with non-negative w.
    """
    in_device = rot6d.device
    in_dtype = rot6d.dtype

    x = rot6d.detach().cpu().numpy().astype(np.float64)
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, 6)

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    r1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + eps)
    dot = np.sum(r1 * a2, axis=1, keepdims=True)
    b_ortho = a2 - r1 * dot

    bad = np.linalg.norm(b_ortho, axis=1) < eps
    if np.any(bad):
        tmp = np.tile(np.array([1.0, 0.0, 0.0]), (x.shape[0], 1))
        use_alt = np.abs(r1[:, 0]) > 0.9
        tmp[use_alt] = np.array([0.0, 1.0, 0.0])
        b_ortho[bad] = tmp[bad] - r1[bad] * np.sum(r1[bad] * tmp[bad], axis=1, keepdims=True)

    r2 = b_ortho / (np.linalg.norm(b_ortho, axis=1, keepdims=True) + eps)
    r3 = np.cross(r1, r2)

    M = np.stack([r1, r2, r3], axis=-1)

    if project_svd:
        Rm = np.empty_like(M)
        for i in range(M.shape[0]):
            U, _, Vt = np.linalg.svd(M[i])
            Ri = U @ Vt
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1
                Ri = U @ Vt
            Rm[i] = Ri
    else:
        Rm = M

    q_xyzw = R.from_matrix(Rm).as_quat()
    q_wxyz = np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=-1)
    neg = q_wxyz[:, 0] < 0
    q_wxyz[neg] *= -1
    q_wxyz = q_wxyz.reshape(*orig_shape, 4)
    return torch.as_tensor(q_wxyz, device=in_device, dtype=in_dtype)


def quat_delta_axis_angle(
    q_t: torch.Tensor,
    q_tp1: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Shortest-path axis-angle vector taking ``q_t`` to ``q_tp1``.

    Inputs are (..., 4) unit quaternions in (w, x, y, z); output is
    (..., 3) rotation vector whose length equals the rotation angle.
    """
    q_t = F.normalize(q_t, dim=-1)
    q_tp1 = F.normalize(q_tp1, dim=-1)

    q_delta = quat_mul(q_tp1, quat_inv(q_t))

    mask = q_delta[..., 0] < 0
    q_delta[mask] = -q_delta[mask]

    w = q_delta[..., 0].clamp(-1.0, 1.0)
    xyz = q_delta[..., 1:]

    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(1.0 - w * w).clamp_min(eps)
    axis = xyz / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def rot6d_to_R(rot6d, device=None, dtype=None) -> torch.Tensor:
    """6D rotation representation → (B, 3, 3) rotation matrix (columns)."""
    rot6d = torch.as_tensor(rot6d, device=device, dtype=dtype)
    rot6d = rot6d.reshape(-1, 6)
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    b1 = F.normalize(a1, dim=-1)
    a2_ortho = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_ortho, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def to_world(q_A_world: torch.Tensor, q_B_local: torch.Tensor) -> torch.Tensor:
    """Compose a local quaternion onto a world-frame parent."""
    q_A_world = F.normalize(q_A_world, dim=-1)
    q_B_local = F.normalize(q_B_local, dim=-1)
    q_B_world = quat_mul(q_A_world, q_B_local)
    return F.normalize(q_B_world, dim=-1)


def quat_angle_error_rad(
    q1_wxyz: torch.Tensor,
    q2_wxyz: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Geodesic angle (radians) between two wxyz quaternions."""
    q1 = F.normalize(q1_wxyz, dim=-1, eps=eps)
    q2 = F.normalize(q2_wxyz, dim=-1, eps=eps)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def quat_mul_wxyz_np(q1, q2) -> np.ndarray:
    """Hamilton product of two wxyz quaternions, numpy (4,)/(4,) → (4,)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def axis_angle_to_quat_wxyz(axis, angle_rad: float) -> np.ndarray:
    """Axis-angle (axis (3,), angle in radians) → wxyz quaternion (4,)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle_rad
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def rot6d_to_rotmat(rot6d) -> np.ndarray:
    """Numpy variant of :func:`rot6d_to_R` — (6,) → (3,3) rotation matrix."""
    rot6d = np.asarray(rot6d, dtype=float).reshape(6)
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2_ortho = a2 - np.dot(b1, a2) * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho) + 1e-12)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)


def compose_base_world_and_eef_local(base_xy_qwxyz, eef_xyz_rot6d):
    """Compose a planar base pose with a 6D EEF-in-base pose into world frame.

    ``base_xy_qwxyz``: (6,) = [x, y, qw, qx, qy, qz] — base pose in world,
    z assumed 0.
    ``eef_xyz_rot6d``: (9,) = [x, y, z, r6d0..r6d5] — EEF pose expressed in
    the base frame.

    Returns (p_WE, q_WE_wxyz, R_WE): world-frame EEF position (3,),
    wxyz quaternion (4,), and scipy ``Rotation`` for downstream composition.
    """
    base_xy_qwxyz = np.asarray(base_xy_qwxyz, dtype=float).reshape(6)
    eef_xyz_rot6d = np.asarray(eef_xyz_rot6d, dtype=float).reshape(9)

    xB, yB, qw, qx, qy, qz = base_xy_qwxyz
    p_WB = np.array([xB, yB, 0.0])
    q_WB_xyzw = np.array([qx, qy, qz, qw])
    R_WB = R.from_quat(q_WB_xyzw)

    p_BE = eef_xyz_rot6d[0:3]
    rot6d = eef_xyz_rot6d[3:9]
    R_BE = R.from_matrix(rot6d_to_rotmat(rot6d))

    p_WE = p_WB + R_WB.apply(p_BE)
    R_WE = R_WB * R_BE

    q_WE_xyzw = R_WE.as_quat()
    q_WE_wxyz = np.array([q_WE_xyzw[3], q_WE_xyzw[0], q_WE_xyzw[1], q_WE_xyzw[2]])

    return p_WE, q_WE_wxyz, R_WE


def translate_in_eef_frame(p_WE, R_WE: R, delta_local) -> np.ndarray:
    """Shift a world-frame position by a delta expressed in the EEF local frame."""
    delta_local = np.asarray(delta_local, dtype=float).reshape(3)
    return p_WE + R_WE.apply(delta_local)
