"""Preserve USD polygon surfaces when using cuRobo's triangle-only reader.

No USD layer is modified. Convex polygons use a winding-preserving fan; concave
or degenerate polygons fail explicitly instead of silently dropping an obstacle.
"""
from __future__ import annotations

import numpy as np


def triangle_indices(points, counts, indices, holes=()):
    points = np.asarray(points, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("mesh points must be finite Nx3 coordinates")
    if counts.ndim != 1 or indices.ndim != 1 or (counts < 3).any():
        raise ValueError("mesh faces require at least three vertices")
    if int(counts.sum()) != len(indices):
        raise ValueError("faceVertexCounts does not match faceVertexIndices")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
        raise ValueError("mesh vertex index out of range")
    holes = set(int(h) for h in holes)
    if any(h < 0 or h >= len(counts) for h in holes):
        raise ValueError("mesh hole index out of range")
    triangles = []
    start = 0
    for face_id, count in enumerate(counts):
        face = indices[start:start + count]
        start += count
        if face_id in holes:
            continue
        if count > 3:
            polygon = points[face]
            # Newell normal and signed corner turns detect a concave polygon.
            # Never create a fan over its exterior and call it exact geometry.
            edges = np.roll(polygon, -1, axis=0) - polygon
            normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
            magnitude = np.linalg.norm(normal)
            if magnitude < 1e-14:
                raise ValueError(f"degenerate polygon at face {face_id}")
            turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ (normal / magnitude)
            if (turns < -1e-10).any():
                raise ValueError(f"concave polygon at face {face_id}; triangulate this asset first")
        triangles.extend((int(face[0]), int(face[k]), int(face[k + 1]))
                         for k in range(1, count - 1))
    return triangles


def install_polygon_mesh_reader(usd_helper):
    """Wrap the module's reader once; pose extraction still receives the real prim."""
    if getattr(usd_helper.get_mesh_attrs, "_simvla_polygons", False):
        return
    original = usd_helper.get_mesh_attrs

    def read(prim, cache=None, transform=None):
        counts = list(prim.GetAttribute("faceVertexCounts").Get() or ())
        holes = list(prim.GetAttribute("holeIndices").Get() or ())
        if all(n == 3 for n in counts) and not holes:
            return original(prim, cache=cache, transform=transform)
        try:
            triangles = triangle_indices(
                prim.GetAttribute("points").Get(), counts,
                prim.GetAttribute("faceVertexIndices").Get(), holes)
        except ValueError as exc:
            raise ValueError(f"Collision mesh {prim.GetPath()}: {exc}") from exc
        if not triangles:
            return None  # An explicitly all-hole surface has no collision faces.
        # Keep cuRobo's real-prim pose/scale extraction; do not mutate the USD
        # stage or pass a Python proxy through the USD C++ boundary.
        matrix, scale = usd_helper.get_prim_world_pose(cache, prim)
        if transform is not None:
            matrix = transform @ matrix
        tensor = usd_helper.torch.as_tensor(matrix, device=usd_helper.torch.device("cuda", 0))
        pose = usd_helper.Pose.from_matrix(tensor).tolist()
        return usd_helper.Mesh(name=str(prim.GetPath()), pose=pose,
                               vertices=[np.ravel(p) for p in prim.GetAttribute("points").Get()],
                               faces=triangles, scale=scale)

    read._simvla_polygons = True
    usd_helper.get_mesh_attrs = read
