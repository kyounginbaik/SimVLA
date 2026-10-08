"""Conservative box-cell sphere cover for broad, asymmetric robot links."""
import math

import numpy as np


def fit_box_grid(points, max_cell_m=0.035):
    """Cover the entire vertex AABB, including triangle interiors, with spheres.

    A long camera bracket attached to a flat wrist is poorly represented by a few
    large principal-axis spheres. Each grid sphere contains its complete box cell;
    no empty cells are removed, so the cover remains conservative between vertices.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points must be a nonempty finite N-by-3 array")
    if not math.isfinite(max_cell_m) or max_cell_m <= 0:
        raise ValueError("max_cell_m must be finite and positive")
    lo, hi = points.min(axis=0), points.max(axis=0)
    counts = np.maximum(1, np.ceil((hi - lo) / max_cell_m).astype(int))
    if math.prod(int(n) for n in counts) > 4096:
        raise ValueError("sphere grid exceeds 4096 cells; check units or increase cell size")
    widths = (hi - lo) / counts
    # Cover five-decimal YAML rounding of both centers and radii.
    radius = max(1e-5, float(np.linalg.norm(widths) / 2) + 2e-5)
    return [(lo + (np.array(index) + 0.5) * widths, radius)
            for index in np.ndindex(tuple(counts))]


def fit_convex_grid(points, max_cell_m=0.015):
    """Conservatively cover a vertex convex hull, including its solid interior.

    Discard a cell only if an entire cell lies outside a hull supporting plane.
    This retains every intersecting cell (and possibly extra cells), unlike a
    nearest-vertex test which can miss long triangle interiors. Useful for diagonal
    finger linkages whose full AABB incorrectly fills the open grasp volume.
    """
    from scipy.spatial import ConvexHull, QhullError

    cells = fit_box_grid(points, max_cell_m)
    points = np.asarray(points, dtype=float)
    extent = points.max(axis=0) - points.min(axis=0)
    counts = np.maximum(1, np.ceil(extent / max_cell_m).astype(int))
    half_width = extent / counts / 2
    try:
        equations = ConvexHull(points).equations
    except QhullError:
        return cells  # Flat/degenerate geometry still gets its conservative box.
    normal, offset = equations[:, :3], equations[:, 3]
    support = np.abs(normal) @ half_width
    return [(centre, radius) for centre, radius in cells
            if np.all(normal @ centre + offset - support <= 1e-10)]
