import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
SPEC = importlib.util.spec_from_file_location("collision_mesh", Path(__file__).resolve().parents[1] /
                                            "scripts/simvla/collision_mesh.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("face", [[0, 1, 2, 3], [3, 2, 1, 0]])
def test_quad_preserves_winding_and_area(face):
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = module.triangle_indices(points, [4], face)
    assert len(result) == 2
    xyz = points[result]
    areas = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])[:, 2] / 2
    assert np.abs(areas).sum() == 1
    assert (areas > 0).all() if face[0] == 0 else (areas < 0).all()


def test_mixed_faces_and_usd_holes():
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    assert module.triangle_indices(points, [4, 3], [0, 1, 2, 3, 0, 1, 2], [0]) == [(0, 1, 2)]


def test_concave_polygon_fails_instead_of_covering_exterior():
    points = [[0, 0, 0], [1, 0, 0], [.2, .2, 0], [0, 1, 0]]
    with pytest.raises(ValueError, match="concave"):
        module.triangle_indices(points, [4], [0, 1, 2, 3])


@pytest.mark.parametrize("counts,indices,message", [([4], [0, 1, 2], "does not match"),
    ([3], [0, 1, 9], "out of range"), ([2], [0, 1], "at least three")])
def test_invalid_topology_fails(counts, indices, message):
    with pytest.raises(ValueError, match=message):
        module.triangle_indices([[0, 0, 0], [1, 0, 0], [0, 1, 0]], counts, indices)
