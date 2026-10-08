import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
SPEC = importlib.util.spec_from_file_location("sphere_fit", Path(__file__).resolve().parents[1] /
                                            "scripts/tools/collision_sphere_fit.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_grid_covers_surface_and_interior_after_yaml_rounding():
    bounds = np.array([[-.06, -.04, -.03], [.14, .04, .02]])
    spheres = module.fit_box_grid(bounds)
    centres = np.round(np.array([c for c, r in spheres]), 5)
    radii = np.round(np.array([r for c, r in spheres]), 5)
    points = np.random.default_rng(0).uniform(bounds[0], bounds[1], (10000, 3))
    corners = np.array(list(__import__("itertools").product(*zip(*bounds))))
    points = np.concatenate([points, corners])
    covered = (np.linalg.norm(points[:, None] - centres[None], axis=2) <= radii).any(axis=1)
    assert covered.all()
    assert radii.max() < .031


@pytest.mark.parametrize("points", [[], [[float("nan"), 0, 0]], [[1, 2]]])
def test_rejects_invalid_points(points):
    with pytest.raises(ValueError, match="points"):
        module.fit_box_grid(points)


def test_rejects_unbounded_grid():
    with pytest.raises(ValueError, match="4096"):
        module.fit_box_grid([[0, 0, 0], [100, 100, 100]])


def test_convex_grid_covers_solid_without_filling_entire_box():
    pytest.importorskip("scipy")
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
    spheres = module.fit_convex_grid(vertices, .01)
    assert len(spheres) < len(module.fit_box_grid(vertices, .01)) / 2
    weights = np.random.default_rng(1).dirichlet(np.ones(4), 10000)
    points = np.concatenate([weights @ vertices, vertices])
    centers = np.round(np.array([c for c, r in spheres]), 5)
    radii = np.round(np.array([r for c, r in spheres]), 5)
    assert (np.linalg.norm(points[:, None] - centers[None], axis=2) <= radii).any(axis=1).all()
