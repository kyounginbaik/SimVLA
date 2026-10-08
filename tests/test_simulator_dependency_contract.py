"""Keep editable-install metadata aligned with the tested raw simulator stack."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('package', ['isaaclab', 'isaaclab_tasks', 'isaaclab_rl'])
def test_raw_simulator_torch_pair_matches_snapshot(package):
    tree = ast.parse((ROOT / 'source' / package / 'setup.py').read_text())
    requirements = next(ast.literal_eval(n.value) for n in tree.body
                        if isinstance(n, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'INSTALL_REQUIRES' for t in n.targets))
    assert 'torch==2.5.1' in requirements
    if package != 'isaaclab':
        assert 'torchvision==0.20.1' in requirements
    snapshot = (ROOT / 'env/simulator-pip.txt').read_text().splitlines()
    assert 'torch==2.5.1' in snapshot
    assert 'torchvision==0.20.1' in snapshot


def test_public_raw_recipe_includes_replay_reader():
    requirements = (ROOT / 'env/simulator-raw.in').read_text().splitlines()
    assert 'datasets' in requirements
    assert 'pyarrow' in requirements
    replay = ast.parse((ROOT / 'scripts/simvla/simvla_replay.py').read_text())
    preflight = next(i for i, n in enumerate(replay.body)
                     if isinstance(n, ast.If) and 'find_spec' in ast.unparse(n.test))
    launcher = next(i for i, n in enumerate(replay.body)
                    if isinstance(n, ast.ImportFrom) and n.module == 'isaaclab.app')
    assert preflight < launcher


def test_public_raw_recipe_includes_geometry_unwrapping_dependency():
    requirements = (ROOT / 'env/simulator-raw.in').read_text().splitlines()
    constraints = (ROOT / 'env/simulator-tested-constraints.txt').read_text().splitlines()
    assert 'xatlas' in requirements
    assert 'xatlas==0.0.11' in constraints
