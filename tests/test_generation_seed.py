"""Headless generation seeds geometry AND appearance before either is built."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("seed", ["-1", str(2**32)])
def test_invalid_seed_fails_before_simulator_boot(tmp_path, seed):
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/simvla/generate_example.py"),
        "--kitchen-id", "99109", "--mesh", str(tmp_path / "missing.obj"),
        "--output", str(tmp_path / "unused"), "--seed", seed,
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--seed must be" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_seed_precedes_geometry_and_materials_and_is_recorded():
    tree = ast.parse((ROOT / "scripts/simvla/build_new_mug_kitchen.py").read_text())
    calls = {ast.unparse(n.func): n for n in ast.walk(tree) if isinstance(n, ast.Call)}
    build = calls["build_kitchen"]
    for function in ("random.seed", "np.random.seed"):
        assert calls[function].lineno < build.lineno < calls["pick_materials"].lineno
        assert ast.unparse(calls[function].args[0]) == "SEED"
    assert ast.unparse(next(k.value for k in build.keywords if k.arg == "seed")) == "SEED"
    facts = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "facts" for t in n.targets))
    keys = {k.value for k in facts.keys}
    assert {"seed", "room_shell_seed", "material_picks", "mesh_sha256"} <= keys
