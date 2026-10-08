"""Collector and controller must measure the same adaptive-finger surfaces."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("distal,proximal,expected", [
    ((12., 0.), (34., 46.), (46., 46.)),
    ((6., 7.), None, (6., 7.)),
    ((0., 0.), (0., 0.), (0., 0.)),
    (None, None, None),
])
def test_whole_finger_contact_measurement(distal, proximal, expected):
    path = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == "_mug_finger_contact_forces")
    def measure(env_id, arm, **kwargs):
        return proximal if kwargs.get("proximal") else distal
    namespace = {"_mug_pad_contact_forces": measure}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace[node.name](0, "l") == expected
