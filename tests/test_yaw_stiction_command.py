import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def controller(minimum):
    path = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "_shaped_yaw")
    namespace = {"torch": torch, "_yaw_kp": 1., "_YAW_W_MAX": .463,
                 "_yaw_min_speed": minimum, "_yaw_slew": .046,
                 "prev_yaw_cmd": torch.zeros(3)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def test_minimum_turning_speed_is_signed_slew_limited_and_zero_at_zero_error():
    step = controller(.08)
    rows = torch.arange(3)
    errors = torch.tensor([.025, -.025, 0.])
    assert step(errors, rows).tolist() == pytest.approx([.046, -.046, 0.])
    assert step(errors, rows).tolist() == pytest.approx([.08, -.08, 0.])
    assert step(errors, rows).tolist() == pytest.approx([.08, -.08, 0.])


def test_default_keeps_existing_proportional_command():
    step = controller(0.)
    assert step(torch.tensor([.025, -.025, 0.]), torch.arange(3)).tolist() == pytest.approx([.025, -.025, 0.])
