"""Production batch entrypoints must fail visibly, never open a debugger."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", [
    "scripts/simvla/simvla_gen.py", "scripts/simvla/simvla_replay.py",
    "scripts/simvla/simvla_data_generator.py", "scripts/simvla/task_emit.py",
    "source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py",
])
def test_no_unconditional_debugger_traps(name):
    path = Path(__file__).parents[1] / name
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == "breakpoint")
            assert not (isinstance(node.func, ast.Attribute) and node.func.attr == "set_trace")
