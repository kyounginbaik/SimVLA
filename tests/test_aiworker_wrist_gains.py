import ast
import math
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("value,expected", [(None, 252), ("1000", 1000), ("1500", 1500),
                                           ("251", None), ("1501", None), ("nan", None)])
def test_wrist_gain_is_bounded_and_reference_default_is_preserved(monkeypatch, value, expected):
    key = "SIMVLA_AIWORKER_WRIST_STIFFNESS"
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    path = Path(__file__).parents[1] / "source/isaaclab_assets/isaaclab_assets/robots/aiworker.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.If))
             and key in ast.unparse(n)]
    namespace = {"os": os, "math": math}
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    if expected is None:
        with pytest.raises(ValueError, match="finite and within"):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert namespace["AIWORKER_WRIST_STIFFNESS"] == expected
