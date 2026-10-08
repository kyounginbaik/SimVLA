"""Planner and jog must not silently choose different standoff distances."""
import ast
from pathlib import Path


def test_right_approach_reads_standoff_once_and_shares_it_with_jog():
    source = (Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py").read_text()
    tree = ast.parse(source)
    assignments = {
        target.id: node.value
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    configured = assignments["_right_pregrasp_standoff"]
    assert isinstance(configured, ast.Call)
    assert configured.func.id == "bounded_float_env"
    assert [arg.value for arg in configured.args] == ["SIMVLA_PREGRASP_STANDOFF", 0., 0., .3]
    assert assignments["_PREGRASP_STANDOFF_M"].id == "_right_pregrasp_standoff"
    jog = assignments["GRASP_STANDOFF_M"]
    assert isinstance(jog, ast.IfExp)
    assert jog.body.id == "_right_pregrasp_standoff"
    assert jog.orelse.value == 0.
