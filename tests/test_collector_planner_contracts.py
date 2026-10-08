"""Import-free guards for collector code that otherwise boots Isaac on import."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("lateral,stage,suppress", [
    (None, 0, False),  # A drawer/ordinary reach must correct orientation.
    (0.12, 0, True),  # Only the initial staged mug translation holds orientation.
    (0.0, 1, False),
    (None, 3, False),
])
def test_left_rotation_hold_is_limited_to_active_mug_staging(lateral, stage, suppress):
    source = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(source.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and "arm_l_pregrasp_stage" in ast.unparse(n.test)
                  and any(isinstance(child, ast.Assign)
                          and "pose_L[env_idx, 3:6]" in ast.unparse(child)
                          for child in n.body))
    condition = compile(ast.Expression(branch.test), str(source), "eval")
    assert bool(eval(condition, {"_pregrasp_lateral_l": lateral,
                                 "arm_l_pregrasp_stage": [stage], "env_idx": 0})) is suppress


def test_collector_never_replaces_curobo_retract_tensor_with_a_python_list():
    source = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(source.read_text())
    assignments = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        assignments.extend(target for target in targets
                           if isinstance(target, ast.Attribute) and target.attr == "retract_config")
    assert assignments == [], "Use the plan JointState; keep cuRobo retract tensor identity/shape intact"


def test_both_failed_lifts_reset_before_the_next_physics_step():
    source = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(source.read_text())
    failures = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "add"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "post_lift_failed_envs"]
    assert len(failures) == 2
    reset = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "post_lift_failed_envs")
    block = ast.unparse(reset)
    assert "reason='post_lift_object_not_retained'" in block
    for name in ("finished_mask", "pose_L", "pose_R", "delta_pose_base", "env_lift_r", "env_lift_l"):
        assert f"{name}[failed_lift_ids] = " in block
    physics_step = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "step"
                        and n.lineno > reset.lineno)
    assert max(n.lineno for n in failures) < reset.lineno < physics_step.lineno
