"""Replay must not cross an automatic environment reset inside one recorded pass."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "scripts/simvla/simvla_replay.py"


@pytest.mark.parametrize("reset,exhausted,expected", [
    (False, False, False), (True, False, True),
    (False, True, True), (True, True, True),
])
def test_failed_reset_ends_pass_before_remaining_actions(reset, exhausted, expected):
    tree = ast.parse(PATH.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "failed_reset" for t in n.targets))
    boundary = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.BoolOp)
                    and ast.unparse(n.test).startswith("failed_reset or"))
    namespace = {
        "torch": SimpleNamespace(any=lambda value: SimpleNamespace(item=lambda: value)),
        "env": SimpleNamespace(reset_buf=reset), "start_idx": int(exhausted), "end_idx": 0,
        "settle_steps": 0,
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(PATH), "exec"), namespace)
    assert eval(compile(ast.Expression(boundary.test), str(PATH), "eval"), namespace) is expected


def test_success_selection_excludes_failed_resets():
    tree = ast.parse(PATH.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "success_env" for t in n.targets))
    assert ast.unparse(assignment.value) == "env.reset_buf & env.termination_manager.get_term('success')"


@pytest.mark.parametrize("start_idx,reset,expected", [
    (11, False, False), (12, False, False), (13, False, True), (11, True, True),
])
def test_terminal_settling_is_bounded_and_never_crosses_reset(start_idx, reset, expected):
    tree = ast.parse(PATH.read_text())
    boundary = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.BoolOp)
                    and ast.unparse(n.test).startswith("failed_reset or"))
    assert eval(compile(ast.Expression(boundary.test), str(PATH), "eval"), {
        "failed_reset": reset, "start_idx": start_idx, "end_idx": 10, "settle_steps": 2,
    }) is expected


def test_terminal_hold_does_not_resample_vqa_or_repeat_base_velocity():
    tree = ast.parse(PATH.read_text())
    checks = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.If)]
    assert any(c.startswith("if simvqa_save and (not settling):") and "simvqa(" in c for c in checks)
    assert any(c.startswith("if settling:") and "delta_pose_base_world.zero_()" in c for c in checks)
    assert any(c.startswith("if settling and motor_targets.ndim == 3:")
               and "motor_targets[:, -1, :]" in c for c in checks)


def test_failure_video_keeps_both_wrists_and_control_fps():
    source = PATH.read_text()
    failure = source[source.index('reason = "environment terminated'):]
    assert '("wrist_left", wrist_left_frames)' in failure
    assert 'fps=args_cli.step_hz' in failure
    assert 'fps=20' not in failure
