import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_terminal_snapshot_selects_requested_environments_and_skips_initial_reset():
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "TerminalStateRecorder")
    namespace = {"RecorderTerm": object, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    recorder = namespace[node.name]()
    state = {"rigid_object": {"mug0": {"root_pose": torch.arange(21).reshape(3, 7)}}}
    recorder._env = SimpleNamespace(
        episode_length_buf=torch.tensor([0, 10, 20]),
        scene=SimpleNamespace(get_state=lambda **kwargs: state))
    assert recorder.record_pre_reset([0]) == (None, None)
    key, value = recorder.record_pre_reset([2, 1])
    assert key == "terminal_state"
    assert value["rigid_object"]["mug0"]["root_pose"].tolist() == [list(range(14, 21)), list(range(7, 14))]
