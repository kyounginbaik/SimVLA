"""Recorded targets remain physical actuator commands, not joint-state writes."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[1]


def manager():
    path = ROOT / "source/isaaclab/isaaclab/managers/action_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActionManager")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"set_joint_position_replay_targets", "apply_action", "reset"}]
    namespace = {"torch": torch, "Sequence": Sequence}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    fake_class = type("Manager", (), {method.name: namespace[method.name] for method in methods})
    obj = fake_class()
    calls = []
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2, 3)),
                            set_joint_position_target=lambda value, **kw: calls.append((value.clone(), kw)))
    obj._env = SimpleNamespace(scene=SimpleNamespace(articulations={"robot": robot}))
    obj.device = "cpu"
    obj._joint_position_replay_targets = None
    obj._joint_position_replay_mask = torch.zeros(2, dtype=torch.bool)
    obj._action = torch.zeros(2, 4)
    obj._prev_action = torch.zeros(2, 4)
    obj._terms = {"normal": SimpleNamespace(apply_actions=lambda: calls.append("normal"), reset=lambda **kw: None)}
    return obj, calls


def test_joint_targets_override_after_normal_commands_without_state_writes():
    obj, calls = manager()
    obj.apply_action()
    assert calls == ["normal"]
    targets = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    obj.set_joint_position_replay_targets(targets)
    targets[:] = 0
    obj.apply_action()
    assert calls[-2] == "normal"
    assert calls[-1][0].tolist() == [[1., 2., 3.], [4., 5., 6.]]
    assert calls[-1][1]["env_ids"].tolist() == [0, 1]
    assert not torch.any(obj._env.scene.articulations["robot"].data.joint_pos)


def test_subset_reset_and_disabling_never_reapply_stale_targets():
    obj, calls = manager()
    obj.set_joint_position_replay_targets(torch.ones(2, 3))
    obj.reset([1])
    obj.apply_action()
    assert calls[-1][1]["env_ids"].tolist() == [0]
    obj.set_joint_position_replay_targets(None)
    calls.clear()
    obj.apply_action()
    assert calls == ["normal"]


@pytest.mark.parametrize("targets", [torch.zeros(2, 2), torch.full((2, 3), float("nan"))])
def test_joint_targets_reject_shape_and_nonfinite_values(targets):
    with pytest.raises(ValueError):
        manager()[0].set_joint_position_replay_targets(targets)


def test_joint_name_alignment_is_explicit():
    path = ROOT / "scripts/simvla/replay_targets.py"
    spec = importlib.util.spec_from_file_location("replay_targets_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    feature = {"shape": [2], "names": {"action.joint": ["b", "a"]}}
    assert mod.joint_target_order(feature, ["a", "b"]) == [1, 0]
    for names in (["a", "a"], ["a", "c"], ["a"]):
        with pytest.raises(ValueError):
            mod.joint_target_order(feature, names)
