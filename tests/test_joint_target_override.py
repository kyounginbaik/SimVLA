"""Exercise motor override methods without importing or launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def manager():
    source = Path(__file__).parents[1] / "source/isaaclab/isaaclab/managers/action_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionManager")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("set_joint_position_replay_targets", "apply_action")]
    namespace = {"torch": torch, "Sequence": Sequence}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    calls = []
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2, 5)),
                            set_joint_position_target=lambda targets, **kwargs: calls.append((targets, kwargs)))
    instance = SimpleNamespace(device="cpu", _terms={},
                              _env=SimpleNamespace(cfg=SimpleNamespace(decimation=3),
                                                   scene=SimpleNamespace(articulations={"robot": robot})),
                              _joint_position_replay_mask=torch.zeros(2, dtype=torch.bool))
    instance.set = lambda targets, **kwargs: namespace["set_joint_position_replay_targets"](instance, targets, **kwargs)
    instance.apply = lambda: namespace["apply_action"](instance)
    return instance, calls


def test_subset_does_not_command_other_joints_or_environments(manager):
    obj, calls = manager
    target = torch.arange(10.).reshape(2, 5)
    obj.set(target, joint_ids=[3, 1], env_ids=[1])
    target[:] = -99  # stored commands must not alias caller-owned buffers
    obj.apply()
    assert calls[0][0].tolist() == [[8., 6.]]
    assert calls[0][1]["joint_ids"] == [3, 1]
    assert calls[0][1]["env_ids"].tolist() == [1]
    obj.set(None)
    obj.apply()
    assert len(calls) == 1


def test_full_replay_keeps_previous_behavior(manager):
    obj, calls = manager
    obj.set(torch.ones(2, 5))
    obj.apply()
    assert calls[0][0].shape == (2, 5)
    assert calls[0][1]["joint_ids"] is None
    assert calls[0][1]["env_ids"].tolist() == [0, 1]


def test_substeps_preserve_order_and_do_not_reuse_last_target(manager):
    obj, calls = manager
    targets = torch.arange(30.).reshape(2, 3, 5)
    obj.set(targets, joint_ids=[1, 3])
    for i in range(3):
        obj.apply()
        assert torch.equal(calls[i][0], targets[:, i, [1, 3]])
    with pytest.raises(RuntimeError, match="exhausted"):
        obj.apply()
    obj.set(targets)
    obj.apply()
    assert torch.equal(calls[-1][0], targets[:, 0])


def test_wrong_substep_count_rejected(manager):
    obj, _ = manager
    with pytest.raises(ValueError):
        obj.set(torch.zeros(2, 2, 5))


@pytest.mark.parametrize("kwargs", [dict(joint_ids=[]), dict(joint_ids=[1, 1]),
                                    dict(joint_ids=[5]), dict(joint_ids=[1.5]),
                                    dict(env_ids=[2]), dict(env_ids=[True])])
def test_invalid_subsets_fail_closed(manager, kwargs):
    obj, _ = manager
    with pytest.raises(ValueError):
        obj.set(torch.zeros(2, 5), **kwargs)
