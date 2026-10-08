"""Exercise the actual action methods without importing Isaac or starting Kit."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def action_class():
    path = Path(__file__).resolve().parents[1] / (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/contact_gripper.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "ContactHoldingBinaryJointPositionAction")

    class Base:
        def process_actions(self, actions):
            self._processed_actions = torch.zeros(actions.numel(), 4)

        def apply_actions(self):
            pass

    namespace = {"torch": torch, "BinaryJointPositionAction": Base,
                 "ContactHoldingBinaryJointPositionActionCfg": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def test_boolean_close_follows_binary_action_convention():
    action = object.__new__(action_class())
    action.num_envs = 2
    action._latched = torch.tensor([True, True])
    action._balance_frozen = torch.tensor([True, True])
    action._hold_target = torch.ones(2, 4)
    action.process_actions(torch.tensor([[False], [True]]))
    assert action._closing.tolist() == [True, False]
    assert action._latched.tolist() == [True, False]
    assert action._balance_frozen.tolist() == [True, False]
    assert action._processed_actions[0].tolist() == [1] * 4


def test_missing_object_filtered_force_cannot_latch_on_counter_contact():
    action = object.__new__(action_class())
    action.num_envs, action.device = 2, "cpu"
    sensor = SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=None, net_forces_w=torch.full((2, 1, 3), 100.)))
    assert action._pad_load(sensor).tolist() == [0, 0]


def test_filtered_force_rejects_nonfinite_measurements():
    action = object.__new__(action_class())
    action.num_envs, action.device = 3, "cpu"
    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.tensor(
        [[[[3., 4., 0.]]], [[[float("nan"), 1., 1.]]], [[[float("inf"), 0., 0.]]]])))
    assert action._pad_load(sensor).tolist() == [5, 0, 0]


def test_force_balance_updates_named_jaw_only_in_holding_environments():
    action = object.__new__(action_class())
    action.cfg = SimpleNamespace(force_balance_enabled=True, target_pad_force_n=6,
                                 maximum_pad_force_n=12, minimum_pad_force_n=1,
                                 balance_step_fraction=.005, freeze_at_target_force=False)
    action._closing = torch.tensor([False, True])
    action._latched = torch.tensor([False, True])
    action._contact_sensors = (torch.tensor([0., 0.]), torch.tensor([0., 20.]))
    action._additional_contact_sensors = ((), ())
    action._pad_load = lambda sensor: sensor
    action._preload_by_pad = torch.full((2, 2), .4)
    action._hold_target = torch.zeros(2, 4)
    action._contact_pos = torch.zeros(2, 4)
    action._close_command = torch.ones(4)
    action._processed_actions = torch.zeros(2, 4)
    action._balance_ticks = torch.zeros(2, dtype=torch.long)
    action._pad_joint_indices = ((0, 2), (1, 3))
    action.apply_actions()
    assert action._hold_target[0].tolist() == [0] * 4
    assert action._hold_target[1].tolist() == pytest.approx([.405, .395, .405, .395])
    assert torch.equal(action._processed_actions, action._hold_target)
    # Optional position-hold mode freezes only after BOTH fingers reach force.
    action.cfg.freeze_at_target_force = True
    action._balance_frozen = torch.zeros(2, dtype=torch.bool)
    action._contact_sensors = (torch.tensor([0., 8.]), torch.tensor([0., 9.]))
    targets = action._hold_target.clone()
    action.apply_actions()
    assert action._balance_frozen.tolist() == [False, True]
    assert torch.equal(action._hold_target, targets)
    action._contact_sensors = (torch.zeros(2), torch.zeros(2))
    action.apply_actions()
    assert torch.equal(action._hold_target, targets)


def test_adaptive_finger_load_does_not_confuse_a_free_tip_with_a_free_finger():
    action = object.__new__(action_class())
    action.num_envs, action.device = 1, "cpu"
    def sensor(force):
        return SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.tensor([[[[force, 0., 0.]]]])))
    action._contact_sensors = (sensor(12.), sensor(0.))
    action._additional_contact_sensors = ((sensor(34.),), (sensor(46.),))
    assert action._finger_load(0).item() == 46.
    assert action._finger_load(1).item() == 46.


def test_balance_increment_can_preserve_rate_across_physics_substeps(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/contact_gripper.py").read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "ContactHoldingBinaryJointPositionActionCfg")
    value = next(n.value for n in cfg.body if isinstance(n, ast.AnnAssign)
                 and n.target.id == "balance_step_fraction")
    expression = compile(ast.Expression(value), "contact_gripper.py", "eval")
    monkeypatch.delenv("SIMVLA_GRIPPER_BALANCE_STEP_FRACTION", raising=False)
    assert eval(expression, {"os": os}) == .005
    for substeps in (6, 12, 24):
        monkeypatch.setenv("SIMVLA_GRIPPER_BALANCE_STEP_FRACTION", str(.03 / substeps))
        assert eval(expression, {"os": os}) * substeps * 20 == pytest.approx(.6)
    assert "SIMVLA_GRIPPER_BALANCE_STEP_FRACTION" in (
        root / "scripts/tools/release_evidence.py").read_text()
