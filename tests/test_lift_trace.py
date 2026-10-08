"""Run collector lift telemetry without booting Isaac or changing robot control."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def trace_function(namespace):
    path = Path(__file__).resolve().parents[1] / "scripts/simvla/simvla_gen.py"
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "_trace_lift")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def test_trace_is_inert_when_disabled(monkeypatch):
    import os
    monkeypatch.delenv("SIMVLA_TRACE_LIFT", raising=False)
    # No environment, robot, sensors, or tensor accesses in the disabled path.
    trace_function({"os": os})("r", None, None)


@pytest.mark.parametrize("arm,names", [
    ("r", ["gripper_finger_r1", "gripper_finger_l1", "gripper_finger_r2"]),
    ("l", ["gripper_l_joint1", "gripper_r_joint1", "gripper_l_joint2"]),
])
def test_trace_captures_measured_state_without_mutation(monkeypatch, capsys, arm, names):
    import os
    monkeypatch.setenv("SIMVLA_TRACE_LIFT", "1")
    positions = torch.tensor([[.1, .2, .3]])
    targets = positions + .1
    robot = SimpleNamespace(
        joint_names=names, body_names=["pad1", "pad2"],
        data=SimpleNamespace(joint_pos=positions, joint_pos_target=targets,
                             body_pos_w=torch.zeros(1, 2, 3),
                             body_quat_w=torch.tensor([[[1., 0, 0, 0]] * 2])))
    obj = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=torch.tensor([[[1., 2, 3]]])),
        root_physx_view=SimpleNamespace(
            get_masses=lambda: torch.tensor([[.5]]),
            get_material_properties=lambda: torch.tensor([[[.8, .7, 0.]]])))
    function = trace_function({
        "os": os, "json": json,
        "env": SimpleNamespace(scene=SimpleNamespace(
            articulations={"robot": robot}, rigid_objects={"mug0": obj})),
        "args_cli": SimpleNamespace(obj_name="mug0", obj_name_l="mug0"),
        "jaw_body_indices": lambda robot, side: (0, 1),
        "_mug_pad_contact_forces": lambda env_id, side, proximal=False: (5., 6.) if proximal else (3., 4.),
    })
    function(arm, torch.tensor([0]), torch.tensor([11]))
    assert capsys.readouterr().out == ""
    function(arm, torch.tensor([0]), torch.tensor([10]))
    record = json.loads(capsys.readouterr().out.removeprefix("[lift-trace] "))
    assert record["joint_names"] == [names[0], names[2]]
    assert record["object_mass_kg"] == [.5]
    assert record["pad_load_n"] == [3., 4.]
    assert record["proximal_load_n"] == [5., 6.]
    assert record["remaining_ticks"] == 10
    assert torch.equal(robot.data.joint_pos, positions)
    assert torch.equal(robot.data.joint_pos_target, targets)
