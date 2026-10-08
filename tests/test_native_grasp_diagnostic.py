import importlib.util
import copy
import json
import ast
import hashlib
from pathlib import Path

import pytest


def test_diagnostic_rejects_unverified_pickle_before_loading_it(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/tools/make_aiworker_native97_diagnostic.py"
    spec = importlib.util.spec_from_file_location("native_grasp_diagnostic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = tmp_path / "untrusted.npy"
    bank.write_bytes(b"not the pinned public bank")
    with pytest.raises(ValueError, match="hash-verified"):
        module.make_goal(bank, tmp_path / "not-read.json")


def test_sink_clearance_preserves_grasp_and_success_gates():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "native_grasp_diagnostic", root / "scripts/tools/make_aiworker_native97_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    goal = json.loads((root / "examples/goals/Isaac-Kitchen-v813a-00.json").read_text())
    original = copy.deepcopy(goal)
    module.sink_clearance_goal(goal)
    assert goal["goals"][0][:4] == original["goals"][0][:4]
    assert goal["goals"][0][6:] == original["goals"][0][6:]
    assert goal["run_config"] == original["run_config"]
    assert goal["goals"][0][4]["goal"][:2] == [.9, -2.4]
    assert goal["goals"][0][5]["params"]["down_m"] == 0
    module.sink_clearance_goal(goal, wall_clearance=True)
    assert goal["goals"][0][4]["goal"][:2] == [1., -2.1]
    assert goal["goals"][0][5]["params"]["lateral_m"] == -.25


def test_both_place_executors_consume_lateral_payload():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/simvla/simvla_gen.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "resolve_skill"
             and n.args and isinstance(n.args[0], ast.Constant)
             and n.args[0].value == "arm.bowl_place"]
    assert len(calls) == 2
    for call in calls:
        value = next(k.value for k in call.keywords if k.arg == "lateral_m")
        assert isinstance(value, ast.Subscript)
        assert isinstance(value.slice, ast.Tuple)
        assert isinstance(value.slice.elts[0], ast.Slice)
        assert value.slice.elts[1].value == 4


@pytest.mark.parametrize("extension", [-.01, .21, float("nan"), float("inf")])
def test_carry_extension_is_bounded_before_reading_bank(extension):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "native_grasp_diagnostic", root / "scripts/tools/make_aiworker_native97_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="carry_forward_m"):
        module.make_goal(Path("missing-bank"), Path("missing-goal"), carry_forward_m=extension)


@pytest.mark.parametrize("candidate", [-1, 100, 1.5, True, None])
def test_candidate_is_bounded_before_reading_bank(candidate):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "native_grasp_diagnostic", root / "scripts/tools/make_aiworker_native97_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="candidate_index"):
        module.make_goal(Path("missing-bank"), Path("missing-goal"), candidate_index=candidate)


def test_angled_sink_preserves_native_orientation_and_extends_counter_forward(tmp_path, monkeypatch):
    import numpy as np
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    from simvla.skills import grasp_tool_frame

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "native_grasp_diagnostic", root / "scripts/tools/make_aiworker_native97_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    offset = np.asarray(grasp_tool_frame("aiworker")["q_offset"])
    native = rotation.from_euler("y", -90, degrees=True) * rotation.from_quat(offset[[1, 2, 3, 0]]).inv()
    poses = np.zeros((1, 98, 1, 7))
    poses[0, 97, 0, 3:] = native.as_quat()[[3, 0, 1, 2]]
    alternate = rotation.from_euler("z", 20, degrees=True) * native
    poses[0, 1, 0, 3:] = alternate.as_quat()[[3, 0, 1, 2]]
    bank = tmp_path / "synthetic-trusted-bank.npy"
    np.save(bank, {"robot_pose": poses})
    # The synthetic test bank is trusted only inside this test; production keeps
    # the fixed public hash and refuses arbitrary pickle deserialization.
    monkeypatch.setattr(module, "EXPECTED_BANK_SHA256", hashlib.sha256(bank.read_bytes()).hexdigest())
    source = root / "examples/goals/Isaac-Kitchen-v813a-00.json"
    base = module.make_goal(bank, source, preserve_grasp_carry=True, wall_clearance=True)
    alternative = module.make_goal(bank, source, candidate_index=1)
    assert alternative["diagnostic_grasp_source"]["native_index"] == 1
    expected = alternate * rotation.from_quat(offset[[1, 2, 3, 0]])
    np.testing.assert_allclose(alternative["goals"][0][1]["goal"][0][3:],
                               expected.as_quat()[[3, 0, 1, 2]], atol=1e-12)
    assert alternative["diagnostic_grasp_source"]["accepted_episode"] is False
    with pytest.raises(ValueError, match="outside the verified bank"):
        module.make_goal(bank, source, candidate_index=99)
    angled = module.make_goal(bank, source, preserve_grasp_carry=True,
                              wall_clearance=True, carry_forward_m=.15, angled_sink=True)
    a, b = base["goals"][0], angled["goals"][0]
    assert b[1] == a[1]  # BoDex grasp untouched.
    assert b[3]["goal"][0] == a[3]["goal"][0]
    assert b[3]["goal"][1] - a[3]["goal"][1] == pytest.approx(.15)
    assert b[3]["goal"][2:] == a[3]["goal"][2:]
    assert b[4]["goal"][2] == -.5
    assert b[5]["params"]["forward_m"] == pytest.approx(.18)
    assert b[5]["params"]["lateral_m"] == -.03
    assert angled["run_config"] == base["run_config"]
    home = module.make_goal(bank, source, return_home=True,
                           wall_clearance=True, angled_sink=True)
    h = home["goals"][0]
    assert h[1] == b[1]  # Native grasp and halfway approach remain unchanged.
    assert h[2] == b[2]  # Keep the existing gripper-close step.
    assert h[3]["skill"] == "arm.reset"
    assert h[3]["action"] == "A_l"
    assert h[3]["goal"] is None
    assert h[6:] == b[6:]  # Release only at the sink, then reset again.
    assert home["diagnostic_grasp_source"]["return_home"] is True
    aisle = module.make_goal(bank, source, return_home=True,
                             wall_clearance=True, angled_sink=True, aisle_waypoint=True)
    a_steps = aisle["goals"][0]
    assert a_steps[:4] == h[:4]
    assert a_steps[5:] == h[4:]
    assert a_steps[4]["goal"][:2] == [.80, -.88]
    assert a_steps[4]["skill"] == "nav.to_prim"
    assert aisle["run_config"]["export_groups"] == "3;4,5,6,7,8"
    assert aisle["diagnostic_grasp_source"]["accepted_episode"] is False
    raised = module.make_goal(bank, source, return_home=True, wall_clearance=True,
                              angled_sink=True, aisle_waypoint=True, raised_home_clearance=True)
    raised_steps = raised["goals"][0]
    assert raised_steps[:5] == a_steps[:5]  # Actual home remains before any clearance lift.
    assert raised_steps[6:] == a_steps[5:]  # Gripper stays closed until the same sink release.
    assert raised_steps[5]["skill"] == "arm.pose"
    assert raised_steps[5]["goal"][2] == pytest.approx(1.22299793)
    assert np.linalg.norm(raised_steps[5]["goal"][3:]) == pytest.approx(1.)
    assert raised["run_config"]["export_groups"] == "3;4,5,6,7,8,9"
    assert raised["diagnostic_grasp_source"]["accepted_episode"] is False
    absolute = module.make_goal(bank, source, return_home=True, wall_clearance=True,
                                angled_sink=True, aisle_waypoint=True, raised_home_clearance=True,
                                absolute_sink_placement=True)
    absolute_steps = absolute["goals"][0]
    assert absolute_steps[:7] == raised_steps[:7]
    assert absolute_steps[8:] == raised_steps[8:]
    assert absolute_steps[7]["skill"] == "arm.pose"
    assert absolute_steps[7]["goal"][:3] == [1.685, -2.240, 1.100]
    assert np.linalg.norm(absolute_steps[7]["goal"][3:]) == pytest.approx(1.)
    assert absolute["diagnostic_grasp_source"]["accepted_episode"] is False
    with pytest.raises(ValueError, match="absolute sink placement requires"):
        module.make_goal(bank, source, absolute_sink_placement=True)
    with pytest.raises(ValueError, match="raised home clearance requires"):
        module.make_goal(bank, source, raised_home_clearance=True)
    with pytest.raises(ValueError, match="aisle waypoint requires"):
        module.make_goal(bank, source, aisle_waypoint=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.make_goal(bank, source, return_home=True, preserve_grasp_carry=True)
