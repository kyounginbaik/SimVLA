"""CPU checks for the emitted-kitchen RB-Y1 adapter."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/simvla"))

import rby1_kitchen_cfg as rkc  # noqa: E402


def _out():
    return rkc.transform((REPO / "scripts/simvla/kitchen_env_cfg_source.py").read_text())


def test_transform_runs_on_the_shipped_template_and_replaces_the_robot():
    out = _out()
    assert "from isaaclab_assets.robots.rby1 import RBY1_CFG" in out
    assert "ANUBIS" not in out and "anubis" not in out
    assert '"right_arm_0"' in out
    assert '"left_arm_0"' in out
    assert "self.sim.dt = 1 / 120" in out
    assert "self.decimation = 6" in out


def test_transform_accepts_checked_in_kitchen_without_optional_sensors():
    source = REPO / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_813_00.py"
    out = rkc.transform(source.read_text())
    assert "RBY1_CFG" in out
    assert "ANUBIS" not in out and "anubis" not in out


def test_selected_candidate_07_is_used_for_both_wrist_cameras():
    out = _out()
    right = out.split("wrist_right = TiledCameraCfg(", 1)[1].split("wrist_left =", 1)[0]
    left = out.split("wrist_left = TiledCameraCfg(", 1)[1].split("@configclass", 1)[0]
    assert "pos=(0.0, 0.1, 0.16)" in right
    assert "rot=(0.984807753012208, -0.17364817766693033, 0.0, 0.0)" in right
    assert "pos=(0.0, -0.1, 0.16)" in left
    assert "rot=(0.0, 0.0, -0.17364817766693033, 0.984807753012208)" in left


def test_checked_in_rby1_wrist_mounts_match_the_adapter():
    import ast
    def mounts(text):
        tree = ast.parse(text)
        result = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('wrist_left', 'wrist_right'):
                        offset = next(k.value for k in node.value.keywords if k.arg == 'offset')
                        result[target.id] = {k.arg: ast.literal_eval(k.value) for k in offset.keywords}
        return result
    checked_in = REPO / 'source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_813r_00.py'
    assert mounts(checked_in.read_text()) == mounts(_out())


def test_transform_instruments_both_mug_pads_on_each_arm():
    out = _out()
    assert out.count("mdp.ContactHoldingBinaryJointPositionActionCfg(") == 2
    for side, finger in (("r", "r"), ("l", "l")):
        assert f'contact_sensor_names=("touch_mug_{side}_pad1", "touch_mug_{side}_pad2")' in out
        for pad in (1, 2):
            name = f"touch_mug_{side}_pad{pad}"
            assert f"{name}: ContactSensorCfg" in out
            assert f'prim_path="{{ENV_REGEX_NS}}/Robot/ee_finger_{finger}{pad}"' in out
    assert out.count('filter_prim_paths_expr=["{ENV_REGEX_NS}/Kitchen/mug.*"]') == 4
    compile(out, "<generated-rby1-kitchen-cfg>", "exec")


def test_collector_selects_the_matching_rby1_pad_pair_for_each_arm():
    source = (REPO / "scripts/simvla/simvla_gen.py").read_text()
    assert 'if args_cli.robot == "rby1":' in source
    assert 'return (f"touch_mug_{arm}_pad1", f"touch_mug_{arm}_pad2")' in source
    assert '_mug_finger_contact_forces(int(_env), arm)' in source


def test_goal_rename_is_idempotent_and_uses_registered_task_id(tmp_path):
    source = tmp_path / "Isaac-Kitchen-v99000-00.json"
    source.write_text("{}")
    expected = tmp_path / "Isaac-Kitchen-v99000r-00.json"
    assert rkc._rename_goals(tmp_path, 99000, [0]) == [expected]
    assert expected.read_text() == "{}"
    assert rkc._rename_goals(tmp_path, 99000, [0]) == [expected]
