"""CPU gate on the AI Worker env-cfg transform.

Runs the real transform over the real template. Text in, text out -- no Isaac, no GPU. A stale
anchor here is a silent no-op that ships an env spawning Anubis while the planner runs AI Worker
kinematics, and that only surfaces on video.

Run: pytest scripts/simvla/test_aiworker_kitchen_cfg.py -v
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "scripts/simvla/kitchen_env_cfg_source.py"
sys.path.insert(0, str(REPO / "scripts/simvla"))

import aiworker_kitchen_cfg as akc  # noqa: E402


def _out():
    return akc.transform(TEMPLATE.read_text())


def test_the_transform_runs_over_the_real_template():
    """Every anchor present exactly the expected number of times. transform() raises SystemExit
    on the first that is not, naming it."""
    assert _out()


def test_adaptive_finger_segments_have_separate_mug_measurements():
    out = _out()
    for arm in ("l", "r"):
        assert f'additional_contact_sensor_names=(("touch_mug_{arm}_r1",), ("touch_mug_{arm}_l1",))' in out
        for finger in ("l", "r"):
            for segment in (1, 2):
                name = f"{finger}{segment}"
                assert f"touch_mug_{arm}_{name}: ContactSensorCfg" in out
                assert f'prim_path="{{ENV_REGEX_NS}}/Robot/gripper_{arm}_rh_p12_rn_{name}"' in out


def test_transform_accepts_checked_in_kitchen_without_optional_sensors():
    source = REPO / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/kitchen_813_00.py"
    out = akc.transform(source.read_text())
    assert "AIWORKER_CFG" in out
    assert "ANUBIS" not in out and "anubis" not in out


def test_no_anubis_reference_survives():
    out = _out()
    assert "ANUBIS" not in out and "anubis" not in out


def test_the_robot_is_the_ai_worker():
    out = _out()
    assert "from isaaclab_assets.robots.aiworker import AIWORKER_CFG" in out
    assert out.count("AIWORKER_CFG") == 4      # the import plus three uses


def test_arm_joint_names_are_seven_per_side():
    out = _out()
    for side in ("r", "l"):
        for i in range(1, 8):
            assert f'"arm_{side}_joint{i}"' in out
    assert 'joint_names=["arm_r_joint.*"]' in out
    assert 'joint_names=["arm_l_joint.*"]' in out


def test_head_camera_is_on_head_link2_at_forty_degrees():
    out = _out()
    assert 'prim_path="{ENV_REGEX_NS}/Robot/head_link2/head_cam"' in out
    # base_link is on the floor and a camera hung off it sits inside this robot's torso shell.
    assert "base_link/head_cam" not in out
    assert "pos=(0.10, 0.0, 0.02)" in out
    assert "rot=(0.93969, 0, 0.34202, 0)" in out          # 40 degrees down about +Y


def test_wrist_cameras_are_on_link7_at_the_assets_own_d405_mount():
    out = _out()
    assert 'prim_path="{ENV_REGEX_NS}/Robot/arm_r_link7/wrist_r_cam"' in out
    assert 'prim_path="{ENV_REGEX_NS}/Robot/arm_l_link7/wrist_l_cam"' in out
    # Measured relative to arm_?_link7 and IDENTICAL on both sides -- not mirrored.
    assert out.count("pos=(0.09824, 0.0, -0.07249)") == 2
    assert out.count("rot=(0.7062927455, -0.0339198709, 0.0339198709, -0.7062927455)") == 2
    assert out.count('convention="opengl"') == 2


def test_a_cctv_camera_is_added_for_watching_the_robot():
    """SIMVLA_RECORD_CAM=cctv is how these runs are recorded. Without this block that variable
    names a sensor the env does not have and the run writes no video at all -- which is exactly
    what the first smoke run did."""
    out = _out()
    assert 'prim_path="{ENV_REGEX_NS}/Robot/base_link/cctv_cam"' in out
    assert "cctv = TiledCameraCfg(" in out
    assert "pos=(0.0, 0.0, 3.02)" in out            # above the robot, not behind a wall
    assert "rot=(0.7967, 0.0, 0.6044, 0.0)" in out  # 74.4 deg down at the grasp plane
    assert "focal_length=14.0" in out               # a human-readable lens, not the fisheye


def test_wrist_lens_is_the_d405_not_the_templates_25mm():
    """At 25 mm the grasp centre sits 38.3 degrees off axis against a 22.7 degree half-angle,
    so the camera cannot see its own gripper at any pose."""
    out = _out()
    assert out.count("focal_length=11.04") == 2
    assert "focal_length=25.0" not in out


def test_ee_frame_offset_is_measured_in_ee_link1s_own_frame():
    """ee_link1 already sits at the jaws on this robot (the USD puts it at -0.178 from link7,
    rolled 180 deg about X, and the contact patch is at -0.1620), so the offset is the 0.016 m
    between them -- not Anubis's 0.1034 m reach out to its own jaws."""
    out = _out()
    assert out.count("pos=(0.0, 0.0, -0.0160)") == 2
    assert "pos=(0.0, 0.0, 0.1034)" not in out


def test_the_ik_deadzone_is_off_for_both_arms():
    """Seven-joint redundancy spreads one Cartesian step across every joint, so the 1e-2
    default discards whole commands. RB-Y1 measured that on its own seven-joint arm."""
    assert _out().count("delta_joint_deadzone=0.0") == 2


def test_grippers_are_the_revolute_rh_p12_rn_not_a_prismatic_pair():
    out = _out()
    assert out.count("mdp.ContactHoldingBinaryJointPositionActionCfg(") == 2
    assert 'contact_sensor_names=("touch_mug_l_r2", "touch_mug_l_l2")' in out
    assert 'contact_sensor_names=("touch_mug_r_r2", "touch_mug_r_l2")' in out
    for name in ("touch_mug_l_r2", "touch_mug_l_l2", "touch_mug_r_r2", "touch_mug_r_l2"):
        assert f"{name}: ContactSensorCfg" in out
    assert 'joint_names=["gripper_r_joint.*"]' in out
    assert 'joint_names=["gripper_l_joint.*"]' in out
    # Zero is OPEN on this linkage, the opposite of Anubis's prismatic jaws.
    assert '{"gripper_r_joint.*": 0.0}' in out
    assert '"gripper_r_joint1": 1.1002' in out
    assert "gripper1" not in out and "gripper2" not in out


def test_physics_runs_at_120hz_with_decimation_six():
    """At a 50 ms contact step a jaw crosses its whole travel inside one timestep. RB-Y1
    measured the object being held for three frames and then extruded."""
    out = _out()
    assert "self.decimation = 6" in out
    assert "self.sim.dt = 1 / 120" in out
    assert "self.sim.dt = 1 / 20" not in out


def test_the_class_is_renamed_so_the_register_entry_point_resolves():
    out = _out()
    assert "class AIWorkerKitchenEnvCfg(ManagerBasedRLEnvCfg):" in out
    assert "AIWorkerKitchenEnvCfg" in akc._REGISTER


def test_registration_uses_the_a_suffix_task_ids():
    reg = akc._REGISTER.format(knum=1215, sub=0)
    assert 'id="Isaac-Kitchen-v1215a-00"' in reg
    assert "kitchen_1215a_00:AIWorkerKitchenEnvCfg" in reg


def test_a_stale_anchor_raises_rather_than_silently_doing_nothing():
    import pytest
    with pytest.raises(SystemExit):
        akc.transform("nothing that looks like a kitchen env cfg")


def test_goal_names_are_aligned_with_the_a_suffix(tmp_path):
    original = tmp_path / "Isaac-Kitchen-v1215-00.json"
    original.write_text('{"version": 2}')

    renamed = akc._rename_goals(tmp_path, 1215, [0])

    expected = tmp_path / "Isaac-Kitchen-v1215a-00.json"
    assert renamed == [expected]
    assert expected.read_text() == '{"version": 2}'
    assert not original.exists()
    assert akc._rename_goals(tmp_path, 1215, [0]) == [expected]


def test_goal_rename_refuses_missing_or_conflicting_files(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="SIMVLA_ROBOT=aiworker"):
        akc._rename_goals(tmp_path, 1215, [0])
    (tmp_path / "Isaac-Kitchen-v1215-00.json").write_text("source")
    (tmp_path / "Isaac-Kitchen-v1215a-00.json").write_text("target")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        akc._rename_goals(tmp_path, 1215, [0])


def test_main_writes_registered_config_and_matching_goal(monkeypatch, tmp_path):
    package = tmp_path / "kitchen"
    package.mkdir()
    (package / "__init__.py").write_text("import gymnasium as gym\n")
    (package / "kitchen_1215_00.py").write_text(TEMPLATE.read_text())
    goals = tmp_path / "goals"
    goals.mkdir()
    (goals / "Isaac-Kitchen-v1215-00.json").write_text('{"version": 2}')
    monkeypatch.setattr(akc, "KITCHEN_PKG", package)

    assert akc.main(["--kitchen", "1215", "--subs", "0", "--goals-dir", str(goals)]) == 0

    config = package / "kitchen_1215a_00.py"
    assert "class AIWorkerKitchenEnvCfg" in config.read_text()
    registration = (package / "__init__.py").read_text()
    assert 'id="Isaac-Kitchen-v1215a-00"' in registration
    assert (goals / "Isaac-Kitchen-v1215a-00.json").is_file()
