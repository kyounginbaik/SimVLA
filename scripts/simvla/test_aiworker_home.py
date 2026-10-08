"""CPU checks for the AI Worker reset/planner home contract."""
import ast
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/simvla"))

from aiworker_home import sync_planner_home


def _config(side: str) -> dict:
    path = ROOT / f"configs/curobo/robot/aiworker_{side}_arm.yml"
    return yaml.safe_load(path.read_text())["robot_cfg"]


def test_default_lift_is_measured_counter_height_home():
    source = (ROOT / "source/isaaclab_assets/isaaclab_assets/robots/aiworker.py").read_text()
    assert 'os.environ.get("SIMVLA_AIWORKER_LIFT_M", "-0.30")' in source


def test_aiworker_gripper_speed_is_explicitly_bounded_and_defaults_to_one_rad_per_second():
    source = (ROOT / "source/isaaclab_assets/isaaclab_assets/robots/aiworker.py").read_text()
    assert 'os.environ.get("SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S", "1.0")' in source
    assert "0.1 <= AIWORKER_GRIPPER_SPEED_RAD_S <= 1.0" in source
    assert "velocity_limit_sim=AIWORKER_GRIPPER_SPEED_RAD_S" in source
    tree = ast.parse(source)
    config = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "AIWORKER_CFG" for t in n.targets))
    actuators = next(k.value for k in config.keywords if k.arg == "actuators")
    groups = {ast.literal_eval(k): v for k, v in zip(actuators.keys, actuators.values)}
    speed = lambda group: next(k.value for k in groups[group].keywords if k.arg == "velocity_limit_sim")
    assert ast.literal_eval(speed("lift")) == 1.0
    assert isinstance(speed("grippers"), ast.Name)
    assert speed("grippers").id == "AIWORKER_GRIPPER_SPEED_RAD_S"


def _lift_homes() -> dict[str, dict[str, float]]:
    source = ROOT / "source/isaaclab_assets/isaaclab_assets/robots/aiworker.py"
    tree = ast.parse(source.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "AIWORKER_LIFT_HOMES"
    )
    return ast.literal_eval(assignment.value)


@pytest.mark.parametrize("lift", ["-0.30", "-0.35"])
def test_lowered_arm_home_is_an_exact_kinematic_mirror(lift):
    """SG2 mirrors R->L with (+,-,-,+,-,+,-) across the sagittal plane."""
    home = _lift_homes()[lift]
    signs = (1, -1, -1, 1, -1, 1, -1)
    for index, sign in enumerate(signs, 1):
        right = home[f"arm_r_joint{index}"]
        left = home[f"arm_l_joint{index}"]
        assert left == pytest.approx(sign * right, abs=1e-9)


@pytest.mark.parametrize(
    ("lift", "right_roll", "left_roll"),
    [("-0.30", -0.7211956, 0.7211956), ("-0.35", -1.0316, 1.0316)],
)
def test_lowered_home_keeps_both_grippers_horizontal(lift, right_roll, left_roll):
    home = _lift_homes()[lift]
    assert home["arm_r_joint7"] == pytest.approx(right_roll, abs=1e-9)
    assert home["arm_l_joint7"] == pytest.approx(left_roll, abs=1e-9)


def test_selected_pose_one_is_the_default_aiworker_arm_home():
    home = _lift_homes()["-0.30"]
    expected_right = (0.9455975, -0.8036, -0.9631013, -2.7254865,
                      -1.3239744, 0.5481815, -0.7211956)
    expected_left = (0.9455975, 0.8036, 0.9631013, -2.7254865,
                     1.3239744, 0.5481815, 0.7211956)
    assert tuple(home[f"arm_r_joint{i}"] for i in range(1, 8)) == pytest.approx(
        expected_right, abs=1e-9
    )
    assert tuple(home[f"arm_l_joint{i}"] for i in range(1, 8)) == pytest.approx(
        expected_left, abs=1e-9
    )


def test_success_predicate_uses_the_symmetric_default_home():
    source = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/homes.py"
    tree = ast.parse(source.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "AIWORKER" for target in node.targets)
    )
    right, left = ast.literal_eval(assignment.value)
    assert right == pytest.approx((0.3249344, -0.3930621, 1.0229979), abs=1e-9)
    assert left == pytest.approx((0.3249344, 0.3930621, 1.0229979), abs=1e-9)


@pytest.mark.parametrize("lift,expected", [("-.30", .3930621), ("-.35", .4875063), ("0", .2444529)])
def test_success_home_tracks_the_configured_lift(monkeypatch, lift, expected):
    import importlib.util
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/mdp/homes.py"
    spec = importlib.util.spec_from_file_location("homes_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SIMVLA_AIWORKER_LIFT_M", lift)
    assert module.home_for(["arm_r_joint1"])[1][1] == pytest.approx(expected)


def test_left_mug_task_requires_close_return_to_the_correct_home():
    template_path = ROOT / "src/simvla/templates/mug_to_sink_left.json"
    template = json.loads(template_path.read_text())
    all_terms = template["success"]["all"]
    home = next(term["eef_home"] for term in all_terms if "eef_home" in term)
    assert home["arm"] == "both"
    assert home["radius"] == pytest.approx(0.05)


def test_collector_holds_the_uncovered_lift_at_its_reset_target():
    source = (ROOT / "scripts/simvla/simvla_gen.py").read_text()
    assert 'robot.find_joints(["lift_joint"])' in source
    assert "robot.data.default_joint_pos[:, aiworker_lift_ids].clone()" in source
    assert "joint_ids=aiworker_lift_ids" in source


def test_left_reset_returns_to_the_original_home_before_navigation():
    """A closed gripper must not let arm.reset pass through the loose carry gate."""
    source = (ROOT / "scripts/simvla/simvla_gen.py").read_text()

    assert (
        'reset_l_js = env.scene.articulations["robot"].data.joint_pos[:, l_j_index].clone()'
        in source
    )
    assert "reset_l_js[needs_plan_indices_temp_l[grasp_env_l]] =" not in source
    assert "_home_fk_l = motion_gen_l.kinematics.forward(" in source
    assert "_gjs_l.position.unsqueeze(0), link_name=\"ee_link2\"" in source
    reset_planning = source.split(
        "if _is_reset_l and env_idx not in _postrelease_reset_envs_l:", 1
    )[1].split("elif (_segmented_grasp_standoff_l", 1)[0]
    assert "motion_gen_l.plan_single_js(" in reset_planning
    left_planning = source.split("for i, env_idx in enumerate(needs_plan_indices_l):", 1)[1]
    left_planning = left_planning.split("# Convert goal to world frame", 1)[0]
    assert "sim_data.joint_limits[env_idx, l_j_index]" in left_planning
    assert "torch.clamp(" in left_planning
    assert "_jl_l[:, 0] + 1e-4" in left_planning
    assert "_jl_l[:, 1] - 1e-4" in left_planning

    arrival_gate = source.split("_sk_now_l = int(skill_ids_tensor[env_idx, _si_l])", 1)[1]
    reset_gate = arrival_gate.index("if _sk_now_l == SID_RESET:")
    carry_gate = arrival_gate.index("elif _holding_l:")
    assert reset_gate < carry_gate
    assert "pos_threshold = RESET_POS_TOL_M" in arrival_gate[:carry_gate]
    assert "rot_tol_l = RESET_ROT_TOL_DEG" in arrival_gate[:carry_gate]
    assert "RESET_POS_TOL_M = 0.02" in source
    assert ('RESET_ROT_TOL_DEG = bounded_float_env("SIMVLA_RESET_ROT_TOL_DEG", '
            "10.0, 0.0, 180.0)") in source


def test_smoke_disables_generated_task_predicates_before_stepping():
    source = (ROOT / "scripts/simvla/smoke_test.py").read_text()
    assert "for term_name in vars(cfg.terminations)" in source
    assert "setattr(cfg.terminations, term_name, None)" in source


@pytest.mark.parametrize("side", ["left", "right"])
def test_tracked_planner_config_and_spheres_are_complete(side):
    cfg = _config(side)
    kin = cfg["kinematics"]
    spheres = ROOT / kin["collision_spheres"]
    assert spheres.is_file()
    sphere_links = yaml.safe_load(spheres.read_text())["collision_spheres"]
    assert set(kin["collision_link_names"]) <= set(sphere_links)


@pytest.mark.parametrize("side", ["left", "right"])
def test_live_reset_pose_rewrites_retract_and_locked_lift(side):
    cfg = _config(side)
    names = cfg["kinematics"]["cspace"]["joint_names"]
    start = {name: i / 10 for i, name in enumerate(names)}
    start["lift_joint"] = -0.30

    sync_planner_home(cfg, start)

    kin = cfg["kinematics"]
    assert kin["lock_joints"]["lift_joint"] == -0.30
    assert kin["cspace"]["retract_config"] == [start[name] for name in names]


def test_home_sync_rejects_partial_reset_state():
    cfg = _config("right")
    with pytest.raises(ValueError, match="missing cuRobo joints"):
        sync_planner_home(cfg, {"lift_joint": -0.30})
