"""User-facing behavior exercised through the installed distribution."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from simvla.tasks import list_templates, load_template


def test_templates_are_wheel_resources():
    for name in list_templates():
        assert load_template(name).steps


def test_left_arm_mug_template_is_explicitly_robot_agnostic():
    task = load_template("mug_to_sink_left")
    assert task.name == "mug_to_sink_left"
    assert [step.action for step in task.steps] == ["N_s", "A_l", "G_l", "A_l", "N_s", "A_l", "G_l", "A_l"]
    assert task.steps[0].params["which_arm"] == "Left"


def test_task_api_does_not_load_simulator_or_torch(tmp_path):
    code = """
import sys
from simvla.tasks import load_template
assert len(load_template('bowl_to_drawer').steps) == 12
assert not {'torch', 'isaaclab', 'isaacsim', 'numpy', 'scene_synthesizer'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_cli_validation_from_unrelated_directory(tmp_path):
    result = subprocess.run([sys.executable, "-m", "simvla", "validate", "bowl_to_drawer"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["valid"]


def test_cli_skill_introspection_is_available_without_simulator(tmp_path):
    result = subprocess.run([sys.executable, "-m", "simvla", "skills", "nav.to_prim"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    skill = json.loads(result.stdout)
    assert skill["id"] == "nav.to_prim"
    assert skill["actions"]
    assert skill["params"]
    assert "Traceback" not in result.stderr


def test_cli_skill_introspection_encodes_non_finite_defaults_as_unset(tmp_path):
    result = subprocess.run([sys.executable, "-m", "simvla", "skills", "arm.bowl_place"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    skill = json.loads(result.stdout)
    params = {item["name"]: item for item in skill["params"]}
    assert params["min_eef_z"]["default"] is None
    assert "Infinity" not in result.stdout


def test_cli_initializes_editable_task_and_refuses_overwrite(tmp_path):
    output = tmp_path / "nested/my_task.json"
    command = [sys.executable, "-m", "simvla", "init-task", "--output", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["name"] == "my_task"
    assert load_template(output).name == "my_task"

    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert second.returncode == 1
    assert output.exists()
    assert "Traceback" not in second.stderr


def test_cpu_demo_builds_scene_task_and_manifest_atomically(tmp_path):
    pytest.importorskip("scene_synthesizer")
    from simvla import cli

    output = tmp_path / "quickstart"
    assert cli.main(["demo", "--output-dir", str(output), "--seed", "7"]) == 0
    assert (output / "kitchen.glb").stat().st_size > 0
    assert load_template(output / "task.json").name == "quickstart_bowl_to_drawer"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed"] == 7 and manifest["steps"] == 12
    assert manifest["registered_skills"] >= 24

    assert cli.main(["demo", "--output-dir", str(output)]) == 1


def test_cli_preflights_bundled_legacy_goal_from_unrelated_directory(tmp_path):
    goal = Path(__file__).resolve().parents[1] / "examples/goals/Isaac-Kitchen-v813-00.json"
    result = subprocess.run([sys.executable, "-m", "simvla", "validate-goal", str(goal)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["valid"] and report["version"] == 1 and report["steps"] == 12
    assert "regenerate as version 2" in report["detail"]


def test_goal_preflight_rejects_unknown_v2_skill(tmp_path):
    goal = tmp_path / "bad-goal.json"
    goal.write_text(json.dumps({"version": 2, "goals": [[
        {"skill": "arm.teleport", "action": "A_r", "params": {}, "goal": None}
    ]]}))
    result = subprocess.run([sys.executable, "-m", "simvla", "validate-goal", str(goal)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "unknown skill 'arm.teleport'" in result.stderr
    assert "Traceback" not in result.stderr


def test_goal_preflight_checks_v2_params_and_resolution_mode(tmp_path):
    from simvla.goal_validate import GoalValidationError, validate_goal
    goal = tmp_path / "goal.json"
    payload = {"version": 2, "goals": [[
        {"skill": "nav.to_prim", "action": "N_s",
         "params": {"prim_path": "/world/bowl0", "safety": 0.18},
         "goal": [0.1, 0.2, 0.0]}
    ]]}
    goal.write_text(json.dumps(payload))
    assert validate_goal(goal)["valid"]

    payload["goals"][0][0]["params"]["safety"] = "far"
    goal.write_text(json.dumps(payload))
    with pytest.raises(GoalValidationError, match="invalid value"):
        validate_goal(goal)

    payload["goals"][0][0]["params"]["safety"] = 0.18
    payload["goals"][0][0]["goal"] = None
    goal.write_text(json.dumps(payload))
    with pytest.raises(GoalValidationError, match="authored value"):
        validate_goal(goal)


def test_doctor_reports_missing_aiworker_assets(monkeypatch, tmp_path):
    from simvla import cli
    # Keep this test independent of a developer's staged research-asset bundle.
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout="Test GPU\n", stderr=""))
    report = cli.doctor(root=tmp_path, robot="aiworker")
    assert report["simulator_ready"]
    assert isinstance(report["planner_ready"], bool)
    assert not report["robot"]["ready"]
    assert set(report["robot"]["missing"]) == {
        "usd", "urdf", "curobo_left", "curobo_right", "spheres_left", "spheres_right"
    }


def test_doctor_reports_anubis_asset_contract(monkeypatch, tmp_path):
    from simvla import cli
    assets = tmp_path / "assets"
    models = tmp_path / "models"
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(assets))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(models))

    report = cli.doctor(root=tmp_path, robot="anubis")

    assert report["robot"]["files"] == {
        "usd": str(assets / "Robots/anubis_simvla.usd"),
        "urdf": str(models / "anubis/anubis_final.urdf"),
        "curobo_left": str(assets / "curobo/robot/anubis_left_arm.yml"),
        "curobo_right": str(assets / "curobo/robot/anubis_right_arm.yml"),
    }
    assert set(report["robot"]["missing"]) == set(report["robot"]["files"])

    for path in report["robot"]["files"].values():
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    complete = cli.doctor(root=tmp_path, robot="anubis")
    assert complete["robot"]["ready"]
    assert complete["robot"]["missing"] == []


def test_doctor_reports_rby1_asset_contract(monkeypatch, tmp_path):
    from simvla import cli
    assets = tmp_path / "assets"
    models = tmp_path / "models"
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(assets))
    monkeypatch.setenv("SIMVLA_RBY1M_DIR", str(models))
    report = cli.doctor(root=tmp_path, robot="rby1")
    assert report["robot"]["files"] == {
        "usd": str(models / "models/rby1m/urdf/model/model_simvla_black_gripper.usd"),
        "urdf": str(models / "models/rby1m/urdf/model.urdf"),
        "curobo_left": str(tmp_path / "configs/curobo/robot/rby1_left_arm.yml"),
        "curobo_right": str(tmp_path / "configs/curobo/robot/rby1_right_arm.yml"),
        "spheres_left": str(tmp_path / "configs/curobo/robot/spheres/rby1_spheres_left.yml"),
        "spheres_right": str(tmp_path / "configs/curobo/robot/spheres/rby1_spheres_right.yml"),
    }
    assert set(report["robot"]["missing"]) == {
        "usd", "urdf", "curobo_left", "curobo_right", "spheres_left", "spheres_right",
    }


def test_doctor_accepts_complete_aiworker_asset_manifest(monkeypatch, tmp_path):
    from simvla import cli
    assets = tmp_path / "assets"
    models = tmp_path / "models"
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(assets))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(models))
    for path in (
        assets / "Robots/MM/aiworker/ffw_sg2.usd",
        tmp_path / "configs/curobo/robot/aiworker_left_arm.yml",
        tmp_path / "configs/curobo/robot/aiworker_right_arm.yml",
        tmp_path / "configs/curobo/robot/spheres/aiworker_spheres_left.yml",
        tmp_path / "configs/curobo/robot/spheres/aiworker_spheres_right.yml",
        models / "ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    report = cli.doctor(root=tmp_path, robot="aiworker")
    assert report["robot"]["ready"]
    assert report["robot"]["missing"] == []


def test_invalid_template_fails_cleanly(tmp_path):
    from simvla.task_template import to_json
    task = load_template("bowl_to_drawer")
    task.steps = task.steps[2:]
    task.subtask_groups = []
    path = tmp_path / "broken.json"
    path.write_text(to_json(task))
    result = subprocess.run([sys.executable, "-m", "simvla", "validate", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "closes on nothing" in result.stderr
    assert "Traceback" not in result.stderr


def test_export_refuses_overwrite_before_generating(tmp_path):
    from simvla.scenes import export_kitchen
    path = tmp_path / "existing.glb"
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        export_kitchen(path)
    assert path.read_bytes() == b"keep me"


def test_glb_has_finite_nonempty_geometry(tmp_path):
    pytest.importorskip("scene_synthesizer")
    import numpy as np
    import trimesh
    from simvla.scenes import export_kitchen
    path = export_kitchen(tmp_path / "kitchen.glb", seed=0)
    scene = trimesh.load(path, force="scene")
    assert len(scene.geometry) > 10
    assert np.isfinite(scene.bounds).all()
    assert (scene.extents > 0).all()


def test_workflow_uses_selected_checkout_and_preserves_arguments(monkeypatch, tmp_path):
    from simvla import cli
    script = tmp_path / "scripts/simvla/smoke_test.py"
    script.parent.mkdir(parents=True)
    script.touch()
    monkeypatch.setenv("PYTHONPATH", "/some/other/checkout")
    monkeypatch.setattr(cli, "doctor", lambda: {"simulator_ready": True})
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or 7)
    assert cli.main(["run", "smoke", "--repo-root", str(tmp_path), "--",
                     "--task", "a task", "--headless"]) == 7
    cmd, options = calls[0]
    assert cmd == [sys.executable, str(script), "--task", "a task", "--headless"]
    assert options["cwd"] == tmp_path
    assert options["env"]["SIMVLA_REPO_ROOT"] == str(tmp_path)
    paths = options["env"]["PYTHONPATH"].split(__import__("os").pathsep)
    assert paths[0] == str(tmp_path / "source/isaaclab")
    assert paths[-1] == "/some/other/checkout"


def test_aiworker_adaptation_uses_selected_checkout_without_simulator(monkeypatch, tmp_path):
    from simvla import cli
    script = tmp_path / "scripts/simvla/aiworker_kitchen_cfg.py"
    script.parent.mkdir(parents=True)
    script.touch()
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or 0)

    goals = tmp_path / "goals"
    assert cli.main(["adapt-aiworker", "--repo-root", str(tmp_path),
                     "--kitchen", "99000", "--subs", "0,3",
                     "--goals-dir", str(goals)]) == 0

    cmd, options = calls[0]
    assert cmd == [sys.executable, str(script), "--kitchen", "99000", "--subs", "0,3",
                   "--goals-dir", str(goals.resolve())]
    assert options["cwd"] == tmp_path
    assert options["env"]["SIMVLA_REPO_ROOT"] == str(tmp_path)


@pytest.mark.parametrize('reported,expected', [(0, 0), (1, 1), (None, 1)])
def test_workflow_result_survives_kit_zero_exit(monkeypatch, tmp_path, reported, expected):
    import json
    from pathlib import Path
    from simvla import cli
    script = tmp_path / 'scripts/simvla/smoke_test.py'
    script.parent.mkdir(parents=True)
    script.touch()
    monkeypatch.setattr(cli, 'doctor', lambda: {'simulator_ready': True})

    def kit_shutdown(cmd, **kwargs):
        if reported is not None:
            Path(kwargs['env']['SIMVLA_STATUS_FILE']).write_text(json.dumps({'exit_code': reported}))
        return 0

    monkeypatch.setattr(cli.subprocess, 'call', kit_shutdown)
    assert cli.main(['run', 'smoke', '--repo-root', str(tmp_path)]) == expected
