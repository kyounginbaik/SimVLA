"""Portable collection preflight must not start Isaac or write data."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/slurm/collect_public_lerobot.sbatch"


def launcher_env(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SIMVLA_", "SLURM_", "BODEX_", "OMNI_KIT_"))}
    for name in ("assets", "models", "bodex"):
        (tmp_path / name).mkdir()
    env.update(SIMVLA_PYTHON=sys.executable, SIMVLA_ASSETS_DIR=str(tmp_path / "assets"),
               SIMVLA_ROBOT_MODELS_DIR=str(tmp_path / "models"),
               BODEX_OBJ_DIR=str(tmp_path / "bodex"),
               SIMVLA_COLLECTION_ROOT=str(tmp_path / "output"), SIMVLA_DRY_RUN="1")
    return env


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_dry_run_is_portable_and_side_effect_free(tmp_path, robot):
    env = launcher_env(tmp_path)
    env["ROBOT"] = robot
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"robot={robot}\n" in result.stdout
    assert f"repo={ROOT}\n" in result.stdout
    assert "overlay=none" in result.stdout
    assert not (tmp_path / "output").exists()


def test_missing_asset_setting_is_actionable(tmp_path):
    env = launcher_env(tmp_path)
    del env["SIMVLA_ASSETS_DIR"]
    env["ROBOT"] = "aiworker"
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Set SIMVLA_ASSETS_DIR" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("count", ["0", "-1", "01", "1.5", "1000000"])
def test_invalid_bulk_count_is_rejected(tmp_path, count):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_NUM_DEMOS=count)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "SIMVLA_NUM_DEMOS must be" in result.stderr
    assert not (tmp_path / "output").exists()


def test_bulk_count_preflight(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_NUM_DEMOS="10")
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "num_demos=10\n" in result.stdout
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("task", ["Isaac-Kitchen-v813r-00", "../../bad", "Isaac-Kitchen-v435-03"])
def test_task_override_must_match_robot_and_goal(tmp_path, task):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_TASK=task,
               SIMVLA_SOURCE_GOAL=str(ROOT / "examples/goals/Isaac-Kitchen-v813-00.json"))
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()


def test_other_registered_scene_can_be_preflighted(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_TASK="Isaac-Kitchen-v435-03",
               SIMVLA_SOURCE_GOAL=str(ROOT / "examples/goals/Isaac-Kitchen-v435-03.json"))
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "task=Isaac-Kitchen-v435-03\n" in result.stdout


def test_arm_override_rejects_goal_without_selected_arm_grasp(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="rby1", SIMVLA_COLLECT_ARM="left")
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "no target object for selected left grasp arm" in result.stderr


def test_arm_override_checks_executable_grasp_channel(tmp_path):
    env = launcher_env(tmp_path)
    goal = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    goal["run_config"]["obj_name_l"] = "bowl0"
    path = tmp_path / "wrong-arm-goal.json"
    path.write_text(json.dumps(goal))
    env.update(ROBOT="rby1", SIMVLA_COLLECT_ARM="left", SIMVLA_SOURCE_GOAL=str(path))
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "no arm.grasp step on selected left arm" in result.stderr


def test_separate_export_python_preflight(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="aiworker", SIMVLA_EXPORT_PYTHON=sys.executable)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"export_python={sys.executable}" in result.stdout
    assert not (tmp_path / "output").exists()


def test_unexecutable_legacy_goal_fails_before_gpu_allocation(tmp_path):
    env = launcher_env(tmp_path)
    goal = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    goal["goals"][0].insert(4, ["N", [0.97, -.78, 1.57]])
    path = tmp_path / "invalid-goal.json"
    path.write_text(json.dumps(goal))
    env.update(ROBOT="anubis", SIMVLA_SOURCE_GOAL=str(path))
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "UnrecognisedStep" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("seed", ["0", "42", "4294967295"])
def test_collection_seed_is_explicit_in_preflight(tmp_path, seed):
    env = launcher_env(tmp_path)
    env.update(ROBOT="rby1", SIMVLA_SEED=seed)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"seed={seed}\n" in result.stdout
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("seed", ["-1", "01", "1.5", "x", "4294967296", "9" * 40])
def test_invalid_seed_fails_before_gpu_or_filesystem_writes(tmp_path, seed):
    env = launcher_env(tmp_path)
    env.update(ROBOT="rby1", SIMVLA_SEED=seed)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "SIMVLA_SEED must be an integer" in result.stderr
    assert not (tmp_path / "output").exists()


def test_launch_does_not_accept_eula_on_users_behalf(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_DRY_RUN="0")
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "if you accept it" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("allocated,override,expected", [
    (None, None, "8"), ("4", None, "4"), ("16", "6", "6"),
])
def test_cpu_workers_follow_allocation_or_explicit_override(tmp_path, allocated, override, expected):
    env = launcher_env(tmp_path)
    env["ROBOT"] = "anubis"
    if allocated is not None:
        env["SLURM_CPUS_PER_TASK"] = allocated
    if override is not None:
        env["SIMVLA_CPU_THREADS"] = override
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"cpu_threads={expected}\n" in result.stdout
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("threads", ["0", "-1", "01", "1.5", "x", "257"])
def test_invalid_cpu_workers_fail_before_boot(tmp_path, threads):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_CPU_THREADS=threads)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "SIMVLA_CPU_THREADS must be an integer" in result.stderr
    assert not (tmp_path / "output").exists()


def test_existing_run_is_never_overwritten(tmp_path):
    env = launcher_env(tmp_path)
    env.update(ROBOT="anubis", SIMVLA_DRY_RUN="0", OMNI_KIT_ACCEPT_EULA="YES",
               SLURM_JOB_ID="123", SIMVLA_REPO_ROOT=str(ROOT))
    run = tmp_path / "output/simvla-public-lerobot-anubis-123"
    run.mkdir(parents=True)
    marker = run / "keep.txt"
    marker.write_text("previous collection")
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "Refusing existing run" in result.stderr
    assert marker.read_text() == "previous collection"


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_goal_authoring_preflight_is_portable(tmp_path, robot):
    env = launcher_env(tmp_path)
    env.update(SIMVLA_ROBOT=robot, SIMVLA_GOAL_OUTPUT_ROOT=str(tmp_path / "goals"))
    result = subprocess.run(["bash", str(SCRIPT.with_name("emit_public_goals.sbatch"))],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"robot={robot}" in result.stdout
    assert f"repo={ROOT}" in result.stdout
    assert not (tmp_path / "goals").exists()


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_goal_authoring_runs_robot_adapter_after_geometry(tmp_path, robot):
    import shlex
    env = launcher_env(tmp_path)
    (tmp_path / "rby1m").mkdir()
    env["SIMVLA_RBY1M_DIR"] = str(tmp_path / "rby1m")
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> " + shlex.quote(str(calls)) + "\n")
    fake.chmod(0o755)
    env.update(SIMVLA_ROBOT=robot, SIMVLA_DRY_RUN="0", OMNI_KIT_ACCEPT_EULA="YES",
               SIMVLA_GOAL_OUTPUT_ROOT=str(tmp_path / "goals"), SIMVLA_PYTHON=str(fake),
               SIMVLA_AIWORKER_URDF_PATH=str(tmp_path / "prebuilt.urdf"))
    result = subprocess.run(["bash", str(SCRIPT.with_name("emit_public_goals.sbatch"))],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert "scripts/simvla/task_emit.py" in lines[0]
    if robot == "anubis":
        assert len(lines) == 1
    else:
        assert len(lines) == 2
        assert f"scripts/simvla/{robot}_kitchen_cfg.py --kitchen 813 --subs 0" in lines[1]
        assert f"--goals-dir {tmp_path / 'goals'}" in lines[1]
