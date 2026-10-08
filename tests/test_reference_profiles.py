"""Reference recipes must not silently inherit a different planner budget."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_reference_search_budget_is_explicit(robot):
    path = Path(__file__).parents[1] / f"configs/collection/{robot}-kitchen813.env"
    settings = dict(line.split("=", 1) for line in path.read_text().splitlines()
                    if line and not line.startswith("#"))
    for name, value in {"SIMVLA_PLANNER_ATTEMPTS": "2", "SIMVLA_GRAPH_SEEDS": "4",
                        "SIMVLA_TRAJOPT_SEEDS": "4", "SIMVLA_FINETUNE_TRAJOPT": "1"}.items():
        assert settings[name] == value


@pytest.mark.parametrize("profile,robot,frames", [
    ("rby1-kitchen813-basin", "rby1", "4500"),
    ("aiworker-kitchen813-loaded-basin", "aiworker", "8000"),
])
def test_basin_profiles_keep_collision_and_acceptance_flags_together(profile, robot, frames):
    path = Path(__file__).parents[1] / f"configs/collection/{profile}.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert settings["ROBOT"] == robot
    assert settings["SIMVLA_MAXFRAMES"] == frames
    assert settings["SIMVLA_SINK_CAVITY_COLLISIONS"] == "1"
    assert settings["SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"] == "1"
    assert settings["SIMVLA_REPLAY_EPISODE_INDEX"] == "0"
    if robot == "aiworker":
        assert settings["SIMVLA_LOADED_HOME_JOINT_TARGETS"] == "1"
        assert settings["SIMVLA_AIWORKER_PAYLOAD_COLLISION"] == "1"
        assert settings["SIMVLA_AIWORKER_WRIST_STIFFNESS"] == "1000"


def test_aiworker_substep_profile_inherits_base_and_declares_distinct_replay():
    path = Path(__file__).parents[1] / "configs/collection/aiworker-kitchen813-substeps.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for name, value in {
        "ROBOT": "aiworker", "SIMVLA_AIWORKER_LIFT_M": "-.30",
        "SIMVLA_REPLAY_BASE_TRACKING": "1", "SIMVLA_REPLAY_JOINT_TARGETS": "1",
        "SIMVLA_RECORD_JOINT_SUBSTEPS": "1", "SIMVLA_REPLAY_JOINT_SUBSTEPS": "1",
        "SIMVLA_REPLAY_ADAPTIVE_GRIPPER": "1", "SIMVLA_REPLAY_INITIAL_JOINT_STATE": "1",
        "SIMVLA_REPLAY_EPISODE_INDEX": "0", "SIMVLA_REPLAY_SETTLE_STEPS": "100",
        "SIMVLA_PHYSICS_SUBSTEPS": "6",
    }.items():
        assert settings[name] == value


def test_experimental_prelude_profile_declares_nominal_targets_and_state_restore():
    path = Path(__file__).parents[1] / "configs/collection/aiworker-kitchen813-loaded-prelude.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for name in ("SIMVLA_REPLAY_INITIAL_STEP", "SIMVLA_REPLAY_INITIAL_JOINT_STATE",
                 "SIMVLA_REPLAY_JOINT_SUBSTEPS", "SIMVLA_SINK_CAVITY_COLLISIONS",
                 "SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"):
        assert settings[name] == "1"
    for name in ("SIMVLA_GOAL_NAV_XY_STD_M", "SIMVLA_GOAL_NAV_YAW_STD_RAD",
                 "SIMVLA_GOAL_ARM_XYZ_STD_M"):
        assert settings[name] == "0"


def test_aiworker_loaded_physics12_profile_preserves_gripper_rate_and_replay_mode():
    path = Path(__file__).parents[1] / "configs/collection/aiworker-kitchen813-loaded-physics12.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for name, value in {
        "ROBOT": "aiworker", "SIMVLA_PHYSICS_SUBSTEPS": "12",
        "SIMVLA_GRIPPER_BALANCE_STEP_FRACTION": ".0025",
        "SIMVLA_MAXFRAMES": "6500", "SIMVLA_EPISODE_STEPS": "5500",
        "SIMVLA_REPLAY_INITIAL_STEP": "1", "SIMVLA_REPLAY_INITIAL_JOINT_STATE": "1",
        "SIMVLA_REPLAY_JOINT_TARGETS": "1", "SIMVLA_REPLAY_ADAPTIVE_GRIPPER": "1",
        "SIMVLA_REPLAY_BASE_TRACKING": "1", "SIMVLA_REPLAY_SETTLE_STEPS": "100",
        "SIMVLA_SIMVQA": "1", "SIMVLA_RECORD_JOINT_SUBSTEPS": "1",
        "SIMVLA_REPLAY_JOINT_SUBSTEPS": "1",
    }.items():
        assert settings[name] == value


@pytest.mark.parametrize("profile", ["nominal", "approach"])
def test_nominal_rby1_batch_retains_action_only_replay_and_strict_basin(profile):
    path = Path(__file__).parents[1] / f"configs/collection/rby1-kitchen813-basin-{profile}.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert settings["SIMVLA_NUM_DEMOS"] == "3"
    assert settings["SIMVLA_MAXFRAMES"] == "8500"
    assert settings["SIMVLA_SINK_CAVITY_COLLISIONS"] == "1"
    assert settings["SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"] == "1"
    if profile == "approach":
        assert settings["SIMVLA_REPLAY_INITIAL_JOINT_STATE"] == "1"
        assert settings["SIMVLA_REPLAY_INITIAL_STEP"] == "1"
        assert settings["SIMVLA_REPLAY_SETTLE_STEPS"] == "0"
    for name in ("SIMVLA_REPLAY_JOINT_TARGETS", "SIMVLA_REPLAY_BASE_TRACKING",
                 "SIMVLA_GOAL_NAV_XY_STD_M", "SIMVLA_GOAL_NAV_YAW_STD_RAD",
                 "SIMVLA_GOAL_ARM_XYZ_STD_M"):
        assert settings[name] == "0"


def test_anubis_prelude_reference_preserves_original_duration():
    path = Path(__file__).parents[1] / "configs/collection/anubis-kitchen813-prelude.env"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; env', "bash", str(path)],
                            check=True, capture_output=True, text=True, cwd="/tmp", env={})
    settings = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert settings["ROBOT"] == "anubis"
    assert settings["SIMVLA_REPLAY_SETTLE_STEPS"] == "0"
    for name in ("SIMVLA_REPLAY_INITIAL_STEP", "SIMVLA_REPLAY_INITIAL_JOINT_STATE",
                 "SIMVLA_REPLAY_JOINT_TARGETS", "SIMVLA_REPLAY_BASE_TRACKING"):
        assert settings[name] == "1"
