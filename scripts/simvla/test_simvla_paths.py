"""Tests for goal-path resolution.

Run with: pytest scripts/simvla/test_simvla_paths.py -v
Pure stdlib — no Omniverse, no scene_synthesizer.
"""

from pathlib import Path

from simvla_paths import goal_files, goals_dir, lerobot_root, repo_root


def test_aiworker_planning_overlay_does_not_change_other_robot_paths(monkeypatch, tmp_path):
    from simvla_paths import robot_asset_path
    monkeypatch.setenv("SIMVLA_AIWORKER_URDF_PATH", str(tmp_path / "corrected.urdf"))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(tmp_path / "models"))
    assert robot_asset_path("ai_worker_min/ffw_sg2_follower.urdf") == tmp_path / "corrected.urdf"
    assert robot_asset_path("anubis/anubis_final.urdf") == tmp_path / "models/anubis/anubis_final.urdf"


def test_robot_assets_can_live_outside_checkout(monkeypatch, tmp_path):
    from simvla_paths import robot_asset_path
    monkeypatch.setenv("SIMVLA_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("SIMVLA_ROBOT_MODELS_DIR", str(tmp_path / "models"))
    assert robot_asset_path("source/isaaclab_assets/data/Robots/robot.usd") == tmp_path / "assets/Robots/robot.usd"
    assert robot_asset_path("anubis/robot.urdf") == tmp_path / "models/anubis/robot.urdf"
    assert robot_asset_path(str(tmp_path / "absolute.urdf")) == tmp_path / "absolute.urdf"


def test_selected_robot_must_match_the_task_articulation():
    from simvla_paths import validate_robot_asset
    validate_robot_asset("anubis", "/assets/Robots/anubis_simvla.usd")
    validate_robot_asset("aiworker", "/assets/Robots/MM/aiworker/ffw_sg2.usd")
    validate_robot_asset("rby1", "/models/model_simvla_black_gripper.usd")


def test_explicit_aiworker_usd_overlay_matches_the_selected_robot(monkeypatch, tmp_path):
    from simvla_paths import validate_robot_asset

    overlay = tmp_path / "ffw_sg2_simvla_grip.usd"
    overlay.touch()
    monkeypatch.setenv("SIMVLA_AIWORKER_USD_PATH", str(overlay))
    validate_robot_asset("aiworker", overlay)


def test_robot_mismatch_fails_before_environment_creation():
    import pytest
    from simvla_paths import validate_robot_asset
    with pytest.raises(ValueError, match="Robot/task mismatch.*Adapt the kitchen task"):
        validate_robot_asset("rby1", "/assets/Robots/anubis_simvla.usd")


def test_unknown_robot_has_a_supported_values_error():
    import pytest
    from simvla_paths import validate_robot_asset
    with pytest.raises(ValueError, match="anubis, aiworker, rby1"):
        validate_robot_asset("unknown", "/assets/robot.usd")


def test_rby1_model_root_is_checked_before_kit_starts(monkeypatch, tmp_path):
    import pytest
    from simvla_paths import validate_robot_model_file

    monkeypatch.delenv("SIMVLA_RBY1M_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="SIMVLA_RBY1M_DIR.*rby1m/"):
        validate_robot_model_file("rby1")
    wrong = tmp_path / "rby1"
    wrong.mkdir()
    monkeypatch.setenv("SIMVLA_RBY1M_DIR", str(wrong))
    with pytest.raises(FileNotFoundError, match="articulation USD is missing"):
        validate_robot_model_file("rby1")


def test_repo_root_honours_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMVLA_REPO_ROOT", str(tmp_path))
    assert repo_root() == tmp_path


def test_repo_root_falls_back_to_the_pyproject_marker(monkeypatch):
    monkeypatch.delenv("SIMVLA_REPO_ROOT", raising=False)
    found = repo_root()
    assert (found / "pyproject.toml").is_file()
    assert (found / "scripts" / "simvla").is_dir()


def test_goals_dir_defaults_under_the_repo(monkeypatch, tmp_path):
    """A fresh clone has no SIMVLA_GOALS_DIR and no symlink — the default must still work."""
    monkeypatch.delenv("SIMVLA_GOALS_DIR", raising=False)
    monkeypatch.setenv("SIMVLA_REPO_ROOT", str(tmp_path))

    resolved = goals_dir()

    assert resolved == tmp_path / "scripts" / "simvla" / "goals"
    assert resolved.is_dir(), "goals_dir must create the directory when it is missing"


def test_goals_dir_honours_the_env_var(monkeypatch, tmp_path):
    """The author's goals live outside the repo; SIMVLA_GOALS_DIR is how they say so."""
    elsewhere = tmp_path / "somewhere" / "else"
    monkeypatch.setenv("SIMVLA_GOALS_DIR", str(elsewhere))
    monkeypatch.setenv("SIMVLA_REPO_ROOT", str(tmp_path))

    resolved = goals_dir()

    assert resolved == elsewhere
    assert resolved.is_dir()


def test_goals_dir_expands_a_leading_tilde(monkeypatch, tmp_path):
    """A single-quoted `SIMVLA_GOALS_DIR='~/simvla-goals'` never gets tilde-expanded by the
    shell. If goals_dir() doesn't expand it either, it resolves to a *relative* path named
    literally '~', and mkdir(parents=True) silently creates '<cwd>/~/simvla-goals' — a
    cwd-dependent location that a later run from a different directory will never find again.
    This must resolve under the real home directory, not a literal '~' relative to cwd.
    """
    monkeypatch.setenv("SIMVLA_GOALS_DIR", "~/simvla_goals_tilde_test")
    monkeypatch.delenv("SIMVLA_REPO_ROOT", raising=False)

    resolved = goals_dir()

    try:
        assert resolved.is_absolute()
        assert "~" not in resolved.parts
        assert resolved == Path.home() / "simvla_goals_tilde_test"
        assert resolved.is_dir()
    finally:
        if resolved.is_dir() and not any(resolved.iterdir()):
            resolved.rmdir()


def test_goals_dir_is_idempotent_on_an_existing_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMVLA_GOALS_DIR", str(tmp_path / "goals"))
    first = goals_dir()
    second = goals_dir()
    assert first == second and second.is_dir()


def test_goal_files_pairs_the_json_with_its_reloadable_twin(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMVLA_GOALS_DIR", str(tmp_path))

    task_json, reloadable_json = goal_files("Isaac-Kitchen-v01-00")

    assert task_json == tmp_path / "Isaac-Kitchen-v01-00.json"
    assert reloadable_json == tmp_path / "Isaac-Kitchen-v01-00.reloadable.json"


def test_goal_files_takes_a_stem_not_a_task_id(monkeypatch, tmp_path):
    """simvla_deploy.py names its files '<task>-<task_type>.json' — the API must not assume
    the stem is a bare task id, or deploy silently looks for the wrong file."""
    monkeypatch.setenv("SIMVLA_GOALS_DIR", str(tmp_path))

    task_json, reloadable_json = goal_files("Isaac-Deploy-RoboCasa-mug2sink")

    assert task_json.name == "Isaac-Deploy-RoboCasa-mug2sink.json"
    assert reloadable_json.name == "Isaac-Deploy-RoboCasa-mug2sink.reloadable.json"


def test_lerobot_root_defaults_to_checkout_outputs(monkeypatch, tmp_path):
    monkeypatch.delenv("SIMVLA_LEROBOT_ROOT", raising=False)
    monkeypatch.setenv("SIMVLA_REPO_ROOT", str(tmp_path))

    assert lerobot_root() == tmp_path / "outputs/lerobot"


def test_lerobot_root_honours_external_path(monkeypatch, tmp_path):
    target = tmp_path / "dataset-export"
    monkeypatch.setenv("SIMVLA_LEROBOT_ROOT", str(target))

    assert lerobot_root() == target.resolve()
